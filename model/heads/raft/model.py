# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from model.heads.raft.config import (
    RAFTHeadOpts,
)
from einops import rearrange
from torchvision.ops import Conv2dNormActivation


class FlowHead(nn.Module):
    """Flow head, part of the update block.
    Source: https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py#L272

    Takes the hidden state of the recurrent unit as input, and outputs the predicted "delta flow".
    """

    def __init__(self, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (tensor): input if shape (B, C, H, W)
        Returns:
            tensor: output of size (B, 3, H, W), two first channels are the predicted flow, last channel is the confidence
        """
        return self.conv2(self.relu(self.conv1(x)))


class WeightPredictor(nn.Module):
    """Weight predictor to be used when upsampling the predicted flow.
    Source: https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py#L311

    It takes the hidden state of the recurrent unit as input and outputs the mask.
    This is not used in the raft-small model.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        patch_size: int = 14,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.convrelu = Conv2dNormActivation(
            in_channels, hidden_dim, norm_layer=None, kernel_size=3
        )
        # patch_size * patch_size * 9 because the predicted flow is downsampled by patch_size, from the downsampling of the initial FeatureEncoder,
        # and we interpolate with all 9 surrounding neighbors. See paper and appendix B.
        self.conv = nn.Conv2d(hidden_dim, patch_size * patch_size * 9, 1, padding=0)

        # In the original code, they use a factor of 0.25 to "downweight the gradients" of that branch.
        # See e.g. https://github.com/princeton-vl/RAFT/issues/119#issuecomment-953950419
        # or https://github.com/princeton-vl/RAFT/issues/24.
        # It doesn't seem to affect epe significantly and can likely be set to 1.
        self.multiplier = multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x: input tensor of shape (B, C, H, W)
        Returns:
            output of shape (B, patch_size * patch_size * 9, H, W) representing the weights to upsample the flow into (B, 2, H*patch_size, W*patch_size)
        """
        x = self.convrelu(x)
        x = self.conv(x)
        return self.multiplier * x  # pyre-ignore


def upsample_flow(
    flow: torch.Tensor, weight: torch.Tensor, patch_size: int = 14
) -> torch.Tensor:
    """Upsample flow by the input factor (default patch_size=14 for DINOv2 features).
    Source: https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/_utils.py#L29
    Args:
        flows (tensor): input flow of shape (B, 2 or 3, H, W)
        weights (tensor): input mask of shape (B, 9 * patch_size * patch_size, H, W), outputs of WeightPredictor
    Returns:
        tensor: upsampled flow of shape (B, 2 or 3, H*patch_size, W*patch_size)
    """
    # Calculate the new height and width after upsampling.
    batch_size, num_channels, h, w = flow.shape
    new_h, new_w = h * patch_size, w * patch_size

    # Reshape the weights from (B, 9 * patch_size * patch_size, H, W) to (B, 1, 9, patch_size, patch_size, H, W)
    weights = weight.view(batch_size, 1, 9, patch_size, patch_size, h, w)

    # We interpolate with all 9 surrounding neighbors so we want to sum over the weights = 1.
    weights = torch.softmax(weights, dim=2)

    upsampled_flows = F.unfold(
        patch_size * flow, kernel_size=3, padding=1
    ).view(  # pyre-ignore
        batch_size, num_channels, 9, 1, 1, h, w
    )
    upsampled_flow = torch.sum(weights * upsampled_flows, dim=2)
    upsampled_flow = upsampled_flow.permute(0, 1, 4, 2, 5, 3).reshape(
        batch_size, num_channels, new_h, new_w
    )
    return upsampled_flow


class RAFTHead(nn.Module):
    """
    Source: https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py#L434

    This is FlowHead from RAFT with a few modifications:
    - Input is a feature map of size (B, C, H/path_size, W/path_size) instead of (B, 2, H/8, W/8)
    - Mask predicted (from RAFT-Large) is used to upsample the flow with learnt weights.
    """

    def __init__(
        self,
        opts: RAFTHeadOpts,
    ) -> None:
        super().__init__()
        self.opts = opts  # pyre-ignore
        self.flow_head = FlowHead(
            in_channels=self.opts.dec_embed_dim,
            hidden_size=self.opts.hidden_dim,
        )
        self.weight_predictor = WeightPredictor(
            in_channels=self.opts.dec_embed_dim,
            hidden_dim=self.opts.hidden_dim,
        )

    def forward(
        self,
        encoder_tokens: List[torch.Tensor],
        crop_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Args:
            encoder_tokens: list of tensors, output of the transformer decoder
                - size of the list is the number of layers in the transformer decoder + 1 (1 is for the input to the transformer decoder)
                - each tensor is of size (B, N, C) where N is the number of tokens, C is the embedding dimension
            crop_size: tuple of integers, (H, W) where H is the height and W is the width of the input image, used to to compute the number of patches in height and width.
        Returns:
            flow: tensor of size (B, 2, H, W) where (H,W) =crop_size
            confidences: tensor of size (B, 1, H, W) if self.opts.predict_confidence is True, otherwise None
        """
        # Number of patches in height and width.
        H, W = crop_size
        N_H = H // self.opts.patch_size
        N_W = W // self.opts.patch_size

        # We use only the output of the last layer of transformer decoder
        tokens = encoder_tokens[-1]

        # Reshape tokens to spatial representation
        features = rearrange(tokens, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W)

        # Predicting the flow in the original resolution of feature map.
        predicted_flow = self.flow_head(features)

        # Predicting the weight to upsample the flow.
        predicted_weight = self.weight_predictor(features)

        prediction = upsample_flow(
            flow=predicted_flow, weight=predicted_weight
        )  # pyre-ignore

        flow = prediction[:, :2, :, :]
        confidence = torch.sigmoid(prediction[:, 2, :, :])
        return flow, confidence
