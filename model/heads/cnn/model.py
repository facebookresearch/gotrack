# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.net_util import (
    conv1x1,
    conv3x3,
    conv_transpose,
    ACTIVATION_FUNCTION_REGISTRY,
)
from model.heads.cnn.config import (
    ConvHeadOpts,
)
from einops import rearrange


class ResConvBlock(nn.Module):
    """Residual convolution block composed of two convolutions and a residual connection."""

    def __init__(
        self, in_dim: int, out_dim: int, activation_func_name: str = "relu"
    ) -> None:
        """Initialization for ConvHead.
        Args:
            in_dim: dimension of the input channels.
            out_dim: dimension of the output channels.
            activation_func_name: name of the activation function. Currently only supports "relu".
        """
        super().__init__()
        assert in_dim == out_dim, "in_dim != out_dim"

        self.conv1: nn.Conv2d = conv3x3(in_dim, out_dim)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(out_dim)

        self.conv2: nn.Conv2d = conv3x3(out_dim, out_dim)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(out_dim)

        self.act: nn.Module = ACTIVATION_FUNCTION_REGISTRY[activation_func_name]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x: features of size (B, C, H, W)
        Returns:
            tensor: output of size (B, C, H, W)
        """
        out = self.bn1(self.conv1(self.act(x)))
        out = self.bn2(self.conv2(self.act(out)))
        return x + out


class UpSampleBlock(nn.Module):
    """Block for upsampling from HxW to (H*2)x(Wx2) using transposed convolution.
    Args:
        in_dim: dimension of the input channels.
        out_dim: dimension of the output channels.
        activation_func_name: name of the activation function. Currently only supports "relu".
    """

    def __init__(
        self, in_dim: int, out_dim: int, activation_func_name: str = "relu"
    ) -> None:
        super().__init__()
        self.deconv: nn.ConvTranspose2d = conv_transpose(
            in_dim=in_dim,
            out_dim=in_dim,
            kernel_size=2,
            stride=2,
        )

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=True)
        self.act: nn.Module = ACTIVATION_FUNCTION_REGISTRY[activation_func_name]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x: feature map of size (B, C, H, W)
        Returns:
            tensor: output of size (B, C, H*2, W*2)
        """
        out = self.deconv(self.act(x))
        out = self.conv(self.act(out))

        out_height, out_width = out.shape[2], out.shape[3]
        in_height, in_width = x.shape[2], x.shape[3]

        # checking the output shape is (H*2)x(W*2)
        assert out_height == 2 * in_height and out_width == 2 * in_width

        return out


class ConvHead(nn.Module):
    """This is a simple CNN block:
    - Input: a feature map of size (B, C, h, w)
    - Output: a dense prediction of size (B, 3, H, W) where H = 16*h and W = 16*w
    """

    def __init__(self, opts: ConvHeadOpts, activation_func_name: str = "relu") -> None:
        super().__init__()
        self.opts = opts

        self.projection: nn.Conv2d = conv1x1(
            self.opts.dec_embed_dim,
            self.opts.proj_dim,
        )

        # Create len(opts.hidden_dims) blocks of ResConvBlock and UpSampleBlock.
        self.layers: nn.ModuleList = nn.ModuleList()
        for idx, out_dim in enumerate(opts.hidden_dims):
            # Get the input dimension of the current block.
            # If it is the first block, the input dimension is the projection dimension.
            # Otherwise, the input dimension is the output dimension of the previous block.
            if idx == 0:
                in_dim = self.opts.proj_dim
            else:
                in_dim = self.opts.hidden_dims[idx - 1]
            out_dim = self.opts.hidden_dims[idx]

            # Create a block of ResConvBlock and UpSampleBlock.
            res_block = ResConvBlock(in_dim, in_dim)
            up_block = UpSampleBlock(in_dim, out_dim)
            self.layers.append(nn.ModuleList([res_block, up_block]))

        self.act: nn.Module = ACTIVATION_FUNCTION_REGISTRY[activation_func_name]()

        if self.opts.predict_confidence:
            out_dim = 3
        else:
            out_dim = 2

        # Final convolution layer to output HxWx3 or HxWx2 channels.
        self.final_conv: nn.Conv2d = conv1x1(opts.hidden_dims[-1], out_dim)

    def forward(
        self,
        encoder_tokens: List[torch.Tensor],
        crop_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        Args:
            encoder_tokens: list of tensors, output of the transformer decoder
                - size of the list is the number of layers in the transformer decoder + 1 (1 is for the input to the transformer decoder)
                - each tensor is of size (B, N, C) where N is the number of tokens, C is the embedding dimension
            crop_size: tuple of integers, (H, W) where H is the height and W is the width of the input image, used to to compute the number of patches in height and width.
        Returns:
            flow: tensor of size (B, 2, H, W) where (H,W) =crop_size
            confidences: optional, tensor of size (B, 1, H, W) if self.opts.predict_confidence is True, otherwise None
        """
        # Number of patches in height and width.
        H, W = crop_size
        N_H = H // self.opts.patch_size
        N_W = W // self.opts.patch_size

        # We use only the last output of ViT layers
        tokens = encoder_tokens[-1]

        # Reshape tokens to spatial representation
        features = rearrange(tokens, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W)

        # Projecting the features to the input dimension of the head
        features = self.projection(self.act(features))

        # Run through the blocks
        for layer in self.layers:
            res_block = layer[0]
            up_block = layer[1]
            features = res_block(features)
            features = up_block(features)

        # Output is a dense prediction of size (B, 3, H, W)
        pred = self.final_conv(self.act(features))

        # Since the head is designed for patch size = 16 = 2**4 while DINOv2 has patch size = 14
        # The output is therefore upsampled by 2x to match the size of DINOv2
        pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=True)
        flow = pred[:, :2, :, :]

        # Apply sigmoid to the confidence to get output in [0, 1] range.
        if self.opts.predict_confidence:
            confidence = torch.sigmoid(pred[:, 2, :, :])
        else:
            confidence = None
        return flow, confidence
