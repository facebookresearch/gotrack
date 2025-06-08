# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict
from typing import List

import torch

import torch.nn as nn
import torch.nn.functional as F  # noqa
from utils.logging import get_logger
from utils.net_util import (
    conv1x1,
    conv3x3,
    conv_transpose,
)
from model.heads.dpt.fusion_block import (
    FeatureFusionBlock,
)

logger = get_logger(__name__)


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(
        self, scale_factor: int, mode: str, align_corners: bool = False
    ) -> None:
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate  # pyre-ignore
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x (tensor): input of size (B, C, H, W)
        Returns:
            tensor of size (B, C, H*scale_factor, W*scale_factor)
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


def create_fusion_layers(
    in_dims: List[int], use_batch_norm: bool = False
) -> nn.ModuleList:
    """
    Create a fusion block which is used to fuse two feature maps of different resolution together
    Please see FeatureFusionBlock for more details.
    Args:
        in_dim (int): number of input channels
        use_batch_norm (bool): whether to use batch norm
    Returns:
        one fusion block to fuse two feature maps of different resolution together
    """
    layers = nn.ModuleList()
    for i in range(len(in_dims)):
        layer = FeatureFusionBlock(
            in_dim=in_dims[i],
            activation_func_name="relu",
            use_batch_norm=use_batch_norm,
        )
        layers.append(layer)
    return layers


def create_projection_layers(
    in_dims: List[int],
    out_dim: int,
) -> nn.ModuleList:
    """
    Source: https://github.com/naver/croco/blob/d3d0ab2858d44bcad54e5bfc24f565983fbe18d9/models/dpt_block.py#L20

    This function is called scratch in DPT + Crocov2 but we change it name to projection_layers to be more clear.
    This function create a set of projection layers which are used to project the input feature maps BxCxHxW to a single feature map Bxout_dimxHxW
    Args:
        in_dims (List[int]): list of input dimensions
        out_dim (int): output dimension
    Returns:
        len(in_dims) projection layers
    """
    layers = nn.ModuleList()
    for i in range(len(in_dims)):
        # This conv3x3 project BxCxHxW to Bxout_dimxHxW
        layer = conv3x3(
            in_dim=in_dims[i],
            out_dim=out_dim,
            bias=False,
        )
        layers.append(layer)
    return layers


def create_processing_blocks(
    in_dim: int,
    out_dims: List[int],
) -> nn.ModuleList:
    """
    Source: https://github.com/naver/croco/blob/d3d0ab2858d44bcad54e5bfc24f565983fbe18d9/models/dpt_block.py#L356
    This is called postprocess layers in DPT + Crocov2 but we change it name to processing_blocks to be more clear.
    This function create a set of processing blocks which are used to process the input feature maps BxCxHxW to:
        Layer 1: B x C1 x (H*4) x (W*4) with 1 conv + 1 deconv
        Layer 2: B x C2 x (H*2) x (W*2) with 1 conv + 1 deconv
        Layer 3 (if used): B x C3 x (H) x (W) with 1 conv
        Layer 4 (if used): B x C4 x (H/2) x (W/2) with 2 conv

    Args:
        in_dim (int): input dimension (output of transformer decoder)
        out_dims (List[int]): list of output dimensions of each block.
            since the resolution of layer 1 = 4x layer 2 = 2x layer 3 = 1x layer 4,
            we have output_dim of layer1 = 2x output_dim of layer 2 = 2x output_dim of layer 3 = 2x output_dim of layer.
    Returns:
        len(out_dims) processing blocks
    """
    # The processing blocks are used to process the feature maps of different layers.
    # So there should be at least two processing blocks.
    # This function is generic and can be used for any number of processing blocks >= 2.
    # By default, we create two first layers and the rest of the layers are created if needed.

    # Layer 1: conv3x3: BxCxHxW -> BxC1xHxW, conv_transpose: BxC1xHxW -> BxC1x(H*4)x(W*4)
    layer1 = nn.Sequential(
        conv1x1(in_dim=in_dim, out_dim=out_dims[0]),
        conv_transpose(
            in_dim=out_dims[0], out_dim=out_dims[0], kernel_size=4, stride=4
        ),
    )
    # Layer 2: conv3x3: BxCxHxW -> BxC2xHxW, conv_transpose: BxC2xHxW -> BxC2x(H*2)x(W*2)
    layer2 = nn.Sequential(
        conv1x1(in_dim=in_dim, out_dim=out_dims[1]),
        conv_transpose(
            in_dim=out_dims[1], out_dim=out_dims[1], kernel_size=2, stride=2
        ),
    )
    layers = nn.ModuleList([layer1, layer2])

    # If there are more than 2 processing blocks, create the rest of the layers.
    if len(out_dims) >= 3:
        # In DPT, layer 3 is only one conv3x3: BxCxHxW -> BxC3xHxW.
        layer3 = conv1x1(in_dim=in_dim, out_dim=out_dims[2])
        layers.append(layer3)
        if len(out_dims) >= 4:
            # In DPT, layer 4 is two conv3x3: BxCxHxW -> BxC4xHxW, BxC4xHxW -> BxC4x(H/2)xW/2).
            layer4 = nn.Sequential(
                conv1x1(in_dim=in_dim, out_dim=out_dims[3]),
                conv3x3(in_dim=out_dims[3], out_dim=out_dims[3], stride=2, padding=1),
            )
            layers.append(layer4)
    return layers


def create_head(
    in_dim: int, hidden_dims: List[int], predict_confidence: bool
) -> nn.Sequential:
    assert len(hidden_dims) == 2, "hidden_dims must be a list of length 3"
    head = nn.Sequential(
        conv3x3(in_dim=in_dim, out_dim=hidden_dims[0]),
        Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        conv3x3(in_dim=hidden_dims[0], out_dim=hidden_dims[1]),
        nn.ReLU(True),
        conv1x1(
            in_dim=hidden_dims[1],
            out_dim=3 if predict_confidence else 2,
        ),
    )
    return head
