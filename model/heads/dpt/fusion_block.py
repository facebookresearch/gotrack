# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from utils.logging import get_logger
from utils.net_util import (
    conv1x1,
    conv3x3,
)


logger = get_logger(__name__)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block which composes of two residual blocks and a convolution:
    - Residual block 1: apply to the second feature maps (if available),
                        then add it to the output of the first residual block
    - Residual block 2: apply to the first input feature maps or the output of the first residual block
    - Convolution: to project the feature maps to the desired number of channels.
    """

    def __init__(
        self,
        in_dim: int,
        activation_func_name: str,
        use_batch_norm: bool,
    ) -> None:
        """
        Args:
            in_dim (int): number of input channels
            activation_func_name (str): name of the activation function
            use_batch_norm (bool): whether to use batch norm
            expand (bool): whether to expand the resolution of the input feature resolution
        """
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(
            in_dim, activation_func_name, use_batch_norm
        )
        self.resConfUnit2 = ResidualConvUnit(
            in_dim, activation_func_name, use_batch_norm
        )
        self.out_conv: nn.Conv2d = conv1x1(in_dim, in_dim)

        # TODO: what is the dfference between skip_add and +
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(
        self, feature1: torch.Tensor, features2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        Args:
            feature1: feature map of N layer, HxWxC1
            features2: feature map of N+1 layer, (H/2)x(W/2)xC2 (C2=2*C1)
        Returns:
            feature map of N resolution,
        """
        h, w = feature1.shape[2:]

        # Fusing the feature map of N+1 layer to the feature map of N layer.
        if features2 is not None:
            res = self.resConfUnit1(features2)
            # Make sure two feature maps have the same resolution.
            res = F.interpolate(res, size=(h, w), mode="bilinear")
            output = self.skip_add.add(feature1, res)
        else:
            output = feature1

        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(
            output,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        output = self.out_conv(output)
        return output


class ResidualConvUnit(nn.Module):
    """Residual convolution module composed of two 3x3 convolutions, batch norm, and ReLU.
    This class outputs the feature maps of same resolution as the input.
    """

    def __init__(
        self, in_dim: int, activation_func_name: str, use_batch_norm: bool
    ) -> None:
        """
        Args:
            in_dim (int): number of input channels
            activation_func_name (str): name of the activation function
            use_batch_norm (bool): whether to use batch norm
        """
        super().__init__()
        self.use_batch_norm = use_batch_norm

        # TODO: why bias is not used when use_batch_norm is True?
        self.conv1: nn.Conv2d = conv3x3(
            in_dim, in_dim, padding=1, bias=not use_batch_norm
        )
        self.conv2: nn.Conv2d = conv3x3(
            in_dim, in_dim, padding=1, bias=not use_batch_norm
        )
        if self.use_batch_norm:
            self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(in_dim)
            self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(in_dim)

        if activation_func_name == "relu":
            self.activation: nn.Module = nn.ReLU(False)
        else:
            raise ValueError(f"Unknown activation function: {activation_func_name}")

        # TODO: what is the dfference between skip_add and +
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.use_batch_norm:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)

        return self.skip_add.add(x, out)
