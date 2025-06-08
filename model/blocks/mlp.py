# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

from typing import Optional

import torch
import torch.nn as nn
from utils import net_util


class MLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks
    Source: https://github.com/naver/croco/blob/743ee71a2a9bf57cea6832a9064a70a0597fcfcb/models/blocks.py#L58

    Args:
        in_dim: int, number of input features.
        act_layer_name: str, activation layer. By default, use ReLU.
        hidden_dim: int, number of features in the hidden layer. If not set, use in_features.
        out_dim: int, number of output features. If not set, use in_features.
        bias: bool, whether to use bias in the linear layers.
        drop_ratio: float, dropout ratio.
    """

    def __init__(
        self,
        in_dim: int,
        act_layer_name: str,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        bias: Optional[bool] = True,
        drop_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        if hidden_dim is None:
            hidden_dim = in_dim
        bias = net_util.to_2tuple(bias)
        drop_probs = net_util.to_2tuple(drop_ratio)

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias[0])
        self.act: nn.Module = net_util.ACTIVATION_FUNCTION_REGISTRY[act_layer_name]()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias[1])

        self.drop1 = nn.Dropout(drop_probs[0])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
