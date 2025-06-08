# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict
from typing import Tuple

import torch
import torch.nn as nn

from model.blocks import (
    attention,
    cross_attention,
    mlp,
    rope2d,
)


class DecoderBlock(nn.Module):
    """
    Transformer decoder block with attention, cross-attention, and MLP.
    Source: https://github.com/naver/croco/blob/743ee71a2a9bf57cea6832a9064a70a0597fcfcb/models/blocks.py#L171
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        act_layer_name: str,
        norm_layer: nn.Module,
        rope: rope2d.RoPE2D,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop_ratio: float = 0.0,
        drop_path_ratio: float = 0.0,
        norm_mem: bool = True,
    ) -> None:
        super().__init__()
        self.norm1: nn.Module = norm_layer(dim)
        self.attn = attention.Attention(
            dim=dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop_ratio,
            proj_drop=drop,
        )

        self.cross_attn = cross_attention.CrossAttention(
            dim=dim,
            rope=rope,  # pyre-ignore
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop_ratio,
            proj_drop=drop,
        )

        self.norm2: nn.Module = norm_layer(dim)
        self.norm3: nn.Module = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp.MLP(
            in_dim=dim,
            hidden_dim=mlp_hidden_dim,
            act_layer_name=act_layer_name,
            drop_ratio=drop,
        )
        self.norm_y: nn.Module = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, xpos: torch.Tensor, ypos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x), xpos)
        y_ = self.norm_y(y)
        x = x + self.cross_attn(self.norm2(x), y_, y_, xpos, ypos)
        x = x + self.mlp(self.norm3(x))
        return x, y
