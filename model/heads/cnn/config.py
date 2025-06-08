# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict
from typing import List, NamedTuple


class ConvHeadOpts(NamedTuple):
    """Options for ConvHead.
    ConvHead takes a feature map of size (B, C, H, W) and outputs flow of size (B, 2, H*patch_size, W*patch_size) + confidence if used.
    Args:
        patch_size: size of the patch.
        dec_embed_dim: decoder embedding dimension.
        proj_dim: dimension of the projection layer, i.e first layer to reduce dimension.
        hidden_dims: dimensions of the hidden layers.
        predict_confidence: whether to predict confidence.
    Returns:
        ConvHead module.
    """

    patch_size: int = 14
    dec_embed_dim: int = 384
    proj_dim: int = 128
    hidden_dims: List[int] = [
        384,
        192,
        96,
        48,
    ]
    predict_confidence: bool = True
