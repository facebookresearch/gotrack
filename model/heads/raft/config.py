# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict
from typing import NamedTuple


class RAFTHeadOpts(NamedTuple):
    """
    Options for DPT Head.
    Args:
        patch_size: size of the patch.
        dec_embed_dim: decoder embedding dimension.
        hidden_dim: dimension of the hidden layers.
        predict_confidence: whether to predict confidence.
    """

    patch_size: int = 14
    dec_embed_dim: int = 384
    hidden_dim: int = 256
    predict_confidence: bool = True
