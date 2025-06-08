# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict
from typing import List, NamedTuple


class DPTHeadOpts(NamedTuple):
    """
    Options for DPT Head.
    Args:
        patch_size: size of the patch.
        dec_embed_dim: decoder embedding dimension.
        hooks: list of hooks to use, i.e features of which layers in the transformer decoder.
                Note that the first layer is the input to the transformer decoder, i.e DINOv2 features.
        process_dims: list of dimensions of the process layers, i.e first set of layers in DPT to up/downsample the features in different resolutions.
        proj_dim: dimension of the projection layers, i.e second set of layers in DPT to reduce dimension.
        head_hidden_dims: dimensions of the hidden layers in the head.
        predict_confidence: whether to predict confidence.
    """

    patch_size: int = 14
    dec_embed_dim: int = 768
    hooks: List[int] = [2, 5, 8, 11]
    process_dims: List[int] = [96, 192, 384, 768]
    proj_dim: int = 256
    head_hidden_dims: List[int] = [64, 256]
    predict_confidence: bool = True
