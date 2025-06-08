# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

from typing import NamedTuple


class DecoderOpts(NamedTuple):
    """Decoder options.
    patch_size: size of the patches used for the decoder
    enc_embed_dim: dimension of the encoder embedding
    dec_embed_dim: dimension of the decoder embedding
    dec_depth: depth of the decoder or number of transformer blocks
    dec_num_heads: number of heads in the decoder
    mlp_ratio: ratio of the MLP layer in each transformer block
    norm_im2_in_dec: whether to normalize the 'memory' = (second image) in the decoder
    act_layer_name: activation layer (relu or gelu)
    norm_layer_name: normalization layer (layernorm or batchnorm)
    """

    patch_size: int = 14
    enc_embed_dim: int = 384
    dec_embed_dim: int = 768
    dec_depth: int = 12
    dec_num_heads: int = 12
    mlp_ratio: float = 4

    norm_im2_in_dec: bool = True
    act_layer_name: str = "relu"
    norm_layer_name: str = "layernorm"
