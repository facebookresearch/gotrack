# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

from typing import List, Tuple

import torch
import torch.nn as nn
from utils.logging import get_logger

from utils import net_util
from model.blocks import (
    decoder_block,
    rope2d,
)
from model.blocks.config import (
    DecoderOpts,
)
from einops import rearrange

logger = get_logger(__name__)


class Decoder(nn.Module):
    """
    Decoder for the GenericRefiner model.
    This decoder is based on the CroCoNet architecture, takes the features of the two images as input
    and outputs the features of the two images after each decoder block. Each decoder block consists of
    a self-attention layer, a cross-attention layer and a feed-forward layer.

    Source: https://github.com/naver/croco/blob/d3d0ab2858d44bcad54e5bfc24f565983fbe18d9/models/croco.py#L95
    """

    def __init__(
        self,
        opts: DecoderOpts,
    ) -> None:
        super(Decoder, self).__init__()
        self.opts = opts

        # Define the positional embedding with RoPE.
        self.rope_embed: rope2d.RoPE2D = rope2d.RoPE2D()
        self.position_getter = rope2d.PositionGetter()

        # Define the mapping from the output of encoder to the input of decoder.
        self.decoder_embed: nn.Linear = nn.Linear(
            self.opts.enc_embed_dim, self.opts.dec_embed_dim, bias=True
        )

        # Define the normalization layer.
        norm_layer = net_util.NORMALIZATION_REGISTRY[self.opts.norm_layer_name]

        # Create the transformer blocks for the decoder.
        self.dec_blocks: nn.ModuleList = nn.ModuleList(
            [
                decoder_block.DecoderBlock(
                    dim=self.opts.dec_embed_dim,
                    num_heads=self.opts.dec_num_heads,
                    act_layer_name=self.opts.act_layer_name,
                    mlp_ratio=self.opts.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,  # pyre-ignore
                    norm_mem=self.opts.norm_im2_in_dec,
                    rope=self.rope_embed,
                )
                for i in range(self.opts.dec_depth)
            ]
        )
        # Create layer norm for the last output.
        self.dec_norm: nn.Module = norm_layer(self.opts.dec_embed_dim)

    def parameters(self) -> List[nn.Parameter]:  # pyre-ignore
        """Returns the parameters of the model."""
        params = list(self.dec_blocks.parameters()) + list(
            self.decoder_embed.parameters()
        )
        return params

    def _decoder(
        self,
        features1: torch.Tensor,
        pos_enc1: torch.Tensor,
        features2: torch.Tensor,
        pos_enc2: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Apply the decoder (self-atteion and cross-attention) to the features of the two images.
        Args:
            features1: features of the first image (B, N, C)
            pos_enc1: positional encoding of the first image (B, N, C)
            features2: features of the second image (B, N, C)
            pos_enc2: positional encoding of the second image (B, N, C)
        Returns:
            list of features of the two images after each decoder block (B, N, C)
        """
        list_feats1 = [features1]
        list_feats2 = [features2]

        # Mapping the input features to desired dimension.
        feats1 = self.decoder_embed(features1)
        feats2 = self.decoder_embed(features2)
        list_feats1.append(feats1)
        list_feats2.append(feats2)

        for block in self.dec_blocks:
            # img1 side
            feats1, _ = block(list_feats1[-1], list_feats2[-1], pos_enc1, pos_enc2)
            # img2 side
            feats2, _ = block(list_feats2[-1], list_feats1[-1], pos_enc2, pos_enc1)
            # store the result
            list_feats1.append(feats1)
            list_feats2.append(feats2)

        # Remove the features of self.decoder_embed.
        del list_feats1[1]
        del list_feats2[1]

        # Normalize last output.
        list_feats1[-1] = self.dec_norm(list_feats1[-1])
        list_feats2[-1] = self.dec_norm(list_feats2[-1])
        return list_feats1, list_feats2

    def forward(
        self,
        features_query: torch.Tensor,
        features_reference: torch.Tensor,
        crop_size: Tuple[int, int],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            - features_query: features of the query image (B, C, H, W)
            - features_reference: features of the reference image (B, C, H, W)
            - crop_size: size of the images (H, W)
        Returns:
            list of features of the two images after each decoder block (B, N, C)
        """

        batch_size = features_query.shape[0]
        device = features_query.device

        # Reshape features to (B, N, C) where N is the number of patches in height and width.
        features_query = rearrange(features_query, "b c h w -> b (h w) c")
        features_reference = rearrange(features_reference, "b c h w -> b (h w) c")

        # Get the positional encoding.
        num_patches_h = int(crop_size[0] / self.opts.patch_size)
        num_patches_w = int(crop_size[1] / self.opts.patch_size)
        pos = self.position_getter(
            batch_size * 2,
            num_patches_h,
            num_patches_w,
            device=device,
        )
        pos_query, pos_reference = pos.chunk(2, dim=0)

        # Apply the decoder.
        list_features_query, list_features_reference = self._decoder(
            features_query,
            pos_query,
            features_reference,
            pos_reference,
        )
        return list_features_query, list_features_reference
