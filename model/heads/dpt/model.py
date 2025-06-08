# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.heads.dpt.config import (
    DPTHeadOpts,
)
from model.heads.dpt.util import (
    create_fusion_layers,
    create_head,
    create_processing_blocks,
    create_projection_layers,
)

from utils.logging import get_logger


logger = get_logger(__name__)


class DPTHead(nn.Module):
    """
    Source: https://github.com/naver/croco/blob/d3d0ab2858d44bcad54e5bfc24f565983fbe18d9/models/dpt_block.py#L264

    DPT Head used in CroCov2 + Dust3r.
    This head is composed of three parts:
    - processing_layers: a list of blocks that process the input features into different resolutions.
    - projection_layers: a list of blocks that project the processed features into the output dimension.
    - fusion_layers: a list of blocks that fuse the output of the processing and projection layers into a single feature map.
    - head: a block that predicts the flow and confidence from the fused feature map.

    """

    def __init__(
        self,
        opts: DPTHeadOpts,
    ) -> None:
        super().__init__()
        self.opts = opts
        assert len(self.opts.process_dims) == len(self.opts.hooks), (
            "Mismatch in number of input features and dimension of the output features"
        )

        # Create processing layers.
        self.process_layers: nn.ModuleList = create_processing_blocks(
            in_dim=opts.dec_embed_dim, out_dims=opts.process_dims
        )

        # Create projection layers.
        self.projection_layers: nn.ModuleList = create_projection_layers(
            in_dims=opts.process_dims, out_dim=opts.proj_dim
        )

        # Create fusion layers.
        self.fusion_layers: nn.ModuleList = create_fusion_layers(
            in_dims=[opts.proj_dim for _ in range(len(opts.process_dims))],
        )

        # Create head.
        self.head: nn.Module = create_head(
            in_dim=opts.proj_dim,
            hidden_dims=opts.head_hidden_dims,
            predict_confidence=opts.predict_confidence,
        )

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

        # Select a subset of the output of the transformer decoder.
        tokens = [encoder_tokens[hook] for hook in self.opts.hooks]

        # Reshape tokens to spatial representation
        tokens = [
            rearrange(token, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W)
            for token in tokens
        ]

        # Process the features to different resolutions:
        # Process layer 1: B x C1 x (H*4) x (W*4) with 1 conv + 1 deconv
        # Process layer 2: B x C2 x (H*2) x (W*2) with 1 conv + 1 deconv
        # Process layer 3 (if used): B x C3 x (H) x (W) with 1 conv
        # Process layer 4 (if used): B x C4 x (H/2) x (W/2) with 2 conv
        tokens = [self.process_layers[idx](token) for idx, token in enumerate(tokens)]

        # Project features to chosen projection dimension C:
        # Projection layer 1: B x C x (H*4) x (W*4)
        # Projection layer 2: B x C x (H*2) x (W*2)
        # Projection layer 3 (if used): B x C x (H) x (W)
        # Projection layer 4 (if used): B x C x (H/2) x (W/2)
        tokens = [
            self.projection_layers[idx](token) for idx, token in enumerate(tokens)
        ]

        assert len(tokens) in [2, 4], "Only 2 or 4 layers are supported"
        # We start from the last layer and go to the first layer.
        fused_tokens = [tokens[-1]]
        if len(self.opts.hooks) == 2:
            layer_indexes = [1, 0]
        else:
            layer_indexes = [3, 2, 1, 0]

        # Fuse layers from N+1 and N to a single feature map, applied in reverse order.
        # Fusion layer 4 (if used): features of projection layer 4 from B x C x (H/2) x (W/2) to B x C x (H) x (W)
        # Fusion layer 3 (if used): fuse features of Fusion layer 4: B x C x (H) x (W) and Projection layer 3 B x C x (H) x (W), then upsample to B x C x (H*2) x (W*2)
        # Fusion layer 2: fuse features of Fusion layer 3: B x C x (H*2) x (W*2) and Projection layer 2 B x C x (H*2) x (W*2), then upsample to B x C x (H*4) x (W*4)
        # Fusion layer 1: fuse features of Fusion layer 2: B x C x (H*4) x (W*4) and Projection layer 1 B x C x (H*4) x (W*4), then upsample to B x C x (H*8) x (W*8)

        # As mentioned above, we start from the last layer and go to the first layer.
        for idx_layer in layer_indexes:
            fusion_layer = self.fusion_layers[idx_layer]

            if idx_layer == len(self.opts.hooks) - 1:
                # There is no fusion in the last layer.
                fused_token = fusion_layer(fused_tokens[-1])
            else:
                # Fusing the feature map of N+1 layer to the feature map of N layer.
                fused_token = fusion_layer(fused_tokens[-1], tokens[idx_layer])
            fused_tokens.append(fused_token)

        # Predicting the flow and confidence.
        pred = self.head(fused_tokens[-1])

        # Since DPT is designed for patch size = 16 = 2**4 while DINOv2 has patch size = 14
        # The output is therefore upsampled by 2x to match the size of DINOv2
        pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=True)
        flow = pred[:, :2, :, :]
        if self.opts.predict_confidence:
            confidence = torch.sigmoid(pred[:, 2, :, :])
        else:
            confidence = None
        return flow, confidence
