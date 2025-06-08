# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict


from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class PositionGetter(object):
    """
    This class is used to get the positions of each patch in the input tensor.
    """

    def __init__(self) -> None:
        self.cache_positions: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        """
        Args:
            batch_size (int): The batch size of the input tensor.
            height (int): The height of the input tensor.
            width (int): The width of the input tensor.
            device (torch.device): The device on which the input tensor will be created.
        Returns:
            torch.Tensor: A tensor of shape (batch_size, height * width, 2) containing the positions of each patch.
        """
        if (height, width) not in self.cache_positions:
            x = torch.arange(width, device=device)
            y = torch.arange(height, device=device)
            self.cache_positions[height, width] = torch.cartesian_prod(y, x)
        pos = self.cache_positions[height, width]
        return pos.view(1, height * width, 2).expand(batch_size, -1, 2).clone()


class RoPE2D(torch.nn.Module):
    """
    This class implements the RoPE (Rotary Position Embedding) 2D method for encoding positional information into a sequence of tokens.
    Table 1 of Crocov2 (https://arxiv.org/pdf/2211.10408) has shown that using RoPE instead of classical positional encoding (e.g. sine/cosine)
    improves the performance of vision transformers.

    RoPE paper: https://arxiv.org/pdf/2104.09864v5
    Source: https://github.com/naver/croco/blob/743ee71a2a9bf57cea6832a9064a70a0597fcfcb/models/pos_embed.py#L112
    Additional source: https://github.com/lucidrains/rotary-embedding-torch

    TODO: understand why Crocov2 uses base_freq=100.0 instead of exp
    """

    def __init__(
        self,
        base_freq: float = 100.0,
    ) -> None:
        super().__init__()
        self.base_freq: float = base_freq
        self.cache: Dict[Any, Tuple[torch.Tensor, torch.Tensor]] = {}  # pyre-ignore

    def get_cos_sin(
        self, d_model: int, max_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function generates the cosine and sine values for each position in the sequence.
        Source: https://github.com/naver/croco/blob/743ee71a2a9bf57cea6832a9064a70a0597fcfcb/models/pos_embed.py#L120
        Reference from Pytorch: https://discuss.pytorch.org/t/positional-encoding/175953

        Args:
            input_dim (int): The dimension of the input sequence.
            seq_len (int): The length of the sequence.
            device (torch.device): The device on which the cosine and sine values will be generated.
            dtype (torch.dtype): The data type of the cosine and sine values.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the cosine and sine values
        """
        # Check if the input parameters have been cached before
        if (d_model, max_len, device, dtype) not in self.cache:
            # Position tensor of size (max_len, ) with values from 0 to max_len - 1:
            position = torch.arange(max_len, device=device)

            # Create a tensor of size (d_model/2,) representing frequencies.
            frequency = torch.arange(0, d_model, 2).float().to(device) / d_model

            # Div term.
            # TODO: check why Crocov2 uses base_freq=100.0 instead of exp
            div_term = 1.0 / (self.base_freq ** (frequency))  # (d_model/2)

            # Get positional encoding (max_len, d_model/2).
            freqs = torch.einsum("i,j->ij", position, div_term).to(dtype)

            # Prepare the cosine and sine values for each position.
            # TODO: check why Crocov2 repeats the freqs ?
            # https://github.com/naver/croco/blob/743ee71a2a9bf57cea6832a9064a70a0597fcfcb/models/pos_embed.py#L125
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)

            # Cache the cosine and sine values
            self.cache[d_model, max_len, device, dtype] = (cos, sin)

        return self.cache[d_model, max_len, device, dtype]

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotate the input tensor [x1, x2] to [-x2, x1] along the last dimension.
        """
        # Split the input tensor x into two halves along the last dimension
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(
        self,
        tokens: torch.Tensor,
        pos1d: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rotary position embedding to the input tokens.
        Args:
            tokens (torch.Tensor): The input tokens to apply RoPE to. Shape: (batch_size, nheads, ntokens, dim)
            pos1d (torch.Tensor): The 1D position of each token. Shape: (batch, ntokens)
            cos (torch.Tensor): The cosine values for each position. Shape: (ntokens, dim)
            sin (torch.Tensor): The sine values for each position. Shape: (ntokens, dim)
        Returns:
            torch.Tensor: The input tokens with RoPE applied. Shape: (batch_size, nheads, ntokens, dim)
        """
        # Check that pos1d has 2 dimensions.
        assert pos1d.ndim == 2

        # TODO: understand why Crocov2 create lookup table for cos and sin in this way.
        cos = nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = nn.functional.embedding(pos1d, sin)[:, None, :, :]

        # Apply the RoPE transformation to the input tokens.
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE2D to the input tokens.
        Args:
            tokens (torch.Tensor): The input tokens to apply RoPE to. Shape: (batch_size, nheads, ntokens, dim)
            positions (torch.Tensor): The 2D position of each token. Shape: (batch, ntokens, 2), i.e output of PositionGetter.
        Returns:
            torch.Tensor: The input tokens with RoPE applied. Shape: (batch_size, nheads, ntokens, dim)
        """
        d_model = tokens.size(3)
        assert d_model % 2 == 0, "d_model should be a multiple of two"
        assert positions.ndim == 3 and positions.shape[-1] == 2, (
            "positions should have shape (batch, ntokens, 2)"
        )

        # Get positional encoding.
        max_len = int(positions.max()) + 1
        cos, sin = self.get_cos_sin(d_model // 2, max_len, tokens.device, tokens.dtype)

        # Split tokens into two parts along the last dimension.
        y, x = tokens.chunk(2, dim=-1)

        # Apply RoPE to each part separately.
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)

        # Concatenate the two parts along the last dimension.
        tokens = torch.cat((y, x), dim=-1)
        return tokens
