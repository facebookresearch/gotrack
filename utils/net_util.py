# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

import collections.abc
from collections.abc import Callable

from functools import partial
from itertools import repeat
from typing import Dict, Tuple, Union

import torch.nn as nn
import numpy as np
import torch
from utils import logging

logger = logging.get_logger(__name__)


def _ntuple(n: int) -> Callable:
    """
    Return a function that takes an argument and returns a tuple of length n.
    If the argument is already a tuple of length n, it is returned as is.
    Otherwise, the argument is repeated n times to form a tuple.
    Args:
        n: The length of the tuple to return.
    Returns:
        A function that takes an argument and returns a tuple of length n.
    """

    def parse(x: int) -> Tuple[int, int]:
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)  # pyre-ignore


def conv1x1(in_dim: int, out_dim: int, stride: int = 1, bias: bool = True) -> nn.Conv2d:
    """
    Create a 1x1 convolutional layer.
    Args:
        in_dim: Number of input channels.
        out_dim: Number of output channels.
        stride: Stride of the convolution.
    Returns:
        a convolutional layer.
    """
    return nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=bias)


def conv3x3(
    in_dim: int,
    out_dim: int,
    stride: int = 1,
    padding: int = 1,
    groups: int = 1,
    bias: bool = True,
) -> nn.Conv2d:
    """
    Create a 3x3 convolutional layer.
    Args:
        in_dim: Number of input channels.
        out_dim: Number of output channels.
        stride: Stride of the convolution.
    Returns:
        a convolutional layer.
    """
    return nn.Conv2d(
        in_dim,
        out_dim,
        kernel_size=3,
        padding=padding,
        stride=stride,
        bias=bias,
        groups=groups,
    )


def conv_transpose(
    in_dim: int,
    out_dim: int,
    kernel_size: int,
    stride: int,
    padding: int = 0,
    groups: int = 1,
    bias: bool = True,
    dilation: int = 1,
) -> nn.ConvTranspose2d:
    """
    Create a 3x3 convolutional layer (commonly used to upsample the resolution of feature maps).
    Args:
        in_dim: Number of input channels.
        out_dim: Number of output channels.
        stride: Stride of the convolution.
        ...
    Returns:
        a convolutional layer.
    """
    return nn.ConvTranspose2d(
        in_channels=in_dim,
        out_channels=out_dim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        dilation=dilation,
        groups=groups,
    )


def kaiming_init(module: nn.Module, bias: float = 0) -> None:
    """
    Initialize a module with the Kaiming initialization.
    Args:
        module: module to initialize.
        bias: bias to use.
    """
    # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Union[Module, Tensor]`.
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if hasattr(module, "bias") and module.bias is not None:
        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Union[Module,
        #  Tensor]`.
        nn.init.constant_(module.bias, bias)


ACTIVATION_FUNCTION_REGISTRY = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "identity": nn.Identity,
}


NORMALIZATION_REGISTRY: Dict[str, Union[nn.Module, Callable[..., nn.Module]]] = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "identity": nn.Identity,
}


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    checkpoint_key: str = None,
    prefix: str = "",
) -> None:
    checkpoint = torch.load(checkpoint_path)
    if checkpoint_key is not None:
        pretrained_dict = checkpoint[checkpoint_key]  # "state_dict"
    else:
        pretrained_dict = checkpoint
    pretrained_dict = {k.replace(prefix, ""): v for k, v in pretrained_dict.items()}
    model_dict = model.state_dict()

    # compare keys and update value
    pretrained_dict_can_load = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    pretrained_dict_cannot_load = [
        k for k, v in pretrained_dict.items() if k not in model_dict
    ]

    model_dict_not_update = [
        k for k, v in model_dict.items() if k not in pretrained_dict
    ]

    module_cannot_load = np.unique(
        [k.split(".")[0] for k in pretrained_dict_cannot_load]  #
    )

    module_not_update = np.unique([k.split(".")[0] for k in model_dict_not_update])

    logger.info(f"Cannot load: {module_cannot_load}")
    logger.info(f"Not update: {module_not_update}")

    size_pretrained = len(pretrained_dict)
    size_pretrained_can_load = len(pretrained_dict_can_load)
    size_pretrained_cannot_load = len(pretrained_dict_cannot_load)
    size_model = len(model_dict)
    logger.info(
        f"Pretrained: {size_pretrained}/ Loaded: {size_pretrained_can_load}/ Cannot loaded: {size_pretrained_cannot_load} VS Current model: {size_model}"
    )
    model_dict.update(pretrained_dict_can_load)
    model.load_state_dict(model_dict)
    logger.info("Load pretrained done!")
