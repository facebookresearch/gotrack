# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

"""Miscellaneous functions."""

import dataclasses
from dataclasses import asdict
from typing import Any, Mapping

import numpy as np

import torch
from utils import logging


logger = logging.get_logger(__name__)


def is_dictlike(obj: Any) -> bool:
    """
    Returns true if the object is a dataclass, NamedTuple, or Mapping.
    """
    return (
        dataclasses.is_dataclass(obj)
        or hasattr(obj, "_asdict")
        or isinstance(obj, Mapping)
    )


def map_fields(func, obj, only_type=object):
    """
    map 'func' recursively over nested collection types.

    >>> map_fields(lambda x: x * 2,
    ...            {'a': 1, 'b': {'x': 2, 'y': 3}})
    {'a': 2, 'b': {'x': 4, 'y': 6}}

    E.g. to detach all tensors in a network output frame:

        frame = map_fields(torch.detach, frame, torch.Tensor)

    The optional 'only_type' parameter only calls `func` for values where
    isinstance(value, only_type) returns True. Other values are returned
    as-is.
    """
    if is_dictlike(obj):
        ty = type(obj)
        if isinstance(obj, Mapping):
            return ty((k, map_fields(func, v, only_type)) for (k, v) in obj.items())
        else:
            # NamedTuple or dataclass
            return ty(
                **{k: map_fields(func, v, only_type) for (k, v) in asdict(obj).items()}
            )
    elif isinstance(obj, tuple):
        return tuple(map_fields(func, v, only_type) for v in obj)
    elif isinstance(obj, list):
        return [map_fields(func, v, only_type) for v in obj]
    elif isinstance(obj, only_type):
        return func(obj)
    else:
        return obj


def array_to_tensor(
    array: np.ndarray, make_array_writeable: bool = True
) -> torch.Tensor:
    """Converts a Numpy array into a tensor.

    Args:
        array: A Numpy array.
        make_array_writeable: Whether to force the array to be writable.
    Returns:
        A tensor.
    """

    # If the array is not writable, make it writable or copy the array.
    # Otherwise, torch.from_numpy() would yield a warning that tensors do not
    # support the writing lock and one could modify the underlying data via them.
    if not array.flags.writeable:
        if make_array_writeable and array.flags.owndata:
            array.setflags(write=True)
        else:
            array = np.array(array)
    return torch.from_numpy(array)


def arrays_to_tensors(data: Any) -> Any:
    """Recursively converts Numpy arrays into tensors.

    Args:
        data: A possibly nested structure with Numpy arrays.
    Returns:
        The same structure but with Numpy arrays converted to tensors.
    """

    return map_fields(lambda x: array_to_tensor(x), data, only_type=np.ndarray)


def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    """Converts a tensor into a Numpy array.

    Args:
        tensor: A tensor (may be in the GPU memory).
    Returns:
        A Numpy array.
    """

    return tensor.detach().cpu().numpy()


def tensors_to_arrays(data: Any) -> Any:
    """Recursively converts tensors into Numpy arrays.

    Args:
        data: A possibly nested structure with tensors.
    Returns:
        The same structure but with tensors converted to Numpy arrays.
    """

    return map_fields(lambda x: tensor_to_array(x), data, only_type=torch.Tensor)
