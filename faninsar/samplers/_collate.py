"""Collate helpers for sampler DataLoader integration."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

from faninsar.logging import setup_logger

logger = setup_logger(__name__)


def identity_collate(batch: list[Any]) -> Any | list[Any]:
    """Return batch as-is; unbox single-item batches for convenience.

    Parameters
    ----------
    batch : list[Any]
        Samples returned by the dataset.

    Returns
    -------
    Any | list[Any]
        A single sample when batch size is 1, otherwise the original list.

    """
    if len(batch) == 1:
        return batch[0]
    return batch


def _to_tensor_all(value: Any) -> Any:  # noqa: PLR0911
    """Recursively convert numpy arrays to CPU torch tensors.

    Parameters
    ----------
    value : Any
        Input value to be converted when it contains numpy arrays.

    Returns
    -------
    Any
        Converted structure with numpy arrays turned into torch tensors.

    """
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype == np.object_:
            return value
        return torch.as_tensor(value)
    if isinstance(value, np.generic):
        return torch.as_tensor(value)
    if isinstance(value, dict):
        return {key: _to_tensor_all(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_tensor_all(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_tensor_all(item) for item in value)
    return value


def _to_tensor_data_only(value: Any) -> Any:
    """Convert only ``data`` fields in nested structures to torch tensors.

    Parameters
    ----------
    value : Any
        Input value that may contain a ``data`` key within nested mappings.

    Returns
    -------
    Any
        Structure with ``data`` entries converted to torch tensors.

    """
    if isinstance(value, dict):
        converted: dict[Any, Any] = {}
        for key, item in value.items():
            if key == "data":
                converted[key] = _to_tensor_all(item)
            else:
                converted[key] = _to_tensor_data_only(item)
        return converted
    if isinstance(value, list):
        return [_to_tensor_data_only(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_tensor_data_only(item) for item in value)
    return value


def tensor_collate(
    batch: list[Any],
    tensor_scope: Literal["data", "all"] = "data",
) -> Any | list[Any]:
    """Convert numpy arrays to torch tensors and unbox single-item batches.

    Parameters
    ----------
    batch : list[Any]
        Samples returned by the dataset.
    tensor_scope : {"data", "all"}, optional
        Conversion scope. ``"data"`` converts only ``data`` fields; ``"all"``
        converts every numpy array recursively. Default is ``"data"``.

    Returns
    -------
    Any | list[Any]
        A single converted sample when batch size is 1, otherwise a list of
        converted samples.

    Raises
    ------
    ValueError
        If ``tensor_scope`` is not one of ``"data"`` or ``"all"``.

    """
    if tensor_scope == "data":
        converter = _to_tensor_data_only
    elif tensor_scope == "all":
        converter = _to_tensor_all
    else:
        msg = f"tensor_scope must be 'data' or 'all'. Got {tensor_scope!r}."
        logger.error(msg)
        raise ValueError(msg)

    if len(batch) == 1:
        return converter(batch[0])
    return [converter(item) for item in batch]
