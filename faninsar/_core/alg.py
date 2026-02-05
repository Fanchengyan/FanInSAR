from __future__ import annotations

from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
import torch

from faninsar.logging import setup_logger

logger = setup_logger(__name__)


def split_group(
    total: int,
    *,
    group_size: int | None = None,
    group_num: int | None = None,
) -> np.ndarray:
    """Split `total` into groups of size `group_size` or number of groups `group_num`.

    Parameters
    ----------
    total : int
        Total number of items to split.
    group_size : int, optional
        Size of each group. Must be specified if :param:`group_num` is not
        specified.
    group_num : int, optional
        Number of groups. Must be specified if :param:`group_size` is not
        specified.

    Returns
    -------
    groups : np.ndarray
        Array of group indices for each item.

    Raises
    ------
    ValueError
        If neither :param:`group_size` nor :param:`group_num` is specified.

    Examples
    --------
    >>> split_group(10, group_size=3)
    array([0., 0., 0., 1., 1., 1., 2., 2., 2., 3.])

    >>> split_group(10, group_num=3)
    array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])

    """
    if group_size is None:
        if group_num is None:
            msg = "Must specify either group_size or group_num"
            logger.error(msg)
            raise ValueError(msg)
        # use floor to automatically balance the group size
        return np.arange(total) * group_num // total
    return np.arange(total) // group_size


@overload
def gradient_magnitude(
    img: torch.Tensor,
    channel_axis: Literal[0, -1, 2] = 0,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor: ...


@overload
def gradient_magnitude(
    img: npt.NDArray[np.floating[Any]],
    channel_axis: Literal[0, -1, 2] = 0,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> npt.NDArray[np.floating[Any]]: ...


def gradient_magnitude(
    img: torch.Tensor | npt.NDArray[np.floating[Any]],
    channel_axis: Literal[0, -1, 2] = 0,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | npt.NDArray[np.floating[Any]]:
    """Calculate gradient magnitude for 2D or 3D images.

    Gradient is computed directly using differences between adjacent pixels.

    Parameters
    ----------
    img: torch.Tensor | npt.NDArray[np.floating[Any]]
        2D array (H, W) or 3D array with channels
        - If channel_axis=0: (C, H, W)
        - If channel_axis=-1 or 2: (H, W, C)
    channel_axis: Literal[0, -1, 2], optional
        Position of channel dimension (0 or -1/2). Ignored for 2D images.
        Default is 0.
    device: str | torch.device | None, optional
        Device to move tensor to. If None, keeps original device.
    dtype: torch.dtype, optional
        Data type for the tensor. Default is torch.float32.

    Returns
    -------
    grad: torch.Tensor | npt.NDArray[np.floating[Any]]
        Gradient magnitude array with the same array container type as ``img``:
        - 2D input (H, W) -> output (H-1, W-1)
        - 3D input with channel_axis=0: (C, H, W) -> (C, H-1, W-1)
        - 3D input with channel_axis=-1: (H, W, C) -> (H-1, W-1, C)

    """
    # Determine original input type to preserve output type
    input_is_numpy = isinstance(img, np.ndarray)

    if isinstance(img, torch.Tensor):
        tensor_img = img.to(device=device, dtype=dtype)
    else:
        tensor_img = torch.as_tensor(img, dtype=dtype, device=device)

    # Check if 2D or 3D
    if tensor_img.ndim == 2:
        # 2D image: (H, W)
        grad_x = tensor_img[:, 1:] - tensor_img[:, :-1]
        grad_y = tensor_img[1:, :] - tensor_img[:-1, :]
        grad_x_cropped = grad_x[:-1, :]
        grad_y_cropped = grad_y[:, :-1]
        grad = torch.sqrt(grad_x_cropped**2 + grad_y_cropped**2)

    elif tensor_img.ndim == 3:
        # 3D image: channels can be at different positions
        if channel_axis == 0:
            # Format: (C, H, W)
            grad_x = tensor_img[:, :, 1:] - tensor_img[:, :, :-1]
            grad_y = tensor_img[:, 1:, :] - tensor_img[:, :-1, :]
            grad_x_cropped = grad_x[:, :-1, :]
            grad_y_cropped = grad_y[:, :, :-1]
        elif channel_axis in {-1, 2}:
            # Format: (H, W, C)
            grad_x = tensor_img[:, 1:, :] - tensor_img[:, :-1, :]
            grad_y = tensor_img[1:, :, :] - tensor_img[:-1, :, :]
            grad_x_cropped = grad_x[:-1, :, :]
            grad_y_cropped = grad_y[:, :-1, :]
        else:
            msg = f"Invalid channel_axis: {channel_axis}, must be 0 or -1/2"
            raise ValueError(msg)

        # Calculate gradient magnitude using torch
        grad = torch.sqrt(grad_x_cropped**2 + grad_y_cropped**2)

    else:
        msg = f"Input image must be 2D or 3D, got {tensor_img.ndim}D"
        raise ValueError(msg)

    if input_is_numpy:
        return grad.detach().cpu().numpy()

    return grad


def percentile_range(
    data: npt.ArrayLike,
    min_percent: float = 0,
    max_percent: float = 100,
    symmetric: bool = False,
) -> tuple[float, float]:
    """Compute percentile-based value range from numeric data percentiles.

    Parameters
    ----------
    data : numpy.ndarray | torch.Tensor | array-like
        Input data for which to compute the percentile range.
    min_percent : float, optional
        Percentile used for the lower bound. Must be within [0, 100]. Default is 0.
    max_percent : float, optional
        Percentile used for the upper bound. Must be within [0, 100]. Default is 100.
    symmetric : bool, optional
        If True, return symmetric range around zero using the larger absolute
        percentile value. Default is False.

    Returns
    -------
    tuple of float
        A tuple ``(vmin, vmax)`` representing the computed ragne.

    Raises
    ------
    TypeError
        If ``data`` cannot be converted to a numeric NumPy array.
    ValueError
        If percentiles lie outside [0, 100] or no finite values are present.

    """
    if isinstance(data, torch.Tensor):
        array = data.detach().cpu().numpy()
    else:
        try:
            array = np.asarray(data)
        except (TypeError, ValueError) as exc:
            msg = "`data` must be convertible to a numeric array."
            raise TypeError(msg) from exc

    if not np.issubdtype(array.dtype, np.number):
        array = np.asarray(array, dtype=np.float64)

    for name, percent in (("min_percent", min_percent), ("max_percent", max_percent)):
        if not (0.0 <= percent <= 100.0):
            msg = f"`{name}` must be within [0, 100]."
            raise ValueError(msg)

    finite = array[np.isfinite(array)]
    if finite.size == 0:
        msg = "`data` must contain at least one finite value."
        raise ValueError(msg)

    low_percent, high_percent = sorted((min_percent, max_percent))
    vmin = float(np.percentile(finite, low_percent))
    vmax = float(np.percentile(finite, high_percent))

    if symmetric:
        bound = max(abs(vmin), abs(vmax))
        return -bound, bound

    return vmin, vmax
