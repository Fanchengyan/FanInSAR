"""Array algorithms shared by Stack and Network processing stages."""

from __future__ import annotations

import math
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


def detrend_surface(
    data: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    x_degree: int = 1,
    y_degree: int = 1,
    max_num_sample: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove a low-order polynomial trend surface from 2-D maps.

    For every 2-D slice ``(H, W)`` of ``data`` (all leading dimensions are
    treated as independent slices), a polynomial surface is fitted by
    unweighted least squares on the valid pixels and then subtracted.

    ``x_degree`` and ``y_degree`` describe two *different* spatial
    directions. ``x_degree`` is the polynomial degree along the x
    dimension (the last axis: columns / image width, i.e. the radar range
    direction), and ``y_degree`` is the degree along the y dimension
    (rows / image height, i.e. the radar azimuth direction). A degree of
    ``0`` means the surface does not depend on that direction at all.
    The fitted basis is the constant ``1``, all pure powers of ``x`` up
    to ``x_degree``, all pure powers of ``y`` up to ``y_degree``, and the
    cross term ``x * y`` only when both degrees are at least 2. In
    particular:

    - ``x_degree=1, y_degree=1`` (default): tilted plane ``b*x + a*y + c``
    - ``x_degree=2, y_degree=0``: parabola along x only, no y dependence
    - ``x_degree=2, y_degree=1``: asymmetric surface with ``x**2`` but no
      cross term; the ``x * y`` cross term requires ``(2, 2)``
    - ``x_degree=0, y_degree=0``: constant, i.e. per-slice mean removal

    Non-finite pixels are excluded from the fit and keep their NaN value
    in the output. Pixels excluded by ``mask`` still have the fitted
    surface subtracted; they only do not contribute to the estimation.

    The fit accumulates separable moments and solves a scaled
    ``n_param x n_param`` normal-equation system in float64 without ever
    materializing the ``(n_pixel, n_param)`` design matrix. The
    evaluation and subtraction run in the output dtype.

    Parameters
    ----------
    data : torch.Tensor
        Floating-point tensor of shape ``(H, W)`` or ``(..., H, W)``.
        Leading dimensions are fitted independently, slice by slice.
    mask : torch.Tensor, optional
        2-D boolean or numeric mask of shape ``(H, W)`` where nonzero
        marks pixels used in the fit. If None, all finite pixels are
        used. Default is None.
    x_degree : int, optional
        Polynomial degree along x (columns / range), in ``[0, 3]``.
        Default is 1.
    y_degree : int, optional
        Polynomial degree along y (rows / azimuth), in ``[0, 3]``.
        Default is 1.
    max_num_sample : int, optional
        Memory fuse: when a slice has more valid pixels than this value,
        a deterministic stride grid subsamples the pixels used in the
        fit. None (default) uses all valid pixels.
    device : torch.device or str, optional
        Device for the computation and the returned tensors. None keeps
        the device of ``data``. Default is None.
    dtype : torch.dtype, optional
        Dtype of the returned tensors. None follows the dtype of
        ``data``. The internal fit always runs in float64 regardless.
        Default is None.

    Returns
    -------
    detrended : torch.Tensor
        ``data`` minus the fitted surface, with the shape, dtype, and
        device described above. Non-finite input pixels stay NaN.
    surface : torch.Tensor
        The fitted trend surface with the same shape, dtype, and device.

    Raises
    ------
    ValueError
        If a degree is not an integer in ``[0, 3]``, if ``data`` is not
        floating-point, if ``mask`` does not match the spatial shape, if
        a slice has fewer valid pixels than surface parameters, or if
        the normal equations are singular for some slice.

    Examples
    --------
    >>> import torch
    >>> from faninsar.processing.alg import detrend_surface
    >>> yy, xx = torch.meshgrid(
    ...     torch.arange(5, dtype=torch.float64),
    ...     torch.arange(7, dtype=torch.float64),
    ...     indexing="ij",
    ... )
    >>> data = 3.0 * xx - 2.0 * yy + 1.0
    >>> detrended, surface = detrend_surface(data)
    >>> bool(detrended.abs().max() < 1e-10)
    True

    A pure-x parabola is removed exactly while leaving y untouched:

    >>> data = 0.5 * xx**2 + 2.0
    >>> detrended, _ = detrend_surface(data, x_degree=2, y_degree=0)
    >>> bool(detrended.abs().max() < 1e-10)
    True

    """
    for name, degree in (("x_degree", x_degree), ("y_degree", y_degree)):
        if (
            isinstance(degree, bool)
            or not isinstance(degree, int)
            or not 0 <= degree <= 3
        ):
            msg = f"{name} must be an integer in [0, 3], got {degree!r}"
            logger.error(msg)
            raise ValueError(msg)
    if not data.dtype.is_floating_point:
        msg = f"data must have a floating dtype, got {data.dtype}"
        logger.error(msg)
        raise ValueError(msg)
    if data.ndim < 2:
        msg = f"data must have at least 2 dimensions, got shape {tuple(data.shape)}"
        logger.error(msg)
        raise ValueError(msg)
    if mask is not None and (
        mask.ndim != 2 or tuple(mask.shape) != tuple(data.shape[-2:])
    ):
        msg = (
            f"mask must be 2-D with shape {tuple(data.shape[-2:])} to match the "
            f"spatial dimensions of data, got shape {tuple(mask.shape)}"
        )
        logger.error(msg)
        raise ValueError(msg)
    if max_num_sample is not None and (
        isinstance(max_num_sample, bool)
        or not isinstance(max_num_sample, int)
        or max_num_sample <= 0
    ):
        msg = f"max_num_sample must be a positive integer, got {max_num_sample!r}"
        logger.error(msg)
        raise ValueError(msg)
    if dtype is not None and not dtype.is_floating_point:
        msg = f"dtype must be a floating dtype, got {dtype}"
        logger.error(msg)
        raise ValueError(msg)

    basis = [(0, 0)]
    basis.extend((i, 0) for i in range(1, y_degree + 1))
    basis.extend((0, j) for j in range(1, x_degree + 1))
    if x_degree >= 2 and y_degree >= 2:
        basis.append((1, 1))
    n_param = len(basis)

    target = data.to(device=device) if device is not None else data
    out_dtype = dtype if dtype is not None else data.dtype
    lead_shape = target.shape[:-2]
    height, width = target.shape[-2], target.shape[-1]
    n_slice = lead_shape.numel() if lead_shape else 1

    slices = target.reshape(n_slice, height, width)
    valid = torch.isfinite(slices)
    if mask is not None:
        valid = valid & mask.to(device=target.device).to(torch.bool).unsqueeze(0)

    if max_num_sample is not None:
        n_valid_max = int(valid.sum(dim=(1, 2)).max().item())
        if n_valid_max > max_num_sample:
            step = math.ceil(math.sqrt(n_valid_max / max_num_sample))
            keep = torch.zeros(height, width, dtype=torch.bool, device=target.device)
            keep[step // 2 :: step, step // 2 :: step] = True
            valid = valid & keep.unsqueeze(0)

    n_valid = valid.sum(dim=(1, 2))
    if bool((n_valid < n_param).any()):
        bad = torch.nonzero(n_valid < n_param).flatten().tolist()
        msg = (
            f"At least {n_param} valid pixels per slice are required for the "
            f"requested surface, but slice(s) starting at {bad[:5]} have as few "
            f"as {int(n_valid.min().item())} valid pixels."
        )
        logger.error(msg)
        raise ValueError(msg)

    acc = torch.float64
    y_lin = torch.arange(height, dtype=acc, device=target.device)
    x_lin = torch.arange(width, dtype=acc, device=target.device)
    max_pow = 2 * max(x_degree, y_degree, 1)
    y_pow = [torch.ones(height, dtype=acc, device=target.device)]
    x_pow = [torch.ones(width, dtype=acc, device=target.device)]
    for _ in range(max_pow):
        y_pow.append(y_pow[-1] * y_lin)
        x_pow.append(x_pow[-1] * x_lin)

    values = slices.to(acc)
    valid_f = valid.to(acc)
    masked = torch.where(
        valid, values, torch.zeros((), dtype=acc, device=target.device)
    )

    moments = torch.empty(n_param, n_slice, dtype=acc, device=target.device)
    for idx, (i, j) in enumerate(basis):
        row = torch.einsum("h,ehw->ew", y_pow[i], masked)
        moments[idx] = torch.einsum("w,ew->e", x_pow[j], row)

    rows_valid: dict[int, torch.Tensor] = {}
    normal = torch.empty(n_param, n_param, n_slice, dtype=acc, device=target.device)
    for ai, (i1, j1) in enumerate(basis):
        for bi, (i2, j2) in enumerate(basis):
            power = i1 + i2
            if power not in rows_valid:
                rows_valid[power] = torch.einsum("h,ehw->ew", y_pow[power], valid_f)
            normal[ai, bi] = torch.einsum("w,ew->e", x_pow[j1 + j2], rows_valid[power])

    # diagonal over the (n_param, n_param) axes -> (n_slice, n_param)
    scale = torch.sqrt(
        torch.diagonal(normal, dim1=0, dim2=1).clamp_min(torch.finfo(acc).tiny)
    )
    normal_scaled = normal / (scale.T[:, None, :] * scale.T[None, :, :])
    moments_scaled = moments / scale.T
    try:
        coef_scaled = torch.linalg.solve(
            normal_scaled.permute(2, 0, 1),
            moments_scaled.T.unsqueeze(-1),
        ).squeeze(-1)
    except torch.linalg.LinAlgError as exc:
        msg = (
            "The detrend surface system is singular for at least one slice; "
            "the valid pixels may not constrain the requested degrees."
        )
        logger.exception(msg)
        raise ValueError(msg) from exc
    coef = coef_scaled / scale

    y_eval = [tensor.to(out_dtype) for tensor in y_pow]
    x_eval = [tensor.to(out_dtype) for tensor in x_pow]
    coef_eval = coef.to(out_dtype)
    surface = torch.zeros(n_slice, height, width, dtype=out_dtype, device=target.device)
    for idx, (i, j) in enumerate(basis):
        surface += (
            coef_eval[:, idx, None, None]
            * y_eval[i][None, :, None]
            * x_eval[j][None, None, :]
        )
    detrended = target.to(out_dtype) - surface

    out_shape = (*lead_shape, height, width)
    return detrended.reshape(out_shape), surface.reshape(out_shape)
