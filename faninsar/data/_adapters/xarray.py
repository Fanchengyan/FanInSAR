"""Xarray accessor registrations."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np
import rioxarray  # noqa: F401
import torch
import xarray as xr
from pyproj.crs import CRS

from faninsar.data.query.points import Points
from faninsar.io.export.kmz import dataarray2kmz
from faninsar.logging import setup_logger
from faninsar.processing.alg import detrend_surface as _detrend_surface

if TYPE_CHECKING:
    from os import PathLike

    from faninsar.core.types import CrsLike

logger = setup_logger(__name__)

_AUTO_DIM_NAME = "auto"
_SPATIAL_DIM_PAIRS = (
    ("x", "y"),
    ("xs", "ys"),
    ("x_coord", "y_coord"),
    ("x_coords", "y_coords"),
    ("xcoord", "ycoord"),
    ("xcoords", "ycoords"),
    ("lon", "lat"),
    ("lons", "lats"),
    ("lng", "lat"),
    ("lngs", "lats"),
    ("long", "lat"),
    ("longs", "lats"),
    ("longitude", "latitude"),
    ("longitudes", "latitudes"),
    ("easting", "northing"),
    ("eastings", "northings"),
    ("east", "north"),
    ("easts", "norths"),
    ("col", "row"),
    ("cols", "rows"),
    ("column", "row"),
    ("columns", "rows"),
    ("sample", "line"),
    ("samples", "lines"),
    ("range", "azimuth"),
    ("ranges", "azimuths"),
)


def _resolve_auto_spatial_dims(
    data_array: xr.DataArray,
    x_dim: str,
    y_dim: str,
) -> tuple[str, str]:
    """Resolve spatial dimension names for a data array.

    Parameters
    ----------
    data_array : xarray.DataArray
        Data array whose dimensions are inspected.
    x_dim : str
        Name of the x dimension, or ``"auto"`` to infer it.
    y_dim : str
        Name of the y dimension, or ``"auto"`` to infer it.

    Returns
    -------
    tuple[str, str]
        Resolved x and y dimension names.

    Raises
    ------
    ValueError
        If either dimension cannot be inferred or is not present in the data
        array dimensions.

    """
    dimension_names = tuple(data_array.dims)
    lower_to_dimension = {name.lower(): name for name in dimension_names}

    resolved_x_dim = x_dim
    resolved_y_dim = y_dim
    if x_dim == _AUTO_DIM_NAME and y_dim == _AUTO_DIM_NAME:
        for x_candidate, y_candidate in _SPATIAL_DIM_PAIRS:
            if x_candidate in lower_to_dimension and y_candidate in lower_to_dimension:
                return lower_to_dimension[x_candidate], lower_to_dimension[y_candidate]

        msg = (
            "Cannot infer spatial dimensions automatically. Expected one of "
            f"{_SPATIAL_DIM_PAIRS}, but found dimensions {dimension_names}. "
            "Please pass x_dim and y_dim explicitly."
        )
        logger.error(msg)
        raise ValueError(msg)

    if x_dim == _AUTO_DIM_NAME:
        for x_candidate, y_candidate in _SPATIAL_DIM_PAIRS:
            if (
                y_candidate in lower_to_dimension
                and lower_to_dimension[y_candidate] == y_dim
                and x_candidate in lower_to_dimension
            ):
                resolved_x_dim = lower_to_dimension[x_candidate]
                break
        else:
            msg = (
                f"Cannot infer x dimension for y_dim={y_dim!r}. "
                f"Found dimensions {dimension_names}. Please pass x_dim explicitly."
            )
            logger.error(msg)
            raise ValueError(msg)

    if y_dim == _AUTO_DIM_NAME:
        for x_candidate, y_candidate in _SPATIAL_DIM_PAIRS:
            if (
                x_candidate in lower_to_dimension
                and lower_to_dimension[x_candidate] == x_dim
                and y_candidate in lower_to_dimension
            ):
                resolved_y_dim = lower_to_dimension[y_candidate]
                break
        else:
            msg = (
                f"Cannot infer y dimension for x_dim={x_dim!r}. "
                f"Found dimensions {dimension_names}. Please pass y_dim explicitly."
            )
            logger.error(msg)
            raise ValueError(msg)

    missing_dimensions = [
        dimension_name
        for dimension_name in (resolved_x_dim, resolved_y_dim)
        if dimension_name not in dimension_names
    ]
    if missing_dimensions:
        msg = (
            f"Spatial dimensions {missing_dimensions} are not present in the "
            f"data array dimensions {dimension_names}."
        )
        logger.error(msg)
        raise ValueError(msg)

    return resolved_x_dim, resolved_y_dim


@xr.register_dataarray_accessor("fis")
class FanInSARDataArrayAccessor:
    """FanInSAR convenience methods for :class:`xarray.DataArray`.

    Parameters
    ----------
    data_array : xarray.DataArray
        Data array bound to the accessor.

    Attributes
    ----------
    data_array : xarray.DataArray
        Data array bound to the accessor.

    """

    def __init__(self, data_array: xr.DataArray) -> None:
        self.data_array = data_array

    def set_spatial_ref(
        self,
        crs: CrsLike,
        x_dim: str = _AUTO_DIM_NAME,
        y_dim: str = _AUTO_DIM_NAME,
    ) -> xr.DataArray:
        """Set rioxarray spatial dimensions and coordinate reference system.

        Parameters
        ----------
        crs : CrsLike
            Coordinate reference system. It can be any value accepted by
            :meth:`pyproj.crs.CRS.from_user_input`.
        x_dim : str, optional
            Name of the x dimension. Use ``"auto"`` to infer a common spatial
            dimension name. Default is ``"auto"``.
        y_dim : str, optional
            Name of the y dimension. Use ``"auto"`` to infer a common spatial
            dimension name. Default is ``"auto"``.

        Returns
        -------
        xarray.DataArray
            Data array with rioxarray spatial dimensions and CRS metadata.

        Raises
        ------
        ValueError
            If spatial dimensions cannot be inferred or the selected dimension
            names are not present in the data array.

        Examples
        --------
        >>> import faninsar  # Register the ``fis`` accessor.
        >>> import xarray as xr
        >>> data_array = xr.DataArray([[1]], dims=("y", "x"))
        >>> spatial_ref = data_array.fis.set_spatial_ref(crs="EPSG:4326")
        >>> spatial_ref.rio.x_dim
        'x'

        """
        resolved_x_dim, resolved_y_dim = _resolve_auto_spatial_dims(
            self.data_array,
            x_dim,
            y_dim,
        )
        data_array = self.data_array.rio.set_spatial_dims(
            x_dim=resolved_x_dim,
            y_dim=resolved_y_dim,
        )
        return data_array.rio.write_crs(crs)

    def to_kmz(
        self,
        out_file: PathLike,
        render_scale: int = 4,
        img_kwargs: dict[str, Any] | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        verbose: bool = True,
        *,
        tile_size: int = 256,
        min_lod_pixels: int = 128,
    ) -> None:
        """Write the data array into a KMZ file.

        Parameters
        ----------
        out_file : str or PathLike
            Path of the KMZ file.
        render_scale : int, optional
            Positive integer scale factor used to repeat source pixels before
            rendering, improving pixel-level clarity in Google Earth.
        img_kwargs : dict[str, Any] | None, optional
            Keyword arguments for :func:`matplotlib.pyplot.imshow`.
        cbar_kwargs : dict[str, Any] | None, optional
            Keyword arguments for :func:`faninsar.io.export.kmz.save_colorbar`,
            excluding ``out_file`` and ``mappable``.
        verbose : bool, optional
            Whether to log the output path.
        tile_size : int, optional
            Maximum tile size in pixels.
        min_lod_pixels : int, optional
            Minimum screen-space threshold used by child ``NetworkLink``
            regions.

        Raises
        ------
        ValueError
            If the data array does not have rioxarray CRS or spatial dimension
            metadata, or if any tiling parameter is invalid.

        """
        dataarray2kmz(
            self.data_array,
            out_file,
            render_scale=render_scale,
            img_kwargs=img_kwargs,
            cbar_kwargs=cbar_kwargs,
            verbose=verbose,
            tile_size=tile_size,
            min_lod_pixels=min_lod_pixels,
        )

    def detrend_surface(
        self,
        x_degree: int = 1,
        y_degree: int = 1,
        *,
        mask: xr.DataArray | None = None,
        ref_points: Points | None = None,
        return_surface: bool = False,
        max_num_sample: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        x_dim: str = _AUTO_DIM_NAME,
        y_dim: str = _AUTO_DIM_NAME,
    ) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
        """Remove a low-order polynomial trend surface from this data array.

        For every 2-D ``(y, x)`` slice of the data array (each ``pair`` or
        ``time`` entry, for example), a polynomial surface is fitted by
        unweighted least squares on the valid pixels and then subtracted.

        ``x_degree`` and ``y_degree`` control two *different* spatial
        directions:

        - ``x_degree`` is the polynomial degree along the **x dimension**
          (columns, image width; the radar range direction). ``0`` means
          the surface does not vary along x at all.
        - ``y_degree`` is the polynomial degree along the **y dimension**
          (rows, image height; the radar azimuth direction). ``0`` means
          the surface does not vary along y at all.

        Setting only one degree greater than zero fits a 1-D polynomial
        along that single direction. Setting **both** degrees greater than
        zero fits a full 2-D surface; the ``x * y`` cross term is included
        once both degrees reach 2. For example ``(1, 1)`` (default) removes
        a tilted plane, ``(2, 2)`` removes a full quadratic surface
        including the cross term, ``(2, 0)`` removes a range-only
        parabola, and ``(0, 0)`` removes the per-slice mean of the valid
        pixels.

        Non-finite (NaN) pixels are excluded from the fit and stay NaN in
        the output. Pixels excluded by ``mask`` are still corrected by the
        fitted surface; they only do not contribute to the estimation.

        When ``ref_points`` is given, the mean detrended value sampled at
        those points (nearest pixel, ``nanmean`` over points) is
        subtracted from every slice *after* detrending, so the output is
        referenced to zero at the reference points. ``ref_points`` with a
        CRS are automatically transformed to the CRS of the data array
        (see :meth:`set_spatial_ref`); points without a CRS are assumed
        to already be in the array coordinates.

        Parameters
        ----------
        x_degree : int, optional
            Polynomial degree along x (columns / range), in ``[0, 3]``.
            Default is 1.
        y_degree : int, optional
            Polynomial degree along y (rows / azimuth), in ``[0, 3]``.
            Default is 1.
        mask : xarray.DataArray, optional
            2-D mask over the spatial dims where nonzero / True marks
            pixels used in the fit, e.g. ``mask=coh >= 0.5``. If None,
            all finite pixels are used. Default is None.
        ref_points : Points, optional
            Reference points
            (:class:`faninsar.data.query.Points`) subtracted after
            detrending. If None, nothing is subtracted. Default is None.
        return_surface : bool, optional
            If True, also return the fitted surface so that
            ``detrended + surface`` reconstructs the input. Default is
            False.
        max_num_sample : int, optional
            Memory fuse passed to
            :func:`faninsar.processing.alg.detrend_surface`. None
            (default) fits on all valid pixels.
        device : torch.device or str, optional
            Torch device for the computation. None keeps everything on
            CPU (the xarray bridge is NumPy-based). Default is None.
        dtype : torch.dtype, optional
            Dtype of the output values. None keeps the dtype of the data
            array. Default is None.
        x_dim : str, optional
            Name of the x dimension. Use ``"auto"`` to infer it. Default
            is ``"auto"``.
        y_dim : str, optional
            Name of the y dimension. Use ``"auto"`` to infer it. Default
            is ``"auto"``.

        Returns
        -------
        xarray.DataArray or tuple of xarray.DataArray
            The detrended data array with the dims, coords, and attrs of
            the input, or a tuple ``(detrended, surface)`` when
            ``return_surface=True``.

        Raises
        ------
        ValueError
            If spatial dims cannot be inferred, if ``mask`` does not
            align with the data array, if the array has no CRS while the
            reference points have one, or if a slice has too few valid
            pixels for the requested surface.
        TypeError
            If ``mask`` or ``ref_points`` have an unexpected type.

        See Also
        --------
        faninsar.processing.alg.detrend_surface :
            The underlying Torch implementation.

        Examples
        --------
        >>> import faninsar  # Register the ``fis`` accessor.
        >>> import numpy as np
        >>> import xarray as xr
        >>> x = np.arange(6.0)
        >>> y = np.arange(4.0)
        >>> data = 2.0 * x[None, :] - 1.0 * y[:, None]
        >>> da = xr.DataArray(data, coords={"y": y, "x": x}, dims=("y", "x"))
        >>> detrended = da.fis.detrend_surface()
        >>> float(np.abs(detrended.values).max()) < 1e-10
        True

        Both degrees together give a 2-D surface; each pair is fitted
        independently:

        >>> da_pair = xr.DataArray(
        ...     np.stack([data, data * 0.5]),
        ...     coords={"pair": ["a", "b"], "y": y, "x": x},
        ...     dims=("pair", "y", "x"),
        ... )
        >>> detrended_pair, surface_pair = da_pair.fis.detrend_surface(
        ...     return_surface=True
        ... )
        >>> float(np.abs(detrended_pair.values).max()) < 1e-10
        True

        """
        resolved_x_dim, resolved_y_dim = _resolve_auto_spatial_dims(
            self.data_array,
            x_dim,
            y_dim,
        )
        data_array = self.data_array
        if data_array.chunks:
            data_array = data_array.load()

        other_dims = [
            dim
            for dim in data_array.dims
            if dim not in (resolved_x_dim, resolved_y_dim)
        ]
        transposed = data_array.transpose(*other_dims, resolved_y_dim, resolved_x_dim)

        mask_tensor = None
        if mask is not None:
            if not isinstance(mask, xr.DataArray):
                msg = f"mask must be an xarray.DataArray, got {type(mask).__name__}"
                logger.error(msg)
                raise TypeError(msg)
            if not set(mask.dims) <= {resolved_y_dim, resolved_x_dim}:
                msg = (
                    f"mask dims {tuple(mask.dims)} must be a subset of "
                    f"({resolved_y_dim!r}, {resolved_x_dim!r})."
                )
                logger.error(msg)
                raise ValueError(msg)
            mask = mask.transpose(resolved_y_dim, resolved_x_dim)
            transposed, mask = xr.align(transposed, mask, join="exact")
            mask_tensor = torch.as_tensor(np.ascontiguousarray(mask.values))

        tensor = torch.from_numpy(np.ascontiguousarray(transposed.values))
        detrended_t, surface_t = _detrend_surface(
            tensor,
            mask_tensor,
            x_degree=x_degree,
            y_degree=y_degree,
            max_num_sample=max_num_sample,
            device=device,
            dtype=dtype,
        )
        detrended = transposed.copy(data=detrended_t.detach().cpu().numpy())
        surface = transposed.copy(data=surface_t.detach().cpu().numpy())

        if ref_points is not None:
            if not isinstance(ref_points, Points):
                msg = (
                    "ref_points must be a faninsar.data.query.Points instance, "
                    f"got {type(ref_points).__name__}"
                )
                logger.error(msg)
                raise TypeError(msg)
            array_crs = None
            with contextlib.suppress(RuntimeError):
                array_crs = transposed.rio.crs
            points = ref_points
            if array_crs is None:
                if points.crs is not None:
                    msg = (
                        "The data array has no CRS, so reference points with a "
                        "CRS cannot be transformed. Call set_spatial_ref first "
                        "or pass points without a CRS."
                    )
                    logger.error(msg)
                    raise ValueError(msg)
            elif (
                points.crs is not None and CRS.from_user_input(array_crs) != points.crs
            ):
                points = points.to_crs(array_crs)

            x_indexer = xr.DataArray(
                np.asarray(points.x, dtype=np.float64), dims="point"
            )
            y_indexer = xr.DataArray(
                np.asarray(points.y, dtype=np.float64), dims="point"
            )
            sampled = detrended.sel(
                {resolved_x_dim: x_indexer, resolved_y_dim: y_indexer},
                method="nearest",
            )
            offset = sampled.mean(dim="point", skipna=True)
            if bool(offset.isnull().any()):
                msg = (
                    "All reference points sample NaN values for at least one "
                    "slice; choose reference points on valid pixels."
                )
                logger.error(msg)
                raise ValueError(msg)
            detrended = detrended - offset
            surface = surface + offset

        detrended = detrended.transpose(*data_array.dims)
        if return_surface:
            return detrended, surface.transpose(*data_array.dims)
        return detrended
