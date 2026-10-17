"""Thin forward-geocoding shell composing the LUT, coordinates, and kernels.

:class:`Geocoder` is the single entry point that owns one geographic-to-radar
lookup table and applies radar-domain layers onto its destination grid through
the kind-driven kernel table. Multi-layer flows that need shared coordinates
(TOPS reramp, joint validity, terrain phase) bypass :meth:`Geocoder.apply` and
call :meth:`Geocoder.coordinates` plus
:func:`faninsar.processing.geocoding.geo_resample.sample` directly, so
per-chunk coordinates are evaluated exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Callable

    from faninsar.processing.geocoding.coordinates import RadarCoordinates
    from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT
    from faninsar.processing.geocoding.geo_resample import GeocodeKind
    from faninsar.processing.geometry.api import DEM
    from faninsar.processing.geometry.grids import GridSpec

__all__ = ["GeocodedRaster", "Geocoder"]

logger = setup_logger(__name__)

_DEFAULT_CACHE_NAMESPACE = "geocoder"


@dataclass(frozen=True, slots=True)
class GeocodedRaster:
    """Geographic-domain raster produced by :meth:`Geocoder.apply`.

    Attributes
    ----------
    values : numpy.ndarray
        Sampled values on the destination grid. This is the caller-provided
        ``out`` array when one was supplied (for example a disk-backed
        memmap), otherwise an in-memory array allocated by the engine.
    valid : numpy.ndarray
        Boolean mask of destination pixels with admissible sampled values,
        narrowed by the request ``mask`` and any per-chunk correction.
    grid : GridSpec
        Grid of the produced frame: a pixel-centre-aligned view of the
        geocoder grid when the request windowed or multilooked it.

    """

    values: np.ndarray
    valid: np.ndarray
    grid: GridSpec


class Geocoder:
    """Forward-geocoding engine for one geometry, grid, and DEM context.

    The engine lazily builds (or accepts) one :class:`Geo2RdrLUT` and turns it
    into :class:`RadarCoordinates` on demand; layers are applied with the
    kernel uniquely determined by their physical ``kind``. It is a thin shell:
    coordinate math lives in :class:`RadarCoordinates`, kernels in
    :func:`faninsar.processing.geocoding.geo_resample.sample`, and disk
    identity in :func:`faninsar.processing.geocoding.geo_lut.build_geo2rdr_lut`.

    Notes
    -----
    Instances cache the lookup table after first use and are not thread-safe;
    build one engine per thread or share the built :attr:`lut` explicitly.

    """

    def __init__(
        self,
        geometry: object,
        grid: GridSpec,
        *,
        dem: DEM | None,
        full_radar_shape: tuple[int, int],
        bounds: tuple[int, int, int, int] | None = None,
        fallback_height: float = 0.0,
        device: str = "auto",
        cache_dir: str | Path | None = None,
        cache_key: str | None = None,
        storage_dir: str | Path | None = None,
        lut: Geo2RdrLUT | None = None,
    ) -> None:
        """Store the geocoding context; the lookup table builds lazily.

        Parameters
        ----------
        geometry : RadarGeometryModel
            Reference-scene radar geometry forwarded to the LUT builder.
        grid : GridSpec
            Destination geographic grid.
        dem : DEM or None
            Per-pixel ellipsoidal height source; ``None`` solves at
            ``fallback_height``.
        full_radar_shape : tuple[int, int]
            Full-resolution radar image shape.
        bounds : tuple[int, int, int, int], optional
            Half-open ``(row0, row1, col0, col1)`` window of ``grid`` covered
            by this engine. ``None`` covers the full grid.
        fallback_height : float, optional
            Fallback ellipsoidal height used where no DEM sample exists.
        device : str, optional
            Execution device forwarded to the LUT solve and the kernels;
            ``auto`` resolves through the standard device policy.
        cache_dir : str or pathlib.Path, optional
            Shared content-addressed LUT cache root. Requires
            ``storage_dir``; ``cache_key`` defaults to a generic namespace
            because the on-disk identity is content-derived.
        cache_key : str, optional
            Readable cache namespace (scene, swath, burst). Never part of the
            entry identity.
        storage_dir : str or pathlib.Path, optional
            Directory for the returned disk-backed LUT arrays; required when
            ``cache_dir`` is given, mirroring ``build_geo2rdr_lut``.
        lut : Geo2RdrLUT, optional
            Pre-built lookup table that bypasses ``build_geo2rdr_lut``. It
            must match ``full_radar_shape`` and ``bounds``.

        Raises
        ------
        InvalidProcessingStateError
            If bounds are malformed, caching is half-configured, or a
            pre-built LUT does not match the declared context.

        """
        if bounds is not None and not (
            isinstance(bounds, tuple)
            and len(bounds) == 4
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in bounds
            )
        ):
            reject_invalid_state(
                "bounds must be a (row0, row1, col0, col1) integer tuple; "
                f"got {bounds!r}"
            )
        if (cache_dir is None) != (storage_dir is None):
            reject_invalid_state(
                "Geocoder caching requires both cache_dir and storage_dir"
            )
        self._geometry = geometry
        self._grid = grid
        self._dem = dem
        self._full_radar_shape = (int(full_radar_shape[0]), int(full_radar_shape[1]))
        self._bounds = bounds
        self._fallback_height = float(fallback_height)
        self._device = device
        self._cache_dir = None if cache_dir is None else Path(cache_dir)
        self._cache_key = cache_key
        self._storage_dir = None if storage_dir is None else Path(storage_dir)
        self._lut = lut
        if lut is not None:
            if tuple(lut.full_radar_shape) != self._full_radar_shape:
                reject_invalid_state(
                    f"pre-built Geo2RdrLUT radar shape {lut.full_radar_shape} "
                    f"does not match Geocoder full_radar_shape "
                    f"{self._full_radar_shape}"
                )
            if bounds is not None and tuple(bounds) != self._coverage:
                reject_invalid_state(
                    "pre-built Geo2RdrLUT coverage does not match Geocoder bounds"
                )

    @property
    def device(self) -> str:
        """Return the configured execution device."""
        return self._device

    @property
    def grid(self) -> GridSpec:
        """Return the destination geographic grid."""
        return self._grid

    @property
    def _coverage(self) -> tuple[int, int, int, int]:
        """Return the ``(row0, row1, col0, col1)`` coverage of the LUT frame."""
        lut = self._lut
        if lut is not None:
            row0, col0 = int(lut.row0), int(lut.col0)
            return (row0, row0 + lut.shape[0], col0, col0 + lut.shape[1])
        if self._bounds is not None:
            return (
                int(self._bounds[0]),
                int(self._bounds[1]),
                int(self._bounds[2]),
                int(self._bounds[3]),
            )
        return (0, int(self._grid.height), 0, int(self._grid.width))

    @property
    def lut(self) -> Geo2RdrLUT:
        """Return the lookup table, building it on first access."""
        if self._lut is None:
            from faninsar.processing.geocoding.geo_lut import build_geo2rdr_lut

            row0, row1, col0, col1 = self._coverage
            cropped = (row0, col0) != (0, 0) or (row1, col1) != (
                int(self._grid.height),
                int(self._grid.width),
            )
            self._lut = build_geo2rdr_lut(
                geometry=self._geometry,
                grid=self._grid,
                full_radar_shape=self._full_radar_shape,
                height_m=self._fallback_height,
                dem=self._dem,
                device=self._device,
                storage_dir=self._storage_dir,
                cache_dir=self._cache_dir,
                cache_key=(
                    (self._cache_key or _DEFAULT_CACHE_NAMESPACE)
                    if self._cache_dir is not None
                    else None
                ),
                row_range=(row0, row1) if cropped else None,
                col_range=(col0, col1) if cropped else None,
            )
        return self._lut

    def coordinates(
        self,
        rows: slice | tuple[int, int] | None = None,
        *,
        window: tuple[int, int, int, int] | None = None,
        looks: tuple[int, int] = (1, 1),
    ) -> RadarCoordinates:
        """Return radar source coordinates for one destination frame.

        Parameters
        ----------
        rows : slice or tuple[int, int] or None, optional
            Destination rows of the engine's own frame (its LUT coverage);
            ``None`` selects every row.
        window : tuple[int, int, int, int], optional
            Absolute ``(row0, row1, col0, col1)`` sub-window of the
            destination grid; must lie inside :attr:`lut`. Mutually exclusive
            with ``rows``.
        looks : tuple[int, int], optional
            Positive ``(azimuth, range)`` look factors of the destination
            layer. ``(1, 1)`` yields one coordinate per window pixel; other
            factors yield one coordinate per look block, taken at the
            box-average block centre — the exact centre row for odd factors
            and the mean of the two bracketing rows for even factors — mapped
            into the layer frame with
            :meth:`RadarCoordinates.to_layer_frame`.

        Returns
        -------
        RadarCoordinates
            Coordinates in the layer frame described by ``window`` and
            ``looks``.

        """
        from faninsar.processing.geocoding.coordinates import RadarCoordinates

        az_looks, rg_looks = _validate_looks(looks)
        frame_row, frame_col, _frame_height, _frame_width = self._frame_selection(
            rows,
            window,
        )
        lut = self.lut
        azimuth = lut.az_full[frame_row, frame_col]
        range_index = lut.rg_full[frame_row, frame_col]
        valid = lut.valid[frame_row, frame_col]
        if (az_looks, rg_looks) != (1, 1):
            block_az = _center_reduce(azimuth, az_looks, axis=0)
            block_az = _center_reduce(block_az, rg_looks, axis=1)
            block_rg = _center_reduce(range_index, az_looks, axis=0)
            block_rg = _center_reduce(block_rg, rg_looks, axis=1)
            block_valid = _center_reduce_valid(valid, az_looks, axis=0)
            block_valid = _center_reduce_valid(block_valid, rg_looks, axis=1)
            return RadarCoordinates(block_az, block_rg, block_valid).to_layer_frame(
                origin=(0, 0),
                looks=(az_looks, rg_looks),
            )
        return RadarCoordinates(azimuth, range_index, valid)

    def apply(
        self,
        layer: np.ndarray,
        *,
        kind: GeocodeKind,
        window: tuple[int, int, int, int] | None = None,
        looks: tuple[int, int] = (1, 1),
        mask: np.ndarray | object | None = None,
        polygon_dilate_px: int = 0,
        correction: Callable[[RadarCoordinates], RadarCoordinates] | None = None,
        out: np.ndarray | None = None,
        rows_per_chunk: int | None = None,
        on_chunk: Callable[[slice], None] | None = None,
    ) -> GeocodedRaster:
        """Resample one radar layer onto the destination grid.

        The destination is iterated in row blocks; each block's coordinates
        pass through ``correction`` first (offset composition, layer-frame
        mapping), then through the kernel selected by ``kind``.

        Parameters
        ----------
        layer : numpy.ndarray
            2-D radar-domain source layer.
        kind : {"complex", "phase", "continuous", "label"}
            Physical kind selecting the resampling kernel.
        window : tuple[int, int, int, int], optional
            Absolute destination window as in :meth:`coordinates`.
        looks : tuple[int, int], optional
            Destination look factors as in :meth:`coordinates`.
        mask : numpy.ndarray or shapely geometry, optional
            Application-time validity restriction. A boolean array must match
            the destination frame; a shapely ``(lon, lat)`` Polygon or
            MultiPolygon is rasterized on the destination look blocks.
        polygon_dilate_px : int, optional
            Dilation applied when ``mask`` is a geometry.
        correction : callable, optional
            Per-chunk ``RadarCoordinates -> RadarCoordinates`` transform
            (dense offset composition, crop-to-layer mapping).
        out : numpy.ndarray, optional
            Slice-assignable destination for the values (``ndarray``,
            ``memmap``, ...). Allocated in memory when omitted; the validity
            mask is always allocated in memory.
        rows_per_chunk : int, optional
            Destination rows per block; the full frame is one block when
            omitted.
        on_chunk : callable, optional
            Callback invoked with each completed destination row slice, after
            its values are written; host page eviction and watchdog sampling
            hook in here.

        Returns
        -------
        GeocodedRaster
            Values, validity, and the pixel-centre-aligned destination grid.

        """
        from faninsar.processing.geocoding.coordinates import RadarCoordinates
        from faninsar.processing.geocoding.geo_resample import sample

        _validate_looks(looks)
        frame_row, frame_col, frame_height, frame_width = self._frame_selection(
            None,
            window,
        )
        frame = self.coordinates(window=window, looks=looks)
        destination_grid = self._destination_grid(
            (frame_row.start, frame_col.start),
            (frame_height, frame_width),
            looks,
        )
        if rows_per_chunk is not None and rows_per_chunk < 1:
            reject_invalid_state("rows_per_chunk must be >= 1")
        if out is not None and np.shape(out) != frame.shape:
            reject_invalid_state(
                f"out shape {np.shape(out)} does not match the destination "
                f"frame {frame.shape}"
            )
        values = _allocate_values(frame.shape, kind, layer) if out is None else out
        valid = np.zeros(frame.shape, dtype=bool)
        request_mask = self._rasterize_mask(
            mask,
            destination_grid,
            (frame_row.start, frame_col.start),
            looks,
            polygon_dilate_px,
        )
        step = frame.shape[0] if rows_per_chunk is None else int(rows_per_chunk)
        for row_start in range(0, frame.shape[0], step):
            row_stop = min(row_start + step, frame.shape[0])
            rows = slice(row_start, row_stop)
            chunk = RadarCoordinates(
                frame.az[rows],
                frame.rg[rows],
                frame.valid[rows],
            )
            if correction is not None:
                chunk = correction(chunk)
            if request_mask is not None:
                chunk = RadarCoordinates(
                    chunk.az,
                    chunk.rg,
                    chunk.valid & request_mask[rows],
                )
            chunk_values, chunk_valid = sample(
                layer,
                chunk,
                kind=kind,
                device=self._device,
            )
            values[rows] = chunk_values
            valid[rows] = chunk_valid
            if on_chunk is not None:
                on_chunk(rows)
        return GeocodedRaster(values=values, valid=valid, grid=destination_grid)

    def _frame_selection(
        self,
        rows: slice | tuple[int, int] | None,
        window: tuple[int, int, int, int] | None,
    ) -> tuple[slice, slice, int, int]:
        """Return LUT ``(row, col)`` indexers and frame dimensions."""
        if rows is not None and window is not None:
            reject_invalid_state("pass either rows or window, not both")
        coverage_row0, coverage_row1, coverage_col0, coverage_col1 = self._coverage
        coverage_height = coverage_row1 - coverage_row0
        if window is None:
            row_start, row_stop = (
                (0, coverage_height) if rows is None else _normalize_rows(rows)
            )
            if not (0 <= row_start <= row_stop <= coverage_height):
                reject_invalid_state(f"rows {rows!r} leave the geocoder frame")
            return (
                slice(coverage_row0 + row_start, coverage_row0 + row_stop),
                slice(coverage_col0, coverage_col1),
                row_stop - row_start,
                coverage_col1 - coverage_col0,
            )
        _validate_window(window)
        row0, row1, col0, col1 = window
        if not (
            coverage_row0 <= row0 <= row1 <= coverage_row1
            and coverage_col0 <= col0 <= col1 <= coverage_col1
        ):
            reject_invalid_state(
                f"window {window!r} is not contained in the geocoder coverage "
                f"{self._coverage}"
            )
        return (
            slice(row0, row1),
            slice(col0, col1),
            row1 - row0,
            col1 - col0,
        )

    def _destination_grid(
        self,
        origin: tuple[int, int],
        frame_shape: tuple[int, int],
        looks: tuple[int, int],
    ) -> GridSpec:
        """Return the pixel-centre-aligned grid of the produced frame."""
        from affine import Affine

        from faninsar.processing.mosaicking.grid import GeoGridSpec

        az_looks, rg_looks = looks
        window_row0, window_col0 = origin
        frame_height, frame_width = frame_shape
        transform = tuple(float(value) for value in self._grid.transform)[:6]
        a, b, c, d, e, f = transform
        return GeoGridSpec(
            crs=self._grid.crs,
            transform=Affine(
                a * rg_looks,
                b,
                c + window_col0 * a + window_row0 * b,
                d,
                e * az_looks,
                f + window_col0 * d + window_row0 * e,
            ),
            shape=(frame_height // az_looks, frame_width // rg_looks),
            resolution_m=(abs(a) * rg_looks, abs(e) * az_looks),
        )

    def _rasterize_mask(
        self,
        mask: np.ndarray | object | None,
        destination_grid: GridSpec,
        origin: tuple[int, int],
        looks: tuple[int, int],
        polygon_dilate_px: int,
    ) -> np.ndarray | None:
        """Return the boolean application-time mask, or ``None``."""
        if mask is None:
            return None
        if isinstance(mask, np.ndarray):
            if not np.issubdtype(mask.dtype, np.bool_):
                reject_invalid_state("mask array must be boolean")
            if mask.shape != destination_grid.shape:
                reject_invalid_state(
                    f"mask shape {mask.shape} does not match the destination "
                    f"grid {destination_grid.shape}"
                )
            return mask
        from faninsar.processing.geocoding.geo_lut import roi_geo_mask

        height, width = destination_grid.shape
        window_row0, window_col0 = origin
        window_mask = roi_geo_mask(
            mask,
            self._grid,
            window_row0,
            window_row0 + height * looks[0],
            window_col0,
            window_col0 + width * looks[1],
            dilate_px=int(polygon_dilate_px),
        )
        return _center_reduce_valid(
            _center_reduce_valid(window_mask, looks[0], axis=0),
            looks[1],
            axis=1,
        )


def _normalize_rows(rows: slice | tuple[int, int]) -> tuple[int, int]:
    """Return the ``(start, stop)`` bounds of a frame-relative row selection."""
    if isinstance(rows, tuple):
        if len(rows) != 2 or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in rows
        ):
            reject_invalid_state(
                f"rows must be a slice or (start, stop) pair; got {rows!r}"
            )
        return (int(rows[0]), int(rows[1]))
    if isinstance(rows, slice):
        if rows.step is not None:
            reject_invalid_state("row slices must not carry a step")
        if rows.stop is None:
            reject_invalid_state("row slices must carry a stop bound")
        return (0 if rows.start is None else int(rows.start), int(rows.stop))
    return reject_invalid_state(
        f"rows must be a slice or (start, stop) pair; got {rows!r}"
    )


def _validate_window(window: tuple[int, int, int, int]) -> None:
    """Reject non-integer or malformed destination windows."""
    if not (
        isinstance(window, tuple)
        and len(window) == 4
        and all(
            isinstance(value, int) and not isinstance(value, bool) for value in window
        )
    ):
        reject_invalid_state(
            f"window must be a (row0, row1, col0, col1) integer tuple; got {window!r}"
        )


def _validate_looks(looks: tuple[int, int]) -> tuple[int, int]:
    """Return ``(azimuth, range)`` look factors when both are positive ints."""
    if (
        not isinstance(looks, tuple)
        or len(looks) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in looks
        )
    ):
        reject_invalid_state(f"looks must be two positive integers; got {looks!r}")
    return int(looks[0]), int(looks[1])


def _center_indices(
    block_count: int,
    look: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return the low (and optional high) centre indices of each look block."""
    half = (look - 1) / 2.0
    first = int(np.floor(half))
    low = first + look * np.arange(block_count)
    if half == first:
        return low, None
    return low, low + 1


def _center_reduce(values: np.ndarray, look: int, *, axis: int) -> np.ndarray:
    """Reduce one axis to the box-average centre sample of each look block."""
    block_count = values.shape[axis] // look
    low, high = _center_indices(block_count, look)
    reduced = np.take(values, low, axis=axis)
    if high is None:
        return reduced
    return 0.5 * (reduced + np.take(values, high, axis=axis))


def _center_reduce_valid(valid: np.ndarray, look: int, *, axis: int) -> np.ndarray:
    """Reduce a validity mask over look-block centre samples (AND for even)."""
    block_count = valid.shape[axis] // look
    low, high = _center_indices(block_count, look)
    reduced = np.take(valid, low, axis=axis)
    if high is None:
        return reduced
    return reduced & np.take(valid, high, axis=axis)


def _allocate_values(
    shape: tuple[int, int],
    kind: GeocodeKind,
    layer: np.ndarray,
) -> np.ndarray:
    """Allocate the in-memory destination values for one kernel kind."""
    if kind == "complex":
        return np.full(shape, np.nan + 1j * np.nan, dtype=np.complex64)
    if kind in ("phase", "continuous"):
        return np.full(shape, np.nan, dtype=np.float32)
    if kind != "label":
        reject_invalid_state(f"unknown geocode resampling kind: {kind!r}")
    return np.zeros(shape, dtype=np.asarray(layer).dtype)
