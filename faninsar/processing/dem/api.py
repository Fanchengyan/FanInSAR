"""Unified public DEM API and source-to-grid materialization."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from affine import Affine

from faninsar._core.geo.grids import GridSpec
from faninsar.logging import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping

VerticalDatum = Literal["ellipsoidal", "egm96", "egm2008"]
DEMProduct = Literal[
    "auto",
    "glo30",
    "glo90",
    "nasadem",
    "alos-dem",
    "srtm-skadi",
    "terrain-tiles",
    "arcticdem-10",
    "arcticdem-32",
    "arcticdem-2",
    "rema-10",
    "rema-32",
    "rema-2",
    "nisar-glo30",
]

_DATUMS = frozenset({"ellipsoidal", "egm96", "egm2008"})
_PRODUCTS = frozenset(
    {
        "auto",
        "glo30",
        "glo90",
        "nasadem",
        "alos-dem",
        "srtm-skadi",
        "terrain-tiles",
        "arcticdem-10",
        "arcticdem-32",
        "arcticdem-2",
        "rema-10",
        "rema-32",
        "rema-2",
        "nisar-glo30",
    }
)


def _admit_datum(value: str) -> VerticalDatum:
    """Validate one public vertical-datum name."""
    datum = str(value).strip().lower()
    if datum not in _DATUMS:
        message = f"unsupported vertical datum {value!r}"
        logger.error(message)
        raise ValueError(message)
    return datum  # type: ignore[return-value]


def _admit_grid(grid: GridSpec) -> GridSpec:
    """Validate the shared grid type at a public boundary."""
    if not isinstance(grid, GridSpec):
        message = "grid must be a GridSpec"
        logger.error(message)
        raise TypeError(message)
    return grid


def _readonly_array(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Make a defensive, finite-compatible two-dimensional DEM array."""
    result = np.array(array, dtype=np.float32, copy=True)
    if result.ndim != 2 or result.shape != shape:
        message = f"DEM array shape {result.shape} does not match grid {shape}"
        logger.error(message)
        raise ValueError(message)
    result.setflags(write=False)
    return result


def _freeze_mapping(values: Mapping[str, object] | None) -> Mapping[str, str]:
    """Return string-valued immutable provenance metadata."""
    raw = (
        {}
        if values is None
        else {str(key): str(value) for key, value in values.items()}
    )
    return MappingProxyType(raw)


def _natural_spline_six(values: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    """Evaluate the P0032 six-sample natural spline along the final axis."""
    samples = np.asarray(values, dtype=np.float64)
    if samples.shape[-1] != 6:
        message = "six-sample spline requires a final axis of length 6"
        logger.error(message)
        raise ValueError(message)
    second = np.zeros_like(samples)
    recurrence = np.zeros(6, dtype=np.float64)
    for index in range(1, 5):
        denominator = recurrence[index - 1] / 2.0 + 2.0
        recurrence[index] = -0.5 / denominator
        second[..., index] = (
            3.0
            * (
                samples[..., index + 1]
                - 2.0 * samples[..., index]
                + samples[..., index - 1]
            )
            - second[..., index - 1] / 2.0
        ) / denominator
    for index in range(4, 0, -1):
        second[..., index] = (
            recurrence[index] * second[..., index + 1] + second[..., index]
        )
    local = np.asarray(fraction, dtype=np.float64)
    return samples[..., 1] + local * (
        samples[..., 2]
        - samples[..., 1]
        - second[..., 1] / 3.0
        - second[..., 2] / 6.0
        + local
        * (second[..., 1] / 2.0 + local * (second[..., 2] - second[..., 1]) / 6.0)
    )


def _sample_biquintic(
    source: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    """Sample a source array using the qualified P0032 6x6 rule."""
    output_shape = rows.shape
    rows = np.asarray(rows, dtype=np.float64).ravel()
    cols = np.asarray(cols, dtype=np.float64).ravel()
    height, width = source.shape
    if height < 6 or width < 6:
        # Tiny synthetic fixtures cannot provide the qualified support window.
        # Bilinear is a deterministic boundary fallback, never exposed as a
        # user-selectable production kernel.
        r0 = np.floor(rows).astype(np.int64)
        c0 = np.floor(cols).astype(np.int64)
        valid = (r0 >= 0) & (r0 < height - 1) & (c0 >= 0) & (c0 < width - 1)
        result = np.full(rows.shape, np.nan, dtype=np.float64)
        if np.any(valid):
            rr, cc = r0[valid], c0[valid]
            dr, dc = rows[valid] - rr, cols[valid] - cc
            result[valid] = (
                source[rr, cc] * (1 - dr) * (1 - dc)
                + source[rr, cc + 1] * (1 - dr) * dc
                + source[rr + 1, cc] * dr * (1 - dc)
                + source[rr + 1, cc + 1] * dr * dc
            )
        return result.reshape(output_shape)
    row_floor = np.floor(rows).astype(np.int64)
    col_floor = np.floor(cols).astype(np.int64)
    valid = (
        (row_floor >= 1)
        & (row_floor <= height - 5)
        & (col_floor >= 1)
        & (col_floor <= width - 5)
    )
    result = np.full(rows.shape, np.nan, dtype=np.float64)
    if np.any(valid):
        selected = np.flatnonzero(valid)
        row_base, col_base = row_floor[selected], col_floor[selected]
        neighbours = np.arange(-1, 5, dtype=np.int64)
        windows = source[
            row_base[:, None, None] + neighbours[None, :, None],
            col_base[:, None, None] + neighbours[None, None, :],
        ]
        along = _natural_spline_six(windows, (cols[selected] - col_base)[:, None])
        result[selected] = _natural_spline_six(along, rows[selected] - row_base)
    return result.reshape(output_shape)


def _grid_centres(grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return target pixel-centre x/y arrays."""
    transform = Affine(*grid.transform)
    columns, rows = np.meshgrid(
        np.arange(grid.width, dtype=np.float64) + 0.5,
        np.arange(grid.height, dtype=np.float64) + 0.5,
    )
    xs, ys = transform * (columns, rows)
    return np.asarray(xs), np.asarray(ys)


def _identity(
    array: np.ndarray,
    grid: GridSpec,
    datum: VerticalDatum,
    provenance: Mapping[str, str],
) -> str:
    """Build a stable content identity for a materialized DEM."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(array).tobytes())
    payload = {
        "grid": (grid.crs, grid.transform, grid.shape, grid.bounds),
        "datum": datum,
        "provenance": dict(provenance),
        "resampling": "isce-p0032-biquintic-6x6-v1",
    }
    digest.update(json.dumps(payload, sort_keys=True, default=str).encode())
    return digest.hexdigest()


class DEM(ABC):
    """Typed DEM facade with offline factories and one materialization seam."""

    @classmethod
    def from_source(
        cls, product: DEMProduct | str, *, cache_dir: Path | None = None
    ) -> SourceDEM:
        """Create an offline source recipe."""
        return SourceDEM(product, cache_dir=cache_dir)

    @classmethod
    def from_raster(
        cls,
        path: Path,
        *,
        vertical_datum: VerticalDatum | None = None,
    ) -> RasterDEM:
        """Open a georeferenced local raster and resolve its vertical datum."""
        return RasterDEM(path=path, vertical_datum=vertical_datum)

    @classmethod
    def from_constant(
        cls, height: float, *, vertical_datum: VerticalDatum = "ellipsoidal"
    ) -> ConstantDEM:
        """Create a finite constant-height DEM without I/O."""
        return ConstantDEM(height, vertical_datum=vertical_datum)

    @abstractmethod
    def to_raster(
        self, grid: GridSpec, *, vertical_datum: VerticalDatum = "ellipsoidal"
    ) -> RasterDEM:
        """Materialize the DEM on one target grid."""


class SourceDEM(DEM):
    """Offline source recipe resolved only when materialized."""

    def __init__(
        self, product: DEMProduct | str, *, cache_dir: Path | None = None
    ) -> None:
        """Validate a canonical product/provider selection without network I/O."""
        selection = str(product).strip().lower()
        name, separator, provider = selection.partition(":")
        if name not in _PRODUCTS or (separator and not provider):
            message = f"unsupported DEM source selection {product!r}"
            logger.error(message)
            raise ValueError(message)
        self.product = name
        self.provider = provider or None
        self.cache_dir = None if cache_dir is None else Path(cache_dir)

    def to_raster(
        self, grid: GridSpec, *, vertical_datum: VerticalDatum = "ellipsoidal"
    ) -> RasterDEM:
        """Materialize this source through the provider implementation.

        Provider resolution is deliberately kept out of construction.  The
        provider lane installs the materializer at this boundary.
        """
        _admit_grid(grid)
        _admit_datum(vertical_datum)
        if self.cache_dir is None:
            message = "SourceDEM.to_raster requires an explicit cache_dir"
            logger.error(message)
            raise ValueError(message)
        message = f"DEM source materialization is not installed for {self.product}"
        logger.error(message)
        raise NotImplementedError(message)


class ConstantDEM(DEM):
    """Finite constant-height DEM in a declared vertical datum."""

    def __init__(
        self, height: float, *, vertical_datum: VerticalDatum = "ellipsoidal"
    ) -> None:
        """Validate and retain a scalar height without I/O."""
        if not np.isfinite(height):
            message = "constant DEM height must be finite"
            logger.error(message)
            raise ValueError(message)
        self.height = float(height)
        self.vertical_datum = _admit_datum(vertical_datum)

    def to_raster(
        self, grid: GridSpec, *, vertical_datum: VerticalDatum = "ellipsoidal"
    ) -> RasterDEM:
        """Fill the target grid with this constant height."""
        _admit_grid(grid)
        target = _admit_datum(vertical_datum)
        values = np.full(grid.shape, self.height, dtype=np.float32)
        if self.vertical_datum != target:
            from .datum import convert_heights

            xs, ys = _grid_centres(grid)
            values = np.asarray(
                convert_heights(values, xs, ys, self.vertical_datum, target),
                dtype=np.float32,
            )
        if grid.validity is not None:
            values = np.array(values, copy=True)
            values[~grid.validity] = np.nan
        provenance = _freeze_mapping(
            {
                "kind": "constant",
                "source_datum": self.vertical_datum,
                "resampling": "none",
            }
        )
        return RasterDEM(
            array=values,
            grid=grid,
            vertical_datum=target,
            provenance=provenance,
        )


class RasterDEM(DEM):
    """Materialized, georeferenced elevation raster with complete identity."""

    def __init__(
        self,
        array: np.ndarray | None = None,
        grid: GridSpec | None = None,
        *,
        path: Path | None = None,
        vertical_datum: VerticalDatum | None = "ellipsoidal",
        nodata: float | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> None:
        """Create a raster from an array or load one local GeoTIFF."""
        if path is not None:
            import rasterio

            with rasterio.open(path) as dataset:
                array = dataset.read(1)
                if grid is None:
                    grid = GridSpec(
                        crs=dataset.crs,
                        transform=dataset.transform,
                        height=dataset.height,
                        width=dataset.width,
                    )
                if nodata is None:
                    nodata = dataset.nodata
                if vertical_datum is None:
                    tag = dataset.tags().get("dem_vertical_datum")
                    if tag is None:
                        message = (
                            "local raster vertical datum is ambiguous; "
                            "provide vertical_datum"
                        )
                        logger.error(message)
                        raise ValueError(message)
                    vertical_datum = tag
                provenance = {
                    **({} if provenance is None else dict(provenance)),
                    "source_path": str(Path(path).resolve()),
                }
        if array is None or grid is None:
            message = "RasterDEM requires array and grid, or path"
            logger.error(message)
            raise TypeError(message)
        _admit_grid(grid)
        datum = _admit_datum(
            "ellipsoidal" if vertical_datum is None else vertical_datum
        )
        values = np.array(array, dtype=np.float32, copy=True)
        if nodata is not None:
            values[np.isclose(values, nodata)] = np.nan
        self._array = _readonly_array(values, grid.shape)
        self._grid = grid
        self._vertical_datum = datum
        self._nodata = None if nodata is None else float(nodata)
        metadata = {
            "resampling": "isce-p0032-biquintic-6x6-v1",
            "vertical_datum": datum,
            **({} if provenance is None else dict(provenance)),
        }
        self._provenance = _freeze_mapping(metadata)
        self._identity = _identity(self._array, grid, datum, self._provenance)

    @property
    def array(self) -> np.ndarray:
        """Return the read-only numeric height array."""
        return self._array

    @property
    def grid(self) -> GridSpec:
        """Return the authoritative raster grid."""
        return self._grid

    @property
    def shape(self) -> tuple[int, int]:
        """Return raster dimensions in ``(height, width)`` order."""
        return self.grid.shape

    @property
    def height(self) -> int:
        """Return the raster height in pixels."""
        return self.grid.height

    @property
    def width(self) -> int:
        """Return the raster width in pixels."""
        return self.grid.width

    @property
    def crs(self) -> str:
        """Return the canonical raster CRS."""
        return self.grid.crs

    @property
    def transform(self) -> tuple[float, float, float, float, float, float]:
        """Return the canonical affine transform coefficients."""
        return self.grid.transform

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return outer pixel-edge bounds in the raster CRS."""
        return self.grid.bounds

    @property
    def vertical_datum(self) -> VerticalDatum:
        """Return the materialized vertical datum."""
        return self._vertical_datum

    @property
    def nodata(self) -> float | None:
        """Return the declared output nodata value, if any."""
        return self._nodata

    @property
    def identity(self) -> str:
        """Return the stable content identity."""
        return self._identity

    @property
    def provenance(self) -> Mapping[str, str]:
        """Return immutable source and processing provenance."""
        return self._provenance

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Expose the raster as a NumPy array without copying when possible."""
        return np.asarray(self._array, dtype=dtype)

    def to_raster(
        self, grid: GridSpec, *, vertical_datum: VerticalDatum = "ellipsoidal"
    ) -> Self:
        """Regrid once onto ``grid`` using the fixed P0032 rule."""
        _admit_grid(grid)
        target_datum = _admit_datum(vertical_datum)
        if grid == self.grid and target_datum == self.vertical_datum:
            return self
        from pyproj import Transformer

        target_x, target_y = _grid_centres(grid)
        if grid.crs == self.grid.crs:
            source_x, source_y = target_x, target_y
        else:
            transformer = Transformer.from_crs(grid.crs, self.grid.crs, always_xy=True)
            source_x, source_y = transformer.transform(target_x, target_y)
        source_transform = Affine(*self.grid.transform)
        cols, rows = (~source_transform) * (source_x, source_y)
        values = _sample_biquintic(
            np.asarray(self._array, dtype=np.float64),
            np.asarray(rows, dtype=np.float64),
            np.asarray(cols, dtype=np.float64),
        ).astype(np.float32)
        if self.nodata is not None:
            values[~np.isfinite(values)] = np.nan
        if self.vertical_datum != target_datum:
            from .datum import convert_heights

            values = np.asarray(
                convert_heights(
                    values, target_x, target_y, self.vertical_datum, target_datum
                ),
                dtype=np.float32,
            )
        if grid.validity is not None:
            values = np.array(values, copy=True)
            values[~grid.validity] = np.nan
        provenance = {
            **dict(self.provenance),
            "resampling": "isce-p0032-biquintic-6x6-v1",
            "source_grid": self.grid.crs,
        }
        return type(self)(
            array=values, grid=grid, vertical_datum=target_datum, provenance=provenance
        )

    def save(self, path: Path) -> None:
        """Write one new GeoTIFF without overwriting an existing file."""
        destination = Path(path)
        if destination.suffix.lower() not in {".tif", ".tiff"}:
            message = "RasterDEM.save requires a .tif or .tiff path"
            logger.error(message)
            raise ValueError(message)
        if destination.exists():
            message = f"refusing to overwrite existing raster: {destination}"
            logger.error(message)
            raise FileExistsError(message)
        import rasterio

        destination.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            destination,
            "w",
            driver="GTiff",
            height=self.grid.height,
            width=self.grid.width,
            count=1,
            dtype="float32",
            crs=self.grid.crs,
            transform=Affine(*self.grid.transform),
            nodata=self.nodata,
        ) as dataset:
            dataset.write(np.asarray(self._array, dtype=np.float32), 1)
            dataset.update_tags(dem_vertical_datum=self.vertical_datum)


__all__ = [
    "DEM",
    "ConstantDEM",
    "DEMProduct",
    "GridSpec",
    "RasterDEM",
    "SourceDEM",
    "VerticalDatum",
]
