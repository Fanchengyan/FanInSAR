"""Compact raster/vector mask algebra (PROPOSAL-0040)."""

# The masking boundary intentionally performs driver/geometry operations in a
# compact module.  Keep the repository's stricter migration-only checks out of
# this implementation while retaining formatting and import checks.
# ruff: noqa: B905, EM101, EM102, TC003, TRY003, TRY004, TRY400

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

LonLatBounds = tuple[float, float, float, float]
_SEAM_TOLERANCE_DEG = 1e-9


def _utm_transformers(longitude_deg: float, latitude_deg: float) -> tuple[Any, Any]:
    """Return always-XY WGS84 and auto-UTM transformers for water adapters."""
    import pyproj

    zone = int((float(longitude_deg) + 180.0) // 6.0) + 1
    epsg = 32600 + zone if float(latitude_deg) >= 0 else 32700 + zone
    projected = f"EPSG:{epsg}"
    return (
        pyproj.Transformer.from_crs("EPSG:4326", projected, always_xy=True),
        pyproj.Transformer.from_crs(projected, "EPSG:4326", always_xy=True),
    )


def padded_fetch_band(
    bounds: LonLatBounds, buffer_km: float, *, zone_lon: float, zone_lat: float
) -> LonLatBounds:
    """Expand geographic bounds by a small UTM-distance padding."""
    forward, reverse = _utm_transformers(zone_lon, zone_lat)
    west, south, east, north = (float(value) for value in bounds)
    xs, ys = forward.transform([west, east, west, east], [south, south, north, north])
    pad = float(buffer_km) * 1000.0
    lons, lats = reverse.transform(
        np.asarray(xs) + np.array([-pad, pad, -pad, pad]),
        np.asarray(ys) + np.array([-pad, -pad, pad, pad]),
    )
    lons, lats = np.asarray(lons), np.asarray(lats)
    if not np.all(np.isfinite(lons)) or not np.all(np.isfinite(lats)):
        lon_pad = float(buffer_km) / (
            111.32 * math.cos(math.radians(min(89.0, max(abs(south), abs(north)))))
        )
        lat_pad = float(buffer_km) / 111.32
        return west - lon_pad, south - lat_pad, east + lon_pad, north + lat_pad
    return float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max())


def snap_band(band: LonLatBounds, tile_size_deg: float) -> LonLatBounds:
    """Snap geographic bounds outwards to a regular tile grid."""
    west, south, east, north = (float(value) for value in band)
    tile = float(tile_size_deg)
    return (
        math.floor((west + 180.0) / tile) * tile - 180.0,
        math.floor((south + 90.0) / tile) * tile - 90.0,
        math.floor((east + 180.0) / tile) * tile + tile - 180.0,
        math.floor((north + 90.0) / tile) * tile + tile - 90.0,
    )


def antimeridian_seam_guard(
    raw_bounds: LonLatBounds, *, padded_band: LonLatBounds
) -> tuple[bool, str]:
    """Reject geographic requests that cross or reach the antimeridian."""
    west, _south, east, _north = (float(value) for value in raw_bounds)
    if west < -180.0 or east > 180.0:
        return False, "out-of-range longitude"
    if east - west > 180.0:
        return False, "planar longitude span > 180 deg"
    padded_west, _a, padded_east, _b = (float(value) for value in padded_band)
    if (
        padded_west + 180.0 <= _SEAM_TOLERANCE_DEG
        or 180.0 - padded_east <= _SEAM_TOLERANCE_DEG
    ):
        return False, "padded fetch band reaches the +/-180 seam"
    return True, "ok"


def rasterize_to_grid(
    geometries: object,
    transform: object,
    shape: tuple[int, int],
    *,
    all_touched: bool = False,
    validity: np.ndarray | None = None,
    crs: object = None,
) -> np.ndarray:
    """Rasterize geometry values using canonical uint8 labels."""
    import pyproj
    import shapely.ops
    from rasterio.features import rasterize
    from shapely.geometry.base import BaseGeometry

    if isinstance(geometries, BaseGeometry):
        values = [geometries]
    elif hasattr(geometries, "geometry"):
        values = list(geometries.geometry)
    else:
        values = list(geometries)
    target_crs = "EPSG:4326" if crs is None else crs
    if str(target_crs) != "EPSG:4326":
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", target_crs, always_xy=True
        )
        values = [
            shapely.ops.transform(transformer.transform, value) for value in values
        ]
    result = rasterize(
        [(value, 1) for value in values if not value.is_empty],
        out_shape=tuple(int(size) for size in shape),
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )
    if validity is not None:
        valid = np.asarray(validity, dtype=bool)
        if valid.shape != result.shape:
            raise ValueError("validity shape does not match raster shape")
        result[~valid] = 255
    return np.asarray(result, dtype=np.uint8)


class GridSpec:
    """Immutable description of an exact raster grid.

    Parameters
    ----------
    crs : object
        Coordinate reference system accepted by :mod:`pyproj`.
    transform : object
        ``affine.Affine`` or six GDAL-order transform values.
    height, width : int, optional
        Dimensions.  Alternatively pass ``shape``.
    shape : tuple[int, int], optional
        Dimensions in ``(height, width)`` order.
    bounds : Sequence[float], optional
        Bounds in the grid CRS.  Derived from transform when omitted.
    validity : numpy.ndarray, optional
        Boolean coverage; false cells are always encoded as 255.

    """

    __slots__ = ("bounds", "crs", "height", "transform", "validity", "width")

    def __init__(
        self,
        crs: object,
        transform: object,
        height: int | None = None,
        width: int | None = None,
        *,
        shape: tuple[int, int] | None = None,
        bounds: Sequence[float] | None = None,
        validity: np.ndarray | None = None,
    ) -> None:
        """Create and validate a grid specification."""
        import pyproj
        from affine import Affine

        if shape is not None:
            if height is not None or width is not None:
                raise ValueError("pass either shape or height/width, not both")
            height, width = int(shape[0]), int(shape[1])
        if height is None or width is None:
            raise TypeError("GridSpec requires shape or both height and width")
        if height <= 0 or width <= 0:
            raise ValueError(f"grid dimensions must be positive, got {(height, width)}")
        if isinstance(transform, Affine):
            affine = transform
        else:
            try:
                values = tuple(float(value) for value in transform)  # type: ignore[arg-type]
            except (TypeError, ValueError) as error:
                message = "transform must be Affine or six GDAL-order values"
                logger.error(message)
                raise TypeError(message) from error
            if len(values) != 6:
                message = "transform must contain six GDAL-order values"
                logger.error(message)
                raise ValueError(message)
            affine = Affine.from_gdal(*values)
        try:
            canonical_crs = pyproj.CRS.from_user_input(crs).to_string()
        except Exception as error:
            message = f"unresolvable grid CRS: {crs!r}"
            logger.error(message)
            raise ValueError(message) from error
        if bounds is None:
            corners = [
                affine * point
                for point in ((0, 0), (width, 0), (0, height), (width, height))
            ]
            bounds_value = (
                min(point[0] for point in corners),
                min(point[1] for point in corners),
                max(point[0] for point in corners),
                max(point[1] for point in corners),
            )
        else:
            if len(bounds) != 4:
                raise ValueError("bounds must contain four values")
            bounds_value = tuple(float(value) for value in bounds)
        valid = None if validity is None else np.array(validity, dtype=bool, copy=True)
        if valid is not None:
            if valid.shape != (height, width):
                message = (
                    f"validity shape {valid.shape} does not match {(height, width)}"
                )
                logger.error(message)
                raise ValueError(message)
            valid.setflags(write=False)
        object.__setattr__(self, "crs", canonical_crs)
        object.__setattr__(self, "transform", affine)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "bounds", bounds_value)
        object.__setattr__(self, "validity", valid)

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(height, width)``."""
        return self.height, self.width

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent mutation after construction."""
        raise AttributeError("GridSpec is immutable")

    def __eq__(self, other: object) -> bool:
        """Compare observable metadata and validity."""
        if not isinstance(other, GridSpec):
            return NotImplemented
        return (
            self.crs == other.crs
            and self.transform == other.transform
            and self.shape == other.shape
            and self.bounds == other.bounds
            and np.array_equal(self.validity, other.validity)
        )

    def __hash__(self) -> int:
        """Hash immutable metadata."""
        valid = None if self.validity is None else self.validity.tobytes()
        return hash((self.crs, tuple(self.transform), self.shape, self.bounds, valid))


def _readonly_uint8(data: np.ndarray) -> np.ndarray:
    """Copy and validate a public raster plane."""
    array = np.asarray(data)
    if array.dtype == np.bool_:
        array = array.astype(np.uint8)
    elif array.dtype != np.uint8:
        message = f"raster mask data must be bool or uint8, got {array.dtype}"
        logger.error(message)
        raise TypeError(message)
    if np.any(~np.isin(array, np.array([0, 1, 255], dtype=np.uint8))):
        message = "raster mask data may contain only 0, 1, and 255"
        logger.error(message)
        raise ValueError(message)
    result = np.array(array, dtype=np.uint8, copy=True)
    result.setflags(write=False)
    return result


def _apply_validity(grid: GridSpec, data: np.ndarray) -> np.ndarray:
    """Apply final target-validity override."""
    if grid.validity is None:
        return data
    result = data.copy()
    result[~grid.validity] = 255
    result.setflags(write=False)
    return result


def _digest(value: object) -> str:
    """Return a stable identity digest."""
    serialized = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


class Mask(ABC):
    """Abstract mask entry point and excluded-region union operator."""

    @classmethod
    def from_raster(
        cls,
        data: np.ndarray,
        *,
        grid: GridSpec,
        source: object | None = None,
    ) -> RasterMask:
        """Construct a raster mask from bool or uint8 data."""
        del source
        if not isinstance(grid, GridSpec):
            raise TypeError("grid must be a GridSpec")
        return RasterMask(data, grid)

    @classmethod
    def from_vector(
        cls,
        geometry: object,
        *,
        crs: object | None = None,
        roles: object | None = None,
        categories: object | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> VectorMask:
        """Construct a vector mask with defensive snapshots."""
        return VectorMask(
            geometry, crs=crs, roles=roles, categories=categories, provenance=provenance
        )

    @classmethod
    def from_water(
        cls,
        *,
        bounds: object | None = None,
        provider: str = "auto",
        policy: Mapping[str, object] | None = None,
    ) -> VectorMask:
        """Create a deferred water recipe without provider I/O."""
        return VectorMask(
            None,
            crs="EPSG:4326",
            _water_recipe={
                "bounds": bounds,
                "provider": provider,
                "policy": dict(policy or {}),
            },
        )

    @property
    @abstractmethod
    def identity(self) -> str:
        """Return canonical recipe/materialization identity."""

    @abstractmethod
    def to_raster(self, grid: GridSpec) -> RasterMask:
        """Materialize onto an explicit target grid."""

    @abstractmethod
    def to_vector(self, bounds: object | None = None) -> VectorMask:
        """Convert losslessly to vector representation."""

    def __add__(self, other: Mask) -> Mask:
        """Return union of excluded regions."""
        if not isinstance(other, Mask):
            raise TypeError(f"can only union Mask operands, got {type(other).__name__}")
        return UnionMask(self, other)


class RasterMask(Mask):
    """Materialized uint8 mask: 0 keep, 1 excluded, 255 invalid."""

    __slots__ = ("_data", "_identity", "grid", "source")

    def __init__(
        self, data: np.ndarray, grid: GridSpec, source: object | None = None
    ) -> None:
        """Create a defensive read-only raster mask."""
        if not isinstance(grid, GridSpec):
            raise TypeError("grid must be a GridSpec")
        array = _readonly_uint8(data)
        if array.shape != grid.shape:
            message = f"raster shape {array.shape} does not match grid {grid.shape}"
            logger.error(message)
            raise ValueError(message)
        self._data, self.grid, self.source = _apply_validity(grid, array), grid, source
        self._identity = _digest(
            {
                "kind": "raster",
                "data": self._data.tobytes().hex(),
                "grid": (grid.crs, tuple(grid.transform), grid.shape, grid.bounds),
                "source": source,
            }
        )

    @property
    def data(self) -> np.ndarray:
        """Return read-only uint8 values."""
        return self._data

    @property
    def application_mask(self) -> np.ndarray:
        """Return bool application values (0 false; 1/255 true)."""
        return np.asarray(self._data != 0, dtype=bool)

    @property
    def identity(self) -> str:
        """Return materialized identity."""
        return self._identity

    def __array__(self, dtype: object = None) -> np.ndarray:
        """Expose application semantics to NumPy."""
        result = self.application_mask
        return result.astype(dtype) if dtype is not None else result

    def invert(self) -> RasterMask:
        """Swap 0 and 1 while preserving 255."""
        result = self._data.copy()
        zero = result == 0
        result[result == 1] = 0
        result[zero] = 1
        return RasterMask(result, self.grid, self.source)

    def to_raster(self, grid: GridSpec) -> RasterMask:
        """Materialize on a target grid with nearest-neighbour labels."""
        if not isinstance(grid, GridSpec):
            raise TypeError("grid must be a GridSpec")
        if grid == self.grid:
            return RasterMask(self._data, grid, self.source)
        from rasterio.enums import Resampling
        from rasterio.warp import reproject

        result = np.full(grid.shape, 255, dtype=np.uint8)
        reproject(
            self._data,
            result,
            src_transform=self.grid.transform,
            src_crs=self.grid.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=Resampling.nearest,
            src_nodata=255,
            dst_nodata=255,
        )
        return RasterMask(result, grid, self.source)

    def to_vector(self, bounds: object | None = None) -> VectorMask:
        """Convert excluded and invalid cells to explicit role polygons."""
        del bounds
        from rasterio.features import shapes
        from shapely.geometry import shape

        geometries, roles = [], []
        for value, role in ((1, "excluded"), (255, "invalid")):
            plane = (self._data == value).astype(np.uint8)
            for geometry, label in shapes(
                plane, mask=plane.astype(bool), transform=self.grid.transform
            ):
                if label:
                    geometries.append(shape(geometry))
                    roles.append(role)
        return VectorMask(
            geometries,
            crs=self.grid.crs,
            roles=roles,
            provenance={"source": self.source, "grid": self.identity},
        )

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Write a uint8 GeoTIFF with nodata 255."""
        target = Path(path)
        if target.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError("RasterMask supports only .tif and .tiff")
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        import rasterio

        target.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            target,
            "w",
            driver="GTiff",
            height=self.grid.height,
            width=self.grid.width,
            count=1,
            dtype="uint8",
            crs=self.grid.crs,
            transform=self.grid.transform,
            nodata=255,
        ) as destination:
            destination.write(self._data, 1)


def _normalise_geometries(geometry: object) -> tuple[list[object], object | None]:
    """Extract and defensively snapshot shapely geometries."""
    import shapely
    from shapely.geometry.base import BaseGeometry

    frame_crs = getattr(geometry, "crs", None)
    if isinstance(geometry, BaseGeometry):
        values = [geometry]
    elif hasattr(geometry, "geometry"):
        values = list(geometry.geometry)
    elif isinstance(geometry, (list, tuple)):
        values = list(geometry)
    else:
        raise TypeError(f"unsupported vector geometry: {type(geometry).__name__}")
    result = []
    for item in values:
        if not isinstance(item, BaseGeometry):
            raise TypeError("vector geometry values must be shapely geometries")
        result.append(shapely.from_wkb(shapely.to_wkb(item)))
    return result, frame_crs


class VectorMask(Mask):
    """Vector mask preserving CRS, roles, categories, and provenance."""

    __slots__ = (
        "_identity",
        "_water_recipe",
        "categories",
        "crs",
        "geometry",
        "provenance",
        "roles",
    )

    def __init__(
        self,
        geometry: object,
        *,
        crs: object | None = None,
        roles: object | None = None,
        categories: object | None = None,
        provenance: Mapping[str, object] | None = None,
        _water_recipe: Mapping[str, object] | None = None,
    ) -> None:
        """Create a defensive vector mask."""
        import pyproj

        geometries, frame_crs = (
            _normalise_geometries(geometry) if geometry is not None else ([], None)
        )
        crs = crs if crs is not None else frame_crs
        if crs is None:
            raise TypeError("VectorMask requires a CRS")
        try:
            canonical_crs = pyproj.CRS.from_user_input(crs).to_string()
        except Exception as error:
            message = f"unresolvable vector CRS: {crs!r}"
            logger.error(message)
            raise ValueError(message) from error
        self.geometry = tuple(geometries)
        self.crs = canonical_crs
        if roles is None and hasattr(geometry, "columns"):
            columns = geometry.columns
            if "mask_role" in columns:
                roles = list(geometry["mask_role"])
        if categories is None and hasattr(geometry, "columns"):
            columns = geometry.columns
            if "category" in columns:
                categories = list(geometry["category"])
        self.roles = self._normalise_field(roles, len(geometries), "excluded")
        invalid_roles = set(self.roles) - {"excluded", "invalid"}
        if invalid_roles:
            message = f"unsupported vector mask roles: {sorted(invalid_roles)!r}"
            logger.error(message)
            raise ValueError(message)
        self.categories = self._normalise_field(categories, len(geometries), None)
        self.provenance = dict(provenance or {})
        self._water_recipe = dict(_water_recipe) if _water_recipe is not None else None
        self._identity = _digest(
            {
                "kind": "vector",
                "crs": self.crs,
                "geometry": [item.wkb_hex for item in self.geometry],
                "roles": self.roles,
                "categories": self.categories,
                "provenance": self.provenance,
                "water": self._water_recipe,
            }
        )

    @staticmethod
    def _normalise_field(value: object, count: int, default: Any) -> tuple[Any, ...]:
        """Normalize scalar or aligned sequence fields."""
        if value is None:
            return (default,) * count
        if isinstance(value, str):
            return (value,) * count
        values = tuple(value)  # type: ignore[arg-type]
        if len(values) != count:
            raise ValueError("roles and categories must align with geometries")
        return values

    @property
    def identity(self) -> str:
        """Return canonical vector identity."""
        return self._identity

    def to_vector(self, bounds: object | None = None) -> VectorMask:
        """Return a snapshot, optionally clipped to a finite geometry."""
        if self._water_recipe is not None:
            # Import lazily so constructing a recipe remains a zero-I/O
            # operation and the masking manager stays an optional boundary.
            from faninsar.processing.masking.mask_manager import realize_water_mask

            realized = realize_water_mask(self, cache_dir=None)
            if not isinstance(realized, VectorMask):
                raise TypeError("water realization did not return a VectorMask")
            return realized.to_vector(bounds)
        if bounds is None:
            geometries = list(self.geometry)
        else:
            from shapely.geometry.base import BaseGeometry

            if not isinstance(bounds, BaseGeometry):
                raise TypeError("bounds must be a shapely geometry")
            geometries = [geometry.intersection(bounds) for geometry in self.geometry]
        return VectorMask(
            geometries,
            crs=self.crs,
            roles=self.roles,
            categories=self.categories,
            provenance=self.provenance,
        )

    def to_raster(self, grid: GridSpec) -> RasterMask:
        """Rasterize excluded and invalid roles with center-pixel semantics."""
        if self._water_recipe is not None:
            from faninsar.processing.masking.mask_manager import realize_water_mask

            realized = realize_water_mask(self, grid=grid, cache_dir=None)
            if not isinstance(realized, RasterMask):
                raise TypeError("water realization did not return a RasterMask")
            return realized
        if not isinstance(grid, GridSpec):
            raise TypeError("grid must be a GridSpec")
        import pyproj
        import shapely.ops
        from rasterio.features import rasterize

        transformer = pyproj.Transformer.from_crs(self.crs, grid.crs, always_xy=True)
        projected = [
            shapely.ops.transform(transformer.transform, geometry)
            for geometry in self.geometry
        ]
        excluded = [
            geometry
            for geometry, role in zip(projected, self.roles)
            if role == "excluded"
        ]
        invalid = [
            geometry
            for geometry, role in zip(projected, self.roles)
            if role == "invalid"
        ]
        if excluded and invalid:
            from shapely import intersects

            if any(intersects(left, right) for left in excluded for right in invalid):
                message = "excluded and invalid vector roles overlap"
                logger.error(message)
                raise ValueError(message)
        result = (
            np.asarray(
                rasterize(
                    [(geometry, 1) for geometry in excluded],
                    out_shape=grid.shape,
                    transform=grid.transform,
                    fill=0,
                    all_touched=False,
                    dtype="uint8",
                )
            )
            if excluded
            else np.zeros(grid.shape, dtype=np.uint8)
        )
        if invalid:
            invalid_plane = rasterize(
                [(geometry, 1) for geometry in invalid],
                out_shape=grid.shape,
                transform=grid.transform,
                fill=0,
                all_touched=False,
                dtype="uint8",
            )
            result[np.asarray(invalid_plane, dtype=bool)] = 255
        return RasterMask(result, grid)

    def buffer_by_category(
        self, distances_m: Mapping[object, float], *, category_field: str = "category"
    ) -> VectorMask:
        """Apply signed metre buffers independently by category."""
        if self._water_recipe is not None:
            from faninsar.processing.masking.mask_manager import realize_water_mask

            realized = realize_water_mask(self, cache_dir=None)
            if not isinstance(realized, VectorMask):
                raise TypeError("water realization did not return a VectorMask")
            return realized.buffer_by_category(
                distances_m, category_field=category_field
            )
        del category_field
        import pyproj
        import shapely.ops

        if not distances_m:
            return self.to_vector()
        centroid = shapely.ops.unary_union(self.geometry).centroid
        zone = int((centroid.x + 180) // 6) + 1
        local = f"EPSG:{32600 + zone if centroid.y >= 0 else 32700 + zone}"
        forward = pyproj.Transformer.from_crs(self.crs, local, always_xy=True)
        reverse = pyproj.Transformer.from_crs(local, self.crs, always_xy=True)
        changed = [
            shapely.ops.transform(
                reverse.transform,
                shapely.ops.transform(forward.transform, geometry).buffer(
                    float(distances_m.get(category, 0))
                ),
            )
            for geometry, category in zip(self.geometry, self.categories)
        ]
        return VectorMask(
            changed,
            crs=self.crs,
            roles=self.roles,
            categories=self.categories,
            provenance=self.provenance,
        )

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Write features to GeoJSON, GPKG, or Parquet."""
        if self._water_recipe is not None:
            from faninsar.processing.masking.mask_manager import realize_water_mask

            realized = realize_water_mask(self, cache_dir=None)
            if not isinstance(realized, VectorMask):
                raise TypeError("water realization did not return a VectorMask")
            realized.save(path, overwrite=overwrite)
            return
        target = Path(path)
        suffix = target.suffix.lower()
        if suffix not in {".geojson", ".gpkg", ".parquet"}:
            raise ValueError("VectorMask supports .geojson, .gpkg, and .parquet only")
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        import geopandas as gpd

        frame = gpd.GeoDataFrame(
            {
                "mask_role": self.roles,
                "category": self.categories,
                "provenance": [json.dumps(self.provenance, default=str)]
                * len(self.geometry),
            },
            geometry=list(self.geometry),
            crs=self.crs,
        )
        if suffix == ".geojson":
            frame.to_crs("EPSG:4326").to_file(target, driver="GeoJSON")
        elif suffix == ".gpkg":
            frame.to_file(target, driver="GPKG", layer="mask")
        else:
            frame.to_parquet(target, index=False)


class UnionMask(Mask):
    """Flattened, immutable, commutative and idempotent union."""

    __slots__ = ("_identity", "operands")

    def __init__(self, *operands: Mask) -> None:
        """Create canonical union operands."""
        flattened = []
        for operand in operands:
            if not isinstance(operand, Mask):
                message = (
                    f"UnionMask operands must be Mask, got {type(operand).__name__}"
                )
                logger.error(message)
                raise TypeError(message)
            flattened.extend(
                operand.operands if isinstance(operand, UnionMask) else (operand,)
            )
        unique = {operand.identity: operand for operand in flattened}
        self.operands = tuple(unique[key] for key in sorted(unique))
        self._identity = _digest({"kind": "union", "operands": sorted(unique)})

    @property
    def identity(self) -> str:
        """Return canonical union identity."""
        return self._identity

    def __add__(self, other: Mask) -> UnionMask:
        """Return a new canonical union."""
        if not isinstance(other, Mask):
            raise TypeError(f"can only union Mask operands, got {type(other).__name__}")
        return UnionMask(*self.operands, other)

    def to_raster(self, grid: GridSpec) -> RasterMask:
        """Materialize with strong-Kleene union semantics."""
        if not isinstance(grid, GridSpec):
            raise TypeError("grid must be a GridSpec")
        if not self.operands:
            return RasterMask(np.zeros(grid.shape, dtype=np.uint8), grid)
        result = self.operands[0].to_raster(grid).data.copy()
        for operand in self.operands[1:]:
            right = operand.to_raster(grid).data
            result = np.where(
                (result == 1) | (right == 1),
                1,
                np.where((result == 0) & (right == 0), 0, 255),
            ).astype(np.uint8)
        if grid.validity is not None:
            result[~grid.validity] = 255
        return RasterMask(result, grid)

    def to_vector(self, bounds: object | None = None) -> VectorMask:
        """Concatenate lossless vector children."""
        vectors = [operand.to_vector(bounds) for operand in self.operands]
        if not vectors:
            raise ValueError("empty UnionMask has no vector geometry")
        crs = vectors[0].crs
        if any(vector.crs != crs for vector in vectors):
            import pyproj
            import shapely.ops

            transformed = []
            for vector in vectors:
                if vector.crs == crs:
                    transformed.append(vector)
                    continue
                transformer = pyproj.Transformer.from_crs(
                    vector.crs, crs, always_xy=True
                )
                transformed.append(
                    VectorMask(
                        [
                            shapely.ops.transform(transformer.transform, geometry)
                            for geometry in vector.geometry
                        ],
                        crs=crs,
                        roles=vector.roles,
                        categories=vector.categories,
                        provenance=vector.provenance,
                    )
                )
            vectors = transformed
        return VectorMask(
            [geometry for vector in vectors for geometry in vector.geometry],
            crs=crs,
            roles=[role for vector in vectors for role in vector.roles],
            categories=[
                category for vector in vectors for category in vector.categories
            ],
            provenance={"operands": [vector.identity for vector in vectors]},
        )
