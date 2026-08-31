"""Private bridge from the P0030 registry to the public DEM materializer."""

from __future__ import annotations

import fnmatch
import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.processing.dem import GridSpec, RasterDEM
    from faninsar.processing.dem.providers import _P0030Source
    from faninsar.processing.dem.resources import ResourceBudget
    from faninsar.processing.geometry.dem_sources import DemSource
    from faninsar.processing.geometry.dem_transport import FetchPlan

logger = setup_logger(__name__)


def _bounds(grid: GridSpec) -> tuple[float, float, float, float]:
    """Return the WGS84 bounds of a target grid."""
    from pyproj import Transformer

    left, bottom, right, top = grid.bounds
    transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(
        np.array([left, left, right, right]),
        np.array([bottom, top, bottom, top]),
    )
    return (
        float(np.min(longitude)),
        float(np.min(latitude)),
        float(np.max(longitude)),
        float(np.max(latitude)),
    )


def _execute_plan(
    plan: FetchPlan,
    cache_dir: Path,
    *,
    budget: ResourceBudget | None,
) -> list[Path]:
    """Resolve cached files and execute only missing P0030 plan units."""
    from faninsar.processing.geometry.dem_transport import (
        Artifact,
        TileSet,
        expand_tile_parts,
        fetch_plan,
        validate_cache_target,
    )

    max_fetch_bytes = budget.max_fetch_bytes if budget is not None else 2**33
    if isinstance(plan, TileSet):
        units = [unit for tile in plan.tiles for unit in expand_tile_parts(tile)]
        hits: list[Path] = []
        missing = []
        for unit in units:
            target = cache_dir / unit.cache_path
            validate_cache_target(cache_dir, target)
            if target.is_file() and target.stat().st_size >= unit.min_bytes:
                hits.append(target)
            else:
                missing.append(unit)
        if missing:
            hits.extend(
                fetch_plan(
                    replace(plan, tiles=tuple(missing)),
                    cache_dir,
                    max_fetch_bytes=max_fetch_bytes,
                )
            )
        by_name = {path.name: path for path in hits}
        resolved: list[Path] = []
        for unit in units:
            target = cache_dir / unit.cache_path
            path = target if target.is_file() else by_name.get(target.name)
            if path is None:
                message = f"P0030 fetch did not produce {target}"
                logger.error(message)
                raise RuntimeError(message)
            if path not in resolved:
                resolved.append(path)
        return resolved

    executed = fetch_plan(plan, cache_dir, max_fetch_bytes=max_fetch_bytes)
    if isinstance(plan, Artifact) and plan.expand == "zip" and plan.cache_path:
        staging = cache_dir / plan.cache_path.parent / (
            plan.cache_path.name + ".zip-staging"
        )
        pattern = plan.member_pattern or "*"
        members = [
            path
            for path in sorted(staging.rglob("*"))
            if path.is_file() and fnmatch.fnmatch(path.name, pattern)
        ]
        if members:
            return members
    return executed


def _source_path(path: Path, recipe: object) -> str:
    """Apply the registry's GDAL open prefix to a local cache path."""
    template = str(getattr(recipe, "gdal_open", "{path}"))
    if "{member}" in template:
        return str(path)
    return template.format(path=path)


def _sample_dataset(
    dataset: object,
    target_x: np.ndarray,
    target_y: np.ndarray,
    *,
    target_crs: str,
    source_crs: object,
    source_nodata: float | None,
) -> np.ndarray:
    """Evaluate one dataset directly at target centres in bounded blocks."""
    from affine import Affine
    from pyproj import Transformer
    from rasterio.windows import Window

    from faninsar.processing.dem.api import _sample_biquintic

    height, width = target_x.shape
    result = np.full((height, width), np.nan, dtype=np.float64)
    transformer = Transformer.from_crs(target_crs, source_crs, always_xy=True)
    source_x, source_y = transformer.transform(target_x, target_y)
    source_x = np.asarray(source_x, dtype=np.float64)
    source_y = np.asarray(source_y, dtype=np.float64)
    transform = Affine(*dataset.transform)
    if str(source_crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
        source_center = float(transform.c) + 0.5 * float(transform.a) * dataset.width
        source_x += 360.0 * np.round((source_center - source_x) / 360.0)
    columns, rows = (~transform) * (source_x, source_y)

    for row_start in range(0, height, 128):
        row_stop = min(height, row_start + 128)
        for col_start in range(0, width, 128):
            col_stop = min(width, col_start + 128)
            block_rows = rows[row_start:row_stop, col_start:col_stop]
            block_cols = columns[row_start:row_stop, col_start:col_stop]
            finite = np.isfinite(block_rows) & np.isfinite(block_cols)
            if not np.any(finite):
                continue
            row_min = math.floor(float(np.min(block_rows[finite]))) - 2
            col_min = math.floor(float(np.min(block_cols[finite]))) - 2
            row_max = math.ceil(float(np.max(block_rows[finite]))) + 3
            col_max = math.ceil(float(np.max(block_cols[finite]))) + 3
            window = Window(
                col_min,
                row_min,
                max(1, col_max - col_min + 1),
                max(1, row_max - row_min + 1),
            )
            fill = np.nan if np.issubdtype(dataset.dtypes[0], np.floating) else 0
            values = np.asarray(
                dataset.read(1, window=window, boundless=True, fill_value=fill),
                dtype=np.float64,
            )
            if source_nodata is not None:
                values[np.isclose(values, source_nodata)] = np.nan
            result[row_start:row_stop, col_start:col_stop] = _sample_biquintic(
                values,
                block_rows - row_min,
                block_cols - col_min,
            )
    return result


def _sample_paths(paths: list[Path], entry: DemSource, grid: GridSpec) -> np.ndarray:
    """Sample all native source files directly onto one target grid."""
    import rasterio
    from affine import Affine

    columns, rows = np.meshgrid(
        np.arange(grid.width, dtype=np.float64) + 0.5,
        np.arange(grid.height, dtype=np.float64) + 0.5,
    )
    target_x, target_y = Affine(*grid.transform) * (columns, rows)
    output = np.full(grid.shape, np.nan, dtype=np.float64)
    recipe = entry.mosaic_recipe()
    for path in paths:
        with rasterio.open(_source_path(path, recipe)) as dataset:
            source_crs = (
                dataset.crs
                or getattr(recipe, "source_crs", None)
                or "EPSG:4326"
            )
            sampled = _sample_dataset(
                dataset,
                np.asarray(target_x),
                np.asarray(target_y),
                target_crs=grid.crs,
                source_crs=source_crs,
                source_nodata=(
                    getattr(recipe, "nodata", None)
                    if getattr(recipe, "nodata", None) is not None
                    else dataset.nodata
                ),
            )
            overlap = np.isfinite(output) & np.isfinite(sampled)
            if np.any(overlap & ~np.isclose(output, sampled, atol=1e-3, rtol=1e-6)):
                message = f"overlapping DEM source files disagree: {path.name}"
                logger.error(message)
                raise RuntimeError(message)
            fill = np.isnan(output) & np.isfinite(sampled)
            output[fill] = sampled[fill]
    return output.astype(np.float32)


def _materialize_entry(
    entry: DemSource,
    bounds: tuple[float, float, float, float],
    grid: GridSpec,
    *,
    cache_dir: Path,
    budget: ResourceBudget | None,
) -> RasterDEM:
    """Plan, fetch, and directly sample one P0030 registry entry."""
    from faninsar.processing.dem.api import RasterDEM

    paths = _execute_plan(entry.plan(bounds), cache_dir, budget=budget)
    values = _sample_paths(paths, entry, grid)
    return RasterDEM(
        array=values,
        grid=grid,
        vertical_datum=entry.vertical_datum,
        provenance={
            "provider": entry.provider,
            "product": entry.product,
            "resampling": "direct-source-target",
        },
    )


def _materialize_auto(
    grid: GridSpec,
    *,
    cache_dir: Path,
    budget: ResourceBudget | None,
) -> RasterDEM:
    """Materialize GLO-30 with per-cell GLO-90 rescue for withheld cells."""
    from faninsar.processing.dem.api import RasterDEM
    from faninsar.processing.geometry.dem_sources import get_dem_source

    bounds = _bounds(grid)
    primary = get_dem_source("auto")
    fallback = get_dem_source("glo90")
    primary_plan = primary.plan(bounds)
    fallback_plan = fallback.plan(bounds)
    statuses: dict[str, int | None] = {}
    from faninsar.processing.geometry import dem_transport

    for tile in primary_plan.tiles:
        try:
            response = dem_transport.thread_local_session().head(
                tile.url, timeout=dem_transport.REQUEST_TIMEOUT
            )
            statuses[tile.url] = int(response.status_code)
            response.close()
        except Exception:
            statuses[tile.url] = None
    primary_tiles = []
    fallback_tiles = []
    for tile in primary_plan.tiles:
        if statuses.get(tile.url) == 404:
            tag = tile.cache_path.parts[0]
            twin = next(
                (
                    candidate
                    for candidate in fallback_plan.tiles
                    if candidate.cache_path.parts[0] == tag
                ),
                None,
            )
            if twin is not None:
                fallback_tiles.append(twin)
        else:
            primary_tiles.append(tile)
    primary_paths = (
        _execute_plan(
            replace(primary_plan, tiles=tuple(primary_tiles)),
            cache_dir,
            budget=budget,
        )
        if primary_tiles
        else []
    )
    fallback_paths = (
        _execute_plan(
            replace(fallback_plan, tiles=tuple(fallback_tiles)),
            cache_dir,
            budget=budget,
        )
        if fallback_tiles
        else []
    )
    output = _sample_paths(primary_paths, primary, grid)
    if fallback_paths:
        rescued = _sample_paths(fallback_paths, fallback, grid)
        fill = np.isnan(output) & np.isfinite(rescued)
        output[fill] = rescued[fill]
    return RasterDEM(
        array=output,
        grid=grid,
        vertical_datum=primary.vertical_datum,
        provenance={
            "provider": primary.provider,
            "product": "auto",
            "fallback": "glo90" if fallback_paths else "none",
            "resampling": "direct-source-target",
        },
    )


def materialize(
    source: _P0030Source,
    grid: GridSpec,
    *,
    cache_dir: Path,
    budget: ResourceBudget | None = None,
) -> RasterDEM:
    """Materialize one registered P0030 source without the legacy manager."""
    from faninsar.processing.geometry.dem_sources import get_dem_source

    if source.product == "auto":
        return _materialize_auto(grid, cache_dir=Path(cache_dir), budget=budget)
    return _materialize_entry(
        get_dem_source(source.collection_id),
        _bounds(grid),
        grid,
        cache_dir=Path(cache_dir),
        budget=budget,
    )
