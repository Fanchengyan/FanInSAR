"""Bounded NISAR RSLC scene provider (PROPOSAL-0035).

The provider is intentionally small.  It reads one common radar window from
each normalized RSLC, optionally maps both windows to one caller-supplied
geographic grid, and publishes the resulting pair as a normal scene store.
Stack remains responsible for lifecycle, markers, alignment identity, and all
downstream interferometric products.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.coordinates import GeoGrid, RadarGrid
from faninsar.processing.errors import InvalidProcessingStateError, reject_invalid_state
from faninsar.processing.slc import RadarSLC
from faninsar.processing.stack.provider import UnsupportedStackCapabilityError
from faninsar.processing.stack.scene_store import (
    scene_grid_identity,
    write_scene_unit,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from faninsar.processing.contracts import SLCProduct
    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.processing.stack.provider import SceneProductionCallback

logger = setup_logger(__name__)


def _source_digest(path: Path) -> tuple[str, str]:
    """Return canonical source id and content SHA-256 for one RSLC path."""
    source_id = str(path.expanduser().resolve(strict=False))
    digest = hashlib.sha256()
    try:
        with Path(source_id).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except (FileNotFoundError, NotADirectoryError):
        digest.update(f"missing:{source_id}".encode())
    except OSError as error:
        logger.exception("NISAR source cannot be snapshotted: %s", source_id)
        reject_invalid_state(
            f"NISAR source cannot be snapshotted: {source_id}: {error}"
        )
    return source_id, digest.hexdigest()


def _dem_identity(dem: object) -> str:
    """Return a stable, human-readable identity for a geometry DEM."""
    qualified_name = f"{type(dem).__module__}.{type(dem).__qualname__}"
    height = getattr(dem, "height_m", None)
    if height is not None:
        try:
            return f"{qualified_name}:height_m={float(height):.17g}"
        except (TypeError, ValueError):
            return f"{qualified_name}:height_m={height!r}"
    path = getattr(dem, "path", None)
    if path is not None:
        return f"{qualified_name}:path={Path(path).expanduser().resolve(strict=False)}"
    return qualified_name


@dataclass(frozen=True, slots=True)
class NisarPairState:
    """Minimal state consumed by :meth:`Stack.coregister_scenes`.

    The bounded provider does not estimate residual offsets.  Explicit zero
    values make that fact part of the state rather than leaving marker
    generation to infer missing fields.
    """

    pair_id: str
    range_shift_px: float = 0.0
    azimuth_shift_px: float = 0.0
    esd_azimuth_shift_px: float = 0.0
    amplitude_residual_rg_px: float = 0.0
    stage_timings_s: dict[str, float] | None = None
    coregistration_timings_s: dict[str, float] | None = None


def _window(value: object, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Validate a row/column crop against one source shape."""
    if value is None:
        mission = "NISAR RSLC Stack"
        capability = "scene-production"
        reason = (
            "bounded scene production requires an explicit nisar_window; "
            "full-scene promotion is unsupported"
        )
        raise UnsupportedStackCapabilityError(
            mission,
            capability,
            reason,
        )
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        reject_invalid_state(
            "NISAR provider window must be (row_start, row_stop, col_start, col_stop)"
        )
    try:
        row_start, row_stop, col_start, col_stop = (int(item) for item in value)
    except (TypeError, ValueError) as error:
        reject_invalid_state(f"NISAR provider window is not integral: {error}")
    if not (
        0 <= row_start < row_stop <= shape[0] and 0 <= col_start < col_stop <= shape[1]
    ):
        reject_invalid_state(
            f"NISAR provider window {(row_start, row_stop, col_start, col_stop)} "
            f"is outside source shape {shape}"
        )
    return row_start, row_stop, col_start, col_stop


def _radar_crop(
    product: SLCProduct,
    array: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> RadarSLC:
    """Attach one native crop to a matching radar product contract."""
    if not isinstance(product.grid, RadarGrid):
        reject_invalid_state("NISAR RSLC provider requires radar-grid products")
    row_start, row_stop, col_start, col_stop = bounds
    grid = replace(
        product.grid,
        shape=(row_stop - row_start, col_stop - col_start),
        starting_slant_range_m=(
            product.grid.starting_slant_range_m
            + col_start * product.grid.range_spacing_m
        ),
        sensing_start=product.grid.sensing_start
        + timedelta(seconds=row_start * product.grid.azimuth_time_interval_s),
    )
    descriptor = replace(product.samples, shape=grid.shape)
    return RadarSLC(
        product=replace(product, grid=grid, samples=descriptor),
        samples=np.asarray(array, dtype=np.complex64),
    )


def _shared_radar_window(
    reference: RadarGrid,
    secondary: RadarGrid,
    reference_bounds: tuple[int, int, int, int],
    *,
    reference_product: SLCProduct | None = None,
    secondary_product: SLCProduct | None = None,
    device: str = "cpu",
    dem: DEMSampler | None = None,
    height_m: float | None = None,
) -> tuple[int, int, int, int]:
    """Map a reference crop onto the secondary radar grid physically.

    NISAR acquisitions can have different zero-Doppler starts and slant-range
    origins.  A common pixel index is therefore not a common physical target;
    this mapping keeps the target range/time and only then selects secondary
    indices.
    """
    row_start, row_stop, col_start, col_stop = reference_bounds
    target_time = reference.sensing_start + timedelta(
        seconds=row_start * reference.azimuth_time_interval_s
    )
    target_range = (
        reference.starting_slant_range_m + col_start * reference.range_spacing_m
    )
    secondary_row_start = round(
        (target_time - secondary.sensing_start).total_seconds()
        / secondary.azimuth_time_interval_s
    )
    secondary_col_start = round(
        (target_range - secondary.starting_slant_range_m) / secondary.range_spacing_m
    )
    rows = row_stop - row_start
    cols = col_stop - col_start
    secondary_bounds = (
        secondary_row_start,
        secondary_row_start + rows,
        secondary_col_start,
        secondary_col_start + cols,
    )
    if reference_product is None or secondary_product is None:
        # Explicit degraded mode for callers that only have grid metadata.
        # A normalized NISAR pair always supplies products and therefore uses
        # the physical geometry seam below, even when this seed is in bounds.
        _window(secondary_bounds, secondary.shape)
        return secondary_bounds
    return _geometry_shared_radar_window(
        reference_product,
        secondary_product,
        reference_bounds,
        device=device,
        dem=dem,
        height_m=height_m,
    )


def _geometry_shared_radar_window(
    reference_product: SLCProduct,
    secondary_product: SLCProduct,
    reference_bounds: tuple[int, int, int, int],
    *,
    device: str,
    dem: DEMSampler | None = None,
    height_m: float | None = None,
) -> tuple[int, int, int, int]:
    """Map a bounded crop through the shared Radar→Geo→Radar geometry seam."""
    from faninsar.processing.geometry import RadarGeometryModel
    from faninsar.processing.geometry.dem import ConstantHeightDEM
    from faninsar.processing.geometry.prepare_production import (
        run_geo2rdr,
        run_rdr2geo,
    )

    if not isinstance(reference_product.grid, RadarGrid) or not isinstance(
        secondary_product.grid, RadarGrid
    ):
        reject_invalid_state("NISAR geometry crop mapping requires radar products")
    reference_bounds = _window(reference_bounds, reference_product.grid.shape)
    if dem is None and height_m is None:
        reject_invalid_state(
            "NISAR geometry crop mapping requires an explicit DEM or height"
        )
    if dem is not None and height_m is not None:
        reject_invalid_state(
            "NISAR geometry crop mapping cannot combine DEM and height inputs"
        )
    if height_m is not None:
        try:
            resolved_height = float(height_m)
        except (TypeError, ValueError) as error:
            reject_invalid_state(f"NISAR geometry mapping height is invalid: {error}")
        if not np.isfinite(resolved_height):
            reject_invalid_state("NISAR geometry mapping height must be finite")
        dem = ConstantHeightDEM(resolved_height)
    row_start, row_stop, col_start, col_stop = reference_bounds
    center_row = np.array(
        [[(row_start + row_stop - 1) / 2.0]],
        dtype=np.float64,
    )
    center_col = np.array(
        [[(col_start + col_stop - 1) / 2.0]],
        dtype=np.float64,
    )
    reference_model = RadarGeometryModel.from_radar_grid(
        reference_product.grid,
        reference_product.orbit,
    )
    secondary_model = RadarGeometryModel.from_radar_grid(
        secondary_product.grid,
        secondary_product.orbit,
    )
    try:
        ground = run_rdr2geo(
            reference_model,
            center_row,
            center_col,
            dem,
            device=device,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        logger.exception("NISAR reference crop geometry mapping failed")
        reject_invalid_state(f"NISAR reference crop rdr2geo failed: {error}")
    converged = getattr(ground, "converged", None)
    converged_values = (
        np.asarray(converged, dtype=bool).reshape(-1)
        if converged is not None
        else np.array([], dtype=bool)
    )
    if not converged_values.size or not bool(converged_values[0]):
        reject_invalid_state("NISAR reference crop target did not converge in rdr2geo")
    try:
        ground_latitude = ground.latitude_deg
        ground_longitude = ground.longitude_deg
        ground_height = ground.height_m
    except AttributeError as error:
        reject_invalid_state(
            f"NISAR reference crop rdr2geo result is missing coordinates: {error}"
        )
    try:
        mapped = run_geo2rdr(
            secondary_model,
            ground_latitude,
            ground_longitude,
            ground_height,
            device=device,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        logger.exception("NISAR secondary crop geometry mapping failed")
        reject_invalid_state(f"NISAR secondary crop geo2rdr failed: {error}")
    mapped_converged = getattr(mapped, "converged", None)
    mapped_converged_values = (
        np.asarray(mapped_converged, dtype=bool).reshape(-1)
        if mapped_converged is not None
        else np.array([], dtype=bool)
    )
    if not mapped_converged_values.size or not bool(mapped_converged_values[0]):
        reject_invalid_state("NISAR shared crop target did not converge in geo2rdr")
    mapped_azimuth = np.asarray(
        getattr(mapped, "azimuth_index", np.array([])), dtype=np.float64
    ).reshape(-1)
    mapped_range = np.asarray(
        getattr(mapped, "range_index", np.array([])), dtype=np.float64
    ).reshape(-1)
    if (
        mapped_azimuth.size == 0
        or mapped_range.size == 0
        or not np.isfinite(mapped_azimuth[0])
        or not np.isfinite(mapped_range[0])
    ):
        reject_invalid_state(
            "NISAR shared crop target returned non-finite radar indices in geo2rdr"
        )
    rows = row_stop - row_start
    cols = col_stop - col_start
    secondary_row_start = round(float(mapped_azimuth[0])) - rows // 2
    secondary_col_start = round(float(mapped_range[0])) - cols // 2
    secondary_bounds = (
        secondary_row_start,
        secondary_row_start + rows,
        secondary_col_start,
        secondary_col_start + cols,
    )
    try:
        return _window(secondary_bounds, secondary_product.grid.shape)
    except InvalidProcessingStateError as error:
        reject_invalid_state(
            "NISAR mapped secondary radar crop lies outside the source grid; "
            f"full-window rejection: {error}"
        )


def _geo_target(value: object) -> GeoGrid:
    """Normalize a public ``GeoGrid`` or ``GeoGridSpec`` target."""
    if isinstance(value, GeoGrid):
        return value
    if value is None:
        reject_invalid_state(
            "NISAR geographic processing requires an explicit geo_grid target"
        )
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = (getattr(value, "height", 0), getattr(value, "width", 0))
    crs = getattr(value, "crs", None)
    transform = getattr(value, "transform", None)
    if not crs or transform is None:
        reject_invalid_state("NISAR geo_grid must expose crs, transform, and shape")
    try:
        # ``GeoGridSpec`` stores ``(x0, dx, 0, y0, 0, dy)`` while the
        # processing ``GeoGrid`` contract stores ``(dx, 0, x0, 0, dy, y0)``.
        x0, dx, _, y0, _, dy = tuple(transform)
        return GeoGrid(
            shape=(int(shape[0]), int(shape[1])),
            crs=str(crs),
            transform=(float(dx), 0.0, float(x0), 0.0, float(dy), float(y0)),
        )
    except (TypeError, ValueError, IndexError) as error:
        reject_invalid_state(f"invalid NISAR geo_grid target: {error}")


def _provider_window(options: Mapping[str, Any], configured: object) -> object:
    """Resolve the explicit NISAR crop option without changing Stack options."""
    for key in ("nisar_window", "rslc_window", "window"):
        if key in options:
            return options[key]
    return configured


def _geometry_inputs(
    options: Mapping[str, Any],
    *,
    configured_dem: DEMSampler | None,
    configured_height: float | None,
) -> tuple[DEMSampler | None, float | None]:
    """Resolve per-call geometry inputs over callback-level configuration."""
    option_dem = options.get("dem")
    dem = option_dem if option_dem is not None else configured_dem
    height = options.get("height_m", options.get("height"))
    if height is None:
        height = configured_height
    return dem, height


def make_nisar_scene_provider(
    *,
    sensor: Any,
    handles: Mapping[Path, Any],
    products: Mapping[str, SLCProduct],
    lineage: Mapping[str, str],
    master: str,
    channel: tuple[str, str],
    configured_window: object = None,
    configured_dem: DEMSampler | None = None,
    configured_height: float | None = None,
    admission_lineage: Mapping[str, Mapping[str, object]] | None = None,
) -> SceneProductionCallback:
    """Build a callback that publishes one bounded NISAR pair scene.

    Parameters
    ----------
    sensor : object
        Normalized sensor exposing ``read_slc_window``.
    handles : mapping of pathlib.Path to object
        Lazy reader handles opened during stack construction.
    products, lineage : mappings
        Date-keyed normalized products and source paths.
    master : str
        Alignment master date used in scene manifests.
    channel : tuple of str
        Admitted ``(frequency, polarization)`` pair.
    configured_window : sequence of int, optional
        Default crop as ``(row_start, row_stop, col_start, col_stop)``.
    configured_dem : DEMSampler, optional
        Callback-level DEM used when a scene-production call omits ``dem``.
    configured_height : float, optional
        Callback-level constant height used when a scene-production call omits
        both ``dem`` and ``height``.
    admission_lineage : mapping, optional
        Date-keyed trusted pre-open metadata.  It is copied into the scene
        manifest so source identity and policy survive stack publication.

    Returns
    -------
    SceneProductionCallback
        Provider callback accepted by :class:`StackSceneProvider`.

    """
    path_dates = {Path(path): date_id for date_id, path in lineage.items()}
    admission_lineage = dict(admission_lineage or {})
    admitted_sources = {
        date_id: _source_digest(Path(path)) for date_id, path in lineage.items()
    }

    def produce_pair(
        reference_path: Path,
        secondary_path: Path,
        *,
        output_dir: Path,
        options: Mapping[str, Any],
    ) -> NisarPairState:
        reference_date = path_dates.get(Path(reference_path))
        secondary_date = path_dates.get(Path(secondary_path))
        if reference_date is None or secondary_date is None:
            reject_invalid_state("NISAR provider received an unadmitted source path")
        reference_product = products[reference_date]
        secondary_product = products[secondary_date]
        mapping_dem, mapping_height = _geometry_inputs(
            options,
            configured_dem=configured_dem,
            configured_height=configured_height,
        )
        for date_id, source_path in (
            (reference_date, reference_path),
            (secondary_date, secondary_path),
        ):
            current = _source_digest(Path(source_path))
            if current != admitted_sources[date_id]:
                reject_invalid_state(
                    "NISAR RSLC source path or content changed after admission"
                )
        if not isinstance(reference_product.grid, RadarGrid) or not isinstance(
            secondary_product.grid, RadarGrid
        ):
            reject_invalid_state("NISAR provider requires normalized radar products")
        if reference_product.grid.wavelength_m != secondary_product.grid.wavelength_m:
            reject_invalid_state("NISAR pair radar wavelengths do not match")
        shape = (
            min(reference_product.grid.shape[0], secondary_product.grid.shape[0]),
            min(reference_product.grid.shape[1], secondary_product.grid.shape[1]),
        )
        bounds = _window(_provider_window(options, configured_window), shape)
        secondary_bounds = _shared_radar_window(
            reference_product.grid,
            secondary_product.grid,
            bounds,
            reference_product=reference_product,
            secondary_product=secondary_product,
            device=str(options.get("device", "cpu")),
            dem=mapping_dem,
            height_m=mapping_height,
        )
        row_start, row_stop, col_start, col_stop = bounds
        (
            secondary_row_start,
            secondary_row_stop,
            secondary_col_start,
            secondary_col_stop,
        ) = secondary_bounds
        reference_window = (slice(row_start, row_stop), slice(col_start, col_stop))
        secondary_window = (
            slice(secondary_row_start, secondary_row_stop),
            slice(secondary_col_start, secondary_col_stop),
        )
        try:
            reference_samples = sensor.read_slc_window(
                handles[Path(reference_path)],
                reference_window,
                frequency=channel[0],
                polarization=channel[1],
            )
            secondary_samples = sensor.read_slc_window(
                handles[Path(secondary_path)],
                secondary_window,
                frequency=channel[0],
                polarization=channel[1],
            )
        except KeyError as error:
            reject_invalid_state(f"NISAR source handle is unavailable: {error}")
        reference_radar = _radar_crop(reference_product, reference_samples, bounds)
        secondary_radar = _radar_crop(
            secondary_product,
            secondary_samples,
            secondary_bounds,
        )
        domain = str(options.get("coregistration_grid", "radar")).lower()
        if mapping_dem is None and mapping_height is None:
            reject_invalid_state(
                "NISAR geometry crop mapping requires an explicit DEM or height "
                "in provider options"
            )
        if mapping_dem is not None and mapping_height is not None:
            reject_invalid_state(
                "NISAR geometry crop mapping cannot combine DEM and height inputs"
            )
        if mapping_dem is None:
            from faninsar.processing.geometry.dem import ConstantHeightDEM

            mapping_dem = ConstantHeightDEM(float(mapping_height))
        dem_identity = _dem_identity(mapping_dem)
        if domain == "radar":
            reference_array = reference_radar.samples
            secondary_array = secondary_radar.samples
            grid_shape = reference_radar.grid.shape
            grid_identity = scene_grid_identity("radar", grid_shape)
        elif domain == "geo":
            target = _geo_target(options.get("geo_grid"))
            dem = options.get("dem")
            reference_geo = reference_radar.rdr2geo(
                device=str(options.get("device", "auto")),
                dem=dem,
                geo_grid=target,
                use_cache=False,
            )
            secondary_geo = secondary_radar.rdr2geo(
                device=str(options.get("device", "auto")),
                dem=dem,
                geo_grid=target,
                use_cache=False,
            )
            reference_array = reference_geo.samples
            secondary_array = secondary_geo.samples
            grid_shape = target.shape
            grid_identity = scene_grid_identity(
                "geo",
                grid_shape,
                {"crs": target.crs, "transform": list(target.transform)},
            )
        else:
            reject_invalid_state(
                f"NISAR provider does not support coordinate domain {domain!r}"
            )
        scenes = output_dir / "scenes"
        write_scene_unit(
            scenes,
            date_id=secondary_date,
            master_id=master,
            domain=domain,
            tag="NISAR_b0",
            reference=np.asarray(reference_array, dtype=np.complex64),
            secondary=np.asarray(secondary_array, dtype=np.complex64),
            row_origin=0,
            col_origin=0,
            grid_shape=grid_shape,
            wavelength_m=reference_product.grid.wavelength_m,
            grid_identity=grid_identity,
            scientific_lineage=(
                {
                    "stage": "nisar_rslc_window",
                    "source": str(reference_path),
                    "source_id": admitted_sources[reference_date][0],
                    "source_digest": admitted_sources[reference_date][1],
                    "admission_policy": admission_lineage.get(reference_date, {}).get(
                        "policy"
                    ),
                    "channel": f"{channel[0]}/{channel[1]}",
                    "lineage": "reference",
                    "dem_identity": dem_identity,
                },
                {
                    "stage": "nisar_rslc_window",
                    "source": str(secondary_path),
                    "source_id": admitted_sources[secondary_date][0],
                    "source_digest": admitted_sources[secondary_date][1],
                    "admission_policy": admission_lineage.get(secondary_date, {}).get(
                        "policy"
                    ),
                    "channel": f"{channel[0]}/{channel[1]}",
                    "lineage": "secondary",
                    "dem_identity": dem_identity,
                },
            ),
        )
        return NisarPairState(
            pair_id=f"{reference_date}_{secondary_date}",
            stage_timings_s={},
            coregistration_timings_s={},
        )

    return produce_pair


__all__ = ["NisarPairState", "make_nisar_scene_provider"]
