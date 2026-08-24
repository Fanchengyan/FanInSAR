"""Bounded NISAR RSLC scene provider (PROPOSAL-0035).

The provider is intentionally small.  It reads one common radar window from
each normalized RSLC, optionally maps both windows to one caller-supplied
geographic grid, and publishes the resulting pair as a normal scene store.
Stack remains responsible for lifecycle, markers, alignment identity, and all
downstream interferometric products.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.coordinates import GeoGrid, RadarGrid
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.slc import RadarSLC
from faninsar.processing.stack.scene_store import (
    scene_grid_identity,
    write_scene_unit,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from faninsar.processing.contracts import SLCProduct
    from faninsar.processing.stack.provider import SceneProductionCallback

logger = setup_logger(__name__)


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
        reject_invalid_state(
            "NISAR provider requires an explicit bounded nisar_window; "
            "full-scene promotion is unsupported"
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
        0 <= row_start < row_stop <= shape[0]
        and 0 <= col_start < col_stop <= shape[1]
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
) -> tuple[int, int, int, int]:
    """Map a reference crop onto the secondary radar grid physically.

    NISAR acquisitions can have different zero-Doppler starts and slant-range
    origins.  A common pixel index is therefore not a common physical target;
    this mapping keeps the target range/time and only then selects secondary
    indices.
    """
    row_start, row_stop, col_start, col_stop = reference_bounds
    # Acquisition epochs differ by days, so use the within-scene zero-Doppler
    # offset rather than comparing absolute datetimes across acquisitions.
    target_azimuth_offset = row_start * reference.azimuth_time_interval_s
    target_range = (
        reference.starting_slant_range_m + col_start * reference.range_spacing_m
    )
    secondary_row_start = round(
        target_azimuth_offset / secondary.azimuth_time_interval_s
    )
    secondary_col_start = round(
        (target_range - secondary.starting_slant_range_m)
        / secondary.range_spacing_m
    )
    rows = row_stop - row_start
    cols = col_stop - col_start
    secondary_bounds = (
        secondary_row_start,
        secondary_row_start + rows,
        secondary_col_start,
        secondary_col_start + cols,
    )
    _window(secondary_bounds, secondary.shape)
    return secondary_bounds


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


def make_nisar_scene_provider(
    *,
    sensor: Any,
    handles: Mapping[Path, Any],
    products: Mapping[str, SLCProduct],
    lineage: Mapping[str, str],
    master: str,
    channel: tuple[str, str],
    configured_window: object = None,
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

    Returns
    -------
    SceneProductionCallback
        Provider callback accepted by :class:`StackSceneProvider`.

    """
    path_dates = {Path(path): date_id for date_id, path in lineage.items()}

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
        secondary_radar = _radar_crop(secondary_product, secondary_samples, bounds)
        domain = str(options.get("coregistration_grid", "radar")).lower()
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
                {"stage": "nisar_rslc_window", "source": str(reference_path)},
                {"stage": "nisar_rslc_window", "source": str(secondary_path)},
            ),
        )
        return NisarPairState(
            pair_id=f"{reference_date}_{secondary_date}",
            stage_timings_s={},
            coregistration_timings_s={},
        )

    return produce_pair


__all__ = ["NisarPairState", "make_nisar_scene_provider"]
