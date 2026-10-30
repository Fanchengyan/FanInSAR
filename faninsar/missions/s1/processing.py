"""Sentinel-1 production stages on radar or geographic grids."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np

from faninsar.data.query import BoundingBox, Polygons
from faninsar.logging import setup_logger
from faninsar.missions.s1 import (
    open_safe_product,
    read_eof_orbit,
    read_full_burst,
    read_swath_bursts,
    stitch_bursts,
)
from faninsar.missions.s1.safe import open_acquisition_products
from faninsar.processing.coregistration import (
    combine_offset_fields,
    dense_geometry_offsets,
    estimate_azimuth_shift_esd,
    geometry_offset_window_extent,
    refine_shift_with_correlation,
    resample_complex,
    resample_complex_deramped_reramp,
    resolve_ampcor_policy,
)
from faninsar.processing.coregistration.tops.carrier import carrier_from_swath
from faninsar.processing.coregistration.tops.deramp import (
    TOPSCarrierModel,
    deramp,
    reramp,
)
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.geometry import (
    DEM,
    RadarGeometryModel,
)
from faninsar.processing.geometry.baseline import BaselineComponents
from faninsar.processing.geometry.coordinates import RadarGrid
from faninsar.processing.geometry.prepare_production import (
    run_geo2rdr,
)
from faninsar.processing.interferometry.flatten import (
    azimuth_ramp_device_kwargs,
    compute_geometric_phase_from_geo,
    compute_topographic_phase,
    estimate_residual_azimuth_ramp,
    estimate_residual_topographic_scale,
    remove_azimuth_phase_ramp,
    remove_topographic_phase,
)
from faninsar.processing.interferometry.pair import (
    form_interferogram,
    goldstein_filter,
    mask_invalid_looks,
)
from faninsar.processing.interferometry.products import (
    PairProductArrays,
    write_pair_stac_item,
    write_pair_zarr,
)
from faninsar.processing.runtime.memory import close_memmap, release_memmap_pages
from faninsar.processing.unwrapping import SnaphuConfig, UnwrapBackend
from faninsar.processing.unwrapping import unwrap as unwrap_dispatch

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from faninsar.missions.s1.io import BurstArray
    from faninsar.missions.s1.types import S1Burst, S1Product, S1Swath
    from faninsar.processing.coregistration.offsets import OffsetFieldResult
    from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT
    from faninsar.processing.geometry.prepared import (
        PhaseState,
        ProviderLeaseToken,
        ResourceLimits,
    )
    from faninsar.processing.mosaicking.grid import GeoGridSpec
    from faninsar.processing.runtime.memory import MemoryWatchdog
    from faninsar.processing.unwrapping.common import SpatialUnwrapResult

logger = setup_logger(__name__)

DEM_BOUNDS_BUFFER_M = 2000.0


def _sync_admitted_device(device: object) -> None:
    """Block until admitted CUDA work is visible to the host clock."""
    ident = str(device)
    if not ident.startswith("cuda"):
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _clock_start(device: object) -> float:
    """Return a device-synchronized perf_counter start time."""
    _sync_admitted_device(device)
    return time.perf_counter()


def _clock_stop(device: object, started: float) -> float:
    """Return elapsed seconds after a device-synchronized stop."""
    _sync_admitted_device(device)
    return time.perf_counter() - started


def _merge_coreg_stage_times(
    stage_times: dict[str, float],
    coregistration_timings_s: Mapping[str, float],
) -> None:
    """Copy exclusive coreg substages into the per-burst timing record."""
    for key, value in coregistration_timings_s.items():
        stage_times[f"coreg.{key}"] = float(value)


SPEED_OF_LIGHT_M_S = 299_792_458.0
ScopeMode = Literal["burst", "swath"]
CoregistrationGrid = Literal["radar", "geo"]


class ScientificTransitionRecord(TypedDict):
    """Hash-bound record for one scientific phase transition."""

    operation: str
    operation_id: str
    input_state_digest: str
    output_state_digest: str
    input_payload_digest: str
    output_payload_digest: str
    metadata: str


#: Lanczos half-width of the radar resampler (``resample_complex`` default).
_LANCZOS_RADIUS_PX = 4
#: Extra crop margin beyond the measured offset extent (absorbs coarse-probe
#: curvature and ESD/amplitude-residual drift between growth iterations).
_WINDOW_HALO_MARGIN_PX = 16
#: Floor for the ROI resampling halo when the pair geometry is near zero.
_WINDOW_HALO_FLOOR_PX = 64
#: Window-growth attempts before falling back to full-burst coregistration.
_MAX_WINDOW_HALO_GROWTH = 3


def _lineage_payload_digest(*arrays: np.ndarray | None) -> str:
    """Hash array payloads with shape and dtype for scientific lineage."""
    digest = hashlib.sha256()
    chunk_bytes = 4 * 1024 * 1024
    for array in arrays:
        if array is None:
            digest.update(b"<none>")
            continue
        value = np.asarray(array)
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        if value.ndim == 0:
            digest.update(np.ascontiguousarray(value).view(np.uint8).tobytes())
            continue
        if value.shape[0] == 0:
            continue
        # Hash in bounded C-order chunks.  This preserves the prior digest for
        # contiguous arrays while avoiding a full temporary copy for memmaps or
        # strided views when lineage is enabled.
        rows_per_chunk = max(
            1,
            chunk_bytes // max(1, value[0].nbytes),
        )
        for row_start in range(0, value.shape[0], rows_per_chunk):
            chunk = np.ascontiguousarray(value[row_start : row_start + rows_per_chunk])
            digest.update(chunk.view(np.uint8).tobytes())
    return digest.hexdigest()


def _phase_state_manifest(state: ProductionPairState) -> dict[str, object] | None:
    """Serialize the typed phase state for a scene-unit manifest."""
    phase_state = state.phase_state
    if phase_state is None:
        return None
    return {
        "carrier": phase_state.carrier.value,
        "registration_model": phase_state.registration_model,
        "geometric_phase": phase_state.geometric_phase.value,
        "phase_model_id": phase_state.phase_model_id,
        "phase_lineage_id": phase_state.phase_lineage_id,
        "residual_solution_id": phase_state.residual_solution_id,
        "residual_application_id": phase_state.residual_application_id,
        "transition_digest": phase_state.transition_digest,
        "transition_history": [
            {
                "operation_id": transition.operation_id,
                "input_state_digest": transition.input_state_digest,
                "output_state_digest": transition.output_state_digest,
                "input_payload_digest": transition.input_payload_digest,
                "output_payload_digest": transition.output_payload_digest,
            }
            for transition in phase_state.transition_history
        ],
    }


def _record_scientific_transition(
    state: ProductionPairState,
    operation: str,
    input_arrays: tuple[np.ndarray | None, ...],
    output_arrays: tuple[np.ndarray | None, ...],
    **metadata: object,
) -> None:
    """Append one ordered, hash-bound scientific operation when enabled."""
    if not state.record_scientific_lineage:
        return
    operation_order = {
        "deramp": 0,
        "measure_residual_solution": 1,
        "apply_carrier_residual_range_phase": 2,
        "apply_geo_carrier_residual_geometric_phase": 2,
        "form_interferogram": 3,
        "apply_geometric_phase_flatten": 4,
        "spatial_unwrap_and_residual_screen": 5,
    }
    if operation not in operation_order:
        reject_invalid_state(f"unknown scientific lineage operation: {operation}")
    if any(item["operation"] == operation for item in state.scientific_lineage):
        reject_invalid_state(f"scientific lineage operation repeated: {operation}")
    if state.scientific_lineage:
        previous_operation = state.scientific_lineage[-1]["operation"]
        if operation_order[operation] < operation_order[previous_operation]:
            reject_invalid_state(
                "scientific lineage operation out of order: "
                + previous_operation
                + " -> "
                + operation
            )
    input_payload_digest = _lineage_payload_digest(*input_arrays)
    output_payload_digest = _lineage_payload_digest(*output_arrays)
    previous_state_digest = (
        state.scientific_lineage[-1]["output_state_digest"]
        if state.scientific_lineage
        else hashlib.sha256(state.pair_id.encode()).hexdigest()
    )
    operation_id = hashlib.sha256(
        json.dumps(
            [state.pair_id, operation, previous_state_digest, input_payload_digest],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    output_state_digest = hashlib.sha256(
        json.dumps(
            [previous_state_digest, operation_id, output_payload_digest],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    state.scientific_lineage.append(
        {
            "operation": operation,
            "operation_id": operation_id,
            "input_state_digest": previous_state_digest,
            "output_state_digest": output_state_digest,
            "input_payload_digest": input_payload_digest,
            "output_payload_digest": output_payload_digest,
            "metadata": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        }
    )
    if operation in {
        "apply_carrier_residual_range_phase",
        "apply_geo_carrier_residual_geometric_phase",
    }:
        from faninsar.processing.geometry.prepared import (
            GeometricPhase,
            PhaseCarrier,
            PhaseState,
            ResidualSolution,
            ResidualStatus,
        )

        if state.phase_state is None:
            state.phase_state = PhaseState(
                carrier=PhaseCarrier.DERAMPED,
                registration_model="network_relative",
                geometric_phase=GeometricPhase.NONE,
                phase_model_id="p18-p19.scientific-lineage.v1",
                phase_lineage_id=hashlib.sha256(
                    state.pair_id.encode("utf-8")
                ).hexdigest(),
                residual_solution_id=None,
                residual_application_id=None,
            )
        solution = ResidualSolution(
            solution_id=operation_id,
            kind=(
                "network"
                if metadata.get("residual_policy", metadata.get("solution_kind"))
                == "network"
                else "pair"
            ),
            parent_observation_ids=(operation_id,),
            payload_digest=output_payload_digest,
            application_id=operation_id,
            status=ResidualStatus.VALID,
        )
        state.phase_state = state.phase_state.apply_solution(
            solution,
            payload_digest=input_payload_digest,
            operation_id=operation_id,
        )


def _align_crop_end(end: int, start: int, stride: int, limit: int) -> int:
    """Extend a crop end so ``end - 1`` is a full-burst control coordinate."""
    if end >= limit:
        return limit
    aligned = start + ((end - 1 - start) // stride + 1) * stride + 1
    return min(aligned, limit)


def _conform_to_shape(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Crop or zero-pad a 2-D array at its trailing edges to ``shape``."""
    rows = min(values.shape[0], shape[0])
    cols = min(values.shape[1], shape[1])
    conformed = np.zeros(shape, dtype=values.dtype)
    conformed[:rows, :cols] = values[:rows, :cols]
    return conformed


def reference_window_crop(
    window: tuple[int, int, int, int],
    shape: tuple[int, int],
    control_spacing: int | None = None,
) -> tuple[int, int, int, int]:
    """Return the pair-independent Reference crop of one radar ROI window.

    Every pair crop is built from the same window with a halo of at least
    ``_WINDOW_HALO_FLOOR_PX`` (it only grows with the pair's offsets), so the
    floor-halo crop is contained in every pair crop of the window. Persisting
    the Reference and all secondaries on this crop keeps one Reference payload
    valid for the whole Stack.

    Parameters
    ----------
    window : tuple[int, int, int, int]
        Radar ROI window ``(row0, row1, col0, col1)`` inside the burst.
    shape : tuple[int, int]
        Full deramped burst shape.
    control_spacing : int, optional
        Geometry control stride; ``None`` uses the coregistration default.

    Returns
    -------
    tuple[int, int, int, int]
        Crop ``(row0, row1, col0, col1)`` in burst coordinates.

    """
    stride = 8 if control_spacing is None else int(control_spacing)
    return _window_crop_bounds(window, shape, stride, _WINDOW_HALO_FLOOR_PX)


def _window_crop_bounds(
    window: tuple[int, int, int, int],
    shape: tuple[int, int],
    stride: int,
    halo: int,
) -> tuple[int, int, int, int]:
    """Return the stride-aligned crop around a radar window.

    The leading edge is floored to a control-grid multiple and the trailing
    edge is extended so the crop's appended edge control point (``row1 - 1``)
    coincides with the full-burst control grid.  The windowed control grid is
    then a subset of the full-burst grid, which keeps the windowed offset
    field bitwise identical to the full-burst field inside the crop.
    """
    height, width = shape
    wr0, wr1, wc0, wc1 = window
    row0 = max(0, (wr0 - halo) // stride * stride)
    col0 = max(0, (wc0 - halo) // stride * stride)
    row1 = _align_crop_end(wr1 + halo, row0, stride, height)
    col1 = _align_crop_end(wc1 + halo, col0, stride, width)
    return row0, row1, col0, col1


def _window_resample_halo_required(
    offsets: OffsetFieldResult,
    window: tuple[int, int, int, int],
    origin: tuple[int, int],
    lanczos_radius_px: int = _LANCZOS_RADIUS_PX,
) -> int:
    """Return the crop margin needed for lossless windowed resampling.

    ``resample_complex`` samples ``source = output - offset`` with a Lanczos
    kernel of half-width ``lanczos_radius_px``, so the crop must extend past
    the window by at least ``max|offset| + kernel radius + 1`` on every side;
    a smaller margin silently reads zero-filled pixels inside the ROI.
    """
    wr0, wr1, wc0, wc1 = window
    row0, col0 = origin
    az = offsets.azimuth_offset_px[wr0 - row0 : wr1 - row0, wc0 - col0 : wc1 - col0]
    rg = offsets.range_offset_px[wr0 - row0 : wr1 - row0, wc0 - col0 : wc1 - col0]
    if az.size == 0:
        return lanczos_radius_px + 1
    magnitude = np.hypot(rg, az)
    finite = np.isfinite(magnitude)
    extent = 0.0 if not np.any(finite) else float(np.nanmax(magnitude[finite]))
    return int(np.ceil(extent)) + lanczos_radius_px + 1


def _window_halo_sufficient(
    offsets: OffsetFieldResult,
    window: tuple[int, int, int, int],
    origin: tuple[int, int],
    crop: tuple[int, int, int, int],
    burst_shape: tuple[int, int],
    lanczos_radius_px: int = _LANCZOS_RADIUS_PX,
) -> bool:
    """Return whether the crop margin covers the windowed resample footprint.

    Sides touching the burst edge are exempt: the full-burst run reads
    zero-filled samples there as well, so both results stay identical.
    """
    required = _window_resample_halo_required(
        offsets, window, origin, lanczos_radius_px
    )
    wr0, wr1, wc0, wc1 = window
    cr0, cr1, cc0, cc1 = crop
    height, width = burst_shape
    if cr0 > 0 and wr0 - cr0 < required:
        return False
    if cr1 < height and cr1 - wr1 < required:
        return False
    if cc0 > 0 and wc0 - cc0 < required:
        return False
    return not (cc1 < width and cc1 - wc1 < required)


def _scene_id(path: Path) -> str:
    stem = path.stem.replace(".SAFE", "")
    for part in stem.split("_"):
        if len(part) >= 8 and part[:8].isdigit():
            return part[:8]
    return stem


def _radar_model(
    swath: S1Swath,
    burst: S1Burst,
    *,
    shape: tuple[int, int],
    row0: int,
    col0: int,
) -> RadarGeometryModel:
    """Build geometry model whose (0,0) is the array origin."""
    local_line = row0 - burst.index * swath.lines_per_burst
    sensing_start = burst.azimuth_time + timedelta(
        seconds=float(local_line) * swath.azimuth_time_interval_s
    )
    starting_slant_range_m = (
        swath.slant_range_time_s * SPEED_OF_LIGHT_M_S / 2.0
        + float(col0) * swath.range_pixel_spacing_m
    )
    grid = RadarGrid(
        shape=shape,
        starting_slant_range_m=starting_slant_range_m,
        range_spacing_m=swath.range_pixel_spacing_m,
        sensing_start=sensing_start,
        azimuth_time_interval_s=swath.azimuth_time_interval_s,
        wavelength_m=SPEED_OF_LIGHT_M_S / swath.radar_frequency_hz,
        look_direction="right",
    )
    return RadarGeometryModel.from_radar_grid(grid, swath.orbit)


@dataclass(frozen=True, slots=True)
class ProductionScene:
    """Full-burst or stitched-swath scene ready for pair processing."""

    scene_id: str
    path: Path
    product: S1Product
    swath: S1Swath
    burst: S1Burst
    array: BurstArray
    carrier: TOPSCarrierModel
    geometry: RadarGeometryModel


@dataclass(frozen=True, slots=True)
class PreparedGeometryField:
    """Exact dense geometry result reusable by a later product pass.

    The field is captured after the existing ``dense_geometry_offsets`` call
    and carries the crop origin that was used for its geometry models.  Reuse
    therefore bypasses only the repeated solver call; residual addition,
    Lanczos resampling, phase handling, and all dtypes remain unchanged.
    """

    field: OffsetFieldResult
    source_shape: tuple[int, int]
    crop_bounds: tuple[int, int, int, int]
    control_spacing: int

    def __post_init__(self) -> None:
        """Validate that the prepared field covers its declared crop."""
        if len(self.source_shape) != 2 or any(size <= 0 for size in self.source_shape):
            reject_invalid_state("prepared geometry source shape must be positive")
        if len(self.crop_bounds) != 4 or any(value < 0 for value in self.crop_bounds):
            reject_invalid_state("prepared geometry crop bounds are invalid")
        row0, row1, col0, col1 = self.crop_bounds
        if not (row0 < row1 <= self.source_shape[0]):
            reject_invalid_state("prepared geometry row crop is outside the source")
        if not (col0 < col1 <= self.source_shape[1]):
            reject_invalid_state("prepared geometry column crop is outside the source")
        expected_shape = (row1 - row0, col1 - col0)
        if any(
            array.shape != expected_shape
            for array in (
                self.field.range_offset_px,
                self.field.azimuth_offset_px,
                self.field.coverage,
                self.field.uncertainty_px,
            )
        ):
            reject_invalid_state(
                "prepared geometry field shape does not match its crop"
            )
        if (
            self.field.range_offset_px.dtype != np.float32
            or self.field.azimuth_offset_px.dtype != np.float32
            or self.field.uncertainty_px.dtype != np.float32
            or self.field.coverage.dtype != bool
        ):
            reject_invalid_state("prepared geometry field dtypes are not canonical")
        if self.control_spacing < 1:
            reject_invalid_state("prepared geometry control spacing must be positive")


def read_prepared_geometry_field(
    provider: object,
    geometry_handle: object,
    token: object,
) -> PreparedGeometryField:
    """Read and validate one provider-bound geometry field for production.

    The provider owns generation pinning, payload hashes, and NumPy decoding;
    this adapter only maps its validated read-only arrays into the production
    ``OffsetFieldResult`` consumed by :func:`stage_coregister`.  It deliberately
    does not accept raw paths or caller-supplied shapes.

    Parameters
    ----------
    provider : object
        Provider implementing ``read_prepared_geometry(handle, token)``.
    geometry_handle : object
        Capability-bound geometry handle returned by the provider.
    token : object
        Live provider lease token.

    Returns
    -------
    PreparedGeometryField
        Validated, immutable dense geometry field for a later product pass.

    Raises
    ------
    InvalidProcessingStateError
        If the provider does not expose the read seam or returns an invalid
        payload record.

    """
    reader = getattr(provider, "read_prepared_geometry", None)
    if not callable(reader):
        reject_invalid_state("prepared provider does not expose read_prepared_geometry")
    from faninsar.processing.geometry.prepared import (
        PreparedGeometryArrayPayload,
    )

    payload = reader(geometry_handle, token)
    if not isinstance(payload, PreparedGeometryArrayPayload):
        reject_invalid_state(
            "prepared provider returned an invalid geometry payload record"
        )
    from faninsar.processing.coregistration.offsets import OffsetFieldResult

    return PreparedGeometryField(
        field=OffsetFieldResult(
            range_offset_px=payload.range_offset_px,
            azimuth_offset_px=payload.azimuth_offset_px,
            coverage=payload.coverage,
            uncertainty_px=payload.uncertainty_px,
        ),
        source_shape=payload.source_shape,
        crop_bounds=payload.crop_bounds,
        control_spacing=payload.control_spacing,
    )


def read_prepared_lut(
    provider: object,
    lut_handle: object,
    token: object,
) -> Geo2RdrLUT:
    """Read one provider-owned LUT without reopening a raw memmap path.

    Parameters
    ----------
    provider : object
        Provider implementing ``read_prepared_lut(handle, token)``.
    lut_handle : object
        Capability-bound LUT handle returned by the provider.
    token : object
        Live provider lease token.

    Returns
    -------
    Geo2RdrLUT
        Immutable-array LUT with the provider's explicit crop origin and
        full radar shape.

    Raises
    ------
    InvalidProcessingStateError
        If the provider does not expose the read seam or returns an invalid
        payload record.

    """
    reader = getattr(provider, "read_prepared_lut", None)
    if not callable(reader):
        reject_invalid_state("prepared provider does not expose read_prepared_lut")
    from faninsar.processing.geometry.prepared import (
        PreparedLutArrayPayload,
    )

    payload = reader(lut_handle, token)
    if not isinstance(payload, PreparedLutArrayPayload):
        reject_invalid_state("prepared provider returned an invalid LUT payload")
    from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT

    row0, _, col0, _ = payload.crop_bounds
    return Geo2RdrLUT(
        az_full=payload.az_full,
        rg_full=payload.rg_full,
        valid=payload.valid,
        full_radar_shape=payload.full_radar_shape,
        height_m=payload.height_m,
        height_full=payload.height_full,
        row0=row0,
        col0=col0,
    )


def _worker_prepared_provider(
    root: str | Path,
    token: ProviderLeaseToken,
    resource_limits: ResourceLimits | None,
) -> object:
    """Attach a worker-local provider to an already pinned generation.

    Prepared generations are owned and pinned by the parent process.  A
    spawned worker therefore receives only the serializable store root and
    lease capability, then asks the provider adapter to attach that lease to
    its own read registry.  The compatibility fallback supports providers
    exposing the instance-level ``attach_worker_lease`` seam while the
    class-level ``from_worker_generation`` adapter is deployed.

    Parameters
    ----------
    root : str or pathlib.Path
        Provider-owned prepared-generation store root.
    token : ProviderLeaseToken
        Serialized capability for the pinned generation.
    resource_limits : ResourceLimits or None
        Authenticated limits bound to the token, when available.

    Returns
    -------
    object
        Provider adapter suitable for :func:`read_prepared_lut`.

    Raises
    ------
    InvalidProcessingStateError
        If no worker attach seam is exposed by the provider.

    """
    from faninsar.processing.geometry.prepared_provider import (
        LocalPreparedGeometryProvider,
    )

    factory = getattr(LocalPreparedGeometryProvider, "from_worker_generation", None)
    if callable(factory):
        if resource_limits is None:
            reject_invalid_state(
                "prepared LUT worker attachment requires resource_limits"
            )
        return factory(Path(root), token, resource_limits)

    # This branch is intentionally narrow: it remains useful for a provider
    # implementation that has landed the instance seam but not the factory.
    def callback(**_kwargs: object) -> object:
        """Reject accidental worker-side publication attempts."""
        reject_invalid_state("worker prepared provider cannot publish generations")

    provider = LocalPreparedGeometryProvider(Path(root), callback)
    attach = getattr(provider, "attach_worker_lease", None)
    if not callable(attach):
        reject_invalid_state(
            "prepared provider does not expose a worker generation attach seam"
        )
    if resource_limits is None:
        reject_invalid_state("prepared LUT worker attachment requires resource_limits")
    attach(token, resource_limits)
    return provider


def _own_geo_valid_mask(array: np.ndarray | None) -> np.ndarray | None:
    """Copy a geographic validity map before its backing memmap is released.

    Parameters
    ----------
    array : numpy.ndarray or None
        Validity map that may be a disk-backed ``numpy.memmap``.

    Returns
    -------
    numpy.ndarray or None
        Owning boolean array safe to use after the source mapping is closed.

    """
    if array is None:
        return None
    return np.array(array, dtype=bool, copy=True)


@dataclass
class ProductionPairState:
    """Mutable state for the production pair workflow with stage log."""

    pair_id: str
    primary: ProductionScene
    secondary: ProductionScene
    dem: DEM
    primary_deramped: np.ndarray | None = None
    secondary_deramped: np.ndarray | None = None
    range_shift_px: float | None = None
    azimuth_shift_px: float | None = None
    esd_azimuth_shift_px: float | None = None
    amplitude_residual_rg_px: float | None = None
    secondary_aligned: np.ndarray | None = None
    secondary_aligned_is_flattened: bool = False
    range_offset_flatten_phase: np.ndarray | None = None
    complex_ifg: np.ndarray | None = None
    complex_ifg_flat: np.ndarray | None = None
    topo_phase: np.ndarray | None = None
    coherence: np.ndarray | None = None
    wrapped_phase: np.ndarray | None = None
    unwrapped_phase: np.ndarray | None = None
    connected_components: np.ndarray | None = None
    baseline: BaselineComponents | None = None
    geocoded: dict[str, np.ndarray] | None = None
    geo2rdr_lut: Geo2RdrLUT | None = None
    geo_height_field: np.ndarray | None = None
    geo_grid: GeoGridSpec | None = None
    primary_geocoded_slc: np.ndarray | None = None
    secondary_geocoded_slc: np.ndarray | None = None
    geocoded_slc_valid: np.ndarray | None = None
    zarr_path: Path | None = None
    stac_path: Path | None = None
    log: list[str] = field(default_factory=list)
    stage_timings_s: dict[str, float] = field(default_factory=dict)
    coregistration_timings_s: dict[str, float] = field(default_factory=dict)
    coregistration_grid: CoregistrationGrid = "radar"
    dem_id: str = ""
    coreg_executor: str = "torch"
    coreg_device: str = "auto"
    multilook: tuple[int, int] | None = None
    goldstein_alpha: float = 0.5
    unwrap_method: str = "snaphu"
    geo_grid_meta: dict[str, Any] | None = None
    geo_work_dir: Path | None = None
    record_scientific_lineage: bool = False
    scientific_lineage: list[ScientificTransitionRecord] = field(default_factory=list)
    phase_state: PhaseState | None = None
    memory_watchdog: MemoryWatchdog | None = None
    geo_bbox: tuple[int, int, int, int] | None = None
    radar_roi_origin: tuple[int, int] | None = None
    prepared_geometry_field: PreparedGeometryField | None = None

    def note(self, message: str) -> None:
        """Append a stage log line."""
        logger.info("[%s] %s", self.pair_id, message)
        self.log.append(message)


def load_production_scene(
    path: str | Path,
    *,
    swath: str = "IW1",
    scope: ScopeMode = "burst",
    burst_index: int = 0,
    dem: DEM | None = None,
    orbit_path: str | Path | None = None,
    coregistration_grid: CoregistrationGrid = "radar",
    full_range: bool = False,
    read_samples: bool = True,
    product: S1Product | None = None,
) -> ProductionScene:
    """Load a full burst or full stitched sub-swath for production processing.

    Parameters
    ----------
    path : path
        SAFE ZIP or directory.  Used as the scene identity even when
        ``product`` is supplied.
    product : S1Product, optional
        Already-opened product, including merged one-burst ASF extracts
        and complementary same-date sub-swath canvases.  When omitted,
        ``path`` is opened with :func:`open_safe_product`.
    swath : str, optional
        Sub-swath name.
    scope : {"burst", "swath"}, optional
        ``burst`` reads one full burst; ``swath`` reads and stitches all bursts.
    burst_index : int, optional
        Burst index when ``scope="burst"``.
    dem : DEM, optional
        DEM used only to attach geometry context (not required for load).
    orbit_path : path, optional
        Precise ESA EOF orbit. Annotation orbit vectors are used when omitted.
    coregistration_grid : {"radar", "geo"}, optional
        Select the burst sample layout required by the coregistration grid.
    full_range : bool, optional
        Read the full swath range extent (column 0 through the burst width)
        instead of the valid-sample envelope, matching ISCE2's full-width
        burst layout. Invalid samples are still zeroed via the valid mask.
    read_samples : bool, optional
        Read the burst pixels. ``False`` loads only placement, carrier, and
        geometry (burst scope only).

    Returns
    -------
    ProductionScene
        Full complex array with carrier and geometry model.

    """
    _ = dem
    product = open_safe_product(path) if product is None else product
    s1_swath = product.swath(swath)
    if orbit_path is not None:
        s1_swath = replace(s1_swath, orbit=read_eof_orbit(orbit_path))
    if scope == "burst":
        array = read_full_burst(
            s1_swath,
            burst_index=burst_index,
            geocoding_layout=coregistration_grid == "geo",
            full_range=full_range,
            read_samples=read_samples,
        )
        burst = s1_swath.bursts[burst_index]
    elif scope == "swath":
        bursts = read_swath_bursts(s1_swath)
        array = stitch_bursts(bursts, overlap_blend=True)
        burst = s1_swath.bursts[0]
    else:
        reject_invalid_state(f"unknown scope: {scope}")

    carrier = carrier_from_swath(s1_swath, burst, first_range_sample=array.col0)
    geometry = _radar_model(
        s1_swath,
        burst,
        shape=array.samples.shape,
        row0=array.row0,
        col0=array.col0,
    )
    scene_id = _scene_id(Path(path))
    logger.info(
        "Loaded production scene %s scope=%s shape=%s origin=(%s,%s)",
        scene_id,
        scope,
        array.samples.shape,
        array.row0,
        array.col0,
    )
    return ProductionScene(
        scene_id=scene_id,
        path=Path(path),
        product=product,
        swath=s1_swath,
        burst=burst,
        array=array,
        carrier=carrier,
        geometry=geometry,
    )


def stage_deramp(
    state: ProductionPairState,
    *,
    device: str = "auto",
    dask_client: Any | None = None,
    secondary_only: bool = False,
) -> ProductionPairState:
    """Deramp both full scenes with annotation carriers.

    Parameters
    ----------
    state : ProductionPairState
        Pair state with loaded reference and secondary scenes.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Numerical device. CPU and CUDA use the unified Torch carrier; MPS uses
        the logged NumPy CPU fallback.
    dask_client : object or None, optional
        Explicitly trusted Dask client. Ambient clients are ignored.
    secondary_only : bool, optional
        Deramp only the secondary. The Reference was prepared once for the
        whole Stack; its pixels are not read here.

    Returns
    -------
    ProductionPairState
        Updated state with deramped scenes.

    """
    from faninsar.processing.runtime.dask_gpu import should_accelerate

    if secondary_only:
        samples = state.secondary.array.samples
        if not should_accelerate(device, dask_client, kernel="carrier_multiply"):
            deramped = deramp(samples, state.secondary.carrier)
        else:
            from faninsar.processing.runtime.dask_gpu import run_carrier_multiply

            deramped = run_carrier_multiply(
                samples,
                state.secondary.carrier,
                sign=-1.0,
                device=device,
                client=dask_client,
            )
        state.secondary_deramped = np.where(
            state.secondary.array.valid_mask, deramped, 0
        )
        state.note(f"DERAMP secondary shape={state.secondary_deramped.shape}")
        return state

    primary_input = state.primary.array.samples
    secondary_input = state.secondary.array.samples
    if not should_accelerate(device, dask_client, kernel="carrier_multiply"):
        state.primary_deramped = deramp(
            state.primary.array.samples, state.primary.carrier
        )
        state.secondary_deramped = deramp(
            state.secondary.array.samples, state.secondary.carrier
        )
    else:
        from faninsar.processing.runtime.dask_gpu import run_carrier_multiply

        state.primary_deramped = run_carrier_multiply(
            state.primary.array.samples,
            state.primary.carrier,
            sign=-1.0,
            device=device,
            client=dask_client,
        )
        state.secondary_deramped = run_carrier_multiply(
            state.secondary.array.samples,
            state.secondary.carrier,
            sign=-1.0,
            device=device,
            client=dask_client,
        )
    # mask invalid
    state.primary_deramped = np.where(
        state.primary.array.valid_mask,
        state.primary_deramped,
        0,
    )
    state.secondary_deramped = np.where(
        state.secondary.array.valid_mask,
        state.secondary_deramped,
        0,
    )
    state.note(
        f"DERAMP shape={state.primary_deramped.shape} "
        f"mean|z|={float(np.mean(np.abs(state.primary_deramped))):.3f}"
    )
    _record_scientific_transition(
        state,
        "deramp",
        (primary_input, secondary_input),
        (state.primary_deramped, state.secondary_deramped),
        carrier="tops",
    )
    return state


def _apply_geo_topographic_phase_chunked(
    state: ProductionPairState,
    lut: Geo2RdrLUT,
    offsets: OffsetFieldResult,
    reference_geo: np.memmap | None,
    secondary_geo: np.memmap,
    valid: np.memmap,
    *,
    grid: GeoGridSpec,
    output_dir: Path,
    chunk_size: int,
    watchdog: MemoryWatchdog | None,
    row0_offset: int = 0,
    col0_offset: int = 0,
    timings_out: dict[str, float] | None = None,
    device: str = "cpu",
) -> tuple[np.memmap, np.ndarray]:
    """Apply geometric phase to disk-backed geographic SLC row tiles.

    Parameters
    ----------
    state : ProductionPairState
        Pair geometry and DEM state.
    lut : Geo2RdrLUT
        Geographic-to-reference-radar coordinates.
    offsets : OffsetFieldResult
        Dense reference-to-secondary radar offsets.
    reference_geo, secondary_geo, valid : numpy.memmap
        Disk-backed geographic SLCs and validity mask, updated in place.
    grid : GeoGridSpec
        Geographic grid defining the shared ground targets.
    output_dir : pathlib.Path
        Directory for disk-backed phase and height fields.
    chunk_size : int
        Number of geographic rows per tile.
    watchdog : MemoryWatchdog, optional
        Memory guard sampled after every completed tile.
    row0_offset, col0_offset : int
        Offset of the cropped LUT inside the full geographic grid, used to
        map cropped rows/cols back to global grid coordinates.
    timings_out : dict, optional
        Exclusive topographic substages (``compose``, ``phase``, ``apply``).
    device : str, optional
        Admitted compose device. CUDA uses Torch ``grid_sample``; CPU uses SciPy.

    Returns
    -------
    topo_phase, height : tuple[numpy.memmap, numpy.ndarray]
        Complete disk-backed geographic fields.

    """
    from faninsar.processing.geocoding.coordinates import RadarCoordinates
    from faninsar.processing.geocoding.geo_lut import grid_lonlat_rows

    topo_phase = np.memmap(
        output_dir / "topographic_phase.float32",
        mode="w+",
        dtype=np.float32,
        shape=lut.shape,
    )
    height_field = lut.height_full
    if height_field is None:
        height_field = np.memmap(
            output_dir / "height.float64",
            mode="w+",
            dtype=np.float64,
            shape=lut.shape,
        )
    invalid = np.complex64(np.nan + 1j * np.nan)
    frame_coordinates = RadarCoordinates(lut.az_full, lut.rg_full, lut.valid)
    for row_start in range(0, lut.shape[0], chunk_size):
        row_stop = min(row_start + chunk_size, lut.shape[0])
        rows = slice(row_start, row_stop)
        chunk_coordinates = RadarCoordinates(
            frame_coordinates.az[rows],
            frame_coordinates.rg[rows],
            frame_coordinates.valid[rows],
        )
        compose_started = time.perf_counter()
        secondary_coordinates = chunk_coordinates.minus_offsets(
            offsets,
            device=device,
        )
        secondary_azimuth = secondary_coordinates.az
        coordinate_valid = secondary_coordinates.valid
        latitude, longitude = grid_lonlat_rows(
            grid,
            row_start + row0_offset,
            row_stop + row0_offset,
        )
        col_stop = col0_offset + lut.shape[1]
        if col0_offset > 0 or col_stop < grid.width:
            latitude = latitude[:, col0_offset:col_stop]
            longitude = longitude[:, col0_offset:col_stop]
        if lut.height_full is None:
            height = np.asarray(
                state.dem.sample(latitude, longitude),
                dtype=np.float64,
            )
        else:
            height = np.asarray(lut.height_full[rows], dtype=np.float64)
        if timings_out is not None:
            timings_out["compose"] = timings_out.get("compose", 0.0) + (
                time.perf_counter() - compose_started
            )
        phase_started = time.perf_counter()
        phase = compute_geometric_phase_from_geo(
            state.primary.geometry,
            state.secondary.geometry,
            latitude,
            longitude,
            height,
            chunk_coordinates.az,
            secondary_azimuth,
            reference_range_index=chunk_coordinates.rg,
        )
        if timings_out is not None:
            timings_out["phase"] = timings_out.get("phase", 0.0) + (
                time.perf_counter() - phase_started
            )
        apply_started = time.perf_counter()
        tile_valid = valid[rows] & coordinate_valid & np.isfinite(phase)
        corrected_secondary = secondary_geo[rows] * np.exp(1j * phase).astype(
            np.complex64
        )
        # The reference keeps its own support: one Reference payload is shared
        # by every secondary and must not inherit one pair's coverage.
        secondary_geo[rows] = np.where(
            tile_valid,
            corrected_secondary,
            invalid,
        ).astype(np.complex64)
        valid[rows] = tile_valid
        topo_phase[rows] = phase.astype(np.float32)
        if lut.height_full is None:
            height_field[rows] = height
        del (
            chunk_coordinates,
            corrected_secondary,
            coordinate_valid,
            height,
            latitude,
            longitude,
            phase,
            secondary_azimuth,
            tile_valid,
        )
        for array in (
            reference_geo,
            secondary_geo,
            valid,
            topo_phase,
        ):
            if array is not None:
                release_memmap_pages(array)
        for array in (height_field, lut.az_full, lut.rg_full, lut.valid):
            if isinstance(array, np.memmap):
                release_memmap_pages(array)
        if watchdog is not None:
            watchdog.sample(f"geo_topographic_phase:{row_start}:{row_stop}")
        if timings_out is not None:
            timings_out["apply"] = timings_out.get("apply", 0.0) + (
                time.perf_counter() - apply_started
            )
    if reference_geo is not None:
        reference_geo.flush()
    secondary_geo.flush()
    valid.flush()
    topo_phase.flush()
    if isinstance(height_field, np.memmap):
        height_field.flush()
    return topo_phase, height_field


def build_reference_geo_lut(
    primary: ProductionScene,
    *,
    radar_shape: tuple[int, int],
    dem: DEM,
    geo_grid: GeoGridSpec,
    storage_dir: str | Path,
    geo_lut_cache_dir: str | Path | None = None,
    roi: BoundingBox | Polygons | None = None,
    roi_buffer_m: float = 320.0,
    geo_footprint_mask_enabled: bool = True,
    geo_height_m: float = 0.0,
    geo_chunk_size: int = 128,
    device: str = "auto",
    timings: dict[str, float] | None = None,
) -> Geo2RdrLUT:
    """Build the geo2rdr LUT of one Reference burst on the Stack geo grid.

    The LUT depends only on the Reference burst geometry, the DEM, the grid,
    and the ROI, so a Stack builds it once per Reference burst and every
    secondary date reuses it.

    Parameters
    ----------
    primary : ProductionScene
        Reference burst (pixels are not needed).
    radar_shape : tuple[int, int]
        Native Reference burst shape.
    dem : DEM
        Height sampler.
    geo_grid : GeoGridSpec
        Stack geographic grid.
    storage_dir : path-like
        Directory for the LUT memmaps when not served from the cache.
    geo_lut_cache_dir : path-like, optional
        Content-addressed LUT cache root.
    roi : BoundingBox or Polygons, optional
        Restricts the LUT window to the buffered ROI.
    roi_buffer_m : float, optional
        ROI buffer in metres.
    geo_footprint_mask_enabled : bool, optional
        Prefilter the LUT with the burst footprint hull (no ROI).
    geo_height_m : float, optional
        Fallback ellipsoidal height.
    geo_chunk_size : int, optional
        Destination rows per geometry call.
    device : str, optional
        Numerical device.
    timings : dict, optional
        Receives substage timings.

    Returns
    -------
    Geo2RdrLUT
        LUT whose ``row0``/``col0`` locate it on ``geo_grid``.

    """
    from faninsar.processing.geocoding.geo_lut import (
        build_geo2rdr_lut,
        burst_geo_footprint_lonlat,
        burst_geo_quad_lonlat,
        derive_burst_geo_bbox,
        roi_geo_bbox,
    )

    timings = {} if timings is None else timings
    bbox_started = _clock_start(device)
    burst_row0, burst_row1, burst_col0, burst_col1 = derive_burst_geo_bbox(
        geometry=primary.geometry,
        radar_shape=radar_shape,
        dem=dem,
        grid=geo_grid,
        device=device,
    )
    timings["geo_bbox"] = _clock_stop(device, bbox_started)
    footprint_lonlat = None
    roi_geometry = None
    if geo_footprint_mask_enabled and roi is None:
        footprint_started = _clock_start(device)
        footprint_lonlat = burst_geo_footprint_lonlat(
            geometry=primary.geometry,
            radar_shape=radar_shape,
            dem=dem,
            grid=geo_grid,
            device=device,
        )
        timings["geo_footprint"] = _clock_stop(device, footprint_started)
        if footprint_lonlat is None:
            logger.warning(
                "burst footprint polygon could not be derived; "
                "continuing without geo footprint mask"
            )
    if roi is not None:
        from shapely.geometry import MultiPolygon
        from shapely.geometry import Polygon as ShapelyPolygon

        from faninsar.processing.geocoding.geo_lut import polygon_parts

        burst_quad = burst_geo_quad_lonlat(
            geometry=primary.geometry,
            radar_shape=radar_shape,
            dem=dem,
            device=device,
        )
        if burst_quad is not None:
            intersection = _roi_geometry(roi).intersection(
                ShapelyPolygon(burst_quad).buffer(0)
            )
            parts = [part for part in polygon_parts(intersection) if not part.is_empty]
            if parts:
                roi_polygon = parts[0] if len(parts) == 1 else MultiPolygon(parts)
                roi_geometry = _buffer_geometry_meters(
                    roi_polygon,
                    geo_grid.crs,
                    roi_buffer_m,
                )
                burst_row0, burst_row1, burst_col0, burst_col1 = roi_geo_bbox(
                    roi_geometry,
                    geo_grid,
                    margin_px=2,
                )
    lut_cache_dir = Path(geo_lut_cache_dir) if geo_lut_cache_dir is not None else None
    lut_cache_key = None
    if lut_cache_dir is not None:
        primary_swath = str(getattr(primary.swath, "swath", ""))
        reference_burst = int(getattr(getattr(primary, "burst", None), "index", 0))
        # Readable namespace only: the on-disk identity is derived from
        # geometry/orbit/DEM/grid/solver content inside build_geo2rdr_lut.
        lut_cache_key = f"{primary.scene_id}_{primary_swath}_b{reference_burst}"
    substage_started = _clock_start(device)
    lut_timings: dict[str, float] = {}
    lut = build_geo2rdr_lut(
        geometry=primary.geometry,
        grid=geo_grid,
        full_radar_shape=radar_shape,
        height_m=geo_height_m,
        dem=dem,
        device=device,
        chunk_size=geo_chunk_size,
        storage_dir=Path(storage_dir),
        cache_dir=lut_cache_dir,
        cache_key=lut_cache_key,
        row_range=(burst_row0, burst_row1),
        col_range=(burst_col0, burst_col1),
        footprint_lonlat=footprint_lonlat,
        roi_geometry=roi_geometry,
        polygon_dilate_px=2,
        footprint_dilate_px=64,
        timings_out=lut_timings,
    )
    timings["geo2rdr_lut"] = _clock_stop(device, substage_started)
    for key, value in lut_timings.items():
        timings[f"geo2rdr_lut.{key}"] = float(value)
    return lut


def stage_coregister(
    state: ProductionPairState,
    *,
    control_spacing: int | None = None,
    esd_enabled: bool = False,
    amplitude_refinement_enabled: bool = False,
    misreg_az_px: float = 0.0,
    misreg_rg_px: float = 0.0,
    force_amplitude_residual_rg: float | None = None,
    force_esd_azimuth_shift_px: float | None = None,
    residuals_only: bool = False,
    executor: str = "torch",
    device: str = "auto",
    dask_client: Any | None = None,
    coregistration_grid: CoregistrationGrid = "radar",
    geo_grid: GeoGridSpec | None = None,
    geo_height_m: float = 0.0,
    geo_chunk_size: int = 128,
    geo_work_dir: str | Path | None = None,
    geo_lut_cache_dir: str | Path | None = None,
    geo_footprint_mask_enabled: bool = True,
    roi: BoundingBox | Polygons | None = None,
    roi_buffer_m: float = 320.0,
    roi_window: tuple[int, int, int, int] | None = None,
    memory_watchdog: MemoryWatchdog | None = None,
    prepared_geometry_field: PreparedGeometryField | None = None,
    prepared_geo_lut: Geo2RdrLUT | None = None,
    secondary_only: bool = False,
    reference_shape: tuple[int, int] | None = None,
    reference_geo_valid: np.ndarray | None = None,
) -> ProductionPairState:
    """Estimate dense offsets and coregister on a radar or geographic grid.

    Memory notes
    ------------
    Full-burst Lanczos resampling is chunked inside :func:`resample_complex`.
    ESD uses a cheaper bilinear pre-align (``order=1``) so the expensive
    phase-preserving Lanczos path runs only once for the final secondary.
    Offset fields and the secondary deramped buffer are released after the
    final resample so peak RSS does not stack geometry + pre + final copies.

    Parameters
    ----------
    state : ProductionPairState
        Mutable pair workflow state (requires deramp outputs).
    control_spacing : int, optional
        Stride between geometry control points on the **radar** deramped grid
        (PROPOSAL-0017: residual / dense geometry is always measured in radar,
        independent of ``coregistration_grid``). Default **8** for both radar
        and geo product modes so the offset field density matches.
    esd_enabled : bool, optional
        Run ESD azimuth residual estimation. Default False.
    amplitude_refinement_enabled : bool, optional
        Refine geometry offsets with multi-window magnitude Ampcor (ISCE2
        topsApp ``runRangeCoreg`` style). On TOPS, only the **range** residual
        is applied; azimuth residual comes from ESD (``runESD``), matching
        ISCE2. Disabled by default; pair/network modes enable it.
    force_amplitude_residual_rg : float or None, optional
        When set, skip Ampcor measurement and apply this range residual
        (ISCE2-style common residual shared across all bursts of a pair).
    force_esd_azimuth_shift_px : float or None, optional
        When set, skip ESD measurement and apply this azimuth residual
        (common residual for multi-burst pairs).
    residuals_only : bool, optional
        When True, measure Ampcor/ESD residuals and return without final
        SLC resampling (used for multi-burst residual consensus).
    misreg_az_px, misreg_rg_px : float, optional
        External azimuth and range residual corrections applied to the
        prepared offset field.
    executor : {"numpy", "torch"}, optional
        Ampcor executor. ``"numpy"`` with ``device="cpu"`` or
        ``device="auto"`` selects the portable Ampcor path; the final
        phase-preserving resample remains Torch-owned.
    device : {"auto","cpu","cuda","mps"}, optional
        Numerical device. ``"auto"`` selects local CUDA or CPU; explicit MPS
        uses the stage's logged NumPy CPU fallback where this proposal applies.
    dask_client : object or None, optional
        Explicitly trusted Dask client used for distributed Torch tasks.
    coregistration_grid : {"radar", "geo"}, optional
        Grid on which both aligned SLCs are produced.
    geo_grid : GeoGridSpec, optional
        Native-resolution SLC grid required for geographic coregistration.
    geo_height_m : float, optional
        Fallback ellipsoidal height for geographic coregistration.
    geo_chunk_size : int, optional
        Geo2rdr row chunk size.
    geo_work_dir : str or pathlib.Path, optional
        Directory for disk-backed Geo intermediate arrays.
    geo_lut_cache_dir : str or pathlib.Path, optional
        Shared cache for reference-scene geo2rdr LUTs. Entries are
        content-addressed inside :func:`build_geo2rdr_lut`; the per-scene
        key is only a readable namespace, so every date pair solving the
        same reference geometry, DEM, and grid reuses one entry.
    geo_footprint_mask_enabled : bool, optional
        When True (default), crop the geo LUT with the eight-point burst
        hull and skip destination pixels outside the dilated footprint.
    roi : BoundingBox or Polygons, optional
        Restricts geo processing to the ROI-burst quad intersection
        (buffered by ``roi_buffer_m``) instead of the full burst bbox.
    roi_buffer_m : float, optional
        Physical margin in meters kept around the ROI in geo mode, applied in
        the grid CRS so the extension is isotropic on the ground.
    roi_window : tuple[int, int, int, int], optional
        Radar window ``(row0, row1, col0, col1)`` to coregister instead of the
        full burst (radar mode with an ROI).
    memory_watchdog : MemoryWatchdog, optional
        Memory guard sampled after each geographic tile.
    prepared_geometry_field : PreparedGeometryField, optional
        A field captured by a preceding residual-measurement pass.  When
        supplied, the exact field/crop is reused and no geometry solve or
        local hole-fill is performed.  Ampcor/ESD must be supplied through
        the existing ``force_*`` arguments for exactly-once residual use.
    prepared_geo_lut : Geo2RdrLUT, optional
        A provider-owned, read-only LUT captured for the exact geographic
        grid/view.  When supplied in geo mode, LUT construction and all
        footprint/ROI recomputation are skipped; its explicit crop origin is
        used for every downstream remap and topographic-phase tile.
    secondary_only : bool, optional
        Resample only the secondary. The Reference contributes its geometry,
        carrier, and shape; its pixels are never read or deramped here
        (PROPOSAL-0059). ESD and amplitude refinement need Reference pixels
        and are rejected in this mode.
    reference_shape : tuple[int, int], optional
        Full deramped Reference burst shape; required with ``secondary_only``.
    reference_geo_valid : numpy.ndarray, optional
        Persisted Reference support on the geographic LUT rows; required with
        ``secondary_only`` in geo mode.

    """
    if secondary_only:
        if state.secondary_deramped is None:
            reject_invalid_state("coregister requires a deramped secondary")
        if reference_shape is None:
            reject_invalid_state(
                "secondary-only coregistration requires reference_shape"
            )
        if esd_enabled or amplitude_refinement_enabled:
            reject_invalid_state(
                "secondary-only coregistration cannot run ESD or amplitude "
                "refinement; they read Reference pixels"
            )
        if coregistration_grid == "geo" and (
            prepared_geo_lut is None or reference_geo_valid is None
        ):
            reject_invalid_state(
                "secondary-only geo coregistration requires the Reference LUT "
                "and support"
            )
        ref = np.broadcast_to(np.complex64(0), tuple(reference_shape))
    elif state.primary_deramped is None or state.secondary_deramped is None:
        reject_invalid_state("coregister requires deramp")
    else:
        ref = state.primary_deramped

    # The reference grid is never cropped: one Reference payload is shared by
    # every secondary of a Stack. A secondary of another shape is conformed to
    # it; padded samples are zero, the deramp stage's invalid-sample value.
    sec = state.secondary_deramped
    if ref.shape != sec.shape:
        sec = _conform_to_shape(sec, ref.shape)
        state.secondary_deramped = sec
        state.note(f"COREG conformed secondary to reference shape {ref.shape}")

    if prepared_geometry_field is not None:
        if coregistration_grid != "radar":
            reject_invalid_state(
                "prepared geometry reuse requires a provider-bound Geo LUT/view "
                "for non-radar products"
            )
        if prepared_geometry_field.source_shape != ref.shape:
            reject_invalid_state(
                "prepared geometry source shape does not match the product pair"
            )
        if amplitude_refinement_enabled and force_amplitude_residual_rg is None:
            reject_invalid_state(
                "prepared geometry reuse requires a forced Ampcor range residual"
            )
        if esd_enabled and force_esd_azimuth_shift_px is None:
            reject_invalid_state(
                "prepared geometry reuse requires a forced ESD azimuth residual"
            )
    if prepared_geo_lut is not None and coregistration_grid != "geo":
        reject_invalid_state("prepared Geo LUT reuse requires geo coregistration")

    resolved_ampcor_executor, resolved_ampcor_device = resolve_ampcor_policy(
        executor, device
    )
    # Ampcor's qualified CPU fallback for ``auto`` must not disable the
    # phase-preserving Torch resampler's own automatic device selection.
    resolved_torch_device = (
        "auto" if device.strip().lower() == "auto" else resolved_ampcor_device
    )

    dem = state.dem
    # Dense geometry + Ampcor/ESD always run on radar deramped samples. Product
    # grid (radar vs geo) only changes how those offsets are *applied*. Keep the
    # same default stride for both so geo does not silently use an 8x coarser
    # offset field than radar.
    resolved_control_spacing = 8 if control_spacing is None else control_spacing
    if resolved_control_spacing < 1:
        reject_invalid_state("control_spacing must be >= 1")
    window: tuple[int, int, int, int] | None = (
        roi_window if coregistration_grid == "radar" else None
    )
    window_origin: tuple[int, int] | None = None
    window_halo_px = 0
    if window is not None:
        wr0, wr1, wc0, wc1 = window
        if not (0 <= wr0 < wr1 <= ref.shape[0] and 0 <= wc0 < wc1 <= ref.shape[1]):
            reject_invalid_state("roi_window must lie inside the burst")
        probe_extent = geometry_offset_window_extent(
            window,
            burst_shape=ref.shape,
            reference_model=state.primary.geometry,
            secondary_model=state.secondary.geometry,
            device=device,
            dem=dem,
            probe_stride=max(32, 8 * resolved_control_spacing),
        )
        window_halo_px = max(
            _WINDOW_HALO_FLOOR_PX,
            int(np.ceil(probe_extent)) + _LANCZOS_RADIUS_PX + _WINDOW_HALO_MARGIN_PX,
        )
    # The carrier centre row is defined by the native Reference burst height.
    reference_native_height = int(
        reference_shape[0] if secondary_only else state.primary.array.samples.shape[0]
    )
    ref_full = ref
    sec_full = sec
    growth_attempts = 0
    if prepared_geometry_field is not None:
        cr0, cr1, cc0, cc1 = prepared_geometry_field.crop_bounds
        ref = ref_full[cr0:cr1, cc0:cc1]
        sec = sec_full[cr0:cr1, cc0:cc1]
        window = None
        window_origin = (cr0, cc0)
        state.primary_deramped = ref
        state.secondary_deramped = sec
        state.radar_roi_origin = window_origin
    while True:
        if window is not None:
            crop_bounds = _window_crop_bounds(
                window,
                ref_full.shape,
                resolved_control_spacing,
                window_halo_px,
            )
            cr0, cr1, cc0, cc1 = crop_bounds
            ref = ref_full[cr0:cr1, cc0:cc1]
            sec = sec_full[cr0:cr1, cc0:cc1]
            window_origin = (cr0, cc0)
            state.primary_deramped = ref
            state.secondary_deramped = sec
            state.radar_roi_origin = window_origin
        substage_started = _clock_start(device)
        if prepared_geometry_field is None:
            geometry_field = dense_geometry_offsets(
                shape=ref.shape,
                reference_model=state.primary.geometry,
                secondary_model=state.secondary.geometry,
                device=device,
                dem=dem,
                stride=resolved_control_spacing,
                row0=0 if window_origin is None else window_origin[0],
                col0=0 if window_origin is None else window_origin[1],
            )
            state.coregistration_timings_s["dense_geometry_offsets"] = _clock_stop(
                device, substage_started
            )
        else:
            geometry_field = prepared_geometry_field.field
            state.coregistration_timings_s["dense_geometry_offsets_reused"] = 0.0

        prior_rg = float(
            np.nanmedian(geometry_field.range_offset_px[geometry_field.coverage])
        )
        prior_az = float(
            np.nanmedian(geometry_field.azimuth_offset_px[geometry_field.coverage])
        )
        amp_res_rg = 0.0
        amp_res_az = 0.0
        amp_az_measured = 0.0
        esd_az = 0.0
        ampcor_started = _clock_start(device)
        # Residual measure is always on radar deramped samples (PROPOSAL-0017);
        # independent of final product grid (radar vs geo).
        # Multi-burst pairs pass force_* so every burst shares one residual
        # (ISCE2 secondaryRangeCorrection / secondaryAzimuthCorrection style);
        # per-burst residuals create constant phase jumps at burst seams.
        if force_amplitude_residual_rg is not None:
            amp_res_rg = float(force_amplitude_residual_rg)
            amp_res_az = 0.0
            state.note(
                f"Ampcor residual rg={amp_res_rg:.4f} px (forced common residual)"
            )
        elif (
            amplitude_refinement_enabled
            and np.isfinite(prior_rg)
            and np.isfinite(prior_az)
        ):
            amp_rg, amp_az = refine_shift_with_correlation(
                ref,
                sec,
                prior_rg=prior_rg,
                prior_az=prior_az,
                search_radius=16,
                executor=resolved_ampcor_executor,
                device=resolved_ampcor_device,
            )
            if np.isfinite(amp_rg) and np.isfinite(amp_az):
                amp_res_rg = amp_rg - prior_rg
                amp_az_measured = amp_az - prior_az
                # TOPS (ISCE2 topsApp): Ampcor residual is applied in range only.
                # Azimuth residual is estimated by ESD on burst spectral diversity;
                # applying magnitude-match azimuth on a full burst injects envelope
                # bias (~0.4 px) and N-S residual fringes.
                amp_res_az = 0.0
                state.note(
                    f"Ampcor residual rg={amp_res_rg:.4f} px "
                    f"(measured az={amp_az_measured:.4f} px not applied; ESD owns az)"
                )
            else:
                state.note("amplitude refinement skipped (non-finite result)")
        elif amplitude_refinement_enabled:
            state.note("amplitude refinement skipped (non-finite geometry prior)")
        state.coregistration_timings_s["ampcor"] = _clock_stop(device, ampcor_started)
        esd_started = _clock_start(device)
        if force_esd_azimuth_shift_px is not None:
            esd_az = float(force_esd_azimuth_shift_px)
            state.esd_azimuth_shift_px = esd_az
            state.note(f"ESD az={esd_az:.4f} px (forced common residual)")
        elif esd_enabled and np.isfinite(prior_rg) and np.isfinite(prior_az):
            # Bilinear pre-align is sufficient for ESD spectral estimation and
            # avoids a second full-burst Lanczos pass (~minutes and peak RSS).
            # Pre-align uses geometry + range Ampcor only (azimuth residual is ESD).
            pre = resample_complex(
                sec,
                range_offset_px=geometry_field.range_offset_px + amp_res_rg,
                azimuth_offset_px=geometry_field.azimuth_offset_px + amp_res_az,
                order=1,
                executor="torch",
                device=resolved_torch_device,
            )
            from faninsar.processing.runtime.dask_gpu import should_accelerate

            if should_accelerate(
                resolved_torch_device,
                dask_client,
                kernel="esd_azimuth_shift",
            ):
                from faninsar.processing.runtime.dask_gpu import run_esd_azimuth_shift

                esd = run_esd_azimuth_shift(
                    ref,
                    pre,
                    device=resolved_torch_device,
                    client=dask_client,
                )
            else:
                esd = estimate_azimuth_shift_esd(ref, pre)
            esd_az = float(esd.azimuth_shift_px)
            state.esd_azimuth_shift_px = esd_az
            state.note(f"ESD az={esd_az:.4f} px coherence={esd.coherence:.3f}")
            del pre
            gc.collect()
        state.coregistration_timings_s["esd"] = _clock_stop(device, esd_started)
        state.amplitude_residual_rg_px = float(amp_res_rg)

        offsets = combine_offset_fields(
            geometry_field,
            esd_azimuth_shift_px=esd_az,
            amplitude_residual_rg=amp_res_rg,
            amplitude_residual_az=amp_res_az,
            misreg_az_px=misreg_az_px,
            misreg_rg_px=misreg_rg_px,
        )
        if residuals_only:
            crop_bounds_for_field = (
                (0, ref_full.shape[0], 0, ref_full.shape[1])
                if window_origin is None
                else (
                    window_origin[0],
                    window_origin[0] + ref.shape[0],
                    window_origin[1],
                    window_origin[1] + ref.shape[1],
                )
            )
            state.prepared_geometry_field = PreparedGeometryField(
                field=geometry_field,
                source_shape=ref_full.shape,
                crop_bounds=crop_bounds_for_field,
                control_spacing=resolved_control_spacing,
            )
        # Drop geometry-only fields once combined; offsets retains the dense maps.
        del geometry_field
        gc.collect()
        if window is None:
            break
        if _window_halo_sufficient(
            offsets,
            window,
            window_origin,
            crop_bounds,
            ref_full.shape,
        ):
            break
        growth_attempts += 1
        if growth_attempts >= _MAX_WINDOW_HALO_GROWTH:
            logger.warning(
                "ROI window halo did not converge after %d growths; "
                "falling back to full-burst coregistration",
                growth_attempts,
            )
            window = None
            window_origin = None
            state.radar_roi_origin = None
            ref = ref_full
            sec = sec_full
            state.primary_deramped = ref
            state.secondary_deramped = sec
            continue
        window_halo_px = (
            _window_resample_halo_required(offsets, window, window_origin)
            + _WINDOW_HALO_MARGIN_PX
        )

    state.range_shift_px = float(
        np.nanmedian(offsets.range_offset_px[offsets.coverage])
    )
    state.azimuth_shift_px = float(
        np.nanmedian(offsets.azimuth_offset_px[offsets.coverage])
    )
    coverage = float(np.mean(offsets.coverage))
    if residuals_only:
        # Multi-burst residual consensus pass: keep residual diagnostics only.
        _record_scientific_transition(
            state,
            "measure_residual_solution",
            (ref, sec),
            (offsets.range_offset_px, offsets.azimuth_offset_px),
            ampcor_range_px=state.amplitude_residual_rg_px,
            ampcor_azimuth_px=amp_az_measured,
            esd_azimuth_px=state.esd_azimuth_shift_px,
            misreg_az_px=misreg_az_px,
            misreg_rg_px=misreg_rg_px,
            solution_kind=(
                "network" if force_esd_azimuth_shift_px is not None else "pair"
            ),
        )
        del offsets
        gc.collect()
        state.note(
            f"COREG residuals_only amp_rg={state.amplitude_residual_rg_px} "
            f"esd_az={state.esd_azimuth_shift_px} "
            f"median_rg={state.range_shift_px:.3f} "
            f"median_az={state.azimuth_shift_px:.3f}"
        )
        return state
    if coregistration_grid == "geo":
        if geo_grid is None:
            reject_invalid_state("geo coregistration requires geo_grid")
        from faninsar.processing.geocoding.geo_modes import (
            coregister_geocoded_slcs_chunked,
            geocode_secondary_slc_chunked,
        )

        if geo_work_dir is None:
            reject_invalid_state("geo coregistration requires geo_work_dir")
        work_directory = Path(geo_work_dir)
        work_directory.mkdir(parents=True, exist_ok=True)
        if prepared_geo_lut is not None:
            if prepared_geo_lut.full_radar_shape != ref.shape:
                reject_invalid_state(
                    "prepared Geo LUT radar shape does not match the product pair"
                )
            if (
                prepared_geo_lut.row0 + prepared_geo_lut.shape[0] > geo_grid.height
                or prepared_geo_lut.col0 + prepared_geo_lut.shape[1] > geo_grid.width
            ):
                reject_invalid_state(
                    "prepared Geo LUT crop lies outside the requested geo grid"
                )
            lut = prepared_geo_lut
            state.coregistration_timings_s["geo2rdr_lut_reused"] = 0.0
            state.note("COREG reused prepared Reference Geo LUT")
        else:
            lut = build_reference_geo_lut(
                state.primary,
                radar_shape=ref.shape,
                dem=state.dem,
                geo_grid=geo_grid,
                storage_dir=work_directory / "lut",
                geo_lut_cache_dir=geo_lut_cache_dir,
                roi=roi,
                roi_buffer_m=roi_buffer_m,
                geo_footprint_mask_enabled=geo_footprint_mask_enabled,
                geo_height_m=geo_height_m,
                geo_chunk_size=geo_chunk_size,
                device=device,
                timings=state.coregistration_timings_s,
            )
        burst_row0, burst_col0 = int(lut.row0), int(lut.col0)
        state.geo_bbox = (
            burst_row0,
            burst_row0 + lut.shape[0],
            burst_col0,
            burst_col0 + lut.shape[1],
        )
        substage_started = _clock_start(device)
        if secondary_only:
            secondary_geo, valid = geocode_secondary_slc_chunked(
                sec,
                secondary_carrier=state.secondary.carrier,
                reference_lut=lut,
                reference_valid=reference_geo_valid,
                offsets=offsets,
                output_dir=work_directory / "slc",
                row_chunk=geo_chunk_size,
                device=resolved_torch_device,
                dask_client=dask_client,
                watchdog=memory_watchdog,
            )
            reference_geo = None
        else:
            reference_geo, secondary_geo, valid = coregister_geocoded_slcs_chunked(
                ref,
                sec,
                reference_carrier=state.primary.carrier,
                secondary_carrier=state.secondary.carrier,
                reference_lut=lut,
                offsets=offsets,
                output_dir=work_directory / "slc",
                row_chunk=geo_chunk_size,
                device=resolved_torch_device,
                dask_client=dask_client,
                watchdog=memory_watchdog,
            )
        state.coregistration_timings_s["geo_slc_resample"] = _clock_stop(
            device, substage_started
        )
        substage_started = _clock_start(device)
        topo_timings: dict[str, float] = {}
        topo_phase, height_field = _apply_geo_topographic_phase_chunked(
            state,
            lut,
            offsets,
            reference_geo,
            secondary_geo,
            valid,
            grid=geo_grid,
            output_dir=work_directory,
            chunk_size=geo_chunk_size,
            watchdog=memory_watchdog,
            row0_offset=burst_row0,
            col0_offset=burst_col0,
            timings_out=topo_timings,
            device=str(resolved_torch_device),
        )
        state.coregistration_timings_s["geo_topographic_phase"] = _clock_stop(
            device, substage_started
        )
        for key, value in topo_timings.items():
            state.coregistration_timings_s[f"geo_topographic_phase.{key}"] = float(
                value
            )
        state.primary_geocoded_slc = reference_geo
        state.secondary_geocoded_slc = secondary_geo
        state.geocoded_slc_valid = valid
        geo_input_reference = ref
        geo_input_secondary = sec
        state.primary_deramped = reference_geo
        state.secondary_deramped = None
        state.secondary_aligned = secondary_geo
        state.geo2rdr_lut = lut
        state.geo_grid = geo_grid
        state.topo_phase = topo_phase
        state.geo_height_field = height_field
        state.geo_work_dir = work_directory
        state.memory_watchdog = memory_watchdog
        geo_offset_digest: str | None = None
        geo_lut_digest: str | None = None
        geo_topographic_phase_digest: str | None = None
        if state.record_scientific_lineage:
            geo_offset_digest = _lineage_payload_digest(
                offsets.range_offset_px,
                offsets.azimuth_offset_px,
            )
            geo_lut_digest = _lineage_payload_digest(
                lut.az_full,
                lut.rg_full,
                lut.valid,
            )
            geo_topographic_phase_digest = _lineage_payload_digest(topo_phase)
        del ref, sec, offsets
        gc.collect()
        state.note(
            f"COREG geo grid={geo_grid.shape} median_rg={state.range_shift_px:.3f} "
            f"median_az={state.azimuth_shift_px:.3f} "
            f"coverage={float(valid.mean()):.3f} "
            "(two deramped single-remaps + fractional reramp + SLC flatten)"
        )
        state.note(f"COREG timings={state.coregistration_timings_s}")
        _record_scientific_transition(
            state,
            "apply_geo_carrier_residual_geometric_phase",
            (geo_input_reference, geo_input_secondary),
            (state.primary_geocoded_slc, state.secondary_aligned),
            amplitude_residual_rg_px=state.amplitude_residual_rg_px,
            esd_azimuth_shift_px=state.esd_azimuth_shift_px,
            misreg_az_px=misreg_az_px,
            misreg_rg_px=misreg_rg_px,
            offset_field_digest=geo_offset_digest,
            lut_digest=geo_lut_digest,
            topographic_phase_digest=geo_topographic_phase_digest,
            carrier_model="tops",
            solution_kind="network" if (misreg_az_px or misreg_rg_px) else "pair",
            geometric_phase="topographic",
        )
        return state

    substage_started = _clock_start(device)
    secondary_input = state.secondary_deramped
    sec_resamp = resample_complex_deramped_reramp(
        sec,
        secondary_carrier=state.secondary.carrier,
        range_offset_px=offsets.range_offset_px,
        azimuth_offset_px=offsets.azimuth_offset_px,
        executor="torch",
        device=resolved_torch_device,
        row0=0 if window_origin is None else window_origin[0],
        col0=0 if window_origin is None else window_origin[1],
        native_height=reference_native_height,
    )
    state.coregistration_timings_s["radar_resample"] = _clock_stop(
        device, substage_started
    )
    secondary_geometry = state.secondary.geometry
    phase_per_range_pixel = (
        4.0
        * np.pi
        * secondary_geometry.range_spacing_m
        / secondary_geometry.wavelength_m
    )
    range_carrier_phase = phase_per_range_pixel * offsets.range_offset_px
    flatten_started = _clock_start(device)
    sec_resamp = _flatten_complex_ifg(
        sec_resamp,
        range_carrier_phase,
        resolved_torch_device,
        dask_client,
    ).astype(np.complex64, copy=False)
    state.coregistration_timings_s["radar_range_flatten"] = _clock_stop(
        device, flatten_started
    )
    state.coregistration_timings_s["radar_resample_and_flatten"] = _clock_stop(
        device, substage_started
    )
    state.secondary_aligned_is_flattened = True
    state.range_offset_flatten_phase = (
        phase_per_range_pixel * offsets.range_offset_px
    ).astype(np.float32)
    state.secondary_deramped = None
    from faninsar.processing.runtime.dask_gpu import (
        run_carrier_multiply,
        should_accelerate,
    )

    reramp_started = _clock_start(device)
    if secondary_only:
        state.primary_deramped = None
    elif should_accelerate(
        resolved_torch_device,
        dask_client,
        kernel="carrier_multiply",
    ):
        state.primary_deramped = run_carrier_multiply(
            ref,
            state.primary.carrier,
            sign=1.0,
            row0=0 if window_origin is None else window_origin[0],
            col0=0 if window_origin is None else window_origin[1],
            native_height=reference_native_height,
            device=resolved_torch_device,
            client=dask_client,
        )
    else:
        state.primary_deramped = reramp(
            ref,
            state.primary.carrier,
            row0=0 if window_origin is None else window_origin[0],
            col0=0 if window_origin is None else window_origin[1],
            native_height=reference_native_height,
        )
    state.coregistration_timings_s["radar_reramp"] = _clock_stop(device, reramp_started)
    state.secondary_aligned = sec_resamp
    radar_offset_digest: str | None = None
    radar_range_phase_digest: str | None = None
    if state.record_scientific_lineage:
        radar_offset_digest = _lineage_payload_digest(
            offsets.range_offset_px,
            offsets.azimuth_offset_px,
        )
        radar_range_phase_digest = _lineage_payload_digest(
            state.range_offset_flatten_phase,
        )
    del sec_resamp, sec, ref, offsets
    gc.collect()
    state.note(
        f"COREG geometry median_rg={state.range_shift_px:.3f} "
        f"median_az={state.azimuth_shift_px:.3f} "
        f"esd={esd_enabled} amplitude_refinement={amplitude_refinement_enabled} "
        f"control_spacing={resolved_control_spacing} "
        f"coverage={coverage:.3f} "
        "(deramped single-remap + fractional reramp + range-offset flatten)"
    )
    state.note(f"COREG timings={state.coregistration_timings_s}")
    _record_scientific_transition(
        state,
        "apply_carrier_residual_range_phase",
        (secondary_input,),
        (state.secondary_aligned,),
        amplitude_residual_rg_px=state.amplitude_residual_rg_px,
        esd_azimuth_shift_px=state.esd_azimuth_shift_px,
        misreg_az_px=misreg_az_px,
        misreg_rg_px=misreg_rg_px,
        offset_field_digest=radar_offset_digest,
        range_phase_screen_digest=radar_range_phase_digest,
        carrier_model="tops",
        solution_kind="network" if (misreg_az_px or misreg_rg_px) else "pair",
        geometric_phase="range_offset_screen",
    )
    return state


def stage_interferogram(
    state: ProductionPairState,
    *,
    multilook: tuple[int, int] = (4, 20),
    goldstein_alpha: float = 0.5,
    dead_pixel_amp_threshold: float = 0.0,
    coherence_window: tuple[int, int] | None = None,
    device: str = "auto",
    dask_client: Any | None = None,
) -> ProductionPairState:
    """Form multilooked interferogram and apply Goldstein filter.

    Parameters
    ----------
    state : ProductionPairState
        Pair state after coregistration.
    multilook : tuple[int, int], optional
        Non-overlapping azimuth and range look factors.
    goldstein_alpha : float, optional
        Goldstein filter exponent.
    dead_pixel_amp_threshold : float, optional
        SLC amplitude below which a pixel is excluded from the multilook
        average (dead-pixel masking). Set to 0 to disable. The production
        Stack provider entry point passes 3.0 for real S1
        data; synthetic tests use the default 0.
    coherence_window : tuple[int, int] or None, optional
        Centered stride-one support for the two-stage coherence estimate
        (sliding-window MLE then per-block averaging). ``None`` keeps the
        direct per-block MLE.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Numerical device. CPU and CUDA use the unified Torch kernels; MPS uses
        the logged NumPy CPU fallback.
    dask_client : object or None, optional
        Explicitly trusted Dask client used for distributed Torch tasks.

    """
    if state.primary_deramped is None or state.secondary_aligned is None:
        reject_invalid_state("interferogram requires coregister")
    state.multilook = multilook
    primary_input = state.primary_deramped
    secondary_input = state.secondary_aligned
    # Geographic coregistration keeps the two aligned SLCs in memmaps that
    # are released below.  Scientific lineage hashes are computed after the
    # interferogram is materialized, so retain owning copies only when the
    # opt-in lineage record is enabled; otherwise preserve the zero-copy
    # production path.
    if state.record_scientific_lineage:
        primary_lineage_input = np.array(primary_input, copy=True)
        secondary_lineage_input = np.array(secondary_input, copy=True)
    else:
        primary_lineage_input = primary_input
        secondary_lineage_input = secondary_input
    from faninsar.processing.runtime.dask_gpu import should_accelerate

    accelerated = should_accelerate(
        device,
        dask_client,
        kernel="multilook_interferogram",
    )
    if not accelerated:
        ifg = form_interferogram(
            state.primary_deramped,
            state.secondary_aligned,
            multilook=multilook,
            dead_pixel_amp_threshold=dead_pixel_amp_threshold,
            coherence_window=coherence_window,
        )
    else:
        from faninsar.processing.runtime.dask_gpu import run_multilook_interferogram

        ifg = run_multilook_interferogram(
            state.primary_deramped,
            state.secondary_aligned,
            multilook=multilook,
            dead_pixel_amp_threshold=dead_pixel_amp_threshold,
            coherence_window=coherence_window,
            device=device,
            client=dask_client,
        )
    state.primary_deramped = None
    state.secondary_aligned = None
    if state.coregistration_grid == "geo" and state.geo_work_dir is not None:
        for array in (
            state.primary_geocoded_slc,
            state.secondary_geocoded_slc,
            state.geocoded_slc_valid,
        ):
            if isinstance(array, np.memmap):
                close_memmap(array)
        state.primary_geocoded_slc = None
        state.secondary_geocoded_slc = None
        state.geocoded_slc_valid = None
        if state.geo2rdr_lut is not None:
            for array in (
                state.geo2rdr_lut.az_full,
                state.geo2rdr_lut.rg_full,
                state.geo2rdr_lut.valid,
            ):
                if isinstance(array, np.memmap):
                    close_memmap(array)
            state.geo2rdr_lut = None
        if isinstance(state.topo_phase, np.memmap):
            close_memmap(state.topo_phase)
            state.topo_phase = None
    gc.collect()
    # alpha=0 is not a no-op in the windowed Goldstein implementation (Hann
    # taper still smooths). Skip the filter entirely so high-rate geometric
    # fringes stay intact for flattening.
    if goldstein_alpha > 0.0:
        if should_accelerate(device, dask_client, kernel="goldstein_filter"):
            from faninsar.processing.runtime.dask_gpu import run_goldstein_filter

            complex_ifg = run_goldstein_filter(
                ifg.complex_ifg,
                alpha=goldstein_alpha,
                window=32,
                device=device,
                client=dask_client,
            )
        else:
            complex_ifg = goldstein_filter(ifg.complex_ifg, alpha=goldstein_alpha)
    else:
        complex_ifg = ifg.complex_ifg
    complex_ifg, coherence, wrapped = mask_invalid_looks(complex_ifg, ifg.coherence)
    state.complex_ifg = complex_ifg
    state.coherence = coherence if coherence is not None else ifg.coherence
    state.wrapped_phase = wrapped
    n_invalid = int(np.isnan(state.wrapped_phase).sum())
    state.note(
        f"IFG multilook={multilook} goldstein={goldstein_alpha} "
        f"mean_coh={float(np.nanmean(state.coherence)):.3f} "
        f"invalid_looks={n_invalid}"
    )
    _record_scientific_transition(
        state,
        "form_interferogram",
        (primary_lineage_input, secondary_lineage_input),
        (state.complex_ifg, state.coherence, state.wrapped_phase),
        multilook=multilook,
        goldstein_alpha=goldstein_alpha,
    )
    return state


def _flatten_complex_ifg(
    complex_ifg: np.ndarray,
    topo_phase: np.ndarray,
    device: str,
    dask_client: Any | None,
) -> np.ndarray:
    """Remove a phase screen with Torch on CPU/CUDA and NumPy on MPS."""
    from faninsar.processing.runtime.dask_gpu import should_accelerate

    if not should_accelerate(
        device,
        dask_client,
        kernel="remove_topographic_phase",
    ):
        return remove_topographic_phase(complex_ifg, topo_phase)
    from faninsar.processing.runtime.dask_gpu import run_remove_topographic_phase

    return run_remove_topographic_phase(
        complex_ifg,
        topo_phase,
        device=device,
        client=dask_client,
    )


def stage_flatten(
    state: ProductionPairState,
    *,
    device: str = "auto",
    dask_client: Any | None = None,
) -> ProductionPairState:
    """Remove topographic phase using DEM + dual-orbit geometry.

    Parameters
    ----------
    state : ProductionPairState
        Pair state after interferogram formation.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Numerical device. CPU and CUDA use the Torch complex multiply; MPS
        uses the logged NumPy CPU fallback.
    dask_client : object or None, optional
        Explicitly trusted Dask client used for distributed Torch tasks.

    Returns
    -------
    ProductionPairState
        Updated state with flattened products.

    When radar coreg already applied a *range-offset* phase screen on the
    secondary SLC (``secondary_aligned_is_flattened=True``), that screen is only
    a first-order approximation of the path-length topographic phase.  Residual
    DEM topography — especially where dense geometry offsets are incomplete
    (DEM edge, outer subswath) — is still removed here with the full dual-orbit
    geometric model.  If the range-offset screen already matched the DEM model,
    the residual is near zero and this step is a no-op in practice.

    """
    if state.complex_ifg is None:
        reject_invalid_state("flatten requires interferogram")
    complex_input = state.complex_ifg
    height, width = state.complex_ifg.shape
    # Multilooked grid → full-res radar indices on the geometry model.
    # Geometry (0,0) is the array origin (already cropped). Use look-window
    # centres rather than leading edges so path-length phase matches the
    # multilooked ifg sampling.
    if state.multilook is not None:
        az_looks, rg_looks = state.multilook
    else:
        full_h, full_w = state.primary.array.samples.shape
        az_looks = max(full_h // max(height, 1), 1)
        rg_looks = max(full_w // max(width, 1), 1)
    az_full = (np.arange(height, dtype=np.float64) + 0.5) * az_looks - 0.5
    rg_full = (np.arange(width, dtype=np.float64) + 0.5) * rg_looks - 0.5
    roi_origin = state.radar_roi_origin or (0, 0)
    az_full = az_full + float(roi_origin[0])
    rg_full = rg_full + float(roi_origin[1])
    az_grid, rg_grid = np.meshgrid(az_full, rg_full, indexing="ij")

    def _topographic_phase() -> tuple[np.ndarray, float, float, float]:
        """Compute the full DEM topo phase and its spread (lazy, costly)."""
        topo = compute_topographic_phase(
            state.primary.geometry,
            state.secondary.geometry,
            az_grid,
            rg_grid,
            state.dem,
            device=device,
        )
        topo_finite = topo[np.isfinite(topo)]
        topo_valid_frac = float(np.isfinite(topo).mean())
        if topo_finite.size:
            rms = float(np.std(topo_finite))
            span = float(np.max(topo_finite) - np.min(topo_finite))
        else:
            rms, span = float("nan"), float("nan")
        return topo, topo_valid_frac, rms, span

    if state.secondary_aligned_is_flattened:
        # ISCE2's burstifg flattens the interferogram with the full-resolution
        # range-offset screen only (fact * range_offset); no second DEM phase
        # removal is applied.  The residual ``topo + range_offset_phase``
        # computed on the multilooked grid is nonzero only because the
        # geometric topo model evaluates the secondary orbit at the reference
        # azimuth, which does not match the resampled source azimuth; removing
        # it corrupts the flattened phase (observed ~0.6 rad on real S1 pairs).
        if state.range_offset_flatten_phase is not None:
            residual_span = 0.0
            screen = state.range_offset_flatten_phase
            screen_std = (
                float(np.std(screen)) if np.isfinite(screen).all() else float("nan")
            )
            state.note(
                f"FLATTEN range-offset screen only (ISCE2 parity); "
                f"screen_std={screen_std:.3f} residual_span={residual_span:.3f} rad"
            )
            flat = state.complex_ifg
            state.topo_phase = np.zeros((height, width), dtype=np.float32)
            # Do NOT remove a residual azimuth ramp here: ISCE2 does not apply
            # one in its flatten step, and the estimated ramp was found to
            # introduce ~2 rad of spurious phase on IW3_b0.
        else:
            topo, topo_valid_frac, rms, span = _topographic_phase()
            # Fallback: no stored range-offset phase, use scale estimation.
            scale, residual_rms = estimate_residual_topographic_scale(
                state.complex_ifg,
                topo,
                coherence=state.coherence,
            )
            if abs(scale) > 1e-3 and topo_valid_frac > 0.05:
                residual_topo = (scale * topo).astype(np.float64)
                flat = _flatten_complex_ifg(
                    state.complex_ifg,
                    residual_topo,
                    device,
                    dask_client,
                )
                state.note(
                    f"FLATTEN residual DEM topo after range-offset "
                    f"scale={scale:.3f} model_rms={rms:.3f} "
                    f"residual_rms={residual_rms:.3f} "
                    f"topo_valid_frac={topo_valid_frac:.3f}"
                )
            else:
                flat = state.complex_ifg
                state.note(
                    f"FLATTEN range-offset only (no residual DEM; "
                    f"scale={scale:.3f} residual_rms={residual_rms:.3f} "
                    f"topo_valid_frac={topo_valid_frac:.3f})"
                )
            state.topo_phase = (
                np.asarray(scale * topo, dtype=np.float32)
                if abs(scale) > 1e-3
                else np.zeros_like(topo, dtype=np.float32)
            )
            if topo_valid_frac > 0.05:
                az_ramp = estimate_residual_azimuth_ramp(
                    flat,
                    topo,
                    coherence=state.coherence,
                    **azimuth_ramp_device_kwargs(device),
                )
                if abs(az_ramp) > 1e-6:
                    flat = remove_azimuth_phase_ramp(flat, az_ramp)
                    state.note(f"FLATTEN residual_az_ramp={az_ramp:.5f} rad/az_sample")
        flat_masked, coh_flat, wrapped_flat = mask_invalid_looks(
            np.asarray(flat, dtype=np.complex64), state.coherence
        )
        state.complex_ifg_flat = flat_masked
        if coh_flat is not None:
            state.coherence = coh_flat
        state.wrapped_phase = wrapped_flat
        _record_scientific_transition(
            state,
            "apply_geometric_phase_flatten",
            (complex_input,),
            (state.complex_ifg_flat, state.wrapped_phase),
            phase_model="range_offset_screen_or_dem_residual",
        )
        return state

    # Residual Doppler / differential TOPS carrier leaves a near-linear
    # azimuth phase ramp on the original-domain ifg. Estimate it against the
    # path-length geometric model and remove it from the unflattened product.
    topo, topo_valid_frac, rms, span = _topographic_phase()
    az_ramp = estimate_residual_azimuth_ramp(
        state.complex_ifg,
        topo,
        coherence=state.coherence,
        **azimuth_ramp_device_kwargs(device),
    )
    if abs(az_ramp) > 1e-6:
        state.complex_ifg = remove_azimuth_phase_ramp(state.complex_ifg, az_ramp)
        state.note(f"FLATTEN residual_az_ramp={az_ramp:.5f} rad/az_sample")

    flat = _flatten_complex_ifg(
        state.complex_ifg,
        topo,
        device,
        dask_client,
    )
    state.topo_phase = topo.astype(np.float32)
    # Keep invalid looks as NaN through ramp/topo multiply (0·e^{iφ}=0 would
    # otherwise repaint a solid phase=0 black edge on the burst margin).
    # Remask the ramp-corrected unflattened ifg for intermediate products.
    state.complex_ifg, state.coherence, _ = mask_invalid_looks(
        state.complex_ifg, state.coherence
    )
    # wrapped_phase must track the flattened product that stage_write archives
    # as complex_ifg (complex_ifg_flat), not the unflattened angle.
    flat_masked, coh_flat, wrapped_flat = mask_invalid_looks(
        flat.astype(np.complex64, copy=False), state.coherence
    )
    state.complex_ifg_flat = flat_masked
    if coh_flat is not None:
        state.coherence = coh_flat
    state.wrapped_phase = wrapped_flat
    state.note(
        f"FLATTEN geometric_phase rms={rms:.3f} rad span={span:.1f} rad "
        f"topo_valid_frac={topo_valid_frac:.3f}"
    )
    _record_scientific_transition(
        state,
        "apply_geometric_phase_flatten",
        (complex_input,),
        (state.complex_ifg_flat, state.wrapped_phase),
        phase_model="topographic",
    )
    return state


def stage_unwrap(
    state: ProductionPairState,
    *,
    method: UnwrapBackend = "snaphu",
    config: SnaphuConfig | None = None,
    irls_kwargs: dict[str, Any] | None = None,
) -> ProductionPairState:
    """Unwrap the flattened interferogram with the selected backend.

    Parameters
    ----------
    state : ProductionPairState
        Pair state containing a flattened interferogram and coherence.
    method : {"irls", "snaphu"}, optional
        Spatial unwrapping backend.
    config : SnaphuConfig, optional
        SNAPHU configuration when ``method="snaphu"``.
    irls_kwargs : dict, optional
        Arguments forwarded to the IRLS backend.

    Returns
    -------
    ProductionPairState
        Updated state containing unwrapped phase and component labels.

    """
    ifg = (
        state.complex_ifg_flat
        if state.complex_ifg_flat is not None
        else state.complex_ifg
    )
    if ifg is None or state.coherence is None:
        reject_invalid_state("unwrap requires interferogram")
    unwrap_input = ifg
    result: SpatialUnwrapResult = unwrap_dispatch(
        ifg,
        state.coherence,
        method=method,
        snaphu_config=config,
        irls_kwargs=irls_kwargs,
    )
    state.unwrapped_phase = result.phase.detach().cpu().numpy()
    state.connected_components = result.component_labels.detach().cpu().numpy()
    state.unwrap_method = method
    state.note(
        f"UNWRAP method={method} converged={result.converged} "
        f"iterations={result.iterations} pcg_iterations={result.pcg_iterations}"
    )

    # Non-DEM residual range/azimuth poly (orbit residual / APS / far-range).
    # Non-DEM residual range/azimuth poly (orbit residual / APS / far-range).
    # Fit on unwrapped phase (stable for multi-fringe), apply to both unwrap
    # and the flattened complex product used by merge.
    from faninsar.processing.interferometry.flatten import (
        apply_residual_phase_screen_to_products,
    )

    flat_src = (
        state.complex_ifg_flat
        if state.complex_ifg_flat is not None
        else state.complex_ifg
    )
    unw_corr, z_corr, _screen, span, applied = apply_residual_phase_screen_to_products(
        unwrapped_phase=state.unwrapped_phase,
        complex_ifg=flat_src,
        coherence=state.coherence,
        range_degree=2,
        azimuth_degree=1,
        min_span_rad=1.0,
        max_span_rad=40.0,
        connected_components=state.connected_components,
    )
    if applied:
        state.unwrapped_phase = np.asarray(unw_corr, dtype=np.float32)
        if z_corr is not None:
            state.complex_ifg_flat = np.asarray(z_corr, dtype=np.complex64)
            state.wrapped_phase = np.angle(state.complex_ifg_flat).astype(np.float32)
        state.note(
            f"UNWRAP residual phase screen span={span:.2f} rad rg_deg=2 az_deg=1"
        )
    elif span >= 1.0:
        state.note(f"UNWRAP residual phase screen skipped span={span:.2f} rad")
    _record_scientific_transition(
        state,
        "spatial_unwrap_and_residual_screen",
        (unwrap_input,),
        (state.unwrapped_phase, state.complex_ifg_flat),
        method=method,
        residual_screen_applied=applied,
    )
    return state


def stage_baseline(state: ProductionPairState) -> ProductionPairState:
    """Compute geometric baseline at burst mid-time for metadata."""
    mid_az = 0.5 * (state.primary.array.samples.shape[0] - 1)
    ref_time = state.primary.geometry.azimuth_time(mid_az)
    sec_time = state.secondary.geometry.azimuth_time(mid_az)
    from faninsar.processing.geometry.orbit import OrbitInterpolator

    ref_interp = OrbitInterpolator.from_orbit(state.primary.swath.orbit)
    sec_interp = OrbitInterpolator.from_orbit(state.secondary.swath.orbit)
    state_ref = ref_interp.evaluate(ref_time)
    state_sec = sec_interp.evaluate(sec_time)
    sat = np.asarray(state_ref.position_m, dtype=np.float64)
    look = -sat / np.linalg.norm(sat)
    baseline_vec = np.asarray(state_sec.position_m, dtype=np.float64) - sat
    parallel = float(np.dot(baseline_vec, look))
    perp_vec = baseline_vec - parallel * look
    state.baseline = BaselineComponents(
        parallel_m=parallel,
        perpendicular_m=float(np.linalg.norm(perp_vec)),
        magnitude_m=float(np.linalg.norm(baseline_vec)),
    )
    state.note(
        f"BASELINE parallel={state.baseline.parallel_m:.1f} m "
        f"perp={state.baseline.perpendicular_m:.1f} m"
    )
    return state


def stage_write(
    state: ProductionPairState,
    output_dir: str | Path,
    *,
    preserve_geo_work_dir: bool = False,
    stac_item_id: str | None = None,
) -> ProductionPairState:
    """Write radar products, geocoded layers, baselines, and STAC with bbox."""
    if (
        state.complex_ifg is None
        or state.coherence is None
        or state.wrapped_phase is None
        or state.unwrapped_phase is None
        or state.connected_components is None
    ):
        reject_invalid_state("write requires completed products")

    try:
        temporal_days = float(
            (
                state.secondary.burst.azimuth_time - state.primary.burst.azimuth_time
            ).total_seconds()
            / 86400.0
        )
    except (TypeError, AttributeError):
        temporal_days = float("nan")
    meta: dict[str, Any] = {
        "unwrap_method": state.unwrap_method,
        "coregistration_grid": state.coregistration_grid,
        "dem_id": state.dem_id,
        "multilook": list(state.multilook or (1, 1)),
        "goldstein_alpha": state.goldstein_alpha,
        "coreg_executor": state.coreg_executor,
        "coreg_device": state.coreg_device,
        "lanczos_a": 4,
        "geo_grid": state.geo_grid_meta,
        "range_shift_px": state.range_shift_px,
        "azimuth_shift_px": state.azimuth_shift_px,
        "esd_azimuth_shift_px": state.esd_azimuth_shift_px,
        "reference_scene": state.primary.scene_id,
        "secondary_scene": state.secondary.scene_id,
        "swath": state.primary.swath.swath,
        "burst_index": state.primary.burst.index,
        "shape": list(state.unwrapped_phase.shape),
        "full_burst_shape": list(state.primary.array.samples.shape),
        "temporal_baseline_days": temporal_days,
        "stages": list(state.log),
        "stage_timings_s": dict(state.stage_timings_s),
        "coregistration_timings_s": dict(state.coregistration_timings_s),
        "scientific_lineage": list(state.scientific_lineage),
    }
    if state.baseline is not None:
        meta["baseline_parallel_m"] = state.baseline.parallel_m
        meta["baseline_perpendicular_m"] = state.baseline.perpendicular_m
        meta["baseline_magnitude_m"] = state.baseline.magnitude_m

    product = PairProductArrays(
        pair_id=state.pair_id,
        complex_ifg=(
            state.complex_ifg_flat
            if state.complex_ifg_flat is not None
            else state.complex_ifg
        ),
        coherence=state.coherence,
        wrapped_phase=state.wrapped_phase,
        unwrapped_phase=state.unwrapped_phase,
        connected_components=state.connected_components,
        metadata=meta,
    )
    out = Path(output_dir)
    zarr_path = write_pair_zarr(product, out / f"{state.pair_id}.zarr")
    reference_slc = state.primary_geocoded_slc
    secondary_slc = state.secondary_geocoded_slc
    slc_valid = state.geocoded_slc_valid
    transform_azimuth = None if state.geo2rdr_lut is None else state.geo2rdr_lut.az_full
    transform_range = None if state.geo2rdr_lut is None else state.geo2rdr_lut.rg_full
    transform_valid = None if state.geo2rdr_lut is None else state.geo2rdr_lut.valid
    topo_phase = state.topo_phase
    reopened_from_work = state.geo_work_dir is not None and state.geo_grid is not None
    if reopened_from_work:
        assert state.geo_work_dir is not None
        assert state.geo_grid is not None
        shape = state.geo_grid.shape
        reference_slc = np.memmap(
            state.geo_work_dir / "slc" / "reference_geo.complex64",
            mode="r",
            dtype=np.complex64,
            shape=shape,
        )
        secondary_slc = np.memmap(
            state.geo_work_dir / "slc" / "secondary_geo.complex64",
            mode="r",
            dtype=np.complex64,
            shape=shape,
        )
        slc_valid = np.memmap(
            state.geo_work_dir / "slc" / "geo_valid.bool",
            mode="r",
            dtype=np.bool_,
            shape=shape,
        )
        transform_azimuth = np.memmap(
            state.geo_work_dir / "lut" / "reference_azimuth.float64",
            mode="r",
            dtype=np.float64,
            shape=shape,
        )
        transform_range = np.memmap(
            state.geo_work_dir / "lut" / "reference_range.float64",
            mode="r",
            dtype=np.float64,
            shape=shape,
        )
        transform_valid = np.memmap(
            state.geo_work_dir / "lut" / "reference_valid.bool",
            mode="r",
            dtype=np.bool_,
            shape=shape,
        )
        topo_phase = np.memmap(
            state.geo_work_dir / "topographic_phase.float32",
            mode="r",
            dtype=np.float32,
            shape=shape,
        )
    if (
        reference_slc is not None
        and secondary_slc is not None
        and state.geo_grid is not None
    ):
        import zarr

        root = zarr.open_group(str(zarr_path), mode="a")
        slc = root.require_group("slc")
        slc.create_array(
            "reference",
            data=reference_slc,
            overwrite=True,
        )
        if reopened_from_work and isinstance(reference_slc, np.memmap):
            close_memmap(reference_slc)
        slc.create_array(
            "secondary",
            data=secondary_slc,
            overwrite=True,
        )
        if reopened_from_work and isinstance(secondary_slc, np.memmap):
            close_memmap(secondary_slc)
        if slc_valid is not None:
            slc.create_array(
                "valid",
                data=slc_valid.astype(np.uint8),
                overwrite=True,
            )
            if reopened_from_work and isinstance(slc_valid, np.memmap):
                close_memmap(slc_valid)
        from affine import Affine

        affine = Affine(*state.geo_grid.transform)
        x0, dx, y0, dy = affine.c, affine.a, affine.f, affine.e
        x = x0 + dx * (0.5 + np.arange(state.geo_grid.width, dtype=np.float64))
        y = y0 + dy * (0.5 + np.arange(state.geo_grid.height, dtype=np.float64))
        slc.create_array("x", data=x, overwrite=True)
        slc.create_array("y", data=y, overwrite=True)
        slc.attrs.update(
            {
                "crs": state.geo_grid.crs,
                "transform": list(state.geo_grid.transform),
                "phase_domain": "reramped_flattened",
            }
        )
        if (
            transform_azimuth is not None
            and transform_range is not None
            and transform_valid is not None
        ):
            transform = root.require_group("transform")
            transform.create_array(
                "reference_azimuth_index",
                data=transform_azimuth,
                overwrite=True,
            )
            if reopened_from_work and isinstance(transform_azimuth, np.memmap):
                close_memmap(transform_azimuth)
            transform.create_array(
                "reference_range_index",
                data=transform_range,
                overwrite=True,
            )
            if reopened_from_work and isinstance(transform_range, np.memmap):
                close_memmap(transform_range)
            transform.create_array(
                "valid",
                data=transform_valid.astype(np.uint8),
                overwrite=True,
            )
            if reopened_from_work and isinstance(transform_valid, np.memmap):
                close_memmap(transform_valid)
    if state.geocoded is not None:
        import zarr

        root = zarr.open_group(str(zarr_path), mode="a")
        geo = root.require_group("geocoded")
        for name, array in state.geocoded.items():
            geo.create_array(name, data=array, overwrite=True)
        if topo_phase is not None:
            root.create_array("topo_phase", data=topo_phase, overwrite=True)
            if reopened_from_work and isinstance(topo_phase, np.memmap):
                close_memmap(topo_phase)

    # STAC with bbox from geocoded lon/lat when available
    stac_path = out / f"{state.pair_id}.json"
    if state.geocoded is not None:
        conv = state.geocoded["converged"].astype(bool)
        if np.any(conv):
            lats = state.geocoded["latitude_deg"][conv]
            lons = state.geocoded["longitude_deg"][conv]
            bbox = [
                float(np.min(lons)),
                float(np.min(lats)),
                float(np.max(lons)),
                float(np.max(lats)),
            ]
            geometry = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]],
                    ]
                ],
            }
            # write custom STAC with geometry
            import json
            from datetime import UTC, datetime

            item = {
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": stac_item_id if stac_item_id is not None else state.pair_id,
                "geometry": geometry,
                "bbox": bbox,
                "properties": {
                    "datetime": datetime.now(UTC).isoformat(),
                    "faninsar:pair_id": state.pair_id,
                    **{
                        f"faninsar:{k}": (
                            v
                            if isinstance(v, (str, int, float, bool, list, type(None)))
                            else str(v)
                        )
                        for k, v in meta.items()
                        if k != "stages"
                    },
                },
                "assets": {
                    "zarr": {
                        "href": str(zarr_path),
                        "type": "application/vnd+zarr",
                        "roles": ["data"],
                    }
                },
                "links": [],
                "stac_extensions": [],
            }
            stac_path.write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
        else:
            write_pair_stac_item(
                product, zarr_path, stac_path, stac_item_id=stac_item_id
            )
    else:
        write_pair_stac_item(product, zarr_path, stac_path, stac_item_id=stac_item_id)

    state.zarr_path = zarr_path
    state.stac_path = stac_path
    if state.memory_watchdog is not None:
        state.memory_watchdog.sample("write:complete")
    if state.geo_work_dir is not None and not preserve_geo_work_dir:
        work_directory = state.geo_work_dir
        state.primary_geocoded_slc = None
        state.secondary_geocoded_slc = None
        state.geocoded_slc_valid = None
        state.geo2rdr_lut = None
        state.geo_height_field = None
        state.topo_phase = None
        state.geo_work_dir = None
        gc.collect()
        shutil.rmtree(work_directory)
    state.note(f"WRITE {zarr_path}")
    return state


def _dem_id(dem: DEM) -> str:
    """Stable DEM identity string for product provenance."""
    name = type(dem).__name__
    path = getattr(dem, "path", None)
    if path is not None:
        return f"{name}:{Path(path)}"
    height = getattr(dem, "height_m", None)
    if height is not None:
        return f"{name}:{float(height)}"
    return name


def _geo_grid_meta(geo_grid: GeoGridSpec | None) -> dict[str, Any] | None:
    """Serialize GeoGridSpec fields for Zarr/STAC metadata."""
    if geo_grid is None:
        return None
    return {
        "crs": geo_grid.crs,
        "transform": list(geo_grid.transform),
        "width": geo_grid.width,
        "height": geo_grid.height,
        "resolution_m": list(geo_grid.resolution_m),
    }


def _multilooked_geo_grid(
    grid: GeoGridSpec,
    multilook: tuple[int, int],
) -> GeoGridSpec:
    """Return the pixel-centre-aligned product grid after block multilooking."""
    from faninsar.processing.mosaicking.grid import GeoGridSpec

    azimuth_looks, range_looks = multilook
    from affine import Affine

    affine = Affine(*grid.transform)
    return GeoGridSpec(
        crs=grid.crs,
        transform=Affine(
            affine.a * range_looks,
            affine.b,
            affine.c,
            affine.d,
            affine.e * azimuth_looks,
            affine.f,
        ),
        width=grid.width // range_looks,
        height=grid.height // azimuth_looks,
        resolution_m=(
            grid.resolution_m[0] * range_looks,
            grid.resolution_m[1] * azimuth_looks,
        ),
    )


def _multilook_real_field(
    field: np.ndarray,
    multilook: tuple[int, int],
) -> np.ndarray:
    """Average a real geographic field over the SLC multilook blocks."""
    azimuth_looks, range_looks = multilook
    height = field.shape[0] // azimuth_looks * azimuth_looks
    width = field.shape[1] // range_looks * range_looks
    return np.nanmean(
        field[:height, :width].reshape(
            height // azimuth_looks,
            azimuth_looks,
            width // range_looks,
            range_looks,
        ),
        axis=(1, 3),
    )


def _finalize_geo_products(
    state: ProductionPairState,
    *,
    geo_grid: GeoGridSpec,
    multilook: tuple[int, int],
    geo_height_m: float,
) -> ProductionPairState:
    """Assemble the multilooked geographic product grid from unwrapped phase."""
    from faninsar.processing.geocoding.geo_lut import grid_lonlat

    assert state.unwrapped_phase is not None
    assert state.coherence is not None
    assert state.wrapped_phase is not None
    product_grid = _multilooked_geo_grid(geo_grid, multilook)
    lat, lon = grid_lonlat(product_grid)
    converged = np.isfinite(state.unwrapped_phase).astype(np.uint8)
    height_field = state.geo_height_field
    if height_field is None:
        height_field = np.full(
            product_grid.shape,
            float(geo_height_m),
            dtype=np.float64,
        )
    elif height_field.shape != product_grid.shape:
        height_field = _multilook_real_field(height_field, multilook)
    state.geocoded = {
        "unwrapped_phase": np.asarray(state.unwrapped_phase, dtype=np.float32),
        "coherence": np.asarray(state.coherence, dtype=np.float32),
        "wrapped_phase": np.asarray(state.wrapped_phase, dtype=np.float32),
        "latitude_deg": lat.astype(np.float64),
        "longitude_deg": lon.astype(np.float64),
        "height_m": np.asarray(height_field, dtype=np.float64),
        "converged": converged,
    }
    if isinstance(state.geo_height_field, np.memmap):
        close_memmap(state.geo_height_field)
        state.geo_height_field = None
    state.geo_grid_meta = _geo_grid_meta(product_grid)
    state.note("GEO products complete on multilooked geographic grid")
    return state


BurstSelection = (
    dict[str, list[int] | range | str] | list[dict[str, list[int] | range | str]]
)


def _as_frame_paths(
    value: str | Path | Sequence[str | Path],
    name: str,
) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    paths = [Path(item) for item in value]
    if not paths:
        reject_invalid_state(name + " must contain at least one SAFE product")
    return paths


def _as_optional_frame_sequence(
    value: str | Path | Sequence[str | Path] | None,
    frame_count: int,
    name: str,
) -> list[Path | None]:
    if value is None:
        return [None] * frame_count
    paths = _as_frame_paths(value, name)
    if len(paths) == 1 and frame_count > 1:
        return [paths[0]] * frame_count
    if len(paths) != frame_count:
        reject_invalid_state(
            name
            + " has "
            + str(len(paths))
            + " entries for "
            + str(frame_count)
            + " frames"
        )
    return list(paths)


def _snapshot_paths(
    paths: Sequence[Path],
    snapshot_root: str | Path,
) -> list[Path]:
    """Copy local input paths into one immutable source snapshot root."""
    from faninsar.processing.runtime.source_snapshots import snapshot_local_source

    root = Path(snapshot_root)
    cache: dict[str, Path] = {}
    snapshots: list[Path] = []
    for path in paths:
        key = str(path.absolute())
        snapshot = cache.get(key)
        if snapshot is None:
            snapshot = snapshot_local_source(path, root).path
            cache[key] = snapshot
        snapshots.append(snapshot)
    return snapshots


def _snapshot_optional_paths(
    paths: Sequence[Path | None],
    snapshot_root: str | Path,
) -> list[Path | None]:
    """Snapshot optional orbit paths while preserving frame cardinality."""
    present = [path for path in paths if path is not None]
    if not present:
        return [None] * len(paths)
    snapshots = iter(_snapshot_paths(present, snapshot_root))
    return [next(snapshots) if path is not None else None for path in paths]


def _burst_index_list(
    value: list[int] | range | str,
    swath: str,
    count: int,
) -> list[int]:
    if isinstance(value, str):
        if value.strip() == "all":
            return list(range(count))
        parts = value.split(":")
        if len(parts) == 2 and all(part.strip().isdigit() for part in parts):
            value = range(int(parts[0]), int(parts[1]))
        else:
            reject_invalid_state(
                "invalid burst selection "
                + repr(value)
                + " for "
                + swath
                + "; expected 'all' or 'start:stop'"
            )
    if isinstance(value, range):
        indices = list(value)
    else:
        indices = [int(index) for index in value]
    if not indices:
        reject_invalid_state("empty burst selection for " + swath)
    for index in indices:
        if index < 0 or index >= count:
            reject_invalid_state(
                "burst index "
                + str(index)
                + " out of range 0.."
                + str(count - 1)
                + " for "
                + swath
            )
    return sorted(set(indices))


def _normalize_burst_selection(
    bursts: BurstSelection | None,
    frame_count: int,
    swaths: tuple[str, ...],
    burst_counts: dict[tuple[int, str], int],
) -> dict[tuple[int, str], list[int]]:
    resolved: dict[tuple[int, str], list[int]] = {}
    if bursts is None:
        for frame_index in range(frame_count):
            for swath in swaths:
                resolved[(frame_index, swath)] = list(
                    range(burst_counts[(frame_index, swath)])
                )
        return resolved
    if isinstance(bursts, list):
        if len(bursts) != frame_count:
            reject_invalid_state(
                "per-frame burst selection has "
                + str(len(bursts))
                + " entries for "
                + str(frame_count)
                + " frames"
            )
        per_frame: list[dict[str, list[int] | range | str] | None] = list(bursts)
    else:
        per_frame = [bursts] * frame_count
    for frame_index, selection in enumerate(per_frame):
        if selection is None:
            for swath in swaths:
                resolved[(frame_index, swath)] = list(
                    range(burst_counts[(frame_index, swath)])
                )
            continue
        for swath in swaths:
            resolved[(frame_index, swath)] = _burst_index_list(
                selection.get(swath, "all"),
                swath,
                burst_counts[(frame_index, swath)],
            )
    return resolved


def _roi_geometry(roi: BoundingBox | Polygons) -> Any:
    from shapely.geometry import box

    if isinstance(roi, BoundingBox):
        return box(roi.left, roi.bottom, roi.right, roi.top)
    series = roi.geometry
    union = series.union_all() if hasattr(series, "union_all") else series.unary_union
    crs = getattr(roi, "crs", None)
    if crs is not None and str(crs) != "EPSG:4326":
        import geopandas as gpd

        return gpd.GeoSeries([union], crs=crs).to_crs("EPSG:4326").iloc[0]
    return union


def _auto_dem_bounds(
    roi: BoundingBox | Polygons | None,
    resolved: dict[tuple[int, str], list[int]],
    primary_products: list,
    orbits: Sequence[str | Path | None] | None = None,
    dem: DEM | None = None,
) -> tuple[float, float, float, float]:
    """Return EPSG:4326 bounds covering the ROI or the selected bursts.

    Parameters
    ----------
    roi : BoundingBox, Polygons, or None
        ROI passed to the Stack provider.
    resolved : dict
        Selected burst indices per (frame_index, swath).
    primary_products : list
        Opened reference SAFE products, one per frame.
    orbits : sequence of path or None, optional
        Per-frame precise orbit files used to build the burst quads.
    dem : DEM, optional
        DEM used by the burst quad rdr2geo.

    Returns
    -------
    tuple[float, float, float, float]
        (min_lon, min_lat, max_lon, max_lat) covering the ROI, or the burst
        quads buffered by DEM_BOUNDS_BUFFER_M.

    """
    if isinstance(roi, BoundingBox):
        return (
            float(roi.left),
            float(roi.bottom),
            float(roi.right),
            float(roi.top),
        )
    if isinstance(roi, Polygons):
        total = roi.to_geodataframe().total_bounds
        return (
            float(total[0]),
            float(total[1]),
            float(total[2]),
            float(total[3]),
        )
    from dataclasses import replace as _replace

    from faninsar.missions.s1 import read_eof_orbit
    from faninsar.processing.geocoding.geo_lut import burst_geo_quad_lonlat

    quads: list[np.ndarray] = []
    for (frame_index, swath), indices in resolved.items():
        s1_swath = primary_products[frame_index].swath(swath)
        orbit_path = None if orbits is None else orbits[frame_index]
        if orbit_path is not None:
            s1_swath = _replace(s1_swath, orbit=read_eof_orbit(orbit_path))
        shape = (s1_swath.lines_per_burst, s1_swath.samples_per_burst)
        for burst_index in indices:
            burst = s1_swath.bursts[burst_index]
            geometry = _radar_model(
                s1_swath,
                burst,
                shape=shape,
                row0=burst.index * s1_swath.lines_per_burst,
                col0=0,
            )
            quad = burst_geo_quad_lonlat(
                geometry=geometry,
                radar_shape=shape,
                dem=dem,
                device="cpu",
            )
            if quad is not None:
                quads.append(quad)
    return _quad_bounds_with_buffer_m(quads, DEM_BOUNDS_BUFFER_M)


def _quad_bounds_with_buffer_m(
    quads: list[np.ndarray],
    buffer_m: float,
) -> tuple[float, float, float, float]:
    """Return the EPSG:4326 bounds of burst quads buffered by meters."""
    from pyproj import Transformer
    from shapely.geometry import Polygon as ShapelyPolygon

    if not quads:
        reject_invalid_state("cannot derive DEM bounds: no burst geometry available")
    all_lon = np.concatenate([quad[:, 0] for quad in quads])
    all_lat = np.concatenate([quad[:, 1] for quad in quads])
    mean_lon = float(np.mean(all_lon))
    mean_lat = float(np.mean(all_lat))
    zone = int(np.floor((mean_lon + 180.0) / 6.0)) + 1
    epsg = (32600 if mean_lat >= 0 else 32700) + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inverse = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    min_lon = min_lat = float("inf")
    max_lon = max_lat = float("-inf")
    for quad in quads:
        xs, ys = transformer.transform(quad[:, 0], quad[:, 1])
        buffered = ShapelyPolygon(np.column_stack([xs, ys])).buffer(buffer_m)
        bx, by = inverse.transform(*buffered.exterior.xy)
        min_lon = min(min_lon, float(np.min(bx)))
        max_lon = max(max_lon, float(np.max(bx)))
        min_lat = min(min_lat, float(np.min(by)))
        max_lat = max(max_lat, float(np.max(by)))
    return (min_lon, min_lat, max_lon, max_lat)


def resolve_auto_dem(
    bounds: tuple[float, float, float, float],
    *,
    output_dir: str | Path,
    geoid_correction: bool = True,
    dem_source: str | None = None,
    output_name: str | None = None,
) -> DEM:
    """Build (or reuse) the automatic DEM raster on one projected grid.

    This is the single datum-aware DEM entry point shared by
    the Stack provider, its sweep helper, and ``faninsar frame``.
    The provider is selected through the public DEM facade and the datum
    conversion is applied at target pixel centres when requested.  The
    target is the same projected UTM/UPS policy used by :class:`Stack`; no
    EPSG:4326 raster is materialized as a relay.

    Parameters
    ----------
    bounds : tuple of float
        (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    output_dir : path
        Directory receiving ``dem/<name>.tif``.
    geoid_correction : bool, optional
        Whether orthometric heights should be converted to ellipsoidal values.
        Defaults to True.
    dem_source : str, optional
        Selection grammar value (``<product>`` / ``<product>:<provider>``);
        ``None`` defers to ``FANINSAR_DEM_SOURCE`` / the ``glo30`` default.
    output_name : str, optional
        Output file name; defaults to ``FANINSAR_DEM_NAME`` or ``dem.tif``.

    Returns
    -------
    DEM
        The ready-to-use DEM sampler for the workflow.

    Raises
    ------
    ValueError
        If bounds or the selected source grammar is invalid.

    """
    west, south, east, north = (float(value) for value in bounds)
    if not (-180.0 <= west <= east <= 180.0 and -90.0 <= south <= north <= 90.0):
        message = f"invalid automatic DEM bounds: {bounds!r}"
        logger.error(message)
        raise ValueError(message)
    name = output_name or os.environ.get("FANINSAR_DEM_NAME", "dem.tif")
    out_path = Path(output_dir) / "dem" / name
    if out_path.is_file():
        return DEM.from_raster(out_path, vertical_datum="ellipsoidal")

    selection = dem_source or os.environ.get("FANINSAR_DEM_SOURCE", "glo30")
    cache_root = Path(
        os.environ.get("FANINSAR_DEM_CACHE_DIR", str(Path(output_dir) / "dem-cache"))
    )
    from shapely.geometry import box

    from faninsar.stack.grid import utm_ups_grid

    # Resolve the authoritative projected target before opening or fetching
    # source resources.  The provider then warps directly to this grid, so a
    # geographic source view remains logical/windowed rather than becoming a
    # full EPSG:4326 ndarray.
    grid = utm_ups_grid(
        box(west, south, east, north),
        resolution_m=30.0,
    )
    source = DEM.from_source(selection, cache_dir=cache_root)
    target_datum = "ellipsoidal" if geoid_correction else "egm2008"
    sampler = source.to_raster(
        grid,
        vertical_datum=target_datum,
    )
    sampler.save(out_path)
    logger.info("Automatic DEM built for %s: %s", bounds, out_path)
    return sampler


def _select_bursts_by_roi(
    roi: BoundingBox | Polygons,
    frame_paths: list[Path],
    swaths: tuple[str, ...],
    orbits: Sequence[str | Path | None] | None = None,
    dem: DEM | None = None,
) -> dict[tuple[int, str], list[int]]:
    """Select bursts whose radar-frame ground quad intersects the ROI."""
    from dataclasses import replace as _replace

    from shapely.geometry import Polygon as ShapelyPolygon

    from faninsar.missions.s1 import read_eof_orbit
    from faninsar.processing.geocoding.geo_lut import burst_geo_quad_lonlat

    region = _roi_geometry(roi)
    products = open_acquisition_products(frame_paths)
    resolved: dict[tuple[int, str], list[int]] = {}
    any_quad = False
    for frame_index, product in enumerate(products):
        orbit_path = None if orbits is None else orbits[frame_index]
        for swath in swaths:
            s1_swath = product.swath(swath)
            if orbit_path is not None:
                s1_swath = _replace(s1_swath, orbit=read_eof_orbit(orbit_path))
            indices: list[int] = []
            shape = (s1_swath.lines_per_burst, s1_swath.samples_per_burst)
            for burst in s1_swath.bursts:
                geometry = _radar_model(
                    s1_swath,
                    burst,
                    shape=shape,
                    row0=burst.index * s1_swath.lines_per_burst,
                    col0=0,
                )
                quad = burst_geo_quad_lonlat(
                    geometry=geometry,
                    radar_shape=shape,
                    dem=dem,
                    device="cpu",
                )
                if quad is None:
                    continue
                any_quad = True
                if region.intersects(ShapelyPolygon(quad)):
                    indices.append(burst.index)
            resolved[(frame_index, swath)] = indices
    if not any_quad:
        reject_invalid_state(
            "ROI selection requires radar geometry; burst rdr2geo quads "
            "could not be built for any burst"
        )
    return resolved


def _buffer_geometry_meters(
    geometry: object,
    target_crs: object,
    buffer_m: float,
) -> object:
    """Buffer a WGS84 shapely geometry by meters in a projected CRS."""
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    inverse = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    projected = shp_transform(
        transformer.transform,
        geometry,
    )
    buffered = projected.buffer(buffer_m)
    return shp_transform(
        inverse.transform,
        buffered,
    )


def _roi_burst_window(
    roi: BoundingBox | Polygons,
    geometry: RadarGeometryModel,
    dem: DEM,
    shape: tuple[int, int],
    *,
    device: str,
    buffer_m: float = 320.0,
) -> tuple[int, int, int, int] | None:
    """Return the radar window covering the ROI-burst quad intersection."""
    from shapely.geometry import Polygon as ShapelyPolygon

    from faninsar.processing.geocoding.geo_lut import (
        burst_geo_quad_lonlat,
        polygon_parts,
    )

    region = _roi_geometry(roi)
    quad = burst_geo_quad_lonlat(geometry, shape, dem, device=device)
    if quad is not None:
        intersection = region.intersection(ShapelyPolygon(quad).buffer(0))
        parts = [part for part in polygon_parts(intersection) if not part.is_empty]
    else:
        parts = []
    if parts:
        rings: list[np.ndarray] = []
        for polygon in parts:
            rings.append(np.asarray(polygon.exterior.coords))
            rings.extend(np.asarray(interior.coords) for interior in polygon.interiors)
        points = np.concatenate(rings)
        lon = points[:, 0]
        lat = points[:, 1]
    else:
        min_lon, min_lat, max_lon, max_lat = region.bounds
        lon = np.asarray([min_lon, max_lon, max_lon, min_lon], dtype=np.float64)
        lat = np.asarray([max_lat, max_lat, min_lat, min_lat], dtype=np.float64)
    height = np.asarray(dem.sample(lat, lon), dtype=np.float64)
    transform = run_geo2rdr(geometry, lat, lon, height, device=device)
    azimuth = np.asarray(transform.azimuth_index, dtype=np.float64)
    range_index = np.asarray(transform.range_index, dtype=np.float64)
    ok = (
        np.asarray(transform.converged, dtype=bool)
        & np.isfinite(azimuth)
        & np.isfinite(range_index)
    )
    if not np.any(ok):
        return None
    # Sentinel-1 azimuth ground speed (~7.0 km/s); a fixed approximation is
    # acceptable for a buffer margin (under- or over-sized by a few percent).
    az_px = max(1, int(np.ceil(buffer_m / (7000.0 * geometry.azimuth_time_interval_s))))
    rg_px = max(1, int(np.ceil(buffer_m / geometry.range_spacing_m)))
    row0 = max(0, int(np.floor(np.min(azimuth[ok]))) - az_px)
    row1 = min(shape[0], int(np.ceil(np.max(azimuth[ok]))) + 1 + az_px)
    col0 = max(0, int(np.floor(np.min(range_index[ok]))) - rg_px)
    col1 = min(shape[1], int(np.ceil(np.max(range_index[ok]))) + 1 + rg_px)
    if row1 <= row0 or col1 <= col0:
        return None
    return row0, row1, col0, col1


def _seconds_of_day(value: datetime) -> float:
    midnight = value.replace(hour=0, minute=0, second=0, microsecond=0)
    return (value - midnight).total_seconds()


def _burst_anx_time_s(burst: S1Burst) -> float | None:
    """Return the burst along-track ascending-node time, or ``None``."""
    value = getattr(burst, "azimuth_anx_time_s", None)
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def _burst_window_seconds(swath: S1Swath) -> float:
    """Return one burst's azimuth extent in seconds."""
    return float(swath.lines_per_burst) * float(swath.azimuth_time_interval_s)


def _burst_time_axis(
    primary_swath: S1Swath,
    secondary_swath: S1Swath,
) -> tuple[list[float], list[float], bool]:
    """Return per-burst along-track times for both swaths.

    Prefers ``azimuth_anx_time_s`` (orbit position), which is identical for the
    same ground strip across Sentinel-1 units.  Falls back to
    ``azimuth_time`` seconds-of-day when either side lacks ANX times; in that
    case the returned flag requests circular (same-day) comparison.
    """
    primary_anx = [_burst_anx_time_s(burst) for burst in primary_swath.bursts]
    secondary_anx = [_burst_anx_time_s(burst) for burst in secondary_swath.bursts]
    anx = [value for value in primary_anx + secondary_anx if value is not None]
    # Use ANX times only when they carry per-burst discrimination: real
    # products set them from the annotation, while stubs may leave a single
    # constant (e.g. 0.0) on every burst.
    if len(primary_anx + secondary_anx) == len(anx) and len(set(anx)) > 1:
        return (
            [float(value) for value in primary_anx],
            [float(value) for value in secondary_anx],
            False,
        )
    primary_tod = [
        _seconds_of_day(burst.azimuth_time) if burst.azimuth_time is not None else None
        for burst in primary_swath.bursts
    ]
    secondary_tod = [
        _seconds_of_day(burst.azimuth_time) if burst.azimuth_time is not None else None
        for burst in secondary_swath.bursts
    ]
    return (
        [value if value is not None else np.nan for value in primary_tod],
        [value if value is not None else np.nan for value in secondary_tod],
        True,
    )


def _common_burst_pairs(
    primary_swath: S1Swath,
    secondary_swath: S1Swath,
) -> dict[int, int]:
    """Return ``{primary_index: secondary_index}`` for bursts sharing ground.

    Matches each primary burst to the secondary burst nearest in along-track
    position (ISCE2 ``getBurstOffset`` model), so a whole-burst index shift
    between products — as between Sentinel-1C and Sentinel-1D, whose bursts
    are clipped to different start lines — still pairs the same ground strip.
    A candidate is accepted when its position gap is within half of one
    burst's azimuth extent.
    """
    primary_axis, secondary_axis, circular = _burst_time_axis(
        primary_swath, secondary_swath
    )
    window_s = _burst_window_seconds(primary_swath)
    tolerance = 0.5 * window_s
    pairs: dict[int, int] = {}
    for primary_index, primary_time in enumerate(primary_axis):
        if not np.isfinite(primary_time):
            pairs[primary_index] = primary_index
            continue
        best_index = -1
        best_gap = np.inf
        for secondary_index, secondary_time in enumerate(secondary_axis):
            if not np.isfinite(secondary_time):
                continue
            gap = abs(primary_time - secondary_time)
            if circular and gap > 12 * 3600:
                gap = 24 * 3600 - gap
            if gap < best_gap:
                best_gap = gap
                best_index = secondary_index
        if best_index >= 0 and best_gap <= tolerance:
            pairs[primary_index] = best_index
    return pairs


def _common_burst_indices(
    primary_swath: S1Swath,
    secondary_swath: S1Swath,
) -> list[int]:
    """Return primary burst indices that have a ground-matched secondary."""
    return sorted(_common_burst_pairs(primary_swath, secondary_swath))


def _swath_range_offsets(
    swaths: tuple[str, ...],
    primary_products: Sequence[S1Product],
) -> dict[str, int]:
    first = primary_products[0].swath(swaths[0])
    offsets: dict[str, int] = {}
    for name in swaths:
        item = primary_products[0].swath(name)
        if name == swaths[0]:
            offsets[name] = 0
        else:
            offset = round(
                (item.slant_range_time_s - first.slant_range_time_s)
                * item.range_sampling_rate_hz
            )
            if offset < 0:
                reject_invalid_state(
                    "swath " + name + " starts before frame swath " + swaths[0]
                )
            offsets[name] = offset
    return offsets
