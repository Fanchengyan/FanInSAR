"""Explicit production pair workflow stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from faninsar.missions.sentinel1.types import S1Burst, S1Product, S1Swath
    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.typing import DeviceLike

import numpy as np

from faninsar.logging import setup_logger
from faninsar.missions.sentinel1 import open_safe_product, read_burst_window
from faninsar.processing.coreg.geometry_coreg import (
    build_offset_field,
    geometry_coarse_shift,
    refine_shift_with_correlation,
    resolve_ampcor_policy,
)
from faninsar.processing.coreg.offsets import resample_complex
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.interferometry.pair import form_interferogram, goldstein_filter
from faninsar.processing.pipeline.geocode import GeocodedLayer, geocode_layer
from faninsar.processing.pipeline.products import (
    PairProductArrays,
    write_pair_stac_item,
    write_pair_zarr,
)
from faninsar.processing.tops.carrier import carrier_from_swath
from faninsar.processing.tops.deramp import TOPSCarrierModel, deramp, reramp
from faninsar.processing.unwrap import SnaphuConfig, snaphu_unwrap

logger = setup_logger(__name__)


def _scene_id(path: Path) -> str:
    stem = path.stem.replace(".SAFE", "")
    for part in stem.split("_"):
        if len(part) >= 8 and part[:8].isdigit():
            return part[:8]
    return stem


@dataclass(frozen=True, slots=True)
class SceneBurstData:
    """One scene burst window with annotation-backed geometry."""

    scene_id: str
    path: Path
    product: S1Product
    swath: S1Swath
    burst: S1Burst
    samples: np.ndarray
    row0: int
    col0: int
    carrier: TOPSCarrierModel


@dataclass
class PairWorkflowState:
    """Mutable intermediate state for an explicit pair workflow."""

    pair_id: str
    reference: SceneBurstData
    secondary: SceneBurstData
    reference_deramped: np.ndarray | None = None
    secondary_deramped: np.ndarray | None = None
    range_shift_px: float | None = None
    azimuth_shift_px: float | None = None
    secondary_aligned: np.ndarray | None = None
    complex_ifg: np.ndarray | None = None
    coherence: np.ndarray | None = None
    wrapped_phase: np.ndarray | None = None
    unwrapped_phase: np.ndarray | None = None
    connected_components: np.ndarray | None = None
    geocoded_unwrapped: GeocodedLayer | None = None
    geocoded_coherence: GeocodedLayer | None = None
    zarr_path: Path | None = None
    stac_path: Path | None = None
    log: list[str] = field(default_factory=list)

    def _note(self, message: str) -> None:
        logger.info("[%s] %s", self.pair_id, message)
        self.log.append(message)


def stage_read_scene(
    path: str | Path,
    *,
    swath: str = "IW1",
    burst_index: int = 0,
    height: int = 256,
    width: int = 256,
    row_offset: int = 100,
    col_offset: int = 100,
) -> SceneBurstData:
    """Stage 1: open SAFE, build carrier, read a complex burst window.

    Parameters
    ----------
    path : str or pathlib.Path
        SAFE ZIP or directory.
    swath : str, optional
        Sub-swath name.
    burst_index : int, optional
        Burst index.
    height, width : int, optional
        Window size in radar samples.
    row_offset, col_offset : int, optional
        Offsets inside the valid burst region.

    Returns
    -------
    SceneBurstData
        Scene window with carrier and geometry metadata.

    """
    product = open_safe_product(path)
    s1_swath = product.swath(swath)
    if burst_index < 0 or burst_index >= len(s1_swath.bursts):
        reject_invalid_state(f"burst_index {burst_index} out of range")
    burst = s1_swath.bursts[burst_index]
    window = read_burst_window(
        s1_swath,
        burst_index=burst_index,
        height=height,
        width=width,
        row_offset=row_offset,
        col_offset=col_offset,
    )
    carrier = carrier_from_swath(s1_swath, burst, first_range_sample=window.col0)
    scene_id = _scene_id(Path(path))
    logger.info(
        "READ %s %s burst=%s window=%s origin=(%s,%s)",
        scene_id,
        swath,
        burst_index,
        window.samples.shape,
        window.row0,
        window.col0,
    )
    return SceneBurstData(
        scene_id=scene_id,
        path=Path(path),
        product=product,
        swath=s1_swath,
        burst=burst,
        samples=window.samples,
        row0=window.row0,
        col0=window.col0,
        carrier=carrier,
    )


def stage_deramp(state: PairWorkflowState) -> PairWorkflowState:
    """Stage 2: TOPS deramp both scenes using annotation-derived carriers."""
    ref = deramp(state.reference.samples, state.reference.carrier)
    sec = deramp(state.secondary.samples, state.secondary.carrier)
    state.reference_deramped = ref
    state.secondary_deramped = sec
    state._note(
        f"DERAMP complete |z|_ref={float(np.mean(np.abs(ref))):.3f} "
        f"|z|_sec={float(np.mean(np.abs(sec))):.3f}"
    )
    return state


def stage_coregister(
    state: PairWorkflowState,
    *,
    search_radius: int = 32,
    executor: str = "torch",
    device: str = "auto",
) -> PairWorkflowState:
    """Stage 3: geometry coarse offsets + correlation refinement + resampling.

    Parameters
    ----------
    state : PairWorkflowState
        Pair workflow state after deramp.
    search_radius : int, optional
        Correlation search radius in pixels.
    executor : {"numpy", "torch"}, optional
        Ampcor executor. ``"numpy"`` with ``device="cpu"`` or
        ``device="auto"`` selects the documented portable Ampcor rollback;
        phase-preserving resampling remains Torch-owned. Explicit CUDA is
        qualified only for the Ampcor search radii 8 and 16; the historical
        default ``search_radius=32`` therefore fails closed for CUDA on this
        deprecated route.
    device : {"auto","cpu","cuda","mps"}, optional
        Torch compute device. Default ``"auto"``.

    """
    if state.reference_deramped is None or state.secondary_deramped is None:
        reject_invalid_state("stage_coregister requires stage_deramp first")

    resolved_executor, resolved_ampcor_device = resolve_ampcor_policy(executor, device)
    resolved_torch_device = (
        "auto" if device.strip().lower() == "auto" else resolved_ampcor_device
    )

    prior_rg, prior_az = geometry_coarse_shift(
        state.reference.swath,
        state.secondary.swath,
        reference_burst=state.reference.burst,
        secondary_burst=state.secondary.burst,
    )
    rg, az = refine_shift_with_correlation(
        state.reference_deramped,
        state.secondary_deramped,
        prior_rg=prior_rg,
        prior_az=prior_az,
        search_radius=search_radius,
        executor=resolved_executor,
        device=resolved_ampcor_device,
    )
    offsets = build_offset_field(
        state.reference_deramped.shape,
        range_shift_px=rg,
        azimuth_shift_px=az,
    )
    # Offsets from deramped domain; final resample of original-domain secondary
    # (see production.stage_coregister).
    sec_orig = reramp(state.secondary_deramped, state.secondary.carrier)
    sec_resamp = resample_complex(
        sec_orig,
        range_offset_px=offsets.range_offset_px,
        azimuth_offset_px=offsets.azimuth_offset_px,
        executor="torch",
        device=resolved_torch_device,
    )
    ref_aligned = reramp(state.reference_deramped, state.reference.carrier)
    state.range_shift_px = rg
    state.azimuth_shift_px = az
    state.secondary_aligned = sec_resamp
    state.reference_deramped = ref_aligned
    state._note(
        f"COREG geometry+correlation rg={rg:.3f} px az={az:.3f} px "
        f"(prior rg={prior_rg:.3f} az={prior_az:.3f}; resample original-domain)"
    )
    return state


def stage_interferogram(
    state: PairWorkflowState,
    *,
    multilook: tuple[int, int] = (2, 8),
    goldstein_alpha: float = 0.5,
) -> PairWorkflowState:
    """Stage 4: form complex interferogram, multilook, Goldstein filter."""
    if state.reference_deramped is None or state.secondary_aligned is None:
        reject_invalid_state("stage_interferogram requires stage_coregister first")
    ifg = form_interferogram(
        state.reference_deramped,
        state.secondary_aligned,
        multilook=multilook,
    )
    if goldstein_alpha > 0.0:
        complex_ifg = goldstein_filter(ifg.complex_ifg, alpha=goldstein_alpha)
    else:
        # alpha=0 is not a no-op in windowed Goldstein (Hann still smooths).
        complex_ifg = ifg.complex_ifg
    state.complex_ifg = complex_ifg.astype(np.complex64, copy=False)
    state.coherence = ifg.coherence
    state.wrapped_phase = np.angle(state.complex_ifg).astype(np.float32)
    state._note(
        f"IFG multilook={multilook} goldstein={goldstein_alpha} "
        f"mean_coh={float(np.nanmean(ifg.coherence)):.3f}"
    )
    return state


def stage_unwrap(
    state: PairWorkflowState,
    *,
    config: SnaphuConfig | None = None,
) -> PairWorkflowState:
    """Stage 5: unwrap with snaphu-py.

    Parameters
    ----------
    state : PairWorkflowState
        Pair workflow state after interferogram formation.
    config : SnaphuConfig, optional
        snaphu-py configuration.

    Returns
    -------
    PairWorkflowState
        State with unwrapped phase and connected components filled.

    """
    if state.wrapped_phase is None or state.coherence is None:
        reject_invalid_state("stage_unwrap requires stage_interferogram first")
    ifg = state.complex_ifg
    if ifg is None:
        ifg = np.exp(1j * np.asarray(state.wrapped_phase, dtype=np.float64)).astype(
            np.complex64
        )
    dispatched = snaphu_unwrap(ifg, state.coherence, config=config)
    state.unwrapped_phase = dispatched.unwrapped_phase
    state.connected_components = dispatched.connected_components
    state._note(f"UNWRAP method={dispatched.method} metrics={dispatched.metrics}")
    return state


def stage_geocode(
    state: PairWorkflowState,
    *,
    dem: DEMSampler | None = None,
    device: DeviceLike,
    stride: int = 1,
) -> PairWorkflowState:
    """Stage 6: geocode unwrapped phase and coherence to lon/lat."""
    if state.unwrapped_phase is None or state.coherence is None:
        reject_invalid_state("stage_geocode requires stage_unwrap first")
    state.geocoded_unwrapped = geocode_layer(
        state.unwrapped_phase,
        swath=state.reference.swath,
        burst=state.reference.burst,
        row0=state.reference.row0,
        col0=state.reference.col0,
        dem=dem,
        device=device,
        stride=stride,
    )
    state.geocoded_coherence = geocode_layer(
        state.coherence,
        swath=state.reference.swath,
        burst=state.reference.burst,
        row0=state.reference.row0,
        col0=state.reference.col0,
        dem=dem,
        device=device,
        stride=stride,
    )
    n_ok = int(np.count_nonzero(state.geocoded_unwrapped.converged))
    state._note(f"GEOCODE converged={n_ok}/{state.geocoded_unwrapped.converged.size}")
    return state


def stage_write(
    state: PairWorkflowState,
    output_dir: str | Path,
) -> PairWorkflowState:
    """Stage 7: write radar-coordinate products and geocoded arrays to Zarr/STAC."""
    if (
        state.complex_ifg is None
        or state.coherence is None
        or state.wrapped_phase is None
        or state.unwrapped_phase is None
        or state.connected_components is None
    ):
        reject_invalid_state("stage_write requires completed interferogram/unwrap")

    meta: dict[str, Any] = {
        "unwrap_method": "snaphu",
        "range_shift_px": state.range_shift_px,
        "azimuth_shift_px": state.azimuth_shift_px,
        "reference_scene": state.reference.scene_id,
        "secondary_scene": state.secondary.scene_id,
        "reference_swath": state.reference.swath.swath,
        "burst_index": state.reference.burst.index,
        "window_origin_row": state.reference.row0,
        "window_origin_col": state.reference.col0,
        "stages": list(state.log),
    }
    product = PairProductArrays(
        pair_id=state.pair_id,
        complex_ifg=state.complex_ifg,
        coherence=state.coherence,
        wrapped_phase=state.wrapped_phase,
        unwrapped_phase=state.unwrapped_phase,
        connected_components=state.connected_components,
        metadata=meta,
    )
    out = Path(output_dir)
    zarr_path = write_pair_zarr(product, out / f"{state.pair_id}.zarr")
    # Append geocoded layers into the same store.
    if state.geocoded_unwrapped is not None:
        import zarr

        root = zarr.open_group(str(zarr_path), mode="a")
        geo = root.require_group("geocoded")
        geo.create_array(
            "unwrapped_phase",
            data=state.geocoded_unwrapped.values,
            overwrite=True,
        )
        geo.create_array(
            "latitude_deg",
            data=state.geocoded_unwrapped.latitude_deg,
            overwrite=True,
        )
        geo.create_array(
            "longitude_deg",
            data=state.geocoded_unwrapped.longitude_deg,
            overwrite=True,
        )
        geo.create_array(
            "height_m",
            data=state.geocoded_unwrapped.height_m,
            overwrite=True,
        )
        geo.create_array(
            "converged",
            data=state.geocoded_unwrapped.converged.astype(np.uint8),
            overwrite=True,
        )
        if state.geocoded_coherence is not None:
            geo.create_array(
                "coherence",
                data=state.geocoded_coherence.values,
                overwrite=True,
            )
    stac_path = write_pair_stac_item(product, zarr_path, out / f"{state.pair_id}.json")
    state.zarr_path = zarr_path
    state.stac_path = stac_path
    state._note(f"WRITE {zarr_path} {stac_path}")
    return state


__all__ = [
    "PairWorkflowState",
    "SceneBurstData",
    "stage_coregister",
    "stage_deramp",
    "stage_geocode",
    "stage_interferogram",
    "stage_read_scene",
    "stage_unwrap",
    "stage_write",
]
