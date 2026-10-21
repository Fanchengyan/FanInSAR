"""Stack-owned ionosphere estimation, correction, and date inversion.

These adapters wire the mission-neutral :mod:`faninsar.processing.atmosphere`
core into the Stack session (PROPOSAL-0036). They mirror the
``form_interferograms`` admission discipline: persisted scene products are
reopened fail-closed, results are committed as ion products bound to the
consumed scene ``write_id``s and the canonical Stack runtime fingerprint,
and degraded estimates can never be consumed silently.

All frequency-domain quantities (carrier, subband centroids, processed
bandwidth, range sampling rate) are caller-supplied parameters; nothing is
inferred from product metadata (owner decision 2026-08-27).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from faninsar.io.storage.ion_store import (
    ION_CORRECTED_NAME,
    IonosphereArtifactStore,
    ion_dir,
    write_ion_correction_artifact,
    write_ionosphere_artifact,
)
from faninsar.io.storage.product import open_product
from faninsar.logging import setup_logger
from faninsar.processing.atmosphere.estimation import estimate_disp_nondisp
from faninsar.processing.atmosphere.filter import (
    remove_small_components,
    smooth_inverse_variance,
)
from faninsar.processing.atmosphere.network import (
    IonosphereNetworkResult,
    invert_ionosphere_network,
)
from faninsar.processing.atmosphere.split_spectrum import split_range_spectrum
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from pathlib import Path

    from faninsar.io.storage.scene_store import CoregisteredSceneStore
    from faninsar.processing.atmosphere.config import IonosphereEstimationConfig
    from faninsar.processing.unwrapping.snaphu_backend import SnaphuConfig
    from faninsar.stack.session import Stack

logger = setup_logger(__name__)

IONOSPHERE_METHOD_NAME = "ionosphere_split_spectrum"


def _runtime_identity(stack: Stack) -> str:
    """Return the canonical Stack runtime fingerprint for admission."""
    from faninsar.stack.session import (
        _provider_callback,
        _runtime_fingerprint,
    )

    return _runtime_fingerprint(_provider_callback(stack.scene_provider))


def _resolve_device(stack: Stack, device: str | None) -> str:
    """Resolve the explicit numerical device for the estimation lane."""
    resolved = device or stack.config.device
    if resolved in {"", "auto"}:
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    return resolved


def _scene_pair_store(stack: Stack, date_id: str) -> CoregisteredSceneStore:
    """Open one qualified single-unit radar-domain scene generation."""
    store = stack.scenes.get(date_id)
    if store is None:
        reject_invalid_state(
            f"Stack scene product missing for {date_id}; run coregister_scenes first"
        )
    if store.domain != "radar":
        reject_invalid_state(
            "ionosphere estimation requires radar-domain scene generations"
        )
    units = store.unit_map()
    if len(units) != 1:
        reject_invalid_state(
            "ionosphere estimation requires single-unit scene generations"
        )
    return store


def _load_scene_payload(store: CoregisteredSceneStore) -> np.ndarray:
    """Load the single persisted payload of one acquisition store."""
    tag = next(iter(store.unit_map()))
    payload, _ = store.read(tag)
    return payload


def estimate_ionosphere(
    stack: Stack,
    *,
    config: IonosphereEstimationConfig,
    range_sampling_rate_hz: float,
    pairs: Any | None = None,
    ion_multilook: tuple[int, int] | None = None,
    low_offset_hz: float | None = None,
    high_offset_hz: float | None = None,
    valid_mask: np.ndarray | None = None,
    filter_sigma: tuple[float, float] = (2.0, 2.0),
    min_cluster_pixels: int = 40,
    unwrap_method: Literal["snaphu", "irls"] = "snaphu",
    snaphu_config: SnaphuConfig | None = None,
    device: str | None = None,
    overwrite: bool = False,
    degraded: bool = False,
    degradation_reason: str | None = None,
) -> list[IonosphereArtifactStore]:
    """Estimate per-pair ionospheric screens and publish ion artifacts.

    The lane reopens the same persisted, master-aligned radar-grid scene
    generations consumed by IFG formation (single-unit, radar-domain
    generations only; anything else fails closed), splits both dates with
    the configured range bandpass, forms low/high subband interferograms
    without Goldstein filtering, unwraps them, and runs the dispersive /
    non-dispersive estimation and inverse-variance smoothing. Each pair
    result is committed as the product ``work_dir/ion/ml_<az>x<rg>/<pair>``
    bound to the consumed scene ``write_id``s and the canonical Stack
    runtime fingerprint.

    Parameters
    ----------
    stack : Stack
        Prepared Stack session with qualified scene generations.
    config : IonosphereEstimationConfig
        Caller-supplied carrier and subband frequency configuration.
    range_sampling_rate_hz : float
        Range sampling rate of the persisted radar-grid scenes.
    pairs : Pairs, optional
        Pair network to process. Defaults to the configured network.
    ion_multilook : tuple[int, int], optional
        Look factors of the ion grid. Defaults to the configured Stack
        multilook.
    low_offset_hz, high_offset_hz : float, optional
        Subband center offsets relative to the carrier. Defaults to
        ``∓config.effective_split_hz / 2`` with a subband width of
        ``config.effective_split_hz / 2`` (the alosStack thirds split).
    valid_mask : numpy.ndarray, optional
        Boolean full-resolution mask of pixels to use (True = valid);
        water or exterior exclusions belong here.
    filter_sigma : tuple[float, float], optional
        Azimuth and range Gaussian sigma of the inverse-variance smoothing.
    min_cluster_pixels : int, optional
        Four-connected valid clusters below this size are dropped.
    unwrap_method : {"snaphu", "irls"}, optional
        Subband unwrapping backend. There is no silent fallback.
    snaphu_config : SnaphuConfig, optional
        Configuration passed to the SNAPHU backend when ``unwrap_method`` is
        ``"snaphu"``.
    device : str, optional
        Numerical device for the Torch lanes. Defaults to the Stack
        device with ``auto`` resolved explicitly.
    overwrite : bool, optional
        Replace existing ion products. Without it a product bound to the
        current scenes is reused and a stale one is rejected.
    degraded : bool, optional
        Mark every committed estimate degraded (requires a reason).
    degradation_reason : str, optional
        Required non-empty reason when ``degraded`` is true.

    Returns
    -------
    list[IonosphereArtifactStore]
        One validated store per processed pair, in pair order.

    """
    from faninsar.processing.interferometry.pair import form_interferogram
    from faninsar.processing.unwrapping.api import unwrap
    from faninsar.stack.session import _iter_pair_dates

    stack._ensure_prepared()
    stack._require_qualified_activation_record()

    ion_looks = tuple(int(value) for value in (ion_multilook or stack.config.multilook))
    if len(ion_looks) != 2 or any(value < 1 for value in ion_looks):
        reject_invalid_state("ion multilook must contain two positive integers")
    resolved_device = _resolve_device(stack, device)
    torch_device = torch.device(resolved_device)
    split_low = (
        -config.effective_split_hz / 2.0
        if low_offset_hz is None
        else float(low_offset_hz)
    )
    split_high = (
        config.effective_split_hz / 2.0
        if high_offset_hz is None
        else float(high_offset_hz)
    )
    if split_low >= 0.0 or split_high <= 0.0:
        reject_invalid_state(
            "subband offsets must straddle the carrier: low < 0 < high"
        )
    runtime_fingerprint = _runtime_identity(stack)

    stores: list[IonosphereArtifactStore] = []
    for primary, secondary in _iter_pair_dates(pairs or stack.pairs):
        if primary not in stack.catalog.paths or secondary not in stack.catalog.paths:
            continue
        pair_id = f"{primary}_{secondary}"
        root = ion_dir(stack.config.work_dir, pair_id, ion_looks)
        primary_store = _scene_pair_store(stack, primary)
        secondary_store = _scene_pair_store(stack, secondary)
        if (root / "item.json").exists() and not overwrite:
            store = IonosphereArtifactStore.open(root)
            if store.pair != (primary, secondary):
                reject_invalid_state(
                    "persisted ion artifact pair does not match its directory"
                )
            if store.source_manifest_digests != {
                "primary": primary_store.write_id,
                "secondary": secondary_store.write_id,
            }:
                reject_invalid_state(
                    f"ion product for {pair_id} is bound to replaced scenes; "
                    "pass overwrite=True to re-estimate it"
                )
            stores.append(store)
            continue
        for attribute in (
            "grid_shape",
            "grid_identity",
            "wavelength_m",
            "reference_id",
        ):
            if getattr(primary_store, attribute) != getattr(secondary_store, attribute):
                reject_invalid_state(
                    f"scene generations disagree on {attribute}; ionosphere "
                    "estimation requires one common aligned grid"
                )
        primary_slc = _load_scene_payload(primary_store)
        secondary_slc = _load_scene_payload(secondary_store)
        if primary_slc.shape != secondary_slc.shape:
            reject_invalid_state("scene payloads do not share one grid")

        def _subband(slc: np.ndarray, center_offset_hz: float) -> torch.Tensor:
            tensor = torch.as_tensor(slc).to(torch_device)
            half_width = config.effective_split_hz / 4.0
            return split_range_spectrum(
                tensor,
                range_sampling_rate_hz=range_sampling_rate_hz,
                low_offset_hz=center_offset_hz - half_width,
                high_offset_hz=center_offset_hz + half_width,
                window_function=config.window_function,
                window_shape=config.window_shape,
            ).data

        primary_low = _subband(primary_slc, split_low)
        primary_high = _subband(primary_slc, split_high)
        secondary_low = _subband(secondary_slc, split_low)
        secondary_high = _subband(secondary_slc, split_high)

        subband_products = [
            form_interferogram(
                sub[0].cpu().numpy(),
                sub[1].cpu().numpy(),
                multilook=ion_looks,
                coherence_window=(3, 3),
            )
            for sub in (
                (primary_low, secondary_low),
                (primary_high, secondary_high),
            )
        ]
        unwrap_results = [
            unwrap(
                product.complex_ifg,
                # mask-before-estimate: the water mask must gate the unwrap
                # INPUT, not only the solve afterwards; otherwise snaphu/IRLS
                # integrate decorrelated ocean pixels (ALOS2 WB1 ion grid).
                product.coherence
                if valid_mask is None
                else np.asarray(product.coherence, dtype=np.float32)
                * np.asarray(valid_mask, dtype=np.float32),
                method=unwrap_method,
                snaphu_config=snaphu_config,
                irls_kwargs={"device": resolved_device}
                if unwrap_method == "irls"
                else None,
            )
            for product in subband_products
        ]
        low_phase = torch.as_tensor(
            np.asarray(unwrap_results[0].phase), dtype=torch.float64
        ).to(torch_device)
        high_phase = torch.as_tensor(
            np.asarray(unwrap_results[1].phase), dtype=torch.float64
        ).to(torch_device)
        coherence = np.minimum(
            subband_products[0].coherence, subband_products[1].coherence
        ).astype(np.float64)
        pixel_valid = (
            torch.as_tensor(
                coherence >= config.coherence_threshold, dtype=torch.bool
            ).to(torch_device)
            & torch.isfinite(low_phase)
            & torch.isfinite(high_phase)
        )
        if valid_mask is not None:
            pixel_valid &= torch.as_tensor(
                np.asarray(valid_mask, dtype=bool), dtype=torch.bool
            ).to(torch_device)
        dispersive, nondispersive = estimate_disp_nondisp(
            low_phase,
            high_phase,
            config,
            valid_mask=pixel_valid,
            coherence=torch.as_tensor(coherence, dtype=torch.float64).to(torch_device),
        )
        weights = torch.where(
            pixel_valid,
            torch.as_tensor(coherence, dtype=torch.float64).to(torch_device),
            torch.zeros((), dtype=torch.float64, device=torch_device),
        )
        smoothed = smooth_inverse_variance(
            dispersive,
            weights,
            sigma_y=float(filter_sigma[0]),
            sigma_x=float(filter_sigma[1]),
        )
        cleaned = remove_small_components(
            torch.isfinite(smoothed) & pixel_valid,
            int(min_cluster_pixels),
        )
        screen = torch.where(cleaned, smoothed, torch.nan)
        screen_np = screen.to(dtype=torch.float32).cpu().numpy().astype(np.float32)
        nondisp_np = (
            nondispersive.to(dtype=torch.float32).cpu().numpy().astype(np.float32)
        )
        weight_np = np.where(
            cleaned.cpu().numpy(),
            weights.to(dtype=torch.float32).cpu().numpy(),
            np.float32(0.0),
        ).astype(np.float32)
        published = write_ionosphere_artifact(
            root,
            pair=(primary, secondary),
            looks=ion_looks,
            domain=primary_store.domain,
            wavelength_m=primary_store.wavelength_m,
            grid_identity=primary_store.grid_identity,
            method_name=IONOSPHERE_METHOD_NAME,
            method_parameters={
                "f0_hz": float(config.f0),
                "freq_low_hz": float(config.freq_low),
                "freq_high_hz": float(config.freq_high),
                "low_offset_hz": float(split_low),
                "high_offset_hz": float(split_high),
                "range_sampling_rate_hz": float(range_sampling_rate_hz),
                "window_function": config.window_function,
                "window_shape": float(config.window_shape),
                "alignment_strategy": config.alignment_strategy,
                "coherence_threshold": float(config.coherence_threshold),
                "looks": list(ion_looks),
                "filter_sigma": [float(filter_sigma[0]), float(filter_sigma[1])],
                "min_cluster_pixels": int(min_cluster_pixels),
                "unwrap_method": str(unwrap_method),
            },
            degraded=degraded,
            degradation_reason=degradation_reason,
            runtime_fingerprint=runtime_fingerprint,
            devices=(resolved_device,),
            source_manifest_digests={
                "primary": primary_store.manifest_digest,
                "secondary": secondary_store.manifest_digest,
            },
            ionosphere_phase=screen_np,
            nondispersive_phase=nondisp_np,
            weight=weight_np,
            replace_existing=overwrite,
        )
        stores.append(published)
    return stores


def apply_ionosphere_correction(
    stack: Stack,
    *,
    ion_multilook: tuple[int, int],
    multilook: tuple[int, int] | None = None,
    allow_degraded: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Subtract qualified ion screens from unwrapped pair phases.

    For every planned Pair the IFG, its current unwrap, and the ion product
    are reopened and cross-validated (pair ids, grid identity, scene write
    ids, layer shapes). Degraded ion products are refused unless
    ``allow_degraded`` is true. Each corrected phase is committed as the
    ``ion_corrected`` child of the Pair's unwrap product, bound to the unwrap
    and ion ``write_id``s; upstream layers are never rewritten.

    Parameters
    ----------
    stack : Stack
        Session whose planned Pairs carry current unwrap products.
    ion_multilook : tuple[int, int]
        Look factors of the ion products to consume. Their shape must match
        the unwrap grid exactly (fail-closed otherwise).
    multilook : tuple[int, int], optional
        IFG product view to correct. Defaults to the configured multilook.
    allow_degraded : bool, optional
        Explicitly consume degraded ion products when true.
    overwrite : bool, optional
        Replace existing corrections. Without it a correction bound to the
        current unwrap and ion products is reused and a stale one is
        rejected.

    Returns
    -------
    dict[str, pathlib.Path]
        Pair id to the committed correction directory.

    """
    looks = (int(ion_multilook[0]), int(ion_multilook[1]))
    stores = stack._pair_artifact_stores(looks=multilook or stack.config.multilook)
    published: dict[str, Path] = {}
    for store in stores:
        pair_id = f"{store.pair[0]}_{store.pair[1]}"
        if not store.has_unwrapped():
            reject_invalid_state(
                f"Pair {pair_id} has no unwrap product bound to its current IFG; "
                "call Stack.unwrap(unwrapper) first"
            )
        ion_store = IonosphereArtifactStore.open(
            ion_dir(stack.config.work_dir, pair_id, looks)
        )
        if ion_store.pair != store.pair:
            reject_invalid_state(f"ion artifact pair mismatch for {pair_id}")
        if ion_store.degraded and not allow_degraded:
            reject_invalid_state(
                f"ion product for {pair_id} is degraded "
                f"({ion_store.degradation_reason!r}); pass "
                "allow_degraded=True to consume it explicitly"
            )
        if ion_store.grid_identity != store.grid_identity:
            reject_invalid_state(
                f"ion artifact for {pair_id} is bound to a different grid"
            )
        for role in ("primary", "secondary"):
            if ion_store.source_manifest_digests.get(
                role
            ) != store.source_manifest_digests.get(role):
                reject_invalid_state(
                    f"ion artifact for {pair_id} was estimated from "
                    f"different {role} scenes than the IFG"
                )
        unwrap = store.unwrap_product()
        target = store.unwrap_root / ION_CORRECTED_NAME
        if (target / "item.json").is_file() and not overwrite:
            existing = open_product(target)
            if existing.inputs == {
                "unwrap": unwrap.write_id,
                "ion": ion_store.write_id,
            }:
                published[pair_id] = target
                continue
            reject_invalid_state(
                f"ion correction for {pair_id} is bound to replaced products; "
                "pass overwrite=True to recompute it"
            )
        unwrapped_phase = unwrap.read("unwrapped_phase")
        if unwrapped_phase.shape != store.shape:
            reject_invalid_state(
                f"unwrap grid shape for {pair_id} differs from the IFG grid"
            )
        ion = ion_store.read()
        if ion.ionosphere_phase.shape != unwrapped_phase.shape:
            reject_invalid_state(
                f"ion grid shape for {pair_id} differs from the IFG "
                "unwrap grid; estimate the ionosphere at the same "
                "multilook to correct it"
            )
        write_ion_correction_artifact(
            store.root,
            unwrapped_phase=(unwrapped_phase - ion.ionosphere_phase).astype(np.float32),
            unwrap_write_id=unwrap.write_id,
            ion_write_id=ion_store.write_id,
            ion_looks=looks,
            degraded_consumed=bool(ion_store.degraded),
            replace_existing=overwrite,
        )
        published[pair_id] = target
    return published


def invert_ionosphere_dates(
    stack: Stack,
    *,
    ion_multilook: tuple[int, int],
    reference_date: str | None = None,
    excluded_dates: tuple[str, ...] | list[str] = (),
    excluded_pairs: tuple[str, ...] | list[str] = (),
    screening_iterations: int = 6,
    screening_threshold: float = 3.0,
    device: str = "cpu",
) -> IonosphereNetworkResult:
    """Invert committed pair ion screens into per-date screens.

    Loads every ion product of the configured network (fail-closed on
    missing or mismatched products) and solves the
    alosStack ``ion_ls`` date-level weighted least-squares system with a
    fixed reference date. Weights are the reciprocal of the persisted
    per-pixel filter weights (larger persisted weight = smaller window
    reciprocal = more confident observation).

    Parameters
    ----------
    stack : Stack
        Session whose pairs carry committed ion products.
    ion_multilook : tuple[int, int]
        Look factors of the ion products to consume.
    reference_date : str, optional
        Date whose screen is pinned to zero; defaults to the earliest
        date in the network.
    excluded_dates : sequence of str, optional
        Every pair touching one of these dates is dropped before the
        connectivity gate.
    excluded_pairs : sequence of str, optional
        Pair identifiers dropped before the connectivity gate.
    screening_iterations : int, optional
        Maximum robust screening rounds; ``0`` disables screening.
    screening_threshold : float, optional
        Standardized residual magnitude where Huber reweighting begins.
    device : str, optional
        Torch device hosting the network solve. The result is NumPy.

    Returns
    -------
    IonosphereNetworkResult
        Per-date screens plus exact pair provenance.

    """
    from faninsar.stack.session import _iter_pair_dates

    expected_pairs = [
        f"{primary}_{secondary}" for primary, secondary in _iter_pair_dates(stack.pairs)
    ]
    pair_screens: dict[str, np.ndarray] = {}
    pair_windows: dict[str, np.ndarray] = {}
    for pair_id in expected_pairs:
        primary, secondary = pair_id.split("_")
        if primary not in stack.catalog.paths or secondary not in stack.catalog.paths:
            continue
        root = ion_dir(
            stack.config.work_dir,
            pair_id,
            (int(ion_multilook[0]), int(ion_multilook[1])),
        )
        ion_store = IonosphereArtifactStore.open(root)
        try:
            if ion_store.pair != (primary, secondary):
                reject_invalid_state(f"ion artifact pair mismatch for {pair_id}")
            artifact = ion_store.read()
            pair_screens[pair_id] = artifact.ionosphere_phase.astype(np.float64)
            weight = artifact.weight.astype(np.float64)
            valid_window = np.isfinite(weight) & (weight > 0.0)
            pair_windows[pair_id] = np.where(
                valid_window,
                1.0 / np.where(valid_window, weight, 1.0),
                np.nan,
            )
        finally:
            ion_store.close()
    if not pair_screens:
        reject_invalid_state(
            "no published ion artifacts match the configured Stack network"
        )
    return invert_ionosphere_network(
        pair_screens,
        pair_windows,
        reference_date=reference_date,
        excluded_dates=excluded_dates,
        excluded_pairs=excluded_pairs,
        screening_iterations=screening_iterations,
        screening_threshold=screening_threshold,
        device=device,
    )


__all__ = [
    "IONOSPHERE_METHOD_NAME",
    "apply_ionosphere_correction",
    "estimate_ionosphere",
    "invert_ionosphere_dates",
]
