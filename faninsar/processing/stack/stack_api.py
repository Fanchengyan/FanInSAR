"""Stack-owned ionosphere estimation, correction, and date inversion.

These adapters wire the mission-neutral :mod:`faninsar.processing.atmosphere`
core into the Stack session (PROPOSAL-0036). They mirror the
``form_interferograms`` admission discipline: persisted scene generations
are reopened fail-closed, results are published as transactional ion
artifact generations bound to the consumed scene manifests and the
canonical Stack runtime fingerprint, and degraded generations can never be
consumed silently.

All frequency-domain quantities (carrier, subband centroids, processed
bandwidth, range sampling rate) are caller-supplied parameters; nothing is
inferred from product metadata (owner decision 2026-08-27).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

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
from faninsar.processing.stack.ion_store import (
    IonosphereArtifactStore,
    write_ion_correction_artifact,
    write_ionosphere_artifact,
)
from faninsar.processing.stack.scene_store import CoregisteredSceneStore
from faninsar.processing.unwrap.snaphu_backend import SnaphuConfig

if TYPE_CHECKING:
    from faninsar.processing.atmosphere.config import IonosphereEstimationConfig
    from faninsar.processing.stack.session import Stack

logger = setup_logger(__name__)

IONOSPHERE_METHOD_NAME = "ionosphere_split_spectrum"


def _runtime_identity(stack: Stack) -> str:
    """Return the canonical Stack runtime fingerprint for admission."""
    from faninsar.processing.stack.session import (
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
    if date_id not in stack.coreg_paths:
        reject_invalid_state(
            f"Stack scene generation missing for {date_id}; run coregister_scenes first"
        )
    store = CoregisteredSceneStore.open(stack.coreg_paths[date_id] / "scenes")
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


def _load_role_payload(
    store: CoregisteredSceneStore,
    *,
    date_id: str,
    master: str,
) -> np.ndarray:
    """Load the exact payload role used by IFG formation for this date."""
    role = "reference" if date_id == master else "secondary"
    tag = next(iter(store.unit_map()))
    reference_payload, secondary_payload, _ = store.read(tag)
    if role == "reference":
        return reference_payload
    return secondary_payload


def _ion_root(
    stack: Stack,
    pair_id: str,
    ion_looks: tuple[int, int],
    ion_root: str | Path | None,
) -> Path:
    """Resolve the ion artifact directory in its own namespace tree.

    Ion estimate generations live under ``<ion_root>/ml_<az>x<rg>/<pair>``
    instead of inside the IFG pair directories: the IFG writer fail-closes
    on pair directories that exist without an IFG manifest, so nesting any
    other generation there would block IFG formation.
    """
    base = Path(ion_root) if ion_root is not None else stack.config.work_dir / "ion"
    return base / f"ml_{ion_looks[0]}x{ion_looks[1]}" / pair_id


def estimate_ionosphere(
    stack: Stack,
    *,
    config: IonosphereEstimationConfig,
    range_sampling_rate_hz: float,
    pairs: Any | None = None,
    ion_multilook: tuple[int, int] | None = None,
    ion_root: str | Path | None = None,
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
    result is published as ``ml_<az>x<rg>/<pair>`` under the ion artifact
    root with its own transactional generation bound to the consumed
    scene manifests and the canonical Stack runtime fingerprint.

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
    ion_root : path, optional
        Ion artifact root owning the per-pair ion directories. Defaults
        to ``<work_dir>/ion``.
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
    device : str, optional
        Numerical device for the Torch lanes. Defaults to the Stack
        device with ``auto`` resolved explicitly.
    overwrite : bool, optional
        Republish ion generations that already exist when true.
    degraded : bool, optional
        Mark every published generation degraded (requires a reason).
    degradation_reason : str, optional
        Required non-empty reason when ``degraded`` is true.

    Returns
    -------
    list[IonosphereArtifactStore]
        One validated store per processed pair, in pair order.

    """
    from faninsar.processing.interferometry.pair import form_interferogram
    from faninsar.processing.stack.session import _iter_pair_dates
    from faninsar.processing.unwrap.api import unwrap

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
        root = _ion_root(stack, pair_id, ion_looks, ion_root)
        if (root / "ion_manifest.json").exists() and not overwrite:
            store = IonosphereArtifactStore.open(root)
            if store.pair != (primary, secondary):
                reject_invalid_state(
                    "persisted ion artifact pair does not match its directory"
                )
            stores.append(store)
            continue
        primary_store = _scene_pair_store(stack, primary)
        secondary_store = _scene_pair_store(stack, secondary)
        for attribute in ("grid_shape", "grid_identity", "wavelength_m", "master_id"):
            if getattr(primary_store, attribute) != getattr(secondary_store, attribute):
                reject_invalid_state(
                    f"scene generations disagree on {attribute}; ionosphere "
                    "estimation requires one common aligned grid"
                )
        primary_slc = _load_role_payload(
            primary_store, date_id=primary, master=stack.master
        )
        secondary_slc = _load_role_payload(
            secondary_store, date_id=secondary, master=stack.master
        )
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
            )
            for product in subband_products
        ]
        low_phase = torch.as_tensor(
            np.asarray(unwrap_results[0].unwrapped_phase), dtype=torch.float64
        ).to(torch_device)
        high_phase = torch.as_tensor(
            np.asarray(unwrap_results[1].unwrapped_phase), dtype=torch.float64
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
            coherence=torch.as_tensor(coherence, dtype=torch.float64).to(
                torch_device
            ),
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
    ifg_root: str | Path | None = None,
    ion_root: str | Path | None = None,
    allow_degraded: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Subtract qualified ion screens from unwrapped pair phases.

    For every pair in the configured network the matching IFG and
    ionosphere artifacts are reopened and cross-validated (pair ids, grid
    identity, scene manifest digests, payload shapes). Degraded ion
    generations are refused unless ``allow_degraded`` is true. The
    corrected phases are published as separate ``ion_correction``
    generations inside each pair directory; upstream layers are never
    rewritten.

    Parameters    ----------
    stack : Stack
        Session whose IFG artifacts carry qualified unwrap generations.
    ion_multilook : tuple[int, int]
        Look factors of the ion generation to consume. Its payload shape
        must match the IFG unwrap grid exactly (fail-closed otherwise).
    multilook : tuple[int, int], optional
        IFG artifact view to correct. Defaults to the configured Stack
        multilook.
    ifg_root : path, optional
        IFG artifact root owning the pair directories.
    ion_root : path, optional
        Ion artifact root owning the per-pair ion directories. Defaults
        to ``<work_dir>/ion``.
    allow_degraded : bool, optional
        Explicitly consume degraded ion generations when true.
    overwrite : bool, optional
        Republish correction generations that already exist when true.

    Returns
    -------
    dict[str, pathlib.Path]
        Pair id → published correction generation directory.

    """
    stores = stack._pair_artifact_stores(
        looks=multilook or stack.config.multilook,
        ifg_root=ifg_root,
    )
    published: dict[str, Path] = {}
    try:
        for store in stores:
            pair_id = f"{store.pair[0]}_{store.pair[1]}"
            ion_store_root = _ion_root(
                stack, pair_id, tuple(int(value) for value in ion_multilook), ion_root
            )
            ion_store = IonosphereArtifactStore.open(ion_store_root)
            try:
                if ion_store.pair != store.pair:
                    reject_invalid_state(f"ion artifact pair mismatch for {pair_id}")
                if ion_store.degraded and not allow_degraded:
                    reject_invalid_state(
                        f"ion generation for {pair_id} is degraded "
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
                            f"different {role} scenes than the IFG generation"
                        )
                unwrapped = store.read_unwrapped()
                ion = ion_store.read()
                if ion.ionosphere_phase.shape != unwrapped.unwrapped_phase.shape:
                    reject_invalid_state(
                        f"ion grid shape for {pair_id} differs from the IFG "
                        "unwrap grid; estimate the ionosphere at the same "
                        "multilook to correct it"
                    )
                target = store.root / "ion_correction"
                if (
                    (target / "ION_CORRECTION_CURRENT").exists()
                    or (target / "ion_correction_manifest.json").exists()
                ) and not overwrite:
                    published[pair_id] = target
                    continue
                write_ion_correction_artifact(
                    store.root,
                    unwrapped_phase=(
                        unwrapped.unwrapped_phase - ion.ionosphere_phase
                    ).astype(np.float32),
                    ion_manifest_digest=ion_store.manifest_digest,
                    ifg_manifest_digest=store.manifest_digest,
                    degraded_consumed=bool(ion_store.degraded),
                    replace_existing=overwrite,
                )
                published[pair_id] = target
            finally:
                ion_store.close()
    finally:
        for store in stores:
            store.close()
    return published


def invert_ionosphere_dates(
    stack: Stack,
    *,
    ion_multilook: tuple[int, int],
    ion_root: str | Path | None = None,
    reference_date: str | None = None,
    excluded_dates: tuple[str, ...] | list[str] = (),
    excluded_pairs: tuple[str, ...] | list[str] = (),
    screening_iterations: int = 6,
    screening_threshold: float = 3.0,
    device: str = "cpu",
) -> IonosphereNetworkResult:
    """Invert published pair ion screens into per-date screens.

    Loads every qualified ion generation of the configured network
    (fail-closed on missing or mismatched artifacts) and solves the
    alosStack ``ion_ls`` date-level weighted least-squares system with a
    fixed reference date. Weights are the reciprocal of the persisted
    per-pixel filter weights (larger persisted weight = smaller window
    reciprocal = more confident observation).

    Parameters
    ----------
    stack : Stack
        Session whose pairs carry published ion generations.
    ion_multilook : tuple[int, int]
        Look factors of the ion generation to consume.
    ion_root : path, optional
        Ion artifact root owning the per-pair ion directories. Defaults
        to ``<work_dir>/ion``.
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
    from faninsar.processing.stack.session import _iter_pair_dates

    expected_pairs = [
        f"{primary}_{secondary}" for primary, secondary in _iter_pair_dates(stack.pairs)
    ]
    pair_screens: dict[str, np.ndarray] = {}
    pair_windows: dict[str, np.ndarray] = {}
    for pair_id in expected_pairs:
        primary, secondary = pair_id.split("_")
        if primary not in stack.catalog.paths or secondary not in stack.catalog.paths:
            continue
        root = _ion_root(
            stack,
            pair_id,
            tuple(int(value) for value in ion_multilook),
            ion_root,
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
