"""Topographic phase computation and removal for interferograms."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.dem import DEM
from faninsar.processing.geometry.ellipsoid import llh_to_ecef
from faninsar.processing.geometry.prepare_production import run_rdr2geo

if TYPE_CHECKING:
    from faninsar.processing.dem import RasterDEM
    from faninsar.processing.geometry import RadarGeometryModel
    from faninsar.processing.runtime.types import DeviceLike

logger = setup_logger(__name__)


def compute_geometric_phase_from_geo(
    model_ref: RadarGeometryModel,
    model_sec: RadarGeometryModel,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    height_m: np.ndarray,
    reference_azimuth_index: np.ndarray,
    secondary_azimuth_index: np.ndarray,
    *,
    reference_range_index: np.ndarray | None = None,
    wavelength_m: float | None = None,
) -> np.ndarray:
    """Compute pair geometric phase directly from shared ground targets.

    Parameters
    ----------
    model_ref, model_sec : RadarGeometryModel
        Reference and secondary orbit timing models.
    latitude_deg, longitude_deg, height_m : numpy.ndarray
        Shared geodetic target coordinates and ellipsoidal heights.
    reference_azimuth_index, secondary_azimuth_index : numpy.ndarray
        Zero-Doppler azimuth coordinates for each acquisition.
    reference_range_index : numpy.ndarray, optional
        Reference slant-range indices for the shared targets. Providing these
        reuses the solved Geo2Rdr range instead of evaluating the reference
        orbit and Euclidean range a second time.
    wavelength_m : float, optional
        Radar wavelength. Defaults to the reference model wavelength.

    Returns
    -------
    numpy.ndarray
        Geometric phase in radians, with invalid inputs represented by NaN.

    """
    wavelength = model_ref.wavelength_m if wavelength_m is None else wavelength_m
    if wavelength <= 0:
        message = "wavelength must be positive"
        logger.error(message)
        raise ValueError(message)

    broadcast_inputs = [
        np.asarray(latitude_deg, dtype=np.float64),
        np.asarray(longitude_deg, dtype=np.float64),
        np.asarray(height_m, dtype=np.float64),
        np.asarray(reference_azimuth_index, dtype=np.float64),
        np.asarray(secondary_azimuth_index, dtype=np.float64),
    ]
    if reference_range_index is not None:
        broadcast_inputs.append(np.asarray(reference_range_index, dtype=np.float64))
    broadcast = np.broadcast_arrays(*broadcast_inputs)
    latitude, longitude, height, reference_azimuth, secondary_azimuth = broadcast[:5]
    reference_range = broadcast[5] if reference_range_index is not None else None
    valid = (
        np.isfinite(latitude)
        & np.isfinite(longitude)
        & np.isfinite(height)
        & np.isfinite(reference_azimuth)
        & np.isfinite(secondary_azimuth)
    )
    phase = np.full(latitude.shape, np.nan, dtype=np.float64)
    if reference_range is not None:
        valid &= np.isfinite(reference_range)
    if not np.any(valid):
        return phase

    secondary_times = model_sec.azimuth_time_seconds(secondary_azimuth[valid])
    satellite_sec, _ = model_sec.orbit.evaluate_array(secondary_times)
    target_x, target_y, target_z = llh_to_ecef(
        latitude[valid],
        longitude[valid],
        height[valid],
    )
    targets = np.stack([target_x, target_y, target_z], axis=-1)
    if reference_range is None:
        reference_times = model_ref.azimuth_time_seconds(reference_azimuth[valid])
        satellite_ref, _ = model_ref.orbit.evaluate_array(reference_times)
        range_ref = np.linalg.norm(targets - satellite_ref, axis=-1)
    else:
        range_ref = (
            model_ref.starting_slant_range_m
            + reference_range[valid] * model_ref.range_spacing_m
        )
    range_sec = np.linalg.norm(targets - satellite_sec, axis=-1)
    usable = np.isfinite(range_ref) & np.isfinite(range_sec)
    values = np.full(range_ref.shape, np.nan, dtype=np.float64)
    values[usable] = (4.0 * np.pi / wavelength) * (
        range_sec[usable] - range_ref[usable]
    )
    phase[valid] = values
    return phase


def compute_topographic_phase(
    model_ref: RadarGeometryModel,
    model_sec: RadarGeometryModel,
    azimuth_index: np.ndarray,
    range_index: np.ndarray,
    dem: DEM,
    *,
    device: DeviceLike,
    secondary_azimuth_index: np.ndarray | None = None,
    wavelength_m: float | None = None,
) -> np.ndarray:
    r"""Compute topographic phase for a coregistered pair from geometry and DEM.

    Uses the two-way path-length geometric phase consistent with
    interferogram formation ``primary * conj(secondary)``:

    .. math::
        \phi_{\text{geo}} = +\frac{4\pi}{\lambda}
        \left(R_{\text{sec}} - R_{\text{ref}}\right)

    where :math:`R` is the one-way slant range from the sensor to the DEM
    target and :math:`\lambda` is the wavelength.  This matches the radar
    phase convention :math:`\varphi = -4\pi R / \lambda` so that
    :math:`\varphi_{\text{ref}} - \varphi_{\text{sec}} =
    +4\pi (R_{\text{sec}} - R_{\text{ref}}) / \lambda`.

    Parameters
    ----------
    model_ref : RadarGeometryModel
        Reference acquisition geometry model.
    model_sec : RadarGeometryModel
        Secondary acquisition geometry model.
    azimuth_index, range_index : numpy.ndarray
        Radar sample coordinates on the coregistered grid.
    dem : DEM
        DEM height sampler.
    device : DeviceLike
        Required production device (``auto`` resolves to cpu or cuda).
    secondary_azimuth_index : numpy.ndarray, optional
        Secondary zero-Doppler azimuth coordinates for the same ground targets.
        Direct-remap workflows should pass the fractional source coordinates
        already computed during coregistration. If omitted, the reference
        azimuth coordinates are used for both orbit evaluations.
    wavelength_m : float, optional
        Radar wavelength in metres. If None, uses ``model_ref.wavelength_m``.

    Returns
    -------
    numpy.ndarray
        Topographic phase array in radians. Non-converged pixels are NaN.

    """
    if wavelength_m is None:
        wavelength_m = model_ref.wavelength_m
    if wavelength_m <= 0:
        message = "wavelength must be positive"
        logger.error(message)
        raise ValueError(message)

    geo = run_rdr2geo(
        model_ref,
        azimuth_index,
        range_index,
        dem,
        device=device,
    )

    phase = np.full(geo.latitude_deg.shape, np.nan, dtype=np.float64)
    if not np.any(geo.converged):
        return phase

    conv_mask = (
        geo.converged
        & np.isfinite(geo.latitude_deg)
        & np.isfinite(geo.longitude_deg)
        & np.isfinite(geo.height_m)
    )
    az_b, _range_b = np.broadcast_arrays(
        np.asarray(azimuth_index, dtype=np.float64),
        np.asarray(range_index, dtype=np.float64),
    )
    if secondary_azimuth_index is None:
        secondary_az_b = az_b
    else:
        secondary_az_b = np.broadcast_to(
            np.asarray(secondary_azimuth_index, dtype=np.float64),
            az_b.shape,
        )
        conv_mask &= np.isfinite(secondary_az_b)
    if not np.any(conv_mask):
        return phase

    reference_azimuth = az_b[conv_mask]
    secondary_azimuth = secondary_az_b[conv_mask]

    phase[conv_mask] = compute_geometric_phase_from_geo(
        model_ref,
        model_sec,
        geo.latitude_deg[conv_mask],
        geo.longitude_deg[conv_mask],
        geo.height_m[conv_mask],
        reference_azimuth,
        secondary_azimuth,
        wavelength_m=wavelength_m,
    )
    return phase


# Path-length geometric phase (orbital + topographic).
compute_geometric_phase = compute_topographic_phase


def remove_topographic_phase(
    complex_ifg: np.ndarray,
    topo_phase: np.ndarray,
) -> np.ndarray:
    """Remove topographic phase from a complex interferogram.

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Complex interferogram array.
    topo_phase : numpy.ndarray
        Topographic phase in radians, broadcastable to ``complex_ifg``.

    Returns
    -------
    numpy.ndarray
        Flattened complex interferogram.

    """
    ifg = np.asarray(complex_ifg)
    topo = np.asarray(topo_phase)
    if ifg.shape != topo.shape:
        try:
            np.broadcast_to(topo, ifg.shape)
        except ValueError as exc:
            message = (
                f"topo_phase shape {topo.shape} cannot broadcast to "
                f"ifg shape {ifg.shape}"
            )
            logger.exception(message)
            raise ValueError(message) from exc
    return ifg * np.exp(-1j * topo)


def _azimuth_ramp_weights(
    ifg: np.ndarray,
    reference_phase: np.ndarray,
    coherence: np.ndarray | None,
    coh_thr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build residual phase, weights, and validity mask for ramp scoring."""
    phase = np.angle(ifg)
    mask = np.isfinite(phase) & np.isfinite(reference_phase) & (np.abs(ifg) > 0)
    if coherence is not None:
        coherence_array = np.asarray(coherence)
        mask = mask & np.isfinite(coherence_array) & (coherence_array >= coh_thr)
        weights = np.where(mask, coherence_array.astype(np.float64, copy=False), 0.0)
    else:
        weights = mask.astype(np.float64)
    residual = np.where(mask, phase - reference_phase, 0.0)
    return residual, weights, mask


def _score_azimuth_ramp_candidates(
    row_sums: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Score candidate coefficients against azimuth-row circular sums."""
    azimuth = np.arange(row_sums.shape[0], dtype=np.float64)
    return np.abs(
        (np.exp(-1j * candidates[:, None] * azimuth[None, :]) * row_sums[None, :]).sum(
            axis=1
        )
    )


def _estimate_residual_azimuth_ramp_numpy(
    residual: np.ndarray,
    weights: np.ndarray,
    candidates: np.ndarray,
) -> float:
    """Vectorized NumPy scorer using one azimuth-row reduction."""
    row_sums = np.sum(weights * np.exp(1j * residual), axis=1)
    scores = _score_azimuth_ramp_candidates(row_sums, candidates)
    return float(candidates[int(np.argmax(scores))])


def azimuth_ramp_device_kwargs(device: str) -> dict[str, str]:
    """Translate a Stack ``device=`` into azimuth-ramp executor options.

    The Stack device is forwarded to the Torch executor. Unpublished
    backends fail closed in :func:`resolve_torch_device`; they do not
    silently select NumPy.

    Parameters
    ----------
    device : str
        Stack device string admitted by :func:`~faninsar.processing.runtime.device.parse_device`.

    Returns
    -------
    dict of str
        Keyword arguments for :func:`estimate_residual_azimuth_ramp`.

    """
    requested = str(device).strip()
    return {"executor": "torch", "device": requested or "auto"}


def _estimate_residual_azimuth_ramp_torch(
    residual: np.ndarray,
    weights: np.ndarray,
    candidates: np.ndarray,
    *,
    device: str,
    candidate_chunk: int,
) -> float:
    """Eager Torch CPU/CUDA scorer using the same row-reduction contract.

    Parameters
    ----------
    residual : numpy.ndarray
        Masked residual phase in radians.
    weights : numpy.ndarray
        Finite pixel weights; invalid pixels are already zero.
    candidates : numpy.ndarray
        Ordered candidate coefficients.
    device : {"auto", "cpu", "cuda", "mps"}
        Torch execution device resolved by :func:`resolve_torch_device`.
    candidate_chunk : int
        Number of candidates scored in one workspace.

    Returns
    -------
    float
        Selected ramp coefficient.

    """
    import torch

    from faninsar.processing.torch_kernels import resolve_torch_device

    resolved = resolve_torch_device(device)
    residual_tensor = torch.as_tensor(residual, dtype=torch.float64, device=resolved)
    weights_tensor = torch.as_tensor(weights, dtype=torch.float64, device=resolved)
    row_sums = torch.sum(
        weights_tensor * torch.exp(1j * residual_tensor),
        dim=1,
        dtype=torch.complex128,
    )
    azimuth = torch.arange(row_sums.shape[0], dtype=torch.float64, device=resolved)
    candidate_tensor = torch.as_tensor(candidates, dtype=torch.float64, device=resolved)
    best_score = torch.tensor(-1.0, dtype=torch.float64, device=resolved)
    best_index = torch.tensor(0, dtype=torch.int64, device=resolved)
    for start in range(0, int(candidate_tensor.shape[0]), candidate_chunk):
        chunk = candidate_tensor[start : start + candidate_chunk]
        scores = torch.abs(
            (
                row_sums[None, :] * torch.exp(-1j * chunk[:, None] * azimuth[None, :])
            ).sum(dim=1, dtype=torch.complex128)
        )
        chunk_score, chunk_index = torch.max(scores, dim=0)
        if bool((chunk_score > best_score).item()):
            best_score = chunk_score
            best_index = start + chunk_index
    if resolved.type == "cuda":
        torch.cuda.synchronize()
    return float(candidates[int(best_index.item())])


def estimate_residual_azimuth_ramp(
    complex_ifg: np.ndarray,
    reference_phase: np.ndarray,
    *,
    coherence: np.ndarray | None = None,
    coh_thr: float = 0.2,
    search: tuple[float, float] = (-0.5, 0.5),
    n_grid: int = 201,
    executor: Literal["numpy", "torch"] = "numpy",
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
    candidate_chunk: int = 41,
) -> float:
    r"""Estimate residual linear azimuth phase ramp (rad per azimuth sample).

    Residual Doppler / TOPS differential carrier often leaves a nearly linear
    phase screen ``c \\cdot i_{az}`` on top of the geometric interferogram.
    The coefficient ``c`` is chosen to maximise the circular correlation of
    the interferogram against a reference phase model (typically the
    path-length geometric phase from orbits + DEM):

    .. math::

        \\hat{c} = \\arg\\max_c
        \\left| \\sum_p w_p \\,
        e^{i\\bigl(\\phi_{\\mathrm{ifg}}(p)
        - \\phi_{\\mathrm{ref}}(p) - c\\, i_{az}\\bigr)} \\right|

    Both executors reduce candidate-independent terms to one complex sum per
    azimuth row, then score the ordered candidate grid. The first (lowest-index)
    candidate wins an exact score tie. The default path is portable NumPy
    in/out. ``executor="torch"`` scores on the device admitted by
    :func:`resolve_torch_device` (CPU or CUDA).

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Complex interferogram.
    reference_phase : numpy.ndarray
        Reference phase in radians (e.g. geometric/topo phase), same shape.
    coherence : numpy.ndarray, optional
        Weights; pixels with coherence below ``coh_thr`` are ignored when
        provided.
    coh_thr : float, optional
        Coherence threshold. Default 0.2.
    search : tuple of float, optional
        Inclusive range of ``c`` to search (rad / azimuth sample).
    n_grid : int, optional
        Number of grid points in the search. Default 201.
    executor : {"numpy", "torch"}, optional
        Scoring backend. ``"numpy"`` (default) is the portable CPU path.
        ``"torch"`` runs the same algorithm on the admitted device.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Device for the Torch executor, resolved by
        :func:`resolve_torch_device`. ``"auto"`` selects CUDA when available
        and CPU otherwise. Ignored by NumPy.
    candidate_chunk : int, optional
        Number of candidates scored together by the Torch executor. Default 41.

    Returns
    -------
    float
        Estimated ramp coefficient ``c`` in rad per azimuth sample.

    Raises
    ------
    ValueError
        If the inputs have mismatched shapes or the executor/device is invalid.
    ImportError
        If ``executor="torch"`` is requested and Torch is not installed.
    RuntimeError
        If ``device="cuda"`` is requested and CUDA is unavailable.

    """
    if executor not in {"numpy", "torch"}:
        message = f"unsupported azimuth-ramp executor: {executor!r}"
        logger.error(message)
        raise ValueError(message)
    if int(candidate_chunk) < 1:
        message = "candidate_chunk must be a positive integer"
        logger.error(message)
        raise ValueError(message)
    ifg = np.asarray(complex_ifg)
    ref = np.asarray(reference_phase, dtype=np.float64)
    if ifg.shape != ref.shape:
        message = f"reference_phase shape {ref.shape} must match ifg shape {ifg.shape}"
        logger.error(message)
        raise ValueError(message)
    residual, weights, mask = _azimuth_ramp_weights(ifg, ref, coherence, coh_thr)
    if not np.any(mask):
        return 0.0
    candidates = np.linspace(search[0], search[1], int(n_grid), dtype=np.float64)
    if executor == "numpy":
        return _estimate_residual_azimuth_ramp_numpy(residual, weights, candidates)
    return _estimate_residual_azimuth_ramp_torch(
        residual,
        weights,
        candidates,
        device=str(device),
        candidate_chunk=int(candidate_chunk),
    )


def remove_azimuth_phase_ramp(
    complex_ifg: np.ndarray,
    ramp_rad_per_az: float,
) -> np.ndarray:
    """Multiply interferogram by ``exp(-i * c * i_az)`` to remove a linear ramp.

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Complex interferogram.
    ramp_rad_per_az : float
        Phase ramp coefficient in rad per azimuth sample.

    Returns
    -------
    numpy.ndarray
        Ramp-corrected complex interferogram.

    """
    ifg = np.asarray(complex_ifg)
    if abs(ramp_rad_per_az) < 1e-15:
        return ifg.astype(np.complex64, copy=False)
    az = np.arange(ifg.shape[0], dtype=np.float64)[:, None]
    return (ifg * np.exp(-1j * ramp_rad_per_az * az)).astype(ifg.dtype, copy=False)


def estimate_residual_topographic_scale(
    complex_ifg: np.ndarray,
    topo_phase: np.ndarray,
    *,
    coherence: np.ndarray | None = None,
    coh_thr: float = 0.2,
    search: tuple[float, float] = (-1.5, 1.5),
    n_grid: int = 61,
) -> tuple[float, float]:
    """Estimate residual scale of a DEM topographic phase model.

    Used after a first-order range-offset flatten: the residual IFG is
    correlated against ``s * topo_phase`` to recover leftover topography
    (and under/over-flattening).  Returns ``(scale, residual_rms)`` where
    ``scale`` maximises circular correlation and ``residual_rms`` is the
    post-correction phase standard deviation on high-coherence pixels.

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Complex interferogram (after range-offset flatten when applicable).
    topo_phase : numpy.ndarray
        Dual-orbit DEM geometric phase (radians), same shape.
    coherence : numpy.ndarray, optional
        Weights; pixels below ``coh_thr`` are ignored.
    coh_thr : float, optional
        Coherence threshold. Default 0.2.
    search : tuple of float, optional
        Inclusive range of scale factors to search.
    n_grid : int, optional
        Grid points in the scale search. Default 61.

    Returns
    -------
    scale, residual_rms : tuple[float, float]
        Best-fit scale and residual phase RMS after applying it.

    """
    ifg = np.asarray(complex_ifg)
    topo = np.asarray(topo_phase, dtype=np.float64)
    if ifg.shape != topo.shape:
        message = f"topo_phase shape {topo.shape} must match ifg shape {ifg.shape}"
        logger.error(message)
        raise ValueError(message)
    ph = np.angle(ifg)
    mask = np.isfinite(ph) & np.isfinite(topo) & (np.abs(ifg) > 0)
    if coherence is not None:
        coh = np.asarray(coherence)
        mask = mask & np.isfinite(coh) & (coh >= coh_thr)
        weights = np.where(mask, coh.astype(np.float64), 0.0)
    else:
        weights = mask.astype(np.float64)
    if not np.any(mask):
        return 0.0, float("nan")

    best_s = 0.0
    best_score = -1.0
    for s in np.linspace(search[0], search[1], int(n_grid)):
        score = float(np.abs(np.nansum(weights * np.exp(1j * (ph - s * topo)))))
        if score > best_score:
            best_score = score
            best_s = float(s)
    residual = np.angle(np.exp(1j * (ph[mask] - best_s * topo[mask])))
    residual_rms = float(np.std(residual)) if residual.size else float("nan")
    return best_s, residual_rms


def _largest_connected_component_mask(
    connected_components: np.ndarray,
    base_mask: np.ndarray,
) -> np.ndarray:
    """Return ``base_mask`` restricted to the largest non-zero conncomp label.

    Multi-component unwrap labels often carry independent 2π offsets.  A global
    poly fit across those offsets invents a huge residual screen.  Fitting only
    on the dominant component keeps the screen physical.

    Parameters
    ----------
    connected_components : numpy.ndarray
        Integer component labels (0 = invalid / background).
    base_mask : numpy.ndarray
        Boolean mask of candidate samples (finite + coherence).

    Returns
    -------
    numpy.ndarray
        Boolean mask; equals ``base_mask`` when no positive labels are present.

    """
    labels = np.asarray(connected_components)
    if labels.shape != base_mask.shape:
        message = (
            f"connected_components shape {labels.shape} must match mask shape "
            f"{base_mask.shape}"
        )
        logger.error(message)
        raise ValueError(message)
    valid_labels = labels[base_mask & (labels > 0)]
    if valid_labels.size == 0:
        return base_mask
    # bincount is faster than unique for dense small-integer labels.
    counts = np.bincount(valid_labels.astype(np.int64, copy=False))
    largest = int(np.argmax(counts))
    if largest <= 0 or counts[largest] < 20:
        return base_mask
    return base_mask & (labels == largest)


def estimate_residual_phase_screen_from_unwrapped(
    unwrapped_phase: np.ndarray,
    *,
    coherence: np.ndarray | None = None,
    coh_thr: float = 0.2,
    range_degree: int = 2,
    azimuth_degree: int = 1,
    max_samples: int = 80_000,
    connected_components: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate a smooth residual poly screen from unwrapped phase.

    Far-range residual ramps after DEM flatten are often multi-fringe and not
    DEM-shaped (orbit residual / APS).  Fitting a low-order polynomial on
    *unwrapped* phase is far more stable than circular correlation on the
    wrapped field.  The resulting screen is zero-mean over valid samples.

    When ``connected_components`` is provided, the fit is restricted to the
    largest non-zero component so independent unwrap offsets cannot dominate
    the poly and produce a runaway screen.

    Model::

        φ(x, y) ≈ a + b x + c y + d x² [+ e y² if az_degree≥2] [+ f x y]

    Parameters
    ----------
    unwrapped_phase : numpy.ndarray
        Unwrapped phase in radians.
    coherence : numpy.ndarray, optional
        Weights; samples below ``coh_thr`` are ignored.
    coh_thr : float, optional
        Coherence threshold. Default 0.2.
    range_degree, azimuth_degree : int, optional
        Maximum polynomial degree along range (x) and azimuth (y).
        Defaults 2 and 1.
    max_samples : int, optional
        Subsample size for the weighted least-squares fit.
    connected_components : numpy.ndarray, optional
        Integer component labels from the unwrapper.  When set, only the
        largest non-zero component is used for the fit.

    Returns
    -------
    numpy.ndarray
        Phase screen in radians, same shape as ``unwrapped_phase``.

    """
    unw = np.asarray(unwrapped_phase, dtype=np.float64)
    if unw.ndim != 2:
        message = f"unwrapped_phase must be 2-D, got shape {unw.shape}"
        logger.error(message)
        raise ValueError(message)
    height, width = unw.shape
    mask = np.isfinite(unw)
    if coherence is not None:
        coh = np.asarray(coherence)
        mask = mask & np.isfinite(coh) & (coh >= coh_thr)
        weights = np.where(mask, coh.astype(np.float64), 0.0)
    else:
        weights = mask.astype(np.float64)
    screen = np.zeros((height, width), dtype=np.float64)
    if connected_components is not None:
        mask = _largest_connected_component_mask(connected_components, mask)
        weights = np.where(mask, weights, 0.0)
    if int(mask.sum()) < 20:
        return screen

    yy, xx = np.where(mask)
    u = unw[mask]
    w = weights[mask]
    if yy.size > max_samples:
        rng = np.random.default_rng(0)
        sel = rng.choice(yy.size, max_samples, replace=False)
        yy, xx, u, w = yy[sel], xx[sel], u[sel], w[sel]

    cols: list[np.ndarray] = [np.ones(xx.shape, dtype=np.float64)]
    # range terms
    cols.append(xx.astype(np.float64))
    if range_degree >= 2:
        cols.append(xx.astype(np.float64) ** 2)
    if range_degree >= 3:
        cols.append(xx.astype(np.float64) ** 3)
    # azimuth terms
    if azimuth_degree >= 1:
        cols.append(yy.astype(np.float64))
    if azimuth_degree >= 2:
        cols.append(yy.astype(np.float64) ** 2)
    # cross term when both axes contribute
    if range_degree >= 1 and azimuth_degree >= 1:
        cols.append(xx.astype(np.float64) * yy.astype(np.float64))
    design = np.column_stack(cols)
    sw = np.sqrt(np.maximum(w, 0.0))
    coef, *_ = np.linalg.lstsq(design * sw[:, None], u * sw, rcond=None)

    yg, xg = np.mgrid[0:height, 0:width]
    terms: list[np.ndarray] = [np.full(xg.shape, coef[0], dtype=np.float64)]
    k = 1
    terms.append(coef[k] * xg)
    k += 1
    if range_degree >= 2:
        terms.append(coef[k] * xg**2)
        k += 1
    if range_degree >= 3:
        terms.append(coef[k] * xg**3)
        k += 1
    if azimuth_degree >= 1:
        terms.append(coef[k] * yg)
        k += 1
    if azimuth_degree >= 2:
        terms.append(coef[k] * yg**2)
        k += 1
    if range_degree >= 1 and azimuth_degree >= 1:
        terms.append(coef[k] * xg * yg)
    screen = np.sum(terms, axis=0)
    # zero-mean over the fitting mask (full grid, original weights)
    full_mask = mask
    w_full = weights
    if w_full[full_mask].sum() > 0:
        screen -= float(np.average(screen[full_mask], weights=w_full[full_mask]))
    elif int(full_mask.sum()) > 0:
        screen -= float(screen[full_mask].mean())
    return screen


def estimate_residual_phase_screen(
    complex_ifg: np.ndarray,
    *,
    coherence: np.ndarray | None = None,
    coh_thr: float = 0.2,
    range_degree: int = 2,
    azimuth_degree: int = 1,
    unwrapped_phase: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate a smooth residual phase screen (range/azimuth polynomial).

    Prefer ``unwrapped_phase`` when available (stable multi-fringe fit).
    Without unwrap, fall back to binned complex column/row means with
    1-D unwrap + polyfit (same idea as the range p2p diagnostic).

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Complex interferogram (used for shape / complex-only fallback).
    coherence : numpy.ndarray, optional
        Coherence weights; samples below ``coh_thr`` are ignored.
    coh_thr : float, optional
        Coherence threshold. Default 0.2.
    range_degree, azimuth_degree : int, optional
        Polynomial degrees along range and azimuth. Defaults 2 and 1.
    unwrapped_phase : numpy.ndarray, optional
        Unwrapped phase; preferred data source for the fit.

    Returns
    -------
    numpy.ndarray
        Phase screen in radians, same shape as ``complex_ifg`` (zero-mean).

    """
    ifg = np.asarray(complex_ifg)
    if ifg.ndim != 2:
        message = f"complex_ifg must be 2-D, got shape {ifg.shape}"
        logger.error(message)
        raise ValueError(message)
    if unwrapped_phase is not None:
        return estimate_residual_phase_screen_from_unwrapped(
            unwrapped_phase,
            coherence=coherence,
            coh_thr=coh_thr,
            range_degree=range_degree,
            azimuth_degree=azimuth_degree,
        )

    height, width = ifg.shape
    finite = np.isfinite(ifg.real) & np.isfinite(ifg.imag) & (np.abs(ifg) > 0)
    if coherence is not None:
        coh = np.asarray(coherence)
        mask = finite & np.isfinite(coh) & (coh >= coh_thr)
        weights = np.where(mask, coh.astype(np.float64), 0.0)
    else:
        mask = finite
        weights = mask.astype(np.float64)
    screen = np.zeros((height, width), dtype=np.float64)
    if not np.any(mask):
        return screen

    # Binned complex column means → unwrap → polyfit (range).
    n_bins = min(80, width)
    re = np.where(mask, ifg.real.astype(np.float64), 0.0) * weights
    im = np.where(mask, ifg.imag.astype(np.float64), 0.0) * weights
    col_den = weights.sum(axis=0)
    col_ok = col_den > 1e-6
    col_c = np.zeros(width, dtype=np.complex128)
    col_c[col_ok] = (re.sum(axis=0)[col_ok] + 1j * im.sum(axis=0)[col_ok]) / col_den[
        col_ok
    ]
    centers: list[float] = []
    means: list[float] = []
    for i in range(n_bins):
        c0 = int(i * width / n_bins)
        c1 = int((i + 1) * width / n_bins)
        sel = col_ok[c0:c1]
        if int(sel.sum()) < 1:
            continue
        cc = col_c[c0:c1][sel]
        if not np.any(np.abs(cc) > 0):
            continue
        means.append(float(np.angle(np.mean(cc))))
        centers.append(0.5 * (c0 + c1))
    if len(means) >= max(range_degree + 2, 3):
        c_arr = np.asarray(centers, dtype=np.float64)
        un = np.unwrap(np.asarray(means, dtype=np.float64))
        coef_rg = np.polyfit(c_arr, un, int(range_degree))
        model_rg = np.polyval(coef_rg, np.arange(width, dtype=np.float64))
        screen += model_rg[None, :]
        corrected = ifg * np.exp(-1j * model_rg[None, :])
    else:
        corrected = ifg

    # Binned row means of residual → unwrap → linear az.
    n_az = min(40, height)
    re = np.where(mask, corrected.real.astype(np.float64), 0.0) * weights
    im = np.where(mask, corrected.imag.astype(np.float64), 0.0) * weights
    centers = []
    means = []
    for i in range(n_az):
        r0 = int(i * height / n_az)
        r1 = int((i + 1) * height / n_az)
        mm = mask[r0:r1]
        if int(mm.sum()) < 20:
            continue
        block = corrected[r0:r1][mm]
        means.append(float(np.angle(np.mean(np.exp(1j * np.angle(block))))))
        centers.append(0.5 * (r0 + r1))
    if azimuth_degree >= 1 and len(means) >= 3:
        c_arr = np.asarray(centers, dtype=np.float64)
        un = np.unwrap(np.asarray(means, dtype=np.float64))
        coef_az = np.polyfit(c_arr, un, int(min(azimuth_degree, 1)))
        model_az = np.polyval(coef_az, np.arange(height, dtype=np.float64))
        screen += model_az[:, None]

    if np.any(mask):
        w = weights[mask]
        mean = (
            float(np.average(screen[mask], weights=w))
            if w.sum() > 0
            else float(screen[mask].mean())
        )
        screen -= mean
    return screen


def estimate_residual_height_poly_screen(
    unwrapped_phase: np.ndarray,
    height_m: np.ndarray,
    *,
    coherence: np.ndarray | None = None,
    coh_thr: float = 0.3,
    range_degree: int = 2,
    azimuth_degree: int = 1,
    max_samples: int = 80_000,
    connected_components: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate residual phase correlated with DEM height plus a smooth poly.

    After topsApp / geometric flatten, far-range multi-fringe residuals are
    often proportional to ellipsoidal height (baseline residual * DEM error).
    Fitting::

        φ ≈ a + b h + poly(x, y)

    removes both the DEM-shaped residual and a low-order orbit/APS screen.

    Parameters
    ----------
    unwrapped_phase : numpy.ndarray
        Unwrapped phase (rad).
    height_m : numpy.ndarray
        Ellipsoidal height (m), same shape as ``unwrapped_phase``.
    coherence : numpy.ndarray, optional
        Coherence weights.
    coh_thr : float, optional
        Coherence threshold. Default 0.3.
    range_degree, azimuth_degree : int, optional
        Polynomial degrees for the smooth screen. Defaults 2 and 1.
    max_samples : int, optional
        Subsample size for the weighted least-squares fit.
    connected_components : numpy.ndarray, optional
        When set, fit is restricted to the largest non-zero component.

    Returns
    -------
    numpy.ndarray
        Zero-mean phase screen (rad), same shape as ``unwrapped_phase``.

    """
    unw = np.asarray(unwrapped_phase, dtype=np.float64)
    hgt = np.asarray(height_m, dtype=np.float64)
    if unw.shape != hgt.shape:
        message = f"height_m shape {hgt.shape} must match phase shape {unw.shape}"
        logger.error(message)
        raise ValueError(message)
    height, width = unw.shape
    mask = np.isfinite(unw) & np.isfinite(hgt)
    if coherence is not None:
        coh = np.asarray(coherence)
        mask = mask & np.isfinite(coh) & (coh >= coh_thr)
        weights = np.where(mask, coh.astype(np.float64), 0.0)
    else:
        weights = mask.astype(np.float64)
    screen = np.zeros((height, width), dtype=np.float64)
    if connected_components is not None:
        mask = _largest_connected_component_mask(connected_components, mask)
        weights = np.where(mask, weights, 0.0)
    if int(mask.sum()) < 50:
        return screen

    yy, xx = np.where(mask)
    u = unw[mask]
    w = weights[mask]
    h_s = hgt[mask]
    if yy.size > max_samples:
        rng = np.random.default_rng(0)
        sel = rng.choice(yy.size, max_samples, replace=False)
        yy, xx, u, w, h_s = yy[sel], xx[sel], u[sel], w[sel], h_s[sel]

    cols: list[np.ndarray] = [
        np.ones(xx.shape, dtype=np.float64),
        h_s.astype(np.float64),
        xx.astype(np.float64),
    ]
    if range_degree >= 2:
        cols.append(xx.astype(np.float64) ** 2)
    if range_degree >= 3:
        cols.append(xx.astype(np.float64) ** 3)
    if azimuth_degree >= 1:
        cols.append(yy.astype(np.float64))
    if azimuth_degree >= 2:
        cols.append(yy.astype(np.float64) ** 2)
    if range_degree >= 1 and azimuth_degree >= 1:
        cols.append(xx.astype(np.float64) * yy.astype(np.float64))
    design = np.column_stack(cols)
    sw = np.sqrt(np.maximum(w, 0.0))
    coef, *_ = np.linalg.lstsq(design * sw[:, None], u * sw, rcond=None)

    yg, xg = np.mgrid[0:height, 0:width]
    terms: list[np.ndarray] = [
        np.full(xg.shape, coef[0], dtype=np.float64),
        coef[1] * hgt,
        coef[2] * xg,
    ]
    k = 3
    if range_degree >= 2:
        terms.append(coef[k] * xg**2)
        k += 1
    if range_degree >= 3:
        terms.append(coef[k] * xg**3)
        k += 1
    if azimuth_degree >= 1:
        terms.append(coef[k] * yg)
        k += 1
    if azimuth_degree >= 2:
        terms.append(coef[k] * yg**2)
        k += 1
    if range_degree >= 1 and azimuth_degree >= 1:
        terms.append(coef[k] * xg * yg)
    screen = np.sum(terms, axis=0)
    screen = np.where(np.isfinite(hgt), screen, 0.0)
    if weights[mask].sum() > 0:
        screen -= float(np.average(screen[mask], weights=weights[mask]))
    elif int(mask.sum()) > 0:
        screen -= float(screen[mask].mean())
    return screen


def estimate_tiled_residual_height_screen(
    unwrapped_phase: np.ndarray,
    height_m: np.ndarray,
    *,
    coherence: np.ndarray | None = None,
    coh_thr: float = 0.25,
    n_az_tiles: int = 8,
    n_rg_tiles: int = 32,
    overlap_frac: float = 0.3,
    min_tile_samples: int = 150,
    connected_components: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate a spatially adaptive residual screen from local height fits.

    Global low-order height polynomials leave multi-fringe DEM residual
    texture because residual baseline scale varies with range/look angle.
    Overlapping tiles fit::

        φ ≈ a + b ĥ + c ĥ x̂ + d x̂ + e ŷ + f x̂²

    with local normalised coordinates, then blend with raised-cosine weights.

    Parameters
    ----------
    unwrapped_phase : numpy.ndarray
        Unwrapped phase (rad).
    height_m : numpy.ndarray
        Ellipsoidal height (m), same shape.
    coherence : numpy.ndarray, optional
        Coherence weights.
    coh_thr : float, optional
        Coherence threshold. Default 0.25.
    n_az_tiles, n_rg_tiles : int, optional
        Tile counts along azimuth and range. Defaults 8 and 32.
    overlap_frac : float, optional
        Fractional overlap of each tile for blending. Default 0.3.
    min_tile_samples : int, optional
        Skip tiles with fewer valid samples.
    connected_components : numpy.ndarray, optional
        When set, restrict the fit mask to the largest non-zero component.

    Returns
    -------
    numpy.ndarray
        Blended residual phase screen (rad).

    """
    unw = np.asarray(unwrapped_phase, dtype=np.float64)
    hgt = np.asarray(height_m, dtype=np.float64)
    if unw.shape != hgt.shape:
        message = f"height_m shape {hgt.shape} must match phase shape {unw.shape}"
        logger.error(message)
        raise ValueError(message)
    height, width = unw.shape
    mask = np.isfinite(unw) & np.isfinite(hgt)
    if coherence is not None:
        coh = np.asarray(coherence, dtype=np.float64)
        mask = mask & np.isfinite(coh) & (coh >= coh_thr)
        weights = np.where(mask, coh, 0.0)
    else:
        coh = np.ones_like(unw)
        weights = mask.astype(np.float64)
    if connected_components is not None:
        mask = _largest_connected_component_mask(connected_components, mask)
        weights = np.where(mask, weights, 0.0)
    screen_acc = np.zeros((height, width), dtype=np.float64)
    weight_acc = np.zeros((height, width), dtype=np.float64)
    if int(mask.sum()) < min_tile_samples:
        return screen_acc

    n_az = max(int(n_az_tiles), 1)
    n_rg = max(int(n_rg_tiles), 1)
    overlap = float(np.clip(overlap_frac, 0.0, 0.49))

    for ia in range(n_az):
        for ir in range(n_rg):
            r0 = int(ia * height / n_az)
            r1 = int((ia + 1) * height / n_az)
            c0 = int(ir * width / n_rg)
            c1 = int((ir + 1) * width / n_rg)
            pr = int((r1 - r0) * overlap)
            pc = int((c1 - c0) * overlap)
            r0e = max(0, r0 - pr)
            r1e = min(height, r1 + pr)
            c0e = max(0, c0 - pc)
            c1e = min(width, c1 + pc)
            mm = mask[r0e:r1e, c0e:c1e]
            n_samp = int(mm.sum())
            if n_samp < min_tile_samples:
                continue
            uu = unw[r0e:r1e, c0e:c1e][mm]
            hh = hgt[r0e:r1e, c0e:c1e][mm]
            ww = weights[r0e:r1e, c0e:c1e][mm]
            ry, rx = np.where(mm)
            rx_std = float(np.std(rx)) if rx.size else 1.0
            ry_std = float(np.std(ry)) if ry.size else 1.0
            hh_std = float(np.std(hh)) if hh.size else 1.0
            rx_std = max(rx_std, 1.0)
            ry_std = max(ry_std, 1.0)
            hh_std = max(hh_std, 1.0)
            rx_mean = float(np.mean(rx))
            ry_mean = float(np.mean(ry))
            hh_mean = float(np.mean(hh))
            rxn = (rx - rx_mean) / rx_std
            ryn = (ry - ry_mean) / ry_std
            hhn = (hh - hh_mean) / hh_std
            design = np.column_stack(
                [
                    np.ones(n_samp, dtype=np.float64),
                    hhn,
                    hhn * rxn,
                    rxn,
                    ryn,
                    rxn**2,
                ]
            )
            sw = np.sqrt(np.maximum(ww, 1e-6))
            coef, *_ = np.linalg.lstsq(design * sw[:, None], uu * sw, rcond=None)
            th, tw = r1e - r0e, c1e - c0e
            yg, xg = np.mgrid[0:th, 0:tw]
            xn = (xg - rx_mean) / rx_std
            yn = (yg - ry_mean) / ry_std
            hn = (hgt[r0e:r1e, c0e:c1e] - hh_mean) / hh_std
            tile_screen = (
                coef[0]
                + coef[1] * hn
                + coef[2] * hn * xn
                + coef[3] * xn
                + coef[4] * yn
                + coef[5] * xn**2
            )
            # Uniform weights with core=1; only ramp in the *overlap extension*
            # outside the core tile (never on the outer image boundary alone).
            wt = np.zeros((th, tw), dtype=np.float64)
            cr0 = r0 - r0e
            cr1 = r1 - r0e
            cc0 = c0 - c0e
            cc1 = c1 - c0e
            wt[cr0:cr1, cc0:cc1] = 1.0
            if pr > 0 and cr0 > 0:
                t = np.linspace(0.0, 1.0, cr0, endpoint=False)
                ramp = 0.5 - 0.5 * np.cos(np.pi * t)
                wt[:cr0, cc0:cc1] = np.maximum(wt[:cr0, cc0:cc1], ramp[:, None])
            if pr > 0 and cr1 < th:
                n = th - cr1
                t = np.linspace(0.0, 1.0, n, endpoint=False)
                ramp = 0.5 - 0.5 * np.cos(np.pi * t)
                wt[cr1:, cc0:cc1] = np.maximum(wt[cr1:, cc0:cc1], ramp[::-1, None])
            if pc > 0 and cc0 > 0:
                t = np.linspace(0.0, 1.0, cc0, endpoint=False)
                ramp = 0.5 - 0.5 * np.cos(np.pi * t)
                wt[cr0:cr1, :cc0] = np.maximum(wt[cr0:cr1, :cc0], ramp[None, :])
            if pc > 0 and cc1 < tw:
                n = tw - cc1
                t = np.linspace(0.0, 1.0, n, endpoint=False)
                ramp = 0.5 - 0.5 * np.cos(np.pi * t)
                wt[cr0:cr1, cc1:] = np.maximum(wt[cr0:cr1, cc1:], ramp[None, ::-1])
            finite_h = np.isfinite(hgt[r0e:r1e, c0e:c1e])
            wt = np.where(finite_h, wt, 0.0)
            screen_acc[r0e:r1e, c0e:c1e] += tile_screen * wt
            weight_acc[r0e:r1e, c0e:c1e] += wt

    screen = np.zeros((height, width), dtype=np.float64)
    ok = weight_acc > 0
    screen[ok] = screen_acc[ok] / weight_acc[ok]
    if weights[mask].sum() > 0:
        screen -= float(np.average(screen[mask], weights=weights[mask]))
    elif int(mask.sum()) > 0:
        screen -= float(screen[mask].mean())
    return screen


def remove_residual_phase_screen(
    complex_ifg: np.ndarray,
    phase_screen: np.ndarray,
) -> np.ndarray:
    """Multiply interferogram by ``exp(-i * phase_screen)``."""
    ifg = np.asarray(complex_ifg)
    screen = np.asarray(phase_screen, dtype=np.float64)
    if ifg.shape != screen.shape:
        message = f"phase_screen shape {screen.shape} must match ifg shape {ifg.shape}"
        logger.error(message)
        raise ValueError(message)
    return (ifg * np.exp(-1j * screen)).astype(ifg.dtype, copy=False)


def apply_residual_phase_screen_to_products(
    *,
    unwrapped_phase: np.ndarray,
    complex_ifg: np.ndarray | None = None,
    coherence: np.ndarray | None = None,
    coh_thr: float = 0.2,
    range_degree: int = 2,
    azimuth_degree: int = 1,
    min_span_rad: float = 1.0,
    max_span_rad: float = 40.0,
    min_std_reduction: float = 0.15,
    connected_components: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, float, bool]:
    """Fit residual poly screen from unwrap and apply to phase products.

    Application gates (all must pass for the correction to be written back):

    1. Screen peak-to-peak ``span >= min_span_rad``.
    2. Variance reduction on the fit mask of at least ``min_std_reduction``
       (relative).  Large true residual ramps (many cycles of orbit/APS) are
       kept when they substantially flatten the field.
    3. If ``span > max_span_rad`` *and* no ``connected_components`` was
       provided, skip (unrestricted multi-component poly can invent a huge
       ramp from independent unwrap islands).  With a largest-component mask
       the high span is allowed when variance improves.

    Parameters
    ----------
    unwrapped_phase : numpy.ndarray
        Unwrapped phase to fit and correct.
    complex_ifg : numpy.ndarray, optional
        Flattened complex interferogram to re-phase.
    coherence : numpy.ndarray, optional
        Coherence weights for the fit.
    coh_thr : float, optional
        Coherence threshold.
    range_degree, azimuth_degree : int, optional
        Polynomial degrees.
    min_span_rad : float, optional
        Skip correction when estimated screen peak-to-peak is below this.
    max_span_rad : float, optional
        Soft ceiling: screens larger than this require either a connected-
        component mask or are rejected as multi-component runaways.
    min_std_reduction : float, optional
        Minimum relative reduction of unwrapped std on the fit mask
        (``1 - after/before``). Default 0.15 (15 %).
    connected_components : numpy.ndarray, optional
        Unwrap component labels; fit is restricted to the largest component.

    Returns
    -------
    unwrapped_corr, complex_corr, screen, span, applied
        Corrected products, the screen, its peak-to-peak span, and whether
        the correction was applied.  When skipped, inputs are returned
        unchanged and ``applied`` is False.

    """
    unw = np.asarray(unwrapped_phase)
    screen = estimate_residual_phase_screen_from_unwrapped(
        unw,
        coherence=coherence,
        coh_thr=coh_thr,
        range_degree=range_degree,
        azimuth_degree=azimuth_degree,
        connected_components=connected_components,
    )
    span = float(np.nanmax(screen) - np.nanmin(screen)) if screen.size else 0.0
    fit_mask = np.isfinite(unw)
    if coherence is not None:
        coh = np.asarray(coherence)
        fit_mask = fit_mask & np.isfinite(coh) & (coh >= coh_thr)
    if connected_components is not None:
        fit_mask = _largest_connected_component_mask(connected_components, fit_mask)

    applied = False
    if span < min_span_rad or int(fit_mask.sum()) < 20:
        return (
            np.asarray(unwrapped_phase),
            None if complex_ifg is None else np.asarray(complex_ifg),
            screen,
            span,
            applied,
        )

    before = float(np.std(unw[fit_mask]))
    after = float(np.std((unw - screen)[fit_mask]))
    improves = before > 0.0 and (1.0 - after / before) >= min_std_reduction
    if span > max_span_rad and connected_components is None:
        logger.warning(
            "residual phase screen span=%.2f rad exceeds max_span_rad=%.2f "
            "without connected_components; skipping (multi-component risk)",
            span,
            max_span_rad,
        )
        return (
            np.asarray(unwrapped_phase),
            None if complex_ifg is None else np.asarray(complex_ifg),
            screen,
            span,
            applied,
        )
    if not improves:
        if span > max_span_rad:
            logger.warning(
                "residual phase screen span=%.2f rad does not reduce std "
                "(before=%.3f after=%.3f); skipping",
                span,
                before,
                after,
            )
        return (
            np.asarray(unwrapped_phase),
            None if complex_ifg is None else np.asarray(complex_ifg),
            screen,
            span,
            applied,
        )

    unw_corr = np.asarray(unwrapped_phase, dtype=np.float32) - screen.astype(np.float32)
    z_corr = None
    if complex_ifg is not None:
        z_corr = remove_residual_phase_screen(complex_ifg, screen)
    applied = True
    return unw_corr, z_corr, screen, span, applied


def copernicus_glo30_dem(
    latitude_deg: float,
    longitude_deg: float,
    *,
    base_path: str | Path,
    device: str,
) -> RasterDEM:
    """Return a :class:`RasterDEM` for the Copernicus GLO-30 tile covering a coordinate.

    Expects the standard COG tile naming convention:
    ``Copernicus_DSM_COG_10_{N|S}{lat:02d}_00_{E|W}{lon:03d}_DEM.tif``.

    Parameters
    ----------
    latitude_deg, longitude_deg : float
        Geodetic coordinate in degrees.
    base_path : str or Path
        Directory containing Copernicus GLO-30 COG tiles.
    device : str
        Admitted DEM device identity (``cpu`` / ``cuda`` / ``cuda:N``).

    Returns
    -------
    RasterDEM
        DEM sampler for the tile.

    Raises
    ------
    FileNotFoundError
        If the expected tile file does not exist.

    """
    base = Path(base_path)
    # This helper accepts an already materialized local tile.  Keep the
    # filename admission here so flattening does not compose the legacy
    # network-facing DEM manager into the public geometry path.
    lat_tile = int(np.floor(latitude_deg))
    lon_tile = int(np.floor(longitude_deg))
    ns = "N" if lat_tile >= 0 else "S"
    ew = "E" if lon_tile >= 0 else "W"
    tile_dir = f"{ns}{abs(lat_tile):02d}_{ew}{abs(lon_tile):03d}"
    filename = (
        f"Copernicus_DSM_COG_10_{ns}{abs(lat_tile):02d}_00_"
        f"{ew}{abs(lon_tile):03d}_00_DEM.tif"
    )
    candidates = (
        base / filename,
        base / tile_dir / filename,
        *base.glob(f"**/{filename}"),
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        message = f"Copernicus GLO-30 tile not found for {filename} under {base}"
        logger.error(message)
        raise FileNotFoundError(message)
    del device
    return DEM.from_raster(path, vertical_datum="egm2008")
