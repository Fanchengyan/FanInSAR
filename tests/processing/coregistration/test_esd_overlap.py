"""Tests for burst-overlap ESD azimuth estimation."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.coregistration.esd_overlap import (
    OverlapESDInput,
    OverlapESDRecord,
    aggregate_overlap_esd,
    estimate_overlap_esd,
)
from faninsar.processing.coregistration.tops.deramp import (
    TOPSCarrierModel,
    carrier_phase_at_points,
)
from faninsar.processing.errors import InvalidProcessingStateError

# Full-burst line count. Overlap strips sit at opposite TOPS edges so the
# steering term dominates R_B; a crop-local centre_row collapses that swing.
_NATIVE_HEIGHT = 1500
_N_LINES = 40
_N_SAMPLES = 60
_TOP_ROW0 = 1460
_BOT_ROW0 = 0
_COL0 = 80
_INJECTED_SHIFT_PX = 0.02


def _carrier(*, dc0: float = 80.0, fm0: float = -2200.0) -> TOPSCarrierModel:
    """Return a Sentinel-1-like TOPS carrier (steering term is required)."""
    return TOPSCarrierModel(
        radar_frequency_hz=5.405e9,
        slant_range_time0_s=0.00533,
        range_sampling_rate_hz=6.434e7,
        azimuth_time_interval_s=0.002055,
        doppler_centroid_hz=(dc0, -40.0, 8.0),
        doppler_t0_s=0.00534,
        fm_rate_hz_s=(fm0, 4.5e5, -7.9e7),
        fm_t0_s=0.00533,
        burst_sensing_time_s=0.0,
        burst_start_slant_range_time_s=0.00533,
        azimuth_steering_rate_hz_s=6500.0,
    )


def _carrier_phase(
    model: TOPSCarrierModel,
    *,
    row0: int,
    col0: int,
    native_height: int,
    n_lines: int = _N_LINES,
    n_samples: int = _N_SAMPLES,
    azimuth_shift_px: float = 0.0,
) -> np.ndarray:
    """Evaluate the analytic TOPS carrier on a burst-absolute overlap crop."""
    rows = np.arange(n_lines, dtype=np.float64)[:, None] + float(row0)
    rows = rows - float(azimuth_shift_px)
    cols = np.arange(n_samples, dtype=np.float64)[None, :] + float(col0)
    rows, cols = np.broadcast_arrays(rows, cols)
    return carrier_phase_at_points(
        model,
        rows,
        cols,
        centre_row=float(native_height // 2),
        dtype=np.float64,
    )


def _slc_from_phase(phase: np.ndarray, ground: np.ndarray) -> np.ndarray:
    """Unit-amplitude original-domain SLC with a shared ground phase."""
    return np.exp(1j * (ground + phase)).astype(np.complex64)


def _zero_offsets(
    n_lines: int = _N_LINES,
    n_samples: int = _N_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Geometry-only offset crops (zeros: residual ESD lives in the SLCs)."""
    az = np.zeros((n_lines, n_samples), dtype=np.float64)
    rg = np.zeros((n_lines, n_samples), dtype=np.float64)
    return az, rg


def _poly_eval(coefficients: tuple[float, ...], x: np.ndarray | float) -> np.ndarray:
    """Evaluate ``c0 + c1 x + c2 x^2 + ...`` (test-local copy of the carrier poly)."""
    x64 = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x64, dtype=np.float64)
    power = np.ones_like(x64, dtype=np.float64)
    for coeff in coefficients:
        result = result + float(coeff) * power
        power = power * x64
    return result


def _doppler_centroid_hz(model: TOPSCarrierModel, col: float) -> float:
    """Doppler-centroid polynomial only (no steering / FM term)."""
    tau = model.slant_range_time0_s + float(col) / model.range_sampling_rate_hz
    value = _poly_eval(model.doppler_centroid_hz, tau - model.doppler_t0_s)
    return float(np.asarray(value))


def _make_overlap_input(
    *,
    injected_shift_px: float = _INJECTED_SHIFT_PX,
    top_row0: int = _TOP_ROW0,
    bot_row0: int = _BOT_ROW0,
    top_native_height: int = _NATIVE_HEIGHT,
    bot_native_height: int = _NATIVE_HEIGHT,
    ref_native_height: int | None = _NATIVE_HEIGHT,
    include_ref_carriers: bool = True,
    noise_sigma: float = 0.0,
) -> OverlapESDInput:
    """Build original-domain overlap patches with a known secondary azimuth shift."""
    top_model = _carrier(dc0=90.0, fm0=-2100.0)
    bot_model = _carrier(dc0=70.0, fm0=-2300.0)
    az = np.arange(_N_LINES, dtype=np.float64)[:, None]
    rg = np.arange(_N_SAMPLES, dtype=np.float64)[None, :]
    # Common ground phase cancels in the double-difference.
    ground = 0.35 * rg + 0.08 * az

    ref_top = _slc_from_phase(
        _carrier_phase(
            top_model,
            row0=_TOP_ROW0,
            col0=_COL0,
            native_height=_NATIVE_HEIGHT,
        ),
        ground,
    )
    ref_bot = _slc_from_phase(
        _carrier_phase(
            bot_model,
            row0=_BOT_ROW0,
            col0=_COL0,
            native_height=_NATIVE_HEIGHT,
        ),
        ground,
    )
    sec_top = _slc_from_phase(
        _carrier_phase(
            top_model,
            row0=_TOP_ROW0,
            col0=_COL0,
            native_height=_NATIVE_HEIGHT,
            azimuth_shift_px=injected_shift_px,
        ),
        ground,
    )
    sec_bot = _slc_from_phase(
        _carrier_phase(
            bot_model,
            row0=_BOT_ROW0,
            col0=_COL0,
            native_height=_NATIVE_HEIGHT,
            azimuth_shift_px=injected_shift_px,
        ),
        ground,
    )
    if noise_sigma > 0.0:
        rng = np.random.default_rng(0)

        def _add_noise(arr: np.ndarray) -> np.ndarray:
            noise = noise_sigma * (
                rng.normal(size=arr.shape) + 1j * rng.normal(size=arr.shape)
            )
            return (arr + noise).astype(np.complex64)

        ref_top = _add_noise(ref_top)
        ref_bot = _add_noise(ref_bot)
        sec_top = _add_noise(sec_top)
        sec_bot = _add_noise(sec_bot)

    kwargs: dict[str, object] = {
        "ref_top": ref_top,
        "ref_bot": ref_bot,
        "sec_top": sec_top,
        "sec_bot": sec_bot,
        "sec_top_carrier": top_model,
        "sec_bot_carrier": bot_model,
        "top_row0": top_row0,
        "bot_row0": bot_row0,
        "top_col0": _COL0,
        "bot_col0": _COL0,
        "top_native_height": top_native_height,
        "bot_native_height": bot_native_height,
        "top_offsets": _zero_offsets(),
        "bot_offsets": _zero_offsets(),
        "top_index": 3,
        "bottom_index": 4,
    }
    if include_ref_carriers:
        kwargs.update(
            {
                "ref_top_carrier": top_model,
                "ref_bot_carrier": bot_model,
                "ref_top_row0": _TOP_ROW0,
                "ref_bot_row0": _BOT_ROW0,
                "ref_top_col0": _COL0,
                "ref_bot_col0": _COL0,
                "ref_top_native_height": ref_native_height,
                "ref_bot_native_height": ref_native_height,
            }
        )
    return OverlapESDInput(**kwargs)  # type: ignore[arg-type]


def test_overlap_esd_recovers_known_subpixel_azimuth_shift() -> None:
    """Analytic TOPS carriers recover an injected sub-pixel azimuth shift."""
    data = _make_overlap_input(injected_shift_px=_INJECTED_SHIFT_PX)
    result = estimate_overlap_esd(data)
    assert result is not None
    assert isinstance(result, OverlapESDRecord)
    assert result.top_index == 3
    assert result.bottom_index == 4
    assert result.n_valid > 0
    assert result.coherence >= 0.85
    assert result.azimuth_shift_px == pytest.approx(_INJECTED_SHIFT_PX, abs=0.005)
    # Independent 1-px finite difference of the forward carrier (not the estimator).
    r_top = np.median(
        _carrier_phase(
            data.sec_top_carrier,
            row0=_TOP_ROW0,
            col0=_COL0,
            native_height=_NATIVE_HEIGHT,
            azimuth_shift_px=-1.0,
        )
        - _carrier_phase(
            data.sec_top_carrier,
            row0=_TOP_ROW0,
            col0=_COL0,
            native_height=_NATIVE_HEIGHT,
        )
    )
    r_bot = np.median(
        _carrier_phase(
            data.sec_bot_carrier,
            row0=_BOT_ROW0,
            col0=_COL0,
            native_height=_NATIVE_HEIGHT,
            azimuth_shift_px=-1.0,
        )
        - _carrier_phase(
            data.sec_bot_carrier,
            row0=_BOT_ROW0,
            col0=_COL0,
            native_height=_NATIVE_HEIGHT,
        )
    )
    r_b = float(r_top - r_bot)
    assert result.response_rb_rad_px == pytest.approx(abs(r_b), rel=0.05)
    assert result.response_consistency < 0.05


def test_crop_local_native_height_does_not_recover_shift() -> None:
    """A crop-local centre_row (PR 28139 class) must not recover the true shift."""
    injected = _INJECTED_SHIFT_PX
    good = estimate_overlap_esd(_make_overlap_input(injected_shift_px=injected))
    assert good is not None
    assert good.azimuth_shift_px == pytest.approx(injected, abs=0.005)

    with pytest.raises(InvalidProcessingStateError, match="native_height"):
        estimate_overlap_esd(
            _make_overlap_input(
                injected_shift_px=injected,
                top_native_height=_N_LINES,
                bot_native_height=_N_LINES,
                include_ref_carriers=False,
            )
        )

    # Same crop evaluated as a standalone burst (row0 = 0, centre = crop//2).
    # Data still carry the true burst-absolute carrier, so R_B collapses.
    wrong = estimate_overlap_esd(
        _make_overlap_input(
            injected_shift_px=injected,
            top_row0=0,
            bot_row0=0,
            top_native_height=_N_LINES,
            bot_native_height=_N_LINES,
            include_ref_carriers=False,
        )
    )
    if wrong is None:
        return
    assert abs(wrong.azimuth_shift_px - injected) > 0.05
    assert abs(wrong.response_rb_rad_px - good.response_rb_rad_px) > 10.0


def test_ref_carrier_requires_ref_native_height() -> None:
    """R_A evaluation must not fall back to a crop-local centre_row (F5)."""
    data = _make_overlap_input(ref_native_height=None)
    with pytest.raises(InvalidProcessingStateError, match="native_height"):
        estimate_overlap_esd(data)


def test_doppler_centroid_only_response_does_not_recover_shift() -> None:
    """R built from f_dc alone (no steering) cannot recover the injected shift."""
    injected = _INJECTED_SHIFT_PX
    data = _make_overlap_input(injected_shift_px=injected)
    result = estimate_overlap_esd(data)
    assert result is not None
    assert result.azimuth_shift_px == pytest.approx(injected, abs=0.005)

    col_mid = float(_COL0 + _N_SAMPLES / 2.0)
    f_top = _doppler_centroid_hz(data.sec_top_carrier, col_mid)
    f_bot = _doppler_centroid_hz(data.sec_bot_carrier, col_mid)
    r_dc = 2.0 * np.pi * data.sec_top_carrier.azimuth_time_interval_s * (f_top - f_bot)
    assert abs(r_dc) < 0.1 * result.response_rb_rad_px

    ifg_top = data.ref_top * np.conjugate(data.sec_top)
    ifg_bot = data.ref_bot * np.conjugate(data.sec_bot)
    dd = ifg_top * np.conjugate(ifg_bot)
    amp = np.abs(dd)
    unit = np.divide(dd, amp, out=np.zeros_like(dd), where=amp > 0)
    phi = float(np.angle(np.sum(unit)))
    if abs(r_dc) < 1e-6:
        return
    doppler_only_shift = phi / r_dc
    assert doppler_only_shift != pytest.approx(injected, abs=0.05)


def test_below_coherence_gate_returns_none_not_zero_shift() -> None:
    """No qualifying sample omits the observation instead of stuffing 0."""
    data = _make_overlap_input(injected_shift_px=_INJECTED_SHIFT_PX, noise_sigma=5.0)
    result = estimate_overlap_esd(data, coherence_threshold=0.85)
    assert result is None


def test_aggregate_empty_records_returns_none() -> None:
    """An empty overlap list cannot become a 0-shift aggregate."""
    assert aggregate_overlap_esd([]) is None


def test_aggregate_single_record_respects_sigma_floor() -> None:
    """A single-overlap arc stays invert-legal via the sigma floor."""
    record = OverlapESDRecord(
        top_index=0,
        bottom_index=1,
        azimuth_shift_px=0.012,
        coherence=0.92,
        n_valid=40,
        response_rb_rad_px=55.0,
        response_consistency=0.01,
    )
    aggregate = aggregate_overlap_esd([record], sigma_floor_px=0.01)
    assert aggregate is not None
    assert aggregate.azimuth_shift_px == pytest.approx(0.012)
    assert aggregate.sigma_px >= 0.01
    assert aggregate.n_overlaps == 1
    assert aggregate.n_valid == 40
    assert aggregate.method == "overlap"


def test_aggregate_weights_by_coherence_and_sample_count() -> None:
    """Pair-level azimuth is the coherence times n_valid weighted mean."""
    weak = OverlapESDRecord(
        top_index=0,
        bottom_index=1,
        azimuth_shift_px=0.04,
        coherence=0.86,
        n_valid=10,
        response_rb_rad_px=50.0,
        response_consistency=0.02,
    )
    strong = OverlapESDRecord(
        top_index=1,
        bottom_index=2,
        azimuth_shift_px=0.01,
        coherence=0.95,
        n_valid=90,
        response_rb_rad_px=52.0,
        response_consistency=0.01,
    )
    aggregate = aggregate_overlap_esd([weak, strong])
    assert aggregate is not None
    w_weak = 0.86 * 10
    w_strong = 0.95 * 90
    expected = (w_weak * 0.04 + w_strong * 0.01) / (w_weak + w_strong)
    assert aggregate.azimuth_shift_px == pytest.approx(expected)
    assert aggregate.n_overlaps == 2
    assert aggregate.n_valid == 100
