"""Tests for TOPS deramp/reramp operators."""

from __future__ import annotations

import numpy as np

from faninsar.processing.coregistration.tops import (
    TOPSCarrierModel,
    deramp,
    deramp_reramp_roundtrip_error,
    reramp,
    tops_carrier_phase,
)


def _model() -> TOPSCarrierModel:
    return TOPSCarrierModel(
        radar_frequency_hz=5.405e9,
        slant_range_time0_s=0.00533,
        range_sampling_rate_hz=6.434e7,
        azimuth_time_interval_s=0.002055,
        doppler_centroid_hz=(100.0, -50.0, 10.0),
        doppler_t0_s=0.00534,
        fm_rate_hz_s=(-2300.0, 4.5e5, -7.9e7),
        fm_t0_s=0.00533,
        burst_sensing_time_s=0.0,
        burst_start_slant_range_time_s=0.00533,
        azimuth_steering_rate_hz_s=6500.0,
    )


def test_deramp_reramp_roundtrip_within_tolerance() -> None:
    """Deramp then reramp recovers complex samples to 1e-6 relative error."""
    model = _model()
    rng = np.random.default_rng(0)
    samples = (rng.normal(size=(32, 64)) + 1j * rng.normal(size=(32, 64))).astype(
        np.complex64
    )
    # Apply a synthetic carrier then remove/restore it.
    phase = tops_carrier_phase(model, 32, 64)
    modulated = (samples * np.exp(1j * phase)).astype(np.complex64)
    error = deramp_reramp_roundtrip_error(modulated, model)
    assert error <= 1e-6


def test_deramp_centers_spectrum_energy() -> None:
    """Deramping a pure TOPS carrier reduces low-frequency residual phase."""
    model = _model()
    phase = tops_carrier_phase(model, 32, 64)
    carrier = np.exp(1j * phase).astype(np.complex64)
    deramped = deramp(carrier, model)
    # Ideal deramp of pure carrier yields near-constant complex ones.
    residual = np.angle(deramped * np.conjugate(deramped[0, 0]))
    assert float(np.max(np.abs(residual))) < 1e-5
    restored = reramp(deramped, model)
    np.testing.assert_allclose(restored, carrier, rtol=0, atol=1e-5)


def _crop_model(centre: float | None) -> TOPSCarrierModel:
    """Realistic IW1-like carrier, optionally carrying the physical centre."""
    return TOPSCarrierModel(
        radar_frequency_hz=5.405e9,
        slant_range_time0_s=0.0053340,
        range_sampling_rate_hz=6.4345238e7,
        azimuth_time_interval_s=0.0020555563,
        doppler_centroid_hz=(100.0, -50.0, 10.0),
        doppler_t0_s=0.00534,
        fm_rate_hz_s=(-2300.0, 4.5e5, -7.9e7),
        fm_t0_s=0.0053340,
        burst_sensing_time_s=0.0,
        burst_start_slant_range_time_s=0.005334,
        azimuth_steering_rate_hz_s=6500.0,
        azimuth_centre_row=centre,
    )


def _fractional_error(
    model_truth_centre: float = 374.0,
    model_property: float | None = None,
) -> dict[float, dict[str, float]]:
    """Fold-phase error of crop-deramp + fractional resample + reramp.

    The scene is the analytic envelope ``B(n) = exp(i * 0.03 * n)`` times the
    TOPS carrier built at the physical mid-burst centre, so the fractional
    truth ``B(q)`` is exact and the only error measured is the sampler/kernel
    error of the crop-path carrier handling.
    """
    from faninsar.processing.coregistration.tops.deramp import (
        carrier_phase_at_points,
        deramp,
    )
    from faninsar.processing.geocoding.coordinates import RadarCoordinates
    from faninsar.processing.geocoding.geo_resample import sample

    height, ncol = 1116, 16
    rows = np.broadcast_to(np.arange(height, dtype=np.float64)[:, None], (height, ncol))
    cols = np.broadcast_to(np.arange(ncol, dtype=np.float64)[None, :], (height, ncol))
    truth_phase = carrier_phase_at_points(
        _crop_model(None),
        rows,
        cols,
        centre_row=model_truth_centre,
        dtype=np.float64,
    )
    envelope = np.exp(1j * 0.03 * rows)
    slc = (envelope * np.exp(1j * truth_phase)).astype(np.complex64)
    model_used = _crop_model(model_property)
    centre = (
        model_used.azimuth_centre_row
        if model_used.azimuth_centre_row is not None
        else float(height // 2)
    )

    # Residual azimuth frequency left by the used centre (cycles per pixel).
    def _slope(centre: float) -> np.ndarray:
        plus = carrier_phase_at_points(
            _crop_model(None), rows + 0.5, cols, centre_row=centre, dtype=np.float64
        )
        minus = carrier_phase_at_points(
            _crop_model(None), rows - 0.5, cols, centre_row=centre, dtype=np.float64
        )
        return plus - minus

    d_true = _slope(model_truth_centre)
    d_used = _slope(centre)
    resid = float(np.median((d_true - d_used) / (2.0 * np.pi)))
    n_fold = int(np.rint(resid))
    out = {"resid_cycles_per_px": resid, "n_fold": float(n_fold)}
    inner = slice(64, height - 64)
    two_pi = 2.0 * np.pi
    for frac in (0.25, 0.75):
        q = rows[inner] + frac
        cols_inner = cols[inner]
        coords = RadarCoordinates(
            q, cols_inner, np.ones((height - 128, ncol), dtype=bool)
        )
        der = deramp(slc, model_used)
        got, ok = sample(der, coords, kind="complex", device="cpu")
        phase_used = carrier_phase_at_points(
            model_used, q, cols_inner, centre_row=centre, dtype=np.float64
        )
        est = np.asarray(got, dtype=np.complex128) * np.exp(1j * phase_used)
        truth_phase_q = carrier_phase_at_points(
            _crop_model(None),
            q,
            cols_inner,
            centre_row=model_truth_centre,
            dtype=np.float64,
        )
        truth_q = np.exp(1j * 0.03 * q) * np.exp(1j * truth_phase_q)
        cross = (est * np.conj(truth_q))[np.asarray(ok, dtype=bool)]
        ang = np.angle(cross)
        out[frac] = {
            "mean_phase_rad": float(np.angle(np.exp(1j * ang).mean())),
            "phase_rmse_rad": float(
                np.sqrt(np.mean(((ang + np.pi) % (2 * np.pi) - np.pi) ** 2))
            ),
            "coherence": float(
                np.abs(cross.mean())
                / np.sqrt(np.mean(np.abs(est) ** 2) * np.mean(np.abs(truth_q) ** 2))
            ),
            "expected_fold_rad": float(
                np.angle(np.exp(1j * (-two_pi * n_fold * frac)))
            ),
        }
    return out


def test_cropped_window_fractional_resample_model_centre_recovers_truth() -> None:
    """Model centre: deramp + fractional resample + reramp recovers phase."""
    errs = _fractional_error(model_truth_centre=374.0, model_property=374.0)
    assert errs["n_fold"] == 0.0
    for frac in (0.25, 0.75):
        res = errs[frac]
        # Compare against the analytic envelope at the fractional coordinate;
        # the remaining Lanczos interpolation error must stay below 1 mrad.
        assert abs(res["mean_phase_rad"]) < 1e-3, (frac, res)
        assert res["phase_rmse_rad"] < 1e-3, (frac, res)
        assert res["coherence"] > 0.999, (frac, res)


def test_cropped_window_fractional_resample_legacy_centre_folds() -> None:
    """Legacy window centre: the band-limited signal aliases by one fold."""
    errs = _fractional_error(model_truth_centre=374.0, model_property=None)
    assert errs["n_fold"] == 1.0
    for frac in (0.25, 0.75):
        res = errs[frac]
        # The sampled phase equals the analytic truth rotated by exactly the
        # -2*pi*k*f fold (k = 1 fold of the 1.32 cycles/pixel residual), plus
        # the same sub-0.02 rad kernel bias measured on the model-centre path.
        bias = abs(
            float(
                np.angle(
                    np.exp(1j * (res["mean_phase_rad"] - res["expected_fold_rad"]))
                )
            )
        )
        assert bias < 0.05, (frac, res)
        assert abs(res["mean_phase_rad"]) > 1.0, (frac, res)
        assert res["coherence"] > 0.999, (frac, res)


def test_model_centre_wins_over_explicit_legacy_argument() -> None:
    """The model property overrides a knowingly-legacy window centre."""
    from faninsar.processing.coregistration.tops.deramp import carrier_phase_at_points

    rows = np.arange(1116, dtype=np.float64)[:, None] * np.ones((1, 8))
    cols = np.arange(8, dtype=np.float64)[None, :] * np.ones((1116, 1))
    model = _crop_model(374.0)
    with_property = carrier_phase_at_points(
        model, rows, cols, centre_row=558.0, dtype=np.float64
    )
    expected = carrier_phase_at_points(
        model, rows, cols, centre_row=374.0, dtype=np.float64
    )
    np.testing.assert_allclose(with_property, expected)


def test_torch_carrier_phase_resolves_model_centre_cpu_parity() -> None:
    """Torch backend resolves the model centre exactly like the CPU path."""
    from faninsar.processing.coregistration.tops.deramp import carrier_phase_at_points
    from faninsar.processing.runtime.torch_kernels import carrier_phase_at_points_torch

    rows = (np.arange(1116, dtype=np.float64)[:, None] + 0.25) * np.ones((1, 8))
    cols = (np.arange(8, dtype=np.float64)[None, :] + 0.5) * np.ones((1116, 1))
    model = _crop_model(374.0)
    cpu = carrier_phase_at_points(model, rows, cols, centre_row=None, dtype=np.float64)
    torch_out = carrier_phase_at_points_torch(
        model,
        rows,
        cols,
        centre_row=None,
        dtype=np.float64,
        precision="float64",
    )
    np.testing.assert_allclose(
        np.asarray(torch_out), np.asarray(cpu), rtol=1e-10, atol=1e-6
    )
