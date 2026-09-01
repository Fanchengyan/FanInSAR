"""Synthetic-oracle tests for dispersive/non-dispersive separation."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from faninsar.processing.atmosphere.config import IonosphereEstimationConfig
from faninsar.processing.atmosphere.estimation import (
    align_absolute_jumps,
    estimate_disp_nondisp,
    solve_2x2_low_high,
    solve_guided_split,
)

_TWO_PI = 2.0 * math.pi


def _forward_model(
    dispersive_at_f0: torch.Tensor,
    non_dispersive: torch.Tensor,
    config: IonosphereEstimationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build subband phases with ISCE3-test physical coefficient roles.

    Mirrors ``tests/python/packages/isce3/atmosphere/ionosphere.py``:
    non-dispersive scales linearly with frequency, the dispersive term with
    1/f; normalizing by ``f0`` reproduces the solver's ratio matrix.
    """
    a = config.freq_low / config.f0
    b = config.f0 / config.freq_low
    c = config.freq_high / config.f0
    d = config.f0 / config.freq_high

    disp = dispersive_at_f0.to(torch.float64)
    nondisp = non_dispersive.to(torch.float64)
    low = a * nondisp + b * disp
    high = c * nondisp + d * disp
    return (low.to(torch.float64), high.to(torch.float64))


class TestSolve2x2:
    def test_roundtrip_is_exact(self, fine_mode_config) -> None:
        grid = torch.linspace(-7.0, 7.0, 32, dtype=torch.float64)
        gx, gy = torch.meshgrid(grid, grid, indexing="ij")
        true_disp = 1.3 + 0.4 * gx - 0.2 * gy**2
        true_nondisp = -0.8 * gx + 0.15 * gy

        phi_low, phi_high = _forward_model(true_disp, true_nondisp, fine_mode_config)
        disp, nondisp = solve_2x2_low_high(phi_low, phi_high, fine_mode_config)

        assert disp.dtype == torch.float64
        torch.testing.assert_close(disp, true_disp, rtol=1e-12, atol=1e-9)
        torch.testing.assert_close(nondisp, true_nondisp, rtol=1e-12, atol=1e-9)

    def test_singular_combination_raises(self) -> None:
        valid = IonosphereEstimationConfig(
            f0=1257.5e6, freq_low=1248e6, freq_high=1266e6
        )
        degenerate = dataclasses.replace(valid, degraded=False)
        object.__setattr__(degenerate, "freq_low", degenerate.freq_high)

        with pytest.raises(ZeroDivisionError):
            solve_2x2_low_high(
                torch.zeros(2, 2, dtype=torch.float64),
                torch.zeros(2, 2, dtype=torch.float64),
                degenerate,
            )


class TestJumpAlignment:
    def test_global_jump_restores_module_2pi(
        self, fine_mode_config: IonosphereEstimationConfig
    ) -> None:
        rows = cols = 16
        base_low = torch.full((rows, cols), 0.6, dtype=torch.float64)
        base_high = torch.full((rows, cols), -1.1, dtype=torch.float64)
        shifted_high = base_high + 5.0 * _TWO_PI

        low, high = align_absolute_jumps(base_low, shifted_high, fine_mode_config)

        assert not torch.allclose(high, shifted_high)
        residual = torch.angle(torch.exp(1j * (high - base_high)))
        torch.testing.assert_close(
            residual.abs(), torch.zeros_like(residual), rtol=1e-9, atol=1e-12
        )
        torch.testing.assert_close(low, base_low)

    def test_pixel_rounded_handles_spatially_varying_offsets(self) -> None:
        grid = torch.linspace(-1.0, 1.0, 48, dtype=torch.float64)
        gy, gx = torch.meshgrid(grid, grid, indexing="ij")
        ramp_cycles = torch.round(3.5 * (gx + 0.5 * gy))
        base_low = 0.2 * gx
        base_high = -0.4 * gy
        noisy_high = base_high + ramp_cycles * _TWO_PI

        cfg = IonosphereEstimationConfig(
            f0=1257.5e6,
            freq_low=1257.5e6 - 28e6 / 3,
            freq_high=1257.5e6 + 28e6 / 3,
            alignment_strategy="alosstack_pixel_rounded",
        )
        _, aligned = align_absolute_jumps(base_low, noisy_high, cfg)
        residual = torch.angle(torch.exp(1j * (aligned - base_high)))
        assert float(residual.abs().max()) < 1e-6

    def test_all_invalid_input_raises(self, fine_mode_config) -> None:
        nan_grid = torch.full((4, 4), float("nan"))
        with pytest.raises(ValueError, match="entirely invalid"):
            align_absolute_jumps(nan_grid, nan_grid.clone(), fine_mode_config)


class TestEndToEndRecovery:
    def test_synthetic_oracle_recovers_known_screen(
        self, fine_mode_config: IonosphereEstimationConfig
    ) -> None:
        rows = cols = 24
        grid = torch.linspace(-5.0, 5.0, rows, dtype=torch.float64)
        gy, gx = torch.meshgrid(grid, grid, indexing="ij")
        # ionospheric screen at carrier: smooth large-scale structure
        true_iono = 2.0 + 1.1 * torch.sin(gx / 3.0) + 0.7 * torch.cos(gy / 2.0)
        # geophysical + tropospheric phase: identical across subbands
        true_geo = -0.9 + 0.5 * gx * gy / 10.0

        phi_low, phi_high = _forward_model(true_iono, true_geo, fine_mode_config)
        # inject absolute misalignment between the two bands (4 cycles, high)
        phi_high_shifted = phi_high + 4.0 * _TWO_PI

        disp, nondisp = estimate_disp_nondisp(
            phi_low,
            phi_high_shifted,
            fine_mode_config,
        )

        for recovered, truth in ((disp, true_iono), (nondisp, true_geo)):
            residual = torch.angle(torch.exp(1j * (recovered - truth)))
            assert float(residual.abs().max()) < 1e-9

    def test_no_data_propagates_nan(self, fine_mode_config) -> None:
        valid = torch.ones(8, 8, dtype=torch.bool)
        valid[:2, :] = False
        disp, _ = estimate_disp_nondisp(
            torch.ones(8, 8), torch.zeros(8, 8), fine_mode_config, valid_mask=valid
        )
        assert bool(torch.isnan(disp[:2, :]).all())
        assert not bool(torch.isnan(disp[2:, :]).any())

    def test_unwrap_error_coefficients_applied(
        self, fine_mode_config: IonosphereEstimationConfig
    ) -> None:
        truth_low = torch.full((4, 4), 1.0)
        truth_high = torch.full((4, 4), -2.0)
        comm = torch.full((4, 4), 0.25)
        diff = torch.full((4, 4), 0.125)

        no_corr = estimate_disp_nondisp(truth_low, truth_high, fine_mode_config)
        with_corr = estimate_disp_nondisp(
            truth_low,
            truth_high,
            fine_mode_config,
            comm_unwcor_coef=comm,
            diff_unwcor_coef=diff,
        )

        changed = [
            not torch.equal(a, b) for a, b in zip(no_corr, with_corr, strict=True)
        ]
        assert all(changed)


class TestSolveGuidedSplit:
    def test_roundtrip_is_exact_with_uniform_coherence(
        self, fine_mode_config: IonosphereEstimationConfig
    ) -> None:
        """With uniform coherence (no adjustment needed), guided_split recovers
        the known dispersive/non-dispersive pair exactly.
        """
        grid = torch.linspace(-7.0, 7.0, 32, dtype=torch.float64)
        gx, gy = torch.meshgrid(grid, grid, indexing="ij")
        true_disp = 1.3 + 0.4 * gx - 0.2 * gy**2
        true_nondisp = -0.8 * gx + 0.15 * gy

        phi_low, phi_high = _forward_model(true_disp, true_nondisp, fine_mode_config)
        coh = torch.ones_like(phi_low)

        disp, nondisp = solve_guided_split(phi_low, phi_high, coh, fine_mode_config)

        torch.testing.assert_close(disp, true_disp, rtol=1e-11, atol=1e-9)
        torch.testing.assert_close(nondisp, true_nondisp, rtol=1e-11, atol=1e-9)

    def test_matches_alosstack_numpy_oracle(
        self, fine_mode_config: IonosphereEstimationConfig
    ) -> None:
        """Bit-for-bit semantics against a faithful NumPy port of ISCE2
        ``runIonFilt.computeIonosphere`` (adjFlag=1: weighted degree-2
        surface fit of the subband difference, per-pixel integer-cycle
        adjustment of the upper band, then the physical solve).
        """
        import numpy as np

        shape = (37, 29)
        rng = np.random.default_rng(20260831)
        gy, gx = np.indices(shape, dtype=np.float64)
        # smooth dispersive + non-dispersive fields
        disp_true = 0.8 * np.sin(gx / 9.0) + 0.3 * np.cos(gy / 5.0)
        nondisp_true = -0.4 * gx / 29.0 + 0.2 * gy / 37.0
        f0 = float(fine_mode_config.f0)
        fl = float(fine_mode_config.freq_low)
        fh = float(fine_mode_config.freq_high)
        a = fl / f0
        b = f0 / fl
        c = fh / f0
        d = f0 / fh
        low = a * nondisp_true + b * disp_true
        high = c * nondisp_true + d * disp_true
        # smooth relative unwrap error (integer cycles, slowly varying)
        cycles = np.round(2.5 * np.sin(gx / 12.0))
        high = high + cycles * (2.0 * np.pi)
        coh = 0.3 + 0.7 * np.abs(np.cos(gx / 6.0) * np.sin(gy / 7.0))

        # --- faithful NumPy oracle (computeIonosphere adjFlag=1) ---
        wgt = coh**fine_mode_config.cor_order_adj
        n = low.size
        H = np.zeros((n, 6))
        H[:, 0] = 1.0
        x = gx.ravel()
        y = gy.ravel()
        H[:, 1] = x
        H[:, 2] = y
        H[:, 3] = x**2
        H[:, 4] = x * y
        H[:, 5] = y**2
        sw = np.sqrt(wgt.ravel())
        diff = (low - high).ravel()
        coeff = np.linalg.lstsq(H * sw[:, None], diff * sw, rcond=-1)[0]
        fit = (H @ coeff).reshape(shape)
        unw_adj = np.round((low - high - fit) / (2.0 * np.pi)) * (2.0 * np.pi)
        high_adj = high + unw_adj
        det = fh**2 - fl**2
        disp_ref = fl * fh * (low * fh - high_adj * fl) / f0 / det
        nondisp_ref = f0 * (high_adj * fh - low * fl) / det

        # --- FanInSAR torch implementation ---
        low_t = torch.as_tensor(low, dtype=torch.float64)
        high_t = torch.as_tensor(high, dtype=torch.float64)
        coh_t = torch.as_tensor(coh, dtype=torch.float64)
        disp, nondisp = solve_guided_split(low_t, high_t, coh_t, fine_mode_config)

        # The polynomial surface is affine-invariant, so the fits agree up
        # to lstsq round-off; per-pixel cycle recovery must agree exactly.
        disp_np = disp.numpy()
        nondisp_np = nondisp.numpy()
        assert np.allclose(disp_np, disp_ref, rtol=1e-9, atol=1e-8), (
            f"dispersive mismatch max {np.abs(disp_np - disp_ref).max():.3e}"
        )
        assert np.allclose(nondisp_np, nondisp_ref, rtol=1e-9, atol=1e-8), (
            f"non-dispersive mismatch max {np.abs(nondisp_np - nondisp_ref).max():.3e}"
        )

    def test_mask_propagates_through_estimate_disp_nondisp(
        self, fine_mode_config: IonosphereEstimationConfig
    ) -> None:
        """Masking is the wrapper's job: estimate_disp_nondisp with
        solve_core='guided_split' returns NaN outside the valid mask (the
        direct solve computes everywhere, exactly like alosStack).
        """
        shape = (8, 8)
        valid = torch.ones(shape, dtype=torch.bool)
        valid[2:4, :] = False
        coh = torch.ones(shape, dtype=torch.float64)
        astack_cfg = dataclasses.replace(fine_mode_config, solve_core="guided_split")
        disp, _ = estimate_disp_nondisp(
            torch.ones(shape), torch.zeros(shape), astack_cfg,
            valid_mask=valid, coherence=coh,
        )
        assert bool(torch.isnan(disp[2:4, :]).all())
        assert not bool(torch.isnan(disp[:2, :]).any())

    def test_zero_coherence_pixels_still_get_cycle_adjustment(
        self, fine_mode_config: IonosphereEstimationConfig
    ) -> None:
        """The per-pixel integer-cycle adjustment of the upper band applies
        to ALL lower-band pixels (alosStack ``flag2 = (lowerUnw != 0)``),
        not only to pixels with positive coherence weight.  Coherence gates
        the surface fit only; low-coherence land pixels with valid phase
        must still get their cycles adjusted relative to the fitted surface.
        """
        import numpy as np

        shape = (37, 29)
        rng = np.random.default_rng(7)
        gy, gx = np.indices(shape, dtype=np.float64)
        disp_true = 0.8 * np.sin(gx / 9.0) + 0.3 * np.cos(gy / 5.0)
        nondisp_true = -0.4 * gx / 29.0 + 0.2 * gy / 37.0
        f0 = float(fine_mode_config.f0)
        fl = float(fine_mode_config.freq_low)
        fh = float(fine_mode_config.freq_high)
        a = fl / f0
        b = f0 / fl
        c = fh / f0
        d = f0 / fh
        low = a * nondisp_true + b * disp_true
        high = c * nondisp_true + d * disp_true
        cycles = np.round(2.5 * np.sin(gx / 12.0))
        high = high + cycles * (2.0 * np.pi)

        # coherence: zero in a corner patch (valid phase there), smooth in
        # the rest -- exactly the alosStack scenario after cor thresholding
        coh = 0.3 + 0.7 * np.abs(np.cos(gx / 6.0) * np.sin(gy / 7.0))
        coh[:8, :8] = 0.0

        # --- faithful NumPy oracle (computeIonosphere adjFlag=1) ---
        wgt = coh**fine_mode_config.cor_order_adj
        n = low.size
        H = np.zeros((n, 6))
        H[:, 0] = 1.0
        x = gx.ravel()
        y = gy.ravel()
        H[:, 1] = x
        H[:, 2] = y
        H[:, 3] = x**2
        H[:, 4] = x * y
        H[:, 5] = y**2
        sw = np.sqrt(wgt.ravel())
        diff = (low - high).ravel()
        coeff = np.linalg.lstsq(H * sw[:, None], diff * sw, rcond=-1)[0]
        fit = (H @ coeff).reshape(shape)
        unw_adj = np.round((low - high - fit) / (2.0 * np.pi)) * (2.0 * np.pi)
        high_adj = high + unw_adj  # every pixel has valid (non-zero) low
        det = fh**2 - fl**2
        disp_ref = fl * fh * (low * fh - high_adj * fl) / f0 / det

        # --- FanInSAR torch implementation ---
        disp, _ = solve_guided_split(
            torch.as_tensor(low, dtype=torch.float64),
            torch.as_tensor(high, dtype=torch.float64),
            torch.as_tensor(coh, dtype=torch.float64),
            fine_mode_config,
        )
        disp_np = disp.numpy()
        assert np.allclose(disp_np, disp_ref, rtol=1e-9, atol=1e-8), (
            f"zero-coherence pixels not adjusted like alosStack: "
            f"max {np.abs(disp_np - disp_ref).max():.3e} rad"
        )
        # sanity: the zero-coherence patch was actually adjusted
        patch = (slice(0, 8), slice(0, 8))
        adjusted = np.round((low - high - fit)[patch] / (2.0 * np.pi))
        assert np.any(adjusted != 0), "test scenario: patch needs cycles"

    def test_equivalence_with_isce3_when_adjustment_trivial(
        self, fine_mode_config: IonosphereEstimationConfig
    ) -> None:
        """When the subband difference has no integer-cycle jumps (aligned
        smooth inputs), guided_split and isce3 solves produce the same result
        (the physical solve is algebraically identical to the m21/m22
        coefficients).
        """
        shape = (16, 16)
        grid = torch.linspace(-1.0, 1.0, shape[0], dtype=torch.float64)
        gy, gx = torch.meshgrid(grid, grid, indexing="ij")
        low = 0.3 * gx + 0.2 * gy**2
        high = -0.4 * gy + 0.1 * gx * gy
        coh = torch.ones(shape, dtype=torch.float64)

        alos_disp, _ = solve_guided_split(low, high, coh, fine_mode_config)
        isce3_disp, _ = solve_2x2_low_high(low, high, fine_mode_config)

        diff = (alos_disp - isce3_disp).abs()
        assert float(diff.mean()) < 1e-9

    def test_estimate_disp_nondisp_routes_to_alosstack(
        self, fine_mode_config: IonosphereEstimationConfig
    ) -> None:
        """estimate_disp_nondisp with solve_core='guided_split' and coherence
        recovers the same result as the direct call.
        """
        shape = (12, 12)
        low = torch.randn(shape, dtype=torch.float64)
        high = torch.randn(shape, dtype=torch.float64)
        coh = torch.full(shape, 0.8, dtype=torch.float64)

        astack_cfg = dataclasses.replace(fine_mode_config, solve_core="guided_split")
        disp_direct, _ = solve_guided_split(low, high, coh, fine_mode_config)
        disp_routed, _ = estimate_disp_nondisp(
            low, high, astack_cfg, coherence=coh,
        )
        torch.testing.assert_close(disp_direct, disp_routed, rtol=1e-12, atol=1e-12)

    def test_estimate_disp_nondisp_raises_without_coherence(
        self, fine_mode_config: IonosphereEstimationConfig
    ) -> None:
        astack_cfg = dataclasses.replace(fine_mode_config, solve_core="guided_split")
        with pytest.raises(ValueError, match="requires a coherence"):
            estimate_disp_nondisp(
                torch.zeros(4, 4), torch.zeros(4, 4), astack_cfg,
            )
