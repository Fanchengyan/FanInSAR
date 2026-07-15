"""Unit tests for the pair-product comparison harness."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from faninsar.processing.comparison import (
    PairComparisonError,
    PairComparisonReport,
    coherence_absolute_error,
    compare_pair_products,
    geolocation_residual,
    phase_circular_rmse,
    rewrap_residual_stats,
)


class TestPhaseCircularRmse:
    """Tests for phase_circular_rmse."""

    def test_identical_arrays_yield_zero_error(self) -> None:
        """Identical wrapped phases should produce zero RMSE."""
        arr = np.angle(np.exp(1j * np.random.default_rng(0).random((5, 5))))
        result = phase_circular_rmse(arr, arr)
        assert result.circular_rmse_rad == pytest.approx(0.0, abs=1e-12)
        assert result.linear_rmse_rad == pytest.approx(0.0, abs=1e-12)
        assert result.mean_bias_rad == pytest.approx(0.0, abs=1e-12)
        assert result.max_absolute_error_rad == pytest.approx(0.0, abs=1e-12)
        assert result.rewrap_count == 0

    def test_known_bias_is_recovered(self) -> None:
        """A constant phase bias should appear in circular RMSE."""
        rng = np.random.default_rng(1)
        ref = rng.random((10, 10)) * 2.0 * np.pi
        bias = 0.05
        cand = ref + bias
        result = phase_circular_rmse(ref, cand)
        assert result.circular_rmse_rad == pytest.approx(bias, abs=1e-3)
        assert result.mean_bias_rad == pytest.approx(bias, abs=1e-3)

    def test_rewrap_count_detects_wraparounds(self) -> None:
        """Large linear differences that wrap should be counted."""
        ref = np.zeros((3, 3))
        cand = np.full((3, 3), 3.5 * np.pi)
        result = phase_circular_rmse(ref, cand)
        assert result.rewrap_count == 9
        assert result.circular_rmse_rad == pytest.approx(0.5 * np.pi, abs=1e-12)

    def test_rejects_mismatched_shapes(self) -> None:
        """Shape mismatch must raise PairComparisonError."""
        with pytest.raises(PairComparisonError, match="shape"):
            phase_circular_rmse(np.zeros((2, 2)), np.zeros((3, 3)))

    def test_rejects_non_finite_values(self) -> None:
        """Non-finite values must raise PairComparisonError."""
        with pytest.raises(PairComparisonError, match="finite"):
            phase_circular_rmse(np.zeros((2, 2)), np.full((2, 2), np.nan))


class TestCoherenceAbsoluteError:
    """Tests for coherence_absolute_error."""

    def test_identical_arrays_yield_zero_error(self) -> None:
        """Identical coherence maps should produce zero error."""
        arr = np.full((4, 4), 0.8)
        result = coherence_absolute_error(arr, arr)
        assert result.mean_absolute_error == pytest.approx(0.0, abs=1e-12)
        assert result.max_absolute_error == pytest.approx(0.0, abs=1e-12)
        assert result.rmse == pytest.approx(0.0, abs=1e-12)

    def test_known_bias_is_recovered(self) -> None:
        """A constant coherence bias should appear in MAE."""
        ref = np.full((5, 5), 0.8)
        bias = -0.02
        cand = ref + bias
        result = coherence_absolute_error(ref, cand)
        assert result.mean_absolute_error == pytest.approx(0.02, abs=1e-12)
        assert result.max_absolute_error == pytest.approx(0.02, abs=1e-12)
        assert result.median_absolute_error == pytest.approx(0.02, abs=1e-12)

    def test_rejects_out_of_bounds(self) -> None:
        """Coherence outside [0, 1] must raise PairComparisonError."""
        with pytest.raises(PairComparisonError, match=r"\[0, 1\]"):
            coherence_absolute_error(np.zeros((2, 2)), np.full((2, 2), 1.1))


class TestRewrapResidualStats:
    """Tests for rewrap_residual_stats."""

    def test_identical_arrays_yield_zero_residual(self) -> None:
        """Identical unwrapped phases should produce zero residual."""
        arr = np.random.default_rng(2).random((5, 5)) * 10.0
        result = rewrap_residual_stats(arr, arr)
        assert result.residual_mean_rad == pytest.approx(0.0, abs=1e-12)
        assert result.residual_std_rad == pytest.approx(0.0, abs=1e-12)
        assert result.residual_rmse_rad == pytest.approx(0.0, abs=1e-12)
        assert result.residual_max_rad == pytest.approx(0.0, abs=1e-12)
        assert result.fraction_within_half_cycle == pytest.approx(1.0, abs=1e-12)

    def test_integer_cycle_difference_yields_zero_residual(self) -> None:
        """Differences of integer cycles should wrap to zero."""
        ref = np.zeros((4, 4))
        cand = np.full((4, 4), 4.0 * np.pi)
        result = rewrap_residual_stats(ref, cand)
        assert result.residual_mean_rad == pytest.approx(0.0, abs=1e-12)
        assert result.residual_rmse_rad == pytest.approx(0.0, abs=1e-12)
        assert result.fraction_within_half_cycle == pytest.approx(1.0, abs=1e-12)

    def test_half_cycle_difference_stats(self) -> None:
        """A half-cycle bias should be reflected in the residual."""
        ref = np.zeros((4, 4))
        cand = np.full((4, 4), np.pi / 2.0)
        result = rewrap_residual_stats(ref, cand)
        assert result.residual_mean_rad == pytest.approx(np.pi / 2.0, abs=1e-12)
        assert result.residual_rmse_rad == pytest.approx(np.pi / 2.0, abs=1e-12)
        assert 0.0 <= result.fraction_within_half_cycle <= 1.0


    def test_full_cycle_difference_is_outside_half_cycle(self) -> None:
        """A full-cycle difference should fall outside the half-cycle band."""
        ref = np.zeros((4, 4))
        cand = np.full((4, 4), np.pi)
        result = rewrap_residual_stats(ref, cand)
        assert result.fraction_within_half_cycle == pytest.approx(0.0, abs=1e-12)


class TestGeolocationResidual:
    """Tests for geolocation_residual."""

    def test_identical_grids_yield_zero(self) -> None:
        """Identical lon/lat grids should produce zero residual."""
        lon = np.linspace(-120, -119, 10)
        lat = np.linspace(34, 35, 10)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        result = geolocation_residual(lon_grid, lat_grid, lon_grid, lat_grid)
        assert result.horizontal_rmse_m == pytest.approx(0.0, abs=1e-6)
        assert result.horizontal_max_m == pytest.approx(0.0, abs=1e-6)

    def test_known_shift_is_recovered(self) -> None:
        """A known lon/lat shift should appear in horizontal RMSE."""
        lon = np.linspace(-120, -119, 10)
        lat = np.linspace(34, 35, 10)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        shift_deg = 0.001
        result = geolocation_residual(
            lon_grid, lat_grid, lon_grid + shift_deg, lat_grid
        )
        # At ~34 deg latitude, ~1 deg lon ≈ 92 km
        expected_m = shift_deg * 111_320.0 * np.cos(np.deg2rad(34.5))
        assert result.horizontal_rmse_m == pytest.approx(expected_m, rel=0.01)

    def test_rejects_mismatched_lon_lat_shapes(self) -> None:
        """Mismatched lon/lat shapes must raise PairComparisonError."""
        with pytest.raises(PairComparisonError, match="shape mismatch"):
            geolocation_residual(
                np.zeros((2, 2)), np.zeros((3, 3)), np.zeros((2, 2)), np.zeros((2, 2))
            )


class TestComparePairProducts:
    """Tests for the high-level compare_pair_products entry point."""

    def test_in_memory_mapping_produces_json(self, tmp_path: "Path") -> None:
        """Passing array mappings should write a valid JSON report."""
        shape = (4, 4)
        faninsar = {
            "wrapped_phase": np.angle(np.exp(1j * np.ones(shape))),
            "coherence": np.full(shape, 0.8),
            "unwrapped_phase": np.ones(shape),
            "lon": np.full(shape, -120.0),
            "lat": np.full(shape, 34.0),
        }
        reference = {
            "wrapped_phase": np.angle(np.exp(1j * np.ones(shape))),
            "coherence": np.full(shape, 0.8),
            "unwrapped_phase": np.ones(shape),
            "lon": np.full(shape, -120.0),
            "lat": np.full(shape, 34.0),
        }
        out = tmp_path / "report.json"
        report = compare_pair_products(
            faninsar,
            reference,
            out,
            pair_id="test-pair",
            reference_source="synthetic",
        )
        assert report.pair_id == "test-pair"
        assert report.phase is not None
        assert report.coherence is not None
        assert report.rewrap_residual is not None
        assert report.geolocation is not None
        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["pair_id"] == "test-pair"
        assert payload["phase"]["circular_rmse_rad"] == pytest.approx(0.0, abs=1e-12)
        assert payload["coherence"]["mean_absolute_error"] == pytest.approx(
            0.0, abs=1e-12
        )

    def test_missing_unwrapped_phase_warns_and_skips_rewrap(
        self, tmp_path: "Path"
    ) -> None:
        """Missing unwrapped phase should skip rewrap residual gracefully."""
        shape = (3, 3)
        faninsar = {
            "wrapped_phase": np.zeros(shape),
            "coherence": np.full(shape, 0.8),
        }
        reference = {
            "wrapped_phase": np.zeros(shape),
            "coherence": np.full(shape, 0.8),
        }
        out = tmp_path / "report.json"
        report = compare_pair_products(
            faninsar,
            reference,
            out,
            pair_id="test-pair",
        )
        assert report.phase is not None
        assert report.coherence is not None
        assert report.rewrap_residual is None

    def test_missing_lon_lat_warns_and_skips_geolocation(
        self, tmp_path: "Path"
    ) -> None:
        """Missing lon/lat should skip geolocation residual gracefully."""
        shape = (3, 3)
        faninsar = {
            "wrapped_phase": np.zeros(shape),
            "coherence": np.full(shape, 0.8),
        }
        reference = {
            "wrapped_phase": np.zeros(shape),
            "coherence": np.full(shape, 0.8),
        }
        out = tmp_path / "report.json"
        report = compare_pair_products(
            faninsar,
            reference,
            out,
            pair_id="test-pair",
            lon_lat_layers=None,
        )
        assert report.geolocation is None

    def test_noisy_arrays_produce_non_zero_metrics(self, tmp_path: "Path") -> None:
        """Noisy candidate arrays should produce non-zero metric values."""
        rng = np.random.default_rng(42)
        shape = (10, 10)
        ref_phase = rng.random(shape) * 2.0 * np.pi
        cand_phase = ref_phase + 0.1 + rng.normal(scale=0.05, size=shape)
        ref_coh = np.full(shape, 0.8)
        cand_coh = np.full(shape, 0.75)
        faninsar = {
            "wrapped_phase": np.angle(np.exp(1j * cand_phase)),
            "coherence": cand_coh,
        }
        reference = {
            "wrapped_phase": np.angle(np.exp(1j * ref_phase)),
            "coherence": ref_coh,
        }
        out = tmp_path / "report.json"
        report = compare_pair_products(
            faninsar,
            reference,
            out,
            pair_id="noisy-pair",
        )
        assert report.phase is not None
        assert report.phase.circular_rmse_rad > 0.0
        assert report.coherence is not None
        assert report.coherence.mean_absolute_error > 0.0

    def test_missing_required_layer_raises(self, tmp_path: "Path") -> None:
        """Missing required layer must raise PairComparisonError."""
        faninsar: dict[str, np.ndarray] = {}
        reference: dict[str, np.ndarray] = {}
        with pytest.raises(PairComparisonError, match="wrapped_phase layer missing"):
            compare_pair_products(
                faninsar,
                reference,
                tmp_path / "report.json",
            )

    def test_zarr_round_trip(self, tmp_path: Path) -> None:
        """Zarr stores should be readable and produce valid reports."""
        zarr = pytest.importorskip("zarr")
        shape = (4, 4)
        faninsar_path = tmp_path / "faninsar.zarr"
        reference_path = tmp_path / "reference.zarr"
        faninsar_root = zarr.open_group(str(faninsar_path), mode="w")
        reference_root = zarr.open_group(str(reference_path), mode="w")
        faninsar_root.create_array("wrapped_phase", data=np.zeros(shape), overwrite=True)
        faninsar_root.create_array("coherence", data=np.full(shape, 0.8), overwrite=True)
        reference_root.create_array("wrapped_phase", data=np.zeros(shape), overwrite=True)
        reference_root.create_array("coherence", data=np.full(shape, 0.8), overwrite=True)
        out = tmp_path / "report.json"
        report = compare_pair_products(
            faninsar_path,
            reference_path,
            out,
            pair_id="zarr-pair",
        )
        assert report.phase is not None
        assert report.phase.circular_rmse_rad == pytest.approx(0.0, abs=1e-12)
        assert out.exists()


class TestPairComparisonReport:
    """Tests for the report dataclass."""

    def test_json_round_trip(self, tmp_path: Path) -> None:
        """A report should round-trip through JSON faithfully."""
        report = PairComparisonReport(
            pair_id="pair-1",
            reference_source="isce2",
            candidate_source="faninsar",
            phase=None,
            coherence=None,
            rewrap_residual=None,
            geolocation=None,
            versions={"numpy": np.__version__},
        )
        path = report.write_json(tmp_path / "report.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["pair_id"] == "pair-1"
        assert payload["reference_source"] == "isce2"
        assert payload["candidate_source"] == "faninsar"
        assert payload["phase"] is None
        assert payload["coherence"] is None
