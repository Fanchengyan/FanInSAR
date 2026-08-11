"""Tests for SBAS time-series inversion over unwrapped pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.stack.ifg_store import ArtifactResourceLimits
from faninsar.processing.timeseries import (
    TimeSeriesResult,
    invert_unwrapped_pairs,
    open_timeseries_zarr,
    write_timeseries_zarr,
)


def _result_for_write() -> TimeSeriesResult:
    """Build one small connected result for publication tests."""
    return invert_unwrapped_pairs(
        {
            "20161207_20161231": np.full((2, 2), 0.1, dtype=np.float32),
            "20161231_20170124": np.full((2, 2), -0.05, dtype=np.float32),
            "20161207_20170124": np.full((2, 2), 0.05, dtype=np.float32),
        },
        device="cpu",
    )


def test_invert_three_pair_network_and_write_zarr(tmp_path: Path) -> None:
    """SBAS inversion of a three-date redundant network is finite and writable."""
    # synthetic displacement increments between 3 dates
    shape = (8, 8)
    true_inc01 = np.full(shape, 0.1, dtype=np.float32)
    true_inc12 = np.full(shape, -0.05, dtype=np.float32)
    pair_phases = {
        "20161207_20161231": true_inc01,
        "20161231_20170124": true_inc12,
        "20161207_20170124": true_inc01 + true_inc12,
    }
    result = invert_unwrapped_pairs(pair_phases, device="cpu")
    assert result.increments.shape[0] == 2
    assert result.cumulative.shape[0] == 3
    assert result.residual_pairs.shape[0] == 3
    assert np.isfinite(result.increments).all()
    assert result.metadata["phase_unit"] == "radian"

    store = write_timeseries_zarr(result, tmp_path / "ts.zarr")
    assert store.exists()
    with open_timeseries_zarr(store) as pinned:
        group = zarr.open_group(str(pinned.path), mode="r")
        assert "increments" in group
        assert "cumulative" in group
        assert "phase_increments_rad" in group
        assert "phase_cumulative_rad" in group
        assert list(group.attrs["pair_ids"]) == sorted(pair_phases)


def test_invert_masks_only_rank_deficient_pixels() -> None:
    """A pixel is solved only when its finite pair graph connects every date."""
    pair_phases = {
        "20161207_20161231": np.array([[1.0, 1.0, np.nan]], dtype=np.float32),
        "20161231_20170124": np.array([[2.0, np.nan, 2.0]], dtype=np.float32),
        "20161207_20170124": np.array([[3.0, 3.0, np.nan]], dtype=np.float32),
    }

    result = invert_unwrapped_pairs(pair_phases, device="cpu")

    # All three pairs and the exact two-edge tree are both connected.
    np.testing.assert_allclose(result.phase_increments_rad[:, 0, 0], [1.0, 2.0])
    np.testing.assert_allclose(result.phase_increments_rad[:, 0, 1], [1.0, 2.0])
    # The final pixel has only one edge and cannot connect all three dates.
    assert np.isnan(result.phase_increments_rad[:, 0, 2]).all()
    assert np.isnan(result.phase_cumulative_rad[:, 0, 2]).all()
    assert result.metadata["n_valid_pixels"] == 2


def test_invert_rejects_globally_disconnected_network() -> None:
    """A globally disconnected acquisition graph is rejected before solving."""
    pair_phases = {
        "20160101_20160102": np.ones((1, 1), dtype=np.float32),
        "20160103_20160104": np.ones((1, 1), dtype=np.float32),
    }

    with pytest.raises(InvalidProcessingStateError, match="disconnected"):
        invert_unwrapped_pairs(pair_phases, device="cpu")


def test_primary_conjugate_secondary_phase_converts_to_los_displacement() -> None:
    """Phase uses the documented negative wavelength over four-pi conversion."""
    wavelength_m = 0.056
    phase_for_one_metre = 4.0 * np.pi / wavelength_m
    result = invert_unwrapped_pairs(
        {
            "20160101_20160102": np.array(
                [[phase_for_one_metre]],
                dtype=np.float64,
            ),
        },
        device="cpu",
        wavelength_m=wavelength_m,
    )

    assert result.displacement_increments_m is not None
    assert result.displacement_cumulative_m is not None
    np.testing.assert_allclose(result.displacement_increments_m, -1.0, rtol=1e-6)
    np.testing.assert_allclose(
        result.displacement_cumulative_m[:, 0, 0],
        [0.0, -1.0],
        rtol=1e-6,
    )
    assert result.metadata["interferogram_convention"] == (
        "primary_times_conjugate_secondary"
    )


@pytest.mark.parametrize("wavelength_m", [0.0, -0.01, np.nan, np.inf])
def test_invert_rejects_invalid_wavelength(wavelength_m: float) -> None:
    """Displacement conversion rejects non-physical wavelengths."""
    with pytest.raises(InvalidProcessingStateError, match="wavelength_m"):
        invert_unwrapped_pairs(
            {"20160101_20160102": np.ones((1, 1), dtype=np.float32)},
            wavelength_m=wavelength_m,
        )


def test_timeseries_zarr_uses_current_and_hash_validated_generation(
    tmp_path: Path,
) -> None:
    """Zarr publication is immutable, reopenable, and manifest-bound."""
    root = tmp_path / "timeseries.zarr"
    write_timeseries_zarr(_result_for_write(), root)

    assert (root / "TIMESERIES_CURRENT").stat().st_size < 4096
    assert not (root / "zarr.json").exists()
    with open_timeseries_zarr(root) as store:
        assert store.path.parent == root / ".timeseries_generations"
        group = zarr.open_group(str(store.path), mode="r")
        assert "phase_cumulative_rad" in group


def test_timeseries_current_and_payload_tampering_fail_closed(tmp_path: Path) -> None:
    """Forged CURRENT and modified Zarr bytes are rejected before opening."""
    import json

    root = tmp_path / "timeseries.zarr"
    write_timeseries_zarr(_result_for_write(), root)
    current = json.loads((root / "TIMESERIES_CURRENT").read_text())
    current["generation_id"] = "0" * 32
    (root / "TIMESERIES_CURRENT").write_text(json.dumps(current), encoding="utf-8")
    with pytest.raises(InvalidProcessingStateError, match="CURRENT"):
        open_timeseries_zarr(root)

    second_root = tmp_path / "second.zarr"
    write_timeseries_zarr(_result_for_write(), second_root)
    with open_timeseries_zarr(second_root) as second_store:
        second_generation = second_store.path
    payload = next(
        path
        for path in second_generation.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    payload.write_bytes(payload.read_bytes() + b"tampered")
    with pytest.raises(InvalidProcessingStateError, match="corrupted"):
        open_timeseries_zarr(second_root)
    assert root.exists()


def test_timeseries_quota_and_legacy_layout_fail_before_publication(
    tmp_path: Path,
) -> None:
    """Resource exhaustion and legacy direct stores never become cache hits."""
    root = tmp_path / "timeseries.zarr"
    with pytest.raises(InvalidProcessingStateError, match="quota"):
        write_timeseries_zarr(
            _result_for_write(),
            root,
            resource_limits=ArtifactResourceLimits(
                max_final_bytes=1,
                max_temporary_bytes=1,
                min_free_bytes=0,
            ),
        )
    assert not (root / "TIMESERIES_CURRENT").exists()

    legacy = tmp_path / "legacy.zarr"
    legacy.mkdir()
    (legacy / "zarr.json").write_text("{}", encoding="utf-8")
    with pytest.raises(InvalidProcessingStateError, match="legacy direct"):
        write_timeseries_zarr(_result_for_write(), legacy)
