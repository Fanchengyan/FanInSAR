"""Tests for SBAS time-series inversion over unwrapped pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.timeseries import invert_unwrapped_pairs, write_timeseries_zarr


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
    group = zarr.open_group(str(store), mode="r")
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
