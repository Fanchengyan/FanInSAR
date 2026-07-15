"""Tests for SBAS time-series inversion over unwrapped pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import zarr

if TYPE_CHECKING:
    from pathlib import Path

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

    store = write_timeseries_zarr(result, tmp_path / "ts.zarr")
    assert store.exists()
    group = zarr.open_group(str(store), mode="r")
    assert "increments" in group
    assert "cumulative" in group
    assert list(group.attrs["pair_ids"]) == sorted(pair_phases)
