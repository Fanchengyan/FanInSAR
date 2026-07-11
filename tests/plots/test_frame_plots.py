"""Tests for faninsar.plots.frame — xarray-aware InSAR plotting (Phase 3)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - non-interactive backend for tests

import numpy as np
import pytest
import xarray as xr

from faninsar.plots.frame import (
    plot_coherence,
    plot_displacement_timeseries,
    plot_interferogram,
    plot_velocity,
)


def _phase_da(shape=(8, 8), pair=None) -> xr.DataArray:
    rng = np.random.default_rng(0)
    data = rng.uniform(-np.pi, np.pi, shape).astype(np.float32)
    coords = {}
    dims = ("y", "x")
    if pair is not None:
        data = data[None]
        dims = ("pair", "y", "x")
        coords["pair"] = [pair]
    return xr.DataArray(data, dims=dims, coords=coords, name="unw_phase")


def _coherence_da(shape=(8, 8)) -> xr.DataArray:
    rng = np.random.default_rng(1)
    data = rng.uniform(0.0, 1.0, shape).astype(np.float32)
    return xr.DataArray(data, dims=("y", "x"), name="coherence")


def _displacement_da(n_time=5, shape=(8, 8)) -> xr.DataArray:
    rng = np.random.default_rng(2)
    data = rng.normal(0, 0.01, (n_time, *shape)).astype(np.float32).cumsum(0)
    times = np.array([np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(n_time)])
    return xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": times, "y": np.arange(shape[0]), "x": np.arange(shape[1])},
        name="displacement",
    )


def _velocity_da(shape=(8, 8)) -> xr.DataArray:
    rng = np.random.default_rng(3)
    data = rng.normal(0, 0.005, shape).astype(np.float32)
    return xr.DataArray(data, dims=("y", "x"), name="velocity")


class TestPlotInterferogram:
    def test_returns_axes(self) -> None:
        ax = plot_interferogram(_phase_da())
        import matplotlib.pyplot as plt

        assert ax is not None
        plt.close("all")

    def test_pair_dim_selected(self) -> None:
        da = _phase_da(pair="20200101_20200201")
        ax = plot_interferogram(da, pair="20200101_20200201")
        import matplotlib.pyplot as plt

        assert "Interferogram" in ax.get_title() or "unw_phase" in ax.get_title()
        plt.close("all")

    def test_accepts_existing_axes(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        out = plot_interferogram(_phase_da(), ax=ax)
        assert out is ax
        plt.close("all")

    def test_nan_data_no_error(self) -> None:
        da = _phase_da()
        da.values[:] = np.nan
        ax = plot_interferogram(da)
        import matplotlib.pyplot as plt

        assert ax is not None
        plt.close("all")


class TestPlotCoherence:
    def test_returns_axes(self) -> None:
        ax = plot_coherence(_coherence_da())
        import matplotlib.pyplot as plt

        assert ax is not None
        plt.close("all")

    def test_accepts_existing_axes(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        out = plot_coherence(_coherence_da(), ax=ax)
        assert out is ax
        plt.close("all")


class TestPlotDisplacement:
    def test_returns_axes(self) -> None:
        ax = plot_displacement_timeseries(_displacement_da())
        import matplotlib.pyplot as plt

        assert ax is not None
        plt.close("all")

    def test_point_selection(self) -> None:
        da = _displacement_da()
        ax = plot_displacement_timeseries(da, point=(4, 4))
        import matplotlib.pyplot as plt

        assert ax is not None
        plt.close("all")

    def test_no_time_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            plot_displacement_timeseries(_coherence_da())


class TestPlotVelocity:
    def test_returns_axes(self) -> None:
        ax = plot_velocity(_velocity_da())
        import matplotlib.pyplot as plt

        assert ax is not None
        plt.close("all")

    def test_accepts_existing_axes(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        out = plot_velocity(_velocity_da(), ax=ax)
        assert out is ax
        plt.close("all")

    def test_nan_data_no_error(self) -> None:
        da = _velocity_da()
        da.values[:] = np.nan
        ax = plot_velocity(da)
        import matplotlib.pyplot as plt

        assert ax is not None
        plt.close("all")


class TestFramePlotWrappers:
    """D3.2: Frame.plot_* / Collection.plot_* wrappers produce figures."""

    def test_collection_plot_interferogram(self, frame_dir: Path) -> None:
        from faninsar.datasets.frame import Frame

        frame = Frame(frame_dir)
        pairs = frame.interferograms.pairs().to_names()
        ax = frame.interferograms.plot_interferogram(pairs[0])
        import matplotlib.pyplot as plt

        assert ax is not None
        plt.close("all")

    def test_collection_plot_coherence(self, frame_dir: Path) -> None:
        from faninsar.datasets.frame import Frame

        frame = Frame(frame_dir)
        pairs = frame.interferograms.pairs().to_names()
        ax = frame.interferograms.plot_coherence(pairs[0])
        import matplotlib.pyplot as plt

        assert ax is not None
        plt.close("all")
