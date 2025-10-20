"""Tests for histogram-embedded colorbar (HistColorbar).

Covers a variety of norm and cmap types, including contourf mappables.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.cm import ScalarMappable

from faninsar.plots.hist_colorbar import HistColorbar


def _expected_rgba(norm: colors.Normalize, cmap: colors.Colormap, value: float) -> tuple[float, float, float, float]:
    sm = ScalarMappable(norm=norm, cmap=cmap)
    return tuple(sm.to_rgba(value))


def _pick_patch_color_for_value(hcb: HistColorbar, value: float) -> tuple[float, float, float, float] | None:
    # Default orientation is vertical colorbar, horizontal histogram bars
    for patch in hcb.ax_hist.patches:
        x, y = patch.get_x(), patch.get_y()
        w, h = patch.get_width(), patch.get_height()
        # Data axis aligns with y for vertical colorbar (horizontal bars)
        if y <= value <= (y + h) and w > 0 and h >= 0:
            fc = patch.get_facecolor()
            # facecolor may be an array-like
            return (float(fc[0]), float(fc[1]), float(fc[2]), float(fc[3]))
    return None


class TestHistColorbarBasic:
    def test_normalize_linear_colormap(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(loc=0, scale=1, size=2000)
        vmin, vmax = -3, 3
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("viridis")

        fig, ax = plt.subplots()
        hcb = fig.hist_colorbar(data=data, cmap=cmap, norm=norm, ax=ax)

        assert isinstance(hcb, HistColorbar)
        assert len(hcb.ax_hist.patches) > 0

        # Check a representative color near 0
        val = 0.0
        got = _pick_patch_color_for_value(hcb, val)
        exp = _expected_rgba(norm, cmap, val)
        assert got is not None
        np.testing.assert_allclose(got[:3], exp[:3], atol=0.07)
        plt.close(fig)

    def test_listed_colormap_without_boundarynorm(self) -> None:
        rng = np.random.default_rng(1)
        data = rng.normal(size=1500)
        vmin, vmax = -2.5, 2.5
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = colors.ListedColormap(["navy", "royalblue", "lightgray", "tomato", "darkred"])

        fig, ax = plt.subplots()
        hcb = fig.hist_colorbar(data=data, cmap=cmap, norm=norm, ax=ax)

        assert len(hcb.ax_hist.patches) > 0
        val = 1.0
        got = _pick_patch_color_for_value(hcb, val)
        exp = _expected_rgba(norm, cmap, val)
        assert got is not None
        np.testing.assert_allclose(got[:3], exp[:3], atol=0.1)
        plt.close(fig)


class TestHistColorbarDiscrete:
    def test_boundarynorm_with_listed_colormap(self) -> None:
        rng = np.random.default_rng(2)
        data = rng.uniform(-2, 2, size=3000)
        boundaries = np.linspace(-2.0, 2.0, 9)  # 8 bins
        cmap = colors.ListedColormap(
            ["#313695", "#4575b4", "#74add1", "#abd9e9", "#fee090", "#fdae61", "#f46d43", "#d73027"]
        )
        norm = colors.BoundaryNorm(boundaries, ncolors=cmap.N)

        fig, ax = plt.subplots()
        hcb = fig.hist_colorbar(data=data, cmap=cmap, norm=norm, ax=ax)

        # Ensure patches exist
        assert len(hcb.ax_hist.patches) > 0

        # Check that boundary edges appear among patch edges on the data axis
        edges = []
        for p in hcb.ax_hist.patches:
            edges.extend([p.get_y(), p.get_y() + p.get_height()])
        edges = np.array(edges)
        for b in boundaries:
            assert np.any(np.isclose(edges, b, atol=1e-6))
        plt.close(fig)


class TestHistColorbarNonlinear:
    def test_lognorm(self) -> None:
        rng = np.random.default_rng(3)
        data = rng.lognormal(mean=0.0, sigma=1.0, size=4000)
        norm = colors.LogNorm(vmin=0.05, vmax=20.0)
        cmap = plt.get_cmap("magma")
        fig, ax = plt.subplots()
        hcb = fig.hist_colorbar(data=data, cmap=cmap, norm=norm, ax=ax)
        assert len(hcb.ax_hist.patches) > 0
        # Color at geometric mean ~1.0
        val = 1.0
        got = _pick_patch_color_for_value(hcb, val)
        exp = _expected_rgba(norm, cmap, val)
        assert got is not None
        np.testing.assert_allclose(got[:3], exp[:3], atol=0.08)
        plt.close(fig)

    def test_two_slope_norm(self) -> None:
        rng = np.random.default_rng(4)
        data = rng.normal(size=3000)
        norm = colors.TwoSlopeNorm(vmin=-3, vcenter=0.0, vmax=3)
        cmap = plt.get_cmap("coolwarm")
        fig, ax = plt.subplots()
        hcb = fig.hist_colorbar(data=data, cmap=cmap, norm=norm, ax=ax)
        assert len(hcb.ax_hist.patches) > 0
        val = 0.0
        got = _pick_patch_color_for_value(hcb, val)
        exp = _expected_rgba(norm, cmap, val)
        assert got is not None
        np.testing.assert_allclose(got[:3], exp[:3], atol=0.07)
        plt.close(fig)


class TestHistColorbarContourf:
    def test_contourf_mappable_levels(self) -> None:
        # Create a smooth field
        x = np.linspace(-3, 3, 101)
        y = np.linspace(-2, 2, 81)
        X, Y = np.meshgrid(x, y)
        Z = np.hypot(X, Y)
        levels = np.linspace(0.0, 3.5, 8)
        cmap = colors.ListedColormap(
            ["#ffffcc", "#c2e699", "#78c679", "#31a354", "#006837", "#004529", "#002b13"]
        )

        fig, ax = plt.subplots()
        cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
        # Use mappable directly
        hcb = fig.hist_colorbar(data=Z.ravel(), mappable=cs, ax=ax)

        # Ensure patches exist
        assert len(hcb.ax_hist.patches) > 0

        # Verify that split boundaries align with contour levels
        edges = []
        for p in hcb.ax_hist.patches:
            edges.extend([p.get_y(), p.get_y() + p.get_height()])
        edges = np.array(edges)
        for b in levels:
            assert np.any(np.isclose(edges, b, atol=1e-6))
        plt.close(fig)
