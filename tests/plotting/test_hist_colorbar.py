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

from faninsar.plotting.colorbar import HistColorbar


def _expected_rgba(
    norm: colors.Normalize, cmap: colors.Colormap, value: float
) -> tuple[float, float, float, float]:
    sm = ScalarMappable(norm=norm, cmap=cmap)
    return tuple(sm.to_rgba(value))


def _pick_patch_color_for_value(
    hcb: HistColorbar, value: float
) -> tuple[float, float, float, float] | None:
    # Default orientation is vertical colorbar, horizontal histogram bars
    for patch in hcb.ax_hist.patches:
        y = patch.get_y()
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
        cmap = colors.ListedColormap(
            ["navy", "royalblue", "lightgray", "tomato", "darkred"]
        )

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
        cmap = colors.ListedColormap([
            "#313695",
            "#4575b4",
            "#74add1",
            "#abd9e9",
            "#fee090",
            "#fdae61",
            "#f46d43",
            "#d73027",
        ])
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
        cmap = colors.ListedColormap([
            "#ffffcc",
            "#c2e699",
            "#78c679",
            "#31a354",
            "#006837",
            "#004529",
            "#002b13",
        ])

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

    def test_contourf_explicit_colors(self) -> None:
        """Test histogram matches contourf with explicit colors parameter."""
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 80)
        X, Y = np.meshgrid(x, y)
        Z = 0.6 * X + 0.4 * Y

        levels = [-1, -0.5, 0, 0.5, 1]
        colors_list = ["blue", "cyan", "yellow", "red"]

        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, Z, levels=levels, colors=colors_list)
        hcb = fig.hist_colorbar(data=Z.ravel(), mappable=cf, ax=ax)

        # Get expected colors from contourf
        expected_rgba = cf.get_facecolors()
        assert expected_rgba is not None
        assert len(expected_rgba) == len(levels) - 1

        # Check that histogram patches have matching colors
        # For each level interval, find a patch and verify its color
        for i in range(len(levels) - 1):
            v_mid = (levels[i] + levels[i + 1]) / 2
            # Find patch containing this value
            patch_found = False
            for p in hcb.ax_hist.patches:
                y = p.get_y()
                h = p.get_height()
                if y <= v_mid <= (y + h):
                    patch_rgba = np.array(p.get_facecolor())
                    expected = np.array(expected_rgba[i])
                    # Allow small tolerance for color matching
                    np.testing.assert_allclose(patch_rgba[:3], expected[:3], atol=0.01)
                    patch_found = True
                    break
            assert patch_found, f"No patch found for level interval {i}"

        plt.close(fig)

    def test_tricontourf_explicit_colors(self) -> None:
        """Test tricontourf with explicit colors parameter."""
        from matplotlib.tri import Triangulation

        # Create triangulated domain
        np.random.seed(42)
        x_tri = np.random.rand(100)
        y_tri = np.random.rand(100)
        z_tri = x_tri + y_tri
        tri = Triangulation(x_tri, y_tri)

        levels = [0.0, 0.5, 1.0, 1.5]
        colors_tri = ["blue", "cyan", "yellow"]

        fig, ax = plt.subplots()
        tcf = ax.tricontourf(x_tri, y_tri, z_tri, levels=levels, colors=colors_tri)
        hcb = fig.hist_colorbar(data=z_tri, mappable=tcf, ax=ax)

        # Get expected colors from tricontourf
        expected_rgba = tcf.get_facecolors()
        assert expected_rgba is not None
        assert len(expected_rgba) == len(levels) - 1

        # Verify histogram patches match tricontourf colors
        for i in range(len(levels) - 1):
            v_mid = (levels[i] + levels[i + 1]) / 2
            # Find patch containing this value
            patch_found = False
            for p in hcb.ax_hist.patches:
                y = p.get_y()
                h = p.get_height()
                if y <= v_mid <= (y + h):
                    patch_rgba = np.array(p.get_facecolor())
                    expected = np.array(expected_rgba[i])
                    # Allow small tolerance
                    np.testing.assert_allclose(patch_rgba[:3], expected[:3], atol=0.01)
                    patch_found = True
                    break
            assert patch_found, f"No patch found for level interval {i}"

        plt.close(fig)
