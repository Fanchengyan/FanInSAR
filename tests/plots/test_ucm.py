"""Tests for UCM plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from faninsar.plots.ucm import (
    plot_ucm,
    plot_ucm_heatmap,
    plot_ucm_spatial_profile,
    plot_ucm_surface_3d,
    plot_ucm_temporal_profile,
)


def _sample_ucm() -> xr.DataArray:
    """Create sample UCM data for plotting tests."""
    return xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]),
        dims=("res", "day"),
        coords={"res": [30, 60], "day": [12, 24, 36]},
        name="velocity",
    )


def test_heatmap_creates_axis_and_sets_custom_labels() -> None:
    """Heatmap should create an axis when ax is omitted."""
    image = plot_ucm_heatmap(
        _sample_ucm(),
        show_hist_colorbar=False,
        xlabel="Temporal baseline",
        ylabel="Spatial resolution",
    )

    assert image.axes.get_xlabel() == "Temporal baseline"
    assert image.axes.get_ylabel() == "Spatial resolution"
    plt.close(image.figure)


def test_spatial_profile_accepts_ax_and_custom_legend_title() -> None:
    """Spatial profile should use the provided ax and custom labels."""
    fig, ax = plt.subplots()
    lines = plot_ucm_spatial_profile(
        _sample_ucm(),
        ax=ax,
        xlabel="Velocity",
        ylabel="Resolution",
        legend_title="Temporal days",
    )

    assert len(lines) == 3
    assert ax.get_xlabel() == "Velocity"
    assert ax.get_ylabel() == "Resolution"
    assert ax.get_legend().get_title().get_text() == "Temporal days"
    plt.close(fig)


def test_temporal_profile_accepts_ax_and_custom_legend_title() -> None:
    """Temporal profile should use the provided ax and custom labels."""
    fig, ax = plt.subplots()
    lines = plot_ucm_temporal_profile(
        _sample_ucm(),
        ax=ax,
        xlabel="Days",
        ylabel="Velocity",
        legend_title="Resolutions",
    )

    assert len(lines) == 2
    assert ax.get_xlabel() == "Days"
    assert ax.get_ylabel() == "Velocity"
    assert ax.get_legend().get_title().get_text() == "Resolutions"
    plt.close(fig)


def test_surface_3d_creates_axis_and_sets_custom_labels() -> None:
    """Surface plot should create a 3D axis when ax is omitted."""
    surface = plot_ucm_surface_3d(
        _sample_ucm(),
        show_hist_colorbar=False,
        xlabel="Days",
        ylabel="Resolution",
        zlabel="Velocity",
    )

    assert surface.axes.get_xlabel() == "Days"
    assert surface.axes.get_ylabel() == "Resolution"
    assert surface.axes.get_zlabel() == "Velocity"
    plt.close(surface.figure)


def test_plot_ucm_forwards_custom_panel_labels() -> None:
    """Composite UCM plot should forward custom labels to panel functions."""
    fig, axes = plot_ucm(
        _sample_ucm(),
        xlabel={
            "heatmap": "Heatmap days",
            "spatial": "Spatial velocity",
            "temporal": "Temporal days",
            "surface_3d": "Surface days",
        },
        ylabel={
            "heatmap": "Heatmap resolution",
            "spatial": "Spatial resolution",
            "temporal": "Temporal velocity",
            "surface_3d": "Surface resolution",
        },
        zlabel={"surface_3d": "Surface velocity"},
        legend_title={
            "spatial": "Spatial legend",
            "temporal": "Temporal legend",
        },
    )

    assert axes["heatmap"].get_xlabel() == "Heatmap days"
    assert axes["heatmap"].get_ylabel() == "Heatmap resolution"
    assert axes["spatial"].get_xlabel() == "Spatial velocity"
    assert axes["spatial"].get_ylabel() == "Spatial resolution"
    assert axes["spatial"].get_legend().get_title().get_text() == "Spatial legend"
    assert axes["temporal"].get_xlabel() == "Temporal days"
    assert axes["temporal"].get_ylabel() == "Temporal velocity"
    assert axes["temporal"].get_legend().get_title().get_text() == "Temporal legend"
    assert axes["surface_3d"].get_xlabel() == "Surface days"
    assert axes["surface_3d"].get_ylabel() == "Surface resolution"
    assert axes["surface_3d"].get_zlabel() == "Surface velocity"
    plt.close(fig)


def test_plot_ucm_maps_quantity_label_to_value_labels() -> None:
    """Composite UCM plot should map quantity label to value-axis defaults."""
    fig, axes = plot_ucm(_sample_ucm(), quantity_label="Custom velocity")

    assert axes["spatial"].get_xlabel() == "Custom velocity"
    assert axes["temporal"].get_ylabel() == "Custom velocity"
    assert axes["surface_3d"].get_zlabel() == "Custom velocity"
    plt.close(fig)
