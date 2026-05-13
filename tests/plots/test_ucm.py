"""Tests for UCM plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from faninsar.plots.ucm import UCM


def _sample_ucm() -> xr.DataArray:
    """Create sample UCM data for plotting tests."""
    return xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]),
        dims=("res", "day"),
        coords={"res": [30, 60], "day": [12, 24, 36]},
        name="velocity",
    )


def test_ucm_init_accepts_custom_dimension_names() -> None:
    """UCM initialization should standardize custom dimension names."""
    data_array = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]),
        dims=("resolution", "baseline"),
        coords={"resolution": [30, 60], "baseline": [12, 24, 36]},
        name="velocity",
    )

    ucm = UCM(data_array, spatial_dim="resolution", temporal_dim="baseline")

    assert ucm.ds_ucm.dims == ("res", "day")
    np.testing.assert_array_equal(ucm.ds_ucm.res.values, [30, 60])
    np.testing.assert_array_equal(ucm.ds_ucm.day.values, [12, 24, 36])
    np.testing.assert_allclose(ucm.ds_ucm.values, _sample_ucm().values)


def test_ucm_init_transposes_custom_dimension_order() -> None:
    """UCM initialization should transpose custom dimensions into standard order."""
    data_array = xr.DataArray(
        np.array([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]),
        dims=("baseline", "resolution"),
        coords={"baseline": [12, 24, 36], "resolution": [30, 60]},
        name="velocity",
    )

    ucm = UCM(data_array, spatial_dim="resolution", temporal_dim="baseline")

    assert ucm.ds_ucm.dims == ("res", "day")
    np.testing.assert_allclose(ucm.ds_ucm.values, _sample_ucm().values)


def test_ucm_init_rejects_missing_custom_dimension() -> None:
    """UCM initialization should reject data missing configured dimensions."""
    data_array = xr.DataArray(
        np.ones((2, 3)),
        dims=("resolution", "day"),
    )

    with pytest.raises(ValueError, match="ds_ucm must contain"):
        UCM(data_array, spatial_dim="res", temporal_dim="day")


def test_ucm_init_rejects_identical_dimension_names() -> None:
    """UCM initialization should reject identical spatial and temporal dimensions."""
    with pytest.raises(ValueError, match="must be different"):
        UCM(_sample_ucm(), spatial_dim="res", temporal_dim="res")


def test_ucm_init_rejects_three_dimensional_data() -> None:
    """UCM initialization should reject data with extra dimensions."""
    data_array = xr.DataArray(
        np.ones((2, 3, 4)),
        dims=("res", "day", "sample"),
    )

    with pytest.raises(ValueError, match="two-dimensional"):
        UCM(data_array)


def test_ucm_from_array_accepts_spatial_temporal_order() -> None:
    """from_array should create UCM data from spatial-temporal arrays."""
    array = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])

    ucm = UCM.from_array(
        array,
        spatial_coords=[30, 60],
        temporal_coords=[12, 24, 36],
        input_dims="st",
        name="velocity",
        attrs={"units": "mm/yr"},
    )

    assert ucm.ds_ucm.dims == ("res", "day")
    assert ucm.ds_ucm.name == "velocity"
    assert ucm.ds_ucm.attrs["units"] == "mm/yr"
    np.testing.assert_array_equal(ucm.ds_ucm.res.values, [30, 60])
    np.testing.assert_array_equal(ucm.ds_ucm.day.values, [12, 24, 36])
    np.testing.assert_allclose(ucm.ds_ucm.values, array)


def test_ucm_from_array_accepts_temporal_spatial_order() -> None:
    """from_array should transpose temporal-spatial arrays into standard order."""
    array = np.array([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])

    ucm = UCM.from_array(
        array,
        spatial_coords=[30, 60],
        temporal_coords=[12, 24, 36],
        input_dims="ts",
    )

    assert ucm.ds_ucm.dims == ("res", "day")
    np.testing.assert_array_equal(ucm.ds_ucm.res.values, [30, 60])
    np.testing.assert_array_equal(ucm.ds_ucm.day.values, [12, 24, 36])
    np.testing.assert_allclose(ucm.ds_ucm.values, _sample_ucm().values)


def test_ucm_from_array_uses_default_coordinates() -> None:
    """from_array should generate integer coordinates when coords are omitted."""
    ucm = UCM.from_array(np.ones((2, 3)))

    np.testing.assert_array_equal(ucm.ds_ucm.res.values, [0, 1])
    np.testing.assert_array_equal(ucm.ds_ucm.day.values, [0, 1, 2])


def test_ucm_from_array_rejects_coordinate_length_mismatch() -> None:
    """from_array should reject coordinates that do not match array axes."""
    with pytest.raises(ValueError, match="spatial_coords"):
        UCM.from_array(
            np.ones((2, 3)),
            spatial_coords=[30],
            temporal_coords=[12, 24, 36],
        )


def test_ucm_from_array_rejects_non_two_dimensional_array() -> None:
    """from_array should reject non-two-dimensional arrays."""
    with pytest.raises(ValueError, match="array must be two-dimensional"):
        UCM.from_array(np.ones((2, 3, 4)))


def test_ucm_from_array_rejects_invalid_input_dims() -> None:
    """from_array should reject unsupported input dimension order values."""
    with pytest.raises(ValueError, match="input_dims"):
        UCM.from_array(np.ones((2, 3)), input_dims="xy")


def test_heatmap_creates_axis_and_sets_custom_labels() -> None:
    """Heatmap should create an axis when ax is omitted."""
    image = UCM(_sample_ucm()).plot_heatmap(
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
    lines = UCM(_sample_ucm()).plot_sprofile(
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
    lines = UCM(_sample_ucm()).plot_tprofile(
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
    surface = UCM(_sample_ucm()).plot_3d_surface(
        show_hist_colorbar=False,
        xlabel="Days",
        ylabel="Resolution",
        zlabel="Velocity",
    )

    assert surface.axes.get_xlabel() == "Days"
    assert surface.axes.get_ylabel() == "Resolution"
    assert surface.axes.get_zlabel() == "Velocity"
    plt.close(surface.figure)


def test_ucm_plot_forwards_custom_panel_labels() -> None:
    """Composite UCM plot should forward custom labels to panel methods."""
    fig, axes = UCM(_sample_ucm()).plot(
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


def test_ucm_plot_maps_quantity_label_to_value_labels() -> None:
    """Composite UCM plot should map quantity label to value-axis defaults."""
    fig, axes = UCM(_sample_ucm(), quantity_label="Custom velocity").plot()

    assert axes["spatial"].get_xlabel() == "Custom velocity"
    assert (
        axes["spatial"].get_legend().get_title().get_text()
        == "Maximum Temporal Baseline (days)"
    )
    assert axes["temporal"].get_ylabel() == "Custom velocity"
    assert axes["surface_3d"].get_zlabel() == "Custom velocity"
    plt.close(fig)


def test_ucm_plot_heatmap_uses_init_labels() -> None:
    """UCM heatmap method should use labels configured at initialization."""
    ucm = UCM(
        _sample_ucm(),
        quantity_label="Velocity",
        spatial_label="Grid size",
        temporal_label="Days",
    )
    image = ucm.plot_heatmap(show_hist_colorbar=False)

    assert image.axes.get_xlabel() == "Days"
    assert image.axes.get_ylabel() == "Grid size"
    plt.close(image.figure)


def test_ucm_profile_methods_use_init_labels() -> None:
    """UCM profile methods should map initialized labels to axes and legends."""
    ucm = UCM(
        _sample_ucm(),
        quantity_label="Velocity",
        spatial_label="Grid size",
        temporal_label="Days",
    )

    fig, axes = plt.subplots(1, 2)
    spatial_lines = ucm.plot_sprofile(ax=axes[0])
    temporal_lines = ucm.plot_tprofile(ax=axes[1])

    assert len(spatial_lines) == 3
    assert axes[0].get_xlabel() == "Velocity"
    assert axes[0].get_ylabel() == "Grid size"
    assert axes[0].get_legend().get_title().get_text() == "Days"
    assert len(temporal_lines) == 2
    assert axes[1].get_xlabel() == "Days"
    assert axes[1].get_ylabel() == "Velocity"
    assert axes[1].get_legend().get_title().get_text() == "Grid size"
    plt.close(fig)


def test_ucm_profile_methods_accept_custom_cmap() -> None:
    """UCM profile methods should sample colors from the requested colormap."""
    ucm = UCM(_sample_ucm())

    fig, axes = plt.subplots(1, 2)
    spatial_lines = ucm.plot_sprofile(ax=axes[0], cmap="viridis")
    temporal_lines = ucm.plot_tprofile(ax=axes[1], cmap="plasma")

    np.testing.assert_allclose(
        spatial_lines[0][0].get_color(),
        plt.get_cmap("viridis", 3)(0),
    )
    np.testing.assert_allclose(
        temporal_lines[0][0].get_color(),
        plt.get_cmap("plasma", 2)(0),
    )
    plt.close(fig)


def test_ucm_3d_surface_uses_init_labels() -> None:
    """UCM 3D surface method should use labels configured at initialization."""
    ucm = UCM(
        _sample_ucm(),
        quantity_label="Velocity",
        spatial_label="Grid size",
        temporal_label="Days",
    )
    surface = ucm.plot_3d_surface(show_hist_colorbar=False)

    assert surface.axes.get_xlabel() == "Days"
    assert surface.axes.get_ylabel() == "Grid size"
    assert surface.axes.get_zlabel() == "Velocity"
    plt.close(surface.figure)


def test_ucm_methods_allow_per_call_label_overrides() -> None:
    """UCM methods should prefer per-call label overrides over init labels."""
    ucm = UCM(
        _sample_ucm(),
        quantity_label="Velocity",
        spatial_label="Grid size",
        temporal_label="Days",
    )

    fig, ax = plt.subplots()
    ucm.plot_sprofile(
        ax=ax,
        xlabel="Custom quantity",
        ylabel="Custom spatial",
        legend_title="Custom temporal",
    )

    assert ax.get_xlabel() == "Custom quantity"
    assert ax.get_ylabel() == "Custom spatial"
    assert ax.get_legend().get_title().get_text() == "Custom temporal"
    plt.close(fig)


def test_ucm_plot_uses_init_labels_and_mapping_overrides() -> None:
    """UCM composite plot should merge init labels with mapping overrides."""
    ucm = UCM(
        _sample_ucm(),
        quantity_label="Velocity",
        spatial_label="Grid size",
        temporal_label="Days",
    )
    fig, axes = ucm.plot(xlabel={"heatmap": "Custom heatmap days"})

    assert axes["heatmap"].get_xlabel() == "Custom heatmap days"
    assert axes["heatmap"].get_ylabel() == "Grid size"
    assert axes["spatial"].get_xlabel() == "Velocity"
    assert axes["spatial"].get_ylabel() == "Grid size"
    assert axes["spatial"].get_legend().get_title().get_text() == "Days"
    assert axes["temporal"].get_xlabel() == "Days"
    assert axes["temporal"].get_ylabel() == "Velocity"
    assert axes["temporal"].get_legend().get_title().get_text() == "Grid size"
    assert axes["surface_3d"].get_xlabel() == "Days"
    assert axes["surface_3d"].get_ylabel() == "Grid size"
    assert axes["surface_3d"].get_zlabel() == "Velocity"
    plt.close(fig)
