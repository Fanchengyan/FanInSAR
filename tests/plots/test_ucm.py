"""Tests for UCM plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.ticker import NullFormatter

from faninsar.plots.ucm import UCM


def _sample_ucm() -> xr.DataArray:
    """Create sample UCM data for plotting tests."""
    return xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]),
        dims=("res", "day"),
        coords={"res": [30, 60], "day": [12, 24, 36]},
        name="velocity",
    )


def _dense_sample_ucm() -> xr.DataArray:
    """Create dense UCM data for 3D surface tick tests."""
    resolutions = np.arange(30, 331, 30)
    days = np.arange(12, 253, 12)
    values = resolutions[:, np.newaxis] * 0.02 + days[np.newaxis, :] * 0.005
    return xr.DataArray(
        values,
        dims=("res", "day"),
        coords={"res": resolutions, "day": days},
        name="velocity",
    )


_CANONICAL_ORIGIN_EXPECTATIONS = [
    ("upper_left", False, True),
    ("upper_right", True, True),
    ("lower_left", False, False),
    ("lower_right", True, False),
]


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


def test_heatmap_uses_lower_left_origin_by_default() -> None:
    """Heatmap should default to a lower-left visual origin."""
    image = UCM(_sample_ucm()).plot_heatmap(show_hist_colorbar=False)

    assert bool(image.axes.xaxis_inverted()) is False
    assert bool(image.axes.yaxis_inverted()) is False
    plt.close(image.figure)


@pytest.mark.parametrize(
    ("origin", "invert_x", "invert_y"),
    _CANONICAL_ORIGIN_EXPECTATIONS,
)
def test_heatmap_supports_all_canonical_origins(
    origin: str,
    invert_x: bool,
    invert_y: bool,
) -> None:
    """Heatmap should support all canonical corner origin settings."""
    image = UCM(_sample_ucm()).plot_heatmap(
        origin=origin,
        show_hist_colorbar=False,
    )

    assert bool(image.axes.xaxis_inverted()) is invert_x
    assert bool(image.axes.yaxis_inverted()) is invert_y
    plt.close(image.figure)


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("ul", "upper_left"),
        ("ll", "lower_left"),
        ("ur", "upper_right"),
        ("lr", "lower_right"),
    ],
)
def test_heatmap_origin_aliases_match_canonical_origins(
    alias: str,
    canonical: str,
) -> None:
    """Heatmap origin aliases should behave like their canonical values."""
    alias_image = UCM(_sample_ucm()).plot_heatmap(
        origin=alias,
        show_hist_colorbar=False,
    )
    canonical_image = UCM(_sample_ucm()).plot_heatmap(
        origin=canonical,
        show_hist_colorbar=False,
    )

    assert bool(alias_image.axes.xaxis_inverted()) == bool(
        canonical_image.axes.xaxis_inverted()
    )
    assert bool(alias_image.axes.yaxis_inverted()) == bool(
        canonical_image.axes.yaxis_inverted()
    )
    plt.close(alias_image.figure)
    plt.close(canonical_image.figure)


@pytest.mark.parametrize("method_name", ["plot_heatmap", "plot_3d_surface"])
def test_ucm_origin_rejects_invalid_values(method_name: str) -> None:
    """Origin-aware UCM methods should reject unsupported origin values."""
    ucm = UCM(_sample_ucm())

    with pytest.raises(ValueError, match="origin must be one of"):
        getattr(ucm, method_name)(origin="UL", show_hist_colorbar=False)


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


def test_surface_3d_uses_lower_left_origin_by_default() -> None:
    """3D surface should default to a lower-left visual origin."""
    surface = UCM(_sample_ucm()).plot_3d_surface(show_hist_colorbar=False)

    assert bool(surface.axes.xaxis_inverted()) is False
    assert bool(surface.axes.yaxis_inverted()) is False
    plt.close(surface.figure)


def test_surface_3d_uses_sparse_auto_major_ticks_for_dense_data() -> None:
    """3D surface should use sparse major ticks and unlabeled minor ticks."""
    dense_ucm = _dense_sample_ucm()
    surface = UCM(dense_ucm).plot_3d_surface(show_hist_colorbar=False)

    x_major_ticks = surface.axes.get_xticks()
    y_major_ticks = surface.axes.get_yticks()
    x_minor_ticks = surface.axes.xaxis.get_minorticklocs()
    y_minor_ticks = surface.axes.yaxis.get_minorticklocs()

    assert len(x_major_ticks) < len(dense_ucm.day.values)
    assert len(y_major_ticks) < len(dense_ucm.res.values)
    assert np.isin(x_major_ticks, dense_ucm.day.values).all()
    assert np.isin(y_major_ticks, dense_ucm.res.values).all()
    assert np.isin(x_minor_ticks, dense_ucm.day.values).all()
    assert np.isin(y_minor_ticks, dense_ucm.res.values).all()
    np.testing.assert_allclose([x_major_ticks[0], x_major_ticks[-1]], [12, 252])
    np.testing.assert_allclose([y_major_ticks[0], y_major_ticks[-1]], [30, 330])
    assert len(x_minor_ticks) == len(dense_ucm.day.values) - len(x_major_ticks)
    assert len(y_minor_ticks) == len(dense_ucm.res.values) - len(y_major_ticks)
    assert isinstance(surface.axes.xaxis.get_minor_formatter(), NullFormatter)
    assert isinstance(surface.axes.yaxis.get_minor_formatter(), NullFormatter)
    plt.close(surface.figure)


def test_surface_3d_all_major_ticks_restore_previous_behavior() -> None:
    """3D surface should support restoring all coordinates as major ticks."""
    dense_ucm = _dense_sample_ucm()
    surface = UCM(dense_ucm).plot_3d_surface(
        show_hist_colorbar=False,
        x_major_tick_count="all",
        y_major_tick_count="all",
    )

    np.testing.assert_array_equal(surface.axes.get_xticks(), dense_ucm.day.values)
    np.testing.assert_array_equal(surface.axes.get_yticks(), dense_ucm.res.values)
    assert surface.axes.xaxis.get_minorticklocs().size == 0
    assert surface.axes.yaxis.get_minorticklocs().size == 0
    plt.close(surface.figure)


def test_surface_3d_supports_explicit_major_tick_counts() -> None:
    """3D surface should use the requested number of major ticks."""
    dense_ucm = _dense_sample_ucm()
    surface = UCM(dense_ucm).plot_3d_surface(
        show_hist_colorbar=False,
        x_major_tick_count=5,
        y_major_tick_count=4,
    )

    x_major_ticks = surface.axes.get_xticks()
    y_major_ticks = surface.axes.get_yticks()

    assert len(x_major_ticks) == 5
    assert len(y_major_ticks) == 4
    np.testing.assert_allclose([x_major_ticks[0], x_major_ticks[-1]], [12, 252])
    np.testing.assert_allclose([y_major_ticks[0], y_major_ticks[-1]], [30, 330])
    assert len(surface.axes.xaxis.get_minorticklocs()) == len(dense_ucm.day.values) - 5
    assert len(surface.axes.yaxis.get_minorticklocs()) == len(dense_ucm.res.values) - 4
    assert isinstance(surface.axes.xaxis.get_minor_formatter(), NullFormatter)
    assert isinstance(surface.axes.yaxis.get_minor_formatter(), NullFormatter)
    plt.close(surface.figure)


@pytest.mark.parametrize(
    "tick_kwargs",
    [
        {"x_major_tick_count": 0},
        {"y_major_tick_count": -1},
        {"x_major_tick_count": "invalid"},
    ],
)
def test_surface_3d_rejects_invalid_major_tick_counts(
    tick_kwargs: dict[str, int | str],
) -> None:
    """3D surface should reject unsupported major tick count values."""
    with pytest.raises(ValueError, match="major_tick_count"):
        UCM(_dense_sample_ucm()).plot_3d_surface(
            show_hist_colorbar=False,
            **tick_kwargs,
        )


def test_surface_3d_small_axes_fall_back_to_all_major_ticks() -> None:
    """Small UCM axes should not create minor ticks in auto mode."""
    sample_ucm = _sample_ucm()
    surface = UCM(sample_ucm).plot_3d_surface(show_hist_colorbar=False)

    np.testing.assert_array_equal(surface.axes.get_xticks(), sample_ucm.day.values)
    np.testing.assert_array_equal(surface.axes.get_yticks(), sample_ucm.res.values)
    assert surface.axes.xaxis.get_minorticklocs().size == 0
    assert surface.axes.yaxis.get_minorticklocs().size == 0
    plt.close(surface.figure)


@pytest.mark.parametrize(
    ("origin", "invert_x", "invert_y"),
    _CANONICAL_ORIGIN_EXPECTATIONS,
)
def test_surface_3d_supports_all_canonical_origins(
    origin: str,
    invert_x: bool,
    invert_y: bool,
) -> None:
    """3D surface should support all canonical corner origin settings."""
    surface = UCM(_sample_ucm()).plot_3d_surface(
        origin=origin,
        show_hist_colorbar=False,
    )

    assert bool(surface.axes.xaxis_inverted()) is invert_x
    assert bool(surface.axes.yaxis_inverted()) is invert_y
    plt.close(surface.figure)


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("ul", "upper_left"),
        ("ll", "lower_left"),
        ("ur", "upper_right"),
        ("lr", "lower_right"),
    ],
)
def test_surface_3d_origin_aliases_match_canonical_origins(
    alias: str,
    canonical: str,
) -> None:
    """3D surface origin aliases should behave like their canonical values."""
    alias_surface = UCM(_sample_ucm()).plot_3d_surface(
        origin=alias,
        show_hist_colorbar=False,
    )
    canonical_surface = UCM(_sample_ucm()).plot_3d_surface(
        origin=canonical,
        show_hist_colorbar=False,
    )

    assert bool(alias_surface.axes.xaxis_inverted()) == bool(
        canonical_surface.axes.xaxis_inverted()
    )
    assert bool(alias_surface.axes.yaxis_inverted()) == bool(
        canonical_surface.axes.yaxis_inverted()
    )
    plt.close(alias_surface.figure)
    plt.close(canonical_surface.figure)


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


def test_ucm_plot_forwards_origin_to_heatmap_and_surface() -> None:
    """Composite UCM plot should forward origin to heatmap and 3D surface."""
    fig, axes = UCM(_sample_ucm()).plot(
        origin="ur",
        show_hist_colorbar=False,
    )

    assert bool(axes["heatmap"].xaxis_inverted()) is True
    assert bool(axes["heatmap"].yaxis_inverted()) is True
    assert bool(axes["surface_3d"].xaxis_inverted()) is True
    assert bool(axes["surface_3d"].yaxis_inverted()) is True
    plt.close(fig)


def test_ucm_plot_forwards_surface_major_tick_counts() -> None:
    """Composite UCM plot should forward 3D surface tick settings."""
    fig, axes = UCM(_dense_sample_ucm()).plot(
        show_hist_colorbar=False,
        surface_x_major_tick_count=4,
        surface_y_major_tick_count=3,
    )

    assert len(axes["surface_3d"].get_xticks()) == 4
    assert len(axes["surface_3d"].get_yticks()) == 3
    assert len(axes["surface_3d"].xaxis.get_minorticklocs()) == 17
    assert len(axes["surface_3d"].yaxis.get_minorticklocs()) == 8
    plt.close(fig)


def test_ucm_plot_maps_quantity_label_to_value_labels() -> None:
    """Composite UCM plot should map quantity label to value-axis defaults."""
    fig, axes = UCM(_sample_ucm(), quantity_label="Custom velocity").plot()

    assert axes["spatial"].get_xlabel() == "Resolution (m)"
    assert axes["spatial"].get_ylabel() == "Custom velocity"
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
    assert axes[0].get_xlabel() == "Grid size"
    assert axes[0].get_ylabel() == "Velocity"
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


def test_ucm_spatial_profile_variable_x_axis_uses_resolution_data() -> None:
    """Spatial profile should use resolution on x-axis in variable mode."""
    fig, ax = plt.subplots()
    lines = UCM(_sample_ucm()).plot_sprofile(ax=ax, x_axis="variable")

    np.testing.assert_array_equal(lines[0][0].get_xdata(), [30, 60])
    np.testing.assert_allclose(lines[0][0].get_ydata(), [1.0, 1.5])
    assert ax.get_xlabel() == "Resolution (m)"
    assert ax.get_ylabel() == "Velocity (mm/yr)"
    plt.close(fig)


def test_ucm_spatial_profile_quantity_x_axis_preserves_legacy_data() -> None:
    """Spatial profile should use quantity on x-axis in quantity mode."""
    fig, ax = plt.subplots()
    lines = UCM(_sample_ucm()).plot_sprofile(ax=ax, x_axis="quantity")

    np.testing.assert_allclose(lines[0][0].get_xdata(), [1.0, 1.5])
    np.testing.assert_array_equal(lines[0][0].get_ydata(), [30, 60])
    assert ax.get_xlabel() == "Velocity (mm/yr)"
    assert ax.get_ylabel() == "Resolution (m)"
    plt.close(fig)


def test_ucm_temporal_profile_variable_x_axis_uses_day_data() -> None:
    """Temporal profile should use day on x-axis in variable mode."""
    fig, ax = plt.subplots()
    lines = UCM(_sample_ucm()).plot_tprofile(ax=ax, x_axis="variable")

    np.testing.assert_array_equal(lines[0][0].get_xdata(), [12, 24, 36])
    np.testing.assert_allclose(lines[0][0].get_ydata(), [1.0, 2.0, 3.0])
    assert ax.get_xlabel() == "Maximum Temporal Baseline (days)"
    assert ax.get_ylabel() == "Velocity (mm/yr)"
    plt.close(fig)


def test_ucm_temporal_profile_quantity_x_axis_uses_quantity_data() -> None:
    """Temporal profile should use quantity on x-axis in quantity mode."""
    fig, ax = plt.subplots()
    lines = UCM(_sample_ucm()).plot_tprofile(ax=ax, x_axis="quantity")

    np.testing.assert_allclose(lines[0][0].get_xdata(), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(lines[0][0].get_ydata(), [12, 24, 36])
    assert ax.get_xlabel() == "Velocity (mm/yr)"
    assert ax.get_ylabel() == "Maximum Temporal Baseline (days)"
    plt.close(fig)


def test_ucm_profile_methods_reject_invalid_x_axis() -> None:
    """Profile methods should reject unsupported x-axis modes."""
    with pytest.raises(ValueError, match="x_axis"):
        UCM(_sample_ucm()).plot_sprofile(x_axis="invalid")


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
    assert axes["spatial"].get_xlabel() == "Grid size"
    assert axes["spatial"].get_ylabel() == "Velocity"
    assert axes["spatial"].get_legend().get_title().get_text() == "Days"
    assert axes["temporal"].get_xlabel() == "Days"
    assert axes["temporal"].get_ylabel() == "Velocity"
    assert axes["temporal"].get_legend().get_title().get_text() == "Grid size"
    assert axes["surface_3d"].get_xlabel() == "Days"
    assert axes["surface_3d"].get_ylabel() == "Grid size"
    assert axes["surface_3d"].get_zlabel() == "Velocity"
    plt.close(fig)


def test_ucm_plot_forwards_profile_x_axis_mapping() -> None:
    """Composite plot should forward per-panel profile x-axis modes."""
    fig, axes = UCM(_sample_ucm()).plot(
        profile_x_axis={"spatial": "quantity", "temporal": "quantity"}
    )

    assert axes["spatial"].get_xlabel() == "Velocity (mm/yr)"
    assert axes["spatial"].get_ylabel() == "Resolution (m)"
    np.testing.assert_allclose(axes["spatial"].lines[0].get_xdata(), [1.0, 1.5])
    np.testing.assert_array_equal(axes["spatial"].lines[0].get_ydata(), [30, 60])
    assert axes["temporal"].get_xlabel() == "Velocity (mm/yr)"
    assert axes["temporal"].get_ylabel() == "Maximum Temporal Baseline (days)"
    np.testing.assert_allclose(axes["temporal"].lines[0].get_xdata(), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(axes["temporal"].lines[0].get_ydata(), [12, 24, 36])
    plt.close(fig)
