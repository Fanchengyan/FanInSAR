"""ProductLayer.plot colorbar placement policy."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from faninsar.network.layers import _resolve_colorbar_location
from tests.network.test_reference_plot import _geo_unwrap_layer

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _colorbar_container(fig: Figure) -> Axes:
    """Return the colorbar container axes created by HistColorbar."""
    return next(ax for ax in fig.axes if ax.get_label() == "<colorbar>")


def test_auto_places_wide_image_bottom(tmp_path: Path) -> None:
    """A wide displayed data box puts the colorbar below the image."""
    layer = _geo_unwrap_layer(tmp_path, np.ones((2, 12), dtype=np.float32))
    fig, _ax, _im = layer.plot(crop=False)
    fig.canvas.draw()
    container = _colorbar_container(fig)
    position = container.get_position(original=False)
    assert position.width > position.height


def test_auto_places_square_image_right(tmp_path: Path) -> None:
    """A non-wide data box puts the colorbar to the right."""
    layer = _geo_unwrap_layer(tmp_path, np.ones((6, 6), dtype=np.float32))
    fig, ax, _im = layer.plot(crop=False)
    fig.canvas.draw()
    container = _colorbar_container(fig)
    position = container.get_position(original=False)
    visible = ax.get_position(original=False)
    assert position.height > position.width
    assert abs(position.height / visible.height - 1.0) < 0.02


def test_explicit_right_matches_visible_height(tmp_path: Path) -> None:
    """An explicit right colorbar matches the displayed image height."""
    layer = _geo_unwrap_layer(tmp_path, np.ones((2, 12), dtype=np.float32))
    fig, ax, _im = layer.plot(crop=False, colorbar_location="right")
    fig.canvas.draw()
    container = _colorbar_container(fig)
    position = container.get_position(original=False)
    visible = ax.get_position(original=False)
    assert position.height > position.width
    assert abs(position.height / visible.height - 1.0) < 0.02


def test_match_extent_false_restores_full_cell(tmp_path: Path) -> None:
    """Disabling extent matching keeps the old full-cell colorbar length."""
    layer = _geo_unwrap_layer(tmp_path, np.ones((2, 12), dtype=np.float32))
    fig, ax, _im = layer.plot(
        crop=False,
        colorbar_location="right",
        colorbar_kwargs={"match_extent": False},
    )
    fig.canvas.draw()
    container = _colorbar_container(fig)
    position = container.get_position(original=False)
    cell = ax.get_position(original=True)
    assert abs(position.height / cell.height - 1.0) < 0.02


def test_reserved_colorbar_kwargs_rejected(tmp_path: Path) -> None:
    """Placement keys cannot be smuggled through colorbar_kwargs."""
    layer = _geo_unwrap_layer(tmp_path, np.ones((6, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="colorbar_kwargs"):
        layer.plot(crop=False, colorbar_kwargs={"location": "bottom"})


def test_plain_colorbar_matches_visible_height(tmp_path: Path) -> None:
    """A plain colorbar also matches the displayed image height."""
    layer = _geo_unwrap_layer(tmp_path, np.ones((2, 12), dtype=np.float32))
    fig, ax, _im = layer.plot(
        crop=False,
        hist_colorbar=False,
        colorbar_location="right",
    )
    fig.canvas.draw()
    container = _colorbar_container(fig)
    position = container.get_position(original=False)
    visible = ax.get_position(original=False)
    assert abs(position.height / visible.height - 1.0) < 0.02


def test_plain_colorbar_match_can_be_disabled(tmp_path: Path) -> None:
    """``match_extent=False`` restores the full-cell plain colorbar."""
    layer = _geo_unwrap_layer(tmp_path, np.ones((2, 12), dtype=np.float32))
    fig, ax, _im = layer.plot(
        crop=False,
        hist_colorbar=False,
        colorbar_location="right",
        colorbar_kwargs={"match_extent": False},
    )
    fig.canvas.draw()
    container = _colorbar_container(fig)
    position = container.get_position(original=False)
    cell = ax.get_position(original=True)
    assert abs(position.height / cell.height - 1.0) < 0.02


def test_auto_avoids_bottom_under_constrained_layout() -> None:
    """constrained_layout keeps ``auto`` on the right side."""
    fig, ax = plt.subplots(layout="constrained")
    try:
        ax.imshow(np.zeros((2, 12)))
        assert _resolve_colorbar_location(ax, "auto") == "right"
    finally:
        plt.close(fig)


def test_layer_plot_auto_under_constrained_stays_right(tmp_path: Path) -> None:
    """A constrained figure gets a vertical colorbar even for wide data."""
    layer = _geo_unwrap_layer(tmp_path, np.ones((2, 12), dtype=np.float32))
    fig, ax = plt.subplots(layout="constrained")
    try:
        fig, ax, _im = layer.plot(ax=ax, crop=False)
        fig.canvas.draw()
        position = _colorbar_container(fig).get_position(original=False)
        assert position.height > position.width
    finally:
        plt.close(fig)


def test_constrained_bottom_warns(tmp_path: Path) -> None:
    """Explicit bottom under constrained_layout warns about approximation."""
    layer = _geo_unwrap_layer(tmp_path, np.ones((2, 12), dtype=np.float32))
    fig, ax = plt.subplots(layout="constrained")
    try:
        with pytest.warns(UserWarning, match="constrained_layout"):
            layer.plot(ax=ax, crop=False, colorbar_location="bottom")
    finally:
        plt.close(fig)


def test_constrained_narrow_right_warns(tmp_path: Path) -> None:
    """A narrow image with a right colorbar warns about horizontal offset."""
    layer = _geo_unwrap_layer(tmp_path, np.ones((12, 2), dtype=np.float32))
    fig, ax = plt.subplots(layout="constrained")
    try:
        with pytest.warns(UserWarning, match="offset horizontally"):
            layer.plot(ax=ax, crop=False, colorbar_location="right")
    finally:
        plt.close(fig)


def test_constrained_wide_right_does_not_warn(tmp_path: Path) -> None:
    """A wide image filling the cell width does not warn for a right bar."""
    layer = _geo_unwrap_layer(tmp_path, np.ones((2, 12), dtype=np.float32))
    fig, ax = plt.subplots(layout="constrained")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            layer.plot(ax=ax, crop=False, colorbar_location="right")
        assert not [
            item for item in caught if "constrained_layout" in str(item.message)
        ]
    finally:
        plt.close(fig)
