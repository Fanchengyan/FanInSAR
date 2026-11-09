"""ProductLayer.plot reference-point markers and display subtraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pytest

from faninsar.data.query.points import Points
from faninsar.io.storage.ifg_store import write_ifg_artifact, write_unwrapped_artifact
from faninsar.io.storage.product import open_product
from faninsar.network.layers import ProductLayer
from faninsar.network.products import AssetKind

if TYPE_CHECKING:
    from pathlib import Path

GEO_TRANSFORM = (1.0, 0.0, 10.0, 0.0, -1.0, 50.0)
GRID_IDENTITY = "a" * 64


def _geo_unwrap_layer(
    tmp_path: Path,
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    kind: AssetKind = AssetKind.UNWRAPPED_PHASE,
    name: str = "unwrapped_phase",
) -> ProductLayer:
    pair = ("20240101", "20240113")
    ifg_root = tmp_path / "ifg" / "ml_1x1" / "20240101_20240113"
    store = write_ifg_artifact(
        ifg_root,
        pair=pair,
        looks=(1, 1),
        domain="geo",
        grid_identity=GRID_IDENTITY,
        crs="EPSG:4326",
        transform=GEO_TRANSFORM,
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"primary": "a" * 64, "secondary": "b" * 64},
        complex_ifg=np.exp(1j * np.nan_to_num(values.astype(np.float32))).astype(
            np.complex64
        ),
        coherence=np.ones(values.shape, dtype=np.float32),
        valid_mask=mask,
    )
    extras = {} if mask is None else {"valid_mask": mask}
    write_unwrapped_artifact(
        store.root,
        unwrapped_phase=np.asarray(values, dtype=np.float32),
        connected_components=np.zeros(values.shape, dtype=np.int32),
        method="test",
        method_parameters={},
        ifg_manifest_digest=store.write_id,
        extra_layers=extras or None,
    )
    if kind is AssetKind.UNWRAPPED_PHASE:
        product = open_product(store.root / "unwrap")
        return ProductLayer(
            product,
            name,
            kind,
            values.shape,
            crs="EPSG:4326",
        )
    product = open_product(store.root)
    return ProductLayer(
        product, "coherence", AssetKind.COHERENCE, values.shape, crs="EPSG:4326"
    )


def test_plot_inherit_without_session_is_noop(tmp_path: Path) -> None:
    """Standalone inherit does not subtract or draw markers."""
    values = np.full((4, 6), 4.0, dtype=np.float32)
    layer = _geo_unwrap_layer(tmp_path, values)
    before = np.array(layer.read(), copy=True)
    _fig, _ax, im = layer.plot(
        ref_points="inherit", crop=False, hist_colorbar=False, colorbar=False
    )
    np.testing.assert_array_equal(layer.read(), before)
    np.testing.assert_allclose(np.asarray(im.get_array()), 4.0)


def test_plot_explicit_geo_points_subtract_and_markers(tmp_path: Path) -> None:
    """Explicit geographic points subtract before vmin/vmax and place markers."""
    values = np.full((4, 6), 4.0, dtype=np.float32)
    layer = _geo_unwrap_layer(tmp_path, values)
    before = np.array(layer.read(), copy=True)
    points = Points([[10.5, 49.5]], crs="EPSG:4326")
    _fig, ax, im = layer.plot(
        ref_points=points, crop=False, hist_colorbar=False, colorbar=False
    )
    np.testing.assert_array_equal(layer.read(), before)
    np.testing.assert_allclose(np.asarray(im.get_array()), 0.0)
    offsets = ax.collections[0].get_offsets()
    np.testing.assert_allclose(offsets[0], [10.5, 49.5])


def test_plot_offsets_independent_of_crop(tmp_path: Path) -> None:
    """Full-layer sampling does not follow the display crop window."""
    values = np.full((8, 8), 5.0, dtype=np.float32)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    layer = _geo_unwrap_layer(tmp_path, values, mask=mask)
    points = Points([[12.5, 47.5]], crs="EPSG:4326")
    _fig, _ax, cropped = layer.plot(
        ref_points=points,
        crop=True,
        masked=True,
        hist_colorbar=False,
        colorbar=False,
    )
    _fig2, _ax2, full = layer.plot(
        ref_points=points,
        crop=False,
        masked=False,
        hist_colorbar=False,
        colorbar=False,
    )
    np.testing.assert_allclose(np.nanmean(np.asarray(cropped.get_array())), 0.0)
    np.testing.assert_allclose(np.nanmean(np.asarray(full.get_array())), 0.0)


def test_plot_none_disables_and_coherence_is_not_subtracted(tmp_path: Path) -> None:
    """None disables overlays; coherence keeps its native values."""
    values = np.full((4, 6), 4.0, dtype=np.float32)
    phase = _geo_unwrap_layer(tmp_path / "phase", values)
    _fig, ax, im = phase.plot(
        ref_points=None,
        crop=False,
        hist_colorbar=False,
        colorbar=False,
    )
    np.testing.assert_allclose(np.asarray(im.get_array()), 4.0)
    assert len(ax.collections) == 0
    coherence = _geo_unwrap_layer(tmp_path / "coh", values, kind=AssetKind.COHERENCE)
    points = Points([[10.5, 49.5]], crs="EPSG:4326")
    _fig, _ax, coh_im = coherence.plot(
        ref_points=points, crop=False, hist_colorbar=False, colorbar=False
    )
    np.testing.assert_allclose(np.asarray(coh_im.get_array()), 1.0)


def test_plot_radar_without_geometry_errors(tmp_path: Path) -> None:
    """Standalone radar layers need geometry context for explicit points."""
    values = np.full((4, 6), 1.0, dtype=np.float32)
    pair = ("20240101", "20240113")
    store = write_ifg_artifact(
        tmp_path / "ifg" / "ml_1x1" / "20240101_20240113",
        pair=pair,
        looks=(1, 1),
        domain="radar",
        grid_identity=GRID_IDENTITY,
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"primary": "a" * 64, "secondary": "b" * 64},
        complex_ifg=np.ones(values.shape, dtype=np.complex64),
        coherence=np.ones(values.shape, dtype=np.float32),
    )
    write_unwrapped_artifact(
        store.root,
        unwrapped_phase=values,
        connected_components=np.zeros(values.shape, dtype=np.int32),
        method="test",
        method_parameters={},
        ifg_manifest_digest=store.write_id,
    )
    layer = ProductLayer(
        open_product(store.root / "unwrap"),
        "unwrapped_phase",
        AssetKind.UNWRAPPED_PHASE,
        values.shape,
    )
    with pytest.raises(ValueError, match="geometry"):
        layer.plot(
            ref_points=Points([[10.5, 49.5]], crs="EPSG:4326"),
            hist_colorbar=False,
            colorbar=False,
        )


def test_plot_iq_overlays_markers_without_subtraction(tmp_path: Path) -> None:
    """Complex layers overlay markers on both panels and keep native values."""
    values = np.full((4, 6), 4.0, dtype=np.float32)
    unwrap = _geo_unwrap_layer(tmp_path, values)
    product = open_product(unwrap.source.path.parent)
    layer = ProductLayer(
        product,
        "complex",
        AssetKind.COMPLEX_INTERFEROGRAM,
        values.shape,
        crs="EPSG:4326",
    )
    points = Points([[10.5, 49.5]], crs="EPSG:4326")
    before = np.array(layer.read(), copy=True)
    _fig, axes, images = layer.plot(
        ref_points=points, crop=False, hist_colorbar=False, colorbar=False
    )
    np.testing.assert_array_equal(layer.read(), before)
    np.testing.assert_allclose(
        np.abs(np.asarray(images[0].get_array())), np.abs(before.real)
    )
    assert len(axes[0].collections) == 1
    assert len(axes[1].collections) == 1
