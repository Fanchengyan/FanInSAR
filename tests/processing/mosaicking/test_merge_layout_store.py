"""merge_layout product roundtrip through the existing product protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.io.storage.merge_layout import (
    LAYOUT_KIND,
    open_merge_layout,
    publish_merge_layout,
)
from faninsar.io.storage.product import publish_product
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.mosaicking import (
    LayoutSourceWindow,
    MergeLayoutWindow,
    MergeScope,
    MergeSeam,
    build_merge_layout,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_dummy(staging: Path) -> dict[str, dict[str, object]]:
    from faninsar.io.storage._zarr_stac import write_zarr_array

    write_zarr_array(staging / "a.zarr", np.zeros((1, 1), dtype=np.float32))
    return {"a": {"href": "a.zarr"}}


def _radar_layout(*, hole: bool = True) -> MergeLayoutWindow:
    scope = MergeScope(
        domain="radar",
        track_id="T100_A",
        channel="vv",
        reference_id="ref-20240101",
        grid_identity="a" * 64,
        polygon_identity="poly-0",
    )
    support_a = np.ones((4, 3), dtype=bool)
    support_b = np.ones((4, 3), dtype=bool)
    if hole:
        support_a[0, 1] = False
        support_a[2, 1] = False
    sources = (
        LayoutSourceWindow("iw1-b1", (0, 4, 0, 3), support_a, "IW1"),
        LayoutSourceWindow("iw1-b2", (2, 6, 0, 3), support_b, "IW1"),
    )
    return build_merge_layout(
        sources,
        scope=scope,
        seams=(MergeSeam("azimuth", 3, "iw1-b1", "iw1-b2"),),
    )


def _geo_layout() -> MergeLayoutWindow:
    scope = MergeScope(
        domain="geo",
        track_id="T100_A",
        channel="vv",
        reference_id="ref-20240101",
        grid_identity="a" * 64,
        polygon_identity="poly-0",
    )
    source = LayoutSourceWindow(
        "iw1-b1",
        (0, 2, 0, 2),
        np.ones((2, 2), dtype=bool),
        "IW1",
    )
    return build_merge_layout([source], scope=scope)


def test_radar_compact_roundtrip_reconstructs_source_holes(tmp_path: Path) -> None:
    """Radar seams plus per-source support reopen without an owner raster."""
    layout = _radar_layout(hole=True)
    product = publish_merge_layout(tmp_path / "layout" / "ref-20240101", layout)
    assert product.kind == LAYOUT_KIND
    assert "owner" not in product.assets
    assert "scientific_support" in product.assets
    assert "support-0" in product.assets
    reopened = open_merge_layout(product.path)
    np.testing.assert_array_equal(reopened.owner, layout.owner)
    np.testing.assert_array_equal(
        reopened.scientific_support, layout.scientific_support
    )
    assert not bool(reopened.scientific_support[0, 1])
    assert reopened.owner[0, 1] == 0
    assert reopened.owner[2, 1] == 2
    assert reopened.seams == layout.seams


def test_geo_layout_persists_owner_raster(tmp_path: Path) -> None:
    """Geo layouts store the ROI-local owner asset."""
    layout = _geo_layout()
    product = publish_merge_layout(tmp_path / "layout" / "ref-geo", layout)
    assert "owner" in product.assets
    reopened = open_merge_layout(product)
    np.testing.assert_array_equal(reopened.owner, layout.owner)
    assert reopened.scope.domain == "geo"


def test_stale_upstream_write_id_is_reported(tmp_path: Path) -> None:
    """Layout products use existing faninsar:inputs stale detection."""
    layout = _geo_layout()
    upstream = publish_product(
        tmp_path / "grid",
        kind="grid",
        properties={},
        write_payloads=_write_dummy,
    )
    product = publish_merge_layout(
        tmp_path / "layout" / "bound",
        layout,
        inputs={"grid": upstream},
    )
    assert product.stale_inputs({"grid": upstream}) == []
    replaced = publish_product(
        tmp_path / "grid",
        kind="grid",
        properties={},
        write_payloads=_write_dummy,
        overwrite=True,
    )
    assert product.stale_inputs({"grid": replaced}) == ["grid"]


def test_open_rejects_missing_layout_kind(tmp_path: Path) -> None:
    """Opening a non-layout product is rejected."""
    other = publish_product(
        tmp_path / "other",
        kind="scene",
        properties={},
        write_payloads=_write_dummy,
    )
    with pytest.raises(InvalidProcessingStateError, match="merge_layout"):
        open_merge_layout(other.path)


def _seam_radar_layout() -> MergeLayoutWindow:
    """Two-burst radar layout with an azimuth midpoint seam (verifier F1 geometry)."""
    scope = MergeScope(
        domain="radar",
        track_id="T1",
        channel="IW",
        reference_id="ref",
        grid_identity="a" * 64,
        polygon_identity="pol",
    )
    support_a = np.ones((5, 4), dtype=bool)
    support_a[0, 0] = False
    support_b = np.ones((6, 4), dtype=bool)
    sources = (
        LayoutSourceWindow("b1", (0, 5, 0, 4), support_a, "iw1"),
        LayoutSourceWindow("b2", (2, 8, 0, 4), support_b, "iw1"),
    )
    return build_merge_layout(
        sources,
        scope=scope,
        seams=(MergeSeam("azimuth", 3, "b1", "b2"),),
        window=(0, 8, 0, 4),
    )


def test_radar_supplied_owner_holes_roundtrip_exactly(tmp_path: Path) -> None:
    """Unowned-but-supported owner holes must survive reopen, not refill."""
    layout = _seam_radar_layout()
    owner = layout.owner.copy()
    owner[1, 2] = 0
    owner[4, 3] = 0
    holed = MergeLayoutWindow(
        scope=layout.scope,
        window=layout.window,
        source_ids=layout.source_ids,
        owner=owner,
        scientific_support=layout.scientific_support,
        seams=layout.seams,
        source_windows=layout.source_windows,
    )
    product = publish_merge_layout(tmp_path / "radar-holed", holed)
    assert product.properties["faninsar:compact"] is False
    assert "owner" in product.assets
    reopened = open_merge_layout(product.path)
    np.testing.assert_array_equal(reopened.owner, holed.owner)
    assert reopened.owner[1, 2] == 0
    assert reopened.owner[4, 3] == 0
    assert bool(reopened.scientific_support[1, 2])
    assert bool(reopened.scientific_support[4, 3])


def test_owner_asset_preserves_per_source_support(tmp_path: Path) -> None:
    """Owner fallback must not replace source footprints with a bounding box."""
    layout = _seam_radar_layout()
    owner = layout.owner.copy()
    owner[1, 2] = 0
    holed = MergeLayoutWindow(
        scope=layout.scope,
        window=layout.window,
        source_ids=layout.source_ids,
        owner=owner,
        scientific_support=layout.scientific_support,
        seams=layout.seams,
        source_windows=layout.source_windows,
    )
    product = publish_merge_layout(tmp_path / "radar-support", holed)
    assert "owner" in product.assets
    assert "support-0" in product.assets
    assert "support-1" in product.assets
    reopened = open_merge_layout(product.path)
    assert len(reopened.source_windows) == 2
    original = {
        item.burst_id: item.scientific_support for item in layout.source_windows
    }
    for source in reopened.source_windows:
        np.testing.assert_array_equal(
            source.scientific_support, original[source.burst_id]
        )
    b1 = next(item for item in reopened.source_windows if item.burst_id == "b1")
    assert not bool(b1.scientific_support[0, 0])
    assert not bool(np.all(b1.scientific_support))
