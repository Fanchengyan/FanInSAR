"""Tests for the ``stack.masks`` facade and mask products (PROPOSAL-0067)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from affine import Affine

from faninsar.io.storage.product import open_product
from faninsar.missions.s1 import S1Stack
from faninsar.processing.masking import GridSpec, Mask
from faninsar.stack.mask_plan import MaskPlan
from faninsar.stack.masks import (
    MASK_KIND,
    CombinedMaskView,
    StackMasks,
    StackMaskView,
    discover_mask_products,
    is_water_definition,
    mask_recipe_identity,
    water_mask_names,
)

if TYPE_CHECKING:
    from pathlib import Path

    from faninsar.stack.session import Stack


def _geo_grid() -> GridSpec:
    """Return a small EPSG:4326 study-area grid."""
    return GridSpec(
        "EPSG:4326",
        Affine(0.01, 0.0, 10.0, 0.0, -0.01, 45.0),
        shape=(10, 10),
    )


def _raster_mask_path(tmp_path: Path) -> Path:
    """Write a deterministic uint8 raster mask source for recipes."""
    import rasterio

    path = tmp_path / "mask.tif"
    data = np.zeros((10, 10), dtype=np.uint8)
    data[:5] = 1
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=Affine(0.01, 0.0, 10.0, 0.0, -0.01, 45.0),
        nodata=255,
    ) as dst:
        dst.write(data, 1)
    return path


def _stack(tmp_path: Path, *, mask_plan: MaskPlan | None = None) -> Stack:
    """Build a minimal two-date Stack with an explicit geo grid."""
    paths = []
    for day in ("20160101", "20160113"):
        path = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000_{day}T000001.SAFE"
        path.mkdir()
        paths.append(path)
    stack = S1Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        multilook=(1, 1),
        geo_grid=_geo_grid(),
        mask_plan=mask_plan,
    )
    return stack.prepare_scenes()


def test_masks_facade_mapping_and_attribute_access(tmp_path: Path) -> None:
    """``stack.masks`` exposes Mapping semantics and attribute access."""
    mask_path = _raster_mask_path(tmp_path)
    plan = MaskPlan(
        {"observed": {"kind": "raster", "path": str(mask_path)}},
        {"interferogram": ["observed"]},
    )
    stack = _stack(tmp_path, mask_plan=plan)

    facade = stack.masks
    assert isinstance(facade, StackMasks)
    assert list(facade) == ["observed"]
    assert len(facade) == 1
    assert isinstance(facade["observed"], StackMaskView)
    assert isinstance(facade.observed, StackMaskView)
    with pytest.raises(KeyError):
        facade["missing"]


def test_combined_is_reserved_in_mask_plan(tmp_path: Path) -> None:
    """``combined`` is rejected as a registry name on every entry path."""
    mask_path = _raster_mask_path(tmp_path)
    with pytest.raises(ValueError, match="reserved"):
        MaskPlan({"combined": {"kind": "raster", "path": str(mask_path)}})
    plan = MaskPlan(
        {"observed": {"kind": "raster", "path": str(mask_path)}},
        {"interferogram": ["observed"]},
    )
    with pytest.raises(ValueError, match="reserved"):
        plan.with_mask("combined", {"kind": "raster", "path": str(mask_path)})


def test_geo_domain_read_commits_product(tmp_path: Path) -> None:
    """First ``.geo`` read materializes and commits ``masks/<name>/geo/``."""
    mask_path = _raster_mask_path(tmp_path)
    plan = MaskPlan(
        {"observed": {"kind": "raster", "path": str(mask_path)}},
        {"interferogram": ["observed"]},
    )
    stack = _stack(tmp_path, mask_plan=plan)

    plane = stack.masks.observed.geo

    assert plane.dtype == np.uint8
    assert plane.shape == (10, 10)
    product_dir = stack.config.work_dir / "masks" / "observed" / "geo"
    assert (product_dir / "item.json").is_file()
    assert (product_dir / "mask.zarr").is_dir()
    product = open_product(product_dir, kind=MASK_KIND)
    assert product.properties["faninsar:mask_name"] == "observed"
    assert product.properties["faninsar:mask_domain"] == "geo"
    assert (
        product.properties["faninsar:mask_recipe_identity"]
        == mask_recipe_identity(plan.masks["observed"])
    )
    # Second read reuses the persisted product (same write id).
    plane_again = stack.masks["observed"].sel("geo")
    np.testing.assert_array_equal(plane, plane_again)


def test_roi_alias_reads_geo_domain(tmp_path: Path) -> None:
    """``.roi`` folds into the geo-domain raster."""
    mask_path = _raster_mask_path(tmp_path)
    plan = MaskPlan(
        {"observed": {"kind": "raster", "path": str(mask_path)}},
        {"roi": ["observed"]},
    )
    stack = _stack(tmp_path, mask_plan=plan)

    roi_plane = stack.masks.observed.roi
    geo_plane = stack.masks.observed.geo
    np.testing.assert_array_equal(roi_plane, geo_plane)
    assert (stack.config.work_dir / "masks" / "observed" / "geo").is_dir()


def test_wrong_domain_access_fails_closed(tmp_path: Path) -> None:
    """An unknown domain raises instead of returning the other raster."""
    mask_path = _raster_mask_path(tmp_path)
    plan = MaskPlan(
        {"observed": {"kind": "raster", "path": str(mask_path)}},
        {"interferogram": ["observed"]},
    )
    stack = _stack(tmp_path, mask_plan=plan)

    with pytest.raises(ValueError, match="unknown mask domain"):
        stack.masks.observed.sel("sideways")


def test_combined_virtual_returns_stage_union(tmp_path: Path) -> None:
    """``combined`` unions the masks the selected stage referenced."""
    mask_path = _raster_mask_path(tmp_path)
    second = np.zeros((10, 10), dtype=np.uint8)
    second[5:] = 1
    concrete = Mask.from_raster(second, grid=_geo_grid())
    plan = MaskPlan(
        {
            "observed": {"kind": "raster", "path": str(mask_path)},
            "extra": concrete,
        },
        {"ionosphere": ["observed", "extra"]},
    )
    stack = _stack(tmp_path, mask_plan=plan)

    combined = stack.masks["combined"]
    assert isinstance(combined, CombinedMaskView)
    union = combined.sel("geo", stage="ionosphere")
    assert union.shape == (10, 10)
    # Union of top-half and bottom-half excluded planes covers all rows.
    assert set(np.unique(union)) == {1}
    # ``combined`` is virtual — no product directory for the reserved name.
    assert not (stack.config.work_dir / "masks" / "combined").exists()


def test_unreferenced_mask_materializes_on_request(tmp_path: Path) -> None:
    """A registered-but-unreferenced mask still resolves through the facade."""
    mask_path = _raster_mask_path(tmp_path)
    plan = MaskPlan(
        {"observed": {"kind": "raster", "path": str(mask_path)}},
        {},
    )
    stack = _stack(tmp_path, mask_plan=plan)

    plane = stack.masks.observed.geo
    assert plane.shape == (10, 10)
    assert (
        stack.config.work_dir / "masks" / "observed" / "geo" / "item.json"
    ).is_file()


def test_status_enumerates_mask_products(tmp_path: Path) -> None:
    """``Stack.status()`` lists committed mask products and flags staleness."""
    mask_path = _raster_mask_path(tmp_path)
    plan = MaskPlan(
        {"observed": {"kind": "raster", "path": str(mask_path)}},
        {"interferogram": ["observed"]},
    )
    stack = _stack(tmp_path, mask_plan=plan)
    _ = stack.masks.observed.geo

    report = stack.status()
    masks = report["masks"]
    assert "observed" in masks["planned"]
    assert "observed/geo" in masks["committed"]
    assert masks["missing"] == []
    assert masks["stale"] == []


def test_discover_mask_products_finds_committed(tmp_path: Path) -> None:
    """``discover_mask_products`` enumerates the masks/<name>/<domain> root."""
    mask_path = _raster_mask_path(tmp_path)
    plan = MaskPlan(
        {"observed": {"kind": "raster", "path": str(mask_path)}},
        {"interferogram": ["observed"]},
    )
    stack = _stack(tmp_path, mask_plan=plan)
    _ = stack.masks.observed.geo

    products = discover_mask_products(stack)
    assert len(products) == 1
    assert products[0].kind == MASK_KIND
    assert products[0].properties["faninsar:mask_name"] == "observed"


def test_verify_covers_mask_products(tmp_path: Path) -> None:
    """``Stack.verify()`` hashes mask product payloads."""
    mask_path = _raster_mask_path(tmp_path)
    plan = MaskPlan(
        {"observed": {"kind": "raster", "path": str(mask_path)}},
        {"interferogram": ["observed"]},
    )
    stack = _stack(tmp_path, mask_plan=plan)
    _ = stack.masks.observed.geo

    assert stack.verify() == []
    # Corrupt the committed payload and confirm verify() flags it.
    chunks = [
        entry
        for entry in (
            stack.config.work_dir / "masks" / "observed" / "geo" / "mask.zarr"
        ).rglob("*")
        if entry.is_file()
    ]
    assert chunks, "mask.zarr payload must contain files"
    chunks[0].write_bytes(b"corrupt")
    problems = stack.verify()
    assert any("masks" in entry for entry in problems)


def test_water_recipe_detection_and_geometry_graceful_fallback(
    tmp_path: Path,
) -> None:
    """Water recipes are detected; Network._geometry degrades without I/O."""
    plan = MaskPlan(
        {"water": Mask.from_water(provider="auto")},
        {"interferogram": ["water"]},
    )
    stack = _stack(tmp_path, mask_plan=plan)

    assert is_water_definition(plan.masks["water"])
    assert water_mask_names(stack) == ("water",)
    # Without provider access the materialization fails closed and the
    # Network geometry stays unset rather than fabricating a water mask.
    assert stack._network_geometry_view() is None


def test_stage_consumption_persists_mask_products(tmp_path: Path) -> None:
    """``_materialize_stage_mask`` commits each referenced mask once."""
    mask_path = _raster_mask_path(tmp_path)
    plan = MaskPlan(
        {"observed": {"kind": "raster", "path": str(mask_path)}},
        {"interferogram": ["observed"], "unwrap": ["observed"]},
    )
    stack = _stack(tmp_path, mask_plan=plan)
    stack.config.coregistration_grid = "geo"

    materialized = stack._materialize_stage_mask("interferogram", _geo_grid())
    assert materialized is not None
    product_dir = stack.config.work_dir / "masks" / "observed" / "geo"
    assert (product_dir / "item.json").is_file()
    product = open_product(product_dir, kind=MASK_KIND)
    first_write_id = product.write_id
    # A second stage sharing (refs, grid) must not rewrite the raster.
    stack._materialize_stage_mask("unwrap", _geo_grid())
    product = open_product(product_dir, kind=MASK_KIND)
    assert product.write_id == first_write_id
