"""Focused PROPOSAL-0040 tests for the canonical mask algebra."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from faninsar.processing.masking.mask import GridSpec, Mask, RasterMask, VectorMask

if TYPE_CHECKING:
    from pathlib import Path

_DEFAULT_TRANSFORM = Affine(1, 0, 0, 0, -1, 2)


def _grid(
    *,
    crs: str = "EPSG:4326",
    transform: Affine = _DEFAULT_TRANSFORM,
    shape: tuple[int, int] = (2, 2),
    validity: np.ndarray | None = None,
) -> GridSpec:
    """Build a small deterministic target grid."""
    return GridSpec(crs=crs, transform=transform, shape=shape, validity=validity)


def test_raster_mask_is_defensive_and_tristate() -> None:
    """Raster masks preserve the canonical 0/1/255 labels."""
    source = np.array([[0, 1], [255, 0]], dtype=np.uint8)
    mask = Mask.from_raster(source, grid=_grid())
    source[0, 0] = 1

    assert isinstance(mask, RasterMask)
    np.testing.assert_array_equal(mask.data, [[0, 1], [255, 0]])
    np.testing.assert_array_equal(mask.application_mask, [[False, True], [True, False]])
    np.testing.assert_array_equal(mask.invert().data, [[1, 0], [255, 1]])
    assert not mask.data.flags.writeable


@pytest.mark.parametrize(
    "values", [np.array([[2]], dtype=np.uint8), np.array([[3]], dtype=np.uint8)]
)
def test_raster_mask_rejects_noncanonical_labels(values: np.ndarray) -> None:
    """Arbitrary integer labels cannot silently become scientific classes."""
    with pytest.raises(ValueError, match="only 0, 1, and 255"):
        Mask.from_raster(values, grid=_grid(shape=(1, 1)))


def test_boolean_raster_is_normalized_to_uint8() -> None:
    """Boolean inputs are accepted as the 0/1 subset of the contract."""
    mask = Mask.from_raster(np.array([[True, False]]), grid=_grid(shape=(1, 2)))
    assert mask.data.dtype == np.uint8
    np.testing.assert_array_equal(mask.data, [[1, 0]])


def test_grid_validity_overrides_labels() -> None:
    """Unavailable target cells are always encoded as 255."""
    grid = _grid(validity=np.array([[True, False], [True, True]]))
    mask = Mask.from_raster(np.ones((2, 2), dtype=np.uint8), grid=grid)
    np.testing.assert_array_equal(mask.data, [[1, 255], [1, 1]])


def test_vector_roles_and_overlap_contract() -> None:
    """Excluded and invalid roles are distinct and may not overlap."""
    vector = Mask.from_vector(
        [box(0, 0, 0.9, 0.9), box(1.1, 1.1, 2, 2)],
        crs="EPSG:4326",
        roles=["excluded", "invalid"],
        categories=["water", "coverage"],
    )
    assert isinstance(vector, VectorMask)
    np.testing.assert_array_equal(vector.to_raster(_grid()).data, [[0, 255], [1, 0]])

    overlap = Mask.from_vector(
        [box(0, 0, 1.1, 1.1), box(0.5, 0.5, 1.5, 1.5)],
        crs="EPSG:4326",
        roles=["excluded", "invalid"],
    )
    with pytest.raises(ValueError, match="overlap"):
        overlap.to_raster(_grid())


def test_nested_identity_is_a_snapshot() -> None:
    """Nested recipe metadata is copied before identity hashing."""
    provenance: dict[str, object] = {"nested": {"values": ["a", "b"]}}
    mask = Mask.from_vector(box(0, 0, 1, 1), crs="EPSG:4326", provenance=provenance)
    identity = mask.identity
    nested = provenance["nested"]
    assert isinstance(nested, dict)
    nested["values"].append("mutated")  # type: ignore[union-attr]
    provenance["new"] = "field"
    assert mask.identity == identity


def test_raster_to_vector_requires_explicit_full_bounds() -> None:
    """Lossless raster conversion accepts only the complete grid bounds."""
    mask = Mask.from_raster(np.array([[0, 1], [255, 0]], dtype=np.uint8), grid=_grid())
    assert isinstance(mask.to_vector(bounds=_grid().bounds), VectorMask)
    with pytest.raises(ValueError, match="exact full grid bounds"):
        mask.to_vector(bounds=(0, 0, 1, 1))


def test_union_is_commutative_idempotent_and_strong_kleene() -> None:
    """Union gives excluded precedence, invalid propagation, and stable identity."""
    left = Mask.from_raster(np.array([[0, 1], [255, 0]], dtype=np.uint8), grid=_grid())
    right = Mask.from_raster(
        np.array([[255, 255], [0, 1]], dtype=np.uint8), grid=_grid()
    )
    expected = np.array([[255, 1], [255, 1]], dtype=np.uint8)
    np.testing.assert_array_equal((left + right).to_raster(_grid()).data, expected)
    assert (left + right).identity == (right + left).identity
    assert len((left + left).operands) == 1  # type: ignore[attr-defined]


def test_water_recipe_is_deferred_and_requires_bounds(tmp_path: Path) -> None:
    """Constructing a water recipe is inert; realization validates its bounds."""
    del tmp_path
    recipe = Mask.from_water(provider="gsw")
    assert recipe.identity
    with pytest.raises(ValueError, match="bounds"):
        recipe.to_raster(_grid())


def test_projected_grid_rasterization() -> None:
    """Vector masks transform WGS84 geometry onto an explicit UTM grid."""
    import pyproj

    forward = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    west, south = forward.transform(3.0, 0.0)
    east, north = forward.transform(3.01, 0.01)
    grid = _grid(
        crs="EPSG:32631",
        transform=Affine((east - west) / 2, 0, west, 0, -(north - south) / 2, north),
    )
    mask = Mask.from_vector(box(3.0, 0.0, 3.01, 0.01), crs="EPSG:4326")
    np.testing.assert_array_equal(mask.to_raster(grid).data, np.ones((2, 2), np.uint8))
