"""Public seam tests for the PROPOSAL-0040 mask core."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from faninsar.processing.masking import (
    GridSpec,
    Mask,
    RasterMask,
    UnionMask,
    VectorMask,
)


def _grid(validity: np.ndarray | None = None) -> GridSpec:
    return GridSpec(
        crs="EPSG:4326",
        transform=Affine(1, 0, 0, 0, -1, 2),
        shape=(2, 2),
        validity=validity,
    )


def test_raster_core_is_defensive_and_tristate() -> None:
    """Factory, NumPy semantics, and inversion preserve the three values."""
    source = np.array([[0, 1], [255, 0]], dtype=np.uint8)
    mask = Mask.from_raster(source, grid=_grid())
    source[0, 0] = 1
    assert isinstance(mask, RasterMask)
    np.testing.assert_array_equal(mask.data, [[0, 1], [255, 0]])
    np.testing.assert_array_equal(mask.application_mask, [[False, True], [True, False]])
    np.testing.assert_array_equal(mask.invert().data, [[1, 0], [255, 1]])
    assert not mask.data.flags.writeable


def test_union_strong_kleene_and_identity() -> None:
    """Union applies definite exclusion precedence and canonical identities."""
    left = Mask.from_raster(np.array([[0, 1], [255, 0]], dtype=np.uint8), grid=_grid())
    right = Mask.from_raster(
        np.array([[255, 255], [0, 1]], dtype=np.uint8), grid=_grid()
    )
    expected = np.array([[255, 1], [255, 1]], dtype=np.uint8)
    np.testing.assert_array_equal((left + right).to_raster(_grid()).data, expected)
    assert (left + right).identity == (right + left).identity
    assert len(UnionMask(left, left).operands) == 1
    np.testing.assert_array_equal(
        UnionMask().to_raster(_grid()).data, np.zeros((2, 2), np.uint8)
    )


def test_vector_roles_and_conversion() -> None:
    """Vector excluded and invalid roles materialize distinctly."""
    vector = Mask.from_vector(
        [box(0, 0, 0.9, 0.9), box(1.1, 1.1, 2, 2)],
        crs="EPSG:4326",
        roles=["excluded", "invalid"],
        categories=["water", "coverage"],
    )
    assert isinstance(vector, VectorMask)
    result = vector.to_raster(_grid()).data
    np.testing.assert_array_equal(result, [[0, 255], [1, 0]])


def test_removed_legacy_surface() -> None:
    """The hard cutover does not provide the old public names."""
    import faninsar.processing.masking.mask as module

    assert not hasattr(module, "MaskOperator")
    assert not hasattr(module, "MaskSampler")
    with pytest.raises(ValueError, match="only 0, 1, and 255"):
        Mask.from_raster(np.array([[2, 0], [0, 0]], dtype=np.uint8), grid=_grid())
