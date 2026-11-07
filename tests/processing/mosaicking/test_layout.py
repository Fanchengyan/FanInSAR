"""Layout assembly from source windows and precomputed seams (PROPOSAL-0074)."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.mosaicking import (
    LayoutSourceWindow,
    MergeScope,
    MergeSeam,
    build_merge_layout,
)

SCOPE = MergeScope(
    domain="radar",
    track_id="T100_A",
    channel="vv",
    reference_id="ref-20240101",
    grid_identity="a" * 64,
    polygon_identity="poly-0",
)


def _source(
    burst_id: str,
    window: tuple[int, int, int, int],
    support: np.ndarray,
    swath: str = "IW1",
) -> LayoutSourceWindow:
    return LayoutSourceWindow(
        burst_id=burst_id,
        window=window,
        scientific_support=np.asarray(support, dtype=bool),
        swath=swath,
    )


def test_unique_coverage_assigns_owner_without_seams() -> None:
    """Non-overlapping support is labeled without inventing a midpoint."""
    left = _source("iw1-b1", (0, 2, 0, 3), np.ones((2, 3), dtype=bool))
    right = _source("iw1-b2", (2, 4, 0, 3), np.ones((2, 3), dtype=bool))
    layout = build_merge_layout([right, left], scope=SCOPE)
    assert layout.source_ids == ("iw1-b1", "iw1-b2")
    assert layout.window == (0, 4, 0, 3)
    np.testing.assert_array_equal(layout.owner[:2], 1)
    np.testing.assert_array_equal(layout.owner[2:], 2)
    np.testing.assert_array_equal(layout.scientific_support, True)


def test_overlap_without_seams_stays_unowned() -> None:
    """ROI-cropped boxes are not treated as physical Geo midpoints."""
    first = _source("iw1-b1", (0, 3, 0, 2), np.ones((3, 2), dtype=bool))
    second = _source("iw1-b2", (1, 4, 0, 2), np.ones((3, 2), dtype=bool))
    layout = build_merge_layout([first, second], scope=SCOPE)
    np.testing.assert_array_equal(layout.owner[0], 1)
    np.testing.assert_array_equal(layout.owner[3], 2)
    np.testing.assert_array_equal(layout.owner[1:3], 0)


def test_precomputed_azimuth_seam_cuts_overlap() -> None:
    """Caller-supplied seams partition overlapping scientific support."""
    first = _source("iw1-b1", (0, 4, 0, 3), np.ones((4, 3), dtype=bool))
    second = _source("iw1-b2", (2, 6, 0, 3), np.ones((4, 3), dtype=bool))
    layout = build_merge_layout(
        [first, second],
        scope=SCOPE,
        seams=(MergeSeam("azimuth", 3, "iw1-b1", "iw1-b2"),),
    )
    np.testing.assert_array_equal(layout.owner[:3], 1)
    np.testing.assert_array_equal(layout.owner[3:], 2)


def test_supplied_owner_is_not_rewritten_by_seams() -> None:
    """A finished owner raster wins; seams remain metadata only."""
    first = _source("iw1-b1", (0, 2, 0, 2), np.ones((2, 2), dtype=bool))
    second = _source("iw1-b2", (0, 2, 0, 2), np.ones((2, 2), dtype=bool))
    owner = np.full((2, 2), 2, dtype=np.uint16)
    layout = build_merge_layout(
        [first, second],
        scope=SCOPE,
        seams=(MergeSeam("azimuth", 1, "iw1-b1", "iw1-b2"),),
        owner=owner,
    )
    np.testing.assert_array_equal(layout.owner, 2)


def test_source_holes_remain_unowned() -> None:
    """Per-source scientific support holes are not filled by a bounding box."""
    support = np.ones((3, 3), dtype=bool)
    support[1, 1] = False
    source = _source("iw1-b1", (0, 3, 0, 3), support)
    layout = build_merge_layout([source], scope=SCOPE)
    assert not bool(layout.scientific_support[1, 1])
    assert layout.owner[1, 1] == 0
    assert layout.owner[0, 0] == 1


def test_empty_sources_are_rejected() -> None:
    """A layout requires at least one source window."""
    with pytest.raises(InvalidProcessingStateError, match="at least one source"):
        build_merge_layout([], scope=SCOPE)
