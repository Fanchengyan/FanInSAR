"""Incremental burst-ready SAFE canvas writes (P0061/P0070)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.windows import Window

from faninsar.missions.s1.annotation import parse_annotation_xml
from faninsar.missions.s1.errors import Sentinel1ProductError
from faninsar.missions.s1.io.canvas import BurstFetchedWindow, _RoiCanvasWriter
from faninsar.missions.s1.overlap import BurstOverlapWindow
from tests.missions.s1.test_span_cache import (
    LINES_PER_BURST,
    SAMPLES_PER_BURST,
    _parent_raster,
    _parent_xml,
)


def test_burst_ready_canvas_resumes_without_losing_samples_or_overlap_mask(
    tmp_path: Path,
) -> None:
    """First burst is durable while later slots stay blank; resume appends safely."""
    xml_text = _parent_xml()
    swath = parse_annotation_xml(
        xml_text,
        annotation_path="s1a-iw1-slc-vv-parent.xml",
        measurement_path="unused.tiff",
    )
    kwargs = dict(
        annotation_xml=xml_text, product_id="INCREMENTAL", listed_indices=(4, 5)
    )
    writer = _RoiCanvasWriter(swath, tmp_path, **kwargs)
    canvas = writer.canvas
    data = _parent_raster()
    roi = BurstFetchedWindow(4, 3, 20, data[43:48, 20:60])
    overlap = BurstOverlapWindow(
        swath="IW1",
        top_index=4,
        bottom_index=5,
        top_row0=6,
        bottom_row0=1,
        col0=30,
        n_lines=3,
        n_samples=25,
    )
    writer.write_burst(4, [roi], [(overlap, "top", data[46:49, 30:55])])
    with rasterio.open(canvas.measurement_path) as src:
        assert src.shape == (2 * LINES_PER_BURST, SAMPLES_PER_BURST)
        assert src.profile["tiled"]
        assert str(src.compression).lower().endswith("deflate")
        first = src.read(1)
    np.testing.assert_array_equal(first[3:8, 20:60], data[43:48, 20:60])
    np.testing.assert_array_equal(first[6:9, 30:55], data[46:49, 30:55])
    assert not np.any(first[10:])
    parsed = parse_annotation_xml(
        canvas.annotation_path.read_text(),
        annotation_path=str(canvas.annotation_path),
        measurement_path=str(canvas.measurement_path),
    )
    assert parsed.lines == 20
    assert parsed.samples_per_burst == 100
    assert parsed.bursts[0].first_valid_sample == (
        -1,
        -1,
        -1,
        20,
        20,
        20,
        20,
        20,
        30,
        -1,
    )
    assert parsed.bursts[0].last_valid_sample == (
        -1,
        -1,
        -1,
        59,
        59,
        59,
        59,
        59,
        54,
        -1,
    )
    assert parsed.bursts[1].first_valid_sample == (-1,) * LINES_PER_BURST

    resumed = _RoiCanvasWriter(swath, tmp_path, resume=True, **kwargs)
    resumed.write_burst(5, [], [(overlap, "bottom", data[51:54, 30:55])])
    with rasterio.open(canvas.measurement_path) as src:
        actual = src.read(1)
    np.testing.assert_array_equal(actual[:10], first[:10])
    np.testing.assert_array_equal(actual[11:14, 30:55], data[51:54, 30:55])
    parsed = parse_annotation_xml(
        canvas.annotation_path.read_text(),
        annotation_path=str(canvas.annotation_path),
        measurement_path=str(canvas.measurement_path),
    )
    assert parsed.bursts[1].first_valid_sample == (
        -1,
        30,
        30,
        30,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
    )
    assert parsed.bursts[1].last_valid_sample == (
        -1,
        54,
        54,
        54,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
    )
    assert resumed.finish() == canvas


def _make_writer(tmp_path: Path, **overrides: object) -> _RoiCanvasWriter:
    """Create or reopen the standard synthetic two-burst canvas."""
    xml_text = _parent_xml()
    swath = parse_annotation_xml(
        xml_text,
        annotation_path="s1a-iw1-slc-vv-parent.xml",
        measurement_path="unused.tiff",
    )
    kwargs = dict(
        annotation_xml=xml_text, product_id="INCREMENTAL", listed_indices=(4, 5)
    )
    kwargs.update(overrides)
    return _RoiCanvasWriter(swath, tmp_path, **kwargs)


def test_rewrite_is_idempotent_and_resets_only_target_burst(tmp_path: Path) -> None:
    """Retries erase stale target samples/masks without changing another burst."""
    writer = _make_writer(tmp_path)
    data = _parent_raster()
    untouched = BurstFetchedWindow(5, 0, 10, data[50:55, 10:50])
    writer.write_burst(5, [untouched], [])
    writer.write_burst(4, [BurstFetchedWindow(4, 0, 0, data[40:50])], [])
    roi = BurstFetchedWindow(4, 3, 20, data[43:48, 20:60])
    writer.write_burst(4, [roi], [])
    writer.write_burst(4, [roi], [])
    with rasterio.open(writer.canvas.measurement_path) as src:
        raster = src.read(1)
    np.testing.assert_array_equal(raster[10:15, 10:50], data[50:55, 10:50])
    assert not np.any(raster[:3])
    assert not np.any(raster[8:10])
    assert not np.any(raster[3:8, :20])
    assert not np.any(raster[3:8, 60:])
    np.testing.assert_array_equal(raster[3:8, 20:60], data[43:48, 20:60])


def test_roi_masks_union_and_exclude_unsupported_line_valid_rows(
    tmp_path: Path,
) -> None:
    """Multiple ROI windows union envelopes but cannot mark unfetched rows valid."""
    writer = _make_writer(tmp_path)
    validity = ((0, 99), (0, 99), (-1, -1), (0, 99)) + ((-1, -1),) * 6
    left = BurstFetchedWindow(4, 1, 20, np.ones((3, 10)), line_valid=validity)
    right = BurstFetchedWindow(4, 1, 40, np.ones((1, 10)))
    writer.write_burst(4, [left, right], [])
    subset = parse_annotation_xml(
        writer.canvas.annotation_path.read_text(),
        annotation_path=str(writer.canvas.annotation_path),
        measurement_path=str(writer.canvas.measurement_path),
    )
    assert subset.bursts[0].first_valid_sample == (
        -1,
        20,
        -1,
        20,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
    )
    assert subset.bursts[0].last_valid_sample == (
        -1,
        49,
        -1,
        29,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
    )


def test_same_size_sample_mutation_does_not_block_structural_resume(
    tmp_path: Path,
) -> None:
    """Same-size raster edits after a successful write are intentionally undetected."""
    writer = _make_writer(tmp_path)
    writer.write_burst(4, [BurstFetchedWindow(4, 1, 20, np.ones((2, 3)))], [])
    with rasterio.open(writer.canvas.measurement_path, "r+") as dst:
        dst.write(
            np.asarray([[2 + 3j]], dtype=np.complex64), 1, window=Window(20, 1, 1, 1)
        )
    resumed = _make_writer(tmp_path, resume=True)
    with rasterio.open(resumed.canvas.measurement_path) as src:
        assert src.read(1)[1, 20] == np.complex64(2 + 3j)


@pytest.mark.parametrize("damage", ["shape", "annotation", "missing", "mask"])
def test_resume_rejects_invalid_canvas(tmp_path: Path, damage: str) -> None:
    """An unusable canvas is rejected, never silently recreated at resume."""
    writer = _make_writer(tmp_path)
    canvas = writer.canvas
    if damage == "shape":
        with rasterio.open(
            canvas.measurement_path,
            "w",
            driver="GTiff",
            width=100,
            height=10,
            count=1,
            dtype="complex64",
        ):
            pass
    elif damage == "annotation":
        canvas.annotation_path.write_text(
            canvas.annotation_path.read_text().replace(
                "2016-12-07T11:18:57.426276", "2016-12-07T11:18:56.426276"
            )
        )
    elif damage == "mask":
        canvas.annotation_path.write_text(
            canvas.annotation_path.read_text().replace(
                "<firstValidSample>-1", "<firstValidSample>1000", 1
            )
        )
    else:
        canvas.measurement_path.unlink()
    with pytest.raises(Sentinel1ProductError):
        _make_writer(tmp_path, resume=True)


@pytest.mark.parametrize("damage", ["other_burst", "overlap_role", "overlap_shape"])
def test_reject_bad_burst_payload_invalidates_target(
    tmp_path: Path, damage: str
) -> None:
    """A rejected payload invalidates the target without a success receipt.

    Incremental writing clears the target before streaming blocks, so a
    bad payload cannot leave the previous samples advertised; the next
    identical write rebuilds the same burst deterministically.
    """
    writer = _make_writer(tmp_path)
    writer.write_burst(4, [BurstFetchedWindow(4, 1, 20, np.ones((2, 3)))], [])
    overlap = BurstOverlapWindow(
        swath="IW1",
        top_index=4,
        bottom_index=5,
        top_row0=6,
        bottom_row0=1,
        col0=30,
        n_lines=3,
        n_samples=25,
    )
    windows = []
    overlaps = []
    if damage == "other_burst":
        windows = [BurstFetchedWindow(5, 1, 20, np.ones((2, 3)))]
    elif damage == "overlap_role":
        overlaps = [(overlap, "bottom", np.ones((3, 25)))]
    else:
        overlaps = [(overlap, "top", np.ones((2, 25)))]
    with pytest.raises(Sentinel1ProductError):
        writer.write_burst(4, windows, overlaps)
    with rasterio.open(writer.canvas.measurement_path) as src:
        assert not np.any(src.read(1)[:10])
    writer.write_burst(4, [BurstFetchedWindow(4, 1, 20, np.ones((2, 3)))], [])
    with rasterio.open(writer.canvas.measurement_path) as src:
        np.testing.assert_array_equal(src.read(1)[1:3, 20:23], np.ones((2, 3)))
