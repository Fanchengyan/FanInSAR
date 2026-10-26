"""Tests for SAFE-subset full-burst-height ROI canvases (PROPOSAL-0061).

The canvas writer materializes a SAFE-compatible subset for ROI transfers:
one full-burst-height x full-samplesPerBurst canvas per kept burst, a
valid-sample XML mask matching the fetched column envelope, geolocation
footprints remapped by original burst identity, and an overlap splice that
restores the parent overlap-window valid samples on spliced lines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile

from faninsar.missions.s1.annotation import parse_annotation_xml
from faninsar.missions.s1.errors import Sentinel1ProductError
from faninsar.missions.s1.io import (
    BurstFetchedWindow,
    splice_overlap_into_canvas,
    write_roi_safe_canvas,
)
from faninsar.missions.s1.io.read import read_full_burst
from faninsar.missions.s1.overlap import BurstOverlapWindow

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import TypeAlias

    from faninsar.missions.s1.io.canvas import RoiSafeCanvas
    from faninsar.missions.s1.types import S1Swath

    ParentInfo: TypeAlias = tuple[S1Swath, MemoryFile, str, np.ndarray]
    CanvasInfo: TypeAlias = tuple[S1Swath, MemoryFile, str, np.ndarray, RoiSafeCanvas]

N_BURSTS = 6
LINES_PER_BURST = 10
SAMPLES_PER_BURST = 100


def _parent_xml(*, grid: bool = True) -> str:
    """Return annotation XML for a 6-burst synthetic IW1 parent swath.

    Modeled on ``tests/data/sentinel1/minimal_annotation_iw1.xml`` with one
    burst block per kept-candidate burst and (optionally) a geolocation grid
    whose per-burst latitudes/longitudes encode the original burst index.
    """
    burst_els: list[str] = []
    grid_els: list[str] = []
    for k in range(N_BURSTS):
        first = " ".join(["0"] * LINES_PER_BURST)
        last = " ".join([str(SAMPLES_PER_BURST - 1)] * LINES_PER_BURST)
        burst_els.append(
            f"""
      <burst>
        <azimuthTime>2016-12-07T11:18:{53 + k:02d}.426276</azimuthTime>
        <azimuthAnxTime>{607.4725 + 2.5731 * k}</azimuthAnxTime>
        <sensingTime>2016-12-07T11:18:{54 + k:02d}.543845</sensingTime>
        <byteOffset>{k * LINES_PER_BURST * SAMPLES_PER_BURST * 4}</byteOffset>
        <firstValidSample>{first}</firstValidSample>
        <lastValidSample>{last}</lastValidSample>
      </burst>"""
        )
        if grid:
            grid_els.extend(
                "        <geolocationGridPoint>"
                f"<line>{k * LINES_PER_BURST + local}</line>"
                f"<latitude>{40.0 + k + 0.01 * local:.2f}</latitude>"
                f"<longitude>{100.0 + k + 0.01 * local:.2f}</longitude>"
                "</geolocationGridPoint>"
                for local in (0, LINES_PER_BURST - 1)
            )
    grid_block = ""
    if grid:
        points = "\n".join(grid_els)
        grid_block = (
            "  <geolocationGrid>\n"
            f'    <geolocationGridPointList count="{len(grid_els)}">\n'
            f"{points}\n"
            "    </geolocationGridPointList>\n"
            "  </geolocationGrid>\n"
        )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<product>
  <adsHeader>
    <missionId>S1A</missionId>
    <productType>SLC</productType>
    <polarisation>VV</polarisation>
    <mode>IW</mode>
    <swath>IW1</swath>
    <startTime>2016-12-07T11:18:53.426275</startTime>
    <stopTime>2016-12-07T11:19:18.571896</stopTime>
  </adsHeader>
  <generalAnnotation>
    <productInformation>
      <rangeSamplingRate>64345238.12571428</rangeSamplingRate>
      <radarFrequency>5405000454.33435</radarFrequency>
      <azimuthSteeringRate>1.397440</azimuthSteeringRate>
    </productInformation>
    <orbitList count="2">
      <orbit>
        <time>2016-12-07T11:17:45.950212</time>
        <frame>Earth Fixed</frame>
        <position><x>-572093.4</x><y>5939124.4</y><z>3795662.9</z></position>
        <velocity><x>2075.1</x><y>-3800.2</y><z>6239.7</z></velocity>
      </orbit>
      <orbit>
        <time>2016-12-07T11:17:55.950213</time>
        <frame>Earth Fixed</frame>
        <position><x>-551342.0</x><y>5901124.0</y><z>3858060.0</z></position>
        <velocity><x>2075.0</x><y>-3800.0</y><z>6240.0</z></velocity>
      </orbit>
    </orbitList>
    <azimuthFmRateList count="1">
      <azimuthFmRate>
        <azimuthTime>2016-12-07T11:18:52.178090</azimuthTime>
        <t0>0.0053317936</t0>
        <azimuthFmRatePolynomial>-2330.5 451066.6</azimuthFmRatePolynomial>
      </azimuthFmRate>
    </azimuthFmRateList>
  </generalAnnotation>
  <imageAnnotation>
    <imageInformation>
      <slantRangeTime>0.0053317936</slantRangeTime>
      <rangePixelSpacing>2.329562</rangePixelSpacing>
      <azimuthPixelSpacing>13.96706</azimuthPixelSpacing>
      <azimuthTimeInterval>0.0020555563</azimuthTimeInterval>
      <numberOfSamples>{SAMPLES_PER_BURST}</numberOfSamples>
      <numberOfLines>{N_BURSTS * LINES_PER_BURST}</numberOfLines>
    </imageInformation>
  </imageAnnotation>
  <dopplerCentroid>
    <dcEstimateList count="1">
      <dcEstimate>
        <azimuthTime>2016-12-07T11:18:53.140830</azimuthTime>
        <t0>0.0053422084</t0>
        <geometryDcPolynomial>-3.992857 -287.3943 134566.6</geometryDcPolynomial>
      </dcEstimate>
    </dcEstimateList>
  </dopplerCentroid>
  <swathTiming>
    <linesPerBurst>{LINES_PER_BURST}</linesPerBurst>
    <samplesPerBurst>{SAMPLES_PER_BURST}</samplesPerBurst>
    <burstList count="{N_BURSTS}">{"".join(burst_els)}
    </burstList>
  </swathTiming>
{grid_block}</product>"""


def _parent_raster() -> np.ndarray:
    """Return the synthetic parent SLC: real part = line, imag part = column."""
    lines = np.arange(N_BURSTS * LINES_PER_BURST, dtype=np.float32)[:, None]
    cols = np.arange(SAMPLES_PER_BURST, dtype=np.float32)[None, :]
    return (lines + 1j * cols).astype(np.complex64)


def _make_parent_swath(
    *,
    grid: bool = True,
) -> tuple[S1Swath, MemoryFile, str, np.ndarray]:
    """Return (swath, memfile, xml_text, raster) of the synthetic parent."""
    xml_text = _parent_xml(grid=grid)
    data = _parent_raster()
    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="complex64",
        crs=None,
        transform=rasterio.Affine.identity(),
    ) as dst:
        dst.write(data, 1)
    swath = parse_annotation_xml(
        xml_text,
        annotation_path="memory://parent-annotation.xml",
        measurement_path=memfile.name,
    )
    return swath, memfile, xml_text, data


def _fetch(
    data: np.ndarray,
    burst_index: int,
    row0: int,
    col0: int,
    n_lines: int,
    n_cols: int,
) -> BurstFetchedWindow:
    """Slice one fetched rectangle out of the parent raster."""
    rows = burst_index * LINES_PER_BURST
    samples = np.ascontiguousarray(
        data[rows + row0 : rows + row0 + n_lines, col0 : col0 + n_cols]
    )
    return BurstFetchedWindow(
        burst_index=burst_index,
        row0=row0,
        col0=col0,
        samples=samples,
    )


def _default_windows(data: np.ndarray) -> list[BurstFetchedWindow]:
    """Return two fetched windows: burst 4 rows 3-7 and burst 5 rows 0-4."""
    return [
        _fetch(data, 4, 3, 20, 5, 40),
        _fetch(data, 5, 0, 10, 5, 40),
    ]


@pytest.fixture
def parent() -> Iterator[ParentInfo]:
    """Yield (swath, memfile, xml_text, raster) of the synthetic parent."""
    swath, memfile, xml_text, data = _make_parent_swath()
    yield swath, memfile, xml_text, data
    memfile.close()


@pytest.fixture
def canvas(parent: ParentInfo, tmp_path: Path) -> CanvasInfo:
    """Return parent info plus a canvas of kept bursts 4 and 5."""
    swath, memfile, xml_text, data = parent
    handle = write_roi_safe_canvas(
        swath,
        _default_windows(data),
        tmp_path,
        annotation_xml=xml_text,
        product_id="ROI_CANVAS",
    )
    return swath, memfile, xml_text, data, handle


class TestRoiCanvasRaster:
    """Requirement 1/3: full-burst-height canvas with placed fetched samples."""

    def test_canvas_raster_is_full_burst_height(
        self,
        parent: ParentInfo,
        tmp_path: Path,
    ) -> None:
        """Canvas is (n_kept * linesPerBurst, samplesPerBurst) for a small ROI."""
        swath, _memfile, xml_text, data = parent
        handle = write_roi_safe_canvas(
            swath,
            _default_windows(data),
            tmp_path,
            annotation_xml=xml_text,
            product_id="ROI_SHAPE",
        )
        with rasterio.open(handle.measurement_path) as ds:
            assert ds.height == 2 * LINES_PER_BURST
            assert ds.width == SAMPLES_PER_BURST
            assert ds.dtypes[0] == "complex64"
            raster = ds.read(1)
        # fetched samples sit at their original burst-local (line, column)
        np.testing.assert_array_equal(raster[3:8, 20:60], data[43:48, 20:60])
        np.testing.assert_array_equal(raster[10:15, 10:50], data[50:55, 10:50])
        # unfetched pixels stay zero
        zeros = np.zeros((3, SAMPLES_PER_BURST), dtype=np.complex64)
        np.testing.assert_array_equal(raster[:3], zeros)
        np.testing.assert_array_equal(raster[8:10], zeros[:2])
        np.testing.assert_array_equal(
            raster[3:8, :20], np.zeros((5, 20), dtype=np.complex64)
        )
        assert not np.any(raster[15:])


class TestRoiCanvasValidSampleMask:
    """Requirement 2/4: the valid-sample XML is the mask from_safes reads."""

    def test_unfetched_lines_are_minus_one_and_roi_lines_match_envelope(
        self,
        canvas: CanvasInfo,
    ) -> None:
        """Unfetched lines parse as -1; ROI lines equal the fetched envelope."""
        _swath, _memfile, _xml_text, _data, handle = canvas
        subset = parse_annotation_xml(
            handle.annotation_path.read_text(encoding="utf-8"),
            annotation_path=str(handle.annotation_path),
            measurement_path=str(handle.measurement_path),
        )
        assert len(subset.bursts) == 2
        first0 = subset.bursts[0].first_valid_sample
        last0 = subset.bursts[0].last_valid_sample
        assert first0 == (-1, -1, -1, 20, 20, 20, 20, 20, -1, -1)
        assert last0 == (-1, -1, -1, 59, 59, 59, 59, 59, -1, -1)
        first1 = subset.bursts[1].first_valid_sample
        last1 = subset.bursts[1].last_valid_sample
        assert first1 == (10, 10, 10, 10, 10, -1, -1, -1, -1, -1)
        assert last1 == (49, 49, 49, 49, 49, -1, -1, -1, -1, -1)

    def test_kept_bursts_keep_parent_metadata(
        self,
        canvas: CanvasInfo,
    ) -> None:
        """Parent linesPerBurst / azimuthTime / byteOffset survive on kept bursts."""
        swath, _memfile, _xml_text, _data, handle = canvas
        subset = parse_annotation_xml(
            handle.annotation_path.read_text(encoding="utf-8"),
            annotation_path=str(handle.annotation_path),
            measurement_path=str(handle.measurement_path),
        )
        assert subset.lines_per_burst == LINES_PER_BURST
        assert subset.samples_per_burst == SAMPLES_PER_BURST
        assert subset.lines == 2 * LINES_PER_BURST
        for slot, original in enumerate((4, 5)):
            parent_burst = swath.bursts[original]
            kept = subset.bursts[slot]
            assert kept.byte_offset == parent_burst.byte_offset
            assert kept.azimuth_time == parent_burst.azimuth_time
            assert kept.sensing_time == parent_burst.sensing_time


class TestSubsetReadsBack:
    """Requirement 8: parse_annotation_xml + read_full_burst accept the subset."""

    def test_parse_and_read_full_burst_keep_parent_dimensions(
        self,
        canvas: CanvasInfo,
    ) -> None:
        """Burst lines stay linesPerBurst and bursts land at row0 = i * lpb."""
        _swath, _memfile, _xml_text, data, handle = canvas
        subset = parse_annotation_xml(
            handle.annotation_path.read_text(encoding="utf-8"),
            annotation_path=str(handle.annotation_path),
            measurement_path=str(handle.measurement_path),
        )
        assert subset.lines == 2 * LINES_PER_BURST
        assert [burst.index for burst in subset.bursts] == [0, 1]
        assert all(burst.lines == LINES_PER_BURST for burst in subset.bursts)
        assert all(
            len(burst.first_valid_sample) == LINES_PER_BURST for burst in subset.bursts
        )

        arr0 = read_full_burst(subset, burst_index=0)
        assert arr0.row0 == 0
        assert arr0.samples.shape == (LINES_PER_BURST, 40)
        np.testing.assert_array_equal(arr0.samples[3:8], data[43:48, 20:60])
        assert not np.any(arr0.samples[0:3])
        assert not np.any(arr0.samples[8:10])
        assert not np.any(arr0.valid_mask[0:3])

        arr1 = read_full_burst(subset, burst_index=1)
        assert arr1.row0 == LINES_PER_BURST
        np.testing.assert_array_equal(arr1.samples[0:5], data[50:55, 10:50])


class TestOverlapSplice:
    """Requirement 7: overlap splice keeps height and rewrites the mask."""

    def test_splice_keeps_height_and_rewrites_valid_samples(
        self,
        canvas: CanvasInfo,
    ) -> None:
        """Spliced lines adopt the parent overlap-window valid samples."""
        _swath, _memfile, _xml_text, data, handle = canvas
        window = BurstOverlapWindow(
            swath="IW1",
            top_index=4,
            bottom_index=5,
            top_row0=6,
            bottom_row0=1,
            col0=30,
            n_lines=3,
            n_samples=25,
        )
        top_rect = (np.arange(3 * 25, dtype=np.float32).reshape(3, 25) + 1j * 7).astype(
            np.complex64
        )
        bottom_rect = (
            np.arange(3 * 25, dtype=np.float32).reshape(3, 25) - 1j * 7
        ).astype(np.complex64)

        with rasterio.open(handle.measurement_path) as ds:
            height_before = ds.height
        splice_overlap_into_canvas(handle, window, role="top", samples=top_rect)
        splice_overlap_into_canvas(
            handle, window, role="bottom", samples=bottom_rect.tobytes()
        )

        with rasterio.open(handle.measurement_path) as ds:
            assert ds.height == height_before == 2 * LINES_PER_BURST
            raster = ds.read(1)
        np.testing.assert_array_equal(raster[6:9, 30:55], top_rect)
        np.testing.assert_array_equal(raster[11:14, 30:55], bottom_rect)
        # non-spliced rows of the same burst are untouched
        np.testing.assert_array_equal(raster[3:6, 20:60], data[43:46, 20:60])

        subset = parse_annotation_xml(
            handle.annotation_path.read_text(encoding="utf-8"),
            annotation_path=str(handle.annotation_path),
            measurement_path=str(handle.measurement_path),
        )
        first0 = subset.bursts[0].first_valid_sample
        last0 = subset.bursts[0].last_valid_sample
        assert first0 == (-1, -1, -1, 20, 20, 20, 30, 30, 30, -1)
        assert last0 == (-1, -1, -1, 59, 59, 59, 54, 54, 54, -1)
        first1 = subset.bursts[1].first_valid_sample
        last1 = subset.bursts[1].last_valid_sample
        assert first1 == (10, 30, 30, 30, 10, -1, -1, -1, -1, -1)
        assert last1 == (49, 54, 54, 54, 49, -1, -1, -1, -1, -1)

    def test_splice_requires_full_overlap_rectangle(
        self,
        canvas: CanvasInfo,
    ) -> None:
        """Completeness is the parent overlap rectangle, not a tight box."""
        _swath, _memfile, _xml_text, _data, handle = canvas
        window = BurstOverlapWindow(
            swath="IW1",
            top_index=4,
            bottom_index=5,
            top_row0=6,
            bottom_row0=1,
            col0=30,
            n_lines=3,
            n_samples=25,
        )
        with pytest.raises(Sentinel1ProductError, match="rectangle"):
            splice_overlap_into_canvas(
                handle,
                window,
                role="top",
                samples=np.zeros((2, 25), dtype=np.complex64),
            )


class TestFootprintRemap:
    """Requirement 6: footprints follow original burst identity."""

    def test_kept_bursts_keep_original_footprints(
        self,
        canvas: CanvasInfo,
    ) -> None:
        """Slot 0 carries burst-4's footprint, not burst-0's."""
        swath, _memfile, _xml_text, _data, handle = canvas
        subset = parse_annotation_xml(
            handle.annotation_path.read_text(encoding="utf-8"),
            annotation_path=str(handle.annotation_path),
            measurement_path=str(handle.measurement_path),
        )
        assert subset.bursts[0].footprint == swath.bursts[4].footprint
        assert subset.bursts[1].footprint == swath.bursts[5].footprint
        assert subset.bursts[0].footprint != swath.bursts[0].footprint
        assert subset.bursts[1].footprint != swath.bursts[1].footprint
