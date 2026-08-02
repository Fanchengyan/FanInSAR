"""Tests for byte-offset burst extraction and SAFE-subset export.

Fast unit tests exercise the fallback path (synthetic in-memory TIFF, where
``extract_burst`` delegates to ``read_full_burst``) and the export round-trip.
Slow integration tests, gated on the local SAFE ZIP corpus, prove the
byte-offset ZIP fast path is bitwise identical to the rasterio-window path.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile

from faninsar.missions.sentinel1 import (
    estimate_burst_bytes,
    export_swath_bursts,
    extract_burst,
    extract_bursts,
    open_safe_product,
    read_full_burst,
)
from faninsar.missions.sentinel1.annotation import parse_annotation_xml
from faninsar.missions.sentinel1.errors import Sentinel1ProductError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from faninsar.missions.sentinel1.types import S1Swath

FIXTURE_XML = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "sentinel1"
    / "minimal_annotation_iw1.xml"
)

REAL_ZIP = Path(
    "/Volumes/DATA2/TEST_sentinel-1/Raw Data/sentinel-slc/"
    "S1A_IW_SLC__1SSV_20161207T111852_20161207T111919_014273_01716D_67BC.zip"
)


def _make_synthetic_swath() -> tuple[S1Swath, MemoryFile]:
    """Return (swath, memfile) for a 2-burst 20x100 synthetic SLC."""
    xml_text = FIXTURE_XML.read_text(encoding="utf-8")
    xml_text = xml_text.replace(
        "<firstValidSample>1 1 1 1 1 1 1 1 1 1</firstValidSample>",
        "<firstValidSample>0 0 0 0 0 0 0 0 0 0</firstValidSample>",
    )
    xml_text = xml_text.replace(
        "<lastValidSample>98 98 98 98 98 98 98 98 98 98</lastValidSample>",
        "<lastValidSample>99 99 99 99 99 99 99 99 99 99</lastValidSample>",
    )
    swath = parse_annotation_xml(
        xml_text,
        annotation_path=str(FIXTURE_XML),
        measurement_path="memory://synthetic.tiff",
    )
    height = swath.lines
    width = swath.samples
    data = np.zeros((height, width), dtype=np.complex64)
    data[:10, :] = np.exp(1j * np.linspace(0, np.pi, width))[None, :]
    data[10:, :] = np.exp(1j * np.linspace(np.pi, 2 * np.pi, width))[None, :]
    data += np.arange(height)[:, None].astype(np.complex64)

    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="complex64",
        crs=None,
        transform=rasterio.Affine.identity(),
    ) as dst:
        dst.write(data, 1)

    swath = type(swath)(
        swath=swath.swath,
        polarization=swath.polarization,
        annotation_path=swath.annotation_path,
        measurement_path=memfile.name,
        lines=swath.lines,
        samples=swath.samples,
        lines_per_burst=swath.lines_per_burst,
        samples_per_burst=swath.samples_per_burst,
        range_pixel_spacing_m=swath.range_pixel_spacing_m,
        azimuth_pixel_spacing_m=swath.azimuth_pixel_spacing_m,
        azimuth_time_interval_s=swath.azimuth_time_interval_s,
        slant_range_time_s=swath.slant_range_time_s,
        range_sampling_rate_hz=swath.range_sampling_rate_hz,
        radar_frequency_hz=swath.radar_frequency_hz,
        sensing_start=swath.sensing_start,
        sensing_stop=swath.sensing_stop,
        bursts=swath.bursts,
        orbit=swath.orbit,
        doppler_centroid=swath.doppler_centroid,
        azimuth_fm_rate=swath.azimuth_fm_rate,
        azimuth_steering_rate_rad_s=swath.azimuth_steering_rate_rad_s,
    )
    return swath, memfile


class TestExtractSynthetic:
    """Fast unit tests against a synthetic in-memory product (fallback path)."""

    @pytest.fixture(scope="class")
    def swath(self) -> Iterator[S1Swath]:
        """Yield a synthetic S1Swath backed by an in-memory TIFF."""
        sw, memfile = _make_synthetic_swath()
        yield sw
        memfile.close()

    def test_extract_matches_read_full_burst(self, swath: S1Swath) -> None:
        """On a non-ZIP source extract delegates to read and returns equal data."""
        for idx in range(len(swath.bursts)):
            extracted = extract_burst(swath, burst_index=idx)
            reference = read_full_burst(swath, burst_index=idx)
            assert extracted.burst_index == idx
            assert extracted.row0 == reference.row0
            assert extracted.col0 == reference.col0
            np.testing.assert_array_equal(extracted.samples, reference.samples)
            np.testing.assert_array_equal(extracted.valid_mask, reference.valid_mask)

    def test_extract_all_bursts(self, swath: S1Swath) -> None:
        """extract_bursts returns every burst in order."""
        bursts = extract_bursts(swath)
        assert len(bursts) == len(swath.bursts)
        for i, b in enumerate(bursts):
            assert b.burst_index == i

    def test_extract_rejects_out_of_range(self, swath: S1Swath) -> None:
        """Out-of-range indices raise Sentinel1ProductError."""
        with pytest.raises(Sentinel1ProductError, match="out of range"):
            extract_burst(swath, burst_index=99)

    def test_estimate_burst_bytes(self, swath: S1Swath) -> None:
        """estimate_burst_bytes returns lines*samples*4."""
        expected = swath.lines_per_burst * swath.samples_per_burst * 4
        assert estimate_burst_bytes(swath, burst_index=0) == expected


class TestExportSynthetic:
    """Export round-trip on the synthetic product."""

    @pytest.fixture(scope="class")
    def swath(self) -> Iterator[S1Swath]:
        """Yield a synthetic S1Swath backed by an in-memory TIFF."""
        sw, memfile = _make_synthetic_swath()
        yield sw
        memfile.close()

    def test_export_writes_safe_subset(self, swath: S1Swath, tmp_path: Path) -> None:
        """Export produces a .SAFE dir with measurement TIFF + annotation."""
        safe_root = export_swath_bursts(
            swath, [0, 1], tmp_path, product_id="SYNTHETIC_TEST"
        )
        assert safe_root.is_dir()
        assert safe_root.name == "SYNTHETIC_TEST.SAFE"
        tiffs = list((safe_root / "measurement").glob("*.tiff"))
        assert len(tiffs) == 1
        anns = list((safe_root / "annotation").glob("*.xml"))
        assert len(anns) == 1

    def test_exported_tiff_round_trips(self, swath: S1Swath, tmp_path: Path) -> None:
        """The exported TIFF contains the concatenated burst samples."""
        bursts = extract_bursts(swath, burst_indices=[0, 1])
        stacked = np.concatenate([b.samples for b in bursts], axis=0)

        safe_root = export_swath_bursts(
            swath, [0, 1], tmp_path, product_id="SYNTHETIC_RT"
        )
        tiff_path = next((safe_root / "measurement").glob("*.tiff"))
        with rasterio.open(tiff_path) as ds:
            assert ds.height == stacked.shape[0]
            assert ds.width == stacked.shape[1]
            assert ds.dtypes[0] == "complex64"
            read_back = ds.read(1)
        np.testing.assert_array_equal(read_back, stacked)

    def test_exported_annotation_keeps_only_selected_bursts(
        self, swath: S1Swath, tmp_path: Path
    ) -> None:
        """The trimmed annotation burstList has the requested count only."""
        import xml.etree.ElementTree as ET

        safe_root = export_swath_bursts(
            swath, [1], tmp_path, product_id="SYNTHETIC_ONE"
        )
        ann_path = next((safe_root / "annotation").glob("*.xml"))
        root = ET.fromstring(ann_path.read_bytes())
        st = next(c for c in root if c.tag.rsplit("}", 1)[-1] == "swathTiming")
        bl = next(c for c in st if c.tag.rsplit("}", 1)[-1] == "burstList")
        bursts = [c for c in bl if c.tag.rsplit("}", 1)[-1] == "burst"]
        assert len(bursts) == 1
        assert bl.get("count") == "1"

    def test_export_rejects_empty_selection(
        self, swath: S1Swath, tmp_path: Path
    ) -> None:
        """Empty burst_indices is rejected."""
        with pytest.raises(Sentinel1ProductError, match="non-empty"):
            export_swath_bursts(swath, [], tmp_path)


@pytest.mark.slow
@pytest.mark.skipif(not REAL_ZIP.exists(), reason="local S1 ZIP corpus unavailable")
class TestExtractRealZip:
    """Integration: byte-offset fast path == rasterio window path on real SAFE."""

    @pytest.fixture(scope="class")
    def iw1(self) -> S1Swath:
        """Yield IW1 of the real product."""
        product = open_safe_product(REAL_ZIP)
        return product.swath("IW1", "VV")

    @pytest.mark.parametrize("burst_index", [0, 4, 8])
    def test_byteoffset_matches_rasterio(self, iw1: S1Swath, burst_index: int) -> None:
        """The ZIP byte-offset path is bitwise identical to rasterio reads."""
        extracted = extract_burst(iw1, burst_index=burst_index)
        reference = read_full_burst(iw1, burst_index=burst_index)
        assert extracted.samples.shape == reference.samples.shape
        np.testing.assert_array_equal(extracted.samples, reference.samples)
        np.testing.assert_array_equal(extracted.valid_mask, reference.valid_mask)
        assert extracted.row0 == reference.row0
        assert extracted.col0 == reference.col0

    def test_estimate_burst_bytes_matches_real_geometry(self, iw1: S1Swath) -> None:
        """estimate_burst_bytes returns lines_per_burst*samples_per_burst*4."""
        expected = iw1.lines_per_burst * iw1.samples_per_burst * 4
        assert estimate_burst_bytes(iw1, burst_index=0) == expected

    def test_extract_all_real_bursts(self, iw1: S1Swath) -> None:
        """Every burst of a real swath can be extracted via the fast path."""
        bursts = extract_bursts(iw1)
        assert len(bursts) == len(iw1.bursts)
        for b in bursts:
            assert b.samples.dtype == np.complex64
            assert b.samples.shape[0] == iw1.lines_per_burst
