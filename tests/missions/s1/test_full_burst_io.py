"""Tests for full-burst I/O and burst stitching.

Non-slow tests use a small synthetic in-memory TIFF backed by the committed
minimal annotation fixture.  Slow tests require local SAFE ZIP scenes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile

from faninsar.missions.s1 import (
    BurstArray,
    open_safe_product,
    read_burst_window,
    read_full_burst,
    read_swath_bursts,
    stitch_bursts,
)
from faninsar.missions.s1.annotation import parse_annotation_xml
from faninsar.missions.s1.errors import Sentinel1ProductError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from faninsar.missions.s1.types import S1Product, S1Swath

FIXTURE_XML = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "sentinel1"
    / "minimal_annotation_iw1.xml"
)

SLC_ROOT = Path("/Volumes/DATA2/TEST_sentinel-1/sentinel-slc")
REAL_ZIP = (
    SLC_ROOT / "S1A_IW_SLC__1SSV_20161207T111852_20161207T111919_014273_01716D_67BC.zip"
)


def _make_synthetic_swath() -> tuple:
    """Return (swath, memfile) for a 2-burst 20x100 synthetic SLC."""
    xml_text = FIXTURE_XML.read_text(encoding="utf-8")
    # Normalize burst 1 valid samples to match burst 0 so stitching tests
    # do not fail on mismatched range crops.
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
    # Build a complex64 GeoTIFF in memory matching the annotation dimensions.
    height = swath.lines
    width = swath.samples
    data = np.zeros((height, width), dtype=np.complex64)
    # Fill burst 0 (lines 0-9) with a known pattern.
    data[:10, :] = np.exp(1j * np.linspace(0, np.pi, width))[None, :]
    # Fill burst 1 (lines 10-19) with a different pattern.
    data[10:, :] = np.exp(1j * np.linspace(np.pi, 2 * np.pi, width))[None, :]
    # Add a ramp along azimuth so stitching continuity is verifiable.
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

    # Patch the swath so rasterio can read from the MemoryFile.
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


class TestSyntheticFullBurst:
    """Fast unit tests against a 20x100 in-memory synthetic product."""

    @pytest.fixture(scope="class")
    def swath(self) -> Iterator[S1Swath]:
        """Yield a synthetic S1Swath backed by an in-memory TIFF."""
        swath, memfile = _make_synthetic_swath()
        yield swath
        memfile.close()

    def test_read_full_burst_shape_matches_lines_per_burst(
        self, swath: S1Swath
    ) -> None:
        """Full burst height equals lines_per_burst; width equals valid columns."""
        burst = read_full_burst(swath, burst_index=0)
        assert burst.samples.shape == (swath.lines_per_burst, swath.samples)
        assert burst.burst_index == 0
        assert burst.row0 == 0
        assert burst.col0 == 0

    def test_read_full_burst_valid_mask_all_true_for_valid_burst(
        self, swath: S1Swath
    ) -> None:
        """When every line is valid the mask is entirely True."""
        burst = read_full_burst(swath, burst_index=0)
        assert np.all(burst.valid_mask)
        assert burst.col0 == 0
        assert burst.samples.shape[1] == swath.samples

    def test_read_full_burst_geocoding_layout_uses_divisible_valid_extent(
        self, swath: S1Swath
    ) -> None:
        """Geocoding layout uses the InSAR.dev-compatible divisible extent."""
        burst = read_full_burst(swath, burst_index=0, geocoding_layout=True)
        assert burst.samples.shape == (8, 100)
        assert burst.row0 == 0
        assert burst.col0 == 0
        assert np.all(burst.valid_mask)

    def test_read_full_burst_rejects_out_of_range_index(self, swath: S1Swath) -> None:
        """Negative or too-large burst indices raise Sentinel1ProductError."""
        with pytest.raises(Sentinel1ProductError, match="out of range"):
            read_full_burst(swath, burst_index=-1)
        with pytest.raises(Sentinel1ProductError, match="out of range"):
            read_full_burst(swath, burst_index=len(swath.bursts))

    def test_read_swath_bursts_returns_all_by_default(self, swath: S1Swath) -> None:
        """Default burst_indices reads every burst in order."""
        bursts = read_swath_bursts(swath)
        assert len(bursts) == len(swath.bursts)
        for i, b in enumerate(bursts):
            assert b.burst_index == i
            assert b.samples.shape[0] == swath.lines_per_burst

    def test_read_swath_bursts_subset(self, swath: S1Swath) -> None:
        """Explicit burst_indices returns only the requested bursts."""
        bursts = read_swath_bursts(swath, burst_indices=[1])
        assert len(bursts) == 1
        assert bursts[0].burst_index == 1
        assert bursts[0].row0 == swath.lines_per_burst

    def test_stitch_two_bursts_continuous_azimuth(self, swath: S1Swath) -> None:
        """Stitching 2 bursts yields a continuous azimuth dimension."""
        bursts = read_swath_bursts(swath, burst_indices=[0, 1])
        stitched = stitch_bursts(bursts, overlap_blend=True)
        expected_height = sum(b.samples.shape[0] for b in bursts)
        assert stitched.samples.shape[0] == expected_height
        assert stitched.samples.shape[1] == bursts[0].samples.shape[1]
        assert stitched.burst_index == -1
        # Because there is no overlap in the synthetic fixture (bursts are
        # contiguous), the stitched azimuth ramp should be monotonic.
        az_mean = stitched.samples.mean(axis=1)
        assert np.all(np.diff(az_mean.real) >= 0)

    def test_stitch_bursts_without_blend(self, swath: S1Swath) -> None:
        """overlap_blend=False stitches by simple valid-mask weighting."""
        bursts = read_swath_bursts(swath, burst_indices=[0, 1])
        stitched = stitch_bursts(bursts, overlap_blend=False)
        assert stitched.samples.shape[0] == sum(b.samples.shape[0] for b in bursts)

    def test_stitch_intersects_mismatched_range(self, swath: S1Swath) -> None:
        """Bursts with different range crops are stitched at the common intersection."""
        b0 = read_full_burst(swath, burst_index=0, range_looks_crop=(0, 50))
        b1 = read_full_burst(swath, burst_index=1, range_looks_crop=(0, 60))
        stitched = stitch_bursts([b0, b1])
        # Common intersection is width 50 (the smaller of the two).
        assert stitched.samples.shape[1] == 50
        assert stitched.col0 == 0

    def test_stitch_rejects_empty_list(self) -> None:
        """Passing an empty burst list raises Sentinel1ProductError."""
        with pytest.raises(Sentinel1ProductError, match="no bursts"):
            stitch_bursts([])

    def test_read_burst_window_is_debug_crop(self, swath: S1Swath) -> None:
        """read_burst_window returns a sub-window of the full burst."""
        window = read_burst_window(
            swath,
            burst_index=0,
            height=4,
            width=8,
            row_offset=2,
            col_offset=3,
        )
        assert window.samples.shape == (4, 8)
        assert window.burst_index == 0
        assert window.row0 == 2
        assert window.col0 == 3


@pytest.mark.slow
@pytest.mark.skipif(not REAL_ZIP.exists(), reason="local S1 ZIP corpus unavailable")
class TestRealFullBurst:
    """Integration tests against real Sentinel-1 SAFE ZIP scenes."""

    @pytest.fixture(scope="class")
    def product(self) -> S1Product:
        """Yield a real S1Product from the local SAFE ZIP corpus."""
        return open_safe_product(REAL_ZIP)

    def test_real_iw1_full_burst_shape(self, product: S1Product) -> None:
        """First burst of IW1 matches lines_per_burst and valid sample width."""
        iw1 = product.swath("IW1", "VV")
        burst = read_full_burst(iw1, burst_index=0)
        assert burst.samples.shape[0] == iw1.lines_per_burst
        assert burst.samples.dtype == np.complex64
        # Width is the valid-sample envelope, not necessarily samples_per_burst.
        assert 0 < burst.samples.shape[1] <= iw1.samples_per_burst
        # At least some samples should be non-zero (the burst is not empty).
        assert float(np.mean(np.abs(burst.samples) > 0)) > 0.1

    def test_real_iw1_valid_mask_non_trivial(self, product: S1Product) -> None:
        """Valid mask covers most but not necessarily all samples."""
        iw1 = product.swath("IW1", "VV")
        burst = read_full_burst(iw1, burst_index=0)
        valid_fraction = float(np.mean(burst.valid_mask))
        assert 0.5 < valid_fraction <= 1.0

    def test_real_read_swath_bursts_all(self, product: S1Product) -> None:
        """Reading all IW1 bursts returns the expected count."""
        iw1 = product.swath("IW1", "VV")
        bursts = read_swath_bursts(iw1)
        assert len(bursts) == len(iw1.bursts)
        for b in bursts:
            assert b.samples.shape[0] == iw1.lines_per_burst

    def test_real_stitch_all_bursts(self, product: S1Product) -> None:
        """Stitching all IW1 bursts yields a continuous swath image."""
        iw1 = product.swath("IW1", "VV")
        bursts = read_swath_bursts(iw1)
        stitched = stitch_bursts(bursts, overlap_blend=True)
        # Height should be at least the sum of burst lines minus overlaps.
        min_height = (
            sum(b.samples.shape[0] for b in bursts)
            - (len(bursts) - 1) * iw1.lines_per_burst // 2
        )
        assert stitched.samples.shape[0] >= min_height
        # Width is the common intersection, so it is <= every burst width.
        assert stitched.samples.shape[1] <= min(b.samples.shape[1] for b in bursts)
        assert stitched.burst_index == -1
