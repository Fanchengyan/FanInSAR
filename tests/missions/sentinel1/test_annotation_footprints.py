"""Tests for burst footprint extraction from annotation geolocation grids."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from faninsar.missions.sentinel1.annotation import _parse_burst_footprints

FIXTURE = (
    Path(__file__).parent
    / "data"
    / "iw1-annotation-20161207-f121.xml"
)
SLC_ROOT = Path("/Volumes/DATA2/TEST_sentinel-1/sentinel-slc")
SLC_ROOT_RAW = Path("/Volumes/DATA2/TEST_sentinel-1/Raw Data/sentinel-slc")
SCENES = sorted(SLC_ROOT.glob("S1A_IW_SLC*.zip")) if SLC_ROOT.exists() else []
if not SCENES and SLC_ROOT_RAW.exists():
    SCENES = sorted(SLC_ROOT_RAW.glob("S1A_IW_SLC*.zip"))


def _grid_xml() -> str:
    return """<product>
  <geolocationGrid>
    <geolocationGridPoint><line>0</line><latitude>38.0</latitude><longitude>101.0</longitude></geolocationGridPoint>
    <geolocationGridPoint><line>700</line><latitude>37.0</latitude><longitude>101.5</longitude></geolocationGridPoint>
    <geolocationGridPoint><line>1493</line><latitude>38.5</latitude><longitude>101.2</longitude></geolocationGridPoint>
    <geolocationGridPoint><line>1500</line><latitude>37.5</latitude><longitude>101.8</longitude></geolocationGridPoint>
    <geolocationGridPoint><line>2200</line><latitude>38.2</latitude><longitude>101.6</longitude></geolocationGridPoint>
    <geolocationGridPoint><line>2987</line><latitude>37.8</latitude><longitude>102.0</longitude></geolocationGridPoint>
  </geolocationGrid>
</product>"""


def test_parse_burst_footprints_splits_by_lines_per_burst() -> None:
    """Each burst gets the bounding box of its geolocation grid points."""
    root = ET.fromstring(_grid_xml())
    footprints = _parse_burst_footprints(root, lines_per_burst=1500)

    assert footprints is not None
    assert len(footprints) == 2
    assert footprints[0] == ((101.0, 37.0), (101.5, 37.0), (101.5, 38.5), (101.0, 38.5))
    assert footprints[1] == ((101.6, 37.5), (102.0, 37.5), (102.0, 38.2), (101.6, 38.2))


def test_parse_burst_footprints_returns_none_without_grid() -> None:
    """A product without a geolocation grid yields no footprints."""
    root = ET.fromstring("<product></product>")
    assert _parse_burst_footprints(root, lines_per_burst=1500) is None


@pytest.mark.skipif(not FIXTURE.exists(), reason="annotation fixture missing")
def test_real_annotation_fixture_corner_values() -> None:
    """The f121 IW1 fixture parses to the known burst-0 footprint."""
    root = ET.fromstring(FIXTURE.read_bytes())
    footprints = _parse_burst_footprints(root, lines_per_burst=1494)
    assert footprints is not None
    assert len(footprints) >= 9
    first = footprints[0]
    assert first[0][0] == pytest.approx(98.448, abs=0.01)
    assert first[0][1] == pytest.approx(37.382, abs=0.01)
    assert first[2][0] == pytest.approx(99.435, abs=0.01)
    assert first[2][1] == pytest.approx(37.528, abs=0.01)
    assert footprints[8][0][1] > footprints[0][2][1]


@pytest.mark.slow
@pytest.mark.skipif(not SCENES, reason="need a local S1 ZIP scene")
def test_real_product_bursts_carry_footprints() -> None:
    """Every IW1 burst of a real SAFE exposes a finite footprint polygon."""
    from faninsar.missions.sentinel1.safe import open_safe_product

    swath = open_safe_product(SCENES[0]).swath("IW1")
    assert len(swath.bursts) >= 2
    for burst in swath.bursts:
        assert burst.footprint is not None
        lon = [point[0] for point in burst.footprint]
        lat = [point[1] for point in burst.footprint]
        assert min(lon) < max(lon)
        assert min(lat) < max(lat)
        assert min(lon) > 60.0
        assert max(lon) < 120.0
        assert min(lat) > -10.0
        assert max(lat) < 60.0
