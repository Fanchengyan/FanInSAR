"""Unit tests for Sentinel-1 SAFE annotation parsing and product open."""

from __future__ import annotations

from pathlib import Path

import pytest

from faninsar.missions.s1.annotation import parse_annotation_xml
from faninsar.missions.s1.errors import UnsupportedPolarizationError
from faninsar.missions.s1.safe import open_safe_product

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "sentinel1"
    / "minimal_annotation_iw1.xml"
)
REAL_ZIP = Path(
    "/Volumes/DATA2/TEST_sentinel-1/sentinel-slc/"
    "S1A_IW_SLC__1SSV_20161207T111852_20161207T111919_014273_01716D_67BC.zip"
)


def test_parse_minimal_annotation_fixture() -> None:
    """Parse the committed minimal annotation into typed swath metadata."""
    xml_text = FIXTURE.read_text(encoding="utf-8")
    swath = parse_annotation_xml(
        xml_text,
        annotation_path=str(FIXTURE),
        measurement_path="memory://measurement.tiff",
    )

    assert swath.swath == "IW1"
    assert swath.polarization == "VV"
    assert swath.lines_per_burst == 10
    assert swath.samples_per_burst == 100
    assert len(swath.bursts) == 2
    assert swath.bursts[0].first_valid_sample[0] == 0
    assert len(swath.orbit.vectors) == 3
    assert len(swath.doppler_centroid) == 1
    assert swath.radar_frequency_hz > 0
    assert swath.range_pixel_spacing_m == pytest.approx(
        299_792_458.0 / (2.0 * swath.range_sampling_rate_hz)
    )


def test_malformed_annotation_without_bursts_is_rejected() -> None:
    """Reject annotation XML that omits the burst list payload."""
    xml_text = FIXTURE.read_text(encoding="utf-8").replace(
        '<burstList count="2">',
        '<burstList count="0">',
    )
    # strip burst bodies
    start = xml_text.index("<burst>")
    end = xml_text.rindex("</burstList>")
    broken = xml_text[:start] + xml_text[end:]
    with pytest.raises(Exception, match="burst"):
        parse_annotation_xml(
            broken,
            annotation_path="broken.xml",
            measurement_path="memory://x.tiff",
        )


@pytest.mark.skipif(not REAL_ZIP.exists(), reason="local S1 ZIP corpus unavailable")
def test_open_real_safe_zip_metadata_only() -> None:
    """Open one real SAFE ZIP and validate sub-swath/burst counts without pixels."""
    product = open_safe_product(REAL_ZIP)

    assert product.product_type == "SLC"
    assert product.polarizations == ("VV",)
    assert {swath.swath for swath in product.swaths} == {"IW1", "IW2", "IW3"}
    iw1 = product.swath("IW1", "VV")
    assert iw1.lines_per_burst == 1494
    assert iw1.samples_per_burst == 21387
    assert len(iw1.bursts) == 9
    assert len(iw1.orbit.vectors) >= 2
    assert iw1.measurement_path.startswith("/vsizip/")
    assert iw1.doppler_centroid


@pytest.mark.skipif(not REAL_ZIP.exists(), reason="local S1 ZIP corpus unavailable")
def test_open_real_safe_zip_rejects_missing_polarization() -> None:
    """Reject a polarization filter that is not present in the product."""
    with pytest.raises(UnsupportedPolarizationError, match="VH"):
        open_safe_product(REAL_ZIP, polarizations=("VH",))
