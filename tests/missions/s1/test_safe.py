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


def test_asf_burst_zip_reads_mission_id_from_inner_safe(tmp_path: Path) -> None:
    """ASF burst ZIP stems start with S1_; identity comes from the inner SAFE."""
    import zipfile

    xml_text = FIXTURE.read_text(encoding="utf-8").replace(
        "<missionId>S1A</missionId>",
        "<missionId>S1C</missionId>",
    )
    zip_path = tmp_path / "S1_053926_IW1_20251003T111838_VV_0E6D-BURST.zip"
    inner = "S1C_IW_SLC__1SSV_20251003T111838_20251003T111838_004397_008B64_4402.SAFE"
    stem = "s1c-iw1-slc-vv-20251003t111838-20251003t111841-004397-008b64-001"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(f"{inner}/annotation/{stem}.xml", xml_text)
        archive.writestr(f"{inner}/measurement/{stem}.tiff", b"tiff-bytes")
    product = open_safe_product(zip_path)
    assert product.mission_id == "S1C"
    assert product.product_id.startswith("S1C_IW_SLC")


def _one_burst_xml(azimuth_time: str, *, lines: int = 40) -> str:
    """Return a one-burst annotation whose sensing window can overlap a neighbour."""
    xml_text = FIXTURE.read_text(encoding="utf-8")
    start = xml_text.index("<burst>")
    end = xml_text.rindex("</burstList>")
    first = " ".join(["0"] * lines)
    last = " ".join(["99"] * lines)
    burst = (
        "<burst>"
        f"<azimuthTime>{azimuth_time}</azimuthTime>"
        "<azimuthAnxTime>0.0</azimuthAnxTime>"
        f"<sensingTime>{azimuth_time}</sensingTime>"
        "<byteOffset>0</byteOffset>"
        f"<firstValidSample>{first}</firstValidSample>"
        f"<lastValidSample>{last}</lastValidSample>"
        "</burst>"
    )
    xml_text = xml_text[:start] + burst + xml_text[end:]
    xml_text = xml_text.replace(
        '<burstList count="2">',
        '<burstList count="1">',
    )
    xml_text = xml_text.replace(
        "<linesPerBurst>10</linesPerBurst>",
        f"<linesPerBurst>{lines}</linesPerBurst>",
    )
    return xml_text.replace(
        "<numberOfLines>20</numberOfLines>",
        f"<numberOfLines>{lines}</numberOfLines>",
    )


def test_same_date_one_burst_safes_merge_into_one_swath(tmp_path: Path) -> None:
    """ASF one-burst ZIPs of one date become adjacent bursts of one swath."""
    import zipfile

    from faninsar.missions.s1.overlap import burst_overlap_windows
    from faninsar.missions.s1.safe import open_safe_acquisition

    first_xml = _one_burst_xml("2016-12-07T11:18:53.426276")
    second_xml = _one_burst_xml("2016-12-07T11:18:53.467387")
    paths: list[Path] = []
    for index, xml_text in enumerate((first_xml, second_xml), start=5):
        zip_path = tmp_path / f"S1_053926_IW1_20161207T11185{index}_VV_BURST.zip"
        inner = (
            "S1C_IW_SLC__1SSV_20161207T111852_20161207T111852_"
            f"014273_01716D_{index:04d}.SAFE"
        )
        stem = (
            f"s1c-iw1-slc-vv-20161207t111852-20161207t111855-014273-01716d-{index:04d}"
        )
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr(f"{inner}/annotation/{stem}.xml", xml_text)
            archive.writestr(f"{inner}/measurement/{stem}.tiff", b"tiff-bytes")
        paths.append(zip_path)

    product = open_safe_acquisition(paths)
    swath = product.swath("IW1")
    assert [burst.index for burst in swath.bursts] == [0, 1]
    assert swath.bursts[0].measurement_path != swath.bursts[1].measurement_path
    windows = burst_overlap_windows(swath, min_lines=1, min_samples=1)
    assert len(windows) == 1
    assert windows[0].top_index == 0
    assert windows[0].bottom_index == 1


def test_same_date_swath_canvases_merge_into_one_product(tmp_path: Path) -> None:
    """ROI canvases that each hold one IW of a date assemble into one product."""
    import zipfile

    from faninsar.missions.s1.safe import open_acquisition_products

    xml_text = FIXTURE.read_text(encoding="utf-8")
    paths: list[Path] = []
    for swath_name, swath_xml in (
        ("IW1", xml_text),
        ("IW2", xml_text.replace("<swath>IW1</swath>", "<swath>IW2</swath>")),
    ):
        zip_path = tmp_path / (
            "S1C_IW_SLC__1SDV_20161207T111852_20161207T111919_"
            f"014273_01716D_{swath_name}-VV.SAFE.zip"
        )
        inner = (
            "S1C_IW_SLC__1SDV_20161207T111852_20161207T111919_"
            f"014273_01716D_{swath_name}-VV.SAFE"
        )
        stem = (
            f"s1c-{swath_name.lower()}-slc-vv-20161207t111852-"
            "20161207t111919-014273-01716d-001"
        )
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr(f"{inner}/annotation/{stem}.xml", swath_xml)
            archive.writestr(f"{inner}/measurement/{stem}.tiff", b"tiff-bytes")
        paths.append(zip_path)

    products = open_acquisition_products(paths)
    assert len(products) == 1
    assert {swath.swath for swath in products[0].swaths} == {"IW1", "IW2"}
    assert len(products[0].swath("IW1").bursts) == 2
    assert len(products[0].swath("IW2").bursts) == 2


def test_one_burst_swath_canvases_keep_native_burst_indices(tmp_path: Path) -> None:
    """One-burst complementary canvases merge by swath, not as ASF burst segments."""
    import zipfile

    from faninsar.missions.s1.safe import open_acquisition_products

    paths: list[Path] = []
    for swath_name in ("IW1", "IW2"):
        xml_text = _one_burst_xml("2016-12-07T11:18:53.426276").replace(
            "<swath>IW1</swath>",
            f"<swath>{swath_name}</swath>",
        )
        zip_path = tmp_path / (
            "S1C_IW_SLC__1SDV_20161207T111852_20161207T111919_"
            f"014273_01716D_{swath_name}-VV.SAFE.zip"
        )
        inner = (
            "S1C_IW_SLC__1SDV_20161207T111852_20161207T111919_"
            f"014273_01716D_{swath_name}-VV.SAFE"
        )
        stem = (
            f"s1c-{swath_name.lower()}-slc-vv-20161207t111852-"
            "20161207t111919-014273-01716d-001"
        )
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr(f"{inner}/annotation/{stem}.xml", xml_text)
            archive.writestr(f"{inner}/measurement/{stem}.tiff", b"tiff-bytes")
        paths.append(zip_path)

    products = open_acquisition_products(paths)
    assert len(products) == 1
    assert {swath.swath for swath in products[0].swaths} == {"IW1", "IW2"}
    assert [burst.index for burst in products[0].swath("IW1").bursts] == [0]
    assert [burst.index for burst in products[0].swath("IW2").bursts] == [0]


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
