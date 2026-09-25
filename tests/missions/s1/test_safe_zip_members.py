"""Regression: SAFE ZIP ingestion must skip measurement-less RFI annotations.

IPF >= 002.36 products ship ``annotation/rfi/*.xml`` metadata without any
matching measurement member.  The ZIP reader must exclude them exactly like
``annotation/calibration/`` entries, or every modern SAFE ZIP fails to open.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from faninsar.missions.s1.safe import open_safe_product

FIXTURE_XML = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "sentinel1"
    / "minimal_annotation_iw1.xml"
)

STEM = "s1a-iw1-slc-vv-20200101t000000-20200101t000027-000001-000001-0001"
ROOT = "S1_TEST.SAFE"
ANNOTATION_NAME = f"{ROOT}/annotation/{STEM}.xml"
RFI_NAME = f"{ROOT}/annotation/rfi/rfi-{STEM.replace('-vv-', '-vh-')}.xml"
CAL_NAME = f"{ROOT}/annotation/calibration/calibration-{STEM}.xml"
MEAS_NAME = f"{ROOT}/measurement/{STEM}.tiff"


def test_safe_zip_skips_rfi_and_calibration_annotations(tmp_path: Path) -> None:
    """RFI/calibration XML members must not be demanded as measurement pairs."""
    zip_path = tmp_path / "S1_TEST.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(ANNOTATION_NAME, FIXTURE_XML.read_text(encoding="utf-8"))
        archive.writestr(RFI_NAME, "<xml/>")
        archive.writestr(CAL_NAME, "<xml/>")
        archive.writestr(MEAS_NAME, b"tiff-bytes")
    product = open_safe_product(zip_path)
    assert len(product.swaths) == 1
    assert product.swaths[0].annotation_path.endswith(f"{STEM}.xml")
