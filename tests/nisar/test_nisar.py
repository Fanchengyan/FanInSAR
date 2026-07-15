"""Tests for NISAR adapter branch selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from faninsar.nisar import NisarProductError, open_nisar_product


def test_open_nisar_rslc_selects_radar_branch(tmp_path: Path) -> None:
    """RSLC products map to the radar dual-coordinate branch."""
    path = tmp_path / "scene.h5"
    path.write_bytes(b"nisar-stub")
    handle = open_nisar_product(path, kind="RSLC", frequency="A", polarization="HH")
    assert handle.coordinate_branch == "radar"
    assert handle.kind == "RSLC"


def test_open_nisar_gslc_selects_geo_branch(tmp_path: Path) -> None:
    """GSLC products map to the geographic dual-coordinate branch."""
    path = tmp_path / "scene.h5"
    path.write_bytes(b"nisar-stub")
    handle = open_nisar_product(path, kind="GSLC")
    assert handle.coordinate_branch == "geo"


def test_missing_nisar_path_raises() -> None:
    """Missing product paths fail before branch selection."""
    with pytest.raises(NisarProductError, match="does not exist"):
        open_nisar_product("/no/such/nisar.h5", kind="RSLC")
