"""Test PROJ environment isolation."""

from __future__ import annotations

import os

import pyproj
import pytest
from rasterio.crs import CRS


def test_epsg_4326_uses_uv_environment_proj_database() -> None:
    """Resolve EPSG:4326 with the uv environment's PROJ database."""
    # Given: pytest was launched from a shell that may define PROJ search paths.

    # When: rasterio resolves a CRS using the uv-managed environment.
    crs = CRS.from_epsg(4326)

    # Then: malformed overrides are absent and the canonical EPSG CRS resolves.
    assert os.environ.get("PROJ_DATA") != ""
    assert os.environ.get("PROJ_LIB") != ""
    assert crs.to_epsg() == 4326


def test_valid_proj_data_override_is_preserved() -> None:
    """Preserve a valid PROJ_DATA override supplied to pytest."""
    # Given: the caller supplies a PROJ_DATA path known to resolve EPSG:4326.
    expected_path = os.environ.get("FANINSAR_EXPECT_PROJ_DATA")
    if expected_path is None:
        pytest.skip("valid PROJ_DATA preservation probe was not requested")

    # When: the root conftest validates the inherited environment at collection.

    # Then: it retains the valid override and Rasterio resolves the CRS.
    assert os.environ.get("PROJ_DATA") == expected_path
    assert pyproj.CRS.from_epsg(4326).to_epsg() == 4326
    assert CRS.from_epsg(4326).to_epsg() == 4326


def test_valid_proj_lib_override_is_preserved() -> None:
    """Preserve a valid PROJ_LIB override supplied to pytest."""
    # Given: the caller supplies a PROJ_LIB path known to resolve EPSG:4326.
    expected_path = os.environ.get("FANINSAR_EXPECT_PROJ_LIB")
    if expected_path is None:
        pytest.skip("valid PROJ_LIB preservation probe was not requested")

    # When: the root conftest validates the inherited environment at collection.

    # Then: it retains the valid override and Rasterio resolves the CRS.
    assert os.environ.get("PROJ_LIB") == expected_path
    assert pyproj.CRS.from_epsg(4326).to_epsg() == 4326
    assert CRS.from_epsg(4326).to_epsg() == 4326
