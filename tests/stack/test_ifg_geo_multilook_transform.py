# The repository intentionally keeps tests as a namespace package.
# ruff: noqa: INP001

"""Geo IFG multilook transform scaling for persisted artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from affine import Affine

from faninsar.missions.s1 import S1Stack
from faninsar.processing.mosaicking.grid import GeoGridSpec
from faninsar.stack.session import _scale_geo_artifact_transform

if TYPE_CHECKING:
    from pathlib import Path

X0 = 320067.4892684915
Y0 = 4284852.990440817


def _session_with_geo_grid(
    tmp_path: Path, transform: Affine, shape: tuple[int, int]
) -> S1Stack:
    """Create a bare Stack session carrying one geographic grid."""
    safes = []
    for date_id in ("20240101", "20240113"):
        safe = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000.SAFE"
        safe.mkdir()
        safes.append(safe)
    return S1Stack.from_safes(
        safes,
        work_dir=tmp_path / "out",
        multilook=(5, 2),
        geo_grid=GeoGridSpec("EPSG:4326", transform, shape=shape),
    )


def test_scale_multiplies_pixel_sizes_and_keeps_origin() -> None:
    """Five azimuth and two range looks scale the two pixel sizes only."""
    transform = (20.0, 0.0, X0, 0.0, -8.0, Y0)

    scaled = _scale_geo_artifact_transform(transform, (5, 2))

    assert scaled == (40.0, 0.0, X0, 0.0, -40.0, Y0)


def test_scale_accepts_affine_and_passthrough_none() -> None:
    """Nine-value affine inputs scale identically; None stays None."""
    affine = Affine(20.0, 0.0, X0, 0.0, -8.0, Y0)

    from_affine = _scale_geo_artifact_transform(affine, (5, 2))

    assert from_affine == (40.0, 0.0, X0, 0.0, -40.0, Y0)
    assert _scale_geo_artifact_transform(None, (5, 2)) is None


def test_ifg_geo_grid_scales_stored_nine_value_transform(
    tmp_path: Path,
) -> None:
    """The mask grid helper reads the stored nine-value affine tuple."""
    stack = _session_with_geo_grid(
        tmp_path,
        Affine(0.0002, 0.0, 12.0, 0.0, -0.0001, 44.0),
        (37555, 11778),
    )

    transform, shape = stack._ifg_geo_grid((5, 2))

    assert transform == Affine(0.0004, 0.0, 12.0, 0.0, -0.0005, 44.0)
    assert shape == (7511, 5889)
