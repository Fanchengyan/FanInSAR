"""Tests for Sentinel-1 external orbit readers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from faninsar.missions.s1 import read_eof_orbit

if TYPE_CHECKING:
    from pathlib import Path


def test_read_eof_orbit_parses_ordered_earth_fixed_vectors(tmp_path: Path) -> None:
    """ESA EOF vectors are exposed as typed UTC orbit metadata."""
    orbit_path = tmp_path / "orbit.EOF"
    orbit_path.write_text(
        """<?xml version="1.0"?>
<Earth_Explorer_File><Data_Block><List_of_OSVs>
<OSV><UTC>UTC=2020-01-01T00:00:00.000000</UTC>
<X>1</X><Y>2</Y><Z>3</Z><VX>4</VX><VY>5</VY><VZ>6</VZ></OSV>
<OSV><UTC>UTC=2020-01-01T00:00:10.000000</UTC>
<X>7</X><Y>8</Y><Z>9</Z><VX>10</VX><VY>11</VY><VZ>12</VZ></OSV>
</List_of_OSVs></Data_Block></Earth_Explorer_File>
""",
        encoding="utf-8",
    )
    orbit = read_eof_orbit(orbit_path)
    assert orbit.reference_frame == "EARTH_FIXED"
    assert len(orbit.vectors) == 2
    assert orbit.vectors[0].time.tzinfo is not None
    assert orbit.vectors[1].position_m == (7.0, 8.0, 9.0)
