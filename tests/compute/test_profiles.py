"""Compile profile registry tests."""

from __future__ import annotations

from faninsar.compute.profiles import PROFILES, get_profile


def test_sentinel1_profiles_exist() -> None:
    assert "sentinel1-cpu" in PROFILES
    assert "sentinel1-cuda" in PROFILES
    assert "sentinel1-mps" in PROFILES


def test_alias_sentinel1() -> None:
    p = get_profile("sentinel1")
    assert p.device == "cpu"
