"""Tests for Stack session scaffolding (PROPOSAL-0017)."""

from __future__ import annotations

from pathlib import Path

import pytest

from faninsar.processing.stack import Stack, StackConfig
from faninsar.processing.stack.catalog import SceneCatalog


def test_scene_catalog_from_paths(tmp_path: Path) -> None:
    """Catalog keys dates from SAFE-like names."""
    a = tmp_path / "S1A_IW_SLC__1SDV_20160101T000000_20160101T000001.SAFE"
    b = tmp_path / "S1A_IW_SLC__1SDV_20160113T000000_20160113T000001.SAFE"
    a.mkdir()
    b.mkdir()
    cat = SceneCatalog.from_paths([a, b])
    assert cat.dates == ("20160101", "20160113")
    assert cat.path_for("20160101") == a


def test_stack_from_safes_defaults(tmp_path: Path) -> None:
    """Stack builds default short-baseline pairs and earliest master."""
    paths = []
    for day in ("20160101", "20160113", "20160125"):
        p = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000_{day}T000001.SAFE"
        p.mkdir()
        paths.append(p)
    stack = Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        coreg_mode="geometry",
        pair_max_interval=2,
        pair_max_days=60,
    )
    assert stack.master == "20160101"
    assert len(stack.catalog) == 3
    stack.prepare_scenes()
    assert (tmp_path / "out" / "coreg").is_dir()
    # geometry mode skips measure/invert
    stack.measure_misreg()
    stack.invert_misreg()
    assert stack.date_misreg is None


def test_stack_config_multilook_normalize(tmp_path: Path) -> None:
    """StackConfig coerces multilook to int pair."""
    cfg = StackConfig(work_dir=tmp_path, multilook=(2, 10))
    assert cfg.multilook == (2, 10)
