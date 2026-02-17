"""Tests for ISCE2 workflow pair selection behavior."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest

from faninsar.isce2 import PathManager
from faninsar.isce2.workflows import InterferogramStack, SLCStack


def _create_dummy_slc_files(slc_dir: Path, dates: list[str]) -> None:
    """Create dummy SAFE files for acquisition discovery.

    Parameters
    ----------
    slc_dir : Path
        Directory to store dummy SAFE files.
    dates : list[str]
        Acquisition dates in YYYYMMDD format.

    Returns
    -------
    None

    """
    for date in dates:
        name = (
            "S1A_IW_SLC__1SDV_"
            f"{date}T000000_{date}T000000_000000_0000.SAFE"
        )
        (slc_dir / name).touch()


def _create_path_manager(tmp_path: Path, slc_dir: Path, dem_path: Path) -> PathManager:
    """Create a PathManager with required directories.

    Parameters
    ----------
    tmp_path : Path
        Base temporary directory.
    slc_dir : Path
        Directory containing dummy SLC files.
    dem_path : Path
        Dummy DEM path.

    Returns
    -------
    PathManager
        Configured PathManager instance.

    """
    work_dir = tmp_path / "work"
    pm = PathManager(work_dir=work_dir, slc_dir=slc_dir, dem=dem_path)
    pm.run_dir.mkdir(parents=True, exist_ok=True)
    pm.config_dir.mkdir(parents=True, exist_ok=True)
    return pm


def test_pairs_argument_does_not_mutate_state() -> None:
    """Ensure pairs argument drives config generation without mutating state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        slc_dir = tmp_path / "slc"
        slc_dir.mkdir()
        dem_path = tmp_path / "dem.tif"
        dem_path.touch()

        _create_dummy_slc_files(slc_dir, ["20240101", "20240113", "20240125"])
        pm = _create_path_manager(tmp_path, slc_dir, dem_path)

        workflow = InterferogramStack(path_manager=pm)
        workflow.generate_run_files(pairs=["20240101_20240125"])

        assert len(workflow.pairs) == 0
        assert list(workflow.acquisitions.strftime("%Y%m%d")) == [
            "20240101",
            "20240113",
            "20240125",
        ]
        config_pattern = "config_generate_igram_*20240101_20240125*.ini"
        config_files = list(pm.config_dir.glob(config_pattern))
        assert len(config_files) == 1


def test_default_interval_pairs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Warn when pairs are not specified and interval=1 is used."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        slc_dir = tmp_path / "slc"
        slc_dir.mkdir()
        dem_path = tmp_path / "dem.tif"
        dem_path.touch()

        _create_dummy_slc_files(slc_dir, ["20240101", "20240113", "20240125"])
        pm = _create_path_manager(tmp_path, slc_dir, dem_path)

        workflow = InterferogramStack(path_manager=pm)
        caplog.set_level(logging.WARNING)
        workflow.generate_run_files()

        assert len(workflow.pairs) == 0
        generated = list(pm.config_dir.glob("config_generate_igram_*.ini"))
        assert len(generated) == 2
        assert any(
            "Pairs not specified" in record.message for record in caplog.records
        )


def test_pairs_factory_and_full_pairs() -> None:
    """Ensure pairs_factory and full_pairs are consistent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        slc_dir = tmp_path / "slc"
        slc_dir.mkdir()
        dem_path = tmp_path / "dem.tif"
        dem_path.touch()

        _create_dummy_slc_files(slc_dir, ["20240101", "20240113"])
        pm = _create_path_manager(tmp_path, slc_dir, dem_path)

        workflow = SLCStack(path_manager=pm)
        assert workflow.full_pairs == workflow.pairs_factory.full_pairs


def test_set_pairs_used_without_pairs_argument(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ensure set_pairs controls pair selection when no argument is provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        slc_dir = tmp_path / "slc"
        slc_dir.mkdir()
        dem_path = tmp_path / "dem.tif"
        dem_path.touch()

        _create_dummy_slc_files(slc_dir, ["20240101", "20240113", "20240125"])
        pm = _create_path_manager(tmp_path, slc_dir, dem_path)

        workflow = InterferogramStack(path_manager=pm)
        workflow.set_pairs(["20240101_20240113"])

        caplog.set_level(logging.WARNING)
        workflow.generate_run_files()

        assert len(workflow.pairs) == 1
        assert list(workflow.acquisitions.strftime("%Y%m%d")) == [
            "20240101",
            "20240113",
            "20240125",
        ]
        config_pattern = "config_generate_igram_*20240101_20240113*.ini"
        config_files = list(pm.config_dir.glob(config_pattern))
        assert len(config_files) == 1
        assert not any(
            "Pairs not specified" in record.message for record in caplog.records
        )
