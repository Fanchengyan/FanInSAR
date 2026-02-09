"""Tests for faninsar.isce2.PathManager.

This module tests the PathManager class including the multilook naming fixes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from faninsar.isce2 import Multilook, PathManager


class TestMultilookNaming:
    """Test multilook variable naming consistency."""

    def test_multilook_iteration_variable(self):
        """Test that multilook iteration uses 'looks' variable."""
        pm = PathManager(
            work_dir="/tmp/test",
            multilook=[(5, 15), (10, 30)],
        )
        # This should not raise any errors
        pm.print_all_paths()

    def test_toml_serialization_uses_looks_key(self):
        """Test that TOML serialization uses 'looks' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PathManager(
                work_dir=tmpdir,
                multilook=[(5, 15)],
            )
            toml_str = pm.to_toml()
            # Check that 'looks' key is used in multilook section
            assert "looks" in toml_str
            # Make sure old 'configs' key is not used in multilook section
            # (it may appear in dir_names mapping, so we check the context)
            lines = toml_str.split("\n")
            in_multilook = False
            for line in lines:
                if "[multilook]" in line:
                    in_multilook = True
                elif line.startswith("[") and in_multilook:
                    in_multilook = False
                if in_multilook and "configs" in line and "looks" not in line:
                    pytest.fail("Found 'configs' in multilook section without 'looks'")

    def test_toml_deserialization_reads_looks_key(self):
        """Test that TOML deserialization reads 'looks' key."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[work]
dir = "/tmp/test"

[multilook]
looks = [
    {azimuth = 5, range = 15},
    {azimuth = 10, range = 30}
]

[metadata]
created = "2024-01-01T00:00:00"
version = "1.1"
""")
            f.flush()
            temp_path = Path(f.name)

        try:
            pm = PathManager.from_toml(temp_path)
            assert pm.has_multilook()
            multilooks = pm.get_multilooks()
            assert len(multilooks) == 2
            assert multilooks[0] == Multilook(5, 15)
            assert multilooks[1] == Multilook(10, 30)
        finally:
            temp_path.unlink()

    def test_create_dirs_uses_looks_variable(self):
        """Test that create_dirs uses 'looks' variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PathManager(
                work_dir=tmpdir,
                multilook=[(5, 15)],
            )
            # Should not raise
            pm.create_all_dirs(include_multilook=True)

            # Verify directories were created
            ml_path = pm.multilook_path(5, 15)
            assert (ml_path / "merged").exists()
            assert (ml_path / "interferograms").exists()


class TestPathManagerBasics:
    """Test basic PathManager functionality."""

    def test_create_path_manager(self):
        """Test creating a PathManager."""
        pm = PathManager(work_dir="/tmp/test")
        # Use resolve() to handle symlinks (e.g., /tmp -> /private/tmp on macOS)
        assert pm.work_dir.resolve() == Path("/tmp/test").resolve()

    def test_multilook_property(self):
        """Test multilook property."""
        pm = PathManager(
            work_dir="/tmp/test",
            multilook=[(5, 15), (10, 30)],
        )
        assert pm.has_multilook()
        assert len(pm.get_multilooks()) == 2

    def test_toml_round_trip(self):
        """Test TOML serialization and deserialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = PathManager(
                work_dir=tmpdir,
                slc_dir="/data/slc",
                dem="/data/dem.tif",
                multilook=[(5, 15)],
            )

            toml_path = Path(tmpdir) / "config.toml"
            original.save(toml_path)

            loaded = PathManager.from_toml(toml_path)

            assert loaded.work_dir == original.work_dir
            assert loaded.slc_dir == original.slc_dir
            assert loaded.dem == original.dem
            assert loaded.get_multilooks() == original.get_multilooks()
