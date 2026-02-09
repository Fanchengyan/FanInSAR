"""Tests for faninsar.isce2.TopsStackCommands.

This module tests the TopsStackCommands class and command execution.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from faninsar.isce2 import Command, PathManager, TopsStackCommands


class TestCommandDataclass:
    """Test Command dataclass."""

    def test_command_creation(self):
        """Test creating a Command."""
        cmd = Command(
            cmd_name="topo",
            params={"reference": "/data/ref", "dem": "/data/dem.tif"},
        )
        assert cmd.cmd_name == "topo"
        assert cmd.params["reference"] == "/data/ref"


class TestTopsStackCommands:
    """Test TopsStackCommands class."""

    @pytest.fixture
    def path_manager(self):
        """Create PathManager for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield PathManager(work_dir=tmpdir, dem="/data/dem.tif")

    @pytest.fixture
    def cmd_manager(self, path_manager):
        """Create TopsStackCommands for testing."""
        return TopsStackCommands(path_manager=path_manager)

    def test_initialization(self, cmd_manager):
        """Test TopsStackCommands initialization."""
        assert cmd_manager.num_process == 1
        assert cmd_manager.use_gpu is False
        assert len(cmd_manager.command_queue) == 0

    def test_add_command(self, cmd_manager):
        """Test adding commands."""
        cmd = Command("test", {})
        cmd_manager.add_command(cmd)
        # With num_process=1, the batch should be flushed immediately
        assert len(cmd_manager.current_batch) == 0
        assert len(cmd_manager.command_queue) == 1

    def test_command_builders_return_command(self, cmd_manager):
        """Test that command builders return Command objects."""
        # Test topo command builder - no arguments needed, uses PathManager
        cmd = cmd_manager.topo_cmd()
        assert isinstance(cmd, Command)
        assert cmd.cmd_name == "topo"
        assert "reference" in cmd.params
        assert "dem" in cmd.params

    def test_sentinel1_tops_cmd(self, cmd_manager):
        """Test Sentinel1_TOPS command builder."""
        cmd = cmd_manager.sentinel1_tops_cmd(
            safe_file=Path("/data/S1A.SAFE"),
            outdir=Path("/data/output"),
            swaths=[1, 2, 3],
            polarization="vv",
            orbit_file=Path("/data/orbit.EOF"),
            orbit_type="precise",
        )
        assert isinstance(cmd, Command)
        assert cmd.cmd_name == "Sentinel1_TOPS"

    def test_geo2rdr_cmd(self, cmd_manager):
        """Test geo2rdr command builder."""
        cmd = cmd_manager.geo2rdr_cmd(date="20240113")
        assert isinstance(cmd, Command)
        assert cmd.cmd_name == "geo2rdr"

    def test_batch_operations(self, cmd_manager):
        """Test batch command operations."""
        # Use num_process > 1 to avoid immediate flushing
        cmd_manager.num_process = 3
        cmd1 = Command("test1", {})
        cmd2 = Command("test2", {})

        cmd_manager.add_command(cmd1)
        cmd_manager.add_command(cmd2)

        # Should still be in current batch since we haven't reached num_process
        assert len(cmd_manager.current_batch) == 2
        assert len(cmd_manager.command_queue) == 0

        cmd_manager.flush_batch()
        assert len(cmd_manager.current_batch) == 0
        assert len(cmd_manager.command_queue) == 1
        assert len(cmd_manager.command_queue[0]) == 2


class TestCommandBuilders:
    """Test all command builder methods."""

    @pytest.fixture
    def cmd_manager(self):
        """Create TopsStackCommands for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PathManager(work_dir=tmpdir)
            yield TopsStackCommands(path_manager=pm)

    def test_resamp_cmd(self, cmd_manager):
        """Test resamp command builder."""
        cmd = cmd_manager.resamp_cmd(
            reference=Path("/data/ref"),
            secondary=Path("/data/sec"),
            coreg_dir=Path("/data/coreg"),
        )
        assert cmd.cmd_name == "resamp"

    def test_generate_igram_cmd(self, cmd_manager):
        """Test generate_igram command builder."""
        cmd = cmd_manager.generate_igram_cmd(
            reference=Path("/data/ref"),
            secondary=Path("/data/sec"),
            coreg_dir=Path("/data/coreg"),
        )
        assert cmd.cmd_name == "generate_igram"

    def test_filter_coherence_cmd(self, cmd_manager):
        """Test filter_coherence command builder."""
        cmd = cmd_manager.filter_coherence_cmd(
            interferogram=Path("/data/int.int"),
            coherence=Path("/data/coh.cor"),
            filtered_int=Path("/data/filt.int"),
        )
        assert cmd.cmd_name == "filter_coherence"

    def test_multilook_cmd(self, cmd_manager):
        """Test multilook command builder."""
        cmd = cmd_manager.multilook_cmd(
            input_path=Path("/data/input.slc"),
            azimuth=5,
            range_looks=15,
        )
        assert cmd.cmd_name == "multilook"
        assert cmd.params["azimuth"] == 5
        assert cmd.params["range_looks"] == 15
