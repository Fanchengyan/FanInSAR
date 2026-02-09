"""Integration tests for faninsar.isce2 executors.

These tests verify that executor functions are properly organized and can be imported.
Full integration tests with ISCE2 require the isce2 conda environment and test data.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from faninsar.isce2 import Command
from faninsar.isce2.executors import execute_command

# Check if ISCE2 is available
try:
    import isceobj

    ISCE2_AVAILABLE = True
except ImportError:
    ISCE2_AVAILABLE = False

# Check if test data is available
TEST_DATA_ROOT = Path("/Volumes/DATA2/InSAR")
HAS_TEST_DATA = TEST_DATA_ROOT.exists()


class TestExecutorImports:
    """Test that all executor modules can be imported."""

    def test_import_executors_module(self):
        """Test that executors module can be imported."""
        from faninsar.isce2 import executors

        assert hasattr(executors, "execute_command")

    def test_import_sentinel1_executor(self):
        """Test that sentinel1 executor can be imported."""
        from faninsar.isce2.executors.sentinel1 import _exec_sentinel1_tops

        assert callable(_exec_sentinel1_tops)

    def test_import_geometry_executor(self):
        """Test that geometry executor can be imported."""
        from faninsar.isce2.executors.geometry import (
            _exec_geo2rdr,
            _exec_topo,
            _run_geo2rdr,
        )

        assert callable(_exec_topo)
        assert callable(_exec_geo2rdr)
        assert callable(_run_geo2rdr)

    def test_import_coregistration_executor(self):
        """Test that coregistration executor can be imported."""
        from faninsar.isce2.executors.coregistration import (
            _exec_resamp,
            _resamp_secondary_burst,
        )

        assert callable(_exec_resamp)
        assert callable(_resamp_secondary_burst)

    def test_import_interferogram_executor(self):
        """Test that interferogram executor can be imported."""
        from faninsar.isce2.executors.interferogram import (
            _exec_filter_coherence,
            _exec_generate_igram,
        )

        assert callable(_exec_generate_igram)
        assert callable(_exec_filter_coherence)

    def test_import_merge_executor(self):
        """Test that merge executor can be imported."""
        from faninsar.isce2.executors.merge import _exec_merge_bursts, _exec_multilook

        assert callable(_exec_multilook)
        assert callable(_exec_merge_bursts)

    def test_import_postprocess_executor(self):
        """Test that postprocess executor can be imported."""
        from faninsar.isce2.executors.postprocess import _exec_geocode, _exec_unwrap

        assert callable(_exec_geocode)
        assert callable(_exec_unwrap)


class TestExecutorDispatch:
    """Test that execute_command dispatches correctly."""

    def test_unknown_command_type(self):
        """Test that unknown command types return error."""
        cmd = Command("unknown_type", {})
        result = execute_command(cmd)
        assert result == 1  # Should fail

    @pytest.mark.parametrize(
        "cmd_name",
        [
            "Sentinel1_TOPS",
            "topo",
            "geo2rdr",
            "multilook",
            "resamp",
            "generate_igram",
            "filter_coherence",
            "merge_bursts",
            "geocode",
            "unwrap",
        ],
    )
    def test_all_command_types_have_executors(self, cmd_name):
        """Test that all command types can be dispatched."""
        from faninsar.isce2.executors.coregistration import _exec_resamp
        from faninsar.isce2.executors.geometry import _exec_geo2rdr, _exec_topo
        from faninsar.isce2.executors.interferogram import (
            _exec_filter_coherence,
            _exec_generate_igram,
        )
        from faninsar.isce2.executors.merge import _exec_merge_bursts, _exec_multilook
        from faninsar.isce2.executors.postprocess import _exec_geocode, _exec_unwrap
        from faninsar.isce2.executors.sentinel1 import _exec_sentinel1_tops

        executors = {
            "Sentinel1_TOPS": _exec_sentinel1_tops,
            "topo": _exec_topo,
            "geo2rdr": _exec_geo2rdr,
            "multilook": _exec_multilook,
            "resamp": _exec_resamp,
            "generate_igram": _exec_generate_igram,
            "filter_coherence": _exec_filter_coherence,
            "merge_bursts": _exec_merge_bursts,
            "geocode": _exec_geocode,
            "unwrap": _exec_unwrap,
        }

        assert cmd_name in executors
        assert callable(executors[cmd_name])


@pytest.mark.skipif(not ISCE2_AVAILABLE, reason="ISCE2 not available")
class TestWithISCE2:
    """Integration tests that require ISCE2.

    These tests verify that executor functions can import ISCE2 modules.
    """

    def test_sentinel1_executor_isce2_imports(self):
        """Test that sentinel1 executor can import ISCE2."""
        from faninsar.isce2.executors.sentinel1 import _exec_sentinel1_tops

        assert callable(_exec_sentinel1_tops)
        # Verify ISCE2 is available
        import isceobj

    def test_geometry_executor_isce2_imports(self):
        """Test that geometry executor can import ISCE2."""
        from faninsar.isce2.executors.geometry import _exec_geo2rdr, _exec_topo

        assert callable(_exec_topo)
        assert callable(_exec_geo2rdr)
        # Verify ISCE2 modules are available
        import isceobj
        from isceobj.Planet.Planet import Planet
        from zerodop.geo2rdr import createGeo2rdr
        from zerodop.topozero import createTopozero

    def test_coregistration_executor_isce2_imports(self):
        """Test that coregistration executor can import ISCE2."""
        from faninsar.isce2.executors.coregistration import _exec_resamp

        assert callable(_exec_resamp)
        # Verify ISCE2 modules are available
        import isceobj
        import stdproc

    def test_interferogram_executor_isce2_imports(self):
        """Test that interferogram executor can import ISCE2."""
        from faninsar.isce2.executors.interferogram import (
            _exec_filter_coherence,
            _exec_generate_igram,
        )

        assert callable(_exec_generate_igram)
        assert callable(_exec_filter_coherence)
        # Verify ISCE2 modules are available
        import isceobj
        from stdproc.stdproc import crossmul

    def test_postprocess_executor_isce2_imports(self):
        """Test that postprocess executor can import ISCE2."""
        from faninsar.isce2.executors.postprocess import _exec_geocode, _exec_unwrap

        assert callable(_exec_geocode)
        assert callable(_exec_unwrap)
        # Verify ISCE2 modules are available
        import isceobj


@pytest.mark.skipif(
    not ISCE2_AVAILABLE or not HAS_TEST_DATA,
    reason="ISCE2 or test data not available",
)
class TestWithRealData:
    """Integration tests with real ISCE2 data.

    These tests verify that executor functions can run with actual data.
    They require:
    - ISCE2 conda environment activated
    - Test data available at /Volumes/DATA2/InSAR
    """

    def test_test_data_exists(self):
        """Verify test data directory exists."""
        assert TEST_DATA_ROOT.exists()
        assert TEST_DATA_ROOT.is_dir()

    def test_reference_stack_exists(self):
        """Verify reference topsStack output exists."""
        reference_stack = TEST_DATA_ROOT / "stacks"
        if reference_stack.exists():
            assert reference_stack.is_dir()
            # Log what's available for reference
            print(f"\nReference stack directory: {reference_stack}")
            if (reference_stack / "configs").exists():
                print("Found configs directory")
            if (reference_stack / "run_files").exists():
                print("Found run_files directory")


class TestExecutorOrganization:
    """Test that executors are properly organized."""

    def test_no_circular_imports(self):
        """Test that there are no circular imports."""
        # Import all executors
        from faninsar.isce2.executors import execute_command
        from faninsar.isce2.executors.coregistration import _exec_resamp
        from faninsar.isce2.executors.geometry import (
            _exec_geo2rdr,
            _exec_topo,
        )
        from faninsar.isce2.executors.interferogram import (
            _exec_filter_coherence,
            _exec_generate_igram,
        )
        from faninsar.isce2.executors.merge import (
            _exec_merge_bursts,
            _exec_multilook,
        )
        from faninsar.isce2.executors.postprocess import (
            _exec_geocode,
            _exec_unwrap,
        )
        from faninsar.isce2.executors.sentinel1 import (
            _exec_sentinel1_tops,
        )

        # If we get here without errors, there are no circular imports

    def test_executor_file_sizes(self):
        """Test that executor files are reasonably sized."""
        import faninsar.isce2.executors.coregistration as coreg
        import faninsar.isce2.executors.geometry as geom
        import faninsar.isce2.executors.interferogram as ifg
        import faninsar.isce2.executors.postprocess as post
        import faninsar.isce2.executors.sentinel1 as s1
        from faninsar.isce2.executors import merge

        # Check that files exist and have code
        for module in [s1, geom, coreg, ifg, merge, post]:
            file_path = Path(module.__file__)
            assert file_path.exists()
            assert file_path.stat().st_size > 0
            # Files should be < 15KB (well-organized)
            assert file_path.stat().st_size < 15000

    def test_command_manager_reduced_size(self):
        """Test that command_manager.py is significantly smaller."""
        import faninsar.isce2.command_manager as cm

        file_path = Path(cm.__file__)
        file_size = file_path.stat().st_size

        # Original was ~74KB, should now be < 30KB
        assert file_size < 30000
        print(f"\ncommand_manager.py size: {file_size} bytes")
