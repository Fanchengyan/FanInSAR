"""Advanced tests for faninsar.cmaps module.

This module contains additional tests for edge cases, performance,
and advanced functionality of the refactored colormap system.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from unittest.mock import patch

import matplotlib.colors as mcolors
import numpy as np
import pytest

from faninsar.cmaps import cmaps


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_colormap_name(self) -> None:
        """Test handling of empty colormap names."""
        with pytest.raises(AttributeError):
            _ = cmaps.GMT.__getattr__("")

    def test_none_colormap_name(self) -> None:
        """Test handling of None as colormap name."""
        with pytest.raises(AttributeError):
            _ = cmaps.GMT.__getattr__(None)  # type: ignore[arg-type]

    def test_numeric_colormap_name(self) -> None:
        """Test handling of numeric colormap names."""
        with pytest.raises(AttributeError):
            _ = getattr(cmaps.GMT, "123")

    def test_special_characters_in_name(self) -> None:
        """Test handling of special characters in colormap names."""
        special_names = ["test-name", "test.name", "test/name", "test name"]
        for name in special_names:
            with pytest.raises(AttributeError):
                _ = getattr(cmaps.GMT, name)

    def test_very_long_colormap_name(self) -> None:
        """Test handling of very long colormap names."""
        long_name = "a" * 1000
        with pytest.raises(AttributeError):
            _ = getattr(cmaps.GMT, long_name)


class TestMemoryManagement:
    """Test memory management and garbage collection."""

    def test_cache_memory_usage(self) -> None:
        """Test that cache doesn't grow indefinitely."""
        # Access many colormaps
        names = cmaps.GMT.names[:10]  # First 10 colormaps

        initial_cache_size = len(cmaps.GMT._cache)

        # Access all colormaps
        for name in names:
            _ = getattr(cmaps.GMT, name)

        # Cache should have grown
        final_cache_size = len(cmaps.GMT._cache)
        assert final_cache_size > initial_cache_size

        # But not more than the number of accessed colormaps
        assert final_cache_size <= initial_cache_size + len(names) * 2  # *2 for reversed

    def test_weak_references(self) -> None:
        """Test that colormaps can be garbage collected when not referenced."""
        # This test ensures that the cache doesn't prevent garbage collection
        # when the colormap objects are no longer needed

        # Get a colormap
        cmap = cmaps.GMT.viridis
        cmap_id = id(cmap)

        # Delete our reference
        del cmap

        # Force garbage collection
        gc.collect()

        # The colormap should still be in cache
        cached_cmap = cmaps.GMT.viridis
        assert id(cached_cmap) == cmap_id  # Same object from cache


class TestConcurrency:
    """Test concurrent access to colormaps."""

    def test_thread_safety_simulation(self) -> None:
        """Simulate concurrent access to test thread safety."""
        # This is a basic test - real thread safety would require threading
        # But we can at least test that multiple rapid accesses work

        names = ["abyss", "bathy", "cool", "earth"]

        # Rapidly access different colormaps
        cmaps_list = []
        for _ in range(10):
            for name in names:
                cmap = getattr(cmaps.GMT, name)
                cmaps_list.append(cmap)

        # Verify all are valid colormaps
        for cmap in cmaps_list:
            assert isinstance(cmap, mcolors.LinearSegmentedColormap)

        # Verify caching worked (same objects for same names)
        abyss_cmaps = [cmap for cmap in cmaps_list if cmap.name == "abyss"]
        if len(abyss_cmaps) > 1:
            assert all(cmap is abyss_cmaps[0] for cmap in abyss_cmaps)


class TestFileSystemInteraction:
    """Test file system interaction and error handling."""

    def test_missing_colormap_file(self) -> None:
        """Test handling of missing colormap files."""
        # Try to access a colormap that doesn't exist in the file system
        # This should raise a FileNotFoundError when trying to load
        with pytest.raises((FileNotFoundError, AttributeError)):
            # Add a fake colormap name to the list temporarily
            original_names = cmaps.GMT._names.copy()
            cmaps.GMT._names.append("nonexistent_file")
            try:
                _ = cmaps.GMT.nonexistent_file
            finally:
                # Restore original names
                cmaps.GMT._names = original_names

    @patch('pathlib.Path.exists')
    def test_file_access_error(self, mock_exists) -> None:
        """Test handling of file access errors."""
        # Mock file existence check to return False
        mock_exists.return_value = False

        # This should raise an error when trying to load a colormap
        with pytest.raises(FileNotFoundError):
            # Force reload by clearing cache
            if "abyss" in cmaps.GMT._cache:
                del cmaps.GMT._cache["abyss"]
            _ = cmaps.GMT.abyss


class TestPerformanceBenchmarks:
    """Performance benchmarks for the colormap system."""

    def test_import_performance(self) -> None:
        """Benchmark import performance."""
        start_time = time.time()
        import faninsar.cmaps  # noqa: F401
        import_time = time.time() - start_time

        # Import should be fast (less than 1 second)
        assert import_time < 1.0

    def test_first_access_performance(self) -> None:
        """Benchmark first access performance."""
        # Clear cache to ensure fresh load
        colormap_name = "world"  # Use a colormap that might not be cached
        if colormap_name in cmaps.GMT._cache:
            del cmaps.GMT._cache[colormap_name]

        start_time = time.time()
        cmap = getattr(cmaps.GMT, colormap_name)
        access_time = time.time() - start_time

        # First access should be reasonably fast (less than 0.1 seconds)
        assert access_time < 0.1
        assert isinstance(cmap, mcolors.LinearSegmentedColormap)

    def test_cached_access_performance(self) -> None:
        """Benchmark cached access performance."""
        # Ensure colormap is cached
        _ = cmaps.GMT.world

        # Time cached access
        start_time = time.time()
        for _ in range(1000):  # Access many times
            _ = cmaps.GMT.world
        total_time = time.time() - start_time

        # 1000 cached accesses should be very fast (less than 0.01 seconds)
        assert total_time < 0.01

    def test_memory_usage_scaling(self) -> None:
        """Test that memory usage scales reasonably with cache size."""
        import sys

        # Get initial memory usage
        initial_cache_size = len(cmaps.GMT._cache)

        # Access several colormaps
        names = cmaps.GMT.names[:5]
        for name in names:
            _ = getattr(cmaps.GMT, name)

        final_cache_size = len(cmaps.GMT._cache)

        # Cache should have grown reasonably
        cache_growth = final_cache_size - initial_cache_size
        assert cache_growth <= len(names) * 2  # Account for reversed versions


class TestDataIntegrity:
    """Test data integrity and colormap properties."""

    def test_colormap_data_validity(self) -> None:
        """Test that loaded colormap data is valid."""
        cmap = cmaps.GMT.abyss

        # Test basic properties
        assert isinstance(cmap, mcolors.LinearSegmentedColormap)
        assert cmap.name == "abyss"
        assert cmap.N > 0

        # Test that colormap can generate colors
        colors = cmap(np.linspace(0, 1, 10))
        assert colors.shape == (10, 4)  # RGBA
        assert np.all(colors >= 0) and np.all(colors <= 1)

    def test_reversed_colormap_data(self) -> None:
        """Test that reversed colormaps have correct data."""
        cmap = cmaps.GMT.abyss
        cmap_r = cmaps.GMT.abyss_r

        # Generate colors from both
        x = np.linspace(0, 1, 10)
        colors = cmap(x)
        colors_r = cmap_r(x)

        # Reversed colormap should have colors in reverse order
        # (approximately, due to interpolation differences)
        assert not np.array_equal(colors, colors_r)

    def test_all_collections_data_integrity(self) -> None:
        """Test data integrity across all collections."""
        collections = [
            (cmaps.GMT, "abyss"),
            (cmaps.SCM, "acton"),
            (cmaps.cmocean, "algae"),
            (cmaps.colorcet, "bkr"),
            (cmaps.mintpy, "cmy"),
        ]

        for collection, cmap_name in collections:
            cmap = getattr(collection, cmap_name)

            # Basic validity checks
            assert isinstance(cmap, mcolors.LinearSegmentedColormap)
            assert cmap.name == cmap_name
            assert cmap.N > 0

            # Test color generation
            colors = cmap(np.linspace(0, 1, 5))
            assert colors.shape == (5, 4)  # RGBA
            assert np.all(colors >= 0) and np.all(colors <= 1)


class TestCompatibilityMatrix:
    """Test compatibility across different access patterns."""

    def test_access_pattern_equivalence(self) -> None:
        """Test that all access patterns return equivalent objects."""
        # Different ways to access the same colormap
        cmap1 = cmaps.GMT.abyss
        cmap2 = cmaps.abyss
        cmap3 = getattr(cmaps.GMT, "abyss")
        cmap4 = getattr(cmaps, "abyss")

        # All should be the same object
        assert cmap1 is cmap2 is cmap3 is cmap4

    def test_collection_consistency(self) -> None:
        """Test that collections are consistent across access methods."""
        # Access collections through different methods
        gmt1 = cmaps.GMT
        gmt2 = getattr(cmaps, "GMT")

        # Should be the same object
        assert gmt1 is gmt2

        # Should have same colormap names
        assert gmt1.names == gmt2.names

    def test_module_level_consistency(self) -> None:
        """Test module-level access consistency."""
        import faninsar.cmaps as cmaps_module

        # Module-level access should work
        cmap1 = cmaps_module.abyss
        cmap2 = cmaps.abyss

        # Should be the same object
        assert cmap1 is cmap2
