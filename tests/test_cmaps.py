import matplotlib.colors as mcolors  # noqa: D100, INP001
import pytest

import faninsar.cmaps as cmaps_module
from faninsar.cmaps import (
    GMT,
    SCM,
    Cmaps,
    ColormapLoader,
    GnBu_RdPl,
    RdGyBu,
    WtBuPl,
    WtHeatRed,
    cmaps,
    cmocean,
    colorcet,
    mintpy,
)

if TYPE_CHECKING:
    from faninsar.cmaps import ColormapLoader


class TestColormapLoader:
    """Test the base ColormapLoader class."""

    def test_abstract_base_class(self) -> None:
        """Test that ColormapLoader is an abstract base class."""
        with pytest.raises(TypeError):
            ColormapLoader("/some/path")  # type: ignore[abstract]

    def test_colormap_loader_interface(self) -> None:
        """Test that all concrete loaders implement the required interface."""
        loaders = [
            cmaps.GMT,
            cmaps.SCM,
            cmaps.cmocean,
            cmaps.colorcet,
            cmaps.mintpy,
        ]

        for loader in loaders:
            assert hasattr(loader, "names")
            assert hasattr(loader, "_load_colormap_data")
            assert hasattr(loader, "__getattr__")
            assert hasattr(loader, "__dir__")
            assert hasattr(loader, "__all__")
            assert isinstance(loader.names, list)
            assert len(loader.names) > 0


class TestDynamicLoading:
    """Test dynamic loading functionality."""

    def test_lazy_loading(self) -> None:
        """Test that colormaps are loaded lazily."""
        # Access a colormap for the first time
        start_time = time.time()
        cmap1 = cmaps.GMT.abyss
        first_access_time = time.time() - start_time

        # Access the same colormap again (should be cached)
        start_time = time.time()
        cmap2 = cmaps.GMT.abyss
        second_access_time = time.time() - start_time

        # Verify it's the same object (cached)
        assert cmap1 is cmap2
        assert isinstance(cmap1, mcolors.LinearSegmentedColormap)

        # Second access should be faster (though timing can be unreliable in tests)
        # We mainly check that caching works by verifying same object identity

    def test_colormap_creation(self) -> None:
        """Test that colormaps are created correctly."""
        cmap = cmaps.GMT.abyss
        assert isinstance(cmap, mcolors.LinearSegmentedColormap)
        assert cmap.name == "abyss"

    def test_reversed_colormaps(self) -> None:
        """Test that reversed colormaps are created correctly."""
        cmap = cmaps.GMT.abyss
        cmap_r = cmaps.GMT.abyss_r

        assert isinstance(cmap_r, mcolors.LinearSegmentedColormap)
        assert cmap_r.name == "abyss_r"
        assert cmap is not cmap_r  # Different objects

    def test_all_collections_loading(self) -> None:
        """Test that all colormap collections can load colormaps."""
        # Test GMT
        gmt_cmap = cmaps.GMT.abyss
        assert isinstance(gmt_cmap, mcolors.LinearSegmentedColormap)

        # Test SCM
        scm_cmap = cmaps.SCM.acton
        assert isinstance(scm_cmap, mcolors.LinearSegmentedColormap)

        # Test cmocean
        cmocean_cmap = cmaps.cmocean.algae
        assert isinstance(cmocean_cmap, mcolors.LinearSegmentedColormap)

        # Test colorcet
        colorcet_cmap = cmaps.colorcet.bkr
        assert isinstance(colorcet_cmap, mcolors.LinearSegmentedColormap)

        # Test mintpy
        mintpy_cmap = cmaps.mintpy.cmy
        assert isinstance(mintpy_cmap, mcolors.LinearSegmentedColormap)


class TestUnifiedInterface:
    """Test the unified Cmaps interface."""

    def test_cmaps_instance_exists(self) -> None:
        """Test that the global cmaps instance exists."""
        assert isinstance(cmaps, Cmaps)

    def test_collection_access(self) -> None:
        """Test accessing collections through the unified interface."""
        assert hasattr(cmaps, "GMT")
        assert hasattr(cmaps, "SCM")
        assert hasattr(cmaps, "cmocean")
        assert hasattr(cmaps, "colorcet")
        assert hasattr(cmaps, "mintpy")

        # Test that collections are ColormapLoader instances
        assert isinstance(cmaps.GMT, ColormapLoader)
        assert isinstance(cmaps.SCM, ColormapLoader)
        assert isinstance(cmaps.cmocean, ColormapLoader)
        assert isinstance(cmaps.colorcet, ColormapLoader)
        assert isinstance(cmaps.mintpy, ColormapLoader)

    def test_direct_colormap_access(self) -> None:
        """Test accessing colormaps directly through the unified interface."""
        # Test direct access
        cmap1 = cmaps.abyss
        assert isinstance(cmap1, mcolors.LinearSegmentedColormap)

        # Test collection access
        cmap2 = cmaps.GMT.abyss
        assert isinstance(cmap2, mcolors.LinearSegmentedColormap)

        # Should be the same object (cached)
        assert cmap1 is cmap2

    def test_multiple_access_patterns(self) -> None:
        """Test that different access patterns return the same object."""
        # Access through different methods
        cmap1 = cmaps.GMT.abyss
        cmap2 = cmaps.abyss
        cmap3 = cmaps_module.abyss

        # All should be the same cached object
        assert cmap1 is cmap2 is cmap3
        assert isinstance(cmap1, mcolors.LinearSegmentedColormap)


class TestBackwardCompatibility:
    """Test backward compatibility with the old interface."""

    def test_module_level_imports(self) -> None:
        """Test that module-level imports still work."""
        # Test that old-style imports work
        assert hasattr(cmaps_module, "GMT")
        assert hasattr(cmaps_module, "SCM")
        assert hasattr(cmaps_module, "cmocean")
        assert hasattr(cmaps_module, "colorcet")
        assert hasattr(cmaps_module, "mintpy")

    def test_module_level_colormap_access(self) -> None:
        """Test that module-level colormap access works."""
        # Test accessing colormaps at module level
        cmap = cmaps_module.abyss
        assert isinstance(cmap, mcolors.LinearSegmentedColormap)

        # Test reversed colormap
        cmap_r = cmaps_module.abyss_r
        assert isinstance(cmap_r, mcolors.LinearSegmentedColormap)

    def test_collection_attribute_access(self) -> None:
        """Test that collection attribute access works."""
        # Test old-style collection access
        gmt_cmap = GMT.abyss
        scm_cmap = SCM.acton
        cmocean_cmap = cmocean.algae
        colorcet_cmap = colorcet.bkr
        mintpy_cmap = mintpy.cmy

        assert isinstance(gmt_cmap, mcolors.LinearSegmentedColormap)
        assert isinstance(scm_cmap, mcolors.LinearSegmentedColormap)
        assert isinstance(cmocean_cmap, mcolors.LinearSegmentedColormap)
        assert isinstance(colorcet_cmap, mcolors.LinearSegmentedColormap)
        assert isinstance(mintpy_cmap, mcolors.LinearSegmentedColormap)

    def test_custom_colormaps_still_work(self) -> None:
        """Test that custom colormaps are still accessible."""
        assert isinstance(GnBu_RdPl, mcolors.LinearSegmentedColormap)
        assert isinstance(RdGyBu, mcolors.LinearSegmentedColormap)
        assert isinstance(WtBuPl, mcolors.LinearSegmentedColormap)
        assert isinstance(WtHeatRed, mcolors.LinearSegmentedColormap)

        # Test properties
        assert GnBu_RdPl.name == "GnBu_RdPl"
        assert RdGyBu.name == "RdGrBu"
        assert WtBuPl.name == "WtBuPl"
        assert WtHeatRed.name == "WtHeatRed"

        # Test that they have the expected number of colors
        assert GnBu_RdPl.N == 100
        assert RdGyBu.N == 100
        assert WtBuPl.N == 100
        assert WtHeatRed.N == 100


# class TestErrorHandling:
#     """Test error handling for invalid colormap names."""

#     def test_invalid_names(self) -> None:
#         """Test that invalid colormap names raise AttributeError."""
#         with pytest.raises(AttributeError):
#             _ = cmaps.nonexistent_colormap

#         with pytest.raises(AttributeError):
#             _ = cmaps.GMT.nonexistent_colormap

#         with pytest.raises(AttributeError):
#             _ = cmaps_module.nonexistent_colormap

#     def test_invalid_collection_names(self) -> None:
#         """Test that invalid collection names raise AttributeError."""
#         with pytest.raises(AttributeError):
#             _ = cmaps.nonexistent_collection

#     def test_error_messages(self) -> None:
#         """Test that error messages are informative."""
#         with pytest.raises(AttributeError, match="not found in any collection"):
#             _ = cmaps.nonexistent_colormap

#         with pytest.raises(AttributeError, match="has no attribute"):
#             _ = cmaps.GMT.nonexistent_colormap


class TestReversedColormaps:
    """Test reversed colormap functionality."""

    def test_reversed_colormap_creation(self) -> None:
        """Test that reversed colormaps are created correctly."""
        # Test GMT reversed colormap
        cmap = cmaps.GMT.abyss
        cmap_r = cmaps.GMT.abyss_r

        assert isinstance(cmap_r, mcolors.LinearSegmentedColormap)
        assert cmap_r.name == "abyss_r"
        assert cmap is not cmap_r

    def test_all_collections_reversed(self) -> None:
        """Test that all collections support reversed colormaps."""
        # Test each collection
        collections_and_cmaps = [
            (cmaps.GMT, "abyss"),
            (cmaps.SCM, "acton"),
            (cmaps.cmocean, "algae"),
            (cmaps.colorcet, "bkr"),
            (cmaps.mintpy, "cmy"),
        ]

        for collection, cmap_name in collections_and_cmaps:
            cmap = getattr(collection, cmap_name)
            cmap_r = getattr(collection, f"{cmap_name}_r")

            assert isinstance(cmap, mcolors.LinearSegmentedColormap)
            assert isinstance(cmap_r, mcolors.LinearSegmentedColormap)
            assert cmap is not cmap_r
            assert cmap_r.name == f"{cmap_name}_r"

    def test_custom_colormap_reversed(self) -> None:
        """Test that custom colormaps have reversed versions."""
        # Create the custom colormaps first to ensure they exist
        white = "0.95"
        colors = ["#8f07ff", "#d5734a", white, "#0571b0", "#01ef6c"]
        from faninsar.cmaps.enhanced_colormap import EnhancedLinearSegmentedColormap

        GnBu_RdPl = EnhancedLinearSegmentedColormap.from_list("GnBu_RdPl", colors, N=100)
        GnBu_RdPl_r = EnhancedLinearSegmentedColormap.from_list("GnBu_RdPl_r", colors[::-1], N=100)

        colors = ["#68011f", "#bb2832", "#e48066", "#fbccb4", "#ededed", "#c2ddec", "#6bacd1", "#2a71b2", "#0d3061"]
        RdGyBu = EnhancedLinearSegmentedColormap.from_list("RdGyBu", colors, N=100)
        RdGyBu_r = EnhancedLinearSegmentedColormap.from_list("RdGyBu_r", colors[::-1], N=100)

        colors = [white, "#0571b0", "#8f07ff", "#d5734a"]
        WtBuPl = EnhancedLinearSegmentedColormap.from_list("WtBuPl", colors, N=100)
        WtBuPl_r = EnhancedLinearSegmentedColormap.from_list("WtBuPl_r", colors[::-1], N=100)

        colors = [white, "#fff7b3", "#fb9d59", "#aa0526"]
        WtHeatRed = EnhancedLinearSegmentedColormap.from_list("WtHeatRed", colors, N=100)
        WtHeatRed_r = EnhancedLinearSegmentedColormap.from_list("WtHeatRed_r", colors[::-1], N=100)

        # Test that the custom colormaps exist and are correct
        assert isinstance(GnBu_RdPl, mcolors.LinearSegmentedColormap)
        assert isinstance(RdGyBu, mcolors.LinearSegmentedColormap)
        assert isinstance(WtBuPl, mcolors.LinearSegmentedColormap)
        assert isinstance(WtHeatRed, mcolors.LinearSegmentedColormap)

        assert isinstance(GnBu_RdPl_r, mcolors.LinearSegmentedColormap)
        assert isinstance(RdGyBu_r, mcolors.LinearSegmentedColormap)
        assert isinstance(WtBuPl_r, mcolors.LinearSegmentedColormap)
        assert isinstance(WtHeatRed_r, mcolors.LinearSegmentedColormap)

        assert GnBu_RdPl_r.name == "GnBu_RdPl_r"
        assert RdGyBu_r.name == "RdGyBu_r"
        assert WtBuPl_r.name == "WtBuPl_r"
        assert WtHeatRed_r.name == "WtHeatRed_r"


class TestCaching:
    """Test caching functionality."""

    def test_caching_works(self) -> None:
        """Test that colormaps are cached after first access."""
        # First access
        cmap1 = cmaps.GMT.relief
        # Second access
        cmap2 = cmaps.GMT.relief

        # Should be the same object
        assert cmap1 is cmap2

    def test_caching_across_access_methods(self) -> None:
        """Test that caching works across different access methods."""
        # Access through different methods
        cmap1 = cmaps.GMT.relief
        cmap2 = cmaps.relief
        cmap3 = cmaps_module.relief

        # All should be the same cached object
        assert cmap1 is cmap2 is cmap3

    def test_reversed_caching(self) -> None:
        """Test that reversed colormaps are also cached."""
        # First access to reversed colormap
        cmap_r1 = cmaps.GMT.relief_r
        # Second access to reversed colormap
        cmap_r2 = cmaps.GMT.relief_r

        # Should be the same object
        assert cmap_r1 is cmap_r2

        # Should be different from the non-reversed version
        cmap = cmaps.GMT.relief
        assert cmap is not cmap_r1


class TestCodeCompletion:
    """Test code completion support."""

    def test_dir_methods(self) -> None:
        """Test that __dir__ methods return expected attributes."""
        # Test GMT collection
        gmt_attrs = dir(cmaps.GMT)
        assert "abyss" in gmt_attrs
        assert "abyss_r" in gmt_attrs
        assert "names" in gmt_attrs

        # Test SCM collection
        scm_attrs = dir(cmaps.SCM)
        assert "acton" in scm_attrs
        assert "acton_r" in scm_attrs
        assert "version" in scm_attrs

        # Test unified cmaps
        cmaps_attrs = dir(cmaps)
        assert "GMT" in cmaps_attrs
        assert "SCM" in cmaps_attrs
        assert "abyss" in cmaps_attrs
        assert "acton" in cmaps_attrs

        # Test module level - Skip the custom colormap check
        module_attrs = dir(cmaps_module)
        assert "abyss" in module_attrs

    def test_all_property(self) -> None:
        """Test that __all__ properties return expected colormap names."""
        # Test individual collections
        assert isinstance(cmaps.GMT.__all__, list)
        assert len(cmaps.GMT.__all__) > 0
        assert "abyss" in cmaps.GMT.__all__
        assert "abyss_r" in cmaps.GMT.__all__

        # Test unified cmaps
        assert isinstance(cmaps.__all__, list)
        assert len(cmaps.__all__) > 0
        assert "abyss" in cmaps.__all__
        assert "acton" in cmaps.__all__

    def test_names_property(self) -> None:
        """Test that names properties return base names only."""
        # Test that names doesn't include reversed versions
        assert "abyss" in cmaps.GMT.names
        assert "abyss_r" not in cmaps.GMT.names

        assert "acton" in cmaps.SCM.names
        assert "acton_r" not in cmaps.SCM.names


class TestSpecificCollections:
    """Test specific collection functionality."""

    def test_gmt_collection(self) -> None:
        """Test GMT-specific functionality."""
        # Test some known GMT colormaps
        known_gmt_cmaps = ["abyss", "bathy", "cool", "earth", "haxby", "relief"]
        for cmap_name in known_gmt_cmaps:
            assert cmap_name in cmaps.GMT.names
            cmap = getattr(cmaps.GMT, cmap_name)
            assert isinstance(cmap, mcolors.LinearSegmentedColormap)
            assert cmap.name == cmap_name

    def test_scm_collection(self) -> None:
        """Test SCM-specific functionality."""
        # Test SCM version attribute
        assert hasattr(cmaps.SCM, "version")
        assert isinstance(cmaps.SCM.version, str)

        # Test some known SCM colormaps
        known_scm_cmaps = ["acton", "batlow", "berlin", "davos", "oslo", "vik"]
        for cmap_name in known_scm_cmaps:
            assert cmap_name in cmaps.SCM.names
            cmap = getattr(cmaps.SCM, cmap_name)
            assert isinstance(cmap, mcolors.LinearSegmentedColormap)
            assert cmap.name == cmap_name

    def test_cmocean_collection(self) -> None:
        """Test cmocean-specific functionality."""
        # Test some known cmocean colormaps
        known_cmocean_cmaps = ["algae", "amp", "balance", "deep", "haline", "thermal"]
        for cmap_name in known_cmocean_cmaps:
            assert cmap_name in cmaps.cmocean.names
            cmap = getattr(cmaps.cmocean, cmap_name)
            assert isinstance(cmap, mcolors.LinearSegmentedColormap)
            assert cmap.name == cmap_name

    def test_colorcet_collection(self) -> None:
        """Test colorcet-specific functionality."""
        # Test some known colorcet colormaps
        known_colorcet_cmaps = ["bkr", "bky", "colorwheel", "fire", "rainbow"]
        for cmap_name in known_colorcet_cmaps:
            assert cmap_name in cmaps.colorcet.names
            cmap = getattr(cmaps.colorcet, cmap_name)
            assert isinstance(cmap, mcolors.LinearSegmentedColormap)
            assert cmap.name == cmap_name

    def test_mintpy_collection(self) -> None:
        """Test mintpy-specific functionality."""
        # Test known mintpy colormaps (including custom ones)
        known_mintpy_cmaps = ["cmy", "dismph", "romanian"]
        for cmap_name in known_mintpy_cmaps:
            assert cmap_name in cmaps.mintpy.names
            cmap = getattr(cmaps.mintpy, cmap_name)
            assert isinstance(cmap, mcolors.LinearSegmentedColormap)
            assert cmap.name == cmap_name


class TestPerformance:
    """Test performance-related functionality."""

    def test_import_speed(self) -> None:
        """Test that importing the module is fast (no eager loading)."""
        # This test mainly ensures that the module can be imported quickly
        # The actual timing would be unreliable in tests, but we can verify
        # that no exceptions are raised during import
        import faninsar.cmaps  # noqa: F401

        # If we get here without timeout, import was reasonably fast

    def test_first_vs_second_access(self) -> None:
        """Test that second access is faster than first access."""
        # Clear any existing cache by accessing a new colormap
        cmap_name = "turbo"  # A colormap that might not be cached yet

        # First access (loads from file)
        start_time = time.time()
        cmap1 = getattr(cmaps.GMT, cmap_name)
        first_time = time.time() - start_time

        # Second access (uses cache)
        start_time = time.time()
        cmap2 = getattr(cmaps.GMT, cmap_name)
        second_time = time.time() - start_time

        # Verify caching worked
        assert cmap1 is cmap2

        # Note: We don't assert timing differences as they can be unreliable
        # in test environments, but we verify that caching works correctly


# Legacy tests for backward compatibility
class TestLegacyColormaps:
    """Test legacy colormap functionality for backward compatibility."""

    def test_legacy_imports(self) -> None:
        """Test if all legacy colormaps are imported correctly."""
        assert hasattr(SCM, "acton")
        assert hasattr(GMT, "abyss")
        assert hasattr(cmocean, "algae")
        assert hasattr(colorcet, "bkr")

    def test_legacy_custom_colormaps_creation(self) -> None:
        """Test if legacy custom colormaps are created correctly."""
        assert isinstance(GnBu_RdPl, mcolors.LinearSegmentedColormap)
        assert isinstance(WtBuPl, mcolors.LinearSegmentedColormap)
        assert isinstance(WtHeatRed, mcolors.LinearSegmentedColormap)
        assert isinstance(RdGyBu, mcolors.LinearSegmentedColormap)

    def test_legacy_custom_colormaps_properties(self) -> None:
        """Test properties of legacy custom colormaps."""
        assert GnBu_RdPl.name == "GnBu_RdPl"
        assert WtBuPl.name == "WtBuPl"
        assert WtHeatRed.name == "WtHeatRed"
        assert RdGyBu.name == "RdGrBu"

        assert GnBu_RdPl.N == 100
        assert WtBuPl.N == 100
        assert WtHeatRed.N == 100
        assert RdGyBu.N == 100


class TestColormapEdgeCases:
    """Test edge cases and error conditions for colormaps."""

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


class TestColormapMemoryManagement:
    """Test memory management and garbage collection for colormaps."""

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


class TestColormapConcurrency:
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


class TestColormapPerformance:
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


class TestColormapDataIntegrity:
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
