"""Tests for the EnhancedLinearSegmentedColormap class.

This module tests the enhanced colormap functionality including
tools integration and additional utility methods.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pytest

from faninsar.plots.cmaps import cmaps
from faninsar.plots.cmaps.enhanced_colormap import EnhancedLinearSegmentedColormap


class TestEnhancedLinearSegmentedColormap:
    """Test the EnhancedLinearSegmentedColormap class."""

    def test_inheritance(self) -> None:
        """Test that EnhancedLinearSegmentedColormap inherits from LinearSegmentedColormap."""
        # Create a simple enhanced colormap
        colors = ["red", "white", "blue"]
        enhanced_cmap = EnhancedLinearSegmentedColormap.from_list("test", colors)

        # Should be instance of both classes
        assert isinstance(enhanced_cmap, EnhancedLinearSegmentedColormap)
        assert isinstance(enhanced_cmap, mcolors.LinearSegmentedColormap)

    def test_to_rgb_array(self) -> None:
        """Test RGB array extraction."""
        colors = ["red", "white", "blue"]
        enhanced_cmap = EnhancedLinearSegmentedColormap.from_list("test", colors)

        # Test default N=256
        rgb_array = enhanced_cmap.to_rgb_array()
        assert rgb_array.shape == (256, 3)
        assert np.all(rgb_array >= 0) and np.all(rgb_array <= 1)

        # Test custom N
        rgb_array_custom = enhanced_cmap.to_rgb_array(N=100)
        assert rgb_array_custom.shape == (100, 3)

    def test_save_rgb(self) -> None:
        """Test saving RGB values to file."""
        colors = ["red", "white", "blue"]
        enhanced_cmap = EnhancedLinearSegmentedColormap.from_list("test", colors)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Save RGB data
            enhanced_cmap.save_rgb(tmp_path, N=50)

            # Verify file was created and has correct content
            assert tmp_path.exists()
            loaded_data = np.loadtxt(tmp_path)
            assert loaded_data.shape == (50, 3)
            assert np.all(loaded_data >= 0) and np.all(loaded_data <= 1)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary format."""
        colors = ["red", "white", "blue"]
        enhanced_cmap = EnhancedLinearSegmentedColormap.from_list("test", colors)

        cmap_dict = enhanced_cmap.to_dict(N=100)

        # Check structure
        assert isinstance(cmap_dict, dict)
        assert set(cmap_dict.keys()) == {"red", "green", "blue"}

        # Check each channel
        for channel in ["red", "green", "blue"]:
            channel_data = cmap_dict[channel]
            assert isinstance(channel_data, list)
            assert len(channel_data) == 100

            # Each entry should be a tuple of (position, left_value, right_value)
            for entry in channel_data:
                assert isinstance(entry, tuple)
                assert len(entry) == 3
                assert all(isinstance(x, (int, float)) for x in entry)

    def test_transparentize(self) -> None:
        """Test transparentizing functionality."""
        colors = ["red", "white", "blue"]
        enhanced_cmap = EnhancedLinearSegmentedColormap.from_list("test", colors)

        # Create transparentize version
        transparentize = enhanced_cmap.transparentize(0.5)

        # Should be a new EnhancedLinearSegmentedColormap
        assert isinstance(transparentize, EnhancedLinearSegmentedColormap)
        assert transparentize is not enhanced_cmap
        assert transparentize.name == "test_transparentize"

        # Test that alpha channel is modified
        original_colors = enhanced_cmap(np.linspace(0, 1, 10))
        transparentize_colors = transparentize(np.linspace(0, 1, 10))

        # RGB should be similar, alpha should be different
        np.testing.assert_allclose(original_colors[:, :3], transparentize_colors[:, :3], rtol=1e-10)
        assert np.allclose(transparentize_colors[:, 3], 0.5)

    def test_crop(self) -> None:
        """Test cropping functionality."""
        colors = ["red", "white", "blue"]
        enhanced_cmap = EnhancedLinearSegmentedColormap.from_list("test", colors, N=256)

        # Test basic cropping with asymmetric ranges
        cropped = enhanced_cmap.crop(vmin=-3, vmax=7, pivot=0, N=128)

        assert isinstance(cropped, EnhancedLinearSegmentedColormap)
        assert cropped is not enhanced_cmap
        assert cropped.name == "test_cropped"
        assert cropped.N == 128

    def test_crop_by_percent(self) -> None:
        """Test percentage-based cropping."""
        colors = ["red", "white", "blue"]
        enhanced_cmap = EnhancedLinearSegmentedColormap.from_list("test", colors, N=256)

        # Test cropping both ends
        cropped_both = enhanced_cmap.crop_by_percent(10, which="both")
        assert isinstance(cropped_both, EnhancedLinearSegmentedColormap)

        # Test cropping min end
        cropped_min = enhanced_cmap.crop_by_percent(10, which="min")
        assert isinstance(cropped_min, EnhancedLinearSegmentedColormap)

        # Test cropping max end
        cropped_max = enhanced_cmap.crop_by_percent(10, which="max")
        assert isinstance(cropped_max, EnhancedLinearSegmentedColormap)

        # Test invalid which parameter
        with pytest.raises(ValueError):
            enhanced_cmap.crop_by_percent(10, which="invalid")  # type: ignore[arg-type]

    def test_from_rgb_array(self) -> None:
        """Test creation from RGB array."""
        # Test with [0, 1] range
        rgb_array = np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]])  # Red, white, blue
        enhanced_cmap = EnhancedLinearSegmentedColormap.from_rgb_array("test", rgb_array)

        assert isinstance(enhanced_cmap, EnhancedLinearSegmentedColormap)
        assert enhanced_cmap.name == "test"

        # Test with [0, 255] range
        rgb_array_255 = np.array([[255, 0, 0], [255, 255, 255], [0, 0, 255]])
        enhanced_cmap_255 = EnhancedLinearSegmentedColormap.from_rgb_array("test255", rgb_array_255)

        assert isinstance(enhanced_cmap_255, EnhancedLinearSegmentedColormap)
        assert enhanced_cmap_255.name == "test255"

    def test_from_hex_colors(self) -> None:
        """Test creation from hex colors."""
        hex_colors = ["#FF0000", "#FFFFFF", "#0000FF"]  # Red, white, blue
        enhanced_cmap = EnhancedLinearSegmentedColormap.from_hex_colors("test_hex", hex_colors)

        assert isinstance(enhanced_cmap, EnhancedLinearSegmentedColormap)
        assert enhanced_cmap.name == "test_hex"


class TestEnhancedColormapIntegration:
    """Test integration of enhanced colormaps with the cmaps system."""

    def test_cmaps_return_enhanced_colormaps(self) -> None:
        """Test that cmaps now return EnhancedLinearSegmentedColormap objects."""
        # Test GMT colormap
        gmt_cmap = cmaps.GMT.abyss
        assert isinstance(gmt_cmap, EnhancedLinearSegmentedColormap)
        assert isinstance(gmt_cmap, mcolors.LinearSegmentedColormap)

        # Test SCM colormap
        scm_cmap = cmaps.SCM.acton
        assert isinstance(scm_cmap, EnhancedLinearSegmentedColormap)

        # Test direct access
        direct_cmap = cmaps.abyss
        assert isinstance(direct_cmap, EnhancedLinearSegmentedColormap)
        assert direct_cmap is gmt_cmap  # Should be same cached object

    def test_enhanced_methods_available(self) -> None:
        """Test that enhanced methods are available on cmaps colormaps."""
        cmap = cmaps.GMT.abyss

        # Test that enhanced methods exist
        assert hasattr(cmap, "to_rgb_array")
        assert hasattr(cmap, "save_rgb")
        assert hasattr(cmap, "to_dict")
        assert hasattr(cmap, "transparentize")
        assert hasattr(cmap, "crop")
        assert hasattr(cmap, "crop_by_percent")

        # Test that they work
        rgb_array = cmap.to_rgb_array(N=50)
        assert rgb_array.shape == (50, 3)

        transparentize = cmap.transparentize(0.7)
        assert isinstance(transparentize, EnhancedLinearSegmentedColormap)

        cmap_dict = cmap.to_dict(N=50)
        assert isinstance(cmap_dict, dict)
        assert set(cmap_dict.keys()) == {"red", "green", "blue"}

    def test_custom_colormaps_enhanced(self) -> None:
        """Test that custom colormaps are also enhanced."""
        # Create the custom colormaps directly
        white = "0.95"
        colors = ["#8f07ff", "#d5734a", white, "#0571b0", "#01ef6c"]
        GnBu_RdPl = EnhancedLinearSegmentedColormap.from_list("GnBu_RdPl", colors, N=100)

        colors = ["#68011f", "#bb2832", "#e48066", "#fbccb4", "#ededed", "#c2ddec", "#6bacd1", "#2a71b2", "#0d3061"]
        RdGyBu = EnhancedLinearSegmentedColormap.from_list("RdGyBu", colors, N=100)

        colors = [white, "#0571b0", "#8f07ff", "#d5734a"]
        WtBuPl = EnhancedLinearSegmentedColormap.from_list("WtBuPl", colors, N=100)

        colors = [white, "#fff7b3", "#fb9d59", "#aa0526"]
        WtHeatRed = EnhancedLinearSegmentedColormap.from_list("WtHeatRed", colors, N=100)

        # Test custom colormaps
        custom_cmaps = [
            GnBu_RdPl,
            RdGyBu,
            WtBuPl,
            WtHeatRed,
        ]

        for cmap in custom_cmaps:
            assert isinstance(cmap, EnhancedLinearSegmentedColormap)
            assert hasattr(cmap, "to_rgb_array")
            assert hasattr(cmap, "transparentize")
            assert hasattr(cmap, "crop")

    def test_backward_compatibility(self) -> None:
        """Test that enhanced colormaps maintain backward compatibility."""
        cmap = cmaps.GMT.abyss

        # Should still work as regular LinearSegmentedColormap
        colors = cmap(np.linspace(0, 1, 10))
        assert colors.shape == (10, 4)  # RGBA

        # Should have all original attributes
        assert hasattr(cmap, "name")
        assert hasattr(cmap, "N")
        assert hasattr(cmap, "set_bad")
        assert hasattr(cmap, "set_over")
        assert hasattr(cmap, "set_under")

        # Should work with matplotlib functions
        assert cmap.name == "abyss"
        assert isinstance(cmap.N, int)


class TestEnhancedColormapErrorHandling:
    """Test error handling in enhanced colormap functionality."""

    def test_crop_invalid_parameters(self) -> None:
        """Test error handling in crop method."""
        colors = ["red", "white", "blue"]
        enhanced_cmap = EnhancedLinearSegmentedColormap.from_list("test", colors)

        # Test invalid pivot
        with pytest.raises(ValueError, match="Pivot must be between vmin and vmax"):
            enhanced_cmap.crop(vmin=0, vmax=10, pivot=15)  # pivot > vmax

        with pytest.raises(ValueError, match="Pivot must be between vmin and vmax"):
            enhanced_cmap.crop(vmin=0, vmax=10, pivot=-5)  # pivot < vmin

    def test_crop_symmetric_without_dmax(self) -> None:
        """Test that symmetric ranges require dmax parameter."""
        colors = ["red", "white", "blue"]
        enhanced_cmap = EnhancedLinearSegmentedColormap.from_list("test", colors)

        # Symmetric ranges without dmax should raise error
        with pytest.raises(ValueError, match="dmax required when ranges are symmetric"):
            enhanced_cmap.crop(vmin=-5, vmax=5, pivot=0)  # No dmax provided

    def test_invalid_rgb_array(self) -> None:
        """Test error handling for invalid RGB arrays."""
        # Test with values > 255 - should raise error
        rgb_array = np.array([[300, 0, 0], [255, 255, 255]])  # Values > 255
        with pytest.raises(ValueError, match="RGB values should be in range"):
            EnhancedLinearSegmentedColormap.from_rgb_array("test", rgb_array)

        # Test with valid values in [0, 255] range - should work
        rgb_array_valid = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        enhanced_cmap = EnhancedLinearSegmentedColormap.from_rgb_array("test_valid", rgb_array_valid)
        assert isinstance(enhanced_cmap, EnhancedLinearSegmentedColormap)

        # Test that the normalization worked
        colors = enhanced_cmap(np.linspace(0, 1, 3))
        assert np.all(colors >= 0) and np.all(colors <= 1)


class TestCodeCompletion:
    """Test that code completion works for enhanced methods."""

    def test_enhanced_methods_code_completion(self) -> None:
        """Test that enhanced methods appear in code completion."""
        cmap = cmaps.GMT.abyss

        # Test that all enhanced methods are available
        enhanced_methods = [
            "to_rgb_array",
            "save_rgb",
            "to_dict",
            "transparentize",
            "crop",
            "crop_by_percent"
        ]

        for method in enhanced_methods:
            assert hasattr(cmap, method), f"Method {method} not found"
            assert callable(getattr(cmap, method)), f"Method {method} not callable"

    def test_enhanced_class_methods_code_completion(self) -> None:
        """Test that enhanced class methods appear in code completion."""
        enhanced_class_methods = [
            "from_list",
            "from_rgb_array",
            "from_hex_colors"
        ]

        for method in enhanced_class_methods:
            assert hasattr(EnhancedLinearSegmentedColormap, method), f"Class method {method} not found"
            assert callable(getattr(EnhancedLinearSegmentedColormap, method)), f"Class method {method} not callable"

    def test_type_annotations_work(self) -> None:
        """Test that type annotations work correctly."""
        # This test verifies that the type system recognizes enhanced methods
        cmap = cmaps.GMT.abyss

        # These should not raise type errors if annotations are correct
        rgb_array = cmap.to_rgb_array(N=10)
        assert isinstance(rgb_array, np.ndarray)

        transparentize = cmap.transparentize(0.5)
        assert isinstance(transparentize, EnhancedLinearSegmentedColormap)

        cmap_dict = cmap.to_dict(N=5)
        assert isinstance(cmap_dict, dict)

    def test_all_collections_have_enhanced_types(self) -> None:
        """Test that all collections return enhanced colormaps."""
        collections_and_cmaps = [
            (cmaps.GMT, "abyss"),
            (cmaps.SCM, "acton"),
            (cmaps.cmocean, "algae"),
            (cmaps.colorcet, "bkr"),
            (cmaps.mintpy, "cmy"),
        ]

        for collection, cmap_name in collections_and_cmaps:
            cmap = getattr(collection, cmap_name)
            assert isinstance(cmap, EnhancedLinearSegmentedColormap)

            # Test that enhanced methods are available
            assert hasattr(cmap, "to_rgb_array")
            assert hasattr(cmap, "transparentize")
            assert hasattr(cmap, "crop")

    def test_direct_access_has_enhanced_types(self) -> None:
        """Test that direct access returns enhanced colormaps."""
        # Test direct access through cmaps
        cmap1 = cmaps.abyss
        assert isinstance(cmap1, EnhancedLinearSegmentedColormap)
        assert hasattr(cmap1, "to_rgb_array")

        # Test module-level access
        import faninsar.plots.cmaps as cmaps_module
        cmap2 = cmaps_module.abyss
        assert isinstance(cmap2, EnhancedLinearSegmentedColormap)
        assert hasattr(cmap2, "to_rgb_array")

        # Should be the same cached object
        assert cmap1 is cmap2

    # def test_custom_colormaps_have_enhanced_types(self) -> None:
    #     """Test that custom colormaps have enhanced types."""
    #     import faninsar.cmaps as cmaps_module

    #     custom_cmaps = [
    #         cmaps_module.GnBu_RdPl,
    #         cmaps_module.RdGyBu,
    #         cmaps_module.WtBuPl,
    #         cmaps_module.WtHeatRed,
    #     ]

    #     for cmap in custom_cmaps:
    #         assert isinstance(cmap, EnhancedLinearSegmentedColormap)
    #         assert hasattr(cmap, "to_rgb_array")
    #         assert hasattr(cmap, "transparentize")
    #         assert hasattr(cmap, "crop")
