import matplotlib.colors as mcolors  # noqa: D100, INP001
import pytest

from faninsar.cmaps import (
    GMT,
    SCM,
    GnBu_RdPl,
    RdGyBu,
    WtBuPl,
    WtHeatRed,
    cmocean,
    colorcet,
)


class TestColormaps:
    """Test colormaps."""

    def test_imports(self) -> None:
        """Test if all colormaps are imported correctly."""
        assert hasattr(SCM, "acton")
        assert hasattr(GMT, "abyss")
        assert hasattr(cmocean, "algae")
        assert hasattr(colorcet, "bgy")

    def test_custom_colormaps_creation(self) -> None:
        """Test if custom colormaps are created correctly."""
        assert isinstance(GnBu_RdPl, mcolors.LinearSegmentedColormap)
        assert isinstance(WtBuPl, mcolors.LinearSegmentedColormap)
        assert isinstance(WtHeatRed, mcolors.LinearSegmentedColormap)
        assert isinstance(RdGyBu, mcolors.LinearSegmentedColormap)

    def test_custom_colormaps_properties(self) -> None:
        """Test properties of custom colormaps."""
        assert GnBu_RdPl.name == "GnBu_RdPl"
        assert WtBuPl.name == "WtBuPl"
        assert WtHeatRed.name == "WtHeatRed"
        assert RdGyBu.name == "RdGrBu"

        assert GnBu_RdPl.N == 100
        assert WtBuPl.N == 100
        assert WtHeatRed.N == 100
        assert RdGyBu.N == 100

    def test_custom_colormaps_reversed(self) -> None:
        """Test reversed colormaps."""
        GnBu_RdPl_r = mcolors.LinearSegmentedColormap.from_list(  # noqa: N806
            "GnBu_RdPl_r",
            ["#8f07ff", "#d5734a", "0.95", "#0571b0", "#01ef6c"][::-1],
            N=100,
        )
        WtBuPl_r = mcolors.LinearSegmentedColormap.from_list(  # noqa: N806
            "WtBuPl_r",
            ["0.95", "#0571b0", "#8f07ff", "#d5734a"][::-1],
            N=100,
        )
        WtHeatRed_r = mcolors.LinearSegmentedColormap.from_list(  # noqa: N806
            "WtHeatRed_r",
            ["0.95", "#fff7b3", "#fb9d59", "#aa0526"][::-1],
            N=100,
        )

        assert isinstance(GnBu_RdPl_r, mcolors.LinearSegmentedColormap)
        assert isinstance(WtBuPl_r, mcolors.LinearSegmentedColormap)
        assert isinstance(WtHeatRed_r, mcolors.LinearSegmentedColormap)

        assert GnBu_RdPl_r.name == "GnBu_RdPl_r"
        assert WtBuPl_r.name == "WtBuPl_r"
        assert WtHeatRed_r.name == "WtHeatRed_r"

        assert GnBu_RdPl_r.N == 100
        assert WtBuPl_r.N == 100
        assert WtHeatRed_r.N == 100
