"""Tests for the canonical plotting colormap registry."""

from __future__ import annotations

import matplotlib.colors as mcolors
import pytest

from faninsar.plots import cmaps
from faninsar.plots.cm import Cmaps, ColormapLoader


def test_registry_is_the_single_public_colormap_entry_point() -> None:
    """The plotting package exposes one unified registry."""
    assert isinstance(cmaps, Cmaps)
    assert isinstance(cmaps.GMT, ColormapLoader)
    assert isinstance(cmaps.SCM, ColormapLoader)
    assert isinstance(cmaps.cmocean, ColormapLoader)
    assert isinstance(cmaps.colorcet, ColormapLoader)
    assert isinstance(cmaps.mintpy, ColormapLoader)


@pytest.mark.parametrize(
    ("collection", "name"),
    [
        ("GMT", "abyss"),
        ("SCM", "acton"),
        ("cmocean", "algae"),
        ("colorcet", "bkr"),
        ("mintpy", "cmy"),
    ],
)
def test_collection_and_unified_access_share_loaded_colormap(
    collection: str, name: str
) -> None:
    """Collection-qualified and unified lookup return the same object."""
    collection_cmap = getattr(getattr(cmaps, collection), name)
    assert getattr(cmaps, name) is collection_cmap
    assert isinstance(collection_cmap, mcolors.LinearSegmentedColormap)


def test_reversed_access_is_cached_and_distinct() -> None:
    """Every collection supports the standard ``_r`` lookup."""
    cmap = cmaps.GMT.relief
    reversed_cmap = cmaps.GMT.relief_r
    assert reversed_cmap is cmaps.relief_r
    assert reversed_cmap is not cmap
    assert reversed_cmap.name == "relief_r"


def test_names_are_base_names_and_all_contains_reversed_aliases() -> None:
    """Registry discovery separates names from reversed aliases."""
    assert "abyss" in cmaps.names
    assert "abyss_r" not in cmaps.names
    assert "abyss" in cmaps.__all__
    assert "abyss_r" in cmaps.__all__
    assert "abyss" in dir(cmaps)


def test_unknown_colormap_fails_with_attribute_error() -> None:
    """Lookup of an unregistered name does not silently use Matplotlib."""
    with pytest.raises(AttributeError):
        _ = cmaps.not_a_faninsar_colormap


def test_cmap_package_does_not_duplicate_registry_names() -> None:
    """The package exports the registry and loader types only."""
    from faninsar.plots import cm as cm_module

    assert set(cm_module.__all__) == {
        "Cmaps",
        "ColormapLoader",
        "EnhancedLinearSegmentedColormap",
        "cmaps",
    }
