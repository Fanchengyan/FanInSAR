"""Tests for the canonical plotting colormap registry."""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
import pytest

from faninsar.plotting.colormaps import Cmaps, ColormapLoader, cmaps

BUILTIN_NAMES = ("RdGyBu", "GnBu_RdPl", "WtBuPl", "WtBuGn", "WtRdPl", "WtHeatRed")


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


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_builtin_colormap_is_registered_and_valid(name: str) -> None:
    """Project-specific maps remain available through the unified registry."""
    cmap = getattr(cmaps, name)

    assert isinstance(cmap, mcolors.LinearSegmentedColormap)
    assert cmap.N == 100
    assert name in cmaps.names
    assert name in cmaps.__all__
    assert f"{name}_r" in cmaps.__all__
    assert name in dir(cmaps)


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_builtin_colormap_reverse_and_cache(name: str) -> None:
    """Built-in maps use the same lazy reverse and caching contract."""
    cmap = getattr(cmaps, name)
    reversed_cmap = getattr(cmaps, f"{name}_r")

    assert reversed_cmap is getattr(cmaps, f"{name}_r")
    assert reversed_cmap is not cmap
    endpoints = np.array([0.0, 1.0])
    np.testing.assert_allclose(reversed_cmap(endpoints), cmap(endpoints[::-1]))


def test_unknown_colormap_fails_with_attribute_error() -> None:
    """Lookup of an unregistered name does not silently use Matplotlib."""
    with pytest.raises(AttributeError):
        _ = cmaps.not_a_faninsar_colormap


def test_cmap_package_does_not_duplicate_registry_names() -> None:
    """The package exports the registry and loader types only."""
    from faninsar.plotting import colormaps as cm_module

    assert set(cm_module.__all__) == {
        "Cmaps",
        "ColormapLoader",
        "EnhancedLinearSegmentedColormap",
        "cmaps",
    }
