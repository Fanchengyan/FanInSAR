"""Lazy exports for optional remote-provider adapters.

Importing this namespace is offline and does not import provider SDKs.  The
adapter modules themselves import optional dependencies at their explicit I/O
boundaries, so applications can inspect or register a provider without making
the base FanInSAR installation depend on every provider package.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ASF_SEARCH_CERTIFIED_VERSION": (".asf_search", "ASF_SEARCH_CERTIFIED_VERSION"),
    "ASF_SEARCH_COMPATIBILITY": (".asf_search", "ASF_SEARCH_COMPATIBILITY"),
    "ASFSearchAdapter": (".asf_search", "ASFSearchAdapter"),
    "ASFSearchEngineAdapter": (".asf_search", "ASFSearchEngineAdapter"),
    "ASFSearchError": (".asf_search", "ASFSearchError"),
    "ASFSearchUnavailableError": (".asf_search", "ASFSearchUnavailableError"),
    "ASFSearchVersionError": (".asf_search", "ASFSearchVersionError"),
    "AsfSearchAdapter": (".asf_search", "AsfSearchAdapter"),
    "EngineUnavailableError": (".asf_search", "EngineUnavailableError"),
    "UnobservableASFSearchError": (".asf_search", "UnobservableASFSearchError"),
    "UnsupportedASFSearchVersionError": (
        ".asf_search",
        "UnsupportedASFSearchVersionError",
    ),
    "register_asf_catalog": (".asf_search", "register_asf_catalog"),
    "register_asf_search_catalog": (".asf_search", "register_asf_search_catalog"),
    "COP_DEM_GLO30_COLLECTION": (
        ".planetary_computer",
        "COP_DEM_GLO30_COLLECTION",
    ),
    "PC_ASSET_HOST": (".planetary_computer", "PC_ASSET_HOST"),
    "PC_STAC_ENDPOINT": (".planetary_computer", "PC_STAC_ENDPOINT"),
    "PlanetaryComputerAdapter": (
        ".planetary_computer",
        "PlanetaryComputerAdapter",
    ),
    "PlanetaryComputerCollectionAdapter": (
        ".planetary_computer",
        "PlanetaryComputerCollectionAdapter",
    ),
    "register_pc_catalog": (".planetary_computer", "register_pc_catalog"),
    "register_planetary_computer": (
        ".planetary_computer",
        "register_planetary_computer",
    ),
}
__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve one provider symbol on demand."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from error
    value = getattr(importlib.import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
