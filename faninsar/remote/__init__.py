"""Provider-neutral remote catalog and complete-file transfer facade.

The implementation is organized by responsibility in :mod:`records`,
:mod:`protocols`, :mod:`catalog`, :mod:`access`, :mod:`auth`, :mod:`fetch`,
:mod:`cache`, and :mod:`errors`.  This module remains the stable developer
facade used by existing FanInSAR adapters.
"""

from __future__ import annotations

import importlib

from faninsar.data.query import BoundingBox, Points, Polygons

from .catalog import AcquisitionMetadata, CatalogItem, search
from .errors import (
    BurstPendingError,
    RemoteAccessError,
    RemoteError,
    RemoteIntegrityError,
    RemoteLimitError,
    RemoteQueryError,
)
from .fetch.complete import download, probe
from .records import RemoteAsset, RemoteResourceBudget

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_CMR_ENDPOINT": ("faninsar.remote.cmr", "DEFAULT_CMR_ENDPOINT"),
    "CMRAdapter": ("faninsar.remote.cmr", "CMRAdapter"),
    "CMRCollectionAdapter": ("faninsar.remote.cmr", "CMRCollectionAdapter"),
    "CMRDiscoveryError": ("faninsar.remote.cmr", "CMRDiscoveryError"),
    "RegisteredCMRCollection": ("faninsar.remote.cmr", "RegisteredCMRCollection"),
    "discover_cmr": ("faninsar.remote.cmr", "discover_cmr"),
    "iter_cmr_records": ("faninsar.remote.cmr", "iter_cmr_records"),
    "normalize_cmr_granule": ("faninsar.remote.cmr", "normalize_cmr_granule"),
    "normalize_umm_granule": ("faninsar.remote.cmr", "normalize_umm_granule"),
    "register_cmr_catalog": ("faninsar.remote.cmr", "register_cmr_catalog"),
    "register_cmr_collection": ("faninsar.remote.cmr", "register_cmr_collection"),
    "STAC_CORE_VERSION": ("faninsar.remote.standards", "STAC_CORE_VERSION"),
    "STAC_PROFILES": ("faninsar.remote.standards", "STAC_PROFILES"),
    "STAC_PROFILE_VERSIONS": ("faninsar.remote.standards", "STAC_PROFILE_VERSIONS"),
    "SUPPORTED_STAC_EXTENSIONS": (
        "faninsar.remote.standards",
        "SUPPORTED_STAC_EXTENSIONS",
    ),
    "SUPPORTED_STAC_PROFILES": (
        "faninsar.remote.standards",
        "SUPPORTED_STAC_PROFILES",
    ),
    "STACProfileError": ("faninsar.remote.standards", "STACProfileError"),
    "map_stac_item": ("faninsar.remote.standards", "map_stac_item"),
    "normalize_stac_item": ("faninsar.remote.standards", "normalize_stac_item"),
    "stac_item_to_record": ("faninsar.remote.standards", "stac_item_to_record"),
    "validate_stac_item": ("faninsar.remote.standards", "validate_stac_item"),
    "ASF_SEARCH_CERTIFIED_VERSION": (
        "faninsar.remote.providers.asf_search",
        "ASF_SEARCH_CERTIFIED_VERSION",
    ),
    "ASF_SEARCH_COMPATIBILITY": (
        "faninsar.remote.providers.asf_search",
        "ASF_SEARCH_COMPATIBILITY",
    ),
    "ASFSearchAdapter": ("faninsar.remote.providers.asf_search", "ASFSearchAdapter"),
    "ASFSearchEngineAdapter": (
        "faninsar.remote.providers.asf_search",
        "ASFSearchEngineAdapter",
    ),
    "ASFSearchError": ("faninsar.remote.providers.asf_search", "ASFSearchError"),
    "AsfSearchAdapter": ("faninsar.remote.providers.asf_search", "AsfSearchAdapter"),
    "register_asf_catalog": (
        "faninsar.remote.providers.asf_search",
        "register_asf_catalog",
    ),
    "register_asf_search_catalog": (
        "faninsar.remote.providers.asf_search",
        "register_asf_search_catalog",
    ),
    "COP_DEM_GLO30_COLLECTION": (
        "faninsar.remote.providers.planetary_computer",
        "COP_DEM_GLO30_COLLECTION",
    ),
    "PC_ASSET_HOST": ("faninsar.remote.providers.planetary_computer", "PC_ASSET_HOST"),
    "PC_STAC_ENDPOINT": (
        "faninsar.remote.providers.planetary_computer",
        "PC_STAC_ENDPOINT",
    ),
    "PlanetaryComputerAdapter": (
        "faninsar.remote.providers.planetary_computer",
        "PlanetaryComputerAdapter",
    ),
    "PlanetaryComputerCollectionAdapter": (
        "faninsar.remote.providers.planetary_computer",
        "PlanetaryComputerCollectionAdapter",
    ),
    "register_pc_catalog": (
        "faninsar.remote.providers.planetary_computer",
        "register_pc_catalog",
    ),
    "register_planetary_computer": (
        "faninsar.remote.providers.planetary_computer",
        "register_planetary_computer",
    ),
    "CDSE_CATALOG": ("faninsar.remote.providers.cdse", "CDSE_CATALOG"),
    "CDSE_COLLECTION": ("faninsar.remote.providers.cdse", "CDSE_COLLECTION"),
    "CDSE_STAC_ENDPOINT": ("faninsar.remote.providers.cdse", "CDSE_STAC_ENDPOINT"),
    "CDSESearchAdapter": ("faninsar.remote.providers.cdse", "CDSESearchAdapter"),
    "ensure_cdse_catalog": ("faninsar.remote.providers.cdse", "ensure_cdse_catalog"),
    "register_cdse_catalog": (
        "faninsar.remote.providers.cdse",
        "register_cdse_catalog",
    ),
    "request_cdse_token": ("faninsar.remote.providers.cdse", "request_cdse_token"),
}

__all__ = [
    "AcquisitionMetadata",
    "BoundingBox",
    "BurstPendingError",
    "CatalogItem",
    "Points",
    "Polygons",
    "RemoteAccessError",
    "RemoteAsset",
    "RemoteIntegrityError",
    "RemoteLimitError",
    "RemoteQueryError",
    "RemoteResourceBudget",
    "download",
    "probe",
    "search",
]
__all__ += list(_LAZY_EXPORTS)


def __getattr__(name: str) -> object:
    """Resolve optional provider and profile exports on first use."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from error
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value
