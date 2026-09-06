"""Provider-neutral remote catalog and complete-file transfer facade.

The implementation is organized by responsibility in :mod:`records`,
:mod:`protocols`, :mod:`catalog`, :mod:`access`, :mod:`auth`, :mod:`transfer`,
:mod:`cache`, and :mod:`errors`.  This module remains the stable developer
facade used by existing FanInSAR adapters.
"""

from __future__ import annotations

import importlib
import time
import urllib

import requests

from faninsar.data.query import BoundingBox, Points, Polygons

from .access import (
    _asf_redirect_url,
    _canonicalize_url,
    _lpdaac_redirect_url,
    _registered_origin,
    _representation_kind,
    _response_is_complete,
    _safe_url,
    _validate_anonymous_delivery_url,
    _validate_representation,
    _validate_url,
)
from .auth import (
    _asf_authenticated_session,
    _asf_request,
    _asf_response_body,
    _drain_asf_response,
    _lpdaac_request,
    _netrc_authorization,
    _netrc_credentials,
    _no_auth,
)
from .cache import _identity, _is_qualified, _manifest_path, _matching_manifest
from .catalog import (
    AcquisitionMetadata,
    CatalogItem,
    _normalize_record,
    _query_geometry,
    _register_adapter,
    _register_fixture,
    search,
)
from .errors import (
    RemoteAccessError,
    RemoteError,
    RemoteIntegrityError,
    RemoteLimitError,
    RemoteQueryError,
    _fail,
)
from .protocols import _accepts_ledger, _Adapter, _adapter_items, _CallLedger
from .records import RemoteAsset, RemoteResourceBudget, _normalize_checksum, _sanitize
from .transfer import _RedirectHandler, _stream_download, download

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_CMR_ENDPOINT": ("faninsar.remote.cmr", "DEFAULT_CMR_ENDPOINT"),
    "CMRAdapter": ("faninsar.remote.cmr", "CMRAdapter"),
    "CMRCollectionAdapter": ("faninsar.remote.cmr", "CMRCollectionAdapter"),
    "CMRDiscoveryError": ("faninsar.remote.cmr", "CMRDiscoveryError"),
    "CMRRegistrationError": ("faninsar.remote.cmr", "CMRRegistrationError"),
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
    "MalformedSTACItemError": ("faninsar.remote.standards", "MalformedSTACItemError"),
    "STACProfileError": ("faninsar.remote.standards", "STACProfileError"),
    "UnknownSTACProfileError": (
        "faninsar.remote.standards",
        "UnknownSTACProfileError",
    ),
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
    "ASFSearchUnavailableError": (
        "faninsar.remote.providers.asf_search",
        "ASFSearchUnavailableError",
    ),
    "ASFSearchVersionError": (
        "faninsar.remote.providers.asf_search",
        "ASFSearchVersionError",
    ),
    "AsfSearchAdapter": ("faninsar.remote.providers.asf_search", "AsfSearchAdapter"),
    "EngineUnavailableError": (
        "faninsar.remote.providers.asf_search",
        "EngineUnavailableError",
    ),
    "UnobservableASFSearchError": (
        "faninsar.remote.providers.asf_search",
        "UnobservableASFSearchError",
    ),
    "UnsupportedASFSearchVersionError": (
        "faninsar.remote.providers.asf_search",
        "UnsupportedASFSearchVersionError",
    ),
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
}

__all__ = [
    "AcquisitionMetadata",
    "BoundingBox",
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
