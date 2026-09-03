"""Lazy, ledger-observable adapter for the optional ``asf-search`` package.

The package is an ASF discovery convenience layer, not a transport boundary.
This module supplies it with one short-lived operation session, meters every
request and response, and immediately converts products through the generic
CMR normalizer.  No package product or download helper crosses this module.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
import urllib.parse
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from faninsar.logging import setup_logger
from faninsar.remote import (
    RemoteAccessError,
    RemoteLimitError,
    RemoteResourceBudget,
    _CallLedger,
    _fail,
    _register_adapter,
)
from faninsar.remote.cmr import CMRCollectionAdapter

if TYPE_CHECKING:
    from datetime import datetime

logger = setup_logger(__name__)

ASF_SEARCH_MIN_MAJOR = 13
ASF_SEARCH_MAX_MAJOR = 14
ASF_SEARCH_CERTIFIED_VERSION = "13.0.0"
ASF_SEARCH_COMPATIBILITY = ">=13,<14"
DEFAULT_CMR_ENDPOINT = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
DEFAULT_DATA_ORIGINS = ("https://datapool.asf.alaska.edu",)


class ASFSearchError(RemoteAccessError):
    """The optional ASF discovery operation failed."""


class EngineUnavailableError(ASFSearchError):
    """The explicitly requested optional engine cannot be used safely."""


class UnsupportedASFSearchVersionError(EngineUnavailableError):
    """The installed ``asf-search`` version is outside the certified line."""


class UnobservableASFSearchError(EngineUnavailableError):
    """The package session cannot be instrumented before provider I/O."""


def _error(error_type: type[ASFSearchError], reason: str, message: str) -> None:
    """Log and raise one typed provider error."""
    logger.error("ASF search failed: %s (%s)", message, reason)
    _fail(error_type, reason, message)


def _version(value: Any) -> tuple[int, ...] | None:
    """Parse a package version without importing a version parser package."""
    if not isinstance(value, str):
        return None
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
    if match is None:
        return None
    return tuple(int(part or 0) for part in match.groups())


def _load_package(package: Any | None) -> Any:
    """Load and validate the optional dependency lazily."""
    if package is None:
        try:
            package = importlib.import_module("asf_search")
        except ImportError as exc:
            _error(
                EngineUnavailableError,
                "asf_search_unavailable",
                "install the optional 'asf' dependency to use asf-search",
            )
            raise AssertionError from exc
    package_version = getattr(package, "__version__", None)
    if package_version is None:
        try:
            package_version = importlib.metadata.version("asf-search")
        except importlib.metadata.PackageNotFoundError as exc:
            _error(
                EngineUnavailableError,
                "asf_search_unavailable",
                "asf-search has no discoverable distribution version",
            )
            raise AssertionError from exc
    parsed = _version(package_version)
    if parsed is None or not (ASF_SEARCH_MIN_MAJOR <= parsed[0] < ASF_SEARCH_MAX_MAJOR):
        _error(
            UnsupportedASFSearchVersionError,
            "unsupported_asf_search_version",
            f"asf-search {package_version!r} is outside {ASF_SEARCH_COMPATIBILITY}",
        )
    return package


def _geometry_wkt(spatial: Any) -> str:
    """Translate a typed FanInSAR geometry into the package's private WKT input."""
    from faninsar.remote import _query_geometry

    geometry, _kind, _points = _query_geometry(spatial)
    from shapely.wkt import dumps

    return dumps(geometry, rounding_precision=8)


_FILTER_NAMES = frozenset(
    {
        "platform",
        "processingLevel",
        "instrument",
        "beamMode",
        "polarization",
        "flightDirection",
        "lookDirection",
        "absoluteOrbit",
        "relativeOrbit",
        "asfFrame",
        "frame",
        "granule_list",
        "product_list",
        "groupID",
    }
)


def _query_parameters(
    collection: str,
    *,
    spatial: Any | None,
    datetime_range: tuple[datetime, datetime] | None,
    filters: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the small allowlisted parameter set accepted by ASF search."""
    unknown = set(filters).difference(_FILTER_NAMES)
    if unknown:
        _error(
            ASFSearchError,
            "unsupported_query_option",
            f"unsupported ASF search option(s): {sorted(unknown)!r}",
        )
    result: dict[str, Any] = {"collections": collection, **filters}
    if spatial is not None:
        result["intersectsWith"] = _geometry_wkt(spatial)
    if datetime_range is not None:
        if len(datetime_range) != 2 or datetime_range[1] < datetime_range[0]:
            _error(ASFSearchError, "invalid_datetime_range", "invalid ASF time range")
        result["start"], result["end"] = datetime_range
    return result


def _record_from_product(product: Any) -> dict[str, Any]:
    """Extract a plain mapping from one package product and nothing else."""
    if isinstance(product, Mapping):
        return dict(product)
    for name in ("umm", "properties"):
        value = getattr(product, name, None)
        if isinstance(value, Mapping):
            return dict(value)
    geojson = getattr(product, "geojson", None)
    if callable(geojson):
        value = geojson()
        if isinstance(value, Mapping):
            return dict(value)
    _error(
        ASFSearchError,
        "malformed_product",
        "asf-search returned a product without a mapping representation",
    )
    raise AssertionError  # pragma: no cover


def _with_asset_link(record: dict[str, Any], product: Any) -> dict[str, Any]:
    """Map ASF's URL property to a CMR data link when needed."""
    if "links" in record or "RelatedUrls" in record or "DataGranule" in record:
        return record
    nested = record.get("properties")
    nested_url = nested.get("url") if isinstance(nested, Mapping) else None
    url = (
        record.get("url")
        or record.get("download_url")
        or nested_url
        or getattr(product, "url", None)
    )
    if isinstance(url, str):
        record["links"] = [{"rel": "data", "href": url, "title": "SAFE ZIP"}]
    return record


def _response_bytes(response: Any, ledger: _CallLedger) -> None:
    """Charge one response body, requiring an observable byte boundary."""
    content = getattr(response, "content", None)
    if not isinstance(content, (bytes, bytearray)):
        content = getattr(response, "_content", None)
    if not isinstance(content, (bytes, bytearray)):
        _error(
            UnobservableASFSearchError,
            "unobservable_response",
            "asf-search response bytes cannot be metered",
        )
    ledger.begin_response()
    ledger.response_bytes(len(content))


def _response_hook(
    response: Any, *args: Any, ledger: _CallLedger, **kwargs: Any
) -> Any:
    """Meter a requests response hook and each redirect response."""
    del args, kwargs
    if getattr(response, "_faninsar_metered", False):
        return response
    response._faninsar_metered = True
    history = getattr(response, "history", ())
    for previous in history if isinstance(history, Iterable) else ():
        if getattr(previous, "_faninsar_metered", False):
            continue
        previous._faninsar_metered = True
        ledger.redirect()
        _response_bytes(previous, ledger)
    if getattr(response, "is_redirect", False):
        ledger.redirect()
    _response_bytes(response, ledger)
    return response


def _request_key(request: Any) -> tuple[Any, ...]:
    """Return a stable key for recognizing package-level repeated attempts."""
    headers = getattr(request, "headers", {})
    selected = tuple(sorted((str(k), str(v)) for k, v in headers.items()))
    return (
        getattr(request, "method", None),
        getattr(request, "url", None),
        selected,
        getattr(request, "body", None),
    )


def _instrument_requests_session(
    session: Any, ledger: _CallLedger
) -> Callable[[], None]:
    """Install request/response hooks on a real requests session."""
    import requests

    if not isinstance(session, requests.Session) or not isinstance(
        getattr(session, "adapters", None), Mapping
    ):
        raise TypeError
    originals = dict(session.adapters)
    retries = []
    for adapter in originals.values():
        retry_config = getattr(adapter, "max_retries", 0)
        total = getattr(retry_config, "total", retry_config)
        if total not in (0, False, None):
            retries.append(total)
    if retries:
        _error(
            UnobservableASFSearchError,
            "unobservable_retries",
            "asf-search session has unmetered transport retries",
        )
    previous_hooks = dict(getattr(session, "hooks", {}))
    session.hooks.setdefault("response", []).append(
        lambda response, *args, **kwargs: _response_hook(
            response, *args, ledger=ledger, **kwargs
        )
    )
    previous_keys: list[tuple[Any, ...]] = []

    def wrap(adapter: Any) -> Any:
        class LedgerAdapter(requests.adapters.HTTPAdapter):
            """Charge each low-level request before delegated transport."""

            def send(self, request: Any, **kwargs: Any) -> Any:
                key = _request_key(request)
                if key in previous_keys:
                    ledger.retry()
                previous_keys.append(key)
                ledger.check_elapsed()
                ledger.request()
                return adapter.send(request, **kwargs)

        return LedgerAdapter()

    for prefix, adapter in originals.items():
        session.mount(prefix, wrap(adapter))

    def restore() -> None:
        """Restore the caller's session hooks and adapters."""
        session.hooks = previous_hooks
        for prefix, adapter in originals.items():
            session.mount(prefix, adapter)

    return restore


def _instrument_generic_session(
    session: Any, ledger: _CallLedger
) -> Callable[[], None]:
    """Instrument a requests-like fixture session used by deterministic tests."""
    post = getattr(session, "post", None)
    if not callable(post):
        _error(
            UnobservableASFSearchError,
            "unobservable_session",
            "supplied ASF session does not expose a POST boundary",
        )
    previous_post = post
    previous_keys: list[tuple[Any, ...]] = []

    def metered_post(*args: Any, **kwargs: Any) -> Any:
        """Charge and execute one fixture POST."""
        key = (args, tuple(sorted((str(k), repr(v)) for k, v in kwargs.items())))
        if key in previous_keys:
            ledger.retry()
        previous_keys.append(key)
        ledger.check_elapsed()
        ledger.request()
        response = previous_post(*args, **kwargs)
        _response_hook(response, ledger=ledger)
        return response

    try:
        session.post = metered_post
    except (AttributeError, TypeError) as exc:
        _error(
            UnobservableASFSearchError,
            "unobservable_session",
            "supplied ASF session cannot be instrumented",
        )
        raise AssertionError from exc

    def restore() -> None:
        """Restore the fixture session method."""
        session.post = previous_post

    return restore


def _instrument_session(session: Any, ledger: _CallLedger) -> Callable[[], None]:
    """Choose a requests or deterministic fixture instrumentation strategy."""
    try:
        return _instrument_requests_session(session, ledger)
    except (ImportError, TypeError):
        return _instrument_generic_session(session, ledger)


@dataclass(slots=True)
class ASFSearchAdapter:
    """Explicit optional ASF discovery adapter.

    Parameters
    ----------
    collection : str
        Registered ASF collection identity.
    endpoint : str, default=DEFAULT_CMR_ENDPOINT
        CMR endpoint used by the package's ASF search implementation.
    package : object, optional
        Injected package module for deterministic tests.  Production code
        leaves this unset so importing ``asf_search`` remains lazy.
    session_factory : callable, optional
        Factory creating one operation-scoped requests-like session.
    session : object, optional
        One already-created operation-scoped session.  It is instrumented and
        closed after this adapter's operation; it must not be reused.
    spatial, datetime_range, filters : optional
        Typed query values translated to the package's allowlisted options.

    """

    collection: str
    endpoint: str = DEFAULT_CMR_ENDPOINT
    package: Any | None = None
    session_factory: Callable[[], Any] | None = None
    session: Any | None = None
    spatial: Any | None = None
    datetime_range: tuple[datetime, datetime] | None = None
    filters: Mapping[str, Any] = field(default_factory=dict)
    provider: str = "ASF"
    data_origins: tuple[str, ...] = DEFAULT_DATA_ORIGINS
    path_prefixes: tuple[str, ...] = ("/",)
    redirect_origins: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ("anonymous", "earthdata-asf")
    engine: str = "asf-search"

    def __post_init__(self) -> None:
        """Validate the explicit engine and register endpoint policy."""
        if self.engine != "asf-search":
            _error(
                ASFSearchError,
                "invalid_engine",
                "ASF adapter engine must be asf-search",
            )
        if not isinstance(self.collection, str) or not self.collection.strip():
            _error(ASFSearchError, "invalid_collection", "ASF collection is required")
        parsed = urllib.parse.urlsplit(self.endpoint)
        if parsed.scheme != "https" or not parsed.hostname:
            _error(ASFSearchError, "invalid_endpoint", "ASF endpoint must be HTTPS")
        if not self.redirect_origins:
            object.__setattr__(
                self,
                "redirect_origins",
                (f"https://{parsed.hostname}", *self.data_origins),
            )

    def _options(
        self, package: Any, session: Any, parameters: Mapping[str, Any]
    ) -> Any:
        """Construct package options with the operation session attached."""
        options_class = getattr(package, "ASFSearchOptions", None)
        if not callable(options_class):
            _error(
                EngineUnavailableError,
                "missing_search_options",
                "asf-search lacks ASFSearchOptions",
            )
        try:
            return options_class(session=session, **dict(parameters))
        except (KeyError, TypeError, ValueError):
            try:
                options = options_class(**dict(parameters))
                options.session = session
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                _error(ASFSearchError, "query_translation_failed", str(exc))
                raise AssertionError from exc
            else:
                return options

    def _pages(self, package: Any, options: Any) -> Iterable[Any]:
        """Invoke only ASF discovery APIs, preferring bounded page iteration."""
        generator = getattr(package, "search_generator", None)
        if callable(generator):
            return generator(opts=options)
        search = getattr(package, "search", None)
        if callable(search):
            return (search(opts=options),)
        _error(
            EngineUnavailableError,
            "missing_search_api",
            "asf-search exposes no discovery API",
        )
        raise AssertionError  # pragma: no cover

    def items(
        self,
        *,
        ledger: _CallLedger | None = None,
        spatial: Any | None = None,
        datetime_range: tuple[datetime, datetime] | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> Iterator[Mapping[str, Any]]:
        """Yield B-normalized records under one operation-scoped ledger."""
        ledger = ledger or _CallLedger(RemoteResourceBudget())
        package = _load_package(self.package)
        session: Any | None = None
        restore: Callable[[], None] | None = None
        try:
            if self.session_factory is None and self.session is None:
                try:
                    import requests

                    session = requests.Session()
                except ImportError as exc:
                    _error(
                        EngineUnavailableError,
                        "asf_search_unavailable",
                        "requests is required by asf-search",
                    )
                    raise AssertionError from exc
            elif self.session_factory is not None:
                session = self.session_factory()
            else:
                session = self.session
            restore = _instrument_session(session, ledger)
            query_filters = dict(self.filters)
            query_filters.update(filters or {})
            parameters = _query_parameters(
                self.collection,
                spatial=spatial if spatial is not None else self.spatial,
                datetime_range=datetime_range
                if datetime_range is not None
                else self.datetime_range,
                filters=query_filters,
            )
            parameters["maxResults"] = ledger.budget.max_items
            options = self._options(package, session, parameters)
            previous_timeout: Any = None
            internal = getattr(package, "INTERNAL", None)
            if internal is not None and hasattr(internal, "CMR_TIMEOUT"):
                previous_timeout = internal.CMR_TIMEOUT
                internal.CMR_TIMEOUT = min(
                    float(previous_timeout), ledger.budget.read_timeout_seconds
                )
            normalizer = CMRCollectionAdapter(
                provider=self.provider,
                collection=self.collection,
                endpoint=self.endpoint,
                data_origins=self.data_origins,
                data_path_prefixes=self.path_prefixes,
            )
            produced = 0
            for page in self._pages(package, options):
                complete = getattr(
                    page, "searchComplete", getattr(page, "search_complete", True)
                )
                if complete is False:
                    _error(
                        ASFSearchError,
                        "incomplete_results",
                        "asf-search returned an incomplete page",
                    )
                values = (
                    page
                    if isinstance(page, Iterable) and not isinstance(page, Mapping)
                    else (page,)
                )
                for product in values:
                    if produced >= ledger.budget.max_items:
                        return
                    record = _with_asset_link(_record_from_product(product), product)
                    if "collection" not in record and "GranuleUR" not in record:
                        record["collection"] = self.collection
                    if "provider" not in record and "Provider" not in record:
                        record["provider"] = self.provider
                    if isinstance(record.get("assets"), Mapping):
                        yield record
                    else:
                        yield normalizer._normalize(record)
                    produced += 1
        except (RemoteLimitError, ASFSearchError):
            raise
        except Exception as exc:
            _error(EngineUnavailableError, "asf_search_failed", str(exc))
        finally:
            if (
                "internal" in locals()
                and internal is not None
                and previous_timeout is not None
            ):
                internal.CMR_TIMEOUT = previous_timeout
            if restore is not None:
                restore()
            if session is not None and callable(getattr(session, "close", None)):
                session.close()


AsfSearchAdapter = ASFSearchAdapter
ASFSearchEngineAdapter = ASFSearchAdapter
ASFSearchUnavailableError = EngineUnavailableError
ASFSearchVersionError = UnsupportedASFSearchVersionError


def register_asf_search_catalog(
    name: str,
    *,
    collection: str,
    endpoint: str = DEFAULT_CMR_ENDPOINT,
    package: Any | None = None,
    session_factory: Callable[[], Any] | None = None,
    **kwargs: Any,
) -> ASFSearchAdapter:
    """Register an explicit optional ASF search catalog without importing it."""
    adapter = ASFSearchAdapter(
        collection=collection,
        endpoint=endpoint,
        package=package,
        session_factory=session_factory,
        **kwargs,
    )
    _register_adapter(name, adapter)
    return adapter


register_asf_catalog = register_asf_search_catalog

__all__ = [
    "ASF_SEARCH_CERTIFIED_VERSION",
    "ASF_SEARCH_COMPATIBILITY",
    "ASFSearchAdapter",
    "ASFSearchEngineAdapter",
    "ASFSearchError",
    "ASFSearchUnavailableError",
    "ASFSearchVersionError",
    "AsfSearchAdapter",
    "EngineUnavailableError",
    "UnobservableASFSearchError",
    "UnsupportedASFSearchVersionError",
    "register_asf_catalog",
    "register_asf_search_catalog",
]
