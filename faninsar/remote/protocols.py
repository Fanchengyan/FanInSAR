"""Private adapter and operation-meter protocols."""

from __future__ import annotations

import inspect
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING, Any, Protocol

from .errors import RemoteLimitError, RemoteQueryError, _fail

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

    from .records import RemoteAsset, RemoteResourceBudget


class _Adapter(Protocol):
    """Private catalog adapter protocol."""

    provider: str
    origins: tuple[str, ...]
    path_prefixes: tuple[str, ...]
    redirect_origins: tuple[str, ...]
    redirect_path_prefixes: Mapping[str, tuple[str, ...]]
    profiles: tuple[str, ...]

    def items(
        self,
        *,
        spatial: Any | None = None,
        spatial_kind: str | None = None,
        point_geometries: tuple[Any, ...] = (),
        datetime_range: tuple[datetime, datetime] | None = None,
        collections: tuple[str, ...] | None = None,
        filters: Mapping[str, Any] | None = None,
        auth_profile: str = "anonymous",
        limit: int = 100,
        budget: RemoteResourceBudget | None = None,
        ledger: _CallLedger | None = None,
    ) -> Iterable[Mapping[str, Any]]: ...

    def fetch(
        self,
        asset: RemoteAsset,
        budget: RemoteResourceBudget,
        *,
        ledger: _CallLedger | None = None,
    ) -> Any: ...


@dataclass(slots=True)
class _CallLedger:
    """Private meter for exactly one public remote operation.

    Adapters receive this object only for the duration of one ``search`` or
    ``download`` call.  Provider code must charge every request, retry,
    redirect, and response byte it performs through the corresponding
    methods.  The public API intentionally exposes neither this object nor a
    cross-call session.
    """

    budget: RemoteResourceBudget
    started: float = 0.0
    requests: int = 0
    retries: int = 0
    redirects: int = 0
    response_bytes_total: int = 0
    _lock: threading.Lock = dataclass_field(init=False, repr=False)
    _local: threading.local = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Capture the operation start before provider work begins."""
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._local = threading.local()

    def check_elapsed(self) -> None:
        """Enforce the operation-wide elapsed-time limit."""
        if time.monotonic() - self.started > self.budget.max_elapsed_seconds:
            _fail(RemoteLimitError, "max_elapsed_seconds")

    def request(self) -> None:
        """Charge one provider request."""
        with self._lock:
            self.requests += 1
            exceeded = self.requests > self.budget.max_requests
        if exceeded:
            _fail(RemoteLimitError, "max_requests")

    def retry(self) -> None:
        """Charge one retry before the next request attempt."""
        self.check_elapsed()
        with self._lock:
            self.retries += 1
            exceeded = self.retries > self.budget.max_retries
        if exceeded:
            _fail(RemoteLimitError, "max_retries")

    def poll_retry(self) -> None:
        """Charge a provider-poll retry without consuming ordinary retries.

        Provider generated products have their own bounded polling budget.
        Keeping that counter separate prevents a slow burst extractor from
        either inheriting the ordinary transfer retry limit or expanding it
        for every other remote transfer.
        """
        self.check_elapsed()
        with self._lock:
            self.retries += 1

    def redirect(self) -> None:
        """Charge one redirect followed by a provider request."""
        self.check_elapsed()
        with self._lock:
            self.redirects += 1
            exceeded = self.redirects > self.budget.max_redirects
        if exceeded:
            _fail(RemoteLimitError, "max_redirects")

    def begin_response(self) -> None:
        """Start accounting for one response body."""
        self._local.response_bytes = 0

    def response_bytes(self, count: int) -> None:
        """Charge bytes from the current response body.

        Parameters
        ----------
        count : int
            Number of newly consumed response bytes.

        """
        if count < 0:
            msg = "response byte count must be non-negative"
            raise ValueError(msg)
        response_bytes = getattr(self._local, "response_bytes", 0) + count
        self._local.response_bytes = response_bytes
        with self._lock:
            self.response_bytes_total += count
            operation_bytes = self.response_bytes_total
        if response_bytes > self.budget.max_response_bytes:
            _fail(RemoteLimitError, "max_response_bytes")
        if operation_bytes > self.budget.max_operation_bytes:
            _fail(RemoteLimitError, "max_operation_bytes")
        self.check_elapsed()


def _accepts_ledger(method: Any) -> bool:
    """Return whether an adapter method accepts the private ledger keyword."""
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == "ledger"
        for parameter in parameters
    )


def _adapter_items(
    adapter: _Adapter,
    *,
    spatial: Any,
    spatial_kind: str,
    point_geometries: tuple[Any, ...],
    datetime_range: tuple[datetime, datetime] | None,
    collections: tuple[str, ...] | None,
    filters: Mapping[str, Any] | None = None,
    auth_profile: str,
    limit: int,
    budget: RemoteResourceBudget,
    ledger: _CallLedger,
) -> Iterable[Mapping[str, Any]]:
    """Invoke an adapter with the normalized query it advertises.

    The registry is private, but adapters from the preceding remote MVP are
    intentionally kept source-compatible.  Keyword arguments are therefore
    limited to parameters accepted by the concrete method; an adapter that
    exposes ``**kwargs`` receives the complete query contract.
    """
    method = adapter.items
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_any = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    values: dict[str, Any] = {
        "spatial": spatial,
        "spatial_kind": spatial_kind,
        "point_geometries": point_geometries,
        "datetime_range": datetime_range,
        "collections": collections,
        "filters": filters or {},
        "auth_profile": auth_profile,
        "limit": limit,
        "budget": budget,
        "ledger": ledger,
    }
    kwargs = (
        values
        if accepts_any
        else {name: value for name, value in values.items() if name in parameters}
    )
    if "ledger" not in kwargs:
        # P0044 adapters predate the private ledger keyword.  Their one
        # catalog operation still receives a conservative request charge.
        ledger.request()
    return method(**kwargs)


def _adapter_query(
    adapter: _Adapter,
    *,
    datetime_range: tuple[datetime, datetime] | None,
    flight_direction: str | None,
    filters: Mapping[str, Any],
) -> dict[str, Any]:
    """Prepare common and provider-specific search arguments before I/O.

    Adapters that advertise ``prepare_query`` own provider vocabulary and
    profile-default merging.  Older adapters remain compatible for the
    empty advanced-query case, while unsupported advanced fields fail before
    a call ledger is created.
    """
    prepare = getattr(adapter, "prepare_query", None)
    if not callable(prepare):
        if filters or flight_direction is not None:
            _fail(
                RemoteQueryError,
                "unsupported_query_option",
                "catalog adapter cannot consume advanced search options",
            )
        return {"datetime_range": datetime_range, "filters": {}}
    try:
        prepared = prepare(
            datetime_range=datetime_range,
            flight_direction=flight_direction,
            filters=filters,
        )
    except RemoteQueryError:
        raise
    except (TypeError, ValueError) as exc:
        _fail(RemoteQueryError, "query_translation_failed", str(exc))
    if not isinstance(prepared, Mapping):
        _fail(
            RemoteQueryError,
            "query_translation_failed",
            "catalog adapter returned invalid prepared search arguments",
        )
    return dict(prepared)


__all__ = [
    "_Adapter",
    "_CallLedger",
    "_accepts_ledger",
    "_adapter_items",
    "_adapter_query",
]
