"""Anonymous SAR-MPC auxiliary-file catalog (Sentinel-1 AUX_CAL).

The ESA SAR Mission Performance Cluster publishes Sentinel-1 auxiliary
data files through a JSON REST API at ``https://sar-mpc.eu/api/v1/``.  This
adapter only lists records and GETs one zip; the download endpoint answers
with a 302 to a short-lived signed object-storage URL, which is followed
once and never persisted.  Which record covers an acquisition is decided by
the mission (:mod:`faninsar.missions.s1.aux_cal_fetch`).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from faninsar.logging import reject, setup_logger
from faninsar.remote.access import _response_is_complete, _validate_url
from faninsar.remote.catalog import _ADAPTERS, _REGISTRY_LOCK
from faninsar.remote.errors import RemoteAccessError, _fail
from faninsar.remote.fetch.complete import _FetchedPayload, _RedirectHandler
from faninsar.remote.protocols import _CallLedger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from faninsar.remote.records import RemoteAsset, RemoteResourceBudget

logger = setup_logger(__name__)

SAR_MPC_CATALOG = "sar-mpc"
SAR_MPC_ORIGIN = "https://sar-mpc.eu"
SAR_MPC_DELIVERY_ORIGIN = "https://obs.eu-nl.otc.t-systems.com"
_API_PATH = "/api/v1/"
_PRODUCT_TYPES = frozenset({"AUX_CAL"})
_MISSIONS = frozenset({"S1A", "S1B", "S1C", "S1D"})
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_MAX_PAGES = 20


class SarMpcHttpError(RemoteAccessError):
    """A SAR-MPC list or download request returned a non-success status."""

    def __init__(self, status: int, host: str, url: str) -> None:
        """Record the failing host, URL, and status code."""
        self.status = status
        self.host = host
        self.url = url
        super().__init__("http_error", f"HTTP {status} from {host}")


def sar_mpc_download_url(uuid: str) -> str:
    """Return the canonical SAR-MPC download URL for one record UUID.

    Parameters
    ----------
    uuid : str
        Lower-case record UUID from a SAR-MPC listing.

    Returns
    -------
    str
        ``https://sar-mpc.eu/download/<uuid>/``.

    Raises
    ------
    ValueError
        If ``uuid`` is not a canonical UUID.

    """
    text = str(uuid).strip().lower()
    if _UUID.fullmatch(text) is None:
        message = f"invalid SAR-MPC record uuid: {uuid!r}"
        reject(message, ValueError, logger=logger)
    return f"{SAR_MPC_ORIGIN}/download/{text}/"


@dataclass(slots=True)
class SarMpcAdapter:
    """Anonymous-only adapter for SAR-MPC auxiliary-file records."""

    provider: str = "sar-mpc"
    origins: tuple[str, ...] = (SAR_MPC_ORIGIN,)
    # ``/s1ece-archive`` is admitted only on the delivery redirect origin.
    path_prefixes: tuple[str, ...] = ("/download", "/api/v1", "/s1ece-archive")
    redirect_origins: tuple[str, ...] = (SAR_MPC_DELIVERY_ORIGIN,)
    redirect_path_prefixes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    profiles: tuple[str, ...] = ("anonymous",)
    _allow_anonymous_delivery: bool = False

    def items(self, **kwargs: Any) -> Iterable[Mapping[str, Any]]:
        """Spatial catalog search is not used for auxiliary-file discovery."""
        del kwargs
        return ()

    def fetch(
        self,
        asset: RemoteAsset,
        budget: RemoteResourceBudget,
        *,
        ledger: _CallLedger | None = None,
    ) -> bytes:
        """GET one auxiliary zip, following the signed delivery redirect."""
        operation = ledger or _CallLedger(budget)
        if asset.auth_profile != "anonymous":
            _fail(RemoteAccessError, "unknown_auth_profile")
        target = _validate_url(asset.href, self)
        payload = self._get(target, budget, operation, accept="application/zip")
        return _FetchedPayload(payload, asset.checksum)

    def list_records(
        self,
        *,
        product_type: str,
        mission_id: str,
        validity_start_lte: str,
        budget: RemoteResourceBudget,
        ledger: _CallLedger | None = None,
    ) -> list[dict[str, Any]]:
        """Return active records of one product type for one mission.

        Parameters
        ----------
        product_type : {"AUX_CAL"}
            SAR-MPC product type.
        mission_id : {"S1A", "S1B", "S1C", "S1D"}
            Sentinel-1 unit.
        validity_start_lte : str
            ISO timestamp; only records whose validity starts no later than
            this instant are returned.
        budget : RemoteResourceBudget
            Finite request budget.
        ledger : _CallLedger, optional
            Shared operation ledger.

        Returns
        -------
        list of dict
            Raw record mappings (``uuid``, ``product_name``, ``hash``,
            ``size``, ``validity_start``, ``creation_date``...).

        """
        if product_type not in _PRODUCT_TYPES:
            _fail(RemoteAccessError, "invalid_listing_prefix", product_type)
        mission = str(mission_id).upper()
        if mission not in _MISSIONS:
            _fail(RemoteAccessError, "invalid_listing_prefix", mission)
        operation = ledger or _CallLedger(budget)
        query = urllib.parse.urlencode(
            {
                "product_type": product_type,
                "sentinel1__mission": mission,
                "adf__active": "1",
                "validity_start__lte": validity_start_lte,
                "ordering": "-validity_start",
                "page_size": "200",
            }
        )
        url: str | None = f"{SAR_MPC_ORIGIN}{_API_PATH}?{query}"
        records: list[dict[str, Any]] = []
        for _page in range(_MAX_PAGES):
            if url is None:
                break
            payload = self._get(
                self._validate_listing_url(url),
                budget,
                operation,
                accept="application/json",
            )
            try:
                document = json.loads(payload)
            except ValueError as error:
                logger.exception("SAR-MPC listing is not JSON")
                _fail(RemoteAccessError, "invalid_listing", str(error))
            results = document.get("results") if isinstance(document, dict) else None
            if not isinstance(results, list):
                _fail(RemoteAccessError, "invalid_listing", "missing results")
            records.extend(item for item in results if isinstance(item, dict))
            next_url = document.get("next")
            url = str(next_url) if isinstance(next_url, str) and next_url else None
        return records

    def _validate_listing_url(self, url: str) -> str:
        """Admit only API listing queries on the registered origin."""
        parsed = urllib.parse.urlsplit(url)
        if (
            f"{parsed.scheme}://{parsed.netloc}" != SAR_MPC_ORIGIN
            or parsed.path != _API_PATH
        ):
            _fail(RemoteAccessError, "unregistered_endpoint")
        return url

    def _get(
        self,
        url: str,
        budget: RemoteResourceBudget,
        ledger: _CallLedger,
        *,
        accept: str,
    ) -> bytes:
        """GET one URL through the redirect policy and return its body."""
        ledger.request()
        ledger.begin_response()
        opener = urllib.request.build_opener(_RedirectHandler(self, budget, ledger))
        request = urllib.request.Request(
            url,
            headers={"Accept": accept, "Accept-Encoding": "identity"},
        )
        try:
            with opener.open(request, timeout=budget.read_timeout_seconds) as response:
                _response_is_complete(response)
                payload = response.read()
        except urllib.error.HTTPError as error:
            host = urllib.parse.urlsplit(url).hostname or ""
            raise SarMpcHttpError(int(error.code), host, url) from error
        except urllib.error.URLError as error:
            host = urllib.parse.urlsplit(url).hostname or ""
            logger.exception("SAR-MPC request failed")
            raise SarMpcHttpError(0, host, url) from error
        ledger.response_bytes(len(payload))
        return bytes(payload)


def get_sar_mpc_adapter() -> SarMpcAdapter:
    """Return the validated adapter registered for SAR-MPC."""
    ensure_sar_mpc_catalog()
    with _REGISTRY_LOCK:
        adapter = _ADAPTERS.get(SAR_MPC_CATALOG)
    if not isinstance(adapter, SarMpcAdapter):
        message = "catalog 'sar-mpc' has an incompatible registration"
        reject(message, TypeError, logger=logger)
    return adapter


def ensure_sar_mpc_catalog(name: str | None = None) -> str:
    """Register or reuse the dedicated anonymous SAR-MPC catalog."""
    catalog = name or SAR_MPC_CATALOG
    if catalog != SAR_MPC_CATALOG:
        message = f"SAR-MPC must use the stable registration name {SAR_MPC_CATALOG!r}"
        reject(message, ValueError, logger=logger)
    expected = SarMpcAdapter()
    identity_fields = ("provider", "origins", "path_prefixes", "redirect_origins")
    with _REGISTRY_LOCK:
        if catalog in _ADAPTERS:
            adapter = _ADAPTERS[catalog]
            if any(
                getattr(adapter, name_, None) != getattr(expected, name_)
                for name_ in identity_fields
            ):
                message = f"catalog {catalog!r} has an incompatible registration"
                reject(message, ValueError, logger=logger)
            return catalog
        _ADAPTERS[catalog] = expected
    return catalog


__all__ = [
    "SAR_MPC_CATALOG",
    "SAR_MPC_DELIVERY_ORIGIN",
    "SAR_MPC_ORIGIN",
    "SarMpcAdapter",
    "SarMpcHttpError",
    "ensure_sar_mpc_catalog",
    "get_sar_mpc_adapter",
    "sar_mpc_download_url",
]
