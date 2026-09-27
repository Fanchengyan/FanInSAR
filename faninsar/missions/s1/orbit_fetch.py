"""Mission-owned covering EOF discovery and download (PROPOSAL-0054).

Orbit fetch runs only after local SAFE files exist.  Covering POEORB then
RESORB objects are listed and GET from the anonymous s1-orbits S3 origin.
Caller-supplied covering files remain a pin.  A catalog miss leaves
SAFE-embedded restituted vectors selected; a failed download is an error.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

from faninsar.logging import setup_logger
from faninsar.missions.s1.search import select_orbit_bindings
from faninsar.remote.catalog import _ADAPTERS
from faninsar.remote.errors import RemoteAccessError
from faninsar.remote.protocols import _CallLedger
from faninsar.remote.providers.s1_orbits import (
    S1_ORBITS_CATALOG,
    S1_ORBITS_ORIGIN,
    S1OrbitsAdapter,
    S1OrbitsHttpError,
    ensure_s1_orbits_catalog,
)
from faninsar.remote.records import RemoteAsset, RemoteResourceBudget

logger = setup_logger(__name__)

_EOF_NAME = re.compile(
    r"^(?P<mission>S1[ABC])_OPER_AUX_(?P<family>POEORB|RESORB)_OPOD_"
    r"(?P<produced>\d{8}T\d{6})_V(?P<start>\d{8}T\d{6})_"
    r"(?P<stop>\d{8}T\d{6})\.EOF$",
    re.IGNORECASE,
)
_MISSIONS = frozenset({"S1A", "S1B", "S1C"})


class S1OrbitDownloadError(RuntimeError):
    """Raised when a covering EOF list or GET fails."""

    def __init__(
        self,
        *,
        dates: Sequence[str],
        host: str,
        status: int,
        message: str | None = None,
    ) -> None:
        """Record the acquisitions, host, and HTTP status of a failed download."""
        self.dates = tuple(str(date_id) for date_id in dates)
        self.host = host
        self.status = int(status)
        text = message or (
            "orbit download failed for "
            f"{', '.join(self.dates)} from {host} (HTTP {self.status})"
        )
        logger.error("%s", text)
        super().__init__(text)


@dataclass(frozen=True, slots=True)
class EofFilename:
    """Parsed ESA AUX_POEORB/AUX_RESORB basename and validity window."""

    mission_id: str
    family: Literal["POEORB", "RESORB"]
    produced: datetime
    valid_start: datetime
    valid_stop: datetime
    filename: str

    @property
    def key(self) -> str:
        """Return the s1-orbits object key for this file."""
        return f"AUX_{self.family}/{self.filename}"


def _compact_utc(value: str) -> datetime:
    """Parse an ESA compact timestamp as timezone-aware UTC."""
    return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)


def parse_eof_filename(value: str | Path) -> EofFilename | None:
    """Parse an EOF basename or AUX_*ORB object key.

    Parameters
    ----------
    value : str or pathlib.Path
        Object key, URL path, or basename. Parent-path and query components
        are ignored.

    Returns
    -------
    EofFilename or None
        Parsed mission, family, and validity window, or ``None`` when the
        name is not a covering POEORB/RESORB file.

    """
    raw = str(value).split("?", 1)[0]
    name = Path(urllib.parse.unquote(raw)).name
    match = _EOF_NAME.fullmatch(name)
    if match is None:
        return None
    try:
        produced = _compact_utc(match["produced"])
        start = _compact_utc(match["start"])
        stop = _compact_utc(match["stop"])
    except ValueError:
        return None
    if stop < start:
        return None
    family = match["family"].upper()
    if family not in {"POEORB", "RESORB"}:
        return None
    return EofFilename(
        mission_id=match["mission"].upper(),
        family=family,  # type: ignore[arg-type]
        produced=produced,
        valid_start=start,
        valid_stop=stop,
        filename=name,
    )


def select_covering_eof_keys(
    keys: Sequence[str],
    *,
    interval: tuple[datetime, datetime],
    mission_id: str,
) -> list[str]:
    """Rank covering s1-orbits keys for one acquisition.

    POEORB files are preferred over RESORB files.  Keys whose mission prefix
    or validity window does not cover the sensing interval are dropped.

    Parameters
    ----------
    keys : sequence of str
        Candidate object keys or basenames.
    interval : tuple of datetime
        Inclusive sensing start and stop.
    mission_id : str
        SAFE ``mission_id`` (``S1A``, ``S1B``, or ``S1C``).

    Returns
    -------
    list of str
        Covering keys in rank order, each ``AUX_{family}/{filename}``.

    """
    sensing_start, sensing_stop = interval
    midpoint = sensing_start + (sensing_stop - sensing_start) / 2
    wanted = mission_id.upper()
    ranked: list[tuple[int, float, str]] = []
    for key in keys:
        parsed = parse_eof_filename(key)
        if parsed is None or parsed.mission_id != wanted:
            continue
        if parsed.valid_start <= sensing_start and parsed.valid_stop >= sensing_stop:
            rank = 0 if parsed.family == "POEORB" else 1
            span = parsed.valid_stop - parsed.valid_start
            orbit_mid = parsed.valid_start + span / 2
            distance = abs((orbit_mid - midpoint).total_seconds())
            ranked.append((rank, distance, parsed.key))
    ranked.sort()
    return [item[2] for item in ranked]


def _orbit_destination(cache_dir: Path, key: str) -> Path:
    """Resolve one EOF destination under ``cache_dir`` from a basename-only key."""
    parsed = parse_eof_filename(key)
    if parsed is None:
        message = f"invalid S1 orbit object key: {key}"
        logger.error(message)
        raise ValueError(message)
    root = cache_dir.resolve()
    destination = (root / parsed.filename).resolve()
    if destination.parent != root:
        message = f"orbit destination escapes cache directory: {destination}"
        logger.error(message)
        raise ValueError(message)
    return destination


def _cached_eof_paths(cache_dir: Path) -> tuple[Path, ...]:
    """Return cached EOF files whose names match the covering grammar."""
    if not cache_dir.is_dir():
        return ()
    return tuple(
        path
        for path in sorted(cache_dir.iterdir())
        if path.is_file() and parse_eof_filename(path.name) is not None
    )


def _eof_asset(key: str) -> RemoteAsset:
    """Build an anonymous RemoteAsset for one s1-orbits EOF object."""
    parsed = parse_eof_filename(key)
    if parsed is None:
        message = f"invalid S1 orbit object key: {key}"
        logger.error(message)
        raise ValueError(message)
    href = f"{S1_ORBITS_ORIGIN}/{parsed.key}"
    return RemoteAsset(
        provider="s1-orbits",
        catalog=S1_ORBITS_CATALOG,
        collection=f"AUX_{parsed.family}",
        item_id=parsed.filename,
        key="eof",
        href=href,
        media_type="application/xml",
        version=parsed.filename,
        representation="complete",
        auth_profile="anonymous",
    )


def _raise_download_error(date_id: str, error: BaseException) -> None:
    """Convert a remote HTTP failure into a dated orbit download error."""
    if isinstance(error, S1OrbitsHttpError):
        raise S1OrbitDownloadError(
            dates=(date_id,),
            host=error.host,
            status=error.status,
        ) from error
    host = S1_ORBITS_ORIGIN.removeprefix("https://")
    status = int(getattr(error, "status", 0) or 0)
    raise S1OrbitDownloadError(
        dates=(date_id,),
        host=host,
        status=status,
    ) from error


def _adapter() -> S1OrbitsAdapter:
    """Return the registered anonymous s1-orbits adapter."""
    ensure_s1_orbits_catalog()
    adapter = _ADAPTERS[S1_ORBITS_CATALOG]
    if not isinstance(adapter, S1OrbitsAdapter):
        message = "catalog 's1-orbits' has an incompatible registration"
        logger.error(message)
        raise TypeError(message)
    return adapter


def ensure_covering_orbit_paths(
    *,
    intervals: Mapping[str, tuple[datetime, datetime]],
    mission_ids: Mapping[str, str],
    cache_dir: str | Path,
    existing: Mapping[str, Path] | Sequence[str | Path] | None = None,
    auto_fetch: bool = True,
    budget: RemoteResourceBudget | None = None,
) -> dict[str, Path]:
    """Return covering local EOF bindings, downloading only missing dates.

    Parameters
    ----------
    intervals : mapping
        Acquisition sensing windows keyed by Stack date ID.
    mission_ids : mapping
        SAFE ``mission_id`` for each date ID.
    cache_dir : path-like
        Confined directory for downloaded EOF files.
    existing : mapping or sequence, optional
        Caller-supplied local candidates. Covering files are a pin and are
        not upgraded to a later POEORB.
    auto_fetch : bool, default True
        When false, skip listing and GET (published-stack reuse).
    budget : RemoteResourceBudget, optional
        Finite list/GET budget inherited from P0044.

    Returns
    -------
    dict
        Date-to-path bindings. An empty mapping leaves SAFE-embedded vectors.

    Raises
    ------
    S1OrbitDownloadError
        If a covering list or object GET fails. A catalog miss is not an
        error; it leaves that date unbound.

    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    pins: dict[str, Path] = {}
    if existing is not None:
        pins = select_orbit_bindings(
            existing,
            intervals,
            mission_ids=mission_ids,
            require_coverage=False,
        )
    missing = [date_id for date_id in intervals if date_id not in pins]
    if not missing:
        return pins
    if not auto_fetch:
        return pins

    cached = _cached_eof_paths(cache)
    if cached:
        from_cache = select_orbit_bindings(
            cached,
            {date_id: intervals[date_id] for date_id in missing},
            mission_ids=mission_ids,
            require_coverage=False,
        )
        pins.update(from_cache)
        missing = [date_id for date_id in intervals if date_id not in pins]
    if not missing:
        return pins

    from faninsar import remote

    adapter = _adapter()
    budget = budget or RemoteResourceBudget()
    ledger = _CallLedger(budget)
    listings: dict[tuple[str, str], list[str]] = {}
    fetched: dict[str, Path] = {}
    for date_id in missing:
        mission = str(mission_ids.get(date_id, "")).upper()
        if mission not in _MISSIONS:
            logger.warning(
                "cannot auto-fetch S1 orbits for %s with mission_id %s",
                date_id,
                mission,
            )
            continue
        interval = intervals[date_id]
        covering: str | None = None
        for family in ("POEORB", "RESORB"):
            cache_key = (mission, family)
            if cache_key not in listings:
                prefix = f"AUX_{family}/{mission}_OPER_AUX_{family}_"
                try:
                    listings[cache_key] = adapter.list_object_keys(
                        prefix,
                        budget=budget,
                        ledger=ledger,
                    )
                except (S1OrbitsHttpError, RemoteAccessError) as error:
                    _raise_download_error(date_id, error)
            ranked = select_covering_eof_keys(
                listings[cache_key],
                interval=interval,
                mission_id=mission,
            )
            if ranked:
                covering = ranked[0]
                break
        if covering is None:
            logger.warning(
                "no covering POEORB/RESORB on s1-orbits for %s (%s); "
                "using SAFE-embedded restituted vectors",
                date_id,
                mission,
            )
            continue
        destination = _orbit_destination(cache, covering)
        overwrite = False
        if destination.is_file():
            bound = select_orbit_bindings(
                [destination],
                {date_id: interval},
                mission_ids={date_id: mission},
                require_coverage=False,
            )
            if date_id in bound:
                fetched[date_id] = destination
                continue
            overwrite = True
        asset = _eof_asset(covering)
        try:
            remote.download(
                asset,
                destination,
                overwrite=overwrite,
                budget=budget,
                ledger=ledger,
            )
        except (S1OrbitsHttpError, RemoteAccessError) as error:
            _raise_download_error(date_id, error)
        fetched[date_id] = destination
    pins.update(fetched)
    if not pins:
        return {}
    return select_orbit_bindings(
        pins,
        {date_id: intervals[date_id] for date_id in pins},
        mission_ids=mission_ids,
        require_coverage=False,
    )


__all__ = [
    "S1_ORBITS_CATALOG",
    "S1_ORBITS_ORIGIN",
    "EofFilename",
    "S1OrbitDownloadError",
    "ensure_covering_orbit_paths",
    "ensure_s1_orbits_catalog",
    "parse_eof_filename",
    "select_covering_eof_keys",
]
