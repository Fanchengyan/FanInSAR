"""Mission-owned AUX_CAL discovery and download for the EAP correction.

AUX_CAL files are fetched only for acquisitions processed by IPF 002.36,
only after local SAFE inputs exist, and only from ``Stack.prepare``.  The
active SAR-MPC record with the latest validity start no later than the
sensing time (latest generation on ties) is downloaded once into the Stack
work directory; caller-supplied files remain a pin.  Each file is ~0.6 MB
and one file usually covers every date of a pre-2015 stack.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from faninsar.logging import reject, setup_logger
from faninsar.missions.s1.aux_cal import parse_aux_cal_name, select_aux_cal_bindings
from faninsar.remote.errors import RemoteAccessError
from faninsar.remote.providers.sar_mpc import (
    SAR_MPC_CATALOG,
    SarMpcHttpError,
    get_sar_mpc_adapter,
    sar_mpc_download_url,
)
from faninsar.remote.records import RemoteAsset, RemoteResourceBudget
from faninsar.stack.preparation import InputUnavailableError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

logger = setup_logger(__name__)

_MISSIONS = frozenset({"S1A", "S1B", "S1C", "S1D"})


def _cached_aux_cal_paths(cache_dir: Path) -> tuple[Path, ...]:
    """Return extracted AUX_CAL SAFEs present in ``cache_dir``."""
    if not cache_dir.is_dir():
        return ()
    return tuple(
        path
        for path in sorted(cache_dir.glob("S1?_AUX_CAL_*.SAFE"))
        if path.is_dir() and any((path / "data").glob("s1?-aux-cal.xml"))
    )


def _best_record(
    records: Sequence[Mapping[str, object]],
    *,
    mission_id: str,
    sensing: datetime,
) -> Mapping[str, object] | None:
    """Return the covering record: latest validity, then latest generation."""
    ranked = []
    for record in records:
        parsed = parse_aux_cal_name(str(record.get("product_name", "")))
        if parsed is None or parsed.mission_id != mission_id:
            continue
        if parsed.validity_start > sensing:
            continue
        ranked.append((parsed.validity_start, parsed.generation, record))
    if not ranked:
        return None
    return max(ranked, key=lambda item: (item[0], item[1]))[2]


def _asset(record: Mapping[str, object]) -> RemoteAsset:
    """Build the anonymous download asset of one SAR-MPC record."""
    name = str(record["product_name"])
    checksum = record.get("hash")
    size = record.get("size")
    return RemoteAsset(
        provider="sar-mpc",
        catalog=SAR_MPC_CATALOG,
        collection="AUX_CAL",
        item_id=name,
        key="aux_cal",
        href=sar_mpc_download_url(str(record["uuid"])),
        media_type="application/zip",
        size_bytes=int(size) if isinstance(size, int) else None,
        checksum=str(checksum) if checksum else None,
        version=name,
        representation="complete",
        auth_profile="anonymous",
    )


def _extract(archive_path: Path, cache_dir: Path, product_name: str) -> Path:
    """Extract one AUX_CAL zip into ``cache_dir`` and return its SAFE root."""
    root = cache_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            target = (root / member).resolve()
            if root not in target.parents and target != root:
                message = f"AUX_CAL archive member escapes cache: {member}"
                reject(message, ValueError, logger=logger)
        archive.extractall(root)
    safe_root = root / f"{product_name}.SAFE"
    if not any((safe_root / "data").glob("s1?-aux-cal.xml")):
        message = f"AUX_CAL archive {archive_path.name} has no data XML"
        reject(message, RuntimeError, logger=logger)
    return safe_root


def ensure_aux_cal_paths(
    *,
    acquisitions: Mapping[str, tuple[str, datetime]],
    cache_dir: str | Path,
    existing: Mapping[str, object] | Sequence[str | Path] | None = None,
    auto_fetch: bool = True,
    budget: RemoteResourceBudget | None = None,
) -> dict[str, Path]:
    """Return one AUX_CAL binding per acquisition, downloading missing files.

    Parameters
    ----------
    acquisitions : mapping
        ``date_id -> (mission_id, sensing_time)`` of every acquisition that
        needs the EAP correction.
    cache_dir : path-like
        Confined directory for downloaded AUX_CAL products.
    existing : mapping or sequence, optional
        Caller-supplied AUX_CAL files; a qualifying file is a pin.
    auto_fetch : bool, default True
        When false, only pins and the cache are consulted.
    budget : RemoteResourceBudget, optional
        Finite request budget.

    Returns
    -------
    dict
        Date-to-file bindings.

    Raises
    ------
    RuntimeError
        If a listing or download fails, or SAR-MPC has no covering record.
    ValueError
        If an acquisition is still uncovered with ``auto_fetch`` disabled.

    """
    cache = Path(cache_dir)
    pins = select_aux_cal_bindings(existing, acquisitions) if existing else {}
    missing = {key: value for key, value in acquisitions.items() if key not in pins}
    if missing:
        pins.update(select_aux_cal_bindings(_cached_aux_cal_paths(cache), missing))
        missing = {key: value for key, value in missing.items() if key not in pins}
    if not missing:
        return pins
    if not auto_fetch:
        message = (
            "IPF 002.36 acquisitions need AUX_CAL files and automatic download "
            f"is disabled; uncovered dates: {sorted(missing)}"
        )
        reject(message, InputUnavailableError, logger=logger)

    from faninsar import remote

    adapter = get_sar_mpc_adapter()
    budget = budget or RemoteResourceBudget()
    cache.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    for date_id, (mission, sensing) in sorted(missing.items()):
        if mission not in _MISSIONS:
            message = f"cannot fetch AUX_CAL for {date_id}: mission_id {mission!r}"
            reject(message, ValueError, logger=logger)
        try:
            records = adapter.list_records(
                product_type="AUX_CAL",
                mission_id=mission,
                validity_start_lte=sensing.strftime("%Y-%m-%dT%H:%M:%S"),
                budget=budget,
            )
        except (SarMpcHttpError, RemoteAccessError) as error:
            message = f"AUX_CAL listing failed for {date_id} from sar-mpc.eu"
            reject(message, InputUnavailableError, cause=error, logger=logger)
        record = _best_record(records, mission_id=mission, sensing=sensing)
        if record is None:
            message = f"SAR-MPC has no active {mission} AUX_CAL covering {date_id}"
            reject(message, InputUnavailableError, logger=logger)
        name = str(record["product_name"])
        if name not in downloaded:
            archive_path = cache / f"{name}.SAFE.zip"
            try:
                remote.download(_asset(record), archive_path, budget=budget)
            except (SarMpcHttpError, RemoteAccessError) as error:
                message = f"AUX_CAL download failed for {date_id}: {name}"
                reject(message, InputUnavailableError, cause=error, logger=logger)
            downloaded[name] = _extract(archive_path, cache, name)
            logger.info("Downloaded AUX_CAL %s for %s", name, date_id)
        pins.update(
            select_aux_cal_bindings(
                [downloaded[name]],
                {date_id: (mission, sensing)},
                require_coverage=True,
            )
        )
    return pins


__all__ = ["ensure_aux_cal_paths"]
