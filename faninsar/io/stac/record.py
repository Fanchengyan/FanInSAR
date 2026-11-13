"""Normalized STAC records and public identity for committed products (P0089).

A committed product is a storage receipt: its ``item.json`` has a null
geometry, null datetime, and a basename-derived id. This module defines the
derived, immutable record a STAC catalog or API exposes for one admitted
product. Normalization itself lives in :mod:`faninsar.io.stac.normalize`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from datetime import datetime

__all__ = [
    "PROJECTION_EXTENSION",
    "NormalizationReason",
    "StacNormalizationError",
    "StacRecord",
    "TemporalCoverage",
    "public_item_id",
]

#: Projection extension schema URI emitted for true geographic Items.
PROJECTION_EXTENSION = "https://stac-extensions.github.io/projection/v2.0.0/schema.json"

_SLUG_LIMIT = 48
_SLUG_DIGEST_LENGTH = 8
_UNSAFE_SLUG = re.compile(r"[^a-z0-9]+")

NormalizationReason = Literal[
    "invalid-item",
    "unsupported-kind",
    "missing-temporal",
    "missing-source",
    "cyclic-inheritance",
    "duplicate-write-id",
    "unsafe-asset",
    "outside-work-dir",
]


class StacNormalizationError(ValueError):
    """Report why one committed product cannot become a public STAC Item.

    Parameters
    ----------
    reason : str
        Exclusion reason used by the index and refresh report, such as
        ``"invalid-item"``, ``"unsupported-kind"``, or ``"missing-temporal"``.
    detail : str
        Human-readable diagnostic naming the offending metadata or write id.

    Notes
    -----
    The exception is a :class:`ValueError` so callers that only reject an
    unsafe candidate can handle it like any other validation failure. Index
    code should report :attr:`reason` and :attr:`detail` separately rather
    than parse the message.

    """

    def __init__(self, reason: NormalizationReason, detail: str) -> None:
        """Record the exclusion reason and its diagnostic."""
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True, slots=True)
class TemporalCoverage:
    """Observation interval of one normalized Item.

    Parameters
    ----------
    start : datetime
        Inclusive UTC start of the observation interval.
    end : datetime
        Inclusive UTC end of the observation interval.
    precision : {"day", "timestamp"}
        ``"day"`` for the explicit UTC calendar-day convention; otherwise
        ``"timestamp"`` for persisted precise times.

    """

    start: datetime
    end: datetime
    precision: Literal["day", "timestamp"]


@dataclass(frozen=True, slots=True)
class StacRecord:
    """Immutable normalized view of one committed product.

    Parameters
    ----------
    public_id : str
        Workspace-unique public Item id including the source write id.
    write_id : str
        Current ``faninsar:write_id`` of the committed product.
    kind : str
        Canonical ``faninsar:kind`` value.
    family : str
        Logical family inferred from the kind, or inherited for ``geocoded``.
    source_item_path : pathlib.Path
        Absolute path of the canonical ``item.json`` inside ``work_dir``.
    source_relative_path : str
        POSIX path of the product directory relative to ``work_dir``.
    collection_id : str or None
        Actual Collection membership, when the index assigns one.
    temporal : TemporalCoverage
        Observation interval used for Item time fields and search.
    bbox : tuple of float or None
        WGS84 min/max longitude and latitude, only for a geographic Item.
    item : mapping
        Read-only normalized STAC Feature mapping.
    asset_paths : mapping
        Read-only mapping of asset key to its absolute registered path.

    """

    public_id: str
    write_id: str
    kind: str
    family: str
    source_item_path: Path
    source_relative_path: str
    collection_id: str | None
    temporal: TemporalCoverage
    bbox: tuple[float, float, float, float] | None
    item: Mapping[str, Any]
    asset_paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        """Freeze both mappings so a record cannot be mutated after creation."""
        object.__setattr__(self, "item", _freeze(self.item))
        object.__setattr__(
            self, "asset_paths", MappingProxyType(dict(self.asset_paths))
        )

    def as_item(self) -> dict[str, Any]:
        """Return a JSON-safe deep copy of the normalized Feature.

        Returns
        -------
        dict
            Mutable, strict-JSON copy detached from this record. Mutating it
            never changes the record, and serialization rejects NaN or
            infinity.

        """
        payload = _thaw(self.item)
        json.dumps(payload, allow_nan=False)
        return payload


def public_item_id(*, kind: str, source_relative_path: str, write_id: str) -> str:
    """Return the write-id-scoped public identity of one product.

    Parameters
    ----------
    kind : str
        Canonical product kind.
    source_relative_path : str
        POSIX product directory relative to ``work_dir``. The source
        directory itself, not a filename, defines the locator.
    write_id : str
        Current ``faninsar:write_id``; it is always the final component.

    Returns
    -------
    str
        ``f"{kind}-{slug}-{write_id}"``. The slug is lowercased, reduced to
        ``[a-z0-9-]``, and truncated with a short digest when longer than 48
        characters. An empty locator becomes ``root``.

    Notes
    -----
    The canonical raw Item id is never changed. Because ``write_id`` is last,
    a replacement yields a distinct identity and two equal basenames in
    different directories remain distinct.

    """
    directory = PurePosixPath(source_relative_path.replace("\\", "/")).as_posix()
    if directory in {"", "."}:
        return f"{kind}-root-{write_id}"
    slug = _UNSAFE_SLUG.sub("-", directory.lower()).strip("-")
    if not slug:
        return f"{kind}-root-{write_id}"
    if len(slug) > _SLUG_LIMIT:
        digest = hashlib.sha256(directory.encode("utf-8")).hexdigest()
        slug = f"{slug[:_SLUG_LIMIT].rstrip('-')}-{digest[:_SLUG_DIGEST_LENGTH]}"
    return f"{kind}-{slug}-{write_id}"


def _freeze(value: Any) -> Any:
    """Return a recursively read-only view of a JSON-like value."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Return a mutable deep copy of a frozen JSON-like value."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_thaw(item) for item in value]
    return value
