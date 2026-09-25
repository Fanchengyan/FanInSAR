"""Small mission-owned search composition seams (PROPOSAL-0049)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from datetime import datetime

    from faninsar.remote.records import CatalogItem


@dataclass(frozen=True, slots=True)
class SelectionRequest:
    """Finite request passed from a Stack search to mission selection.

    Parameters
    ----------
    limit : int
        Maximum number of normalized candidates to select.
    datetime_range : tuple of datetime, optional
        Already validated acquisition interval.
    flight_direction : {"ASCENDING", "DESCENDING"}, optional
        Common orbit direction filter.
    filters : mapping, optional
        Provider filter values after profile defaults have been applied.

    """

    limit: int = 100
    datetime_range: tuple[datetime, datetime] | None = None
    flight_direction: str | None = None
    filters: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Protect nested filter state from mutation."""
        if self.limit <= 0:
            message = "selection limit must be positive"
            raise ValueError(message)
        if self.flight_direction not in (None, "ASCENDING", "DESCENDING"):
            message = "invalid flight direction"
            raise ValueError(message)
        object.__setattr__(self, "filters", MappingProxyType(dict(self.filters)))


@dataclass(frozen=True, slots=True)
class MissionPreparedInputs:
    """Validated mission-owned inputs handed to shared Stack preparation.

    Parameters
    ----------
    selected_footprints : sequence, optional
        Mission-selected source footprints.
    orbit_paths_by_acquisition : mapping, optional
        Local orbit files keyed by acquisition identity.
    calibration_paths : sequence, optional
        Local calibration files required by the mission.
    provenance : mapping, optional
        Scrubbed, human-readable identities. Credentials and signed URLs are
        never accepted as part of this value.

    """

    selected_footprints: tuple[object, ...] = ()
    orbit_paths_by_acquisition: Mapping[str, Path] = field(
        default_factory=lambda: MappingProxyType({})
    )
    calibration_paths: tuple[Path, ...] = ()
    provenance: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Normalize paths and freeze all nested values."""
        object.__setattr__(self, "selected_footprints", tuple(self.selected_footprints))
        object.__setattr__(
            self,
            "orbit_paths_by_acquisition",
            MappingProxyType(
                {
                    str(key): Path(value)
                    for key, value in self.orbit_paths_by_acquisition.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "calibration_paths",
            tuple(Path(value) for value in self.calibration_paths),
        )
        scrubbed: dict[str, str] = {}
        for key, value in self.provenance.items():
            text_key = str(key)
            if any(
                secret in text_key.lower()
                for secret in ("token", "secret", "password", "credential", "signature")
            ):
                message = "mission provenance contains a credential-bearing key"
                raise ValueError(message)
            parsed = urlsplit(str(value))
            scrubbed[text_key] = (
                urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
                if parsed.scheme and parsed.netloc
                else str(value)
            )
        object.__setattr__(self, "provenance", MappingProxyType(scrubbed))


@dataclass(frozen=True, slots=True)
class MissionSearchProfile:
    """Mission-owned defaults and deterministic catalog normalization policy.

    ``normalizer`` and ``selector`` are intentionally private callables.  A
    mission can provide its provider-specific policy without putting mission
    branches in :class:`~faninsar.stack.session.Stack`.
    """

    mission: str
    catalog: str
    collections: tuple[str, ...] = ()
    default_filters: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    normalizer: Callable[[Sequence[CatalogItem]], list[CatalogItem]] | None = None
    selector: (
        Callable[[Sequence[CatalogItem], SelectionRequest], list[CatalogItem]] | None
    ) = None

    def __post_init__(self) -> None:
        """Normalize and freeze profile defaults."""
        if not self.mission or not self.catalog:
            message = "mission search profile requires mission and catalog"
            raise ValueError(message)
        object.__setattr__(self, "collections", tuple(self.collections))
        object.__setattr__(
            self, "default_filters", MappingProxyType(dict(self.default_filters))
        )

    def normalize(self, items: Sequence[CatalogItem]) -> list[CatalogItem]:
        """Normalize provider records using the mission policy."""
        if self.normalizer is None:
            return list(items)
        return list(self.normalizer(items))

    def select(
        self,
        items: Sequence[CatalogItem],
        request: SelectionRequest | None = None,
    ) -> list[CatalogItem]:
        """Select candidates using the mission policy and finite request."""
        request = request or SelectionRequest(limit=len(items) or 1)
        if self.selector is None:
            return list(items)[: request.limit]
        return list(self.selector(items, request))


def merge_profile_filters(
    defaults: Mapping[str, object], filters: Mapping[str, object] | None
) -> dict[str, object]:
    """Merge explicit filters over mission defaults without mutating either."""
    merged = dict(defaults)
    if filters is not None:
        merged.update(filters)
    return merged


__all__ = [
    "MissionPreparedInputs",
    "MissionSearchProfile",
    "SelectionRequest",
    "merge_profile_filters",
]
