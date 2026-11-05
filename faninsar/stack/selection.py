"""Normalized acquisition facts for mission-neutral Stack selection.

A mission's registered :class:`~faninsar.missions.base.Sensor` may declare a
``decode_selection`` hook that turns the mission dictionary persisted in
``stack.json`` (``science_extra``) into :class:`AcquisitionFacts`.  Shared
Stack and Network reader code applies these facts instead of parsing
mission-specific metadata (PROPOSAL-0072 §7).

Channel semantics
-----------------
``unit_channels`` maps each opaque scene-unit identifier to the channels
that unit carries:

* ``None`` — channel availability is unknown;
* ``()`` — availability is known and the unit carries no channel;
* an absent unit is not wildcard support.

An explicit channel request never matches unknown availability as "all
channels".
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeAlias

from faninsar.logging import reject, setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class ChannelKey:
    """One acquisition channel.

    Attributes
    ----------
    frequency : str or None
        Mission frequency identifier (e.g. NISAR ``"A"``/``"B"``).  ``None``
        where frequency is inapplicable (Sentinel-1) or unknown.
    polarization : str
        Upper-case polarization label such as ``"VV"`` or ``"HH"``.

    """

    frequency: str | None
    polarization: str


@dataclass(frozen=True, slots=True)
class AcquisitionFacts:
    """Normalized selection facts of one persisted acquisition.

    Attributes
    ----------
    acquisition_key : str
        Existing acquisition identifier, scoped by the containing workspace
        (the Stack catalog date id).
    group_key : str or None
        Mission grouping label (e.g. a Sentinel-1 track ``"88A"``); grouping
        is not acquisition identity.  ``None`` when unknown.
    unit_channels : Mapping[str, tuple[ChannelKey, ...] or None]
        Read-only map from opaque scene-unit identifier to the channels it
        carries; see the module docstring for ``None``/``()``/absent rules.

    """

    acquisition_key: str
    group_key: str | None
    unit_channels: Mapping[str, tuple[ChannelKey, ...] | None]

    def __post_init__(self) -> None:
        """Freeze ``unit_channels`` into a read-only mapping of tuples."""
        frozen: dict[str, tuple[ChannelKey, ...] | None] = {}
        for unit, channels in dict(self.unit_channels).items():
            if channels is None:
                frozen[str(unit)] = None
                continue
            values = tuple(channels)
            if not all(isinstance(value, ChannelKey) for value in values):
                message = (
                    f"acquisition {self.acquisition_key!r} unit {unit!r} channels "
                    "must be ChannelKey values"
                )
                reject(message, TypeError, logger=logger)
            frozen[str(unit)] = values
        object.__setattr__(self, "unit_channels", MappingProxyType(frozen))

    def carries(self, polarizations: Iterable[str | ChannelKey]) -> bool:
        """Return whether every known unit carries every requested channel.

        Parameters
        ----------
        polarizations : iterable of str
            Requested polarization labels.  A polarization-only request
            matches a channel of any frequency.

        Returns
        -------
        bool
            ``False`` when the acquisition records no unit, when any unit's
            availability is unknown (``None``), or when a requested
            polarization is missing from a unit.

        """
        requested = tuple(polarizations)
        if not self.unit_channels:
            return False
        for channels in self.unit_channels.values():
            if channels is None:
                return False
            for wanted in requested:
                if isinstance(wanted, ChannelKey):
                    if wanted not in channels:
                        return False
                elif not any(
                    channel.polarization == str(wanted).strip().upper()
                    for channel in channels
                ):
                    return False
        return True


SelectionDecoder: TypeAlias = Callable[
    [Mapping[str, object]], tuple[AcquisitionFacts, ...]
]


def select_acquisitions(
    facts: Iterable[AcquisitionFacts],
    *,
    group: str | None = None,
    polarizations: Iterable[str | ChannelKey] | None = None,
    units: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return sorted acquisition keys matching normalized selectors.

    Parameters
    ----------
    facts : iterable of AcquisitionFacts
        Decoded facts of one workspace.
    group : str, optional
        Exact group label; acquisitions with an unknown group never match.
    polarizations : iterable of str or ChannelKey, optional
        Channels every unit of a matching acquisition must carry; unknown
        availability never matches. A ChannelKey requires an exact frequency.
    units : iterable of str, optional
        Opaque unit identifiers that must all be present. Missing units never
        count as channel support.

    Returns
    -------
    tuple of str
        Sorted matching acquisition keys.

    """
    wanted_group = None if group is None else str(group).strip().upper()
    wanted_pols = None if polarizations is None else tuple(polarizations)
    wanted_units = None if units is None else frozenset(units)
    matched: list[str] = []
    for item in facts:
        if wanted_group is not None and (
            item.group_key is None or item.group_key.upper() != wanted_group
        ):
            continue
        if wanted_units is not None and not wanted_units.issubset(item.unit_channels):
            continue
        if wanted_pols is not None and not item.carries(wanted_pols):
            continue
        matched.append(item.acquisition_key)
    return tuple(sorted(matched))


def polygon_track_assignment(
    polygons: Sequence[Any],
    footprints_by_date: Mapping[str, Sequence[Any]],
    date_track: Mapping[str, str | None],
) -> dict[int, str]:
    """Assign each effective-ROI polygon to the track that covers it.

    A polygon is assigned to the track whose union footprint covers the
    largest share of the polygon's area; ties and empty overlaps resolve
    to the lexicographically smallest track so the map is deterministic.
    Polygons that intersect no footprint are assigned to the first
    lexicographic track — the caller's wrong-track error then still
    names a deterministic candidate.

    Parameters
    ----------
    polygons : sequence
        One shapely geometry per effective-ROI polygon, in
        ``_EffectiveRoi.parts`` order.
    footprints_by_date : mapping
        ``{date_id: (footprint geometry, ...)}`` — per-date unioned
        burst/swath footprints in EPSG:4326.
    date_track : mapping
        ``{date_id: track_key}`` of dates eligible for the comparison
        (``None`` values are skipped).

    Returns
    -------
    dict
        ``{polygon_index: track_key}`` covering every polygon index, or
        ``{}`` when no footprint resolves to a track (callers treat a
        missing index as "no track evidence" rather than a wrong-track
        failure).

    """
    track_footprints: dict[str, list[Any]] = {}
    for date_id, footprints in footprints_by_date.items():
        track = date_track.get(date_id)
        if track is None:
            continue
        target = track_footprints.setdefault(track, [])
        target.extend(footprints)
    tracks = sorted(track_footprints)
    assignment: dict[int, str] = {}
    if not tracks:
        return assignment
    for index, polygon in enumerate(polygons):
        best_track = tracks[0]
        best_area = -1.0
        for track in tracks:
            area = 0.0
            for footprint in track_footprints[track]:
                try:
                    if polygon.intersects(footprint):
                        area += float(polygon.intersection(footprint).area)
                except Exception:  # pragma: no cover - degenerate geometry
                    continue
            if area > best_area:
                best_area = area
                best_track = track
        assignment[index] = best_track
    return assignment


__all__ = [
    "AcquisitionFacts",
    "ChannelKey",
    "SelectionDecoder",
    "select_acquisitions",
]
