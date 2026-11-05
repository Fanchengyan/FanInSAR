"""Sentinel-1 acquisition→track metadata for the sel() interface (stage 5).

Resolves each catalog date's ``"<relOrbit><A|D>"`` track label, flight
direction, polarizations, and absolute-burst-id coverage — the per-date
axis selectors that :meth:`StackInterferogramCollection.sel` and
:meth:`StackSceneCollection.sel` consume, plus the polygon→track mapping
the Stack-level ``sel(track=)``/``sel(polygon=)`` rules need.

Track keys come from :func:`~faninsar.missions.s1.types.track_group_key`.
The resolver never hits the network and reads no measurement pixels:

1. **Path layout** — stage-2 path-SAFEs staged under
   ``tracks/<trackKey>/acquisitions`` carry their track in the path, and
   the assembled product id encodes ``<relOrbit><letter>`` in its
   absolute-orbit field, so most groups resolve without parsing XML.
2. **S1C-vs-S1A/B disambiguation** — S1C flies the same relative orbits
   as S1A but 12 days later in the cycle.  When a group's resolved letter
   is ambiguous (``"?"``), :func:`disambiguate_track_letters` borrows the
   direction of the *other* same-orbit group only when the catalog
   resolves exactly one direction for that orbit; a catalog mixing both
   directions keeps ``"?"`` rather than guessing.
3. **Annotation/orbit fallback** — when the path gives nothing, the
   orbit-state vectors of the first swath supply the direction (the mean
   radial→velocity cross-track sense) and the relative orbit stays
   ``None``; annotation parsing is the last resort because it reads the
   XML payloads.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from faninsar.logging import setup_logger
from faninsar.missions.s1.types import track_group_key, track_group_parts
from faninsar.stack.selection import polygon_track_assignment

if TYPE_CHECKING:
    from faninsar.missions.s1.types import S1Product

logger = setup_logger(__name__)

_TRACK_DIR_RE = re.compile(r"(?P<digits>\d{1,5})(?P<letter>[AD?]+)")
# The track field only follows the SAFE start/stop time pair in the
# path-SAFE product name (``_<start>_<stop>_<relOrbit><letter>_<take>_<uid>``).
# Anchoring on the two adjacent ``\d{8}T\d{6}`` fields keeps an ESA frame
# name's ``<absOrbit>_<dataTake>_<uid>`` tail (e.g. ``_009286_01274A_931A``)
# from being misread as ``<relOrbit><letter>`` — ``01274A`` is a data-take
# id, not a track.
_SAFE_TRACK_FIELD_RE = re.compile(
    r"_\d{8}T\d{6}_\d{8}T\d{6}_(?P<orbit>\d{5})(?P<letter>[AD?])_"
)


@dataclass(frozen=True, slots=True)
class TrackAcquisitionInfo:
    """Track-axis metadata of one logical acquisition date.

    Attributes
    ----------
    track_key : str or None
        ``"<relOrbit><A|D>"`` group label, or ``None`` when no stable
        orbit group could be resolved.
    direction : str or None
        ``"ASCENDING"``/``"DESCENDING"`` when known.
    relative_orbit : int or None
        Relative orbit (track) number when known.
    mission_id : str or None
        ``"S1A"``-style platform id of the first product.
    polarizations : tuple of str
        Union of channel values across the date's products.
    absolute_burst_ids : frozenset of int
        Union of known ESA absolute burst ids across the date's bursts;
        empty when annotation/catalog metadata carries none.

    """

    track_key: str | None = None
    direction: str | None = None
    relative_orbit: int | None = None
    mission_id: str | None = None
    polarizations: tuple[str, ...] = ()
    absolute_burst_ids: frozenset[int] = field(default_factory=frozenset)


def _track_key_from_path(path: Path) -> tuple[int | None, str | None]:
    """Return ``(relative_orbit, direction-letter)`` a staged path encodes.

    Stage-2 workspaces keep SAFEs under ``tracks/<key>/acquisitions``;
    the ``<key>`` directory and the path-SAFE product name's
    ``_<relOrbit><letter>_`` field both carry the track.  Either source
    may fill a field the other lacks.
    """
    orbit: int | None = None
    letter: str | None = None
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part != "tracks" or index + 1 >= len(parts):
            continue
        match = _TRACK_DIR_RE.fullmatch(parts[index + 1])
        if match is not None:
            orbit = int(match.group("digits"))
            letter = match.group("letter")
    name = path.name.upper()
    for match in _SAFE_TRACK_FIELD_RE.finditer(name):
        orbit = int(match.group("orbit"))
        candidate = match.group("letter")
        if candidate in {"A", "D"}:
            letter = candidate
    return orbit, letter


def _direction_from_orbit(product: S1Product) -> str | None:
    """Return the flight direction implied by one product's orbit vectors.

    Consecutive state vectors give the ground-track azimuth: the northern
    component of its horizontal projection is positive on ascending
    passes and negative on descending ones at non-polar latitudes.  The
    median across all vector pairs decides, so a single odd vector does
    not flip the answer.
    """
    for swath in product.swaths:
        vectors = swath.orbit.vectors
        if len(vectors) < 2:
            continue
        votes: list[float] = []
        for first, second in itertools.pairwise(vectors):
            p0 = np.asarray(first.position_m, dtype=np.float64)
            p1 = np.asarray(second.position_m, dtype=np.float64)
            # Northern displacement = d(position)·(z x p / |z x p|): the
            # local north direction at p0 in the ECEF frame.
            z_axis = np.array([0.0, 0.0, 1.0])
            east = np.cross(z_axis, p0)
            east_norm = float(np.linalg.norm(east))
            if east_norm <= 0.0:
                continue
            east = east / east_norm
            north = np.cross(p0 / float(np.linalg.norm(p0)), east)
            votes.append(float(np.dot(p1 - p0, north)))
        if votes:
            median = float(np.median(np.asarray(votes)))
            if median > 0.0:
                return "ASCENDING"
            if median < 0.0:
                return "DESCENDING"
    return None


def _open_products(
    paths: Sequence[Path],
    *,
    polarizations: tuple[str, ...] | None,
) -> list[S1Product]:
    """Open a date's SAFE inputs, tolerating unreadable members."""
    from faninsar.missions.s1.safe import open_acquisition_products

    try:
        return open_acquisition_products(paths, polarizations=polarizations)
    except Exception as error:  # pragma: no cover - defensive boundary
        logger.warning("could not open %s for track resolution: %s", paths[0], error)
        return []


def resolve_track_acquisition(
    paths: Sequence[Path],
    *,
    polarizations: tuple[str, ...] | None = None,
    open_products: bool = True,
) -> TrackAcquisitionInfo:
    """Resolve one date's track metadata from its SAFE inputs.

    Parameters
    ----------
    paths : sequence of pathlib.Path
        Source SAFE paths of one acquisition date.
    polarizations : tuple of str, optional
        Channel subset honored when products are opened (defaults to all
        present channels).
    open_products : bool, default=True
        When False, only path-encoded metadata is used and no SAFE
        annotation is parsed (``direction``/``polarizations``/
        ``absolute_burst_ids`` then stay empty).

    Returns
    -------
    TrackAcquisitionInfo
        Track key and per-date axis metadata; fields that cannot be
        resolved stay ``None``/empty rather than guessed.

    """
    paths = tuple(Path(path) for path in paths)
    orbit, letter = _track_key_from_path(paths[0]) if paths else (None, None)
    for path in paths[1:]:
        other_orbit, other_letter = _track_key_from_path(path)
        if orbit is None:
            orbit = other_orbit
        if letter not in {"A", "D"}:
            letter = other_letter

    products: list[S1Product] = []
    if open_products and paths:
        products = _open_products(paths, polarizations=polarizations)

    mission_id = next(
        (product.mission_id for product in products if product.mission_id), None
    )
    pols: list[str] = []
    absolute_ids: set[int] = set()
    for product in products:
        for pol in product.polarizations:
            if pol not in pols:
                pols.append(pol)
        for swath in product.swaths:
            for burst in swath.bursts:
                if burst.absolute_burst_id is not None:
                    absolute_ids.add(int(burst.absolute_burst_id))

    direction: str | None = None
    if letter in {"A", "D"}:
        direction = "ASCENDING" if letter == "A" else "DESCENDING"
    if direction is None and products:
        direction = _direction_from_orbit(products[0])
        if direction is not None:
            letter = "A" if direction == "ASCENDING" else "D"

    key = track_group_key(
        str(mission_id or ""),
        orbit,
        str(direction or letter or ""),
    )
    return TrackAcquisitionInfo(
        track_key=key,
        direction=direction,
        relative_orbit=orbit,
        mission_id=None if mission_id is None else str(mission_id),
        polarizations=tuple(pols),
        absolute_burst_ids=frozenset(absolute_ids),
    )


def decode_track_metadata_dates(
    values: Mapping[str, object],
) -> dict[str, TrackAcquisitionInfo]:
    """Rebuild ``TrackAcquisitionInfo`` objects from persisted ``extra``.

    ``stack.json``'s ``science_extra`` round-trips track metadata as plain
    dicts (``{"track_key": ..., "direction": ..., ...}``); both the base
    :meth:`Stack._track_metadata` and the Sentinel-1 override decode them
    back into the mission type so the sel() surface sees one uniform shape
    regardless of session age.
    """
    decoded: dict[str, TrackAcquisitionInfo] = {}
    for date_id, record in values.items():
        if isinstance(record, TrackAcquisitionInfo):
            decoded[str(date_id)] = record
            continue
        if not isinstance(record, Mapping):
            continue
        decoded[str(date_id)] = TrackAcquisitionInfo(
            track_key=(
                None if record.get("track_key") is None else str(record["track_key"])
            ),
            direction=(
                None if record.get("direction") is None else str(record["direction"])
            ),
            relative_orbit=(
                None
                if record.get("relative_orbit") is None
                else int(record["relative_orbit"])  # type: ignore[arg-type]
            ),
            mission_id=(
                None if record.get("mission_id") is None else str(record["mission_id"])
            ),
            polarizations=tuple(
                str(value) for value in (record.get("polarizations") or ())
            ),
            absolute_burst_ids=frozenset(
                int(value) for value in (record.get("absolute_burst_ids") or ())
            ),
        )
    return decoded


def disambiguate_track_letters(
    infos: Mapping[str, TrackAcquisitionInfo],
) -> dict[str, TrackAcquisitionInfo]:
    """Resolve ``?``-letter tracks against the catalog's S1C/S1A parity.

    An S1C path-SAFE whose own product name could only pin the relative
    orbit (letter ``"?"``) borrows the direction of the *other* same-orbit
    group when exactly one resolved direction exists, or keeps ``?``
    when the catalog mixes both — one date cannot sit on two directions
    of one relative orbit.

    Returns a new ``{date_id: TrackAcquisitionInfo}`` mapping; entries
    that need no change are returned as the same object.
    """
    result = dict(infos)
    by_orbit: dict[int, set[str]] = {}
    for info in infos.values():
        if info.relative_orbit is None or info.direction is None:
            continue
        by_orbit.setdefault(info.relative_orbit, set()).add(info.direction)
    for date_id, info in infos.items():
        if info.relative_orbit is None:
            continue
        _orbit, letter = track_group_parts(info.track_key or "")
        if letter and letter != "?":
            continue
        directions = by_orbit.get(info.relative_orbit, set())
        if len(directions) == 1:
            direction = next(iter(directions))
        elif len(directions) > 1:
            # Mixed-direction same-orbit catalogs keep ``?`` rather than
            # guess one direction for every date.
            continue
        else:
            continue
        key = track_group_key(
            str(info.mission_id or ""), info.relative_orbit, direction
        )
        result[date_id] = TrackAcquisitionInfo(
            track_key=key,
            direction=direction,
            relative_orbit=info.relative_orbit,
            mission_id=info.mission_id,
            polarizations=info.polarizations,
            absolute_burst_ids=info.absolute_burst_ids,
        )
    return result


def _footprint_geometry(
    footprint: Sequence[tuple[float, float]],
) -> Any:
    """Return a shapely ``Polygon`` for one burst footprint tuple."""
    from shapely.geometry import Polygon

    return Polygon([(float(lon), float(lat)) for lon, lat in footprint])


def select_dates(
    infos: Mapping[str, TrackAcquisitionInfo],
    *,
    track: str | None = None,
    direction: str | None = None,
    pol: str | Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return the sorted date ids matching the track-axis selectors.

    ``track`` accepts the full ``"88A"`` label or a bare relative-orbit
    number (``"88"`` selects every direction letter of that orbit).
    ``direction`` accepts ``"ASC"``/``"DESC"`` abbreviations and the full
    ``"ASCENDING"``/``"DESCENDING"`` words.  ``pol`` accepts one channel
    or a list; a date matches when every requested channel is present in
    that acquisition's products.
    """
    requested_track = str(track).strip().upper() if track is not None else None
    requested_direction = _normalize_direction(direction) if direction else None
    requested_pols: set[str] = set()
    if pol is not None:
        values = (pol,) if isinstance(pol, str) else tuple(pol)
        requested_pols = {str(value).strip().upper() for value in values}
        invalid = requested_pols - {"VV", "VH", "HH", "HV"}
        if invalid:
            message = (
                f"unsupported Sentinel-1 polarization(s): {sorted(invalid)}; "
                "allowed=['HH', 'HV', 'VH', 'VV']"
            )
            raise ValueError(message)

    matched: list[str] = []
    for date_id, info in infos.items():
        if requested_track is not None:
            orbit, _letter = track_group_parts(info.track_key or "")
            if requested_track.isdigit():
                if orbit != int(requested_track):
                    continue
            elif (info.track_key or "").upper() != requested_track:
                continue
        if requested_direction is not None and info.direction != requested_direction:
            continue
        if requested_pols and not requested_pols.issubset(set(info.polarizations)):
            continue
        matched.append(date_id)
    return tuple(sorted(matched))


def _normalize_direction(value: str) -> str:
    """Return ``"ASCENDING"``/``"DESCENDING"`` for a direction selector."""
    text = str(value or "").strip().upper()
    if text in {"A", "ASC", "ASCENDING"}:
        return "ASCENDING"
    if text in {"D", "DESC", "DESCENDING"}:
        return "DESCENDING"
    message = f"direction must be ASC/DESC (or ASCENDING/DESCENDING); got {value!r}"
    raise ValueError(message)


def normalize_direction(value: str) -> str:
    """Public wrapper around :func:`_normalize_direction` for sel()."""
    return _normalize_direction(value)


def burst_absolute_ids_for_unit_tags(
    tags: Sequence[str],
    absolute_ids: Mapping[tuple[str, int], int],
) -> frozenset[int]:
    """Map ``f{frame}_{swath}_b{index}`` unit tags to absolute burst ids.

    ``absolute_ids`` keys are ``(swath, burst_index)`` pairs of one
    acquisition; unit tags recover the same pair so a committed scene's
    units map back to absolute ids without reopening the source SAFE.
    """
    selected: set[int] = set()
    for tag in tags:
        parts = tag.split("_")
        if len(parts) != 3 or not parts[2].startswith("b"):
            continue
        swath = parts[1]
        try:
            index = int(parts[2][1:])
        except ValueError:
            continue
        value = absolute_ids.get((swath, index))
        if value is not None:
            selected.add(int(value))
    return frozenset(selected)


def unit_burst_map(products: Sequence[S1Product]) -> dict[tuple[str, int], int]:
    """Return ``{(swath, burst_index): absolute_burst_id}`` for products."""
    mapping: dict[tuple[str, int], int] = {}
    for product in products:
        for swath in product.swaths:
            for burst in swath.bursts:
                if burst.absolute_burst_id is not None:
                    mapping[(swath.swath, int(burst.index))] = int(
                        burst.absolute_burst_id
                    )
    return mapping


def footprints_for_date(
    paths: Sequence[Path],
    *,
    polarizations: tuple[str, ...] | None = None,
) -> tuple[Any, ...]:
    """Return burst-footprint geometries of one date's SAFE inputs.

    Each footprint is a shapely ``Polygon`` in EPSG:4326; bursts without
    parsed footprints are skipped.  The union (not the convex hull) is
    preserved per burst so polygon overlap is measured on real coverage.
    """
    footprints: list[Any] = []
    for product in _open_products(paths, polarizations=polarizations):
        for swath in product.swaths:
            footprints.extend(
                _footprint_geometry(burst.footprint)
                for burst in swath.bursts
                if burst.footprint
            )
    return tuple(footprints)


__all__ = [
    "TrackAcquisitionInfo",
    "burst_absolute_ids_for_unit_tags",
    "decode_track_metadata_dates",
    "disambiguate_track_letters",
    "footprints_for_date",
    "normalize_direction",
    "polygon_track_assignment",
    "resolve_track_acquisition",
    "select_dates",
    "unit_burst_map",
]
