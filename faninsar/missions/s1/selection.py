"""Sentinel-1 selection decoder: persisted mission metadata to facts.

Converts the Sentinel-1 mission dictionary persisted in ``stack.json``
(``science_extra``, i.e. :attr:`StackConfig.extra`) into the normalized
:class:`~faninsar.stack.selection.AcquisitionFacts` that shared Stack and
Network reader selection applies (PROPOSAL-0072 section 7).

Decoding is local-only: it reads the recorded ``track_metadata.dates``
records and never opens SAFE inputs or touches the network.

Normalization rules
-------------------
* ``acquisition_key`` is the recorded catalog date id.
* ``group_key`` is the recorded ``"<relOrbit><A|D>"`` track label, or
  ``None`` when the track was not resolved.
* Sentinel-1 channel availability is recorded per logical acquisition, so
  the acquisition key is the only scene unit with channel facts.
* ``frequency`` is inapplicable to Sentinel-1 and always ``None``.
* An empty recorded polarization list means *unresolved* (the products
  could not be opened), never "known empty": every Sentinel-1 product
  carries at least one channel.  It decodes to ``None`` (unknown).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from faninsar.logging import reject, setup_logger
from faninsar.missions.s1.track_map import normalize_direction
from faninsar.missions.s1.types import track_group_parts
from faninsar.stack.selection import AcquisitionFacts, ChannelKey, select_acquisitions

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = setup_logger(__name__)

_S1_POLARIZATIONS = frozenset({"VV", "VH", "HH", "HV"})


def _channels(
    date_id: str,
    values: object,
) -> tuple[ChannelKey, ...] | None:
    """Return normalized S1 channels of one record, or ``None`` if unknown."""
    if values is None:
        return None
    if isinstance(values, str) or not isinstance(values, (list, tuple, set)):
        message = (
            f"Sentinel-1 track metadata for {date_id!r} records polarizations "
            f"as {type(values).__name__}; expected a list of channel labels"
        )
        reject(message, ValueError, logger=logger)
    labels: list[str] = []
    for value in values:  # type: ignore[union-attr]
        label = str(value).strip().upper()
        if label not in _S1_POLARIZATIONS:
            message = (
                f"Sentinel-1 track metadata for {date_id!r} records unsupported "
                f"polarization {value!r}; allowed={sorted(_S1_POLARIZATIONS)}"
            )
            reject(message, ValueError, logger=logger)
        if label not in labels:
            labels.append(label)
    if not labels:
        return None
    return tuple(ChannelKey(frequency=None, polarization=label) for label in labels)


def _group_key(date_id: str, value: object) -> str | None:
    """Return the normalized track label of one record."""
    if value is None:
        return None
    if not isinstance(value, str):
        message = (
            f"Sentinel-1 track metadata for {date_id!r} records track_key "
            f"as {type(value).__name__}; expected a string"
        )
        reject(message, ValueError, logger=logger)
    text = value.strip().upper()
    return text or None


def _record_fields(record: object) -> tuple[object, object] | None:
    """Return ``(track_key, polarizations)`` of one persisted record."""
    if isinstance(record, Mapping):
        return record.get("track_key"), record.get("polarizations")
    # In-session caches may already hold the mission's typed record.
    if hasattr(record, "track_key") and hasattr(record, "polarizations"):
        return record.track_key, record.polarizations  # type: ignore[attr-defined]
    return None


def decode_s1_selection(
    mission: Mapping[str, object],
) -> tuple[AcquisitionFacts, ...]:
    """Decode a persisted Sentinel-1 mission dictionary into facts.

    Parameters
    ----------
    mission : Mapping[str, object]
        The persisted ``science_extra`` mapping of a Sentinel-1 Stack.  Only
        ``track_metadata.dates`` is read; other keys are ignored.

    Returns
    -------
    tuple of AcquisitionFacts
        One entry per recorded acquisition, sorted by acquisition key.
        Empty when no track metadata is recorded.

    Raises
    ------
    ValueError
        If the recorded metadata has an invalid shape or an unsupported
        channel label.

    Examples
    --------
    >>> facts = decode_s1_selection(
    ...     {
    ...         "track_metadata": {
    ...             "dates": {"20240101": {"track_key": "88A", "polarizations": ["VV"]}}
    ...         }
    ...     }
    ... )
    >>> facts[0].group_key, facts[0].unit_channels["20240101"][0].polarization
    ('88A', 'VV')

    """
    if not isinstance(mission, Mapping):
        message = (
            "Sentinel-1 selection decoding requires the persisted mission "
            f"mapping; got {type(mission).__name__}"
        )
        reject(message, TypeError, logger=logger)
    metadata = mission.get("track_metadata")
    if metadata is None:
        return ()
    if not isinstance(metadata, Mapping):
        message = "Sentinel-1 track_metadata must be a mapping with a 'dates' entry"
        reject(message, ValueError, logger=logger)
    dates = metadata.get("dates")
    if dates is None:
        return ()
    if not isinstance(dates, Mapping):
        message = "Sentinel-1 track_metadata.dates must map date ids to records"
        reject(message, ValueError, logger=logger)
    facts: list[AcquisitionFacts] = []
    for raw_date, record in _sorted_items(dates):
        date_id = str(raw_date)
        fields = _record_fields(record)
        if fields is None:
            message = (
                f"Sentinel-1 track metadata for {date_id!r} is "
                f"{type(record).__name__}; expected a mapping"
            )
            reject(message, ValueError, logger=logger)
        track_key, polarizations = fields
        facts.append(
            AcquisitionFacts(
                acquisition_key=date_id,
                group_key=_group_key(date_id, track_key),
                unit_channels={
                    unit: _channels(date_id, polarizations)
                    for unit in _record_units(date_id, record)
                },
            )
        )
    return tuple(facts)


def _sorted_items(values: Mapping[object, object]) -> Iterable[tuple[object, object]]:
    """Return mapping items in stable acquisition-key order."""
    return sorted(values.items(), key=lambda item: str(item[0]))


def _record_units(date_id: str, record: object) -> tuple[str, ...]:
    """Retain native absolute burst identities when recorded."""
    values = (
        record.get("absolute_burst_ids")
        if isinstance(record, Mapping)
        else getattr(record, "absolute_burst_ids", None)
    )
    return tuple(str(int(value)) for value in values) if values else (date_id,)


def select_s1_acquisitions(
    facts: tuple[AcquisitionFacts, ...],
    *,
    track: str | None = None,
    direction: str | None = None,
    pol: str | Sequence[str] | None = None,
    burst: int | Sequence[int] | None = None,
) -> tuple[str, ...]:
    """Normalize S1's existing selector aliases and apply shared matching.

    Numeric tracks match exactly their relative orbit, never a string prefix.
    Direction abbreviations and absolute burst ids are mission vocabulary;
    the shared matcher sees only facts, exact groups, units and channels.
    """
    candidates = facts
    if track is not None:
        text = str(track).strip().upper()
        if not text:
            message = "track must be a non-empty label"
            reject(message, ValueError, logger=logger)
        candidates = tuple(
            item
            for item in candidates
            if (
                track_group_parts(item.group_key or "")[0] == int(text)
                if text.isdigit()
                else item.group_key == text
            )
        )
    if direction is not None:
        letter = "A" if normalize_direction(direction) == "ASCENDING" else "D"
        candidates = tuple(
            item for item in candidates if (item.group_key or "").endswith(letter)
        )
    values = (pol,) if isinstance(pol, str) else pol
    if values is not None:
        values = tuple(str(value).strip().upper() for value in values)
        if not values or set(values) - _S1_POLARIZATIONS:
            message = f"unsupported Sentinel-1 polarization(s) in pol={pol!r}"
            reject(message, ValueError, logger=logger)
    units: tuple[str, ...] | None = None
    if burst is not None:
        requested = (burst,) if isinstance(burst, int) else burst
        units = tuple(str(int(value)) for value in requested)
        if not units:
            message = "burst requires at least one absolute burst id"
            reject(message, ValueError, logger=logger)
        known = {
            unit
            for item in facts
            for unit in item.unit_channels
            if unit != item.acquisition_key
        }
        if not known:
            message = "sel(burst=) requires absolute-burst metadata; none is available"
            reject(message, ValueError, logger=logger)
        if unknown := set(units) - known:
            message = (
                f"absolute burst id(s) not present in this Stack: {sorted(unknown)}"
            )
            reject(message, ValueError, logger=logger)
    return select_acquisitions(candidates, polarizations=values, units=units)


__all__ = ["decode_s1_selection", "select_s1_acquisitions"]
