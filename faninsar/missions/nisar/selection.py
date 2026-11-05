"""NISAR selection decoder: persisted mission metadata to facts.

Converts the NISAR mission dictionary that :meth:`NISARStack.from_rslc`
persists in ``stack.json`` (``science_extra``: ``frequency``,
``polarization`` and ``source_lineage``) into normalized
:class:`~faninsar.stack.selection.AcquisitionFacts` (PROPOSAL-0072 §7).

Normalization rules
-------------------
* ``acquisition_key`` is each ``source_lineage`` date id; NISARStack admits
  one RSLC per acquisition, so that id is also the only scene unit.
* ``group_key`` is ``None``: NISAR records no grouping label.
* The admitted channel keeps its actual frequency identifier.  A missing
  frequency or polarization means *unknown* channel availability (``None``),
  never an unqualified ``HH`` default.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from faninsar.logging import reject, setup_logger

if TYPE_CHECKING:
    from faninsar.stack.selection import AcquisitionFacts

logger = setup_logger(__name__)


def _identifier(value: object, *, field: str) -> str | None:
    """Return an upper-case NISAR channel identifier, ``None`` if absent."""
    if value is None:
        return None
    if not isinstance(value, str):
        message = (
            f"NISAR persisted {field} must be a string; got {type(value).__name__}"
        )
        reject(message, ValueError, logger=logger)
    text = value.strip().upper()
    return text or None


def decode_nisar_selection(
    mission: Mapping[str, object],
) -> tuple[AcquisitionFacts, ...]:
    """Decode a persisted NISAR mission dictionary into facts.

    Parameters
    ----------
    mission : Mapping[str, object]
        The persisted ``science_extra`` mapping of a NISAR Stack.

    Returns
    -------
    tuple of AcquisitionFacts
        One entry per ``source_lineage`` acquisition, sorted by key; empty
        when no lineage is recorded.

    Raises
    ------
    ValueError
        If the recorded channel identifiers or lineage have invalid shapes.

    Examples
    --------
    >>> facts = decode_nisar_selection(
    ...     {
    ...         "frequency": "A",
    ...         "polarization": "HH",
    ...         "source_lineage": {"20250101": "a.h5"},
    ...     }
    ... )
    >>> facts[0].unit_channels["20250101"][0].frequency
    'A'

    """
    from faninsar.stack.selection import AcquisitionFacts, ChannelKey

    if not isinstance(mission, Mapping):
        message = (
            "NISAR selection decoding requires the persisted mission mapping; "
            f"got {type(mission).__name__}"
        )
        reject(message, TypeError, logger=logger)
    lineage = mission.get("source_lineage")
    if lineage is None:
        return ()
    if not isinstance(lineage, Mapping):
        message = "NISAR source_lineage must map acquisition date ids to sources"
        reject(message, ValueError, logger=logger)
    frequency = _identifier(mission.get("frequency"), field="frequency")
    polarization = _identifier(mission.get("polarization"), field="polarization")
    channels: tuple[ChannelKey, ...] | None = (
        None
        if frequency is None or polarization is None
        else (ChannelKey(frequency=frequency, polarization=polarization),)
    )
    return tuple(
        AcquisitionFacts(
            acquisition_key=date_id,
            group_key=None,
            unit_channels={date_id: channels},
        )
        for date_id in sorted(str(key) for key in lineage)
    )


__all__ = ["decode_nisar_selection"]
