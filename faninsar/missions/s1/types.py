"""Immutable Sentinel-1 product types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from faninsar.core.orbit import OrbitMetadata
    from faninsar.missions.protocols import (
        DopplerCentroidPolynomial,
        ValidSampleMask,
    )


@dataclass(frozen=True, slots=True)
class S1Burst:
    """One TOPS burst within a sub-swath annotation.

    ``measurement_path`` is set when a merged one-burst extract keeps its
    own TIFF instead of sharing the parent swath measurement.

    ``absolute_burst_id`` is the mission-independent ESA burst identifier
    when it is known (e.g. from the ASF ``BURST_ID_ABSOLUTE`` catalog
    attribute).  SAFE annotation XML does not carry it, so annotation
    parsing leaves it ``None``; ``azimuth_anx_time_s`` and
    ``sensing_time`` remain the annotation-derived matching keys.

    Adding ``number`` does not add selection by original parent ordinal.

    Parameters
    ----------
    index : int
        Zero-based Python ordinal in the current local burst sequence. It may
        change when local bursts are appended or reordered. A singleton subset
        of original parent burst 8 has index 0.
    number : int or None
        Zero-based ordinal in the original parent SAFE sub-swath burstList,
        scoped by parent SAFE, sub-swath and polarization. The singleton above
        has number 8, which remains 8 when the local sequence changes. This is
        not absoluteBurstID or absolute_burst_id. None means the original
        ordinal is unknown; it must not be guessed from local order.

    """

    index: int
    azimuth_time: datetime
    sensing_time: datetime
    azimuth_anx_time_s: float
    byte_offset: int
    lines: int
    samples: int
    first_valid_sample: tuple[int, ...]
    last_valid_sample: tuple[int, ...]
    footprint: tuple[tuple[float, float], ...] | None = None
    measurement_path: str | None = None
    absolute_burst_id: int | None = None
    number: int | None = None

    @property
    def valid_samples(self) -> ValidSampleMask:
        """Return valid-sample windows as a processing contract object."""
        from faninsar.missions.protocols import ValidSampleMask

        return ValidSampleMask(
            first_valid_sample=self.first_valid_sample,
            last_valid_sample=self.last_valid_sample,
        )


@dataclass(frozen=True, slots=True)
class ElevationAngleRecord:
    """One annotation ``antennaPattern`` record: elevation angle vs. range time.

    Sentinel-1 annotation lists one record per burst.  The elevation angle
    of each slant-range time locates a range sample on the two-way
    elevation antenna pattern (used by the IPF 002.36 EAP phase correction).
    """

    azimuth_time: datetime
    slant_range_time_s: tuple[float, ...]
    elevation_angle_deg: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class S1Swath:
    """One IW/EW sub-swath with annotation-derived geometry metadata."""

    swath: Literal["IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5"]
    polarization: str
    annotation_path: str
    measurement_path: str
    lines: int
    samples: int
    lines_per_burst: int
    samples_per_burst: int
    range_pixel_spacing_m: float
    azimuth_pixel_spacing_m: float
    azimuth_time_interval_s: float
    slant_range_time_s: float
    range_sampling_rate_hz: float
    radar_frequency_hz: float
    sensing_start: datetime
    sensing_stop: datetime
    bursts: tuple[S1Burst, ...]
    orbit: OrbitMetadata
    doppler_centroid: tuple[DopplerCentroidPolynomial, ...]
    azimuth_fm_rate: tuple[tuple[datetime, float, tuple[float, ...]], ...]
    azimuth_steering_rate_rad_s: float
    ascending_node_time: datetime | None = None
    elevation_angle_records: tuple[ElevationAngleRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class S1SourcePhase:
    """Qualified source-load phase binding retained by persisted SLC products.

    ``input_phase`` names the materialized source convention. ``phase_state``
    records the load-time result before carrier removal or registration.
    Only the small AUX_CAL XML is hashed; SLC samples are never hashed here.
    """

    product_id: str
    ipf_version: str | None
    eap_applicable: bool
    aux_cal_id: str | None
    aux_cal_digest: str | None
    correction_version: str
    phase_state: Literal["raw", "eap_corrected"]
    input_phase: Literal["raw"] = "raw"


@dataclass(frozen=True, slots=True)
class S1Product:
    """SAFE or EOPF Sentinel-1 Level-1 SLC product handle.

    ``ipf_version`` is the Instrument Processing Facility version recorded
    in ``manifest.safe`` (for example ``"002.36"``), or ``None`` when the
    product carries no processing record.
    """

    product_id: str
    source_path: Path
    mission_id: str
    product_type: str
    mode: str
    absolute_orbit: int
    polarizations: tuple[str, ...]
    swaths: tuple[S1Swath, ...]
    ipf_version: str | None = None

    def find_swath(
        self,
        name: str,
        polarization: str | None = None,
    ) -> S1Swath | None:
        """Return one sub-swath, or ``None`` when this product does not carry it.

        ROI canvases and adjacent along-track frames may hold a subset of the
        IW/EW sub-swaths of a date.  Callers that walk a shared swath list
        should use this instead of :meth:`swath`, which rejects a missing name.
        """
        matches = [item for item in self.swaths if item.swath == name]
        if polarization is not None:
            matches = [item for item in matches if item.polarization == polarization]
        if not matches:
            return None
        return matches[0]

    def swath(
        self,
        name: str,
        polarization: str | None = None,
    ) -> S1Swath:
        """Return one sub-swath by name and optional polarization."""
        from .errors import reject_product

        found = self.find_swath(name, polarization)
        if found is not None:
            return found
        if polarization is not None and polarization not in self.polarizations:
            message = (
                f"unsupported polarization {polarization!r} in product "
                f"{self.source_path}; available={self.polarizations}"
            )
            raise ValueError(message)
        return reject_product(f"swath {name!r} not found in product {self.source_path}")


_MISSION_LETTERS = frozenset({"A", "B", "C", "D"})


def _mission_letter(mission_id: str) -> str:
    """Return the Sentinel-1 unit letter for ``"S1A"``- or ``"A"``-style ids."""
    text = str(mission_id or "").strip().upper().replace("-", "")
    if text.startswith("SENTINEL"):
        text = text[-1:]
    text = text.removeprefix("S1")
    return text if text in _MISSION_LETTERS else ""


def track_group_key(
    mission_id: str,
    relative_orbit: int | str | None,
    orbit_direction: str,
) -> str | None:
    """Return the compact track label ``"<rel-orbit><letter>"``.

    The label is a concise single-track grouping key of the form ``"88A"``
    or ``"88D"``: the relative-orbit number plus the pass-direction letter
    (``A`` = ascending, ``D`` = descending).  ``mission_id`` only supplies
    the fallback letter when ``orbit_direction`` is empty or unrecognized;
    within one direction the same relative orbit flown by S1A and S1C
    therefore shares one label, matching cross-mission stack grouping.

    Parameters
    ----------
    mission_id : str
        SAFE mission id; both ``"S1A"`` and ``"A"`` spellings are accepted.
    relative_orbit : int or str or None
        Relative orbit (track) number; strings are coerced to ``int``.
    orbit_direction : str
        ``"ASCENDING"``/``"DESCENDING"`` (case-insensitive, ``"A"``/``"D"``
        accepted); unknown or empty values fall back to the mission letter.

    Returns
    -------
    str or None
        ``"88A"``-style key, or ``None`` when ``relative_orbit`` is missing
        or unparseable and no stable key exists.

    """
    try:
        orbit = int(str(relative_orbit).strip())
    except (TypeError, ValueError):
        return None
    if orbit <= 0:
        return None
    direction = str(orbit_direction or "").strip().upper()
    if direction.startswith("A"):
        suffix = "A"
    elif direction.startswith("D"):
        suffix = "D"
    else:
        suffix = _mission_letter(mission_id) or "?"
    return f"{orbit}{suffix}"


def track_group_parts(key: str) -> tuple[int | None, str]:
    """Split a :func:`track_group_key` label back into ``(orbit, suffix)``.

    Returns ``(None, key)`` when ``key`` does not start with digits.
    """
    text = str(key or "").strip().upper()
    digits = ""
    for char in text:
        if not char.isdigit():
            break
        digits += char
    if not digits:
        return None, text
    return int(digits), text[len(digits) :]


__all__ = [
    "ElevationAngleRecord",
    "S1Burst",
    "S1Product",
    "S1Swath",
    "track_group_key",
    "track_group_parts",
]
