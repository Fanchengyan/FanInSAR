"""Sentinel-1 precise orbit ephemeris readers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from faninsar.logging import setup_logger
from faninsar.missions.s1.errors import reject_product
from faninsar.processing.contracts import OrbitMetadata, OrbitStateVector

logger = setup_logger(__name__)

__all__ = ["read_eof_orbit"]


def _utc_datetime(value: str) -> datetime:
    timestamp = value.removeprefix("UTC=")
    parsed = datetime.fromisoformat(timestamp)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def read_eof_orbit(path: str | Path) -> OrbitMetadata:
    """Read an ESA Sentinel-1 EOF precise orbit file.

    Parameters
    ----------
    path : str or pathlib.Path
        ESA ``AUX_POEORB`` or ``AUX_RESORB`` XML file.

    Returns
    -------
    OrbitMetadata
        Earth-fixed position and velocity state vectors.

    """
    orbit_path = Path(path)
    if not orbit_path.exists():
        reject_product(f"orbit file does not exist: {orbit_path}")
    root = ET.parse(orbit_path).getroot()
    vectors: list[OrbitStateVector] = []
    for state in root.findall(".//OSV"):
        try:
            vectors.append(
                OrbitStateVector(
                    time=_utc_datetime(state.findtext("UTC", "")),
                    position_m=tuple(
                        float(state.findtext(component, "nan"))
                        for component in ("X", "Y", "Z")
                    ),
                    velocity_m_s=tuple(
                        float(state.findtext(component, "nan"))
                        for component in ("VX", "VY", "VZ")
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            message = f"invalid orbit state vector in {orbit_path}"
            logger.exception(message)
            raise ValueError(message) from exc
    if not vectors:
        reject_product(f"orbit file contains no state vectors: {orbit_path}")
    return OrbitMetadata(
        reference_frame="EARTH_FIXED",
        source=str(orbit_path),
        vectors=tuple(vectors),
    )
