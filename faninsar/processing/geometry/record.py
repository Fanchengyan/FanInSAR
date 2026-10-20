"""Strict-JSON record of a Reference radar grid and its orbit.

The Reference scene product persists the merged radar grid exactly as its
units are placed (row 0 / column 0 of the scene grid) together with the
orbit trimmed to the frame span. :func:`radar_geometry_from_record` rebuilds
the :class:`~faninsar.processing.geometry.RadarGeometryModel` that geocoding
and radar mask projection use, so a reopened Stack needs no SAFE access.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.geometry.coordinates import RadarGrid

if TYPE_CHECKING:
    from collections.abc import Mapping

    from faninsar.processing.geometry.transforms import RadarGeometryModel

__all__ = [
    "ORBIT_TRIM_EXTRA_VECTORS",
    "ORBIT_TRIM_MARGIN_S",
    "radar_geometry_from_record",
    "radar_geometry_record",
    "radar_grid_from_record",
]

#: Orbit kept beyond the frame span: a time margin, then extra vectors.
ORBIT_TRIM_MARGIN_S = 60.0
ORBIT_TRIM_EXTRA_VECTORS = 2


def radar_geometry_record(
    grid: RadarGrid,
    orbit: OrbitMetadata,
    **extra: object,
) -> dict[str, object]:
    """Return the strict-JSON record of ``grid`` with a span-trimmed orbit.

    Parameters
    ----------
    grid : RadarGrid
        Merged Reference radar grid whose (0, 0) is scene row/column 0.
    orbit : OrbitMetadata
        Reference orbit; trimmed to the grid span plus
        :data:`ORBIT_TRIM_MARGIN_S` and :data:`ORBIT_TRIM_EXTRA_VECTORS`.
    **extra : object
        Additional JSON-compatible fields (for example per-swath offsets).

    Returns
    -------
    dict
        Record accepted by :func:`radar_geometry_from_record`.

    """
    start = grid.sensing_start
    stop = start + timedelta(seconds=grid.azimuth_time_interval_s * grid.shape[0])
    vectors = list(orbit.vectors)
    low = start - timedelta(seconds=ORBIT_TRIM_MARGIN_S)
    high = stop + timedelta(seconds=ORBIT_TRIM_MARGIN_S)
    inside = [i for i, vector in enumerate(vectors) if low <= vector.time <= high]
    first, last = (inside[0], inside[-1]) if inside else (0, len(vectors) - 1)
    first = max(0, first - ORBIT_TRIM_EXTRA_VECTORS)
    last = min(len(vectors) - 1, last + ORBIT_TRIM_EXTRA_VECTORS)
    return {
        "shape": [int(grid.shape[0]), int(grid.shape[1])],
        "sensing_start": start.isoformat(),
        "azimuth_time_interval_s": float(grid.azimuth_time_interval_s),
        "starting_slant_range_m": float(grid.starting_slant_range_m),
        "range_spacing_m": float(grid.range_spacing_m),
        "wavelength_m": float(grid.wavelength_m),
        "look_direction": str(grid.look_direction),
        "orbit": {
            "reference_frame": orbit.reference_frame,
            "source": orbit.source,
            "vectors": [
                {
                    "time": vector.time.isoformat(),
                    "position_m": [float(value) for value in vector.position_m],
                    "velocity_m_s": [float(value) for value in vector.velocity_m_s],
                }
                for vector in vectors[first : last + 1]
            ],
        },
        **extra,
    }


def radar_grid_from_record(record: Mapping[str, object]) -> RadarGrid:
    """Rebuild the persisted :class:`RadarGrid`.

    Raises
    ------
    InvalidProcessingStateError
        If the record is incomplete.

    """
    try:
        shape = record["shape"]
        return RadarGrid(
            shape=(int(shape[0]), int(shape[1])),  # type: ignore[index]
            starting_slant_range_m=float(record["starting_slant_range_m"]),  # type: ignore[arg-type]
            range_spacing_m=float(record["range_spacing_m"]),  # type: ignore[arg-type]
            sensing_start=datetime.fromisoformat(str(record["sensing_start"])),
            azimuth_time_interval_s=float(record["azimuth_time_interval_s"]),  # type: ignore[arg-type]
            wavelength_m=float(record["wavelength_m"]),  # type: ignore[arg-type]
            look_direction=str(record["look_direction"]),  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError, IndexError) as error:
        reject_invalid_state(f"persisted radar geometry record is incomplete: {error}")
        raise


def _orbit_from_record(record: Mapping[str, object]) -> OrbitMetadata:
    try:
        orbit = record["orbit"]
        return OrbitMetadata(
            reference_frame=str(orbit["reference_frame"]),  # type: ignore[index]
            source=str(orbit["source"]),  # type: ignore[index]
            vectors=tuple(
                OrbitStateVector(
                    time=datetime.fromisoformat(str(vector["time"])),
                    position_m=tuple(float(v) for v in vector["position_m"]),  # type: ignore[arg-type]
                    velocity_m_s=tuple(float(v) for v in vector["velocity_m_s"]),  # type: ignore[arg-type]
                )
                for vector in orbit["vectors"]  # type: ignore[index]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        reject_invalid_state(f"persisted radar orbit record is incomplete: {error}")
        raise


def radar_geometry_from_record(record: Mapping[str, object]) -> RadarGeometryModel:
    """Rebuild the Reference :class:`RadarGeometryModel` from its record.

    Parameters
    ----------
    record : mapping
        Output of :func:`radar_geometry_record`.

    Returns
    -------
    RadarGeometryModel
        Geometry whose (0, 0) is scene row/column 0.

    """
    from faninsar.processing.geometry.transforms import RadarGeometryModel

    return RadarGeometryModel.from_radar_grid(
        radar_grid_from_record(record), _orbit_from_record(record)
    )
