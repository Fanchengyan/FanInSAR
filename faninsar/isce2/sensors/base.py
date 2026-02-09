"""Base sensor class for ISCE2 processing.

This module provides the abstract base class for SAR sensor implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from faninsar._core.sar import Acquisition
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.query import BoundingBox

logger = setup_logger(__name__)


@dataclass
class SafeFileInfo:
    """Information about a SAFE file.

    Parameters
    ----------
    path : Path
        Path to the SAFE file.
    date : str
        Acquisition date in YYYYMMDD format.
    start_time : datetime
        Start time of acquisition.
    stop_time : datetime
        Stop time of acquisition.
    platform : str
        Satellite platform (e.g., 'S1A', 'S1B').
    orbit_file : Path | None
        Path to the orbit file.

    """

    path: Path
    date: str
    start_time: datetime
    stop_time: datetime
    platform: str
    orbit_file: Path | None = None


class BaseSensor(ABC):
    """Abstract base class for SAR sensor implementations.

    This class defines the interface for sensor-specific operations
    in ISCE2 processing workflows.

    Parameters
    ----------
    slc_dir : Path
        Directory containing SLC/SAFE files.
    orbit_dir : Path | None, optional
        Directory containing orbit files. Default is None.
    aux_dir : Path | None, optional
        Directory containing auxiliary files. Default is None.

    Attributes
    ----------
    slc_dir : Path
        SLC directory.
    orbit_dir : Path | None
        Orbit directory.
    aux_dir : Path | None
        Auxiliary directory.
    name : str
        Sensor name.

    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the sensor name."""
        ...

    def __init__(
        self,
        slc_dir: Path,
        orbit_dir: Path | None = None,
        aux_dir: Path | None = None,
    ) -> None:
        """Initialize the BaseSensor."""
        self.slc_dir = Path(slc_dir)
        self.orbit_dir = Path(orbit_dir) if orbit_dir else None
        self.aux_dir = Path(aux_dir) if aux_dir else None

    @abstractmethod
    def find_safe_files(self) -> list[Path]:
        """Find all SAFE/SLC files in the directory.

        Returns
        -------
        list[Path]
            List of SAFE file paths.

        """
        ...

    @abstractmethod
    def parse_safe_file(self, safe_file: Path) -> SafeFileInfo:
        """Parse a SAFE file to extract metadata.

        Parameters
        ----------
        safe_file : Path
            Path to the SAFE file.

        Returns
        -------
        SafeFileInfo
            Parsed SAFE file information.

        """
        ...

    def discover_acquisitions(
        self,
        bbox: BoundingBox | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        exclude_dates: list[str] | None = None,
        include_dates: list[str] | None = None,
    ) -> Acquisition:
        """Discover acquisition dates from SAFE files.

        Parameters
        ----------
        bbox : BoundingBox | None, optional
            Spatial extent filter. Default is None.
        start_date : str | None, optional
            Start date filter (YYYY-MM-DD). Default is None.
        end_date : str | None, optional
            End date filter (YYYY-MM-DD). Default is None.
        exclude_dates : list[str] | None, optional
            Dates to exclude. Default is None.
        include_dates : list[str] | None, optional
            Dates to force include. Default is None.

        Returns
        -------
        Acquisition
            Collection of discovered acquisition dates.

        """
        import pandas as pd

        exclude_set = set(exclude_dates or [])
        include_set = set(include_dates or [])

        # Parse date range
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
        else:
            start_dt = datetime(1958, 1, 31, tzinfo=UTC)  # JPL satellite launch date

        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
        else:
            end_dt = datetime(2158, 1, 31, tzinfo=UTC)  # Far future

        # Find and parse SAFE files
        safe_files = self.find_safe_files()
        dates: list[datetime] = []

        for safe_file in safe_files:
            try:
                info = self.parse_safe_file(safe_file)
            except (ValueError, KeyError) as e:
                logger.warning("Failed to parse %s: %s", safe_file, e)
                continue

            # Check date range
            if info.start_time < start_dt or info.start_time > end_dt:
                continue

            # Check exclude list
            if info.date in exclude_set:
                continue

            # Check bbox if provided
            if bbox is not None and not self._check_bbox_overlap(safe_file, bbox):
                logger.debug("SAFE file %s does not overlap with bbox", safe_file)
                continue

            dates.append(info.start_time)

        # Add forced include dates
        for date_str in include_set:
            try:
                dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=UTC)
                if dt not in dates:
                    dates.append(dt)
            except ValueError:
                logger.warning("Invalid date format: %s", date_str)

        # Sort and create Acquisition
        dates.sort()
        return Acquisition(pd.to_datetime(dates)) if dates else Acquisition([])

    def _check_bbox_overlap(self, _safe_file: Path, _bbox: BoundingBox) -> bool:
        """Check if SAFE file overlaps with bounding box.

        Parameters
        ----------
        _safe_file : Path
            Path to the SAFE file.
        _bbox : BoundingBox
            Bounding box to check.

        Returns
        -------
        bool
            True if overlaps, False otherwise.

        """
        # Default implementation - subclasses can override
        # For now, assume all files overlap
        return True

    @abstractmethod
    def get_orbit_file(self, safe_file: Path) -> Path | None:
        """Find the orbit file for a SAFE file.

        Parameters
        ----------
        safe_file : Path
            Path to the SAFE file.

        Returns
        -------
        Path | None
            Path to the orbit file, or None if not found.

        """
        ...
