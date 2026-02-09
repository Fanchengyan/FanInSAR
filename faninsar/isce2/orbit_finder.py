"""Orbit file finder for Sentinel-1 processing.

This module provides automatic orbit file finding for Sentinel-1 SAFE files.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

# Orbit file name pattern
# S1A_OPER_AUX_POEORB_OPOD_20240110T120000_V20240101T000000_20240102T000000.EOF
ORBIT_PATTERN = re.compile(
    r"^(?P<platform>S1[AB])_OPER_AUX_"
    r"(?P<type>POEORB|RESORB)_OPOD_"
    r"(?P<creation>\d{8}T\d{6})_"
    r"V(?P<start>\d{8}T\d{6})_"
    r"(?P<stop>\d{8}T\d{6})\.EOF$"
)


class OrbitFinder:
    """Orbit file finder for Sentinel-1 SAFE files.

    Automatically finds matching orbit files from orbit_dir.

    Parameters
    ----------
    orbit_dir : Path
        Orbit file directory.

    Examples
    --------
    >>> finder = OrbitFinder(Path("/data/orbits"))
    >>> safe_file = Path("S1A_IW_SLC__1SDV_20240101T120000_....SAFE")
    >>> orbit = finder.find_orbit(safe_file, orbit_type="precise")
    >>> print(orbit)
    /data/orbits/S1A_OPER_AUX_POEORB_OPOD_20240110T120000_V20240101T000000_20240102T000000.EOF

    """

    def __init__(self, orbit_dir: Path) -> None:
        """Initialize the orbit finder."""
        self.orbit_dir = orbit_dir
        self._orbit_cache: dict[str, Path | None] = {}

    @lru_cache(maxsize=128)
    def find_orbit(
        self,
        safe_file: Path,
        orbit_type: Literal["precise", "restituted", "auto"] = "auto",
    ) -> Path | None:
        """Find matching orbit file.

        Parameters
        ----------
        safe_file : Path
            SAFE file path.
        orbit_type : {"precise", "restituted", "auto"}
            Orbit type. "auto" automatically selects: prefer precise, fallback to restituted.

        Returns
        -------
        Path | None
            Matching orbit file path, None if not found.

        """
        # Extract SAFE file info
        platform, start_time = self._parse_safe_filename(safe_file)
        if platform is None or start_time is None:
            logger.warning("Could not parse SAFE filename: %s", safe_file)
            return None

        # Find orbit file
        if orbit_type == "auto":
            # Prefer precise
            orbit = self._find_orbit_by_type(platform, start_time, "precise")
            if orbit is None:
                # Fallback to restituted
                logger.info("Precise orbit not found, trying restituted...")
                orbit = self._find_orbit_by_type(platform, start_time, "restituted")
            return orbit
        else:
            return self._find_orbit_by_type(platform, start_time, orbit_type)

    def _parse_safe_filename(
        self, safe_file: Path
    ) -> tuple[str | None, datetime | None]:
        """Extract platform and time from SAFE filename.

        Parameters
        ----------
        safe_file : Path
            SAFE file path.

        Returns
        -------
        tuple[str | None, datetime | None]
            (platform, start_time) or (None, None) if parsing failed.

        """
        # S1A_IW_SLC__1SDV_20240101T120000_20240101T120030_...
        name = safe_file.name

        if name.startswith("S1A"):
            platform = "S1A"
        elif name.startswith("S1B"):
            platform = "S1B"
        else:
            return None, None

        # Extract start time
        parts = name.split("_")
        if len(parts) >= 6:
            time_str = parts[5]  # 20240101T120000
            try:
                start_time = datetime.strptime(time_str, "%Y%m%dT%H%M%S")
                return platform, start_time
            except ValueError:
                pass

        return None, None

    def _find_orbit_by_type(
        self,
        platform: str,
        acquisition_time: datetime,
        orbit_type: Literal["precise", "restituted"],
    ) -> Path | None:
        """Find orbit file by type.

        Parameters
        ----------
        platform : str
            Platform (S1A or S1B).
        acquisition_time : datetime
            Acquisition time.
        orbit_type : {"precise", "restituted"}
            Orbit type.

        Returns
        -------
        Path | None
            Orbit file path or None if not found.

        """
        type_code = "POEORB" if orbit_type == "precise" else "RESORB"

        # List all orbit files
        pattern = f"{platform}_OPER_AUX_{type_code}_*.EOF"
        orbit_files = list(self.orbit_dir.glob(pattern))

        # Find matching orbit file
        for orbit_file in orbit_files:
            match = ORBIT_PATTERN.match(orbit_file.name)
            if not match:
                continue

            # Parse orbit coverage time
            start_str = match.group("start")
            stop_str = match.group("stop")

            start_time = datetime.strptime(start_str, "%Y%m%dT%H%M%S")
            stop_time = datetime.strptime(stop_str, "%Y%m%dT%H%M%S")

            # Check if acquisition time is covered
            if start_time <= acquisition_time <= stop_time:
                logger.info("Found %s orbit: %s", orbit_type, orbit_file.name)
                return orbit_file

        logger.warning(
            "No %s orbit found for %s at %s",
            orbit_type,
            platform,
            acquisition_time,
        )
        return None

    def find_orbits_batch(
        self,
        safe_files: list[Path],
        orbit_type: Literal["precise", "restituted", "auto"] = "auto",
    ) -> dict[Path, Path | None]:
        """Find orbits for multiple SAFE files.

        Parameters
        ----------
        safe_files : list[Path]
            List of SAFE file paths.
        orbit_type : {"precise", "restituted", "auto"}
            Orbit type.

        Returns
        -------
        dict[Path, Path | None]
            Mapping from SAFE file to orbit file.

        """
        result = {}
        for safe_file in safe_files:
            result[safe_file] = self.find_orbit(safe_file, orbit_type)
        return result
