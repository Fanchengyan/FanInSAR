"""Sentinel-1 sensor implementation for ISCE2 processing.

This module provides the Sentinel-1 specific sensor implementation.
"""

from __future__ import annotations

import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from xml.etree import ElementTree as ET

from faninsar.isce2.sensors.base import BaseSensor, SafeFileInfo
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.query import BoundingBox

logger = setup_logger(__name__)

# Sentinel-1 SAFE file name pattern
# S1A_IW_SLC__1SDV_20240101T120000_20240101T120030_000001_000000_1234.SAFE
SAFE_PATTERN = re.compile(
    r"^(?P<platform>S1[AB])_"
    r"(?P<mode>IW)_"
    r"(?P<product>SLC)__"
    r"(?P<resolution>1[SD])"
    r"(?P<pol>[HV]{2})_"
    r"(?P<start>\d{8}T\d{6})_"
    r"(?P<stop>\d{8}T\d{6})_"
    r"(?P<orbit>\d{6})_"
    r"(?P<take_id>\d{6})_"
    r"(?P<uuid>[0-9A-F]{4}).*$"
)


class Sentinel1Sensor(BaseSensor):
    """Sentinel-1 sensor implementation.

    This class provides Sentinel-1 specific functionality for
    ISCE2 processing workflows, including automatic orbit file discovery.

    Parameters
    ----------
    slc_dir : Path
        Directory containing Sentinel-1 SAFE files.
    orbit_dir : Path | None, optional
        Directory containing orbit files. If provided, enables automatic
        orbit file discovery. Default is None.
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

    Examples
    --------
    >>> sensor = Sentinel1Sensor(
    ...     slc_dir=Path("/data/SLC"),
    ...     orbit_dir=Path("/data/orbits"),
    ... )
    >>> acquisitions = sensor.discover_acquisitions(
    ...     start_date="2020-01-01",
    ...     end_date="2020-12-31",
    ... )
    >>> # Find orbit files automatically
    >>> orbit_map = sensor.find_orbit_files(orbit_type="auto")

    """

    @property
    def name(self) -> str:
        """Get the sensor name."""
        return "Sentinel-1"

    def find_safe_files(self) -> list[Path]:
        """Find all Sentinel-1 SAFE files in the directory.

        Returns
        -------
        list[Path]
            List of SAFE file paths (both .SAFE and .zip).

        """
        safe_files: list[Path] = []

        # Find .SAFE directories
        safe_files.extend(self.slc_dir.glob("S1*_IW_SLC_*.SAFE"))

        # Find .zip files
        safe_files.extend(self.slc_dir.glob("S1*_IW_SLC_*.zip"))

        # Also check if slc_dir is a file list
        if self.slc_dir.is_file():
            content = self.slc_dir.read_text()
            for line in content.strip().split("\n"):
                path = Path(line.strip())
                if path.exists():
                    safe_files.append(path)

        return sorted(safe_files)

    def parse_safe_file(self, safe_file: Path) -> SafeFileInfo:
        """Parse a Sentinel-1 SAFE file to extract metadata.

        Parameters
        ----------
        safe_file : Path
            Path to the SAFE file (.SAFE or .zip).

        Returns
        -------
        SafeFileInfo
            Parsed SAFE file information.

        """
        name = safe_file.name

        # Try to parse from filename first
        match = SAFE_PATTERN.match(name)
        if match:
            platform = match.group("platform")
            start_str = match.group("start")
            stop_str = match.group("stop")

            start_time = datetime.strptime(start_str, "%Y%m%dT%H%M%S").replace(
                tzinfo=UTC
            )
            stop_time = datetime.strptime(stop_str, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
            date = start_time.strftime("%Y%m%d")

            return SafeFileInfo(
                path=safe_file,
                date=date,
                start_time=start_time,
                stop_time=stop_time,
                platform=platform,
                orbit_file=None,
            )

        # Fallback: parse from manifest.safe inside the file
        return self._parse_from_manifest(safe_file)

    def _parse_from_manifest(self, safe_file: Path) -> SafeFileInfo:
        """Parse SAFE file from manifest.safe.

        Parameters
        ----------
        safe_file : Path
            Path to the SAFE file.

        Returns
        -------
        SafeFileInfo
            Parsed SAFE file information.

        """
        manifest_path = "manifest.safe"

        if safe_file.suffix == ".zip":
            with zipfile.ZipFile(safe_file) as zf, zf.open(manifest_path) as f:
                tree = ET.parse(f)
        else:
            tree = ET.parse(safe_file / manifest_path)

        root = tree.getroot()

        # Extract metadata from XML
        # Find acquisition period
        start_str = None
        stop_str = None
        platform = None

        for elem in root.iter():
            if "startTime" in elem.tag:
                start_str = elem.text
            elif "stopTime" in elem.tag:
                stop_str = elem.text
            elif (
                ("platformShortName" in elem.tag or "familyName" in elem.tag)
                and elem.text
                and "Sentinel" in elem.text
            ):
                # Try to get platform from another element
                pass

        # Try to get platform from filename as fallback
        if safe_file.name.startswith("S1A"):
            platform = "S1A"
        elif safe_file.name.startswith("S1B"):
            platform = "S1B"

        if start_str is None or stop_str is None:
            msg = f"Could not parse acquisition times from {safe_file}"
            raise ValueError(msg)

        start_time = datetime.fromisoformat(start_str)
        stop_time = datetime.fromisoformat(stop_str)
        date = start_time.strftime("%Y%m%d")

        return SafeFileInfo(
            path=safe_file,
            date=date,
            start_time=start_time,
            stop_time=stop_time,
            platform=platform or "S1A",
            orbit_file=None,
        )

    def get_orbit_file(
        self,
        safe_file: Path,
        orbit_type: Literal["auto", "precise", "restituted"] = "auto",
    ) -> Path | None:
        """Find the orbit file for a Sentinel-1 SAFE file.

        Uses the OrbitFinder to automatically locate the appropriate orbit file.

        Parameters
        ----------
        safe_file : Path
            Path to the SAFE file.
        orbit_type : {"auto", "precise", "restituted"}, optional
            Orbit type preference. Default is "auto" (tries precise, falls back
            to restituted).

        Returns
        -------
        Path | None
            Path to the orbit file, or None if not found.

        """
        if self.orbit_dir is None:
            return None

        from faninsar.isce2.orbit_finder import OrbitFinder

        finder = OrbitFinder(self.orbit_dir)
        return finder.find_orbit(safe_file, orbit_type=orbit_type)

    def find_orbit_files(
        self, orbit_type: Literal["auto", "precise", "restituted"] = "auto"
    ) -> dict[Path, Path | None]:
        """Find orbit files for all SAFE files in the directory.

        This method uses the OrbitFinder to batch-find orbit files for all
        discovered SAFE files.

        Parameters
        ----------
        orbit_type : {"auto", "precise", "restituted"}, optional
            Orbit type preference. Default is "auto" (tries precise, falls back
            to restituted).

        Returns
        -------
        dict[Path, Path | None]
            Mapping from SAFE file path to orbit file path. Value is None if
            no orbit file was found for that SAFE file.

        Examples
        --------
        >>> sensor = Sentinel1Sensor(
        ...     slc_dir=Path("/data/SLC"),
        ...     orbit_dir=Path("/data/orbits"),
        ... )
        >>> orbit_map = sensor.find_orbit_files(orbit_type="precise")
        >>> for safe_file, orbit_file in orbit_map.items():
        ...     if orbit_file is not None:
        ...         print(f"{safe_file.name} -> {orbit_file.name}")

        """
        if self.orbit_dir is None:
            return {}

        from faninsar.isce2.orbit_finder import OrbitFinder

        finder = OrbitFinder(self.orbit_dir)
        safe_files = self.find_safe_files()
        return finder.find_orbits_batch(safe_files, orbit_type=orbit_type)

    def get_aux_file(self, safe_file: Path) -> Path | None:
        """Find the auxiliary file for a Sentinel-1 SAFE file.

        Parameters
        ----------
        safe_file : Path
            Path to the SAFE file.

        Returns
        -------
        Path | None
            Path to the auxiliary file, or None if not found.

        """
        if self.aux_dir is None:
            return None

        info = self.parse_safe_file(safe_file)

        # AUX file naming convention:
        # S1A_AUX_CAL_V20140915T000000_G20140915T000000.SAFE
        pattern = f"{info.platform}_AUX_CAL_V{info.start_time.strftime('%Y%m%d')}*.SAFE"
        aux_files = list(self.aux_dir.glob(pattern))

        if aux_files:
            return aux_files[0]

        return None

    def _check_bbox_overlap(self, safe_file: Path, bbox: BoundingBox) -> bool:
        """Check if SAFE file overlaps with bounding box.

        Uses the preview map.kml file inside the SAFE archive.

        Parameters
        ----------
        safe_file : Path
            Path to the SAFE file.
        bbox : BoundingBox
            Bounding box to check.

        Returns
        -------
        bool
            True if overlaps, False otherwise.

        """
        try:
            coords = self._get_footprint_coords(safe_file)
        except Exception as e:
            logger.debug("Failed to check bbox overlap for %s: %s", safe_file, e)
            return True  # Assume overlap on error
        else:
            if coords is None:
                return True  # Assume overlap if can't determine

            # Check if any corner is within bbox
            for lon, lat in coords:
                if bbox.left <= lon <= bbox.right and bbox.bottom <= lat <= bbox.top:
                    return True

            return False

    def _get_footprint_coords(
        self, safe_file: Path
    ) -> list[tuple[float, float]] | None:
        """Get footprint coordinates from SAFE file.

        Parameters
        ----------
        safe_file : Path
            Path to the SAFE file.

        Returns
        -------
        list[tuple[float, float]] | None
            List of (lon, lat) coordinates, or None if not found.

        """
        kml_path = "preview/map.kml"

        try:
            if safe_file.suffix == ".zip":
                with zipfile.ZipFile(safe_file) as zf:
                    if kml_path not in zf.namelist():
                        return None
                    with zf.open(kml_path) as f:
                        content = f.read().decode("utf-8")
            else:
                kml_file = safe_file / kml_path
                if not kml_file.exists():
                    return None
                content = kml_file.read_text()

        except Exception as e:
            logger.debug("Failed to get footprint from %s: %s", safe_file, e)
            return None
        else:
            # Parse coordinates from KML
            coords: list[tuple[float, float]] = []

            coord_pattern = re.compile(r"<coordinates>([^<]+)</coordinates>")
            for match in coord_pattern.finditer(content):
                coord_str = match.group(1)
                for point in coord_str.strip().split():
                    parts = point.split(",")
                    if len(parts) >= 2:
                        lon, lat = float(parts[0]), float(parts[1])
                        coords.append((lon, lat))

            return coords or None
