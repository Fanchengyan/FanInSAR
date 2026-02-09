"""Path manager for ISCE2 stack processing.

This module provides centralized path management for ISCE2 InSAR processing workflows,
with support for multilook configurations and TOML serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import starmap
from pathlib import Path
from typing import ClassVar, Self

import tomli_w

from faninsar.logging import setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True)
class Multilook:
    """Multilook settings for ISCE2 processing.

    Parameters
    ----------
    azimuth : int
        Number of azimuth looks for multilooking.
    range : int
        Number of range looks for multilooking.

    Attributes
    ----------
    azimuth : int
        Number of azimuth looks.
    range : int
        Number of range looks.

    Examples
    --------
    >>> config = Multilook(azimuth=2, range=8)
    >>> print(config)
    (2, 8)

    """

    azimuth: int
    range: int

    def __str__(self) -> str:
        """Return string representation of multilook settings."""
        return f"({self.azimuth}, {self.range})"


class PathManager:
    """Centralized path and directory management for ISCE2 processing.

    This class manages all file paths and directory structures using pathlib.Path
    for modern path operations. Supports multilook settings and TOML
    serialization for configuration persistence.

    Parameters
    ----------
    work_dir : str | Path
        Main working directory for the ISCE2 processing workflow.
    slc_dir : str | Path | None, optional
        Directory containing Sentinel-1 SLC data. Default is None.
    orbit_dir : str | Path | None, optional
        Directory containing orbit files. Default is None.
    aux_dir : str | Path | None, optional
        Directory containing auxiliary data. Default is None.
    dem : str | Path | None, optional
        Path to DEM file (GeoTIFF or NetCDF). Default is None.
    multilook : list[tuple[int, int]] | None, optional
        List of (azimuth, range) tuples for multilook settings.
        Default is None. Can be specified at initialization or added
        later using add_multilook() method.
    dir_names : dict[str, str] | None, optional
        Custom directory names for internal directories. Default is None.

    Attributes
    ----------
    work_dir : Path
        Main working directory.
    slc_dir : Path | None
        Directory containing SLC data.
    orbit_dir : Path | None
        Directory containing orbit files.
    aux_dir : Path | None
        Directory containing auxiliary data.
    dem : Path | None
        Path to DEM file.

    Examples
    --------
    >>> pm = PathManager(
    ...     work_dir="/data/processing",
    ...     slc_dir="/data/SLC",
    ...     dem="/data/dem.tif",
    ...     multilook=[(1, 4), (2, 8)],
    ... )
    >>> pm.create_all_dirs()
    >>> pm.save("config.toml")

    Or add multilook settings later:

    >>> pm = PathManager(work_dir="/data/processing", slc_dir="/data/SLC")
    >>> pm.add_multilook(1, 4)
    >>> pm.add_multilook(2, 8)

    Notes
    -----
    The PathManager uses TOML for serialization, which requires Python 3.11+
    for reading and tomli-w for writing. For Python < 3.11, install tomli as well.

    """

    # Default directory names for ISCE2 processing
    DEFAULT_DIR_NAMES: ClassVar[dict[str, str]] = {
        "run": "run_files",
        "config": "configs",
        "reference": "reference",
        "secondarys": "secondarys",
        "coreg_secondarys": "coreg_secondarys",
        "slc": "slc",
        "geom_reference": "geom_reference",
        "geom_slc": "geom_slc",
        "merged": "merged",
        "baselines": "baselines",
        "misreg": "misreg",
        "ion": "ion",
        "stack": "stack",
        "multilooked": "multilooked",
    }

    def __init__(
        self,
        work_dir: str | Path,
        slc_dir: str | Path | None = None,
        orbit_dir: str | Path | None = None,
        aux_dir: str | Path | None = None,
        dem: str | Path | None = None,
        multilook: list[tuple[int, int]] | None = None,
        dir_names: dict[str, str] | None = None,
    ) -> None:
        # Main working directory
        self._work_dir = Path(work_dir).resolve()

        # External input paths
        self._slc_dir = Path(slc_dir).resolve() if slc_dir else None
        self._orbit_dir = Path(orbit_dir).resolve() if orbit_dir else None
        self._aux_dir = Path(aux_dir).resolve() if aux_dir else None
        self._dem = Path(dem).resolve() if dem else None

        # Directory name configuration
        self._dir_names = {**self.DEFAULT_DIR_NAMES, **(dir_names or {})}

        # Multilook settings
        if multilook:
            self._multilooks = list(starmap(Multilook, multilook))
        else:
            self._multilooks: list[Multilook] = []

    @property
    def work_dir(self) -> Path:
        """Get the main working directory."""
        return self._work_dir

    @property
    def slc_dir(self) -> Path | None:
        """Get the SLC directory."""
        return self._slc_dir

    @property
    def orbit_dir(self) -> Path | None:
        """Get the orbit directory."""
        return self._orbit_dir

    @property
    def aux_dir(self) -> Path | None:
        """Get the auxiliary data directory."""
        return self._aux_dir

    @property
    def dem(self) -> Path | None:
        """Get the DEM file path."""
        return self._dem

    @property
    def multilooked_dir(self) -> Path:
        """Get the multilooked output directory."""
        return self._work_dir / self._dir_names["multilooked"]

    def get_dir_name(self, key: str) -> str:
        """Get directory name for a given key.

        Parameters
        ----------
        key : str
            Directory key (e.g., 'run', 'config', 'reference').

        Returns
        -------
        str
            Directory name.

        Examples
        --------
        >>> pm = PathManager(work_dir="/data")
        >>> pm.get_dir_name("run")
        'run_files'

        """
        return self._dir_names.get(key, key)

    def set_dir_name(self, key: str, name: str) -> None:
        """Set a single directory name.

        Parameters
        ----------
        key : str
            Directory key to update.
        name : str
            New directory name.

        Raises
        ------
        ValueError
            If the key is not a valid directory key.

        Examples
        --------
        >>> pm = PathManager(work_dir="/data")
        >>> pm.set_dir_name("run", "my_run_files")

        """
        if key in self._dir_names:
            self._dir_names[key] = name
        else:
            logger.error("Unknown directory key: %s", key)
            msg = f"Unknown directory key: {key}"
            raise ValueError(msg)

    def update_dir_names(self, updates: dict[str, str]) -> None:
        """Batch update directory names.

        Parameters
        ----------
        updates : dict[str, str]
            Dictionary of directory key to name mappings.

        Raises
        ------
        ValueError
            If any key is not a valid directory key.

        Examples
        --------
        >>> pm = PathManager(work_dir="/data")
        >>> pm.update_dir_names({"run": "my_run_files", "slc": "raw_slc"})

        """
        for key, name in updates.items():
            if key in self._dir_names:
                self._dir_names[key] = name
            else:
                logger.error("Unknown directory key: %s", key)
                msg = f"Unknown directory key: {key}"
                raise ValueError(msg)

    def has_multilook(self) -> bool:
        """Check if multilook settings are present.

        Returns
        -------
        bool
            True if multilook settings exist, False otherwise.

        Examples
        --------
        >>> pm = PathManager(work_dir="/data", multilook=[(1, 4)])
        >>> pm.has_multilook()
        True

        """
        return len(self._multilooks) > 0

    def add_multilook(self, azimuth: int, range: int) -> None:
        """Add a multilook setting.

        Parameters
        ----------
        azimuth : int
            Number of azimuth looks.
        range : int
            Number of range looks.

        Examples
        --------
        >>> pm = PathManager(work_dir="/data")
        >>> pm.add_multilook_config(2, 8)

        """
        look = Multilook(azimuth, range)
        if look not in self._multilooks:
            self._multilooks.append(look)

    def get_multilooks(self) -> list[Multilook]:
        """Get all multilook settings.

        Returns
        -------
        list[Multilook]
            List of multilook configurations.

        Examples
        --------
        >>> pm = PathManager(work_dir="/data", multilook=[(1, 4), (2, 8)])
        >>> looks = pm.get_multilooks()
        >>> len(looks)
        2

        """
        return self._multilooks.copy()

    def multilook_path(self, azimuth: int, range: int) -> Path:
        """Get the multilook root directory for specific looks.

        Parameters
        ----------
        azimuth : int
            Number of azimuth looks.
        range : int
            Number of range looks.

        Returns
        -------
        Path
            Path to the multilook directory.

        Examples
        --------
        >>> pm = PathManager(work_dir="/data")
        >>> pm.multilook_path(1, 4)
        PosixPath('/data/multilooked/1_4')

        """
        return self.multilooked_dir / f"{azimuth}_{range}"

    def multilook_merged_path(self, azimuth: int, range: int) -> Path:
        """Get the multilook merged directory for specific looks.

        Parameters
        ----------
        azimuth : int
            Number of azimuth looks.
        range : int
            Number of range looks.

        Returns
        -------
        Path
            Path to the multilook merged directory.

        Examples
        --------
        >>> pm = PathManager(work_dir="/data")
        >>> pm.multilook_merged_path(2, 8)
        PosixPath('/data/multilooked/2_8/merged')

        """
        return self.multilook_path(azimuth, range) / "merged"

    def multilook_interferogram_path(
        self, azimuth: int, range: int, ref_date: str, sec_date: str
    ) -> Path:
        """Get the multilook interferogram directory for specific looks.

        Parameters
        ----------
        azimuth : int
            Number of azimuth looks.
        range : int
            Number of range looks.
        ref_date : str
            Reference date (format: YYYYMMDD).
        sec_date : str
            Secondary date (format: YYYYMMDD).

        Returns
        -------
        Path
            Path to the multilook interferogram directory.

        Examples
        --------
        >>> pm = PathManager(work_dir="/data")
        >>> pm.multilook_interferogram_path(1, 4, "20240101", "20240113")
        PosixPath('/data/multilooked/1_4/interferograms/20240101_20240113')

        """
        return (
            self.multilook_path(azimuth, range)
            / "interferograms"
            / f"{ref_date}_{sec_date}"
        )

    @property
    def run_dir(self) -> Path:
        """Get the run files directory."""
        return self._work_dir / self._dir_names["run"]

    @property
    def config_dir(self) -> Path:
        """Get the configuration files directory."""
        return self._work_dir / self._dir_names["config"]

    def reference_path(self) -> Path:
        """Get the reference scene directory."""
        return self._work_dir / self._dir_names["reference"]

    def secondary_path(self, date: str) -> Path:
        """Get the secondary scene directory for a specific date.

        Parameters
        ----------
        date : str
            Date string (format: YYYYMMDD).

        Returns
        -------
        Path
            Path to the secondary scene directory.

        """
        return self._work_dir / self._dir_names["secondarys"] / date

    def coreg_secondary_path(self, date: str) -> Path:
        """Get the coregistered secondary scene directory for a specific date.

        Parameters
        ----------
        date : str
            Date string (format: YYYYMMDD).

        Returns
        -------
        Path
            Path to the coregistered secondary scene directory.

        """
        return self._work_dir / self._dir_names["coreg_secondarys"] / date

    def slc_path(self, date: str) -> Path:
        """Get the SLC directory for a specific date.

        Parameters
        ----------
        date : str
            Date string (format: YYYYMMDD).

        Returns
        -------
        Path
            Path to the SLC directory.

        """
        return self._work_dir / self._dir_names["slc"] / date

    def geom_reference_path(self) -> Path:
        """Get the geometry reference directory."""
        return self._work_dir / self._dir_names["geom_reference"]

    def geom_slc_path(self, date: str) -> Path:
        """Get the geometry SLC directory for a specific date.

        Parameters
        ----------
        date : str
            Date string (format: YYYYMMDD).

        Returns
        -------
        Path
            Path to the geometry SLC directory.

        """
        return self._work_dir / self._dir_names["geom_slc"] / date

    def interferogram_path(self, ref_date: str, sec_date: str) -> Path:
        """Get the interferogram directory for a reference-secondary pair.

        Parameters
        ----------
        ref_date : str
            Reference date (format: YYYYMMDD).
        sec_date : str
            Secondary date (format: YYYYMMDD).

        Returns
        -------
        Path
            Path to the interferogram directory.

        """
        return (
            self._work_dir
            / self._dir_names["merged"]
            / "interferograms"
            / f"{ref_date}_{sec_date}"
        )

    def config_file_path(self, name: str) -> Path:
        """Get the path to a configuration file.

        Parameters
        ----------
        name : str
            Configuration file name.

        Returns
        -------
        Path
            Path to the configuration file.

        """
        return self.config_dir / name

    def run_file_path(self, name: str) -> Path:
        """Get the path to a run file.

        Parameters
        ----------
        name : str
            Run file name.

        Returns
        -------
        Path
            Path to the run file.

        """
        return self.run_dir / name

    def print_all_paths(self, show_exists: bool = True, prefix: str = "") -> None:
        """Print all path information with optional existence status.

        Parameters
        ----------
        show_exists : bool, optional
            Whether to show existence status with checkmarks. Default is True.
        prefix : str, optional
            Prefix string for indentation. Default is "".

        Examples
        --------
        >>> pm = PathManager(work_dir="/data")
        >>> pm.print_all_paths()

        """
        print(f"{prefix}📁 Working directory: {self._work_dir}")
        print(f"{prefix}\n  External inputs:")
        print(
            f"{prefix}    SLC directory:    {self._fmt_path(self._slc_dir, show_exists)}"
        )
        print(
            f"{prefix}    Orbit directory:   {self._fmt_path(self._orbit_dir, show_exists)}"
        )
        print(
            f"{prefix}    Aux directory:    {self._fmt_path(self._aux_dir, show_exists)}"
        )
        print(f"{prefix}    DEM file:         {self._fmt_path(self._dem, show_exists)}")

        print(f"{prefix}\n  Internal working directories:")
        for key, path in self._get_standard_paths().items():
            print(f"{prefix}    {key}: {self._fmt_path(path, show_exists)}")

        if self.has_multilook():
            print(f"{prefix}\n  Multilook settings:")
            for looks in self._multilooks:
                ml_path = self.multilook_path(looks.azimuth, looks.range)
                print(f"{prefix}    {looks}:")
                print(f"{prefix}      Root:   {self._fmt_path(ml_path, show_exists)}")
                print(
                    f"{prefix}      Merged: {self._fmt_path(ml_path / 'merged', show_exists)}"
                )

    def _fmt_path(self, path: Path | None, show_exists: bool) -> str:
        """Format path for display with optional existence indicator.

        Parameters
        ----------
        path : Path | None
            Path to format.
        show_exists : bool
            Whether to show existence status.

        Returns
        -------
        str
            Formatted path string.

        """
        if path is None:
            return "Not set"
        if show_exists:
            exists = "✓" if path.exists() else "✗"
            return f"[{exists}] {path}"
        return str(path)

    def to_toml(self, filepath: str | Path | None = None) -> str:
        """Export configuration to TOML format.

        Parameters
        ----------
        filepath : str | Path | None, optional
            Path to save the TOML file. If None, returns the TOML string.
            Default is None.

        Returns
        -------
        str
            TOML formatted string.

        Examples
        --------
        >>> pm = PathManager(work_dir="/data")
        >>> toml_str = pm.to_toml()
        >>> pm.to_toml("config.toml")

        """
        data = {
            "work": {"dir": str(self._work_dir)},
            "external": {},
            "dir_names": self._dir_names.copy(),
            "multilook": {
                "looks": [
                    {"azimuth": looks.azimuth, "range": looks.range}
                    for looks in self._multilooks
                ]
            },
            "metadata": {
                "created": datetime.now().isoformat(),
                "version": "1.1",
            },
        }

        # Only add external paths that are set (TOML can't serialize None)
        if self._slc_dir:
            data["external"]["slc_dir"] = str(self._slc_dir)
        if self._orbit_dir:
            data["external"]["orbit_dir"] = str(self._orbit_dir)
        if self._aux_dir:
            data["external"]["aux_dir"] = str(self._aux_dir)
        if self._dem:
            data["external"]["dem"] = str(self._dem)

        toml_str = tomli_w.dumps(data)

        if filepath:
            Path(filepath).write_text(toml_str, encoding="utf-8")
            logger.info("Path configuration saved to: %s", filepath)

        return toml_str

    @classmethod
    def from_toml(cls, filepath: str | Path) -> Self:
        """Load configuration from TOML file.

        Parameters
        ----------
        filepath : str | Path
            Path to the TOML configuration file.

        Returns
        -------
        PathManager
            PathManager instance loaded from TOML.

        Raises
        ------
        FileNotFoundError
            If the TOML file does not exist.

        Examples
        --------
        >>> pm = PathManager.from_toml("config.toml")

        """
        # Try to import tomllib (Python 3.11+) or fall back to tomli
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                msg = (
                    "Neither tomllib (Python 3.11+) nor tomli is available. "
                    "Install tomli: pip install tomli"
                )
                logger.error(msg)
                raise ImportError(msg) from None

        if not Path(filepath).exists():
            logger.error("TOML file not found: %s", filepath)
            raise FileNotFoundError(f"TOML file not found: {filepath}")

        with Path(filepath).open("rb") as f:
            data = tomllib.load(f)

        # Extract multilook settings
        ml_configs: list[tuple[int, int]] | None = None
        if "multilook" in data and "looks" in data["multilook"]:
            ml_configs = [
                (looks["azimuth"], looks["range"])
                for looks in data["multilook"]["looks"]
            ]

        # Create instance
        ext = data.get("external", {})
        pm = cls(
            work_dir=data["work"]["dir"],
            slc_dir=ext.get("slc_dir"),
            orbit_dir=ext.get("orbit_dir"),
            aux_dir=ext.get("aux_dir"),
            dem=ext.get("dem"),
            multilook=ml_configs,
        )

        # Restore directory names
        pm._dir_names = data.get("dir_names", pm._dir_names)

        logger.info("Path configuration loaded from: %s", filepath)
        return pm

    def save(self, filepath: str | Path) -> None:
        """Save configuration to TOML file.

        Parameters
        ----------
        filepath : str | Path
            Path to save the TOML file.

        Examples
        --------
        >>> pm = PathManager(work_dir="/data")
        >>> pm.save("config.toml")

        """
        self.to_toml(filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        """Load configuration from TOML file.

        Parameters
        ----------
        filepath : str | Path
            Path to the TOML configuration file.

        Returns
        -------
        PathManager
            PathManager instance loaded from TOML.

        Examples
        --------
        >>> pm = PathManager.load("config.toml")

        """
        return cls.from_toml(filepath)

    def create_all_dirs(
        self, include_multilook: bool = True, exist_ok: bool = True
    ) -> None:
        """Create all directories defined in the PathManager.

        Parameters
        ----------
        include_multilook : bool, optional
            Whether to create multilook directories. Default is True.
        exist_ok : bool, optional
            Whether to ignore existing directories. Default is True.

        Examples
        --------
        >>> pm = PathManager(work_dir="/data", multilook_configs=[(1, 4)])
        >>> pm.create_all_dirs()

        """
        created: list[str] = []

        # Standard directories
        for key, path in self._get_standard_paths().items():
            if not path.exists():
                path.mkdir(parents=True, exist_ok=exist_ok)
                created.append(key)

        # Multilook directories
        if include_multilook and self.has_multilook():
            for looks in self._multilooks:
                ml_path = self.multilook_path(looks.azimuth, looks.range)
                if not (ml_path / "merged").exists():
                    (ml_path / "merged").mkdir(parents=True, exist_ok=exist_ok)
                    created.append(f"multilook/{looks}/merged")
                if not (ml_path / "interferograms").exists():
                    (ml_path / "interferograms").mkdir(parents=True, exist_ok=exist_ok)
                    created.append(f"multilook/{looks}/interferograms")

        if created:
            logger.info(f"Created {len(created)} directories")
        else:
            logger.info("All directories already exist")

    def create_specific_dirs(self, *dir_keys: str, exist_ok: bool = True) -> None:
        """Create specific directories by key.

        Parameters
        ----------
        *dir_keys : str
            Directory keys to create.
        exist_ok : bool, optional
            Whether to ignore existing directories. Default is True.

        Examples
        --------
        >>> pm = PathManager(work_dir="/data")
        >>> pm.create_specific_dirs("run", "config", "reference")

        """
        for key in dir_keys:
            if key in self._dir_names:
                path = self._work_dir / self._dir_names[key]
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=exist_ok)
                    logger.info("Created: %s", path)
            else:
                logger.warning("Unknown directory key: %s", key)

    def _get_standard_paths(self) -> dict[str, Path]:
        """Get dictionary of standard paths.

        Returns
        -------
        dict[str, Path]
            Dictionary mapping directory keys to Path objects.

        """
        return {
            "run": self.run_dir,
            "config": self.config_dir,
            "reference": self.reference_path(),
            "secondarys": self._work_dir / self._dir_names["secondarys"],
            "coreg_secondarys": self._work_dir / self._dir_names["coreg_secondarys"],
            "slc": self._work_dir / self._dir_names["slc"],
            "geom_reference": self._work_dir / self._dir_names["geom_reference"],
            "geom_slc": self._work_dir / self._dir_names["geom_slc"],
            "merged": self._work_dir / self._dir_names["merged"],
            "baselines": self._work_dir / self._dir_names["baselines"],
            "misreg": self._work_dir / self._dir_names["misreg"],
            "ion": self._work_dir / self._dir_names["ion"],
            "stack": self._work_dir / self._dir_names["stack"],
            "multilooked": self.multilooked_dir,
        }

    def get_existing_paths(self) -> dict[str, Path]:
        """Get all existing paths.

        Returns
        -------
        dict[str, Path]
            Dictionary of existing directory paths.

        Examples
        --------
        >>> pm = PathManager(work_dir="/data")
        >>> existing = pm.get_existing_paths()

        """
        return {k: v for k, v in self._get_standard_paths().items() if v.exists()}

    def get_missing_paths(self) -> dict[str, Path]:
        """Get all missing paths.

        Returns
        -------
        dict[str, Path]
            Dictionary of missing directory paths.

        Examples
        --------
        >>> pm = PathManager(work_dir="/data")
        >>> missing = pm.get_missing_paths()

        """
        return {k: v for k, v in self._get_standard_paths().items() if not v.exists()}
