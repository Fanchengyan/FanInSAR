"""Base workflow class for ISCE2 processing.

This module provides the abstract base class for all ISCE2 processing workflows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from faninsar._core.sar import Acquisition, Baselines, Pairs, PairsFactory
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from pathlib import Path

    from faninsar.isce2 import PathManager, TopsStackCommands
    from faninsar.query import BoundingBox

logger = setup_logger(__name__)


class BaseWorkflow(ABC):
    """Abstract base class for ISCE2 processing workflows.

    This class provides the foundation for all ISCE2 processing workflows,
    using composition pattern where TopsStackCommands is created internally.

    Parameters
    ----------
    path_manager : PathManager
        Path manager instance for all file paths.
    num_process : int, optional
        Number of parallel processes. Default is 1.
    use_gpu : bool, optional
        Whether to enable GPU acceleration. Default is False.
    text_cmd : str, optional
        Command prefix for backward compatibility. Default is "".
    bbox : BoundingBox | None, optional
        Spatial extent filter. If None, processes all data.

    Attributes
    ----------
    paths : PathManager
        Path manager instance.
    cmd_mgr : TopsStackCommands
        Command manager (read-only, created internally).
    acquisitions : Acquisition
        Acquisition date collection.
    pairs : Pairs
        Interferometric pair collection.
    pairs_factory : PairsFactory
        Factory for building interferometric pairs.
    full_pairs : Pairs
        All possible date pairs from acquisitions.
    baselines : Baselines
        Baseline data (lazy loaded).
    bbox : BoundingBox | None
        Spatial extent.

    Examples
    --------
    >>> from faninsar.isce2 import PathManager
    >>> from faninsar.isce2.workflows import InterferogramStack
    >>> pm = PathManager(
    ...     work_dir="/data/processing",
    ...     slc_dir="/data/SLC",
    ...     dem="/data/dem.wgs84",
    ... )
    >>> workflow = InterferogramStack(
    ...     path_manager=pm,
    ...     num_process=4,
    ...     use_gpu=True,
    ... )
    >>> workflow.generate_run_files()
    >>> workflow.execute(parallel=True)

    Notes
    -----
    TopsStackCommands is created internally with the same PathManager instance.
    This ensures a single source of truth for path configuration.

    """

    def __init__(
        self,
        path_manager: PathManager,
        num_process: int = 1,
        use_gpu: bool = False,
        text_cmd: str = "",
        bbox: BoundingBox | None = None,
    ) -> None:
        """Initialize the BaseWorkflow."""
        self._paths = path_manager
        self.bbox = bbox

        # Create TopsStackCommands internally (composition pattern)
        from faninsar.isce2 import TopsStackCommands

        self._cmd_mgr = TopsStackCommands(
            path_manager=path_manager,
            num_process=num_process,
            use_gpu=use_gpu,
            text_cmd=text_cmd,
        )

        # Initialize acquisitions and pairs
        self._acquisitions = self._discover_acquisitions()
        self._pairs = Pairs([])
        self._pairs_factory: PairsFactory | None = None

        # Baseline data (lazy loaded)
        self._baselines: Baselines | None = None

    @property
    def paths(self) -> PathManager:
        """Get the path manager."""
        return self._paths

    @property
    def cmd_mgr(self) -> TopsStackCommands:
        """Access the command manager (read-only)."""
        return self._cmd_mgr

    @property
    def acquisitions(self) -> Acquisition:
        """Get the acquisition date collection."""
        return self._acquisitions

    @property
    def baselines(self) -> Baselines:
        """Get baseline data (lazy loaded)."""
        if self._baselines is None:
            self._baselines = self._load_baselines()
        return self._baselines

    @property
    def pairs(self) -> Pairs:
        """Get the interferometric pair collection."""
        return self._pairs

    def set_pairs(self, pairs: Pairs) -> None:
        """Set the interferometric pair collection."""
        pairs_valid = self.full_pairs.intersect(pairs)
        if len(pairs_valid) != len(pairs):
            pairs_invalid = pairs - pairs_valid
            logger.warning(
                "The following pairs are not valid and will be ignored: \n%s",
                "\n".join(pairs_invalid.to_names()),
            )

        self._pairs = pairs_valid

    @property
    def pairs_factory(self) -> PairsFactory:
        """Return a pairs factory built from current acquisitions."""
        if self._pairs_factory is None:
            self._pairs_factory = PairsFactory(self.acquisitions)
        return self._pairs_factory

    @property
    def full_pairs(self) -> Pairs:
        """Return all possible date pairs from acquisitions."""
        return self.pairs_factory.full_pairs

    @abstractmethod
    def generate_run_files(self) -> None:
        """Generate all run files and config files.

        This method must be implemented by subclasses to generate
        the appropriate run files for each workflow type.

        """
        ...

    def execute(
        self,
        parallel: bool = False,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """Execute all commands in the workflow.

        Parameters
        ----------
        parallel : bool, optional
            Whether to execute commands in parallel within each run file.
            Default is False.
        dry_run : bool, optional
            Whether to only print commands without executing. Default is False.

        Returns
        -------
        dict[str, int]
            Mapping from command name to return code. 0 indicates success.

        """
        results: dict[str, int] = {}

        # Execute each run file in order
        run_dir = self.paths.run_dir
        if not run_dir.exists():
            logger.warning("Run directory does not exist: %s", run_dir)
            return results

        for run_file in sorted(run_dir.glob("run_*")):
            if not run_file.is_file():
                continue

            logger.info("Executing: %s", run_file.name)

            if dry_run:
                content = run_file.read_text()
                logger.info("Would execute:\n%s", content)
                continue

            # Execute commands from run file
            if parallel:
                results.update(self._cmd_mgr.execute_parallel(run_file))
            else:
                results.update(self._cmd_mgr.execute_sequential(run_file))

        return results

    def _discover_acquisitions(self) -> Acquisition:
        """Auto-discover acquisition dates from SLC directory.

        Returns
        -------
        Acquisition
            Collection of discovered acquisition dates.

        """
        slc_dir = self.paths.slc_dir
        if slc_dir is None:
            logger.warning("SLC directory is not set")
            return Acquisition([])

        dates: set[str] = set()

        # Find all SAFE files and extract dates
        for safe_file in slc_dir.glob("S1*_IW_SLC_*.SAFE"):
            # Extract date from filename: S1A_IW_SLC__1SDV_YYYYMMDDTHHMMSS_...
            parts = safe_file.name.split("_")
            if len(parts) >= 6:
                date_str = parts[5].split("T")[0]
                if len(date_str) == 8 and date_str.isdigit():
                    dates.add(date_str)

        # Also check for zip files
        for zip_file in slc_dir.glob("S1*_IW_SLC_*.zip"):
            parts = zip_file.name.split("_")
            if len(parts) >= 6:
                date_str = parts[5].split("T")[0]
                if len(date_str) == 8 and date_str.isdigit():
                    dates.add(date_str)

        # Convert to Acquisition
        import pandas as pd

        sorted_dates = sorted(dates)
        return (
            Acquisition(pd.to_datetime(sorted_dates))
            if sorted_dates
            else Acquisition([])
        )

    def _normalize_acquisitions(
        self,
        acquisitions: Acquisition | list[str] | tuple[str, ...] | None,
    ) -> Acquisition:
        """Normalize acquisition input to an Acquisition object.

        Parameters
        ----------
        acquisitions : Acquisition | list[str] | tuple[str, ...] | None
            Acquisition input.

        Returns
        -------
        Acquisition
            Normalized acquisitions.

        """
        if acquisitions is None:
            return Acquisition([])
        if isinstance(acquisitions, Acquisition):
            return acquisitions
        if isinstance(acquisitions, str):
            acquisition_values: list[str] = [acquisitions]
        else:
            acquisition_values = list(acquisitions)
        import pandas as pd

        return Acquisition(pd.to_datetime(acquisition_values))

    def _validate_acquisitions(self, acquisitions: Acquisition) -> None:
        """Validate acquisitions against available SLC dates.

        Parameters
        ----------
        acquisitions : Acquisition
            Acquisition dates to validate.

        Raises
        ------
        ValueError
            If acquisitions contain dates not found in the SLC directory.

        """
        available = self._discover_acquisitions()
        if len(available) == 0:
            logger.warning(
                "No acquisitions discovered from SLC directory for validation"
            )
            return
        available_dates = set(available.strftime("%Y%m%d"))
        requested_dates = set(acquisitions.strftime("%Y%m%d"))
        missing_dates = sorted(requested_dates - available_dates)
        if missing_dates:
            logger.error(
                "Acquisition dates are missing from SLC directory: %s",
                ", ".join(missing_dates),
            )
            msg = "Acquisition dates are missing from SLC directory"
            raise ValueError(msg)

    def _set_acquisitions(
        self, acquisitions: Acquisition | list[str] | tuple[str, ...] | None
    ) -> None:
        """Set acquisitions and reset dependent cached properties.

        Parameters
        ----------
        acquisitions : Acquisition | list[str] | tuple[str, ...] | None
            Optional acquisitions override. If None, auto-discover from SLC.

        """
        if acquisitions is None:
            self._acquisitions = self._discover_acquisitions()
        else:
            normalized = self._normalize_acquisitions(acquisitions)
            self._validate_acquisitions(normalized)
            self._acquisitions = normalized
        self._pairs_factory = None

    def _load_baselines(self) -> Baselines:
        """Load baseline data from file.

        Returns
        -------
        Baselines
            Loaded baseline data.

        """
        # TODO: Implement baseline loading from baselines directory
        import numpy as np
        import pandas as pd

        # Return empty baselines for now
        return Baselines(
            dates=pd.DatetimeIndex([]),
            values=np.array([]),
        )

    def create_directories(self) -> None:
        """Create all required directories for processing."""
        self.paths.create_all_dirs()

    def _generate_workspace_run_all_script(
        self, script_name: str = "run_all.sh"
    ) -> Path:
        """Generate a helper script to run all run-files sequentially.

        Parameters
        ----------
        script_name : str, optional
            Output script name in workspace root. Default is ``"run_all.sh"``.

        Returns
        -------
        Path
            Path to generated helper script.

        Raises
        ------
        FileNotFoundError
            If no run-files are found in the run directory.

        """
        run_files = sorted(
            path for path in self.paths.run_dir.glob("run_*.sh") if path.is_file()
        )
        if not run_files:
            run_files = sorted(
                path for path in self.paths.run_dir.glob("run_*") if path.is_file()
            )

        if not run_files:
            logger.error("No run files found in %s", self.paths.run_dir)
            msg = f"No run files found in {self.paths.run_dir}"
            raise FileNotFoundError(msg)

        script_path = self.paths.work_dir / script_name
        lines = [
            "#!/bin/sh",
            "",
            "set -eu",
            "",
            f'RUN_DIR="{self.paths.run_dir}"',
            "",
        ]

        for run_file in run_files:
            lines.extend((
                f'echo "[RUN] {run_file.name}"',
                f'sh "$RUN_DIR/{run_file.name}"',
                "",
            ))

        lines.extend(('echo "All run scripts completed."', ""))

        script_path.write_text("\n".join(lines), encoding="utf-8")
        script_path.chmod(0o755)
        logger.info("Writing %s", script_path)
        return script_path

    def print_status(self) -> None:
        """Print current workflow status."""
        logger.info("=" * 60)
        logger.info("Workflow: %s", self.__class__.__name__)
        logger.info("=" * 60)
        logger.info("Acquisitions: %d dates", len(self.acquisitions))
        logger.info(
            "  Range: %s to %s", self.acquisitions.min(), self.acquisitions.max()
        )
        logger.info("Pairs: %d interferometric pairs", len(self.pairs))
        logger.info("Paths:")
        self.paths.print_all_paths()
