"""Base workflow class for ISCE2 processing.

This module provides the abstract base class for all ISCE2 processing workflows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from faninsar._core.sar import Acquisition, Baselines, Pairs, PairsFactory
from faninsar.logging import setup_logger

if TYPE_CHECKING:
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
    acquisitions : Acquisition | None, optional
        Specified acquisition date subset. If None, auto-discovers all dates.
    pairs : Pairs | None, optional
        Specified interferometric pair subset. If None, auto-generates pairs.
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
        acquisitions: Acquisition | None = None,
        pairs: Pairs | None = None,
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

        # Initialize acquisitions
        self.acquisitions = acquisitions or self._discover_acquisitions()

        # Initialize pairs
        self.pairs = pairs or self._generate_pairs()

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
    def baselines(self) -> Baselines:
        """Get baseline data (lazy loaded)."""
        if self._baselines is None:
            self._baselines = self._load_baselines()
        return self._baselines

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

    def _generate_pairs(
        self,
        max_interval: int = 1,
        max_days: int = 180,
    ) -> Pairs:
        """Generate interferometric pairs using PairsFactory.

        Parameters
        ----------
        max_interval : int, optional
            Maximum interval between acquisitions. Default is 1.
        max_days : int, optional
            Maximum days between acquisitions. Default is 180.

        Returns
        -------
        Pairs
            Generated interferometric pairs.

        """
        if len(self.acquisitions) == 0:
            return Pairs([])

        factory = PairsFactory(self.acquisitions)
        return factory.from_interval(max_interval=max_interval, max_days=max_days)

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
