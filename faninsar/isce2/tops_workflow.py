"""ISCE2 TOPS Sentinel stack workflow manager.

This module provides a comprehensive workflow manager for ISCE2 Sentinel-1 TOPS
stack processing, supporting multiple multilook settings and different
workflow types (SLC stack, interferogram stack, ionosphere correction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.isce2.command_manager import Command, TopsStackCommands

logger = setup_logger(__name__)


@dataclass(frozen=True)
class WorkflowConfig:
    """Configuration for TOPS stack workflow.

    Parameters
    ----------
    workflow_type : str
        Type of workflow ('slc', 'interferogram', 'correlation', 'offset').
    coreg_method : str
        Coregistration method ('NESD' or 'geometry').
    num_connections : int | None
        Number of connections per acquisition. None for all connections.
    use_ionosphere : bool
        Whether to perform ionosphere phase estimation.

    Attributes
    ----------
    workflow_type : str
        Workflow type.
    coreg_method : str
        Coregistration method.
    num_connections : int | None
        Number of connections.
    use_ionosphere : bool
        Ionosphere correction flag.

    """

    workflow_type: str
    coreg_method: str
    num_connections: int | None
    use_ionosphere: bool


class TopsStackWorkflow:
    """Workflow manager for ISCE2 TopsStack processing.

    This class orchestrates the complete ISCE2 topsStack processing workflow,
    supporting multiple multilook settings and batch command generation.
    It generates run and config files for all multilook configurations at once,
    with intelligent sharing of intermediate results.

    Parameters
    ----------
    command_manager : TopsStackCommands
        TopsStackCommands instance for command execution. The PathManager is
        accessed via command_manager.paths.
    workflow_type : str, optional
        Type of workflow ('slc', 'interferogram', 'correlation', 'offset').
        Default is 'interferogram'.
    coreg_method : str, optional
        Coregistration method ('NESD' or 'geometry'). Default is 'NESD'.
    num_connections : int | None, optional
        Number of connections per acquisition. Default is None (all).
    use_ionosphere : bool, optional
        Whether to perform ionosphere correction. Default is False.

    Attributes
    ----------
    paths : PathManager
        PathManager instance.
    cmd_mgr : TopsStackCommands
        TopsStackCommands instance.
    config : WorkflowConfig
        Workflow configuration.
    acquisition_dates : list[str]
        List of acquisition dates.
    reference_date : str
        Reference acquisition date.

    Examples
    --------
    >>> from faninsar.isce2 import PathManager, TopsStackCommands, TopsStackWorkflow
    >>> pm = PathManager(
    ...     work_dir="/data/processing",
    ...     slc_dir="/data/SLC",
    ...     orbit_dir="/data/orbits",
    ...     aux_dir="/data/aux",
    ...     dem="/data/dem.wgs84",
    ...     multilook_configs=[(1, 4), (2, 8), (3, 12)],
    ... )
    >>> cmd_mgr = TopsStackCommands(path_manager=pm, num_process=4)
    >>> workflow = TopsStackWorkflow(
    ...     command_manager=cmd_mgr,
    ...     workflow_type="interferogram",
    ... )
    >>> workflow.generate_all()

    Notes
    -----
    The workflow supports sharing of intermediate results across multilook
    configurations. Reference scenes, geometry data, and baselines are shared,
    while merged SLCs and interferograms are computed separately for each
    multilook setting.

    """

    def __init__(
        self,
        command_manager: TopsStackCommands,
        workflow_type: str = "interferogram",
        coreg_method: str = "NESD",
        num_connections: int | None = None,
        use_ionosphere: bool = False,
    ) -> None:
        """Initialize the TopsStackWorkflow."""
        self.cmd_mgr = command_manager
        self.paths = command_manager.paths

        # Validate workflow type
        valid_types = ["slc", "interferogram", "correlation", "offset"]
        if workflow_type not in valid_types:
            logger.error(
                "Invalid workflow type: %s. Must be one of: %s",
                workflow_type,
                valid_types,
            )
            msg = f"Invalid workflow type: {workflow_type}"
            raise ValueError(msg)

        # Validate coregistration method
        valid_methods = ["NESD", "geometry"]
        if coreg_method not in valid_methods:
            logger.error(
                "Invalid coregistration method: %s. Must be one of: %s",
                coreg_method,
                valid_methods,
            )
            msg = f"Invalid coregistration method: {coreg_method}"
            raise ValueError(msg)

        self.config = WorkflowConfig(
            workflow_type=workflow_type,
            coreg_method=coreg_method,
            num_connections=num_connections,
            use_ionosphere=use_ionosphere,
        )

        # Initialize data
        self.acquisition_dates: list[str] = []
        self.reference_date: str = ""

    def generate_all(self) -> None:
        """Generate all run and config files for all multilook configurations.

        This is the main entry point that orchestrates the entire workflow:
        1. Find all acquisition dates
        2. Select reference scene
        3. Build interferometric network
        4. Generate run/config files for each multilook configuration

        Examples
        --------
        >>> workflow.generate_all()

        """
        logger.info("Starting TOPS stack workflow generation")

        # Step 1: Find acquisition dates
        self.acquisition_dates = self._find_acquisition_dates()
        logger.info("Found %d acquisition dates", len(self.acquisition_dates))

        if not self.acquisition_dates:
            logger.warning("No acquisition dates found")
            return

        # Step 2: Select reference
        self.reference_date = self._select_reference()
        logger.info("Selected reference date: %s", self.reference_date)

        # Step 3: Build network
        network = self._build_network()
        logger.info("Built network with %d pairs", len(network))

        # Step 4: Generate workflow for each multilook configuration
        looks = self.paths.get_multilooks()

        if not looks:
            logger.warning("No multilook settings found")
            return

        logger.info("Generating workflow for %d multilook settings", len(looks))

        for ml_config in looks:
            logger.info(
                "Processing multilook setting: azimuth=%d, range=%d",
                ml_config.azimuth,
                ml_config.range,
            )

            if self.config.workflow_type == "slc":
                self._generate_slc_stack(ml_config.azimuth, ml_config.range)
            elif self.config.workflow_type == "interferogram":
                self._generate_interferogram_stack(
                    ml_config.azimuth, ml_config.range, network
                )

        logger.info("Workflow generation completed")

    def _find_acquisition_dates(self) -> list[str]:
        """Find all acquisition dates in the SLC directory.

        Returns
        -------
        list[str]
            Sorted list of acquisition dates (YYYYMMDD format).

        """
        slc_dir = self.paths.slc_dir

        if slc_dir is None:
            logger.warning("SLC directory is not set")
            return []

        # Find all SAFE files and extract dates
        dates: set[str] = set()

        for safe_file in slc_dir.glob("*.SAFE"):
            # Extract date from filename: S1A_IW_SLC__1SDV_YYYYMMDDTHHMMSS_...
            parts = safe_file.name.split("_")
            if len(parts) >= 6:
                date_str = parts[5].split("T")[0]
                if len(date_str) == 8 and date_str.isdigit():
                    dates.add(date_str)

        return sorted(dates)

    def _select_reference(self) -> str:
        """Select reference acquisition date.

        Returns
        -------
        str
            Reference date (YYYYMMDD format).

        Notes
        -----
        Currently selects the first date. Future implementations may
        use more sophisticated selection criteria.

        """
        if not self.acquisition_dates:
            logger.warning("No acquisition dates available for reference selection")
            return ""

        # Select the first date as reference
        return self.acquisition_dates[0]

    def _build_network(self) -> list[tuple[str, str]]:
        """Build interferometric network based on connection strategy.

        Returns
        -------
        list[tuple[str, str]]
            List of (reference_date, secondary_date) pairs.

        Notes
        -----
        Currently builds a sequential network. Future implementations may
        support different network strategies (star, delaunay, etc.).

        """
        network: list[tuple[str, str]] = []

        if len(self.acquisition_dates) < 2:
            logger.warning("Not enough dates to build network")
            return network

        num_conn = self.config.num_connections
        if num_conn is None:
            num_conn = len(self.acquisition_dates)

        # Build sequential network
        for i in range(len(self.acquisition_dates) - 1):
            for j in range(i + 1, min(i + num_conn + 1, len(self.acquisition_dates))):
                network.append((self.acquisition_dates[i], self.acquisition_dates[j]))

        return network

    def _generate_slc_stack(self, azimuth: int, range_looks: int) -> None:
        """Generate run and config files for SLC stack workflow.

        Generates the 10-step SLC stack workflow:
        - run_01_unpack_slc_topo_reference
        - run_02_unpack_secondary_slc
        - run_03_average_baseline
        - run_04_extract_burst_overlaps
        - run_05_overlap_geo2rdr
        - run_06_overlap_resample
        - run_07_pairs_misreg
        - run_08_timeseries_misreg
        - run_09_fullBurst_geo2rdr
        - run_10_fullBurst_resample

        Parameters
        ----------
        azimuth : int
            Number of azimuth looks.
        range_looks : int
            Number of range looks.

        """
        logger.info(
            "Generating SLC stack workflow for looks: (%d, %d)", azimuth, range_looks
        )

        # Generate run files for each step
        run_num = 1

        # Step 1: Unpack reference + topo
        commands = self._generate_reference_commands()
        self.generate_run_file(run_num, "unpack_slc_topo_reference", commands)
        run_num += 1

        # Step 2: Unpack secondary SLCs
        for date in self.acquisition_dates:
            if date != self.reference_date:
                commands = self._generate_secondary_commands(date)
                self.generate_run_file(
                    run_num, f"unpack_secondary_slc_{date}", commands
                )
                run_num += 1

        # Step 3: Average baseline
        commands = self._generate_baseline_commands()
        self.generate_run_file(run_num, "average_baseline", commands)
        run_num += 1

        # Step 4-10: Coregistration workflow
        if self.config.coreg_method == "NESD":
            self._generate_nesd_workflow(run_num)
        else:
            self._generate_geometry_workflow(run_num)

    def _generate_interferogram_stack(
        self, azimuth: int, range_looks: int, network: list[tuple[str, str]]
    ) -> None:
        """Generate run and config files for interferogram stack workflow.

        Extends SLC stack with interferogram generation, multilooking,
        filtering, and unwrapping.

        Parameters
        ----------
        azimuth : int
            Number of azimuth looks.
        range_looks : int
            Number of range looks.
        network : list[tuple[str, str]]
            List of interferometric pairs.

        """
        logger.info(
            "Generating interferogram stack workflow for looks: (%d, %d)",
            azimuth,
            range_looks,
        )

        # First generate SLC stack
        self._generate_slc_stack(azimuth, range_looks)

        # Get the next run number
        run_num = 11

        # Step 11: Extract stack valid region
        commands = self._generate_valid_region_commands()
        self.generate_run_file(run_num, "extract_stack_valid_region", commands)
        run_num += 1

        # Step 12: Merge reference and secondary SLCs
        for date in self.acquisition_dates:
            commands = self._generate_merge_slc_commands(date, azimuth, range_looks)
            self.generate_run_file(run_num, f"merge_slc_{date}", commands)
            run_num += 1

        # Step 13: Generate burst interferograms
        for ref_date, sec_date in network:
            commands = self._generate_burst_igram_commands(ref_date, sec_date)
            self.generate_run_file(
                run_num, f"generate_burst_igram_{ref_date}_{sec_date}", commands
            )
            run_num += 1

        # Step 14: Merge burst interferograms
        for ref_date, sec_date in network:
            commands = self._generate_merge_igram_commands(
                ref_date, sec_date, azimuth, range_looks
            )
            self.generate_run_file(
                run_num, f"merge_igram_{ref_date}_{sec_date}", commands
            )
            run_num += 1

        # Step 15: Filter and coherence
        for ref_date, sec_date in network:
            commands = self._generate_filter_commands(
                ref_date, sec_date, azimuth, range_looks
            )
            self.generate_run_file(
                run_num, f"filter_coherence_{ref_date}_{sec_date}", commands
            )
            run_num += 1

        # Step 16: Unwrap
        for ref_date, sec_date in network:
            commands = self._generate_unwrap_commands(
                ref_date, sec_date, azimuth, range_looks
            )
            self.generate_run_file(run_num, f"unwrap_{ref_date}_{sec_date}", commands)
            run_num += 1

    def _generate_reference_commands(self) -> list[Command]:
        """Generate commands for reference scene unpacking and topo.

        Returns
        -------
        list[Command]
            List of commands.

        """
        commands: list[Command] = []

        # Sentinel1_TOPS command
        # Note: safe_file and orbit_file need to be determined
        # This is a placeholder - actual implementation needs to find files
        if self.paths.slc_dir and self.paths.orbit_dir:
            # Find reference SAFE file
            safe_files = list(self.paths.slc_dir.glob(f"*{self.reference_date}*.SAFE"))
            if safe_files:
                cmd = self.cmd_mgr.sentinel1_tops_cmd(
                    safe_file=safe_files[0],
                    orbit_file=self.paths.orbit_dir,
                    orbit_type="precise",
                    swaths=["IW1", "IW2", "IW3"],
                )
                commands.append(cmd)

        # Topo command
        cmd = self.cmd_mgr.topo_cmd()
        commands.append(cmd)

        return commands

    def _generate_secondary_commands(self, date: str) -> list[Command]:
        """Generate commands for secondary scene unpacking.

        Parameters
        ----------
        date : str
            Secondary date.

        Returns
        -------
        list[Command]
            List of commands.

        """
        commands: list[Command] = []

        if self.paths.slc_dir and self.paths.orbit_dir:
            # Find secondary SAFE file
            safe_files = list(self.paths.slc_dir.glob(f"*{date}*.SAFE"))
            if safe_files:
                cmd = self.cmd_mgr.sentinel1_tops_cmd(
                    safe_file=safe_files[0],
                    orbit_file=self.paths.orbit_dir,
                    orbit_type="precise",
                    swaths=["IW1", "IW2", "IW3"],
                )
                commands.append(cmd)

        return commands

    def _generate_baseline_commands(self) -> list[Command]:
        """Generate commands for baseline calculation.

        Returns
        -------
        list[Command]
            List of commands.

        """
        # Placeholder - baseline calculation needs to be implemented
        return []

    def _generate_nesd_workflow(self, start_run_num: int) -> None:
        """Generate NESD-based coregistration workflow.

        Parameters
        ----------
        start_run_num : int
            Starting run number.

        """
        # Placeholder - NESD workflow implementation

    def _generate_geometry_workflow(self, start_run_num: int) -> None:
        """Generate geometry-based coregistration workflow.

        Parameters
        ----------
        start_run_num : int
            Starting run number.

        """
        # Placeholder - geometry workflow implementation

    def _generate_valid_region_commands(self) -> list[Command]:
        """Generate commands for extracting valid stack region.

        Returns
        -------
        list[Command]
            List of commands.

        """
        # Placeholder - valid region extraction
        return []

    def _generate_merge_slc_commands(
        self, date: str, azimuth: int, range_looks: int
    ) -> list[Command]:
        """Generate commands for merging SLC.

        Parameters
        ----------
        date : str
            Date string.
        azimuth : int
            Number of azimuth looks.
        range_looks : int
            Number of range looks.

        Returns
        -------
        list[Command]
            List of commands.

        """
        # Placeholder - merge SLC implementation
        return []

    def _generate_burst_igram_commands(
        self, ref_date: str, sec_date: str
    ) -> list[Command]:
        """Generate commands for burst interferogram generation.

        Parameters
        ----------
        ref_date : str
            Reference date.
        sec_date : str
            Secondary date.

        Returns
        -------
        list[Command]
            List of commands.

        """
        # Placeholder - burst interferogram generation
        return []

    def _generate_merge_igram_commands(
        self, ref_date: str, sec_date: str, azimuth: int, range_looks: int
    ) -> list[Command]:
        """Generate commands for merging interferograms.

        Parameters
        ----------
        ref_date : str
            Reference date.
        sec_date : str
            Secondary date.
        azimuth : int
            Number of azimuth looks.
        range_looks : int
            Number of range looks.

        Returns
        -------
        list[Command]
            List of commands.

        """
        # Placeholder - merge interferograms
        return []

    def _generate_filter_commands(
        self, ref_date: str, sec_date: str, azimuth: int, range_looks: int
    ) -> list[Command]:
        """Generate commands for filtering and coherence.

        Parameters
        ----------
        ref_date : str
            Reference date.
        sec_date : str
            Secondary date.
        azimuth : int
            Number of azimuth looks.
        range_looks : int
            Number of range looks.

        Returns
        -------
        list[Command]
            List of commands.

        """
        # Placeholder - filter and coherence
        return []

    def _generate_unwrap_commands(
        self, ref_date: str, sec_date: str, azimuth: int, range_looks: int
    ) -> list[Command]:
        """Generate commands for phase unwrapping.

        Parameters
        ----------
        ref_date : str
            Reference date.
        sec_date : str
            Secondary date.
        azimuth : int
            Number of azimuth looks.
        range_looks : int
            Number of range looks.

        Returns
        -------
        list[Command]
            List of commands.

        """
        # Placeholder - phase unwrapping
        return []

    def generate_run_file(
        self,
        run_num: int,
        description: str,
        commands: list[Command],
    ) -> None:
        """Generate a single run file with specified commands.

        Parameters
        ----------
        run_num : int
            Run file number (for ordering).
        description : str
            Description suffix for the run file name.
        commands : list[Command]
            List of commands to include.

        """
        run_name = f"run_{run_num:02d}_{description}"
        self.cmd_mgr.generate_run_file(run_name, commands)
        logger.info("Generated run file: %s", run_name)

    def generate_config_file(self, command: Command) -> str:
        """Generate a config file for a command.

        Parameters
        ----------
        command : Command
            Command to generate config for.

        Returns
        -------
        str
            Path to the generated config file.

        """
        return self.cmd_mgr._generate_config_file(command)
