"""SLC Stack workflow for ISCE2 processing.

This module provides the SLC coregistration stack workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from faninsar.isce2.workflows.base import BaseWorkflow
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar._core.sar import Acquisition, Pairs
    from faninsar.isce2 import PathManager
    from faninsar.query import BoundingBox

logger = setup_logger(__name__)


class SLCStack(BaseWorkflow):
    """SLC coregistration stack workflow.

    Implements the SLC stack processing workflow including:
    - Unpack reference scene and topo (run_01)
    - Unpack secondary SLCs (run_02)
    - Average baseline (run_03)
    - NESD/geometry coregistration (run_04-10)

    Parameters
    ----------
    path_manager : PathManager
        Path manager instance.
    num_process : int, optional
        Number of parallel processes. Default is 1.
    use_gpu : bool, optional
        Whether to enable GPU acceleration. Default is False.
    text_cmd : str, optional
        Command prefix for backward compatibility. Default is "".
    acquisitions : Acquisition | None, optional
        Specified acquisition date subset.
    pairs : Pairs | None, optional
        Specified interferometric pair subset.
    bbox : BoundingBox | None, optional
        Spatial extent filter.
    coreg_method : str, optional
        Coregistration method ('NESD' or 'geometry'). Default is 'NESD'.
    reference_date : str | None, optional
        Reference date for the stack. If None, uses first date.
    num_overlap_connections : int, optional
        Number of overlap connections for NESD. Default is 3.
    esd_coherence_threshold : float, optional
        ESD coherence threshold. Default is 0.85.
    snr_threshold : float, optional
        SNR threshold for range misregistration. Default is 10.0.

    Attributes
    ----------
    coreg_method : str
        Coregistration method.
    reference_date : str
        Reference date for the stack.
    secondary_dates : list[str]
        List of secondary dates.

    Examples
    --------
    >>> from faninsar.isce2 import PathManager
    >>> pm = PathManager(work_dir="/data/processing", slc_dir="/data/SLC", ...)
    >>> workflow = SLCStack(path_manager=pm, num_process=4, coreg_method="NESD")
    >>> workflow.generate_run_files()

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
        coreg_method: Literal["NESD", "geometry"] = "NESD",
        reference_date: str | None = None,
        num_overlap_connections: int = 3,
        esd_coherence_threshold: float = 0.85,
        snr_threshold: float = 10.0,
    ) -> None:
        """Initialize the SLCStack workflow."""
        super().__init__(
            path_manager=path_manager,
            num_process=num_process,
            use_gpu=use_gpu,
            text_cmd=text_cmd,
            acquisitions=acquisitions,
            pairs=pairs,
            bbox=bbox,
        )

        # Validate coregistration method
        if coreg_method not in ("NESD", "geometry"):
            msg = f"Invalid coregistration method: {coreg_method}"
            raise ValueError(msg)

        self.coreg_method = coreg_method
        self.num_overlap_connections = num_overlap_connections
        self.esd_coherence_threshold = esd_coherence_threshold
        self.snr_threshold = snr_threshold

        # Set reference date
        if reference_date is None:
            if len(self.acquisitions) > 0:
                self.reference_date = self.acquisitions[0].strftime("%Y%m%d")
            else:
                self.reference_date = ""
        else:
            self.reference_date = reference_date

        # Set secondary dates
        self.secondary_dates = self._get_secondary_dates()

    def _get_secondary_dates(self) -> list[str]:
        """Get list of secondary dates.

        Returns
        -------
        list[str]
            List of secondary dates in YYYYMMDD format.

        """
        dates = []
        for acq in self.acquisitions:
            date_str = acq.strftime("%Y%m%d")
            if date_str != self.reference_date:
                dates.append(date_str)
        return sorted(dates)

    def generate_run_files(self) -> None:
        """Generate all run files and config files for SLC stack.

        Generates the following run files:
        - run_01: unpack_topo_reference
        - run_02: unpack_secondary_slc
        - run_03: average_baseline
        - run_04-10: coregistration workflow (NESD or geometry)

        """
        logger.info("Generating SLC stack run files")
        logger.info("Reference date: %s", self.reference_date)
        logger.info("Secondary dates: %d", len(self.secondary_dates))

        # Create run directory
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        self.paths.config_dir.mkdir(parents=True, exist_ok=True)

        run_num = 1

        # Step 1: Unpack reference scene and topo
        self._generate_unpack_topo_reference(run_num)
        run_num += 1

        # Step 2: Unpack secondary SLCs
        self._generate_unpack_secondary_slc(run_num)
        run_num += 1

        # Step 3: Average baseline
        self._generate_average_baseline(run_num)
        run_num += 1

        # Steps 4-10: Coregistration workflow
        if self.coreg_method == "NESD":
            run_num = self._generate_nesd_workflow(run_num)
        else:
            run_num = self._generate_geometry_workflow(run_num)

        logger.info("Generated run files 01-%02d", run_num - 1)

    def _generate_unpack_topo_reference(self, run_num: int) -> None:
        """Generate run file for unpacking reference scene and topo.

        Parameters
        ----------
        run_num : int
            Run file number.

        """
        commands = []

        # Find reference SAFE file
        safe_files = self._find_safe_files(self.reference_date)
        if not safe_files:
            logger.warning(
                "No SAFE file found for reference date: %s", self.reference_date
            )

        for safe_file in safe_files:
            # Get orbit file
            orbit_file = self._get_orbit_file(safe_file)
            orbit_path = orbit_file or self.paths.orbit_dir or Path()

            # Sentinel1_TOPS command
            cmd = self.cmd_mgr.sentinel1_tops_cmd(
                safe_file=safe_file,
                orbit_file=orbit_path,
                orbit_type="precise",
                swaths=["IW1", "IW2", "IW3"],
            )
            commands.append(cmd)

        # Topo command
        try:
            cmd = self.cmd_mgr.topo_cmd()
            commands.append(cmd)
        except ValueError as e:
            logger.warning("Could not create topo command: %s", e)

        # Generate run file
        run_name = f"run_{run_num:02d}_unpack_topo_reference"
        self.cmd_mgr.generate_run_file(run_name, commands)

    def _generate_unpack_secondary_slc(self, run_num: int) -> None:
        """Generate run file for unpacking secondary SLCs.

        Parameters
        ----------
        run_num : int
            Run file number.

        """
        commands = []

        for date in self.secondary_dates:
            safe_files = self._find_safe_files(date)
            for safe_file in safe_files:
                orbit_file = self._get_orbit_file(safe_file)
                orbit_path = orbit_file or self.paths.orbit_dir or Path()

                cmd = self.cmd_mgr.sentinel1_tops_cmd(
                    safe_file=safe_file,
                    orbit_file=orbit_path,
                    orbit_type="precise",
                    swaths=["IW1", "IW2", "IW3"],
                )
                commands.append(cmd)

        run_name = f"run_{run_num:02d}_unpack_secondary_slc"
        self.cmd_mgr.generate_run_file(run_name, commands)

    def _generate_average_baseline(self, run_num: int) -> None:
        """Generate run file for average baseline calculation.

        Parameters
        ----------
        run_num : int
            Run file number.

        """
        commands = []

        # Baseline command for each secondary date
        for date in self.secondary_dates:
            cmd = self.cmd_mgr.baseline_cmd(
                reference=self.paths.reference_path(),
                secondary=self.paths.secondary_path(date),
                baseline_file=(
                    self.paths.work_dir
                    / "baselines"
                    / f"{self.reference_date}_{date}"
                    / f"{self.reference_date}_{date}.txt"
                ),
            )
            commands.append(cmd)

        run_name = f"run_{run_num:02d}_average_baseline"
        self.cmd_mgr.generate_run_file(run_name, commands)

    def _generate_nesd_workflow(self, start_run: int) -> int:
        """Generate NESD coregistration workflow.

        Parameters
        ----------
        start_run : int
            Starting run number.

        Returns
        -------
        int
            Next run number after NESD workflow.

        """
        run_num = start_run

        # Step 4: Extract burst overlaps
        run_name = f"run_{run_num:02d}_extract_burst_overlaps"
        self.cmd_mgr.generate_run_file(run_name, [])
        run_num += 1

        # Step 5: Overlap geo2rdr
        run_name = f"run_{run_num:02d}_overlap_geo2rdr"
        commands = []
        for date in self.secondary_dates:
            cmd = self.cmd_mgr.geo2rdr_cmd(date=date, overlap=True)
            commands.append(cmd)
        self.cmd_mgr.generate_run_file(run_name, commands)
        run_num += 1

        # Step 6: Overlap resample
        run_name = f"run_{run_num:02d}_overlap_resample"
        commands = []
        for date in self.secondary_dates:
            cmd = self.cmd_mgr.resamp_cmd(
                reference=self.paths.reference_path(),
                secondary=self.paths.secondary_path(date),
                coreg_dir=self.paths.coreg_secondary_path(date),
                overlap=True,
            )
            commands.append(cmd)
        self.cmd_mgr.generate_run_file(run_name, commands)
        run_num += 1

        # Step 7: Pairs misregistration
        run_name = f"run_{run_num:02d}_pairs_misreg"
        self.cmd_mgr.generate_run_file(run_name, [])
        run_num += 1

        # Step 8: Timeseries misregistration
        run_name = f"run_{run_num:02d}_timeseries_misreg"
        self.cmd_mgr.generate_run_file(run_name, [])
        run_num += 1

        # Step 9: Full burst geo2rdr
        run_name = f"run_{run_num:02d}_fullBurst_geo2rdr"
        commands = []
        for date in self.secondary_dates:
            cmd = self.cmd_mgr.geo2rdr_cmd(date=date, overlap=False)
            commands.append(cmd)
        self.cmd_mgr.generate_run_file(run_name, commands)
        run_num += 1

        # Step 10: Full burst resample
        run_name = f"run_{run_num:02d}_fullBurst_resample"
        commands = []
        for date in self.secondary_dates:
            misreg_az = (
                self.paths.work_dir / "misreg" / "azimuth" / "dates" / f"{date}.txt"
            )
            misreg_rng = (
                self.paths.work_dir / "misreg" / "range" / "dates" / f"{date}.txt"
            )
            cmd = self.cmd_mgr.resamp_cmd(
                reference=self.paths.reference_path(),
                secondary=self.paths.secondary_path(date),
                coreg_dir=self.paths.coreg_secondary_path(date),
                overlap=False,
                misreg_az=misreg_az,
                misreg_rng=misreg_rng,
            )
            commands.append(cmd)
        self.cmd_mgr.generate_run_file(run_name, commands)
        run_num += 1

        return run_num

    def _generate_geometry_workflow(self, start_run: int) -> int:
        """Generate geometry-based coregistration workflow.

        Parameters
        ----------
        start_run : int
            Starting run number.

        Returns
        -------
        int
            Next run number after geometry workflow.

        """
        run_num = start_run

        # Geometry workflow is simpler - just geo2rdr and resample
        # Step 4: geo2rdr
        run_name = f"run_{run_num:02d}_geo2rdr"
        commands = []
        for date in self.secondary_dates:
            cmd = self.cmd_mgr.geo2rdr_cmd(date=date, overlap=False)
            commands.append(cmd)
        self.cmd_mgr.generate_run_file(run_name, commands)
        run_num += 1

        # Step 5: resample
        run_name = f"run_{run_num:02d}_resample"
        commands = []
        for date in self.secondary_dates:
            cmd = self.cmd_mgr.resamp_cmd(
                reference=self.paths.reference_path(),
                secondary=self.paths.secondary_path(date),
                coreg_dir=self.paths.coreg_secondary_path(date),
                overlap=False,
            )
            commands.append(cmd)
        self.cmd_mgr.generate_run_file(run_name, commands)
        run_num += 1

        return run_num

    def _find_safe_files(self, date: str) -> list[Path]:
        """Find SAFE files for a specific date.

        Parameters
        ----------
        date : str
            Date in YYYYMMDD format.

        Returns
        -------
        list[Path]
            List of SAFE file paths.

        """
        slc_dir = self.paths.slc_dir
        if slc_dir is None:
            logger.warning("slc_dir not set")
            return []

        safe_files: list[Path] = []

        # Find .SAFE directories
        safe_files.extend(slc_dir.glob(f"S1*_IW_SLC_*{date}*.SAFE"))

        # Find .zip files
        safe_files.extend(slc_dir.glob(f"S1*_IW_SLC_*{date}*.zip"))

        logger.info("Found %d SAFE files for date %s", len(safe_files), date)
        return sorted(safe_files)

    def _get_orbit_file(self, safe_file: Path) -> Path | None:
        """Get orbit file for a SAFE file.

        Parameters
        ----------
        safe_file : Path
            Path to the SAFE file.

        Returns
        -------
        Path | None
            Path to the orbit file, or None if not found.

        """
        if self.paths.orbit_dir is None:
            logger.warning("orbit_dir not set, skipping orbit file lookup")
            return None

        from faninsar.isce2.orbit_finder import OrbitFinder

        finder = OrbitFinder(self.paths.orbit_dir)
        return finder.find_orbit(safe_file, orbit_type="auto")
