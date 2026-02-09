"""Command manager for ISCE2 stack processing.

This module provides a centralized command management system for ISCE2 processing,
allowing direct command execution without generating intermediate files.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from faninsar.isce2.executors import execute_command
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.isce2.path_manager import PathManager

logger = setup_logger(__name__)


@dataclass
class Command:
    """Command dataclass for ISCE2 processing.

    Parameters
    ----------
    cmd_name : str
        Name of the command (e.g., 'Sentinel1_TOPS', 'topo', 'geo2rdr', 'multilook').
    params : dict[str, object]
        Dictionary of command parameters.

    Attributes
    ----------
    cmd_name : str
        Name of the command.
    params : dict[str, object]
        Command parameters.

    Examples
    --------
    >>> cmd = Command(
    ...     cmd_name="Sentinel1_TOPS",
    ...     params={"safe_file": "/data/S1A.safe"},
    ... )

    """

    cmd_name: str
    params: dict[str, object]


class TopsStackCommands:
    """Command management for ISCE2 processing.

    This class manages command building, queuing, and execution for ISCE2 processing
    workflows. It provides an alternative to generating run/config files by directly
    constructing and executing commands. PathManager is used for all path resolution.

    Parameters
    ----------
    path_manager : PathManager
        PathManager instance for path resolution.
    num_process : int, optional
        Number of parallel processes to use. Default is 1.
    text_cmd : str, optional
        Command prefix for backwards compatibility with run files. Default is "".
    use_gpu : bool, optional
        Whether to enable GPU acceleration for supported commands. Default is False.

    Attributes
    ----------
    paths : PathManager
        PathManager instance for path resolution.
    num_process : int
        Number of parallel processes.
    text_cmd : str
        Command prefix.
    use_gpu : bool
        GPU acceleration flag.
    command_queue : list[list[Command]]
        Queue of command batches.
    current_batch : list[Command]
        Current batch of commands being collected.

    Examples
    --------
    >>> from faninsar.isce2 import PathManager, TopsStackCommands
    >>> pm = PathManager(
    ...     work_dir="/data/processing", slc_dir="/data/SLC", dem="/data/dem.tif"
    ... )
    >>> cmd_mgr = TopsStackCommands(path_manager=pm, num_process=4, use_gpu=True)
    >>> cmd1 = cmd_mgr.sentinel1_tops_cmd(
    ...     safe_file="/data/S1A_IW_SLC__1SDV_20240101/...",
    ...     orbit_file="/data/orbits/...",
    ...     orbit_type="precise",
    ... )
    >>> cmd_mgr.add_command(cmd1)
    >>> cmd_mgr.execute_all(parallel=True)

    Notes
    -----
    The TopsStackCommands supports both direct command execution (for efficiency)
    and backwards-compatible run/config file generation for legacy workflows.

    """

    def __init__(
        self,
        path_manager: PathManager,
        num_process: int = 1,
        text_cmd: str = "",
        use_gpu: bool = False,
    ) -> None:
        """Initialize the TopsStackCommands."""
        self.paths = path_manager
        self.num_process = num_process
        self.text_cmd = text_cmd
        self.use_gpu = use_gpu

        self.command_queue: list[list[Command]] = []
        self.current_batch: list[Command] = []

    def sentinel1_tops_cmd(
        self,
        safe_file: str | Path,
        orbit_file: str | Path,
        orbit_type: str,
        swaths: list[str],
        polarization: str = "vv",
        bbox: list[float] | None = None,
        **kwargs: object,
    ) -> Command:
        """Build a Sentinel1_TOPS command.

        Paths are resolved using PathManager:
        - Output directory: paths.reference_path()

        Parameters
        ----------
        safe_file : str | Path
            Path to Sentinel-1 SAFE file.
        orbit_file : str | Path
            Path to orbit file or directory.
        orbit_type : str
            Type of orbit file ('precise' or 'restituted').
        swaths : list[str]
            List of swaths to process (e.g., ['IW1', 'IW2', 'IW3']).
        polarization : str, optional
            Polarization to extract. Default is 'vv'.
        bbox : list[float] | None, optional
            Bounding box [west, south, east, north]. Default is None.
        **kwargs : object
            Additional keyword arguments.

        Returns
        -------
        Command
            Sentinel1_TOPS command object.

        Examples
        --------
        >>> cmd = cmd_mgr.sentinel1_tops_cmd(
        ...     safe_file="/data/S1A_IW_SLC__1SDV_20240101/...",
        ...     orbit_file="/data/orbits/...",
        ...     orbit_type="precise",
        ...     swaths=["IW1", "IW2", "IW3"],
        ... )

        """
        return Command(
            cmd_name="Sentinel1_TOPS",
            params={
                "safe_file": Path(safe_file),
                "orbit_file": Path(orbit_file),
                "orbit_type": orbit_type,
                "outdir": self.paths.reference_path(),
                "swaths": swaths,
                "polarization": polarization,
                "bbox": bbox,
                **kwargs,
            },
        )

    def topo_cmd(self, num_process: int | None = None) -> Command:
        """Build a topo command.

        Paths are resolved using PathManager:
        - Reference scene: paths.reference_path()
        - DEM: paths.dem
        - Geometry directory: paths.geom_reference_path()

        Parameters
        ----------
        num_process : int | None, optional
            Number of processes for parallel execution. Default is None.

        Returns
        -------
        Command
            Topo command object.

        Examples
        --------
        >>> cmd = cmd_mgr.topo_cmd()

        """
        if self.paths.dem is None:
            msg = "DEM path is not set in PathManager"
            logger.error(msg)
            raise ValueError(msg)

        return Command(
            cmd_name="topo",
            params={
                "reference": self.paths.reference_path(),
                "dem": self.paths.dem,
                "geom_dir": self.paths.geom_reference_path(),
                "num_process": num_process or self.num_process,
            },
        )

    def geo2rdr_cmd(
        self,
        date: str,
        overlap: bool = False,
        misreg_az: str | None = None,
        misreg_rng: str | None = None,
    ) -> Command:
        """Build a geo2rdr command.

        Paths are resolved using PathManager:
        - Secondary scene: paths.secondary_path(date)
        - Reference scene: paths.reference_path()
        - Geometry reference: paths.geom_reference_path()
        - Coregistered directory: paths.coreg_secondary_path(date)

        Parameters
        ----------
        date : str
            Date string (format: YYYYMMDD) for the secondary scene.
        overlap : bool, optional
            Whether to use overlap method. Default is False.
        misreg_az : str | None, optional
            Misregistration in azimuth. Default is None.
        misreg_rng : str | None, optional
            Misregistration in range. Default is None.

        Returns
        -------
        Command
            Geo2rdr command object.

        Examples
        --------
        >>> cmd = cmd_mgr.geo2rdr_cmd(date="20240113")

        """
        return Command(
            cmd_name="geo2rdr",
            params={
                "secondary": self.paths.secondary_path(date),
                "reference": self.paths.reference_path(),
                "geom_reference": self.paths.geom_reference_path(),
                "coreg_dir": self.paths.coreg_secondary_path(date),
                "overlap": overlap,
                "misreg_az": misreg_az,
                "misreg_rng": misreg_rng,
                "use_gpu": self.use_gpu,
            },
        )

    def multilook_cmd(
        self,
        input_path: str | Path,
        azimuth: int,
        range_looks: int,
        multilook_dir: str | Path | None = None,
    ) -> Command:
        """Build a multilook command.

        Parameters
        ----------
        input_path : str | Path
            Path to input image.
        azimuth : int
            Number of azimuth looks.
        range_looks : int
            Number of range looks.
        multilook_dir : str | Path | None, optional
            Output multilook directory. If None, output path is derived from input path
            by appending the look numbers. Default is None.

        Returns
        -------
        Command
            Multilook command object.

        Examples
        --------
        >>> cmd = cmd_mgr.multilook_cmd(
        ...     input_path="/data/input.tif",
        ...     azimuth=2,
        ...     range_looks=8,
        ... )

        """
        input_path = Path(input_path)

        if multilook_dir is None:
            # Derive output path from input path
            multilook_dir = self.paths.multilook_path(azimuth, range_looks)

        multilook_dir = Path(multilook_dir)

        # Build output path: same structure as input but in multilook directory
        relative_path = (
            input_path.relative_to(input_path.parents[0])
            if input_path.is_absolute()
            else input_path
        )
        output_path = multilook_dir / relative_path

        return Command(
            cmd_name="multilook",
            params={
                "input": input_path,
                "output": output_path,
                "azimuth": azimuth,
                "range_looks": range_looks,
            },
        )

    def resamp_cmd(
        self,
        reference: str | Path,
        secondary: str | Path,
        coreg_dir: str | Path,
        misreg_az: str | Path | None = None,
        misreg_rng: str | Path | None = None,
        flatten: bool = True,
        overlap: bool = False,
    ) -> Command:
        """Build a resamp_withCarrier command.

        Paths are resolved using PathManager:
        - Reference scene: paths.reference_path()
        - Secondary scene: paths.secondary_path(date)
        - Coregistered directory: paths.coreg_secondary_path(date)

        Parameters
        ----------
        reference : str | Path
            Path to reference directory.
        secondary : str | Path
            Path to secondary directory.
        coreg_dir : str | Path
            Path to coregistered output directory.
        misreg_az : str | Path | None, optional
            Path to azimuth misregistration file. Default is None.
        misreg_rng : str | Path | None, optional
            Path to range misregistration file. Default is None.
        flatten : bool, optional
            Whether to flatten the interferogram. Default is True.
        overlap : bool, optional
            Whether this is an overlap burst. Default is False.

        Returns
        -------
        Command
            Resamp command object.

        Examples
        --------
        >>> cmd = cmd_mgr.resamp_cmd(
        ...     reference=pm.reference_path(),
        ...     secondary=pm.secondary_path("20240113"),
        ...     coreg_dir=pm.coreg_secondary_path("20240113"),
        ... )

        """
        return Command(
            cmd_name="resamp",
            params={
                "reference": Path(reference),
                "secondary": Path(secondary),
                "coreg_dir": Path(coreg_dir),
                "misreg_az": Path(misreg_az) if misreg_az else None,
                "misreg_rng": Path(misreg_rng) if misreg_rng else None,
                "flatten": flatten,
                "overlap": overlap,
            },
        )

    def generate_igram_cmd(
        self,
        reference: str | Path,
        secondary: str | Path,
        coreg_dir: str | Path,
        overlap: bool = False,
    ) -> Command:
        """Build a generateIgram command for burst interferogram generation.

        Parameters
        ----------
        reference : str | Path
            Path to reference burst directory.
        secondary : str | Path
            Path to secondary burst directory.
        coreg_dir : str | Path
            Path to coregistered directory containing resampled SLC.
        overlap : bool, optional
            Whether this is an overlap burst. Default is False.

        Returns
        -------
        Command
            GenerateIgram command object.

        Examples
        --------
        >>> cmd = cmd_mgr.generate_igram_cmd(
        ...     reference=pm.reference_path(),
        ...     secondary=pm.secondary_path("20240113"),
        ...     coreg_dir=pm.coreg_secondary_path("20240113"),
        ... )

        """
        return Command(
            cmd_name="generate_igram",
            params={
                "reference": Path(reference),
                "secondary": Path(secondary),
                "coreg_dir": Path(coreg_dir),
                "overlap": overlap,
            },
        )

    def filter_coherence_cmd(
        self,
        interferogram: str | Path,
        coherence: str | Path,
        filtered_int: str | Path,
        filter_strength: float = 0.5,
    ) -> Command:
        """Build a FilterAndCoherence command.

        Parameters
        ----------
        interferogram : str | Path
            Path to input interferogram.
        coherence : str | Path
            Path to output coherence file.
        filtered_int : str | Path
            Path to output filtered interferogram.
        filter_strength : float, optional
            Filter strength parameter (0.0-1.0). Default is 0.5.

        Returns
        -------
        Command
            FilterAndCoherence command object.

        Examples
        --------
        >>> cmd = cmd_mgr.filter_coherence_cmd(
        ...     interferogram="burst.int",
        ...     coherence="burst.cor",
        ...     filtered_int="filt_burst.int",
        ...     filter_strength=0.5,
        ... )

        """
        return Command(
            cmd_name="filter_coherence",
            params={
                "interferogram": Path(interferogram),
                "coherence": Path(coherence),
                "filtered_int": Path(filtered_int),
                "filter_strength": filter_strength,
            },
        )

    def merge_bursts_cmd(
        self,
        reference: str | Path,
        dirname: str | Path,
        outfile: str | Path,
        method: str = "avg",
        name_pattern: str = "fine*int",
        valid_only: bool = True,
        use_virtual: bool = False,
    ) -> Command:
        """Build a mergeBursts command.

        Parameters
        ----------
        reference : str | Path
            Path to reference directory.
        dirname : str | Path
            Directory with burst products to merge.
        outfile : str | Path
            Output merged file path.
        method : str, optional
            Merge method: 'top', 'bot', or 'avg'. Default is 'avg'.
        name_pattern : str, optional
            Pattern for burst products to merge. Default is 'fine*int'.
        valid_only : bool, optional
            Whether to merge only valid regions. Default is True.
        use_virtual : bool, optional
            Whether to create virtual (VRT) files instead of real files.
            Default is False.

        Returns
        -------
        Command
            MergeBursts command object.

        Examples
        --------
        >>> cmd = cmd_mgr.merge_bursts_cmd(
        ...     reference=pm.reference_path(),
        ...     dirname="interferograms/20240101_20240113",
        ...     outfile="merged/interferogram.int",
        ...     method="avg",
        ... )

        """
        return Command(
            cmd_name="merge_bursts",
            params={
                "reference": Path(reference),
                "dirname": Path(dirname),
                "outfile": Path(outfile),
                "method": method,
                "name_pattern": name_pattern,
                "valid_only": valid_only,
                "use_virtual": use_virtual,
            },
        )

    def geocode_cmd(
        self,
        lat_file: str | Path,
        lon_file: str | Path,
        input_file: str | Path,
        output_file: str | Path,
        bbox: list[float] | None = None,
        lat_step: float = 0.001,
        lon_step: float = 0.001,
        method: str = "near",
    ) -> Command:
        """Build a geocode command using GDAL.

        Parameters
        ----------
        lat_file : str | Path
            Path to latitude file in radar coordinates.
        lon_file : str | Path
            Path to longitude file in radar coordinates.
        input_file : str | Path
            Path to input file to be geocoded.
        output_file : str | Path
            Path to output geocoded file.
        bbox : list[float] | None, optional
            Bounding box [south, north, west, east]. Default is None.
        lat_step : float, optional
            Output pixel size in latitude (degrees). Default is 0.001.
        lon_step : float, optional
            Output pixel size in longitude (degrees). Default is 0.001.
        method : str, optional
            Resampling method for gdalwarp. Default is 'near'.

        Returns
        -------
        Command
            Geocode command object.

        Examples
        --------
        >>> cmd = cmd_mgr.geocode_cmd(
        ...     lat_file="geom_reference/lat.rdr",
        ...     lon_file="geom_reference/lon.rdr",
        ...     input_file="merged/interferogram.int",
        ...     output_file="geocoded/interferogram.geo",
        ...     bbox=[19, 20, -99.5, -98.5],
        ... )

        """
        return Command(
            cmd_name="geocode",
            params={
                "lat_file": Path(lat_file),
                "lon_file": Path(lon_file),
                "input_file": Path(input_file),
                "output_file": Path(output_file),
                "bbox": bbox,
                "lat_step": lat_step,
                "lon_step": lon_step,
                "method": method,
            },
        )

    def unwrap_cmd(
        self,
        interferogram: str | Path,
        coherence: str | Path,
        unwrapped: str | Path,
        reference: str | Path,
        azimuth_looks: int = 1,
        range_looks: int = 1,
        defo_max: float = 2.0,
        method: str = "snaphu",
        nomcf: bool = False,
    ) -> Command:
        """Build an unwrap command.

        Parameters
        ----------
        interferogram : str | Path
            Path to input interferogram file.
        coherence : str | Path
            Path to coherence file.
        unwrapped : str | Path
            Path to output unwrapped phase file.
        reference : str | Path
            Path to reference directory for metadata.
        azimuth_looks : int, optional
            Number of azimuth looks. Default is 1.
        range_looks : int, optional
            Number of range looks. Default is 1.
        defo_max : float, optional
            Maximum deformation in cycles. Default is 2.0.
        method : str, optional
            Unwrapping method ('snaphu' or 'icu'). Default is 'snaphu'.
        nomcf : bool, optional
            Run full snaphu instead of MCF mode. Default is False.

        Returns
        -------
        Command
            Unwrap command object.

        Examples
        --------
        >>> cmd = cmd_mgr.unwrap_cmd(
        ...     interferogram="filt_interferogram.int",
        ...     coherence="coherence.cor",
        ...     unwrapped="unwrapped.unw",
        ...     reference=pm.reference_path(),
        ...     azimuth_looks=2,
        ...     range_looks=8,
        ... )

        """
        return Command(
            cmd_name="unwrap",
            params={
                "interferogram": Path(interferogram),
                "coherence": Path(coherence),
                "unwrapped": Path(unwrapped),
                "reference": Path(reference),
                "azimuth_looks": azimuth_looks,
                "range_looks": range_looks,
                "defo_max": defo_max,
                "method": method,
                "nomcf": nomcf,
            },
        )

    def baseline_cmd(
        self,
        reference: str | Path,
        secondary: str | Path,
        baseline_file: str | Path,
    ) -> Command:
        """Build a baseline computation command.

        Parameters
        ----------
        reference : str | Path
            Path to reference directory.
        secondary : str | Path
            Path to secondary directory.
        baseline_file : str | Path
            Path to output baseline file.

        Returns
        -------
        Command
            Baseline command object.

        Examples
        --------
        >>> cmd = cmd_mgr.baseline_cmd(
        ...     reference=pm.reference_path(),
        ...     secondary=pm.secondary_path("20240113"),
        ...     baseline_file=pm.baseline_path("20240113"),
        ... )

        """
        return Command(
            cmd_name="baseline",
            params={
                "reference": Path(reference),
                "secondary": Path(secondary),
                "baseline_file": Path(baseline_file),
            },
        )

    def baseline_grid_cmd(
        self,
        reference: str | Path,
        secondary: str | Path,
        baseline_file: str | Path,
    ) -> Command:
        """Build a baseline grid computation command.

        Parameters
        ----------
        reference : str | Path
            Path to reference directory.
        secondary : str | Path
            Path to secondary directory.
        baseline_file : str | Path
            Path to output baseline file.

        Returns
        -------
        Command
            Baseline grid command object.

        """
        return Command(
            cmd_name="baseline_grid",
            params={
                "reference": Path(reference),
                "secondary": Path(secondary),
                "baseline_file": Path(baseline_file),
            },
        )

    def overlap_withdem_cmd(
        self,
        interferogram: str | Path,
        reference_dir: str | Path,
        secondary_dir: str | Path,
        overlap_dir: str | Path,
    ) -> Command:
        """Build an overlap_withDEM command.

        Parameters
        ----------
        interferogram : str | Path
            Path to interferogram directory.
        reference_dir : str | Path
            Path to reference directory.
        secondary_dir : str | Path
            Path to secondary directory.
        overlap_dir : str | Path
            Path to overlap output directory.

        Returns
        -------
        Command
            Overlap_withDEM command object.

        """
        return Command(
            cmd_name="overlap_withdem",
            params={
                "interferogram": Path(interferogram),
                "reference_dir": Path(reference_dir),
                "secondary_dir": Path(secondary_dir),
                "overlap_dir": Path(overlap_dir),
            },
        )

    def azimuth_misreg_cmd(
        self,
        overlap_dir: str | Path,
        out_azimuth: str | Path,
        coh_threshold: float = 0.85,
        plot: bool = False,
    ) -> Command:
        """Build an azimuth misregistration estimation command.

        Parameters
        ----------
        overlap_dir : str | Path
            Path to overlap directory.
        out_azimuth : str | Path
            Path to output azimuth misregistration file.
        coh_threshold : float, optional
            Coherence threshold. Default is 0.85.
        plot : bool, optional
            Whether to plot results. Default is False.

        Returns
        -------
        Command
            Azimuth misregistration command object.

        """
        return Command(
            cmd_name="azimuth_misreg",
            params={
                "overlap_dir": Path(overlap_dir),
                "out_azimuth": Path(out_azimuth),
                "coh_threshold": coh_threshold,
                "plot": plot,
            },
        )

    def range_misreg_cmd(
        self,
        reference: str | Path,
        secondary: str | Path,
        out_range: str | Path,
        snr_threshold: float = 10.0,
    ) -> Command:
        """Build a range misregistration estimation command.

        Parameters
        ----------
        reference : str | Path
            Path to reference directory.
        secondary : str | Path
            Path to secondary directory.
        out_range : str | Path
            Path to output range misregistration file.
        snr_threshold : float, optional
            SNR threshold. Default is 10.0.

        Returns
        -------
        Command
            Range misregistration command object.

        """
        return Command(
            cmd_name="range_misreg",
            params={
                "reference": Path(reference),
                "secondary": Path(secondary),
                "out_range": Path(out_range),
                "snr_threshold": snr_threshold,
            },
        )

    def add_command(self, cmd: Command) -> None:
        """Add a single command to the queue.

        If the batch size reaches num_process, the batch is automatically flushed.

        Parameters
        ----------
        cmd : Command
            Command to add to the queue.

        Examples
        --------
        >>> cmd_mgr.add_command(cmd1)
        >>> cmd_mgr.add_command(cmd2)

        """
        self.current_batch.append(cmd)

        # Trigger batch processing if parallel size reached
        if len(self.current_batch) >= self.num_process:
            self.flush_batch()

    def flush_batch(self) -> None:
        """Flush the current batch of commands to the queue.

        This moves the current batch of commands to the command queue
        and starts a new batch.
        """
        if self.current_batch:
            self.command_queue.append(self.current_batch.copy())
            self.current_batch = []

    def execute_all(self, parallel: bool = True) -> None:
        """Execute all commands in the queue.

        Parameters
        ----------
        parallel : bool, optional
            Whether to execute commands in parallel. Default is True.

        Examples
        --------
        >>> cmd_mgr.execute_all(parallel=True)

        """
        self.flush_batch()  # Ensure last batch is added

        for batch in self.command_queue:
            if parallel and self.num_process > 1:
                self._execute_parallel(batch)
            else:
                self._execute_sequential(batch)

    def _execute_parallel(self, batch: list[Command]) -> None:
        """Execute a batch of commands in parallel.

        Parameters
        ----------
        batch : list[Command]
            List of commands to execute in parallel.

        """
        with ProcessPoolExecutor(max_workers=self.num_process) as executor:
            futures = {executor.submit(execute_command, cmd): cmd for cmd in batch}

            for future in as_completed(futures):
                cmd = futures[future]
                try:
                    result = future.result()
                    logger.info(
                        "Command %s completed with result: %s",
                        cmd.cmd_name,
                        result,
                    )
                except Exception:
                    logger.exception("Command %s failed", cmd.cmd_name)

    def _execute_sequential(self, batch: list[Command]) -> None:
        """Execute a batch of commands sequentially.

        Parameters
        ----------
        batch : list[Command]
            List of commands to execute sequentially.

        """
        for cmd in batch:
            result = execute_command(cmd)
            logger.info(
                "Command %s completed with result: %s",
                cmd.cmd_name,
                result,
            )

    def generate_run_file(self, run_name: str, commands: list[Command]) -> None:
        """Generate a run file for backwards compatibility.

        Parameters
        ----------
        run_name : str
            Name of the run file.
        commands : list[Command]
            List of commands to include in the run file.

        Examples
        --------
        >>> cmd_mgr.generate_run_file("run_stack.txt", [cmd1, cmd2, cmd3])

        """
        run_path = self.paths.run_file_path(run_name)

        with run_path.open("w") as f:
            # Add bash shebang
            f.write("#!/bin/bash\n\n")

            for i, cmd in enumerate(commands):
                config_path = self._generate_config_file(cmd)
                cmd_line = f"{self.text_cmd}SentinelWrapper.py -c {config_path}"

                # Parallel processing markers
                if self.num_process > 1 and (i + 1) % self.num_process != 0:
                    f.write(cmd_line + " &\n")
                else:
                    f.write(cmd_line + "\n")
                    if self.num_process > 1:
                        f.write("wait\n\n")

        # Make the run file executable
        run_path.chmod(0o755)

        logger.info("Writing %s", run_path)

    def _generate_config_file(self, cmd: Command) -> str:
        """Generate a config file for backwards compatibility.

        Parameters
        ----------
        cmd : Command
            Command to generate config file for.

        Returns
        -------
        str
            Path to the generated config file.

        """
        from faninsar.isce2.config_writer import ConfigWriter

        # Generate unique config name
        config_name = f"config_{cmd.cmd_name}_{id(cmd)}"
        config_path = self.paths.config_file_path(config_name).with_suffix(".ini")

        # Create ConfigWriter
        config = ConfigWriter(config_path)

        # Write configuration based on command type
        if cmd.cmd_name == "Sentinel1_TOPS":
            config.write_sentinel1_tops(self._prepare_sentinel1_params(cmd))
        elif cmd.cmd_name == "topo":
            config.write_topo(self._prepare_topo_params(cmd))
        elif cmd.cmd_name == "baseline":
            config.write_baseline(self._prepare_baseline_params(cmd))
        elif cmd.cmd_name == "baseline_grid":
            config.write_baseline_grid(self._prepare_baseline_params(cmd))
        elif cmd.cmd_name == "geo2rdr":
            config.write_geo2rdr(self._prepare_geo2rdr_params(cmd))
        elif cmd.cmd_name == "resamp":
            config.write_resamp_withcarrier(self._prepare_resamp_params(cmd))
        elif cmd.cmd_name == "generate_igram":
            config.write_generate_igram(self._prepare_generate_igram_params(cmd))
        elif cmd.cmd_name == "overlap_withdem":
            config.write_overlap_withdem(self._prepare_overlap_withdem_params(cmd))
        elif cmd.cmd_name == "azimuth_misreg":
            config.write_azimuth_misreg(self._prepare_azimuth_misreg_params(cmd))
        elif cmd.cmd_name == "range_misreg":
            config.write_range_misreg(self._prepare_range_misreg_params(cmd))
        elif cmd.cmd_name == "merge_bursts":
            config.write_merge_bursts(self._prepare_merge_bursts_params(cmd))
        elif cmd.cmd_name == "filter_coherence":
            config.write_filter_coherence(self._prepare_filter_coherence_params(cmd))
        elif cmd.cmd_name == "unwrap":
            config.write_unwrap(self._prepare_unwrap_params(cmd))
        else:
            logger.warning("Unknown command type: %s", cmd.cmd_name)

        config.finalize()
        return str(config_path)

    def _prepare_sentinel1_params(self, cmd: Command) -> dict:
        """Prepare Sentinel1_TOPS configuration parameters."""
        params = cmd.params

        # Handle swaths
        swaths = params.get("swaths", [])
        if isinstance(swaths, list):
            swaths_str = " ".join(str(s).replace("IW", "") for s in swaths)
        else:
            swaths_str = str(swaths)

        # Handle orbit file/dir
        orbit_file = params.get("orbit_file")
        if isinstance(orbit_file, Path):
            orbit_dir = str(orbit_file.parent)
            orbit_file_str = str(orbit_file)
        else:
            orbit_dir = str(params.get("orbit_dir", ""))
            orbit_file_str = str(orbit_file) if orbit_file else ""

        # Handle bbox
        bbox = params.get("bbox")
        if isinstance(bbox, (list, tuple)):
            bbox_str = " ".join(map(str, bbox))
        elif bbox:
            bbox_str = str(bbox)
        else:
            bbox_str = ""

        return {
            "dirname": str(params["safe_file"]),
            "swaths": swaths_str,
            "orbit_type": params.get("orbit_type", "precise"),
            "orbit_dir": orbit_dir,
            "orbit_file": orbit_file_str,
            "outdir": str(params["outdir"]),
            "auxdir": str(self.paths.aux_dir or ""),
            "bbox": bbox_str,
            "pol": params.get("polarization", "vv"),
        }

    def _prepare_topo_params(self, cmd: Command) -> dict:
        """Prepare topo configuration parameters."""
        params = cmd.params
        return {
            "reference": str(params["reference"]),
            "dem": str(params["dem"]),
            "geom_referenceDir": str(params["geom_dir"]),
            "numProcess": params.get("num_process", 1),
        }

    def _prepare_baseline_params(self, cmd: Command) -> dict:
        """Prepare baseline configuration parameters."""
        params = cmd.params
        return {
            "reference": str(params["reference"]),
            "secondary": str(params["secondary"]),
            "baseline_file": str(params["baseline_file"]),
        }

    def _prepare_geo2rdr_params(self, cmd: Command) -> dict:
        """Prepare geo2rdr configuration parameters."""
        params = cmd.params
        return {
            "secondary": str(params["secondary"]),
            "reference": str(params["reference"]),
            "geom_reference": str(params["geom_reference"]),
            "coreg_dir": str(params["coreg_dir"]),
            "overlap": "True" if params.get("overlap") else "False",
            "use_gpu": params.get("use_gpu", False),
            "misreg_az": str(params["misreg_az"]) if params.get("misreg_az") else None,
            "misreg_rng": str(params["misreg_rng"])
            if params.get("misreg_rng")
            else None,
        }

    def _prepare_resamp_params(self, cmd: Command) -> dict:
        """Prepare resamp_withCarrier configuration parameters."""
        params = cmd.params
        return {
            "secondary": str(params["secondary"]),
            "reference": str(params["reference"]),
            "coreg_dir": str(params["coreg_dir"]),
            "overlap": "True" if params.get("overlap") else "False",
            "misreg_az": str(params["misreg_az"]) if params.get("misreg_az") else None,
            "misreg_rng": str(params["misreg_rng"])
            if params.get("misreg_rng")
            else None,
        }

    def _prepare_generate_igram_params(self, cmd: Command) -> dict:
        """Prepare generateIgram configuration parameters."""
        params = cmd.params
        return {
            "reference": str(params["reference"]),
            "secondary": str(params["secondary"]),
            "interferogram": str(
                params.get("interferogram_dir", params.get("coreg_dir", ""))
            ),
            "flatten": "True" if params.get("flatten") else "False",
            "prefix": params.get("prefix", "fine"),
            "overlap": "True" if params.get("overlap") else "False",
            "misreg_az": str(params["misreg_az"]) if params.get("misreg_az") else None,
            "misreg_rng": str(params["misreg_rng"])
            if params.get("misreg_rng")
            else None,
        }

    def _prepare_overlap_withdem_params(self, cmd: Command) -> dict:
        """Prepare overlap_withDEM configuration parameters."""
        params = cmd.params
        return {
            "interferogram": str(params["interferogram"]),
            "reference_dir": str(params["reference_dir"]),
            "secondary_dir": str(params["secondary_dir"]),
            "overlap_dir": str(params["overlap_dir"]),
        }

    def _prepare_azimuth_misreg_params(self, cmd: Command) -> dict:
        """Prepare azimuth misregistration configuration parameters."""
        params = cmd.params
        return {
            "overlap_dir": str(params["overlap_dir"]),
            "out_azimuth": str(params["out_azimuth"]),
            "coh_threshold": str(params.get("coh_threshold", "0.85")),
            "plot": "True" if params.get("plot") else "False",
        }

    def _prepare_range_misreg_params(self, cmd: Command) -> dict:
        """Prepare range misregistration configuration parameters."""
        params = cmd.params
        return {
            "reference": str(params["reference"]),
            "secondary": str(params["secondary"]),
            "out_range": str(params["out_range"]),
            "snr_threshold": str(params.get("snr_threshold", "10.0")),
        }

    def _prepare_merge_bursts_params(self, cmd: Command) -> dict:
        """Prepare mergeBursts configuration parameters."""
        params = cmd.params
        return {
            "stack": str(params["stack"]) if params.get("stack") else None,
            "reference": str(params["reference"]),
            "dirname": str(params["dirname"]),
            "name_pattern": params.get("name_pattern", "fine*int"),
            "outfile": str(params["outfile"]),
            "method": params.get("method", "top"),
            "aligned": "True" if params.get("aligned", True) else "False",
            "valid_only": "True" if params.get("valid_only", True) else "False",
            "use_virtual": "True" if params.get("use_virtual", True) else "False",
            "multilook": "True" if params.get("multilook", True) else "False",
            "range_looks": str(params.get("range_looks", 9)),
            "azimuth_looks": str(params.get("azimuth_looks", 3)),
            "multilook_tool": params.get("multilook_tool"),
            "no_data_value": str(params["no_data_value"])
            if params.get("no_data_value")
            else None,
        }

    def _prepare_filter_coherence_params(self, cmd: Command) -> dict:
        """Prepare FilterAndCoherence configuration parameters."""
        params = cmd.params
        return {
            "input": str(params["interferogram"]),
            "filt": str(params["filtered_int"]),
            "coh": str(params["coherence"]),
            "strength": str(params.get("filter_strength", 0.5)),
            "slc1": str(params.get("slc1", "")),
            "slc2": str(params.get("slc2", "")),
            "complex_coh": str(params.get("complex_coh", "")),
            "range_looks": str(params.get("range_looks", 9)),
            "azimuth_looks": str(params.get("azimuth_looks", 3)),
        }

    def _prepare_unwrap_params(self, cmd: Command) -> dict:
        """Prepare unwrap configuration parameters."""
        params = cmd.params
        return {
            "ifg": str(params["interferogram"]),
            "unw": str(params["unwrapped"]),
            "coh": str(params["coherence"]),
            "nomcf": "True" if params.get("nomcf") else "False",
            "reference": str(params["reference"]),
            "defomax": str(params.get("defo_max", 2)),
            "rlks": str(params.get("range_looks", 9)),
            "alks": str(params.get("azimuth_looks", 3)),
            "rmfilter": "True" if params.get("rmfilter") else "False",
            "method": params.get("method", "snaphu"),
        }
