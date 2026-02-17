"""Command manager for ISCE2 stack processing.

This module provides a centralized command management system for ISCE2 processing,
allowing direct command execution without generating intermediate files.
"""

from __future__ import annotations

import re
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
    suffix : str, optional
        Optional suffix for config file naming (e.g., date string like '20211128').

    Attributes
    ----------
    cmd_name : str
        Name of the command.
    params : dict[str, object]
        Command parameters.
    suffix : str
        Suffix for config file naming.

    Examples
    --------
    >>> cmd = Command(
    ...     cmd_name="Sentinel1_TOPS",
    ...     params={"safe_file": "/data/S1A.safe"},
    ...     suffix="20211128",
    ... )

    """

    cmd_name: str
    params: dict[str, object]
    suffix: str = ""


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
        self._config_sequence_by_cmd: dict[str, int] = {}

    def sentinel1_tops_cmd(
        self,
        safe_file: str | Path,
        orbit_file: str | Path,
        orbit_type: str,
        swaths: list[str],
        polarization: str = "vv",
        bbox: list[float] | None = None,
        outdir: str | Path | None = None,
        suffix: str = "",
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
        outdir : str | Path | None, optional
            Output directory for unpacked products. If None, use
            ``paths.reference_path()``. Default is None.
        suffix : str, optional
            Suffix for config file naming. Default is "".
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
                "outdir": Path(outdir)
                if outdir is not None
                else self.paths.reference_path(),
                "swaths": swaths,
                "polarization": polarization,
                "bbox": bbox,
                **kwargs,
            },
            suffix=suffix,
        )

    def pairs_misreg_cmd(
        self,
        reference_dir: str | Path,
        secondary_dir: str | Path,
        interferogram_dir: str | Path,
        overlap_dir: str | Path,
        out_azimuth: str | Path,
        out_range: str | Path,
        coh_threshold: float = 0.85,
        snr_threshold: float = 10.0,
        suffix: str = "",
    ) -> Command:
        """Build a compound misregistration command.

        Parameters
        ----------
        reference_dir : str | Path
            Path to reference burst directory.
        secondary_dir : str | Path
            Path to secondary burst directory.
        interferogram_dir : str | Path
            Output coarse interferogram directory.
        overlap_dir : str | Path
            Output overlap directory.
        out_azimuth : str | Path
            Output azimuth misregistration file.
        out_range : str | Path
            Output range misregistration file.
        coh_threshold : float, optional
            Coherence threshold for azimuth misregistration. Default is 0.85.
        snr_threshold : float, optional
            SNR threshold for range misregistration. Default is 10.0.
        suffix : str, optional
            Suffix for config file naming. Default is "".

        Returns
        -------
        Command
            Compound misregistration command object.

        """
        reference_path = Path(reference_dir)
        secondary_path = Path(secondary_dir)
        interferogram_path = Path(interferogram_dir)
        return Command(
            cmd_name="pairs_misreg",
            params={
                "reference": reference_path,
                "secondary": secondary_path,
                "interferogram_dir": interferogram_path,
                "overlap_dir": Path(overlap_dir),
                "out_azimuth": Path(out_azimuth),
                "out_range": Path(out_range),
                "coh_threshold": coh_threshold,
                "snr_threshold": snr_threshold,
            },
            suffix=suffix,
        )

    def shell_cmd(self, command_line: str, suffix: str = "") -> Command:
        """Build a shell command for direct run-file execution.

        Parameters
        ----------
        command_line : str
            Command line content to write directly into run file.
        suffix : str, optional
            Suffix for command identification. Default is "".

        Returns
        -------
        Command
            Shell command object.

        """
        return Command(
            cmd_name="shell",
            params={"command_line": command_line},
            suffix=suffix,
        )

    def topo_cmd(self, num_process: int | None = None, suffix: str = "") -> Command:
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
            suffix=suffix,
        )

    def geo2rdr_cmd(
        self,
        date: str,
        overlap: bool = False,
        misreg_az: str | None = None,
        misreg_rng: str | None = None,
        suffix: str = "",
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
            suffix=suffix,
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
        suffix: str = "",
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
            suffix=suffix,
        )

    def generate_igram_cmd(
        self,
        reference: str | Path,
        secondary: str | Path,
        coreg_dir: str | Path,
        overlap: bool = False,
        suffix: str = "",
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
            suffix=suffix,
        )

    def filter_coherence_cmd(
        self,
        interferogram: str | Path,
        coherence: str | Path,
        filtered_int: str | Path,
        filter_strength: float = 0.5,
        slc1: str | Path | None = None,
        slc2: str | Path | None = None,
        complex_coh: str | Path | None = None,
        azimuth_looks: int = 1,
        range_looks: int = 1,
        suffix: str = "",
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
        suffix : str, optional
            Suffix for config file naming (e.g., date string). Default is "".

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
        ...     suffix="20240101_20240113",
        ... )

        """
        return Command(
            cmd_name="filter_coherence",
            params={
                "interferogram": Path(interferogram),
                "coherence": Path(coherence),
                "filtered_int": Path(filtered_int),
                "filter_strength": filter_strength,
                "slc1": Path(slc1) if slc1 is not None else None,
                "slc2": Path(slc2) if slc2 is not None else None,
                "complex_coh": Path(complex_coh) if complex_coh is not None else None,
                "azimuth_looks": azimuth_looks,
                "range_looks": range_looks,
            },
            suffix=suffix,
        )

    def merge_bursts_cmd(
        self,
        reference: str | Path,
        dirname: str | Path,
        outfile: str | Path,
        method: str = "avg",
        name_pattern: str = "fine*int",
        stack: str | Path | None = None,
        aligned: bool = True,
        valid_only: bool = True,
        use_virtual: bool = False,
        multilook: bool = True,
        azimuth_looks: int = 1,
        range_looks: int = 1,
        multilook_tool: str | None = None,
        no_data_value: str | float | None = None,
        suffix: str = "",
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
        azimuth_looks : int, optional
            Number of azimuth looks for multilooking. Default is 1.
        range_looks : int, optional
            Number of range looks for multilooking. Default is 1.
        suffix : str, optional
            Suffix for config file naming (e.g., date string). Default is "".

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
        ...     azimuth_looks=3,
        ...     range_looks=9,
        ...     suffix="20240101_20240113",
        ... )

        """
        return Command(
            cmd_name="merge_bursts",
            params={
                "stack": Path(stack) if stack is not None else None,
                "reference": Path(reference),
                "dirname": Path(dirname),
                "outfile": Path(outfile),
                "method": method,
                "name_pattern": name_pattern,
                "aligned": aligned,
                "valid_only": valid_only,
                "use_virtual": use_virtual,
                "multilook": multilook,
                "azimuth_looks": azimuth_looks,
                "range_looks": range_looks,
                "multilook_tool": multilook_tool,
                "no_data_value": no_data_value,
            },
            suffix=suffix,
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
        suffix: str = "",
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
        suffix : str, optional
            Suffix for config file naming (e.g., date string). Default is "".

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
        ...     suffix="20240101_20240113",
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
            suffix=suffix,
        )

    def baseline_cmd(
        self,
        reference: str | Path,
        secondary: str | Path,
        baseline_file: str | Path,
        suffix: str = "",
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
            suffix=suffix,
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
                    logger.debug(
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
            logger.debug(
                "Command %s completed with result: %s",
                cmd.cmd_name,
                result,
            )

    def generate_run_file(
        self,
        run_name: str,
        commands: list[Command],
        parallelize: bool = True,
    ) -> None:
        """Generate a run file for backwards compatibility.

        Parameters
        ----------
        run_name : str
            Name of the run file.
        commands : list[Command]
            List of commands to include in the run file.
        parallelize : bool, optional
            Whether to emit background execution markers for command-level
            parallelism. Default is True.

        Examples
        --------
        >>> cmd_mgr.generate_run_file("run_stack.txt", [cmd1, cmd2, cmd3])

        """
        normalized_run_name = (
            run_name if Path(run_name).suffix == ".sh" else f"{run_name}.sh"
        )
        run_path = self.paths.run_file_path(normalized_run_name)

        with run_path.open("w") as f:
            # Add POSIX sh shebang
            f.write("#!/bin/sh\n\n")

            for i, cmd in enumerate(commands):
                if cmd.cmd_name == "shell":
                    command_line = cmd.params.get("command_line")
                    if not isinstance(command_line, str):
                        logger.error("Shell command line is invalid: %s", command_line)
                        msg = "Shell command line must be a string"
                        raise ValueError(msg)
                    cmd_line = f"{self.text_cmd}{command_line}"
                else:
                    config_path = self._generate_config_file(cmd)
                    cmd_line = f"{self.text_cmd}SentinelWrapper.py -c {config_path}"

                if not parallelize:
                    f.write(cmd_line + "\n")
                    continue

                # Parallel processing markers
                is_last = i == len(commands) - 1
                batch_boundary = (i + 1) % self.num_process == 0
                should_background = (
                    self.num_process > 1 and not batch_boundary and not is_last
                )

                if should_background:
                    f.write(cmd_line + " &\n")
                else:
                    f.write(cmd_line + "\n")
                    if self.num_process > 1 and (batch_boundary or is_last):
                        f.write("wait\n\n")

        # Make the run file executable
        run_path.chmod(0o755)

        logger.debug("Writing %s", run_path)

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

        # Generate deterministic config name with suffix if provided
        suffix = self._sanitize_config_suffix(cmd.suffix)
        base_name = self._config_base_name(cmd.cmd_name)
        if suffix:
            config_name = f"{base_name}_{suffix}"
        else:
            sequence = self._config_sequence_by_cmd.get(cmd.cmd_name, 0) + 1
            self._config_sequence_by_cmd[cmd.cmd_name] = sequence
            config_name = f"{base_name}_{sequence:04d}"
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
        elif cmd.cmd_name == "pairs_misreg":
            self._write_pairs_misreg_config(config, cmd)
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

    @staticmethod
    def _sanitize_config_suffix(suffix: str) -> str:
        """Sanitize config suffix to be filesystem-safe.

        Parameters
        ----------
        suffix : str
            Raw suffix string.

        Returns
        -------
        str
            Sanitized suffix string.

        """
        if not suffix:
            return ""
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", suffix).strip("_.")
        return sanitized

    def _write_pairs_misreg_config(self, config: ConfigWriter, cmd: Command) -> None:
        """Write compound misregistration config sections.

        Parameters
        ----------
        config : ConfigWriter
            Configuration file writer instance.
        cmd : Command
            Compound misregistration command.

        """
        params = cmd.params
        interferogram_dir = Path(params["interferogram_dir"])
        config.write_generate_igram({
            "reference": str(params["reference"]),
            "secondary": str(params["secondary"]),
            "interferogram": str(interferogram_dir),
            "flatten": "False",
            "prefix": "int",
            "overlap": "True",
        })
        config.write_overlap_withdem({
            "interferogram": str(interferogram_dir / "coarse_ifg"),
            "reference_dir": str(params["reference"]),
            "secondary_dir": str(params["secondary"]),
            "overlap_dir": str(params["overlap_dir"]),
        })
        config.write_azimuth_misreg({
            "overlap_dir": str(params["overlap_dir"]),
            "out_azimuth": str(params["out_azimuth"]),
            "coh_threshold": str(params["coh_threshold"]),
            "plot": "False",
        })
        config.write_range_misreg({
            "reference": str(params["reference"]),
            "secondary": str(params["secondary"]),
            "out_range": str(params["out_range"]),
            "snr_threshold": str(params["snr_threshold"]),
        })

    @staticmethod
    def _config_base_name(cmd_name: str) -> str:
        """Map command names to legacy-compatible config prefixes.

        Parameters
        ----------
        cmd_name : str
            Internal command name.

        Returns
        -------
        str
            Config filename base prefix.

        """
        mapping = {
            "merge_bursts": "config_merge",
            "generate_igram": "config_generate_igram",
            "filter_coherence": "config_igram_filt_coh",
            "unwrap": "config_igram_unw",
            "Sentinel1_TOPS": "config_unpack",
            "pairs_misreg": "config_misreg",
        }
        mapped = mapping.get(cmd_name)
        if mapped is not None:
            return mapped
        return f"config_{cmd_name}"

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
            if len(bbox) == 4:
                west, south, east, north = bbox
                bbox_str = f"{south} {north} {west} {east}"
            else:
                logger.error("Invalid bbox length: %s", len(bbox))
                msg = "bbox must have 4 elements: [west, south, east, north]"
                raise ValueError(msg)
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
        slc1 = params.get("slc1")
        slc2 = params.get("slc2")
        complex_coh = params.get("complex_coh")
        return {
            "input": str(params["interferogram"]),
            "filt": str(params["filtered_int"]),
            "coh": str(params["coherence"]),
            "strength": str(params.get("filter_strength", 0.5)),
            "slc1": str(slc1) if slc1 else "",
            "slc2": str(slc2) if slc2 else "",
            "complex_coh": str(complex_coh) if complex_coh else "",
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
