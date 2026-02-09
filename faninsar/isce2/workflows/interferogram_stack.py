"""Interferogram Stack workflow for ISCE2 processing.

This module provides the interferogram generation workflow with support
for multiple multilook configurations.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import Literal, TypeAlias

from faninsar.isce2.command_manager import Command
from faninsar.isce2.workflows.slc_stack import SLCStack
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar._core.sar import Acquisition, Pairs
    from faninsar.isce2 import PathManager
    from faninsar.isce2.path_manager import Multilook
    from faninsar.query import BoundingBox

logger = setup_logger(__name__)

GeocodeTargetLiteral: TypeAlias = Literal[
    "filt_fine.unw",
    "filt_fine.cor",
    "filt_fine.int",
    "fine.int",
    "fine.cor",
    "hgt.rdr",
    "lat.rdr",
    "lon.rdr",
    "los.rdr",
    "shadowMask.rdr",
    "incLocal.rdr",
]

GeocodeTargetAliasLiteral: TypeAlias = Literal[
    "hgt",
    "lat",
    "lon",
    "los",
    "shadowMask",
    "incLocal",
]

DEFAULT_GEOCODE_TARGETS: tuple[GeocodeTargetLiteral, ...] = (
    "filt_fine.unw",
    "filt_fine.cor",
)

PAIR_GEOCODE_TARGETS: set[str] = {
    "filt_fine.unw",
    "filt_fine.cor",
    "filt_fine.int",
    "fine.int",
    "fine.cor",
}

GEOM_GEOCODE_TARGETS: set[str] = {
    "hgt.rdr",
    "lat.rdr",
    "lon.rdr",
    "los.rdr",
    "shadowMask.rdr",
    "incLocal.rdr",
}

GEOCODE_TARGET_ALIASES: dict[str, GeocodeTargetLiteral] = {
    "hgt": "hgt.rdr",
    "lat": "lat.rdr",
    "lon": "lon.rdr",
    "los": "los.rdr",
    "shadowMask": "shadowMask.rdr",
    "incLocal": "incLocal.rdr",
}


class InterferogramStack(SLCStack):
    """Interferogram stack workflow with multiple multilook support.

    Extends SLCStack with interferogram generation, filtering,
    and phase unwrapping steps. Supports multiple multilook configurations
    through PathManager.

    Parameters
    ----------
    path_manager : PathManager
        Path manager instance. If PathManager has multilook configurations,
        run_12-16 will be generated for each multilook setting.
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
        Reference date for the stack.
    filter_strength : float, optional
        Filter strength for interferogram filtering. Default is 0.5.
    unw_method : str, optional
        Unwrapping method ('snaphu' or 'icu'). Default is 'snaphu'.
    virtual_merge : bool, optional
        Whether to use virtual files for merged SLCs. Default is True.
    geocode_targets : Sequence[GeocodeTargetLiteral | GeocodeTargetAliasLiteral] | None, optional
        Geocoding targets. If None, defaults to
        ``("filt_fine.unw", "filt_fine.cor")``. Supports pair products
        (for each pair) and geometry products (under ``geom_reference``).
        Geometry aliases like ``"hgt"`` and ``"lat"`` are normalized to
        ``"hgt.rdr"`` and ``"lat.rdr"``.

    Attributes
    ----------
    filter_strength : float
        Filter strength for interferogram filtering.
    unw_method : str
        Unwrapping method.
    virtual_merge : bool
        Whether to use virtual files for merged SLCs.
    multilooks : list[Multilook]
        List of multilook configurations from PathManager.

    Examples
    --------
    Single multilook configuration:

    >>> from faninsar.isce2 import PathManager
    >>> pm = PathManager(
    ...     work_dir="/data/processing",
    ...     slc_dir="/data/SLC",
    ...     dem="/data/dem.tif",
    ...     multilook=[(3, 9)],
    ... )
    >>> workflow = InterferogramStack(path_manager=pm, num_process=4)
    >>> workflow.generate_run_files()

    Multiple multilook configurations:

    >>> pm = PathManager(
    ...     work_dir="/data/processing",
    ...     slc_dir="/data/SLC",
    ...     dem="/data/dem.tif",
    ...     multilook=[(1, 4), (2, 8), (3, 12)],
    ... )
    >>> workflow = InterferogramStack(path_manager=pm, num_process=4)
    >>> workflow.generate_run_files()
    # run_01-11: shared SLC processing
    # run_12: generated for each multilook
    # run_13: shared full-resolution burst interferograms
    # run_14-16: generated for each multilook in multilooked/{az}_{rg}/

    Notes
    -----
    When multiple multilook configurations are specified:
    - run_01 to run_11 are shared (SLC coregistration)
    - run_12 is generated for each multilook configuration
    - run_13 is shared and generated once in full resolution
    - run_14 to run_16 are generated for each multilook configuration
    - Output files are organized in multilooked/{az}_{rg}/ directories

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
        filter_strength: float = 0.5,
        unw_method: Literal["snaphu", "icu"] = "snaphu",
        virtual_merge: bool = True,
        geocode_targets: Sequence[
            GeocodeTargetLiteral | GeocodeTargetAliasLiteral
        ]
        | None = None,
    ) -> None:
        """Initialize the InterferogramStack workflow."""
        super().__init__(
            path_manager=path_manager,
            num_process=num_process,
            use_gpu=use_gpu,
            text_cmd=text_cmd,
            acquisitions=acquisitions,
            pairs=pairs,
            bbox=bbox,
            coreg_method=coreg_method,
            reference_date=reference_date,
        )

        self.filter_strength = filter_strength
        self.unw_method = unw_method
        self.virtual_merge = virtual_merge
        self.geocode_targets = self._normalize_geocode_targets(geocode_targets)

        # Get multilook configurations from PathManager
        self.multilooks = path_manager.get_multilooks()

        # If no multilook configured, add a default one
        if not self.multilooks:
            logger.warning(
                "No multilook configuration found in PathManager. "
                "Using default (3, 9). Consider adding multilook settings."
            )
            path_manager.add_multilook(3, 9)
            self.multilooks = path_manager.get_multilooks()

    def generate_run_files(self) -> None:
        """Generate all run files and config files for interferogram stack.

        Generates the following run files:
        - run_01-10: SLC stack (from parent class, shared)
        - run_11: extract_stack_valid_region (shared)
        - run_12: For each multilook configuration:
            - run_12_{az}_{rg}: merge_reference_secondary_slc
        - run_13: Shared full-resolution burst interferograms
            - run_13_generate_burst_igram
        - run_14-16: For each multilook configuration:
            - run_14_{az}_{rg}: merge_burst_igram
            - run_15_{az}_{rg}: filter_coherence
            - run_16_{az}_{rg}: unwrap

        """
        # First generate SLC stack run files (run_01-10)
        super().generate_run_files()

        logger.info("Generating interferogram stack run files")
        logger.info("Multilook configurations: %s", self.multilooks)

        # Step 11: Extract stack valid region (shared)
        self._generate_extract_valid_region(11)

        # Step 12: Merge reference and secondary SLCs per multilook
        for ml in self.multilooks:
            logger.info(
                "Generating run files for multilook: (%d, %d)",
                ml.azimuth,
                ml.range,
            )

            suffix = f"_{ml.azimuth}_{ml.range}"
            self._generate_merge_slc(12, ml, suffix)

        # Step 13: Generate burst interferograms (shared, no multilook)
        self._generate_burst_igram(13)

        # Generate run_14-16 for each multilook configuration
        for ml in self.multilooks:
            logger.info(
                "Generating multilook downstream run files for: (%d, %d)",
                ml.azimuth,
                ml.range,
            )
            self._generate_multilook_downstream_workflow(ml)

        # Step 17: Geocode unwrapped phase and coherence for each multilook
        for ml in self.multilooks:
            self._generate_geocode(17, ml)

        self._generate_workspace_run_all_script()

        logger.info("Generated all interferogram run files")

    def _generate_multilook_downstream_workflow(self, ml: Multilook) -> None:
        """Generate downstream run files for a specific multilook.

        Parameters
        ----------
        ml : Multilook
            Multilook configuration.

        """
        az, rg = ml.azimuth, ml.range
        suffix = f"_{az}_{rg}"

        # Step 14: Merge burst interferograms
        self._generate_merge_igram(14, ml, suffix)

        # Step 15: Filter and coherence
        self._generate_filter_coherence(15, ml, suffix)

        # Step 16: Phase unwrapping
        self._generate_unwrap(16, ml, suffix)

    def _generate_extract_valid_region(self, run_num: int) -> None:
        """Generate run file for extracting valid stack region.

        Parameters
        ----------
        run_num : int
            Run file number.

        """
        commands = []
        cmd = self.cmd_mgr.shell_cmd(
            command_line=(
                "extractCommonValidRegion.py "
                f"-m {self.paths.reference_path()} "
                f"-s {self.paths.work_dir / 'coreg_secondarys'}"
            ),
            suffix="extract_stack_valid_region",
        )
        commands.append(cmd)

        run_name = f"run_{run_num:02d}_extract_stack_valid_region"
        self.cmd_mgr.generate_run_file(run_name, commands)

    def _get_merged_slc_dir(self, ml: Multilook) -> Path:
        """Get the merged SLC output directory for a multilook config.

        Parameters
        ----------
        ml : Multilook
            Multilook configuration.

        Returns
        -------
        Path
            Path to merged SLC directory.

        """
        return self.paths.multilook_merged_path(ml.azimuth, ml.range) / "SLC"

    def _get_merged_igram_dir(
        self, ml: Multilook, ref_date: str, sec_date: str
    ) -> Path:
        """Get the merged interferogram output directory.

        Parameters
        ----------
        ml : Multilook
            Multilook configuration.
        ref_date : str
            Reference date.
        sec_date : str
            Secondary date.

        Returns
        -------
        Path
            Path to merged interferogram directory.

        """
        return (
            self.paths.multilook_merged_path(ml.azimuth, ml.range)
            / "interferograms"
            / f"{ref_date}_{sec_date}"
        )

    def _generate_merge_slc(self, run_num: int, ml: Multilook, suffix: str) -> None:
        """Generate run file for merging SLCs with multilook.

        Parameters
        ----------
        run_num : int
            Run file number.
        ml : Multilook
            Multilook configuration.
        suffix : str
            Suffix for run file name.

        """
        commands = []
        merged_slc_dir = self._get_merged_slc_dir(ml)

        # Merge reference SLC
        ref_cmd = self.cmd_mgr.merge_bursts_cmd(
            stack=self.paths.work_dir / "stack",
            reference=self.paths.reference_path(),
            dirname=self.paths.reference_path(),
            outfile=merged_slc_dir / self.reference_date / f"{self.reference_date}.slc",
            name_pattern="burst*slc",
            method="top",
            aligned=False,
            valid_only=True,
            use_virtual=self.virtual_merge,
            multilook=False,
            azimuth_looks=ml.azimuth,
            range_looks=ml.range,
            suffix=f"{self.reference_date}_{ml.azimuth}_{ml.range}",
        )
        commands.append(ref_cmd)

        # Merge secondary SLCs
        for date in self.secondary_dates:
            sec_cmd = self.cmd_mgr.merge_bursts_cmd(
                stack=self.paths.work_dir / "stack",
                reference=self.paths.coreg_secondary_path(date),
                dirname=self.paths.coreg_secondary_path(date),
                outfile=merged_slc_dir / date / f"{date}.slc",
                name_pattern="burst*slc",
                method="top",
                aligned=True,
                valid_only=True,
                use_virtual=self.virtual_merge,
                multilook=False,
                azimuth_looks=ml.azimuth,
                range_looks=ml.range,
                suffix=f"{date}_{ml.azimuth}_{ml.range}",
            )
            commands.append(sec_cmd)

        run_name = f"run_{run_num:02d}_merge_reference_secondary_slc{suffix}"
        self.cmd_mgr.generate_run_file(run_name, commands)

    def _generate_burst_igram(self, run_num: int) -> None:
        """Generate run file for burst interferogram generation.

        Parameters
        ----------
        run_num : int
            Run file number.
        """
        commands = []

        # Generate interferogram for each pair
        for pair in self.pairs:
            ref_date = pair.primary_string()
            sec_date = pair.secondary_string()

            # Determine the reference and secondary paths
            if ref_date == self.reference_date:
                ref_path = self.paths.reference_path()
            else:
                ref_path = self.paths.coreg_secondary_path(ref_date)

            sec_path = self.paths.coreg_secondary_path(sec_date)

            # Shared interferogram directory (full resolution)
            igram_dir = (
                self.paths.work_dir / "interferograms" / f"{ref_date}_{sec_date}"
            )

            cmd = Command(
                cmd_name="generate_igram",
                params={
                    "reference": ref_path,
                    "secondary": sec_path,
                    "interferogram_dir": igram_dir,
                    "flatten": False,
                    "prefix": "fine",
                    "overlap": False,
                },
                suffix=f"{ref_date}_{sec_date}",
            )
            commands.append(cmd)

        run_name = f"run_{run_num:02d}_generate_burst_igram"
        self.cmd_mgr.generate_run_file(run_name, commands)

    def _generate_merge_igram(self, run_num: int, ml: Multilook, suffix: str) -> None:
        """Generate run file for merging burst interferograms.

        Parameters
        ----------
        run_num : int
            Run file number.
        ml : Multilook
            Multilook configuration.
        suffix : str
            Suffix for run file name.

        """
        commands = []
        self._append_merge_reference_geometry_commands(commands, ml)

        # Merge interferogram for each pair
        for pair in self.pairs:
            ref_date = pair.primary_string()
            sec_date = pair.secondary_string()

            igram_dir = self.paths.work_dir / "interferograms" / f"{ref_date}_{sec_date}"
            merged_dir = self._get_merged_igram_dir(ml, ref_date, sec_date)

            cmd = self.cmd_mgr.merge_bursts_cmd(
                stack=self.paths.work_dir / "stack",
                reference=igram_dir,
                dirname=igram_dir,
                outfile=merged_dir / "fine.int",
                name_pattern="fine*int",
                method="top",
                aligned=True,
                valid_only=True,
                use_virtual=True,
                multilook=True,
                azimuth_looks=ml.azimuth,
                range_looks=ml.range,
                suffix=f"igram_{ref_date}_{sec_date}_{ml.azimuth}_{ml.range}",
            )
            commands.append(cmd)

        run_name = f"run_{run_num:02d}_merge_burst_igram{suffix}"
        self.cmd_mgr.generate_run_file(run_name, commands)

    def _generate_filter_coherence(
        self, run_num: int, ml: Multilook, suffix: str
    ) -> None:
        """Generate run file for filtering and coherence calculation.

        Parameters
        ----------
        run_num : int
            Run file number.
        ml : Multilook
            Multilook configuration.
        suffix : str
            Suffix for run file name.

        """
        commands = []

        # Filter and calculate coherence for each pair
        for pair in self.pairs:
            ref_date = pair.primary_string()
            sec_date = pair.secondary_string()

            merged_dir = self._get_merged_igram_dir(ml, ref_date, sec_date)
            merged_slc_dir = self._get_merged_slc_dir(ml)
            slc1 = merged_slc_dir / ref_date / f"{ref_date}.slc.full"
            slc2 = merged_slc_dir / sec_date / f"{sec_date}.slc.full"

            cmd = self.cmd_mgr.filter_coherence_cmd(
                interferogram=merged_dir / "fine.int",
                coherence=merged_dir / "filt_fine.cor",
                filtered_int=merged_dir / "filt_fine.int",
                filter_strength=self.filter_strength,
                slc1=slc1,
                slc2=slc2,
                complex_coh=merged_dir / "fine.cor",
                azimuth_looks=ml.azimuth,
                range_looks=ml.range,
                suffix=f"{ref_date}_{sec_date}_{ml.azimuth}_{ml.range}",
            )
            commands.append(cmd)

        run_name = f"run_{run_num:02d}_filter_coherence{suffix}"
        self.cmd_mgr.generate_run_file(run_name, commands)

    def _generate_unwrap(self, run_num: int, ml: Multilook, suffix: str) -> None:
        """Generate run file for phase unwrapping.

        Parameters
        ----------
        run_num : int
            Run file number.
        ml : Multilook
            Multilook configuration.
        suffix : str
            Suffix for run file name.

        """
        commands = []

        # Unwrap each interferogram
        for pair in self.pairs:
            ref_date = pair.primary_string()
            sec_date = pair.secondary_string()

            merged_dir = self._get_merged_igram_dir(ml, ref_date, sec_date)

            cmd = self.cmd_mgr.unwrap_cmd(
                interferogram=merged_dir / "filt_fine.int",
                coherence=merged_dir / "filt_fine.cor",
                unwrapped=merged_dir / "filt_fine.unw",
                reference=self.paths.reference_path(),
                azimuth_looks=ml.azimuth,
                range_looks=ml.range,
                method=self.unw_method,
                suffix=f"{ref_date}_{sec_date}_{ml.azimuth}_{ml.range}",
            )
            commands.append(cmd)

        run_name = f"run_{run_num:02d}_unwrap{suffix}"
        self.cmd_mgr.generate_run_file(run_name, commands)

    def _append_merge_reference_geometry_commands(
        self,
        commands: list[Command],
        ml: Multilook,
    ) -> None:
        """Append merge commands for reference geometry products.

        Parameters
        ----------
        commands : list[Command]
            Existing command list to append to.
        ml : Multilook
            Multilook configuration.

        """
        geometry_patterns = [
            "lat*rdr",
            "lon*rdr",
            "los*rdr",
            "hgt*rdr",
            "shadowMask*rdr",
            "incLocal*rdr",
        ]
        multilook_tool = {
            "lat*rdr": "gdal",
            "lon*rdr": "gdal",
            "los*rdr": "gdal",
            "hgt*rdr": "gdal",
            "shadowMask*rdr": "isce",
            "incLocal*rdr": "gdal",
        }
        no_data_values: dict[str, str | None] = {
            "lat*rdr": "0",
            "lon*rdr": "0",
            "los*rdr": "0",
            "hgt*rdr": None,
            "shadowMask*rdr": None,
            "incLocal*rdr": "0",
        }

        for pattern in geometry_patterns:
            stem, ext = pattern.split("*")
            cmd = self.cmd_mgr.merge_bursts_cmd(
                stack=self.paths.work_dir / "stack",
                reference=self.paths.reference_path(),
                dirname=self.paths.geom_reference_path(),
                outfile=(
                    self.paths.multilook_merged_path(ml.azimuth, ml.range)
                    / "geom_reference"
                    / f"{stem}.{ext}"
                ),
                method="top",
                name_pattern=pattern,
                aligned=False,
                valid_only=False,
                use_virtual=self.virtual_merge,
                multilook=True,
                azimuth_looks=ml.azimuth,
                range_looks=ml.range,
                multilook_tool=multilook_tool[pattern],
                no_data_value=no_data_values[pattern],
                suffix=f"{stem}_{ml.azimuth}_{ml.range}",
            )
            commands.append(cmd)

    def _generate_geocode(self, run_num: int, ml: Multilook) -> None:
        """Generate geocode run file for a multilook configuration.

        Parameters
        ----------
        run_num : int
            Run file number.
        ml : Multilook
            Multilook configuration.

        """
        if self.paths.dem is None:
            logger.error("DEM is required for geocoding run generation")
            msg = "DEM is required for geocoding run generation"
            raise ValueError(msg)

        commands: list[Command] = []
        bbox_arg = self._bbox_snwe_arg()
        pair_targets = [
            target for target in self.geocode_targets if target in PAIR_GEOCODE_TARGETS
        ]
        geom_targets = [
            target for target in self.geocode_targets if target in GEOM_GEOCODE_TARGETS
        ]

        if pair_targets:
            for pair in self.pairs:
                ref_date = pair.primary_string()
                sec_date = pair.secondary_string()
                merged_dir = self._get_merged_igram_dir(ml, ref_date, sec_date)
                file_list = " ".join(str(merged_dir / target) for target in pair_targets)

                command_line = (
                    "geocodeIsce.py "
                    f"-f \"{file_list}\" "
                    f"-b \"{bbox_arg}\" "
                    f"-d {self.paths.dem} "
                    f"-m {self._coreg_or_reference_path(ref_date)} "
                    f"-s {self._coreg_or_reference_path(sec_date)} "
                    f"-r {ml.range} "
                    f"-a {ml.azimuth}"
                )
                commands.append(
                    self.cmd_mgr.shell_cmd(
                        command_line=command_line,
                        suffix=f"geocode_{ref_date}_{sec_date}_{ml.azimuth}_{ml.range}",
                    )
                )

        if geom_targets:
            geom_reference_dir = self.paths.multilook_merged_path(ml.azimuth, ml.range) / "geom_reference"
            file_list = " ".join(str(geom_reference_dir / target) for target in geom_targets)
            command_line = (
                "geocodeIsce.py "
                f"-f \"{file_list}\" "
                f"-b \"{bbox_arg}\" "
                f"-d {self.paths.dem} "
                f"-m {self.paths.reference_path()} "
                f"-s {self.paths.reference_path()} "
                f"-r {ml.range} "
                f"-a {ml.azimuth}"
            )
            commands.append(
                self.cmd_mgr.shell_cmd(
                    command_line=command_line,
                    suffix=f"geocode_geom_{ml.azimuth}_{ml.range}",
                )
            )

        if not commands:
            logger.warning(
                "No geocode commands generated for multilook (%d, %d)",
                ml.azimuth,
                ml.range,
            )
            return

        run_name = f"run_{run_num:02d}_geocode_{ml.azimuth}_{ml.range}"
        self.cmd_mgr.generate_run_file(run_name, commands)

    def _bbox_snwe_arg(self) -> str:
        """Get geocode SNWE bounding box argument string.

        Returns
        -------
        str
            Bounding box string in ``south north west east`` order.

        """
        if self.bbox is None:
            logger.warning(
                "Bounding box is not set. Using global SNWE for geocoding: "
                "-90 90 -180 180"
            )
            return "-90 90 -180 180"
        left, bottom, right, top = (float(value) for value in self.bbox)
        return f"{bottom} {top} {left} {right}"

    @staticmethod
    def _normalize_geocode_targets(
        geocode_targets: Sequence[
            GeocodeTargetLiteral | GeocodeTargetAliasLiteral
        ]
        | None,
    ) -> tuple[GeocodeTargetLiteral, ...]:
        """Normalize and validate geocode target list.

        Parameters
        ----------
        geocode_targets : Sequence[GeocodeTargetLiteral | GeocodeTargetAliasLiteral] | None
            Optional geocode target sequence.

        Returns
        -------
        tuple[GeocodeTargetLiteral, ...]
            Normalized, de-duplicated geocode target tuple.

        Raises
        ------
        ValueError
            If the provided target list is empty after normalization.

        """
        if geocode_targets is None:
            return DEFAULT_GEOCODE_TARGETS

        normalized_targets = [
            GEOCODE_TARGET_ALIASES.get(str(target), str(target))
            for target in geocode_targets
        ]

        ordered_targets = list(dict.fromkeys(normalized_targets))
        if not ordered_targets:
            logger.error("geocode_targets must not be empty")
            msg = "geocode_targets must not be empty"
            raise ValueError(msg)

        supported_targets = PAIR_GEOCODE_TARGETS | GEOM_GEOCODE_TARGETS
        invalid_targets = [
            target for target in ordered_targets if target not in supported_targets
        ]
        if invalid_targets:
            logger.error("Unsupported geocode target(s): %s", invalid_targets)
            msg = (
                "Unsupported geocode target(s): "
                f"{', '.join(invalid_targets)}. "
                f"Supported targets: {', '.join(sorted(supported_targets))}"
            )
            raise ValueError(msg)

        return tuple(ordered_targets)
