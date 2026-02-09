"""Interferogram Stack workflow for ISCE2 processing.

This module provides the interferogram generation workflow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from faninsar.isce2.command_manager import Command
from faninsar.isce2.workflows.slc_stack import SLCStack
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar._core.sar import Acquisition, Pairs
    from faninsar.isce2 import PathManager
    from faninsar.query import BoundingBox

logger = setup_logger(__name__)


class InterferogramStack(SLCStack):
    """Interferogram stack workflow.

    Extends SLCStack with interferogram generation, filtering,
    and phase unwrapping steps.

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
        Reference date for the stack.
    azimuth_looks : int, optional
        Number of azimuth looks. Default is 3.
    range_looks : int, optional
        Number of range looks. Default is 9.
    filter_strength : float, optional
        Filter strength for interferogram filtering. Default is 0.5.
    unw_method : str, optional
        Unwrapping method ('snaphu' or 'icu'). Default is 'snaphu'.
    virtual_merge : bool, optional
        Whether to use virtual files for merged SLCs. Default is True.

    Examples
    --------
    >>> from faninsar.isce2 import PathManager
    >>> pm = PathManager(work_dir="/data/processing", ...)
    >>> workflow = InterferogramStack(
    ...     path_manager=pm,
    ...     num_process=4,
    ...     azimuth_looks=3,
    ...     range_looks=9,
    ... )
    >>> workflow.generate_run_files()
    >>> workflow.execute(parallel=True)

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
        azimuth_looks: int = 3,
        range_looks: int = 9,
        filter_strength: float = 0.5,
        unw_method: Literal["snaphu", "icu"] = "snaphu",
        virtual_merge: bool = True,
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

        self.azimuth_looks = azimuth_looks
        self.range_looks = range_looks
        self.filter_strength = filter_strength
        self.unw_method = unw_method
        self.virtual_merge = virtual_merge

    def generate_run_files(self) -> None:
        """Generate all run files and config files for interferogram stack.

        Generates the following run files:
        - run_01-10: SLC stack (from parent class)
        - run_11: extract_stack_valid_region
        - run_12: merge_reference_secondary_slc
        - run_13: generate_burst_igram
        - run_14: merge_burst_igram
        - run_15: filter_coherence
        - run_16: unwrap

        """
        # First generate SLC stack run files
        super().generate_run_files()

        logger.info("Generating interferogram stack run files")

        run_num = 11

        # Step 11: Extract stack valid region
        self._generate_extract_valid_region(run_num)
        run_num += 1

        # Step 12: Merge reference and secondary SLCs
        self._generate_merge_slc(run_num)
        run_num += 1

        # Step 13: Generate burst interferograms
        self._generate_burst_igram(run_num)
        run_num += 1

        # Step 14: Merge burst interferograms
        self._generate_merge_igram(run_num)
        run_num += 1

        # Step 15: Filter and coherence
        self._generate_filter_coherence(run_num)
        run_num += 1

        # Step 16: Phase unwrapping
        self._generate_unwrap(run_num)

        logger.info("Generated run files 11-16")

    def _generate_extract_valid_region(self, run_num: int) -> None:
        """Generate run file for extracting valid stack region.

        Parameters
        ----------
        run_num : int
            Run file number.

        """
        commands = []

        # TODO: Add extract valid region command
        # This extracts the common valid region across all dates

        run_name = f"run_{run_num:02d}_extract_stack_valid_region"
        self.cmd_mgr.generate_run_file(run_name, commands)

    def _generate_merge_slc(self, run_num: int) -> None:
        """Generate run file for merging SLCs.

        Parameters
        ----------
        run_num : int
            Run file number.

        """
        commands = []

        # Merge reference SLC
        ref_cmd = self.cmd_mgr.merge_bursts_cmd(
            reference=self.paths.reference_path(),
            dirname=self.paths.reference_path(),
            outfile=(
                self.paths.work_dir
                / "merged"
                / "SLC"
                / self.reference_date
                / f"{self.reference_date}.slc"
            ),
            name_pattern="burst*slc",
            method="top",
            valid_only=True,
            use_virtual=self.virtual_merge,
        )
        commands.append(ref_cmd)

        # Merge secondary SLCs
        for date in self.secondary_dates:
            sec_cmd = self.cmd_mgr.merge_bursts_cmd(
                reference=self.paths.coreg_secondary_path(date),
                dirname=self.paths.coreg_secondary_path(date),
                outfile=(self.paths.work_dir / "merged" / "SLC" / date / f"{date}.slc"),
                name_pattern="burst*slc",
                method="top",
                valid_only=True,
                use_virtual=self.virtual_merge,
            )
            commands.append(sec_cmd)

        run_name = f"run_{run_num:02d}_merge_reference_secondary_slc"
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

            # Interferogram directory
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
            )
            commands.append(cmd)

        run_name = f"run_{run_num:02d}_generate_burst_igram"
        self.cmd_mgr.generate_run_file(run_name, commands)

    def _generate_merge_igram(self, run_num: int) -> None:
        """Generate run file for merging burst interferograms.

        Parameters
        ----------
        run_num : int
            Run file number.

        """
        commands = []

        # Merge interferogram for each pair
        for pair in self.pairs:
            ref_date = pair.primary_string()
            sec_date = pair.secondary_string()

            igram_dir = (
                self.paths.work_dir / "interferograms" / f"{ref_date}_{sec_date}"
            )
            merged_dir = (
                self.paths.work_dir
                / "merged"
                / "interferograms"
                / f"{ref_date}_{sec_date}"
            )

            cmd = self.cmd_mgr.merge_bursts_cmd(
                reference=igram_dir,
                dirname=igram_dir,
                outfile=merged_dir / "fine.int",
                name_pattern="fine*int",
                method="top",
                valid_only=True,
                use_virtual=True,
            )
            commands.append(cmd)

        run_name = f"run_{run_num:02d}_merge_burst_igram"
        self.cmd_mgr.generate_run_file(run_name, commands)

    def _generate_filter_coherence(self, run_num: int) -> None:
        """Generate run file for filtering and coherence calculation.

        Parameters
        ----------
        run_num : int
            Run file number.

        """
        commands = []

        # Filter and calculate coherence for each pair
        for pair in self.pairs:
            ref_date = pair.primary_string()
            sec_date = pair.secondary_string()

            merged_dir = (
                self.paths.work_dir
                / "merged"
                / "interferograms"
                / f"{ref_date}_{sec_date}"
            )

            cmd = self.cmd_mgr.filter_coherence_cmd(
                interferogram=merged_dir / "fine.int",
                coherence=merged_dir / "fine.cor",
                filtered_int=merged_dir / "filt_fine.int",
                filter_strength=self.filter_strength,
            )
            commands.append(cmd)

        run_name = f"run_{run_num:02d}_filter_coherence"
        self.cmd_mgr.generate_run_file(run_name, commands)

    def _generate_unwrap(self, run_num: int) -> None:
        """Generate run file for phase unwrapping.

        Parameters
        ----------
        run_num : int
            Run file number.

        """
        commands = []

        # Unwrap each interferogram
        for pair in self.pairs:
            ref_date = pair.primary_string()
            sec_date = pair.secondary_string()

            merged_dir = (
                self.paths.work_dir
                / "merged"
                / "interferograms"
                / f"{ref_date}_{sec_date}"
            )

            cmd = self.cmd_mgr.unwrap_cmd(
                interferogram=merged_dir / "filt_fine.int",
                coherence=merged_dir / "fine.cor",
                unwrapped=merged_dir / "filt_fine.unw",
                reference=self.paths.reference_path(),
                azimuth_looks=self.azimuth_looks,
                range_looks=self.range_looks,
                method=self.unw_method,
            )
            commands.append(cmd)

        run_name = f"run_{run_num:02d}_unwrap"
        self.cmd_mgr.generate_run_file(run_name, commands)
