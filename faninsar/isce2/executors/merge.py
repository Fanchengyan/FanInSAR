"""Burst merging and multilooking executors for ISCE2.

This module provides functionality for merging burst products and multilooking.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import TYPE_CHECKING

from isceobj.Util.ImageUtil import ImageLib as IML
from mroipac.looks.Looks import Looks

from faninsar.isce2.isce_utils import get_swath_list, load_product
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.isce2.command_manager import Command

logger = setup_logger(__name__)


def _exec_multilook(cmd: Command) -> int:
    """Execute multilook processing using mroipac.looks.

    This function uses the Looks class from mroipac to perform multilooking
    on ISCE2 images.

    Parameters
    ----------
    cmd : Command
        Multilook command with parameters:
        - input: Path to input image
        - output: Path to output image
        - azimuth: Number of azimuth looks
        - range_looks: Number of range looks

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).

    """
    try:
        # Use Path objects internally
        input_path = Path(cmd.params["input"])
        output_path = Path(cmd.params["output"])
        azimuth_looks = int(cmd.params["azimuth"])
        range_looks = int(cmd.params["range_looks"])

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Multilooking: %s -> %s (az=%d, rg=%d)",
            input_path,
            output_path,
            azimuth_looks,
            range_looks,
        )

        # Create Looks object (ISCE2 requires string paths)
        looks_obj = Looks()
        looks_obj.setInputFilename(str(input_path))
        looks_obj.setOutputFilename(str(output_path))
        looks_obj.setNumberAzimuthLooks(azimuth_looks)
        looks_obj.setNumberRangeLooks(range_looks)

        # Execute multilooking
        looks_obj.looks()

        logger.info("Multilook completed successfully")
        return 0
    except Exception:
        logger.exception("Multilook execution failed")
        return 1


def _exec_merge_bursts(cmd: Command) -> int:
    """Execute burst merging.

    Parameters
    ----------
    cmd : Command
        MergeBursts command with parameters:
        - reference: Path to reference directory
        - dirname: Directory with burst products to merge
        - outfile: Output merged file
        - method: Merge method ('top', 'bot', 'avg')
        - name_pattern: Pattern for burst files
        - valid_only: Whether to merge only valid regions
        - use_virtual: Whether to create VRT files

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).

    """
    try:
        # Extract parameters
        reference_dir = Path(cmd.params["reference"])
        dirname = Path(cmd.params["dirname"])
        outfile = Path(cmd.params["outfile"])
        method = str(cmd.params.get("method", "avg"))
        name_pattern = str(cmd.params.get("name_pattern", "fine*int"))
        valid_only = bool(cmd.params.get("valid_only", True))
        use_virtual = bool(cmd.params.get("use_virtual", False))

        logger.info("Merging bursts from: %s", dirname)
        logger.info("Output file: %s", outfile)
        logger.info("Method: %s", method)
        logger.info("Pattern: %s", name_pattern)

        # Ensure output directory exists
        outfile.parent.mkdir(parents=True, exist_ok=True)

        # Get swath list
        swath_list = get_swath_list(reference_dir)

        # Load reference products for each swath
        reference_frames = []
        for swath in swath_list:
            ref_prod = load_product(reference_dir / f"IW{swath}.xml")
            reference_frames.append(ref_prod)

        # Find burst files to merge
        file_list = sorted(glob.glob(str(dirname / name_pattern)))

        if not file_list:
            logger.warning("No files found matching pattern: %s", name_pattern)
            return 1

        logger.info("Found %d files to merge", len(file_list))

        # Use VRT for virtual merging or actual merging
        if use_virtual:
            # Virtual merge using VRT
            logger.info("Creating virtual merge (VRT)")
            # TODO: Implement VRT merging logic
            logger.warning("VRT merging not yet implemented")
            return 1
        # Physical merge using ImageLib
        logger.info("Creating physical merge")

        # Merge using ISCE ImageLib merge functionality
        # This is a simplified version - full implementation would need
        # to handle swath boundaries, valid regions, etc.
        IML.mergeSwaths(
            reference_frames,
            file_list,
            str(outfile),
            method=method,
            validOnly=valid_only,
        )

        logger.info("Burst merging completed successfully")
        return 0
    except Exception:
        logger.exception("Burst merging failed")
        return 1
