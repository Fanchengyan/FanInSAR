"""Sentinel-1 SLC extraction executor for ISCE2.

This module provides functionality for extracting Sentinel-1 SLC data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isceobj.Sensor.TOPS.Sentinel1 import Sentinel1

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.isce2.command_manager import Command

logger = setup_logger(__name__)


def _exec_sentinel1_tops(cmd: Command) -> int:
    """Execute Sentinel1_TOPS processing.

    Parameters
    ----------
    cmd : Command
        Sentinel1_TOPS command.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).

    """
    try:
        obj = Sentinel1()
        obj.configure()
        obj.safeFile = str(cmd.params["safe_file"])
        obj.outputDir = str(cmd.params["outdir"])
        obj.swaths = cmd.params["swaths"]
        obj.polarization = cmd.params["polarization"]

        # Set orbit
        if cmd.params["orbit_type"] == "precise":
            obj.orbitDir = str(cmd.params["orbit_file"].parent)
        else:
            obj.orbit = str(cmd.params["orbit_file"])

        if cmd.params.get("bbox"):
            obj.bbox = cmd.params["bbox"]

        obj.extractImage()
    except Exception as e:
        logger.error("Sentinel1_TOPS execution failed: %s", e)
        return 1
    else:
        logger.info("Sentinel1_TOPS completed successfully")
        return 0
