"""Executors for ISCE2 commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.isce2.command_manager import Command

logger = setup_logger(__name__)


def execute_command(cmd: Command) -> int:  # noqa: PLR0911
    """Execute a single ISCE2 command.

    Parameters
    ----------
    cmd : Command
        Command to execute.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).

    Notes
    -----
    This function dispatches commands to appropriate executor functions
    based on command type.

    """
    from faninsar.isce2.executors.coregistration import _exec_resamp
    from faninsar.isce2.executors.geometry import _exec_geo2rdr, _exec_topo
    from faninsar.isce2.executors.interferogram import (
        _exec_filter_coherence,
        _exec_generate_igram,
    )
    from faninsar.isce2.executors.merge import _exec_merge_bursts, _exec_multilook
    from faninsar.isce2.executors.postprocess import _exec_geocode, _exec_unwrap
    from faninsar.isce2.executors.sentinel1 import _exec_sentinel1_tops

    cmd_name = cmd.cmd_name

    if cmd_name == "Sentinel1_TOPS":
        return _exec_sentinel1_tops(cmd)
    if cmd_name == "topo":
        return _exec_topo(cmd)
    if cmd_name == "geo2rdr":
        return _exec_geo2rdr(cmd)
    if cmd_name == "multilook":
        return _exec_multilook(cmd)
    if cmd_name == "resamp":
        return _exec_resamp(cmd)
    if cmd_name == "generate_igram":
        return _exec_generate_igram(cmd)
    if cmd_name == "filter_coherence":
        return _exec_filter_coherence(cmd)
    if cmd_name == "merge_bursts":
        return _exec_merge_bursts(cmd)
    if cmd_name == "geocode":
        return _exec_geocode(cmd)
    if cmd_name == "unwrap":
        return _exec_unwrap(cmd)

    logger.error("Unknown command type: %s", cmd_name)
    return 1
