"""Logger utility."""

from __future__ import annotations

import logging
from logging import _nameToLevel as NAME_TO_LEVEL
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faninsar.typing import LogLevel, PathLike


def setup_logger(
    log_file: PathLike | None = None,
    log_level: LogLevel = "info",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_name: str = "FanInSAR",
    capture_warnings: bool = True,
    keep_handlers: bool = True,
    return_handler: bool = False,
) -> logging.Logger:
    """Setups a logger.

    Parameters
    ----------
    log_file : str, optional
        Log file. If not given, log to sys.stderr. Default is None.
    log_level : str, optional
        Log level, one of ["notset", "debug", "info", "warning", "error", "critical"].
        Default is "info".
    log_format : str, optional
        Log format. Default is "%(asctime)s - %(name)s - %(levelname)s - %(message)s".
    log_name : str, optional
        Log name. Default is "FanInSAR".
    capture_warnings : bool, optional
        Capture warnings. Default is True.
    keep_handlers : bool, optional
        Keep existing handlers. Default is True.
    return_handler : bool, optional
        Return handler. Default is False.

    """
    log_level = NAME_TO_LEVEL[log_level.upper()]

    # create a logger
    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)

    # create a formatter
    formatter = logging.Formatter(log_format)

    def create_handler() -> logging.Handler:
        """Create a handler."""
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.set_name(log_name)
            handler.setLevel(log_level)
            handler.setFormatter(formatter)
        else:
            handler = logging.StreamHandler()
            handler.set_name(log_name)
            handler.setLevel(log_level)
            handler.setFormatter(formatter)
        return handler

    handler_names = [i.name for i in logger.handlers]
    if log_name in handler_names:
        handler = logger.handlers[handler_names.index(log_name)]
        if not keep_handlers:
            logger.removeHandler(handler)
            handler = create_handler()
            logger.addHandler(handler)
    else:
        handler = create_handler()
        logger.addHandler(handler)

    if capture_warnings:
        logging.captureWarnings(True)
        warning_logger = logging.getLogger("py.warnings")
        if warning_logger.handlers and not keep_handlers:
            warning_logger.removeHandler(warning_logger.handlers[0])
            warning_logger.addHandler(handler)

    if return_handler:
        return logger, handler
    return logger
