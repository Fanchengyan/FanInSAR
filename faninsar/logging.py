"""Logging utilities for faninsar."""

from __future__ import annotations

import logging

import colorlog
from tqdm import tqdm

__all__ = [
    "color_formatter",
    "formatter",
    "setup_logger",
    "stream_handler",
    "tqdm_handler",
]

# formatters
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%d-%d %H:%M:%S",
)
color_formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%d-%d %H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "white",
        "SUCCESS:": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
)

# handlers
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(color_formatter)


class _TqdmLoggingHandler(logging.StreamHandler):
    """A logging handler that works with tqdm."""

    def __init__(self, tqdm_class: type[tqdm] = tqdm) -> None:
        """Initialize the tqdm logging handler."""
        super().__init__()
        self.tqdm_class = tqdm_class

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record."""
        try:
            msg = self.format(record)
            self.tqdm_class.write(msg, file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


tqdm_handler = _TqdmLoggingHandler()
tqdm_handler.setLevel(logging.INFO)
tqdm_handler.setFormatter(color_formatter)


def setup_logger(
    name: str = "faninsar",
    handler: logging.Handler | list[logging.Handler] = stream_handler,
) -> logging.Logger:
    """Set up logging for the faninsar module.

    Examples
    --------
    default setup:

    >>> from faninsar.logging import setup_logger
    >>> logger = setup_logger(__name__)

    working with tqdm:

    >>> from faninsar.logging import setup_logger, tqdm_handler
    >>> logger = setup_logger(__name__, handler=tqdm_handler)

    Parameters
    ----------
    name : str, optional
        Name of the logger, by default "faninsar"
    handler : logging.Handler, optional
        Logging handler to use, by default stream_handler

    Returns
    -------
    logging.Logger
        Configured logger instance.

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if isinstance(handler, list):
        for h in handler:
            logger.addHandler(h)
    else:
        logger.addHandler(handler)
    return logger
