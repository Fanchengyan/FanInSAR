"""Logging utilities for faninsar."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import colorlog
from tqdm import tqdm

if TYPE_CHECKING:
    from os import PathLike

__all__ = [
    "SUCCESS",  # Add SUCCESS log level to __all__
    "color_formatter",
    "formatter",
    "setup_logger",
    "stream_handler",
    "tqdm_handler",
]

# Custom log levels
SUCCESS = 25  # Between INFO and WARNING

# Add SUCCESS level to logging
logging.addLevelName(SUCCESS, "SUCCESS")


# Add success method to Logger class
def _success(self: logging.Logger, message: object, *args: Any, **kwargs: Any) -> None:
    """Log a message with SUCCESS level."""
    if self.isEnabledFor(SUCCESS):
        self._log(SUCCESS, message, args, **kwargs)


logging.Logger.success = _success

# formatters
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
color_formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "white",
        "SUCCESS": "green",
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
    log_file: str | PathLike[str] | None = None,
    handler: logging.Handler | list[logging.Handler] = stream_handler,
    level: int = logging.DEBUG,
    propagate: bool = True,
) -> logging.Logger:
    """Set up logging for the faninsar module.

    Parameters
    ----------
    name : str, optional
        Name of the logger, by default "faninsar"
    log_file : str | PathLike[str] | None, optional
        If provided, also log to this file, by default None
    handler : logging.Handler | list[logging.Handler], optional
        Logging handler to use, by default stream_handler
    level : int, optional
        Logging level for all handlers, by default logging.DEBUG
    propagate : bool, optional
        Whether to propagate messages to parent loggers, by default True

    Examples
    --------
    default setup:

    >>> from faninsar.logging import setup_logger
    >>> logger = setup_logger(__name__)

    working with tqdm:

    >>> from faninsar.logging import setup_logger, tqdm_handler
    >>> logger = setup_logger(__name__, handler=tqdm_handler)

    logging to a file:

    >>> from faninsar.logging import setup_logger
    >>> logger = setup_logger(
    ...     __name__,
    ...     level=logging.INFO,
    ...     log_file="logfile.log",
    ... )


    Returns
    -------
    logging.Logger
        Configured logger instance.

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = propagate

    # Convert handler to a list for uniform handling
    handlers = [handler] if not isinstance(handler, list) else handler.copy()

    # Set level for all handlers
    for h in handlers:
        h.setLevel(level)

    # Add file handler if log_file is provided
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Add all handlers to the logger
    for h in handlers:
        logger.addHandler(h)

    return logger
