"""Logging utilities for faninsar.

This module provides enhanced logging functionality with colored output,
tqdm integration, custom SUCCESS log level, and smart default levels.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import colorlog
from tqdm import tqdm
from typing_extensions import Literal

if TYPE_CHECKING:
    from os import PathLike

__all__ = [
    "SUCCESS",
    "FaninsarLogger",
    "LogLevel",
    "color_formatter",
    "formatter",
    "get_default_log_level",
    "setup_logger",
    "stream_handler",
    "tqdm_handler",
]

# Custom log levels
SUCCESS: Literal[25] = 25  # Between INFO and WARNING

# Type aliases
LogLevel = Literal["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
HandlerType = logging.Handler | list[logging.Handler]

# Register SUCCESS level name
logging.addLevelName(SUCCESS, "SUCCESS")


class FaninsarLogger(logging.Logger):
    """Enhanced logger with SUCCESS level support.

    This logger extends the standard logging.Logger with a custom
    SUCCESS level (25) that sits between INFO and WARNING.
    """

    def success(self, message: object, *args: Any, **kwargs: Any) -> None:
        """Log a message with SUCCESS level.

        Parameters
        ----------
        message : object
            The message to log.
        *args : Any
            Positional arguments for message formatting.
        **kwargs : Any
            Keyword arguments for logging.

        """
        if self.isEnabledFor(SUCCESS):
            self._log(SUCCESS, message, args, **kwargs)


# Set the custom logger class globally so logging.getLogger returns it
logging.setLoggerClass(FaninsarLogger)

# Formatters
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

# Handlers
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(color_formatter)


class TqdmLoggingHandler(logging.StreamHandler[Any]):
    """A logging handler that works with tqdm progress bars.

    This handler ensures log messages don't interfere with tqdm
    progress bar display by using tqdm.write().
    """

    def __init__(self, tqdm_class: type[tqdm] = tqdm) -> None:
        """Initialize the tqdm logging handler.

        Parameters
        ----------
        tqdm_class : type[tqdm], optional
            The tqdm class to use for writing, by default tqdm.

        """
        super().__init__()
        self.tqdm_class = tqdm_class

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record using tqdm.write()."""
        try:
            msg = self.format(record)
            self.tqdm_class.write(msg, file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:  # pragma: no cover - logging safety net
            self.handleError(record)


tqdm_handler = TqdmLoggingHandler()
tqdm_handler.setLevel(logging.INFO)
tqdm_handler.setFormatter(color_formatter)


def get_default_log_level() -> int:
    """Get default log level based on environment.

    Order of precedence:
    1) FANINSAR_LOG_LEVEL environment variable (DEBUG/INFO/...)
    2) FANINSAR_DEBUG=1 (or true/yes/on)
    3) Heuristic dev detection (DEBUG/ENV/interactive/tests)
    4) Fallback: INFO
    """
    # Explicit level override
    log_level_str = os.getenv("FANINSAR_LOG_LEVEL", "").upper()
    if log_level_str in {"DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}:
        # Map SUCCESS to INFO threshold for enablement
        return getattr(logging, log_level_str if log_level_str != "SUCCESS" else "INFO")

    # Debug flag
    if os.getenv("FANINSAR_DEBUG", "0").lower() in {"1", "true", "yes", "on"}:
        return logging.DEBUG

    # Heuristic dev detection
    dev_indicators = [
        os.getenv("ENVIRONMENT") in ("dev", "development", "local"),
        os.getenv("ENV") in ("dev", "development", "local"),
        os.getenv("DEBUG", "0").lower() in {"1", "true", "yes", "on"},
        sys.argv and sys.argv[0].endswith(("pytest", "python", "ipython")),
        any("test" in arg for arg in sys.argv),
        hasattr(sys, "ps1"),  # Interactive
    ]
    if any(dev_indicators):
        return logging.DEBUG

    # Default production-like
    return logging.INFO


def setup_logger(
    name: str | None = None,
    file: str | PathLike[str] | None = None,
    *,
    # Backward-compatible aliases
    log_name: str | None = None,
    log_file: str | PathLike[str] | None = None,
    handler: HandlerType = stream_handler,
    level: int | None = None,
    propagate: bool = True,
    clear_existing: bool = False,
) -> FaninsarLogger:
    """Create and configure a logger.

    Parameters
    ----------
    name : str | None, optional
        Logger name. If None, uses "faninsar".
    log_name : str | None, optional
        Alias for ``name``; if provided and ``name`` is None, this value is used.
    file : str | PathLike[str] | None, optional
        If provided, also log to this file.
    log_file : str | PathLike[str] | None, optional
        Alias for ``file``; if provided and ``file`` is None, this value is used.
    handler : logging.Handler | list[logging.Handler], optional
        Logging handler(s) to add, by default ``stream_handler``.
    level : int | None, optional
        Logging level for all handlers. If None, uses ``get_default_log_level()``.
    propagate : bool, optional
        Whether to propagate to parent loggers, by default True.
    clear_existing : bool, optional
        Whether to clear existing handlers on the logger, by default False.

    Returns
    -------
    FaninsarLogger
        Configured logger instance with SUCCESS level support.

    """
    # Resolve aliases
    if name is None:
        name = log_name or "faninsar"
    if file is None and log_file is not None:
        file = log_file

    # Determine level
    effective_level = get_default_log_level() if level is None else level
    if not isinstance(effective_level, int) or effective_level < 0:
        msg = f"Invalid logging level: {level}"
        raise ValueError(msg)

    # Get or create logger and set base config
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Allow handlers to filter
    logger.propagate = propagate

    # Clear existing handlers if requested
    if clear_existing:
        logger.handlers.clear()

    # Normalize handlers list
    handlers: list[logging.Handler] = (
        [handler] if not isinstance(handler, list) else list(handler)
    )

    # Validate and set level on handlers
    for h in handlers:
        if not isinstance(h, logging.Handler):
            msg = f"Expected logging.Handler, got {type(h)}"
            raise TypeError(msg)
        h.setLevel(effective_level)

    # Optional file handler
    if file is not None:
        file_path = Path(file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(effective_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Attach handlers (avoid duplicates)
    existing_ids = {id(h) for h in logger.handlers}
    for h in handlers:
        if id(h) not in existing_ids:
            logger.addHandler(h)

    return logger  # type: ignore[return-value]
