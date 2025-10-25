"""Tests for the logging module."""

from __future__ import annotations

# Import by file path to avoid importing full package during test collection
import importlib.util
import logging
import os
import tempfile
from pathlib import Path
from pathlib import Path as _Path

import pytest

_logging_path = _Path(__file__).resolve().parents[1] / "faninsar" / "logging.py"
_spec = importlib.util.spec_from_file_location("faninsar.logging", _logging_path)
_filog = importlib.util.module_from_spec(_spec)
assert _spec is not None
assert _spec.loader is not None
_spec.loader.exec_module(_filog)  # type: ignore[attr-defined]

SUCCESS = _filog.SUCCESS
color_formatter = _filog.color_formatter
formatter = _filog.formatter
get_default_log_level = _filog.get_default_log_level
setup_logger = _filog.setup_logger
stream_handler = _filog.stream_handler
tqdm_handler = _filog.tqdm_handler


class TestLogging:
    """Test the logging module."""

    def test_success_level(self) -> None:
        """Test that the SUCCESS level is defined."""
        assert SUCCESS == 25
        assert logging.getLevelName(SUCCESS) == "SUCCESS"

    def test_logger_success_method(self) -> None:
        """Test that the logger has a success method."""
        logger = logging.getLogger("test_success")
        assert hasattr(logger, "success")
        assert callable(logger.success)

    def test_setup_logger_default(self) -> None:
        """Test the setup_logger function with default parameters."""
        logger = setup_logger("test_default")
        assert logger.name == "test_default"
        assert logger.level == logging.DEBUG
        assert logger.propagate  # Default is True

        # Check handlers
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.level == get_default_log_level()

        # Clean up
        logging.getLogger("test_default").handlers = []

    def test_setup_logger_with_level(self) -> None:
        """Test the setup_logger function with a specified level."""
        logger = setup_logger("test_level", level=logging.INFO)
        assert logger.name == "test_level"
        assert logger.level == logging.DEBUG  # Logger level is always DEBUG

        # Check handlers
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert handler.level == logging.INFO  # Handler level is INFO

        # Clean up
        logging.getLogger("test_level").handlers = []

    def test_setup_logger_with_file(self) -> None:
        """Test the setup_logger function with a file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            # Setup logger with file
            logger = setup_logger(
                "test_file",
                level=logging.INFO,
                file=log_file,
            )

            assert logger.name == "test_file"

            # Check handlers (should have 2: stream and file)
            assert len(logger.handlers) == 2

            # First handler should be stream handler with INFO level
            stream_h = logger.handlers[0]
            assert isinstance(stream_h, logging.StreamHandler)
            assert stream_h.level == logging.INFO

            # Second handler should be file handler with INFO level
            file_h = logger.handlers[1]
            assert isinstance(file_h, logging.FileHandler)
            assert file_h.level == logging.INFO

            # Test logging to file
            test_msg = "Test file logging"
            logger.info(test_msg)  # Will go to both handlers

            # Check that the message was written to the file
            with log_file.open("r") as f:
                log_content = f.read()
                assert test_msg in log_content

            # Clean up
            logging.getLogger("test_file").handlers = []

    def test_setup_logger_with_file_path_types(self) -> None:
        """Test the setup_logger function with different file path types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with string path
            str_path = str(Path(tmpdir) / "string_path.log")
            str_logger = setup_logger("test_str_path", file=str_path)
            assert len(str_logger.handlers) == 2
            assert isinstance(str_logger.handlers[1], logging.FileHandler)

            # Test with Path object
            path_obj = Path(tmpdir) / "path_object.log"
            path_logger = setup_logger("test_path_obj", file=path_obj)
            assert len(path_logger.handlers) == 2
            assert isinstance(path_logger.handlers[1], logging.FileHandler)

            # Clean up
            logging.getLogger("test_str_path").handlers = []
            logging.getLogger("test_path_obj").handlers = []

    def test_setup_logger_with_multiple_handlers(self) -> None:
        """Test the setup_logger function with multiple handlers."""
        handler1 = logging.StreamHandler()
        handler2 = logging.StreamHandler()

        logger = setup_logger(
            "test_multi",
            handler=[handler1, handler2],
            level=logging.WARNING,
        )

        assert len(logger.handlers) == 2
        assert all(h.level == logging.WARNING for h in logger.handlers)

        # Clean up
        logging.getLogger("test_multi").handlers = []

    def test_setup_logger_propagate(self) -> None:
        """Test the setup_logger function with propagate=False."""
        logger = setup_logger("test_propagate", propagate=False)
        assert not logger.propagate

        # Clean up
        logging.getLogger("test_propagate").handlers = []

    def test_tqdm_handler(self) -> None:
        """Test the tqdm_handler."""
        logger = setup_logger("test_tqdm", handler=tqdm_handler)

        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert handler is tqdm_handler

        # Clean up
        logging.getLogger("test_tqdm").handlers = []

    def test_setup_logger_aliases(self) -> None:
        """Ensure legacy alias parameters work (log_name, log_file)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "alias.log"
            logger = setup_logger(
                log_name="alias_mod",
                log_file=log_path,
                level=logging.WARNING,
            )
            assert logger.name == "alias_mod"
            assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
            # Clean up
            logging.getLogger("alias_mod").handlers = []
