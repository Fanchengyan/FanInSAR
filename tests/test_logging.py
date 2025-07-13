"""Tests for the logging module."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pytest

from faninsar.logging import (
    SUCCESS,
    color_formatter,
    formatter,
    setup_logger,
    stream_handler,
    tqdm_handler,
)


class TestLogging:
    """Test the logging module."""

    def test_success_level(self):
        """Test that the SUCCESS level is defined."""
        assert SUCCESS == 25
        assert logging.getLevelName(SUCCESS) == "SUCCESS"

    def test_logger_success_method(self):
        """Test that the logger has a success method."""
        logger = logging.getLogger("test_success")
        assert hasattr(logger, "success")
        assert callable(logger.success)

    def test_setup_logger_default(self):
        """Test the setup_logger function with default parameters."""
        logger = setup_logger("test_default")
        assert logger.name == "test_default"
        assert logger.level == logging.DEBUG
        assert logger.propagate  # Default is now True

        # Check handlers
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.level == logging.DEBUG

        # Clean up
        logging.getLogger("test_default").handlers = []

    def test_setup_logger_with_level(self):
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

    def test_setup_logger_with_file(self):
        """Test the setup_logger function with a file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            # Setup logger with file
            logger = setup_logger(
                "test_file",
                level=logging.INFO,
                log_file=log_file,
            )

            assert logger.name == "test_file"

            # Check handlers (should have 2: stream and file)
            assert len(logger.handlers) == 2

            # First handler should be stream handler with INFO level
            stream_h = logger.handlers[0]
            assert isinstance(stream_h, logging.StreamHandler)
            assert stream_h.level == logging.INFO

            # Second handler should be file handler with INFO level (same as global level)
            file_h = logger.handlers[1]
            assert isinstance(file_h, logging.FileHandler)
            assert file_h.level == logging.INFO

            # Test logging to file
            test_msg = "Test file logging"
            logger.info(test_msg)  # Will go to both handlers

            # Check that the message was written to the file
            with open(log_file, "r") as f:
                log_content = f.read()
                assert test_msg in log_content

            # Clean up
            logging.getLogger("test_file").handlers = []

    def test_setup_logger_with_file_path_types(self):
        """Test the setup_logger function with different file path types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with string path
            str_path = os.path.join(tmpdir, "string_path.log")
            str_logger = setup_logger("test_str_path", log_file=str_path)
            assert len(str_logger.handlers) == 2
            assert isinstance(str_logger.handlers[1], logging.FileHandler)

            # Test with Path object
            path_obj = Path(tmpdir) / "path_object.log"
            path_logger = setup_logger("test_path_obj", log_file=path_obj)
            assert len(path_logger.handlers) == 2
            assert isinstance(path_logger.handlers[1], logging.FileHandler)

            # Clean up
            logging.getLogger("test_str_path").handlers = []
            logging.getLogger("test_path_obj").handlers = []

    def test_setup_logger_with_multiple_handlers(self):
        """Test the setup_logger function with multiple handlers."""
        handler1 = logging.StreamHandler()
        handler2 = logging.StreamHandler()

        logger = setup_logger(
            "test_multi",
            handler=[handler1, handler2],
            level=logging.WARNING
        )

        assert len(logger.handlers) == 2
        assert all(h.level == logging.WARNING for h in logger.handlers)

        # Clean up
        logging.getLogger("test_multi").handlers = []

    def test_setup_logger_propagate(self):
        """Test the setup_logger function with propagate=False."""
        logger = setup_logger("test_propagate", propagate=False)
        assert not logger.propagate

        # Clean up
        logging.getLogger("test_propagate").handlers = []

    def test_tqdm_handler(self):
        """Test the tqdm_handler."""
        logger = setup_logger("test_tqdm", handler=tqdm_handler)

        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert handler is tqdm_handler

        # Clean up
        logging.getLogger("test_tqdm").handlers = []
