"""Typing for logging module."""

from __future__ import annotations

from typing import Literal

LogLevel = Literal["notset", "debug", "info", "warning", "error", "critical"]
