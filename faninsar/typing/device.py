"""Typing for device operations."""

from __future__ import annotations

from torch import device as torch_device

DeviceLike = str | torch_device
