"""Typing for device operations."""

from __future__ import annotations

from typing import Union

from torch import device as torch_device

DeviceLike = Union[str, torch_device]
