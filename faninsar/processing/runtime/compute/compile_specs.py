"""Fixed compile shapes and dtypes for hot kernels."""

from __future__ import annotations

from typing import Final

DEFAULT_CHUNK: Final[int] = 65_536
DEFAULT_DTYPE_NAME: Final[str] = "complex64"
DEFAULT_KERNEL_WIDTH: Final[int] = 8

# Shape variants used by warmup profiles (rows, cols) for 2-D kernels.
WARMUP_SHAPES: tuple[tuple[int, int], ...] = (
    (256, 256),
    (512, 128),
    (DEFAULT_CHUNK // 256, 256),
)


__all__ = [
    "DEFAULT_CHUNK",
    "DEFAULT_DTYPE_NAME",
    "DEFAULT_KERNEL_WIDTH",
    "WARMUP_SHAPES",
]
