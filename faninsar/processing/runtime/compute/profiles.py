"""Warmup / compile profiles for common hardware targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class CompileProfile:
    """Named compile/warmup profile."""

    name: str
    device: str
    chunk_size: int = 65_536
    dtype_name: str = "complex64"
    kernel_groups: tuple[str, ...] = (
        "knab_resample",
        "lanczos_resample",
        "deramp",
        "reramp",
        "form_interferogram",
        "flatten",
        "multilook",
        "coherence",
    )


PROFILES: Final[dict[str, CompileProfile]] = {
    "sentinel1-cpu": CompileProfile(name="sentinel1-cpu", device="cpu"),
    "sentinel1-cuda": CompileProfile(name="sentinel1-cuda", device="cuda"),
    "sentinel1-mps": CompileProfile(name="sentinel1-mps", device="mps"),
    "high-precision-cuda": CompileProfile(
        name="high-precision-cuda",
        device="cuda",
        dtype_name="complex128",
    ),
}


def get_profile(name: str) -> CompileProfile:
    """Return a named profile or raise KeyError."""
    if name not in PROFILES:
        # Allow short aliases used by CLI: sentinel1 → sentinel1-cpu
        alias = f"{name}-cpu"
        if alias in PROFILES:
            return PROFILES[alias]
        message = f"unknown profile {name!r}; known={sorted(PROFILES)}"
        raise KeyError(message)
    return PROFILES[name]


__all__ = ["PROFILES", "CompileProfile", "get_profile"]
