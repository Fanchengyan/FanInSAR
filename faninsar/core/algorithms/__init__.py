"""Domain-neutral numerical algorithms with lazily loaded operation families.

The public operations use float64 Torch computation on the input CPU/CUDA
execution device. Importing this toolbox alone does not load numerical kernels.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "LeastSquaresResult",
    "PolynomialSurface",
    "block_reduce2d",
    "filter2d",
    "fit_surface2d",
    "interpolate2d",
    "least_squares",
    "polyval",
]

_EXPORT_MODULES = {
    "interpolate2d": ".sampling",
    "block_reduce2d": ".reduction",
    "filter2d": ".filtering",
    "polyval": ".fitting",
    "fit_surface2d": ".fitting",
    "PolynomialSurface": ".fitting",
    "least_squares": ".linalg",
    "LeastSquaresResult": ".linalg",
}


def __getattr__(name: str) -> Any:
    """Load only the mathematical family owning a requested public export."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public names in discovery without importing their families."""
    return sorted(set(globals()) | set(__all__))
