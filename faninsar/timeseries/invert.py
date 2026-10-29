"""Public invert / NSBAS / SBAS façades over InterferogramStack."""

from __future__ import annotations

from typing import Any

from faninsar.timeseries.coherence_weight import (
    WeightFunc,
    coherence_to_weight,
    normalize_weight_func,
)
from faninsar.timeseries.models import (
    AnnualSemiannualSinusoidal,
    AnnualSinusoidalModel,
    CubicModel,
    FreezeThawCycleModel,
    FreezeThawCycleModelWithVelocity,
    LinearModel,
    QuadraticModel,
    TimeSeriesModels,
)
from faninsar.timeseries.solver import (
    LstsqResult,
    NSBASResult,
    NSBASSolver,
    batch_lstsq,
    calculate_u,
    censored_lstsq,
)


class NSBAS:
    """Namespace façade matching historical ``faninsar.timeseries`` usage."""

    NSBASSolver = NSBASSolver
    TimeSeriesModels = TimeSeriesModels
    LinearModel = LinearModel
    QuadraticModel = QuadraticModel
    CubicModel = CubicModel
    AnnualSinusoidalModel = AnnualSinusoidalModel
    AnnualSemiannualSinusoidal = AnnualSemiannualSinusoidal
    FreezeThawCycleModel = FreezeThawCycleModel
    FreezeThawCycleModelWithVelocity = FreezeThawCycleModelWithVelocity
    LstsqResult = LstsqResult
    NSBASResult = NSBASResult
    batch_lstsq = staticmethod(batch_lstsq)
    calculate_u = staticmethod(calculate_u)
    censored_lstsq = staticmethod(censored_lstsq)
    coherence_to_weight = staticmethod(coherence_to_weight)
    normalize_weight_func = staticmethod(normalize_weight_func)


class SBAS:
    """SBAS façade — same solver without a temporal model (pure SBAS G)."""

    NSBASSolver = NSBASSolver

    @staticmethod
    def solve(stack: Any, **kwargs: Any) -> NSBASResult:
        """Run pure SBAS inversion (no parametric model)."""
        solver = NSBASSolver(stack=stack, model=None, **kwargs)
        return solver.inverse()


def invert(
    stack: Any,
    model: TimeSeriesModels | None = None,
    **kwargs: Any,
) -> NSBASResult:
    """Invert an :class:`InterferogramStack` with optional NSBAS model.

    Parameters
    ----------
    stack : InterferogramStack
        Unwrapped interferogram stack seam product.
    model : TimeSeriesModels, optional
        Parametric model; omit for pure SBAS.
    **kwargs
        Forwarded to :class:`NSBASSolver`.

    Returns
    -------
    NSBASResult
        Result of :meth:`NSBASSolver.inverse`. Tuple unpacking keeps the
        historical ``incs, params, residual_pair, residual_tsm`` pattern
        working.

    """
    solver = NSBASSolver(stack=stack, model=model, **kwargs)
    return solver.inverse()


__all__ = [
    "NSBAS",
    "SBAS",
    "LstsqResult",
    "NSBASResult",
    "WeightFunc",
    "coherence_to_weight",
    "invert",
    "normalize_weight_func",
]
