"""Public invert / NSBAS / SBAS façades over InterferogramStack."""

from __future__ import annotations

from typing import Any

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
    batch_lstsq = staticmethod(batch_lstsq)
    calculate_u = staticmethod(calculate_u)
    censored_lstsq = staticmethod(censored_lstsq)


class SBAS:
    """SBAS façade — same solver without a temporal model (pure SBAS G)."""

    NSBASSolver = NSBASSolver

    @staticmethod
    def solve(stack: Any, **kwargs: Any) -> Any:
        """Run pure SBAS inversion (no parametric model)."""
        solver = NSBASSolver(stack=stack, model=None, **kwargs)
        return solver.inverse()


def invert(
    stack: Any,
    model: TimeSeriesModels | None = None,
    **kwargs: Any,
) -> Any:
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
    Any
        Result of :meth:`NSBASSolver.inverse`.

    """
    solver = NSBASSolver(stack=stack, model=model, **kwargs)
    return solver.inverse()


__all__ = ["NSBAS", "SBAS", "invert"]
