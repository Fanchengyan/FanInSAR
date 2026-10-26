"""Convert spatial coherence to interferogram observation weights."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import numpy as np

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = setup_logger(__name__)

WeightFunc = Literal["no", "coh", "fim", "var"]

_VALID_WEIGHT_FUNCS = {"no", "sbas", "uniform", "coh", "lin", "fim", "fisher", "var"}


def coherence_to_weight(
    coh: NDArray[np.floating],
    weight_func: WeightFunc | str = "fim",
    n_looks: int = 1,
    epsilon: float = 1e-3,
) -> NDArray[np.floating]:
    """Convert spatial coherence to positive observation weights.

    Parameters
    ----------
    coh : ndarray
        Spatial coherence, typically shaped ``(n_pair, n_pixel)``.
    weight_func : {"no", "coh", "fim", "var"}, optional
        Weight definition. ``coh`` uses coherence itself, ``fim`` uses the
        Fisher information index ``2 L γ² / (1 - γ²)`` (Seymour and Cumming,
        1994), and ``var`` uses the inverse DS phase variance (Tough et al.,
        1995). ``no`` returns unit weights.
    n_looks : int, optional
        Number of independent looks ``L``. Used by ``fim`` and ``var``.
        Values below 1 are raised to 1; values above 80 are capped at 80.
    epsilon : float, optional
        Lower clip applied to coherence to keep weights finite.

    Returns
    -------
    ndarray
        Weights with the same shape as ``coh``.

    Raises
    ------
    ValueError
        If ``weight_func`` is not recognized.

    """
    coh_arr = np.asarray(coh, dtype=np.float64)
    coh_arr = np.where(np.isnan(coh_arr), epsilon, coh_arr)
    coh_arr = np.clip(coh_arr, epsilon, 1.0 - epsilon)
    n_looks = max(int(n_looks), 1)

    name = str(weight_func).lower()
    if name not in _VALID_WEIGHT_FUNCS:
        msg = f"Unrecognized weight_func {weight_func!r}"
        logger.error(msg, extra={"weight_func": weight_func})
        raise ValueError(msg)

    if name in {"no", "sbas", "uniform"}:
        return np.ones_like(coh_arr)
    if name in {"coh", "lin"}:
        return coh_arr
    if name in {"fim", "fisher"}:
        return 2.0 * n_looks * np.square(coh_arr) / (1.0 - np.square(coh_arr))
    return 1.0 / _phase_variance_ds(coh_arr, n_looks)


def _phase_variance_ds(
    coh: NDArray[np.floating],
    n_looks: int,
    coh_step: float = 0.005,
) -> NDArray[np.floating]:
    """Look up DS interferometric phase variance for each coherence value."""
    n_looks = min(int(n_looks), 80)
    var_lut, coh_lut = _ds_variance_lut(n_looks, coh_step)
    coh_min = float(coh_lut[0])
    coh_max = float(coh_lut[-1])
    coh_clip = np.clip(np.asarray(coh, dtype=np.float64), coh_min, coh_max)
    idx = np.array((coh_clip - coh_min) / coh_step, dtype=np.int64)
    idx = np.clip(idx, 0, var_lut.size - 1)
    return var_lut[idx]


@lru_cache(maxsize=16)
def _ds_variance_lut(n_looks: int, coh_step: float) -> tuple[np.ndarray, np.ndarray]:
    """Build a DS phase-variance lookup table for one look number."""
    coh_num = int(1.0 / coh_step)
    coh_lut = np.linspace(coh_step, 1.0, num=coh_num) - coh_step / 2.0
    phi_num = len(coh_lut)
    phi = np.linspace(-np.pi, np.pi, phi_num, dtype=np.float64).reshape(-1, 1)
    phi_step = 2.0 * np.pi / phi_num
    pdf = _phase_pdf_ds(n_looks, coh_lut, phi_num)
    var = np.sum(np.square(phi) * pdf * phi_step, axis=0)
    nonpositive = var <= 0
    if np.any(~nonpositive):
        var[nonpositive] = np.nanmin(var[~nonpositive])
    else:
        var[nonpositive] = np.finfo(np.float64).eps
    return var, coh_lut


def _phase_pdf_ds(n_looks: int, coherence: np.ndarray, phi_num: int) -> np.ndarray:
    """Marginal interferometric phase PDF for distributed scatterers.

    Tough et al. (1995), Eq. 66; Hanssen (2001), Eq. 4.2.23.
    """
    coh = np.asarray(coherence, dtype=np.float64).reshape(1, -1)
    phi = np.linspace(-np.pi, np.pi, phi_num, dtype=np.float64).reshape(-1, 1)
    looks = int(n_looks)

    scale = np.power(1.0 - np.square(coh), looks) / (2.0 * np.pi)
    scale = np.tile(scale, (phi_num, 1))
    pref = _gamma(2 * looks - 1) / ((_gamma(looks) ** 2) * 2 ** (2 * (looks - 1)))
    beta = np.abs(coh) * np.cos(phi)
    one_minus_beta2 = 1.0 - np.square(beta)
    term = ((2 * looks - 1) * beta / np.power(one_minus_beta2, looks + 0.5)) * (
        np.pi / 2.0 + np.arcsin(np.clip(beta, -1.0, 1.0))
    )
    term = term + 1.0 / np.power(one_minus_beta2, looks)

    extra = 0.0
    if looks > 1:
        extra = 0.0
        for rank in range(looks - 1):
            piece = _gamma(looks - 0.5) / _gamma(looks - 0.5 - rank)
            piece *= _gamma(looks - 1 - rank) / _gamma(looks - 1)
            piece *= (1.0 + (2 * rank + 1) * np.square(beta)) / np.power(
                one_minus_beta2,
                rank + 2,
            )
            extra = extra + piece
        extra = extra / (2.0 * (looks - 1))

    return scale * (pref * term + extra)


def _gamma(value: float) -> float:
    """Gamma function with SciPy-like inf on non-positive integers."""
    try:
        return math.gamma(value)
    except ValueError:
        return float("inf")
