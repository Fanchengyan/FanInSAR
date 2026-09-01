"""Configuration for mission-neutral ionosphere estimation.

All physical inputs are plain caller-supplied parameters (owner decision
2026-08-27 in PROPOSAL-0036): this module never reads or mutates persisted
data formats.  The configuration is validated for physical sanity at
construction time and fails closed with structured errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

JumpAlignmentStrategy = Literal["isce3_global_jump", "alosstack_pixel_rounded"]

SolveCore = Literal["isce3", "guided_split"]

DEFAULT_MIN_BANDWIDTH_HZ = 14.0e6


@dataclass(frozen=True)
class IonosphereEstimationConfig:
    """Caller-supplied inputs for one split-spectrum estimation lane.

    Parameters
    ----------
    f0 : float
        Radar carrier center frequency [Hz].  Corrected products are
        referenced to this frequency.
    freq_low : float
        Center frequency of the lower range subband [Hz].
    freq_high : float
        Center frequency of the upper range subband [Hz].
    min_bandwidth_hz : float, optional
        Minimum effective subband separation ``freq_high - freq_low`` for a
        non-degraded run.  Defaults to 14 MHz (narrowest PALSAR-2 ScanSAR
        subswath class).  Below the threshold the constructor raises unless
        ``degraded=True``.
    alignment_strategy : {"isce3_global_jump", "alosstack_pixel_rounded"}, \
            default="isce3_global_jump"
        Absolute phase-jump alignment convention.  The ISCE3 strategy adds
        one global integer-cycle correction computed from the scene mean;
        the alosStack strategy fits a weighted degree-2 surface to the
        subband phase difference and removes per-pixel integer cycles
        (``runIonFilt.computeIonosphere(adjFlag=1)`` semantics).
    solve_core : {"isce3", "guided_split"}, default="isce3"
        Solve core selection.  ``"isce3"`` (default) uses the pure 2x2
        linear system of ``estimate_iono_low_high`` after a global-jump
        alignment.  ``"guided_split"`` uses the surface-guided variant
        (:func:`~faninsar.processing.atmosphere.estimation.solve_guided_split`)
        that adds a coherence-weighted degree-2 surface fit and per-pixel
        integer-cycle adjustment of the upper band before the identical
        physical solve, compatible with the ISCE2 alosStack chain.
    cor_order_adj : int, default=20
        Coherence-power exponent for the weighted surface fit in the
        ``"guided_split"`` lane.  ``cor ** cor_order_adj`` mirrors the
        alosStack ``corOrderAdj`` parameter (default 20).
    looks : tuple[int, int], optional
        Nominal (azimuth, range) multilook factors applied before
        estimation; recorded in manifests only, applied by callers.
    coherence_threshold : float, optional
        Masking threshold for coherence-gated filtering stages.
    degraded : bool, optional
        Explicit opt-in that permits below-threshold bandwidth.  Degraded
        outputs must be flagged by callers (layer metadata) and downstream
        consumers refuse them by default.
    degradation_reason : str, optional
        Required when ``degraded=True``; short machine-readable code.

    Raises
    ------
    ValueError
        On non-positive frequencies, non-monotonic ordering, negative
        thresholds/looks, coherence outside (0, 1), or a below-threshold
        split without explicit degradation.

    Examples
    --------
    >>> cfg = IonosphereEstimationConfig(
    ...     f0=1257.5e6,
    ...     freq_low=1257.5e6 - 28e6 / 3,
    ...     freq_high=1257.5e6 + 28e6 / 3,
    ... )  # doctest: +SKIP

    """

    f0: float
    freq_low: float
    freq_high: float
    min_bandwidth_hz: float = DEFAULT_MIN_BANDWIDTH_HZ
    alignment_strategy: JumpAlignmentStrategy = "isce3_global_jump"
    solve_core: SolveCore = "isce3"
    cor_order_adj: int = 20
    looks: tuple[int, int] = field(default=(16, 16))
    coherence_threshold: float = 0.5
    window_function: str = "tukey"
    window_shape: float = 0.25
    degraded: bool = False
    degradation_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate physical sanity and fail closed on bad caller input."""
        for name in ("f0", "freq_low", "freq_high"):
            value = float(getattr(self, name))
            if not value > 0.0:
                message = f"{name} must be positive, got {value!r}"
                logger.error(message)
                raise ValueError(message)
        if not self.freq_low < self.f0 < self.freq_high:
            message = (
                "frequency roles must satisfy freq_low < f0 < freq_high, got "
                f"{self.freq_low!r}, {self.f0!r}, {self.freq_high!r}"
            )
            logger.error(message)
            raise ValueError(message)
        if self.freq_low == self.freq_high:
            message = (
                "Frequency combination leads to singular matrix (freq_low == freq_high)"
            )
            logger.error(message)
            raise ValueError(message)
        if self.min_bandwidth_hz <= 0.0:
            message = (
                f"min_bandwidth_hz must be positive, got {self.min_bandwidth_hz!r}"
            )
            logger.error(message)
            raise ValueError(message)
        az_looks, rg_looks = self.looks
        if az_looks < 1 or rg_looks < 1:
            message = f"looks must be >= 1, got {self.looks!r}"
            logger.error(message)
            raise ValueError(message)
        if not 0.0 < self.coherence_threshold < 1.0:
            message = (
                "coherence_threshold must lie in (0, 1), got "
                f"{self.coherence_threshold!r}"
            )
            logger.error(message)
            raise ValueError(message)
        if self.solve_core not in ("isce3", "guided_split"):
            message = (
                f"solve_core must be 'isce3' or 'guided_split', "
                f"got {self.solve_core!r}"
            )
            logger.error(message)
            raise ValueError(message)
        if self.cor_order_adj < 1:
            message = f"cor_order_adj must be >= 1, got {self.cor_order_adj!r}"
            logger.error(message)
            raise ValueError(message)
        effective_split_hz = self.freq_high - self.freq_low
        if effective_split_hz < self.min_bandwidth_hz and not self.degraded:
            message = (
                f"effective subband split {effective_split_hz:.3g} Hz is below "
                f"min_bandwidth_hz={self.min_bandwidth_hz:.3g}; split-spectrum "
                "estimation is unreliable. Re-run with degraded=True and a "
                "degradation_reason to force flagged low-confidence outputs."
            )
            logger.error(message)
            raise ValueError(message)
        if self.degraded and not self.degradation_reason:
            message = "degraded=True requires an explicit degradation_reason"
            logger.error(message)
            raise ValueError(message)

    @property
    def effective_split_hz(self) -> float:
        """Return the frequency span covered by the two subband centers."""
        return float(self.freq_high - self.freq_low)
