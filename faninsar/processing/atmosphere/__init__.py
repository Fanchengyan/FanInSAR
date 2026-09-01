"""Mission-neutral ionospheric phase estimation for the Stack pipeline.

Split-spectrum (``split_main_band``) capability per PROPOSAL-0036: all
physical inputs are caller-supplied parameters; this package performs no
data-format changes and fails closed on invalid input.
"""

from __future__ import annotations

from .config import (
    DEFAULT_MIN_BANDWIDTH_HZ,
    IonosphereEstimationConfig,
    JumpAlignmentStrategy,
    SolveCore,
)
from .estimation import (
    align_absolute_jumps,
    estimate_disp_nondisp,
    solve_2x2_low_high,
    solve_guided_split,
)
from .filter import (
    gaussian_kernel_1d,
    remove_small_components,
    smooth_inverse_variance,
)
from .network import IonosphereNetworkResult, invert_ionosphere_network
from .split_spectrum import SubBandResult, split_range_spectrum

__all__ = [
    "DEFAULT_MIN_BANDWIDTH_HZ",
    "IonosphereEstimationConfig",
    "IonosphereNetworkResult",
    "JumpAlignmentStrategy",
    "SolveCore",
    "SubBandResult",
    "align_absolute_jumps",
    "estimate_disp_nondisp",
    "gaussian_kernel_1d",
    "invert_ionosphere_network",
    "remove_small_components",
    "smooth_inverse_variance",
    "solve_2x2_low_high",
    "solve_guided_split",
    "split_range_spectrum",
]
