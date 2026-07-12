"""Burst merge product data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.processing.merge.grid import GeoGridSpec

logger = setup_logger(__name__)

PhaseDomain = Literal["complex", "unwrapped"]
PathPolicy = Literal["same_path_only", "allow_cross_path"]
MergeMode = Literal["complex_average", "phase_network"]

__all__ = ["BurstGeoProduct", "MosaicProduct"]


@dataclass(frozen=True, slots=True)
class BurstGeoProduct:
    """Single burst geocoded onto the common merge grid.

    Attributes
    ----------
    burst_id : str
        Stable identifier, e.g. ``"{scene}_{swath}_b{idx}"``.
    path_id : str
        Relative orbit plus ascend/descend tag, e.g. ``"T100_A"``.
    swath : str
        Sub-swath name (``"IW1"``/``"IW2"``/``"IW3"``).
    date : datetime.date
        Acquisition date.
    grid : GeoGridSpec
        Grid the complex field is sampled on.
    complex : numpy.ndarray
        Complex64 corrected SLC or wrapped interferogram.
    weight : numpy.ndarray
        Float32 per-pixel weight (mask × feather × coh × …).
    coherence : numpy.ndarray or None
        Coherence layer (ifg mode only).
    phase_domain : {"complex", "unwrapped"}
        ``"complex"`` for wrapped/SLC products (default, mergeable).
        ``"unwrapped"`` is rejected by default merge.
    look_direction : str
        ``"ascending"`` or ``"descending"``.
    meta : dict
        Orbit, B⊥, carrier version, etc.

    """

    burst_id: str
    path_id: str
    swath: str
    date: date
    grid: GeoGridSpec
    complex: np.ndarray
    weight: np.ndarray
    coherence: np.ndarray | None = None
    phase_domain: PhaseDomain = "complex"
    look_direction: str = "ascending"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate array shapes and phase-domain consistency."""
        if self.complex.dtype != np.complex64:
            raise ValueError("BurstGeoProduct.complex must be complex64")
        if self.complex.shape != self.grid.shape:
            raise ValueError(
                "BurstGeoProduct.complex shape must match grid.shape"
            )
        if self.weight.shape != self.grid.shape:
            raise ValueError(
                "BurstGeoProduct.weight shape must match grid.shape"
            )
        if self.coherence is not None and self.coherence.shape != self.grid.shape:
            raise ValueError(
                "BurstGeoProduct.coherence shape must match grid.shape"
            )
        if self.phase_domain not in ("complex", "unwrapped"):
            raise ValueError(
                "phase_domain must be 'complex' or 'unwrapped'"
            )


@dataclass(frozen=True, slots=True)
class MosaicProduct:
    """Weighted mosaic of multiple bursts on a common grid.

    Attributes
    ----------
    grid : GeoGridSpec
        Common grid.
    complex : numpy.ndarray
        Complex64 mosaic (wrapped ifg or SLC).
    coherence : numpy.ndarray or None
        Mosaic coherence.
    weight_sum : numpy.ndarray
        Sum of per-burst weights at each pixel.
    n_bursts : numpy.ndarray
        Number of bursts contributing at each pixel (uint8).
    component_id : numpy.ndarray
        Connected-component label per pixel (int16); 0 = nodata.
    network : MergeGraphStats or None
        Phase network statistics (None for complex_average mode).

    """

    grid: GeoGridSpec
    complex: np.ndarray
    coherence: np.ndarray | None
    weight_sum: np.ndarray
    n_bursts: np.ndarray
    component_id: np.ndarray
    network: Any | None = None

    def __post_init__(self) -> None:
        """Validate mosaic array shapes and dtypes."""
        if self.complex.dtype != np.complex64:
            raise ValueError("MosaicProduct.complex must be complex64")
        if self.complex.shape != self.grid.shape:
            raise ValueError("MosaicProduct.complex shape must match grid")
        for arr_name in ("weight_sum", "n_bursts", "component_id"):
            arr = getattr(self, arr_name)
            if arr.shape != self.grid.shape:
                raise ValueError(
                    f"MosaicProduct.{arr_name} shape must match grid"
                )