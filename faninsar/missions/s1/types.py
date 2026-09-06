"""Immutable Sentinel-1 product types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from faninsar.processing.contracts import OrbitMetadata
    from faninsar.processing.readers import (
        DopplerCentroidPolynomial,
        ValidSampleMask,
    )


@dataclass(frozen=True, slots=True)
class S1Burst:
    """One TOPS burst within a sub-swath annotation."""

    index: int
    azimuth_time: datetime
    sensing_time: datetime
    azimuth_anx_time_s: float
    byte_offset: int
    lines: int
    samples: int
    first_valid_sample: tuple[int, ...]
    last_valid_sample: tuple[int, ...]
    footprint: tuple[tuple[float, float], ...] | None = None

    @property
    def valid_samples(self) -> ValidSampleMask:
        """Return valid-sample windows as a processing contract object."""
        from faninsar.processing.readers import ValidSampleMask

        return ValidSampleMask(
            first_valid_sample=self.first_valid_sample,
            last_valid_sample=self.last_valid_sample,
        )


@dataclass(frozen=True, slots=True)
class S1Swath:
    """One IW/EW sub-swath with annotation-derived geometry metadata."""

    swath: Literal["IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5"]
    polarization: str
    annotation_path: str
    measurement_path: str
    lines: int
    samples: int
    lines_per_burst: int
    samples_per_burst: int
    range_pixel_spacing_m: float
    azimuth_pixel_spacing_m: float
    azimuth_time_interval_s: float
    slant_range_time_s: float
    range_sampling_rate_hz: float
    radar_frequency_hz: float
    sensing_start: datetime
    sensing_stop: datetime
    bursts: tuple[S1Burst, ...]
    orbit: OrbitMetadata
    doppler_centroid: tuple[DopplerCentroidPolynomial, ...]
    azimuth_fm_rate: tuple[tuple[datetime, float, tuple[float, ...]], ...]
    azimuth_steering_rate_rad_s: float


@dataclass(frozen=True, slots=True)
class S1Product:
    """SAFE or EOPF Sentinel-1 Level-1 SLC product handle."""

    product_id: str
    source_path: Path
    mission_id: str
    product_type: str
    mode: str
    absolute_orbit: int
    polarizations: tuple[str, ...]
    swaths: tuple[S1Swath, ...]

    def swath(
        self,
        name: str,
        polarization: str | None = None,
    ) -> S1Swath:
        """Return one sub-swath by name and optional polarization."""
        from .errors import UnsupportedPolarizationError, reject_product

        matches = [item for item in self.swaths if item.swath == name]
        if polarization is not None:
            matches = [item for item in matches if item.polarization == polarization]
        if not matches:
            if polarization is not None and polarization not in self.polarizations:
                message = (
                    f"unsupported polarization {polarization!r} in product "
                    f"{self.source_path}; available={self.polarizations}"
                )
                raise UnsupportedPolarizationError(message)
            reject_product(f"swath {name!r} not found in product {self.source_path}")
        return matches[0]
