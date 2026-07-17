"""Direct geographic-grid SLC coregistration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from faninsar.processing.pipeline.geo_resample import (
    compose_secondary_coordinates,
    resample_complex_at_coordinates,
)
from faninsar.processing.tops.deramp import (
    TOPSCarrierModel,
    carrier_phase_at_points,
)

if TYPE_CHECKING:
    from faninsar.processing.coreg.offsets import OffsetFieldResult
    from faninsar.processing.pipeline.geo_lut import Geo2RdrLUT

__all__ = ["coregister_geocoded_slcs"]


def _apply_reramp(
    deramped: np.ndarray,
    carrier: TOPSCarrierModel,
    azimuth: np.ndarray,
    range_index: np.ndarray,
    *,
    native_height: int,
) -> np.ndarray:
    phase = carrier_phase_at_points(
        carrier,
        azimuth,
        range_index,
        centre_row=float(native_height // 2),
        dtype=np.float32,
    )
    output = np.asarray(deramped, dtype=np.complex64) * np.exp(
        1j * np.asarray(phase, dtype=np.float64)
    )
    return output.astype(np.complex64, copy=False)


def coregister_geocoded_slcs(
    reference_deramped: np.ndarray,
    secondary_deramped: np.ndarray,
    *,
    reference_carrier: TOPSCarrierModel,
    secondary_carrier: TOPSCarrierModel,
    reference_lut: Geo2RdrLUT,
    offsets: OffsetFieldResult,
    executor: str = "serial",
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coregister two deramped SLCs directly on a geographic grid.

    Parameters
    ----------
    reference_deramped, secondary_deramped : numpy.ndarray
        Native-resolution deramped SLCs.
    reference_carrier, secondary_carrier : TOPSCarrierModel
        Acquisition-specific TOPS carrier models.
    reference_lut : Geo2RdrLUT
        Reference radar coordinates at geographic pixel centres.
    offsets : OffsetFieldResult
        Dense reference-to-secondary radar offsets.
    executor : {"serial", "dask-torch"}, optional
        Lanczos implementation.
    device : {"auto", "cpu", "cuda"}, optional
        Torch device for the accelerated implementation.

    Returns
    -------
    reference_geo, secondary_geo, valid : tuple[numpy.ndarray, ...]
        Reramped aligned SLCs and their shared validity mask.

    Notes
    -----
    Each SLC is interpolated exactly once in the deramped domain. Reramping is
    evaluated analytically at the same fractional source coordinates used by
    that interpolation.

    """
    secondary_azimuth, secondary_range, secondary_valid = (
        compose_secondary_coordinates(reference_lut, offsets)
    )
    reference_geo, reference_valid = resample_complex_at_coordinates(
        reference_deramped,
        reference_lut.az_full,
        reference_lut.rg_full,
        valid=reference_lut.valid,
        executor=executor,
        device=device,
    )
    secondary_geo, secondary_valid = resample_complex_at_coordinates(
        secondary_deramped,
        secondary_azimuth,
        secondary_range,
        valid=secondary_valid,
        executor=executor,
        device=device,
    )
    valid = (
        reference_valid
        & secondary_valid
        & np.isfinite(reference_geo.real)
        & np.isfinite(reference_geo.imag)
        & np.isfinite(secondary_geo.real)
        & np.isfinite(secondary_geo.imag)
    )
    reference_geo = _apply_reramp(
        reference_geo,
        reference_carrier,
        reference_lut.az_full,
        reference_lut.rg_full,
        native_height=reference_deramped.shape[0],
    )
    secondary_geo = _apply_reramp(
        secondary_geo,
        secondary_carrier,
        secondary_azimuth,
        secondary_range,
        native_height=secondary_deramped.shape[0],
    )
    invalid_value = np.complex64(np.nan + 1j * np.nan)
    reference_geo = np.where(valid, reference_geo, invalid_value).astype(np.complex64)
    secondary_geo = np.where(valid, secondary_geo, invalid_value).astype(np.complex64)
    return reference_geo, secondary_geo, valid
