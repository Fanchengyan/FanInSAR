"""Resample radar fields through geographic lookup tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import map_coordinates

from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.resampling import lanczos_resample

if TYPE_CHECKING:
    from faninsar.processing.coreg.offsets import OffsetFieldResult
    from faninsar.processing.pipeline.geo_lut import Geo2RdrLUT

__all__ = [
    "apply_lut_complex",
    "apply_lut_real",
    "compose_secondary_coordinates",
    "resample_complex_at_coordinates",
]


def resample_complex_at_coordinates(
    complex_radar: np.ndarray,
    azimuth: np.ndarray,
    range_index: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    executor: str = "serial",
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a complex radar image at arbitrary fractional coordinates.

    Parameters
    ----------
    complex_radar : numpy.ndarray
        Full-resolution complex radar image.
    azimuth, range_index : numpy.ndarray
        Zero-based fractional source coordinates on the destination grid.
    valid : numpy.ndarray, optional
        Additional destination validity mask.
    executor : {"serial", "dask-torch"}, optional
        Lanczos implementation.
    device : {"auto", "cpu", "cuda"}, optional
        Torch device for the accelerated implementation.

    Returns
    -------
    output, valid : tuple[numpy.ndarray, numpy.ndarray]
        Resampled complex image and final validity mask.

    """
    source = np.asarray(complex_radar, dtype=np.complex64)
    azimuth = np.asarray(azimuth, dtype=np.float64)
    range_index = np.asarray(range_index, dtype=np.float64)
    if azimuth.shape != range_index.shape:
        reject_invalid_state("azimuth and range coordinates must have matching shapes")
    height, width = source.shape
    coordinate_valid = (
        np.isfinite(azimuth)
        & np.isfinite(range_index)
        & (azimuth >= 0.0)
        & (azimuth <= height - 1.0)
        & (range_index >= 0.0)
        & (range_index <= width - 1.0)
    )
    if valid is not None:
        if np.asarray(valid).shape != azimuth.shape:
            reject_invalid_state("valid mask must match coordinate shape")
        coordinate_valid &= np.asarray(valid, dtype=bool)

    output = np.full(azimuth.shape, np.nan + 1j * np.nan, dtype=np.complex64)
    if not np.any(coordinate_valid):
        return output, coordinate_valid
    coordinates = np.array(
        [azimuth[coordinate_valid], range_index[coordinate_valid]],
        dtype=np.float64,
    )
    if executor == "dask-torch":
        from faninsar.processing.resampling_torch import (
            lanczos_resample_dask_torch,
        )

        values = lanczos_resample_dask_torch(
            source,
            coordinates,
            a=4,
            mode="constant",
            cval=0.0,
            device=device,
            use_dask=False,
        )
    elif executor == "serial":
        values = lanczos_resample(
            source,
            coordinates,
            a=4,
            mode="constant",
            cval=0.0,
        )
    else:
        reject_invalid_state(f"unknown geocode executor: {executor}")
    output[coordinate_valid] = np.asarray(values, dtype=np.complex64)
    return output, coordinate_valid


def compose_secondary_coordinates(
    reference_lut: Geo2RdrLUT,
    offsets: OffsetFieldResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose reference geo2rdr coordinates with dense secondary offsets.

    Parameters
    ----------
    reference_lut : Geo2RdrLUT
        Reference radar coordinates on the geographic grid.
    offsets : OffsetFieldResult
        Dense offsets using ``offset = reference - secondary``.

    Returns
    -------
    secondary_azimuth, secondary_range, valid : tuple[numpy.ndarray, ...]
        Fractional secondary source coordinates and validity mask.

    """
    coordinates = np.array(
        [reference_lut.az_full.ravel(), reference_lut.rg_full.ravel()],
        dtype=np.float64,
    )
    azimuth_offset = map_coordinates(
        np.asarray(offsets.azimuth_offset_px, dtype=np.float32),
        coordinates,
        order=1,
        mode="constant",
        cval=np.nan,
    ).reshape(reference_lut.shape)
    range_offset = map_coordinates(
        np.asarray(offsets.range_offset_px, dtype=np.float32),
        coordinates,
        order=1,
        mode="constant",
        cval=np.nan,
    ).reshape(reference_lut.shape)
    coverage = map_coordinates(
        np.asarray(offsets.coverage, dtype=np.uint8),
        coordinates,
        order=0,
        mode="constant",
        cval=0,
    ).reshape(reference_lut.shape)
    secondary_azimuth = reference_lut.az_full - azimuth_offset
    secondary_range = reference_lut.rg_full - range_offset
    height, width = offsets.coverage.shape
    valid = (
        reference_lut.valid
        & coverage.astype(bool)
        & np.isfinite(secondary_azimuth)
        & np.isfinite(secondary_range)
        & (secondary_azimuth >= 0.0)
        & (secondary_azimuth <= height - 1.0)
        & (secondary_range >= 0.0)
        & (secondary_range <= width - 1.0)
    )
    return secondary_azimuth, secondary_range, valid


def _multilook_coordinates(
    source_shape: tuple[int, int],
    lut: Geo2RdrLUT,
    full_radar_shape: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full_height, full_width = full_radar_shape or lut.full_radar_shape
    source_height, source_width = source_shape
    azimuth = lut.az_full / max(full_height / max(source_height, 1), 1.0)
    range_index = lut.rg_full / max(full_width / max(source_width, 1), 1.0)
    valid = (
        lut.valid
        & np.isfinite(azimuth)
        & np.isfinite(range_index)
        & (azimuth >= 0.0)
        & (azimuth <= source_height - 1.0)
        & (range_index >= 0.0)
        & (range_index <= source_width - 1.0)
    )
    return azimuth, range_index, valid


def apply_lut_complex(
    complex_radar: np.ndarray,
    lut: Geo2RdrLUT,
    *,
    full_radar_shape: tuple[int, int] | None = None,
    executor: str = "serial",
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Lanczos-resample a complex radar field through a shared LUT.

    Parameters
    ----------
    complex_radar : numpy.ndarray
        Full-resolution or multilooked complex radar field.
    lut : Geo2RdrLUT
        Geographic-to-radar lookup table.
    full_radar_shape : tuple[int, int], optional
        Full-resolution shape used to infer look factors.
    executor : {"serial", "dask-torch"}, optional
        Resampling backend.
    device : {"auto", "cpu", "cuda"}, optional
        Torch device for the ``dask-torch`` backend.

    Returns
    -------
    output, valid : tuple[numpy.ndarray, numpy.ndarray]
        Geocoded complex field and validity mask.

    """
    azimuth, range_index, valid = _multilook_coordinates(
        complex_radar.shape,
        lut,
        full_radar_shape,
    )
    return resample_complex_at_coordinates(
        complex_radar,
        azimuth,
        range_index,
        valid=valid,
        executor=executor,
        device=device,
    )


def apply_lut_real(
    real_radar: np.ndarray,
    lut: Geo2RdrLUT,
    *,
    full_radar_shape: tuple[int, int] | None = None,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a real radar field through a shared LUT.

    Parameters
    ----------
    real_radar : numpy.ndarray
        Full-resolution or multilooked real radar field.
    lut : Geo2RdrLUT
        Geographic-to-radar lookup table.
    full_radar_shape : tuple[int, int], optional
        Full-resolution shape used to infer look factors.
    order : int, optional
        Spline interpolation order passed to SciPy.

    Returns
    -------
    output, valid : tuple[numpy.ndarray, numpy.ndarray]
        Geocoded real field and validity mask.

    """
    azimuth, range_index, valid = _multilook_coordinates(
        real_radar.shape,
        lut,
        full_radar_shape,
    )
    output = np.full(lut.shape, np.nan, dtype=np.float32)
    if not np.any(valid):
        return output, valid
    coordinates = np.array([azimuth[valid], range_index[valid]])
    values = map_coordinates(
        np.asarray(real_radar, dtype=np.float32),
        coordinates,
        order=order,
        mode="constant",
        cval=0.0,
    )
    output[valid] = values.astype(np.float32)
    return output, valid
