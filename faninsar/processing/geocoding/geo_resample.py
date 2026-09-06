"""Resample radar fields through geographic lookup tables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.ndimage import map_coordinates

from faninsar.logging import setup_logger
from faninsar.processing.coregistration.resampling import lanczos_resample
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from faninsar.processing.coregistration.offsets import OffsetFieldResult
    from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT

__all__ = [
    "apply_lut_complex",
    "apply_lut_real",
    "compose_secondary_coordinates",
    "resample_complex_at_coordinates",
]

logger = setup_logger(__name__)


def _resample_complex_opencv(
    source: np.ndarray,
    azimuth: np.ndarray,
    range_index: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Resample one dense coordinate tile with OpenCV Lanczos-4."""
    try:
        import cv2
    except ImportError as error:
        message = (
            "the OpenCV geocode executor requires opencv-python; "
            "install FanInSAR with the 'opencv' extra"
        )
        logger.exception(message)
        raise ImportError(message) from error

    azimuth_map = np.where(valid, azimuth, -1.0).astype(np.float32)
    range_map = np.where(valid, range_index, -1.0).astype(np.float32)
    real = cv2.remap(
        source.real,
        range_map,
        azimuth_map,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    imaginary = cv2.remap(
        source.imag,
        range_map,
        azimuth_map,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    output = (real + 1j * imaginary).astype(np.complex64)
    output[~valid] = np.complex64(np.nan + 1j * np.nan)
    return output


def resample_complex_at_coordinates(
    complex_radar: np.ndarray,
    azimuth: np.ndarray,
    range_index: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    executor: Literal["torch", "opencv"] = "torch",
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
    executor : {"torch", "opencv"}, optional
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
    if executor == "opencv":
        return (
            _resample_complex_opencv(
                source,
                azimuth,
                range_index,
                coordinate_valid,
            ),
            coordinate_valid,
        )
    coordinates = np.array(
        [azimuth[coordinate_valid], range_index[coordinate_valid]],
        dtype=np.float64,
    )
    if executor == "torch":
        values = lanczos_resample(
            source,
            coordinates,
            a=4,
            mode="constant",
            cval=0.0,
            device=device,
        )
    else:
        reject_invalid_state(f"unknown geocode executor: {executor}")
    output[coordinate_valid] = np.asarray(values, dtype=np.complex64)
    return output, coordinate_valid


def compose_secondary_coordinates(
    reference_lut: Geo2RdrLUT,
    offsets: OffsetFieldResult,
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose reference geo2rdr coordinates with dense secondary offsets.

    Secondary radar coordinates follow the reference LUT minus the dense
    reference-minus-secondary offset field. The offset field is interpolated
    onto the geographic LUT; it is not a second geo2rdr solve.

    Parameters
    ----------
    reference_lut : Geo2RdrLUT
        Reference radar coordinates on the geographic grid.
    offsets : OffsetFieldResult
        Dense offsets using ``offset = reference - secondary``.
    device : str, optional
        ``cpu`` uses SciPy bilinear ``map_coordinates``. An admitted CUDA
        identity uses Torch ``grid_sample`` with ``align_corners=True``.

    Returns
    -------
    secondary_azimuth, secondary_range, valid : tuple[numpy.ndarray, ...]
        Fractional secondary source coordinates and validity mask.

    """
    identity = str(device)
    if identity.startswith("cuda"):
        return _compose_secondary_coordinates_torch(reference_lut, offsets, identity)
    return _compose_secondary_coordinates_numpy(reference_lut, offsets)


def _compose_secondary_coordinates_numpy(
    reference_lut: Geo2RdrLUT,
    offsets: OffsetFieldResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU bilinear compose used as the identity oracle."""
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
    return _finalize_composed_coordinates(
        reference_lut, offsets, azimuth_offset, range_offset, coverage
    )


def _compose_secondary_coordinates_torch(
    reference_lut: Geo2RdrLUT,
    offsets: OffsetFieldResult,
    identity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same-device bilinear compose for an admitted CUDA identity."""
    import torch
    import torch.nn.functional as torch_functional

    torch_device = torch.device(identity)
    radar_h, radar_w = offsets.coverage.shape
    az = torch.as_tensor(
        np.asarray(reference_lut.az_full, dtype=np.float64),
        device=torch_device,
    )
    rg = torch.as_tensor(
        np.asarray(reference_lut.rg_full, dtype=np.float64),
        device=torch_device,
    )
    denom_h = max(radar_h - 1, 1)
    denom_w = max(radar_w - 1, 1)
    grid_y = (2.0 * az / denom_h) - 1.0
    grid_x = (2.0 * rg / denom_w) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).to(torch.float32)
    az_field = torch.as_tensor(
        np.asarray(offsets.azimuth_offset_px, dtype=np.float32),
        device=torch_device,
    ).view(1, 1, radar_h, radar_w)
    rg_field = torch.as_tensor(
        np.asarray(offsets.range_offset_px, dtype=np.float32),
        device=torch_device,
    ).view(1, 1, radar_h, radar_w)
    cov_field = torch.as_tensor(
        np.asarray(offsets.coverage, dtype=np.float32),
        device=torch_device,
    ).view(1, 1, radar_h, radar_w)
    azimuth_offset = (
        torch_functional.grid_sample(
            az_field, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        .squeeze(0)
        .squeeze(0)
        .to(torch.float64)
    )
    range_offset = (
        torch_functional.grid_sample(
            rg_field, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        .squeeze(0)
        .squeeze(0)
        .to(torch.float64)
    )
    coverage = (
        torch_functional.grid_sample(
            cov_field, grid, mode="nearest", padding_mode="zeros", align_corners=True
        )
        .squeeze(0)
        .squeeze(0)
    )
    finite_src = torch.isfinite(az) & torch.isfinite(rg)
    azimuth_offset = torch.where(
        finite_src, azimuth_offset, torch.full_like(azimuth_offset, float("nan"))
    )
    range_offset = torch.where(
        finite_src, range_offset, torch.full_like(range_offset, float("nan"))
    )
    return _finalize_composed_coordinates(
        reference_lut,
        offsets,
        azimuth_offset.detach().cpu().numpy(),
        range_offset.detach().cpu().numpy(),
        (coverage > 0.5).detach().cpu().numpy(),
    )


def _finalize_composed_coordinates(
    reference_lut: Geo2RdrLUT,
    offsets: OffsetFieldResult,
    azimuth_offset: np.ndarray,
    range_offset: np.ndarray,
    coverage: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the reference-minus-secondary offset convention and in-window mask."""
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
    executor: Literal["torch", "opencv"] = "torch",
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
    executor : {"torch", "opencv"}, optional
        Resampling backend.
    device : {"auto", "cpu", "cuda"}, optional
        Torch device for the unified Torch backend.

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
