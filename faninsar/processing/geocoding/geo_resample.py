"""Resample radar fields through geographic lookup tables.

The public kernel entry point is :func:`sample`: the resampling kernel is
uniquely determined by the ``kind`` of the quantity being sampled — never
inferred from the array dtype — so that every stage geocodes a given layer
with the same physics:

* ``complex`` — Lanczos-4 windowed sinc on the complex samples (preserves
  phase; bilinear on real/imag attenuates in-band signal and leaks aliasing).
* ``phase`` — cyclic wrapped phase: ``exp(i*phi)`` is resampled with
  Lanczos-4 and the angle of the averaged vector is returned in radians.
* ``continuous`` — SciPy bilinear for smooth real fields (unwrapped phase,
  coherence, displacement, amplitude).
* ``label`` — SciPy nearest for hard integer/boolean classes; interpolation
  would invent invalid categories.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.ndimage import map_coordinates

from faninsar.logging import setup_logger
from faninsar.processing.coregistration.resampling import lanczos_resample
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from faninsar.processing.geocoding.coordinates import RadarCoordinates

__all__ = [
    "GeocodeKind",
    "resample_complex_at_coordinates",
    "sample",
]

logger = setup_logger(__name__)

GeocodeKind = Literal["complex", "phase", "continuous", "label"]

_PHASE_MAGNITUDE_EPSILON = 1e-6


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


def sample(
    layer: np.ndarray,
    coords: RadarCoordinates,
    *,
    kind: GeocodeKind,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one radar layer at radar coordinates with the kernel for ``kind``.

    The kernel is fixed by ``kind``; the layer dtype is validated against the
    physical type and never used to pick the kernel. Samples outside the
    layer window or marked invalid in ``coords`` come back NaN (0 for
    integer labels) with ``valid=False``.

    Parameters
    ----------
    layer : numpy.ndarray
        2-D radar-domain source layer.
    coords : RadarCoordinates
        Fractional source coordinates for the destination frame.
    kind : {"complex", "phase", "continuous", "label"}
        Physical kind selecting the kernel: ``complex`` (Lanczos-4 on the
        complex samples), ``phase`` (cyclic wrapped phase through
        ``exp(i*phi)`` + Lanczos-4 + angle, returned as float32 radians),
        ``continuous`` (bilinear), or ``label`` (nearest).
    device : {"auto", "cpu", "cuda"}, optional
        Torch device for the accelerated Lanczos kernels (``complex`` and
        ``phase``). The SciPy bilinear/nearest backends (``continuous`` and
        ``label``) execute on CPU.

    Returns
    -------
    values, valid : tuple[numpy.ndarray, numpy.ndarray]
        Sampled destination frame and its validity mask.

    Raises
    ------
    InvalidProcessingStateError
        If ``kind`` is unknown or the layer dtype does not match the
        physical type required by ``kind``.

    Notes
    -----
    Unwrapped phase sampled as ``continuous`` is locally unreliable across
    connectivity-component boundaries, where the field jumps by ``2*pi``.

    """
    layer = np.asarray(layer)
    if kind == "complex":
        _validate_dtype(layer, kind)
        return resample_complex_at_coordinates(
            layer,
            coords.az,
            coords.rg,
            valid=coords.valid,
            device=device,
        )
    if kind == "phase":
        _validate_dtype(layer, kind)
        return _sample_phase(layer, coords, device=device)
    if kind == "continuous":
        _validate_dtype(layer, kind)
        return _sample_bilinear(layer, coords)
    if kind != "label":
        reject_invalid_state(f"unknown geocode resampling kind: {kind!r}")
    _validate_dtype(layer, "label")
    return _sample_nearest(layer, coords)


def _validate_dtype(layer: np.ndarray, kind: GeocodeKind) -> None:
    """Reject layers whose dtype does not match the physical kind."""
    if kind == "complex":
        if not np.iscomplexobj(layer):
            reject_invalid_state(
                f"kind='complex' requires a complex-valued layer; got {layer.dtype}"
            )
        return
    if kind in ("phase", "continuous"):
        if not np.issubdtype(layer.dtype, np.floating):
            reject_invalid_state(
                f"kind={kind!r} requires a floating-point layer; got {layer.dtype}"
            )
        return
    if not (np.issubdtype(layer.dtype, np.integer) or layer.dtype == np.bool_):
        reject_invalid_state(
            f"kind='label' requires an integer or boolean layer; got {layer.dtype}"
        )


def _sample_phase(
    layer: np.ndarray,
    coords: RadarCoordinates,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample cyclic wrapped phase through its complex representation.

    Non-finite source phase gets zero magnitude so it contributes nothing to
    the kernel average; the destination is valid only where the averaged
    complex vector retains meaningful magnitude.
    """
    phase = np.asarray(layer, dtype=np.float32)
    finite = np.isfinite(phase)
    complex_source = np.where(
        finite,
        np.exp(1j * phase.astype(np.float64)),
        0.0 + 0.0j,
    ).astype(np.complex64)
    complex_values, valid = resample_complex_at_coordinates(
        complex_source,
        coords.az,
        coords.rg,
        valid=coords.valid,
        device=device,
    )
    valid = valid & (np.abs(complex_values) > _PHASE_MAGNITUDE_EPSILON)
    values = np.full(complex_values.shape, np.nan, dtype=np.float32)
    values[valid] = np.angle(complex_values[valid]).astype(np.float32)
    return values, valid


def _sample_bilinear(
    layer: np.ndarray,
    coords: RadarCoordinates,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear (``map_coordinates`` order 1) sampling for smooth fields."""
    values = np.full(coords.shape, np.nan, dtype=np.float32)
    valid = _admissible(layer.shape, coords)
    if not np.any(valid):
        return values, valid
    coordinates = np.array(
        [np.asarray(coords.az)[valid], np.asarray(coords.rg)[valid]],
        dtype=np.float64,
    )
    sampled = map_coordinates(
        np.asarray(layer, dtype=np.float32),
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
    )
    values[valid] = sampled.astype(np.float32)
    return values, valid


def _sample_nearest(
    layer: np.ndarray,
    coords: RadarCoordinates,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest (``map_coordinates`` order 0) sampling for hard labels.

    The layer is passed to SciPy in its own integer/boolean dtype: a
    ``float32`` round trip would silently round label magnitudes above
    ``2**24`` (SciPy keeps integer dtypes exactly up to the ``float64``
    integer range).
    """
    values = np.zeros(coords.shape, dtype=layer.dtype)
    valid = _admissible(layer.shape, coords)
    if not np.any(valid):
        return values, valid
    coordinates = np.array(
        [np.asarray(coords.az)[valid], np.asarray(coords.rg)[valid]],
        dtype=np.float64,
    )
    sampled = map_coordinates(
        np.asarray(layer),
        coordinates,
        order=0,
        mode="constant",
        cval=0,
    )
    values[valid] = sampled.astype(layer.dtype, copy=False)
    return values, valid


def _admissible(
    layer_shape: tuple[int, int],
    coords: RadarCoordinates,
) -> np.ndarray:
    """Return the in-window, finite subset of the coordinate validity mask."""
    azimuth = np.asarray(coords.az)
    range_index = np.asarray(coords.rg)
    height, width = layer_shape
    return (
        np.asarray(coords.valid, dtype=bool)
        & np.isfinite(azimuth)
        & np.isfinite(range_index)
        & (azimuth >= 0.0)
        & (azimuth <= height - 1.0)
        & (range_index >= 0.0)
        & (range_index <= width - 1.0)
    )
