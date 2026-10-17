"""Radar-domain coordinate currency shared by geocoding stages.

A :class:`RadarCoordinates` value carries the fractional radar ``(azimuth,
range)`` source coordinates for one destination frame together with its
validity mask. Transforms between frames (dense offset composition, crop
windows, multilook) produce new immutable coordinate values instead of
mutating arrays, so every downstream stage can sample with one vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import map_coordinates

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.runtime.torch_kernels import resolve_torch_device

if TYPE_CHECKING:
    from faninsar.processing.coregistration.offsets import OffsetFieldResult

__all__ = ["RadarCoordinates"]

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class RadarCoordinates:
    """Fractional radar source coordinates for one destination frame.

    Attributes
    ----------
    az, rg : numpy.ndarray
        Zero-based fractional azimuth and range coordinates in the source
        radar frame, one entry per destination pixel. Arrays are stored
        as given (disk-backed memmaps keep their identity).
    valid : numpy.ndarray
        Boolean mask of destination pixels with admissible source
        coordinates.

    """

    az: np.ndarray
    rg: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        """Reject non-2-D or shape-mismatched coordinate arrays."""
        shape = np.shape(self.az)
        if len(shape) != 2:
            reject_invalid_state("RadarCoordinates arrays must be 2-D")
        if np.shape(self.rg) != shape or np.shape(self.valid) != shape:
            reject_invalid_state(
                "RadarCoordinates az, rg, and valid must share one shape"
            )
        if not np.issubdtype(np.asarray(self.valid).dtype, np.bool_):
            reject_invalid_state("RadarCoordinates valid mask must be boolean")

    @property
    def shape(self) -> tuple[int, int]:
        """Return the destination frame shape."""
        shape = np.shape(self.valid)
        return (int(shape[0]), int(shape[1]))

    def minus_offsets(
        self,
        offsets: OffsetFieldResult,
        *,
        device: str = "auto",
    ) -> RadarCoordinates:
        """Compose these reference coordinates with dense secondary offsets.

        Secondary radar coordinates follow ``reference - offset``; the dense
        reference-minus-secondary offset field is bilinearly interpolated at
        these coordinates. This is an interpolation of an existing offset
        field, not a second geo2rdr solve.

        Parameters
        ----------
        offsets : OffsetFieldResult
            Dense offsets with ``offset = reference - secondary`` and a
            coverage mask over the secondary radar raster.
        device : str, optional
            Execution device for the offset interpolation. ``auto`` resolves
            through the standard device policy; ``cpu`` uses SciPy bilinear
            ``map_coordinates`` and ``cuda*`` uses Torch ``grid_sample`` on
            the same device. Other devices fail closed instead of silently
            falling back to CPU.

        Returns
        -------
        RadarCoordinates
            Secondary coordinates with validity narrowed by the offset
            coverage, finiteness, and the secondary radar window.

        """
        resolved = str(resolve_torch_device(device))
        if resolved.startswith("cuda"):
            azimuth_offset, range_offset, coverage = _offset_fields_torch(
                offsets,
                np.asarray(self.az, dtype=np.float64),
                np.asarray(self.rg, dtype=np.float64),
                resolved,
            )
        elif resolved == "cpu":
            azimuth_offset, range_offset, coverage = _offset_fields_numpy(
                offsets,
                np.asarray(self.az, dtype=np.float64),
                np.asarray(self.rg, dtype=np.float64),
            )
        else:
            reject_invalid_state(
                "offset composition is published for cpu/cuda devices only; "
                f"got {resolved!r}"
            )
        return _subtract_offsets(self, offsets, azimuth_offset, range_offset, coverage)

    def to_layer_frame(
        self,
        *,
        origin: tuple[int, int] = (0, 0),
        looks: tuple[int, int] = (1, 1),
    ) -> RadarCoordinates:
        """Map full-radar coordinates into a cropped, multilooked layer frame.

        The layer pixel ``j`` covers full-radar samples
        ``[origin + j * L, origin + (j + 1) * L)``; its box-average center
        sits at ``origin + j * L + (L - 1) / 2`` (the convention of
        :func:`faninsar.processing.runtime.torch_kernels.multilook_interferogram_torch`).
        Inverting that mapping gives ``(x - origin - (L - 1) / 2) / L`` for
        odd and even look factors alike.

        Parameters
        ----------
        origin : tuple[int, int], optional
            ``(row, column)`` origin of the cropped window in the full
            radar raster.
        looks : tuple[int, int], optional
            Positive ``(azimuth, range)`` look factors of the layer.

        Returns
        -------
        RadarCoordinates
            Coordinates in the layer frame. Validity narrows to finite
            coordinates; the per-layer in-window check happens at sampling
            time against the concrete layer shape.

        """
        row0, col0 = _validate_non_negative_ints(origin, "origin")
        az_looks, rg_looks = _validate_positive_ints(looks, "looks")
        azimuth = (
            np.asarray(self.az, dtype=np.float64) - row0 - (az_looks - 1) / 2.0
        ) / az_looks
        range_index = (
            np.asarray(self.rg, dtype=np.float64) - col0 - (rg_looks - 1) / 2.0
        ) / rg_looks
        valid = (
            np.asarray(self.valid, dtype=bool)
            & np.isfinite(azimuth)
            & np.isfinite(range_index)
        )
        return RadarCoordinates(azimuth, range_index, valid)


def _validate_positive_ints(
    values: tuple[int, int],
    name: str,
) -> tuple[int, int]:
    """Return ``(a, b)`` when both entries are positive integers."""
    if len(values) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in values
    ):
        reject_invalid_state(f"{name} must be two positive integers; got {values!r}")
    return int(values[0]), int(values[1])


def _validate_non_negative_ints(
    values: tuple[int, int],
    name: str,
) -> tuple[int, int]:
    """Return ``(a, b)`` when both entries are non-negative integers."""
    if len(values) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        reject_invalid_state(
            f"{name} must be two non-negative integers; got {values!r}"
        )
    return int(values[0]), int(values[1])


def _offset_fields_numpy(
    offsets: OffsetFieldResult,
    azimuth: np.ndarray,
    range_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU bilinear offset interpolation used as the identity oracle."""
    coordinates = np.array([azimuth.ravel(), range_index.ravel()], dtype=np.float64)
    frame_shape = azimuth.shape
    azimuth_offset = map_coordinates(
        np.asarray(offsets.azimuth_offset_px, dtype=np.float32),
        coordinates,
        order=1,
        mode="constant",
        cval=np.nan,
    ).reshape(frame_shape)
    range_offset = map_coordinates(
        np.asarray(offsets.range_offset_px, dtype=np.float32),
        coordinates,
        order=1,
        mode="constant",
        cval=np.nan,
    ).reshape(frame_shape)
    coverage = map_coordinates(
        np.asarray(offsets.coverage, dtype=np.uint8),
        coordinates,
        order=0,
        mode="constant",
        cval=0,
    ).reshape(frame_shape)
    return azimuth_offset, range_offset, coverage


def _offset_fields_torch(
    offsets: OffsetFieldResult,
    azimuth: np.ndarray,
    range_index: np.ndarray,
    identity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same-device bilinear offset interpolation for an admitted accelerator."""
    import torch
    import torch.nn.functional as torch_functional

    torch_device = torch.device(identity)
    radar_h, radar_w = offsets.coverage.shape
    az = torch.as_tensor(azimuth, device=torch_device)
    rg = torch.as_tensor(range_index, device=torch_device)
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
    return (
        azimuth_offset.detach().cpu().numpy(),
        range_offset.detach().cpu().numpy(),
        (coverage > 0.5).detach().cpu().numpy(),
    )


def _subtract_offsets(
    coordinates: RadarCoordinates,
    offsets: OffsetFieldResult,
    azimuth_offset: np.ndarray,
    range_offset: np.ndarray,
    coverage: np.ndarray,
) -> RadarCoordinates:
    """Apply the reference-minus-secondary convention and window mask."""
    secondary_az = np.asarray(coordinates.az) - azimuth_offset
    secondary_rg = np.asarray(coordinates.rg) - range_offset
    height, width = offsets.coverage.shape
    valid = (
        np.asarray(coordinates.valid, dtype=bool)
        & np.asarray(coverage, dtype=bool)
        & np.isfinite(secondary_az)
        & np.isfinite(secondary_rg)
        & (secondary_az >= 0.0)
        & (secondary_az <= height - 1.0)
        & (secondary_rg >= 0.0)
        & (secondary_rg <= width - 1.0)
    )
    return RadarCoordinates(secondary_az, secondary_rg, valid)


def _minus_offsets_torch(
    coordinates: RadarCoordinates,
    offsets: OffsetFieldResult,
    identity: str,
) -> RadarCoordinates:
    """Compose offsets through the Torch path on an explicit device identity.

    Exposed for oracle tests: the Torch grid_sample path is validated
    against the SciPy bilinear oracle by running it on ``"cpu"``.
    """
    azimuth_offset, range_offset, coverage = _offset_fields_torch(
        offsets,
        np.asarray(coordinates.az, dtype=np.float64),
        np.asarray(coordinates.rg, dtype=np.float64),
        identity,
    )
    return _subtract_offsets(
        coordinates,
        offsets,
        azimuth_offset,
        range_offset,
        coverage,
    )
