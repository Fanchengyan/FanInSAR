"""Structured input contract for the on-demand geocoding entry point.

The Stack-level ``geocode`` entry point accepts any object that structurally
conforms to :class:`GeocodeableProduct` (PROPOSAL-0043 extension point);
:class:`RadarDomainProduct` is the canonical frozen carrier for callers that
hold a bare radar-domain array. The contract carries everything the geocoding
engine needs besides the Stack-owned geometry context: the 2-D radar-domain
raster, its physical asset vocabulary, the full-resolution radar frame it
lives in, and the crop/multilook mapping that relates its array indices to
full-radar coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from pathlib import Path

    from faninsar.processing.geocoding.geocoder import GeocodedRaster

__all__ = [
    "GeocodeSession",
    "GeocodeableProduct",
    "RadarDomainProduct",
    "layer_frame_shape",
]

logger = setup_logger(__name__)


class GeocodeableProduct(Protocol):
    """Structural contract for radar-domain products accepted by geocoding.

    Notes
    -----
    All members are attributes; adapters may expose them as plain attributes
    or as properties. ``crs`` is ``None`` for radar-domain products; a map
    ``crs`` marks a product that is already geocoded. It passes through only
    when no output or resampling change is requested. ``asset_kind`` holds
    the mission-neutral vocabulary (a
    :class:`~faninsar.network.products.AssetKind` or its
    string value) that determines the resampling kernel. It is required for
    radar-domain products and may be ``None`` only for an already
    georeferenced product, which is never resampled.

    """

    @property
    def name(self) -> str:
        """Return the layer name recorded in output metadata."""
        ...

    @property
    def values(self) -> np.ndarray:
        """Return the 2-D radar-domain raster to geocode."""
        ...

    @property
    def crs(self) -> object | None:
        """Return the product CRS, or ``None`` for the radar domain."""
        ...

    @property
    def asset_kind(self) -> object:
        """Return the asset vocabulary tag, or ``None`` when unnamed."""
        ...

    @property
    def full_radar_shape(self) -> tuple[int, int]:
        """Return the full-resolution ``(azimuth, range)`` radar shape."""
        ...

    @property
    def multilook(self) -> tuple[int, int]:
        """Return the ``(azimuth, range)`` look factors of ``values``."""
        ...

    @property
    def origin(self) -> tuple[int, int]:
        """Return the ``(row, column)`` crop origin in the full radar frame."""
        ...


class GeocodeSession(Protocol):
    """A session that can geocode a product with its own geometry context.

    ``Stack`` satisfies this protocol structurally. Committed layer handles
    keep an opaque reference to the session that produced them, so
    ``faninsar.network`` can offer ``layer.geocode()`` without importing the
    ``faninsar.stack`` domain.

    """

    def geocode(
        self,
        product: GeocodeableProduct,
        *,
        grid: object = None,
        path: str | Path | None = None,
        overwrite: bool = False,
    ) -> GeocodedRaster:
        """Geocode ``product`` and return its geographic raster."""
        ...


@dataclass(frozen=True, slots=True)
class RadarDomainProduct:
    """Canonical frozen carrier conforming to :class:`GeocodeableProduct`.

    Parameters
    ----------
    name : str
        Human-readable layer name recorded in output metadata. It must be
        a single safe path component and is not a content fingerprint.
    values : numpy.ndarray
        2-D radar-domain raster. Its dtype must match the physical kind
        selected for geocoding (complex for ``complex``, floating point for
        ``phase``/``continuous``, integer or boolean for ``label``).
    full_radar_shape : tuple[int, int]
        Full-resolution ``(azimuth, range)`` radar shape the LUT solves
        against.
    asset_kind : object
        Asset vocabulary tag (``AssetKind`` or string value) that determines
        the resampling kernel. Required for radar-domain products; use
        :attr:`~faninsar.network.products.AssetKind.LABEL` for label masks.
        May be ``None`` only when ``crs`` is set, because an already
        georeferenced product passes through without resampling.
    multilook : tuple[int, int], optional
        Positive ``(azimuth, range)`` look factors of ``values``.
    origin : tuple[int, int], optional
        ``(row, column)`` origin of the cropped window in the full radar
        frame.
    crs : object, optional
        Product CRS; ``None`` for the radar domain, otherwise a map CRS.

    Raises
    ------
    InvalidProcessingStateError
        If the name is not one safe path component, shapes are malformed,
        look factors and origin are not non-negative integers,
        ``values.shape`` does not match the layer frame implied by
        ``full_radar_shape``, ``origin``, and ``multilook`` (see
        :func:`layer_frame_shape`), or a radar-domain product omits
        ``asset_kind``.

    """

    name: str
    values: np.ndarray
    full_radar_shape: tuple[int, int]
    asset_kind: object | None = None
    multilook: tuple[int, int] = (1, 1)
    origin: tuple[int, int] = (0, 0)
    crs: object | None = None

    def __post_init__(self) -> None:
        """Validate the persisted name and the frame/layer geometry."""
        name = self.name
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
        ):
            reject_invalid_state(
                f"product name must be one safe path component; got {name!r}"
            )
        values = np.asarray(self.values)
        if values.ndim != 2:
            reject_invalid_state(
                f"product values must be a 2-D radar raster; got shape {values.shape}"
            )
        full_radar_shape = _validate_pair(
            self.full_radar_shape, "full_radar_shape", positive=True
        )
        multilook = _validate_pair(self.multilook, "multilook", positive=True)
        origin = _validate_pair(self.origin, "origin", positive=False)
        expected = layer_frame_shape(
            full_radar_shape,
            multilook=multilook,
            origin=origin,
        )
        if values.shape != expected:
            reject_invalid_state(
                f"product values shape {values.shape} does not match the layer "
                f"frame implied by full_radar_shape={full_radar_shape}, "
                f"origin={origin}, multilook={multilook} (expected {expected})"
            )
        if self.crs is None and self.asset_kind is None:
            reject_invalid_state(
                "a radar-domain product must declare asset_kind; use "
                "AssetKind.LABEL for label masks, and leave asset_kind None "
                "only for an already georeferenced product"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the radar-layer ``(height, width)``."""
        height, width = np.asarray(self.values).shape
        return (int(height), int(width))


def layer_frame_shape(
    full_radar_shape: tuple[int, int],
    *,
    multilook: tuple[int, int],
    origin: tuple[int, int],
) -> tuple[int, int]:
    """Return the layer shape implied by the crop/multilook frame mapping.

    Layer pixel ``j`` covers full-radar samples
    ``[origin + j * L, origin + (j + 1) * L)`` with box-average centre
    ``origin + j * L + (L - 1) / 2`` — the convention inverted by
    :meth:`faninsar.processing.geocoding.coordinates.RadarCoordinates.to_layer_frame`.
    A trailing partial look block is not covered by any layer pixel, so the
    extent is the floored division of the covered window by the look factors.

    Parameters
    ----------
    full_radar_shape : tuple[int, int]
        Full-resolution ``(azimuth, range)`` radar frame shape.
    multilook : tuple[int, int]
        Positive ``(azimuth, range)`` look factors of the layer.
    origin : tuple[int, int]
        ``(row, column)`` crop origin of the layer window in the full frame.

    Returns
    -------
    tuple[int, int]
        The ``(height, width)`` a frame-consistent layer must have.

    """
    return (
        (full_radar_shape[0] - origin[0]) // multilook[0],
        (full_radar_shape[1] - origin[1]) // multilook[1],
    )


def _validate_pair(
    values: tuple[int, int],
    name: str,
    *,
    positive: bool,
) -> tuple[int, int]:
    """Return ``(a, b)`` when both entries are (positive) integers."""
    if (
        not isinstance(values, tuple)
        or len(values) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        )
    ):
        reject_invalid_state(f"{name} must be two integers; got {values!r}")
    pair = (int(values[0]), int(values[1]))
    if positive and any(value < 1 for value in pair):
        reject_invalid_state(f"{name} must contain positive integers; got {values!r}")
    if not positive and any(value < 0 for value in pair):
        reject_invalid_state(
            f"{name} must contain non-negative integers; got {values!r}"
        )
    return pair
