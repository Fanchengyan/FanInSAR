"""Sensor adapter port — mission I/O boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from faninsar.logging import reject
from faninsar.processing.errors import InvalidProcessingStateError

if TYPE_CHECKING:
    from faninsar.processing.geometry.coordinates import Shape2D
    from faninsar.processing.slc.products import SLCProduct


@runtime_checkable
class SensorAdapter(Protocol):
    """Mission-neutral adapter for opening and reading SLC products.

    Implementations live under ``faninsar.missions``. Processing stages never
    import missions directly; they consume ``SLCProduct`` handles only.
    """

    name: str

    def open_product(self, uri: str, **kwargs: Any) -> Any:
        """Open a mission product at *uri* and return a handle."""
        ...

    def to_slc_product(self, handle: Any, **kwargs: Any) -> Any:
        """Convert an open handle into a mission-neutral SLCProduct."""
        ...

    def read_slc_window(self, handle: Any, window: Any, **kwargs: Any) -> Any:
        """Read a spatial window from *handle* as a complex ndarray."""
        ...


@dataclass(frozen=True, slots=True)
class ValidSampleMask:
    """Per-line valid-sample window in radar coordinates."""

    first_valid_sample: tuple[int, ...]
    last_valid_sample: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate window lengths and inclusive sample bounds."""
        if len(self.first_valid_sample) != len(self.last_valid_sample):
            message = "valid-sample window lengths must match"
            reject(message, ValueError)
        if not self.first_valid_sample:
            message = "valid-sample windows must not be empty"
            reject(message, ValueError)
        for first, last in zip(
            self.first_valid_sample,
            self.last_valid_sample,
            strict=True,
        ):
            if first < 0 or last < first:
                message = "valid-sample bounds must satisfy 0 <= first <= last"
                reject(message, ValueError)


@dataclass(frozen=True, slots=True)
class DopplerCentroidPolynomial:
    """Range-dependent Doppler centroid polynomial coefficients."""

    coefficients_hz: tuple[float, ...]
    reference_range_m: float
    reference_time_s: float

    def __post_init__(self) -> None:
        """Validate that the Doppler polynomial is non-empty."""
        if not self.coefficients_hz:
            message = "Doppler centroid polynomial requires coefficients"
            reject(message, ValueError)


@dataclass(frozen=True, slots=True)
class SLCReadResult:
    """Typed product plus mission-native critical metadata for one SLC."""

    product: SLCProduct
    valid_samples: ValidSampleMask
    doppler_centroid: DopplerCentroidPolynomial
    source_path: str
    mission_native: dict[str, str]


@runtime_checkable
class SLCReader(Protocol):
    """Open a mission-neutral SLC product without materializing full arrays."""

    def open(self, source_path: str) -> SLCReadResult:
        """Parse metadata and return a lazy-backed SLC product."""


def require_critical_metadata(
    *,
    source_path: str,
    doppler_centroid: DopplerCentroidPolynomial | None,
    valid_samples: ValidSampleMask | None,
) -> tuple[DopplerCentroidPolynomial, ValidSampleMask]:
    """Reject missing Doppler or valid-sample metadata before array access.

    Parameters
    ----------
    source_path : str
        Product path or URI used in diagnostics.
    doppler_centroid : DopplerCentroidPolynomial or None
        Parsed Doppler polynomial, if present.
    valid_samples : ValidSampleMask or None
        Parsed valid-sample windows, if present.

    Returns
    -------
    tuple[DopplerCentroidPolynomial, ValidSampleMask]
        Validated critical metadata pair.

    Raises
    ------
    InvalidProcessingStateError
        If either required field is absent.

    """
    if doppler_centroid is None:
        message = (
            "missing critical metadata 'doppler_centroid' at "
            f"{source_path}: restore the mission Doppler centroid polynomial "
            "from annotation metadata before opening the SLC"
        )
        reject(message, InvalidProcessingStateError)
    if valid_samples is None:
        message = (
            "missing critical metadata 'valid_samples' at "
            f"{source_path}: restore first/last valid sample windows from "
            "annotation metadata before opening the SLC"
        )
        reject(message, InvalidProcessingStateError)
    return doppler_centroid, valid_samples


def normalize_selection(
    shape: Shape2D,
    selection: tuple[slice, slice],
) -> tuple[slice, slice]:
    """Normalize a two-dimensional selection against an array shape.

    Parameters
    ----------
    shape : tuple[int, int]
        Full array height and width.
    selection : tuple[slice, slice]
        Requested row and column slices.

    Returns
    -------
    tuple[slice, slice]
        Normalized slices with concrete start/stop values.

    Raises
    ------
    ValueError
        If the selection is out of bounds or uses a non-unit step.

    """
    normalized: list[slice] = []
    for axis, (size, item) in enumerate(zip(shape, selection, strict=True)):
        if item.step not in (None, 1):
            message = f"selection step must be 1 on axis {axis}"
            reject(message, ValueError)
        start, stop, _ = item.indices(size)
        if start < 0 or stop > size or start >= stop:
            message = f"selection out of bounds on axis {axis}: {item!r}"
            reject(message, ValueError)
        normalized.append(slice(start, stop))
    return normalized[0], normalized[1]


__all__ = [
    "DopplerCentroidPolynomial",
    "SLCReadResult",
    "SLCReader",
    "SensorAdapter",
    "ValidSampleMask",
    "normalize_selection",
    "require_critical_metadata",
]
