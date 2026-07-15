"""Mission-neutral SLC reader contracts and critical-metadata validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from faninsar.logging import setup_logger

from .errors import ProcessingContractError

if TYPE_CHECKING:
    from .contracts import SLCProduct
    from .coordinates import Shape2D

logger = setup_logger(__name__)


class MissingCriticalMetadataError(ProcessingContractError):
    """Raised when a reader is missing metadata required for processing."""

    def __init__(self, *, source_path: str, field: str, remedy: str) -> None:
        """Initialize with the source path, missing field, and remedy.

        Parameters
        ----------
        source_path : str
            Path or URI of the product that failed validation.
        field : str
            Missing critical metadata field name.
        remedy : str
            Actionable guidance for restoring the missing field.

        """
        self.source_path = source_path
        self.field = field
        self.remedy = remedy
        message = f"missing critical metadata {field!r} at {source_path}: {remedy}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ValidSampleMask:
    """Per-line valid-sample window in radar coordinates."""

    first_valid_sample: tuple[int, ...]
    last_valid_sample: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate window lengths and inclusive sample bounds."""
        if len(self.first_valid_sample) != len(self.last_valid_sample):
            message = "valid-sample window lengths must match"
            logger.error(message)
            raise ValueError(message)
        if not self.first_valid_sample:
            message = "valid-sample windows must not be empty"
            logger.error(message)
            raise ValueError(message)
        for first, last in zip(
            self.first_valid_sample,
            self.last_valid_sample,
            strict=True,
        ):
            if first < 0 or last < first:
                message = "valid-sample bounds must satisfy 0 <= first <= last"
                logger.error(message)
                raise ValueError(message)


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
            logger.error(message)
            raise ValueError(message)


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
    MissingCriticalMetadataError
        If either required field is absent.

    """
    if doppler_centroid is None:
        error = MissingCriticalMetadataError(
            source_path=source_path,
            field="doppler_centroid",
            remedy=(
                "restore the mission Doppler centroid polynomial from "
                "annotation metadata before opening the SLC"
            ),
        )
        logger.error("%s", error)
        raise error
    if valid_samples is None:
        error = MissingCriticalMetadataError(
            source_path=source_path,
            field="valid_samples",
            remedy=(
                "restore first/last valid sample windows from annotation "
                "metadata before opening the SLC"
            ),
        )
        logger.error("%s", error)
        raise error
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
            logger.error(message)
            raise ValueError(message)
        start, stop, _ = item.indices(size)
        if start < 0 or stop > size or start >= stop:
            message = f"selection out of bounds on axis {axis}: {item!r}"
            logger.error(message)
            raise ValueError(message)
        normalized.append(slice(start, stop))
    return normalized[0], normalized[1]
