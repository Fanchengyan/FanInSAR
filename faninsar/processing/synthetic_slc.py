"""Synthetic mission-neutral SLC reader for contract tests and fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np  # noqa: TC002

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.logging import setup_logger
from faninsar.processing.interferometry.products import (
    CalibrationState,
    CarrierState,
    CoregistrationState,
)
from faninsar.processing.slc.products import SLCProduct

from .coordinates import ArrayDescriptor, ArrayRepresentation, RadarGrid
from .readers import (
    DopplerCentroidPolynomial,
    SLCReadResult,
    ValidSampleMask,
    require_critical_metadata,
)

if TYPE_CHECKING:
    from pathlib import Path


logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class SyntheticSLCSpec:
    """In-memory specification for a synthetic SLC product."""

    acquisition_id: str
    samples: np.ndarray
    shape: tuple[int, int]
    source_path: str
    array_uri: str
    doppler_centroid: DopplerCentroidPolynomial | None
    valid_samples: ValidSampleMask | None
    mission_native: dict[str, str]


@dataclass(slots=True)
class SyntheticSLCReader:
    """Open synthetic SLCs backed by a provided ArrayReader store."""

    store: Any
    specs: dict[str, SyntheticSLCSpec]

    def open(self, source_path: str) -> SLCReadResult:
        """Validate critical metadata and return a typed lazy SLC product.

        Parameters
        ----------
        source_path : str
            Synthetic product path registered with this reader.

        Returns
        -------
        SLCReadResult
            Product metadata plus critical mission-native fields.

        Raises
        ------
        KeyError
            If the source path is unknown.
        MissingCriticalMetadataError
            If Doppler or valid-sample metadata is absent.

        """
        spec = self.specs.get(source_path)
        if spec is None:
            message = f"unknown synthetic SLC source: {source_path}"
            logger.error(message)
            raise KeyError(message)
        doppler, valid = require_critical_metadata(
            source_path=source_path,
            doppler_centroid=spec.doppler_centroid,
            valid_samples=spec.valid_samples,
        )
        grid = RadarGrid(
            shape=spec.shape,
            starting_slant_range_m=800_000.0,
            range_spacing_m=2.3,
            sensing_start=datetime(2024, 1, 1, tzinfo=UTC),
            azimuth_time_interval_s=0.002,
            wavelength_m=0.0555,
            look_direction="right",
        )
        samples = ArrayDescriptor(
            uri=spec.array_uri,
            shape=spec.shape,
            dtype=str(spec.samples.dtype),
            representation=ArrayRepresentation.COMPLEX,
        )
        orbit = OrbitMetadata(
            reference_frame="ITRF",
            source="synthetic",
            vectors=(
                OrbitStateVector(
                    time=datetime(2024, 1, 1, tzinfo=UTC),
                    position_m=(7_000_000.0, 0.0, 0.0),
                    velocity_m_s=(0.0, 7_500.0, 0.0),
                ),
            ),
        )
        product = SLCProduct(
            acquisition_id=spec.acquisition_id,
            grid=grid,
            samples=samples,
            orbit=orbit,
            carrier=CarrierState.PRESENT,
            coregistration=CoregistrationState.NOT_REGISTERED,
            calibration=CalibrationState.RAW_DN,
        )
        return SLCReadResult(
            product=product,
            valid_samples=valid,
            doppler_centroid=doppler,
            source_path=source_path,
            mission_native=dict(spec.mission_native),
        )


def write_synthetic_slc(
    store: object,
    *,
    source_path: str,
    acquisition_id: str,
    samples: np.ndarray,
    include_doppler: bool = True,
    include_valid_samples: bool = True,
) -> SyntheticSLCSpec:
    """Write synthetic samples into a store and return the openable spec.

    Parameters
    ----------
    store : object
        ArrayWriter-compatible store with a ``write`` method.
    source_path : str
        URI used both as the store key and reader path.
    acquisition_id : str
        Acquisition identity attached to the product.
    samples : numpy.ndarray
        Complex two-dimensional sample array.
    include_doppler : bool, optional
        When false, omit Doppler metadata to exercise rejection.
    include_valid_samples : bool, optional
        When false, omit valid-sample windows to exercise rejection.

    Returns
    -------
    SyntheticSLCSpec
        Spec suitable for registration with :class:`SyntheticSLCReader`.

    """
    height, width = (int(samples.shape[0]), int(samples.shape[1]))
    descriptor = store.write(
        source_path,
        samples,
        metadata={
            "acquisition_id": acquisition_id,
            "mission": "synthetic",
        },
    )
    doppler = (
        DopplerCentroidPolynomial(
            coefficients_hz=(100.0, -0.01),
            reference_range_m=800_000.0,
            reference_time_s=0.0,
        )
        if include_doppler
        else None
    )
    valid = (
        ValidSampleMask(
            first_valid_sample=tuple(0 for _ in range(height)),
            last_valid_sample=tuple(width - 1 for _ in range(height)),
        )
        if include_valid_samples
        else None
    )
    return SyntheticSLCSpec(
        acquisition_id=acquisition_id,
        samples=samples,
        shape=(height, width),
        source_path=source_path,
        array_uri=descriptor.uri,
        doppler_centroid=doppler,
        valid_samples=valid,
        mission_native={"mission": "synthetic", "schema": "v1"},
    )


def register_specs(
    *specs: SyntheticSLCSpec,
) -> dict[str, SyntheticSLCSpec]:
    """Index synthetic specs by source path."""
    return {spec.source_path: spec for spec in specs}


def ensure_parent(path: Path) -> Path:
    """Create parent directories for a path and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
