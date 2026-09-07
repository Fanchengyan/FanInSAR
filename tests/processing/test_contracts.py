from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.processing.interferometry.products import (
    CalibrationState,
    CarrierState,
    ComplexInterferogram,
    CoregistrationState,
    FlatteningState,
    PairProduct,
    UnwrapResult,
)
from faninsar.processing.slc.products import SLCProduct, StackProduct
from faninsar.processing.geometry.coordinates import (
    ArrayDescriptor,
    ArrayRepresentation,
    CoordinateSystem,
    GeoGrid,
    GridMismatchError,
    OffsetField,
    RadarGrid,
    TransformDirection,
    TransformLUT,
)
from faninsar.processing.errors import InvalidProcessingStateError


def radar_grid() -> RadarGrid:
    return RadarGrid(
        shape=(4, 6),
        starting_slant_range_m=800_000.0,
        range_spacing_m=2.3,
        sensing_start=datetime(2024, 1, 1, tzinfo=UTC),
        azimuth_time_interval_s=0.002,
        wavelength_m=0.0555,
        look_direction="right",
    )


def geo_grid() -> GeoGrid:
    return GeoGrid(
        shape=(4, 6),
        crs="EPSG:32649",
        transform=(10.0, 0.0, 500_000.0, 0.0, -10.0, 3_000_000.0),
    )


def orbit() -> OrbitMetadata:
    vector = OrbitStateVector(
        time=datetime(2024, 1, 1, tzinfo=UTC),
        position_m=(1.0, 2.0, 3.0),
        velocity_m_s=(4.0, 5.0, 6.0),
    )
    return OrbitMetadata(reference_frame="ITRF", source="embedded", vectors=(vector,))


def slc(
    acquisition_id: str,
    *,
    grid: RadarGrid | GeoGrid | None = None,
    coregistration: CoregistrationState = CoregistrationState.REGISTERED,
) -> SLCProduct:
    selected_grid = radar_grid() if grid is None else grid
    return SLCProduct(
        acquisition_id=acquisition_id,
        grid=selected_grid,
        samples=ArrayDescriptor(
            uri=f"memory://{acquisition_id}",
            shape=selected_grid.shape,
            dtype="complex64",
            representation=ArrayRepresentation.COMPLEX,
        ),
        orbit=orbit(),
        carrier=CarrierState.PRESENT,
        coregistration=coregistration,
        calibration=CalibrationState.CALIBRATED,
    )


def test_valid_radar_and_geo_tracks_construct_pair_and_stack() -> None:
    primary = slc("20240101")
    secondary = slc("20240113")
    interferogram = ComplexInterferogram.form(primary, secondary)
    unwrapped = UnwrapResult(
        pair_id=interferogram.pair_id,
        grid=interferogram.grid,
        phase=ArrayDescriptor(
            uri="memory://unwrapped",
            shape=interferogram.grid.shape,
            dtype="float32",
            representation=ArrayRepresentation.PHASE,
        ),
        method="irls",
    )
    pair = PairProduct(interferogram=interferogram, unwrap=unwrapped)
    stack = StackProduct(stack_id="demo", pairs=(pair,))

    assert stack.pairs[0].interferogram.flattening is FlatteningState.NOT_APPLIED

    geo = slc("20240101-geo", grid=geo_grid())
    assert geo.coordinate_system is CoordinateSystem.GEO


def test_rejects_double_deramp(caplog: pytest.LogCaptureFixture) -> None:
    deramped = slc("20240101").deramp("tops-carrier-v1")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(InvalidProcessingStateError, match="already deramped"):
            deramped.deramp("tops-carrier-v1")

    assert "processing state rejected: SLC is already deramped" in caplog.messages


def test_rejects_deramp_id_without_deramped_carrier() -> None:
    base = slc("20240101")

    with pytest.raises(InvalidProcessingStateError, match="cannot have"):
        SLCProduct(
            acquisition_id=base.acquisition_id,
            grid=base.grid,
            samples=base.samples,
            orbit=base.orbit,
            carrier=CarrierState.PRESENT,
            coregistration=base.coregistration,
            calibration=base.calibration,
            deramp_application_id="inconsistent",
        )


def test_rejects_interferogram_from_unregistered_slc() -> None:
    secondary = slc("20240113", coregistration=CoregistrationState.NOT_REGISTERED)

    with pytest.raises(InvalidProcessingStateError, match="registered"):
        ComplexInterferogram.form(slc("20240101"), secondary)


def test_rejects_phase_only_resampling() -> None:
    coordinates = ArrayDescriptor(
        uri="memory://coordinates",
        shape=geo_grid().shape,
        dtype="float64",
        representation=ArrayRepresentation.COORDINATE,
    )
    lut = TransformLUT(
        source=radar_grid(),
        target=geo_grid(),
        direction=TransformDirection.RADAR_TO_GEO,
        range_coordinates=coordinates,
        azimuth_coordinates=coordinates,
    )
    phase = ArrayDescriptor(
        uri="memory://phase",
        shape=radar_grid().shape,
        dtype="float32",
        representation=ArrayRepresentation.PHASE,
    )

    with pytest.raises(InvalidProcessingStateError, match="complex samples"):
        lut.validate_resampling_source(phase)


def test_rejects_mismatched_grids() -> None:
    shifted = GeoGrid(
        shape=(4, 6),
        crs="EPSG:32649",
        transform=(10.0, 0.0, 500_010.0, 0.0, -10.0, 3_000_000.0),
    )

    with pytest.raises(GridMismatchError, match="same grid"):
        ComplexInterferogram.form(
            slc("20240101", grid=geo_grid()), slc("20240113", grid=shifted)
        )


def test_transform_and_offset_fields_require_consistent_shapes() -> None:
    coordinates = ArrayDescriptor(
        uri="memory://coords",
        shape=geo_grid().shape,
        dtype="float64",
        representation=ArrayRepresentation.COORDINATE,
    )
    lut = TransformLUT(
        source=radar_grid(),
        target=geo_grid(),
        direction=TransformDirection.RADAR_TO_GEO,
        range_coordinates=coordinates,
        azimuth_coordinates=coordinates,
    )
    offsets_descriptor = ArrayDescriptor(
        uri="memory://offsets",
        shape=geo_grid().shape,
        dtype="float32",
        representation=ArrayRepresentation.OFFSET,
    )
    offsets = OffsetField(
        grid=geo_grid(),
        range_offsets=offsets_descriptor,
        azimuth_offsets=offsets_descriptor,
    )

    assert lut.target.shape == offsets.grid.shape
