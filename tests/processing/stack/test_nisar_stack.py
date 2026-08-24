"""Focused NISAR RSLC Stack seam tests (PROPOSAL-0035)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from faninsar.processing.contracts import (
    CalibrationState,
    CarrierState,
    CoregistrationState,
    OrbitMetadata,
    OrbitStateVector,
    SLCProduct,
)
from faninsar.processing.coordinates import (
    ArrayDescriptor,
    ArrayRepresentation,
    RadarGrid,
)
from faninsar.processing.readers import (
    DopplerCentroidPolynomial,
    SLCReadResult,
    ValidSampleMask,
)
from faninsar.processing.stack import NISARStack


def _result(path: Path, date_id: str) -> SLCReadResult:
    """Build one metadata-only normalized RSLC result for the seam test."""
    sensing_start = datetime.strptime(date_id, "%Y%m%d").replace(tzinfo=UTC)
    grid = RadarGrid(
        shape=(4, 5),
        starting_slant_range_m=800_000.0,
        range_spacing_m=2.3,
        sensing_start=sensing_start,
        azimuth_time_interval_s=0.002,
        wavelength_m=0.24,
        look_direction="right",
    )
    orbit_time = (sensing_start, sensing_start.replace(second=1))
    product = SLCProduct(
        acquisition_id=date_id,
        grid=grid,
        samples=ArrayDescriptor(
            uri=f"hdf5://{path}#/science/LSAR/RSLC/swaths/frequencyB/HH",
            shape=grid.shape,
            dtype="complex64",
            representation=ArrayRepresentation.COMPLEX,
        ),
        orbit=OrbitMetadata(
            "ITRF",
            "nisar-rslc",
            tuple(
                OrbitStateVector(item, (7_000_000.0, 0.0, 0.0), (0.0, 7_500.0, 0.0))
                for item in orbit_time
            ),
        ),
        carrier=CarrierState.PRESENT,
        coregistration=CoregistrationState.NOT_REGISTERED,
        calibration=CalibrationState.RAW_DN,
    )
    return SLCReadResult(
        product=product,
        valid_samples=ValidSampleMask((0, 0, 0, 0), (4, 4, 4, 4)),
        doppler_centroid=DopplerCentroidPolynomial((100.0, -0.1), 800_000.0, 0.0),
        source_path=str(path),
        mission_native={
            "mission": "NISAR",
            "product": "RSLC",
            "frequency": "B",
            "polarization": "HH",
        },
    )


def test_nisar_stack_builds_shared_catalog_and_pair_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three RSLCs become one shared Stack catalog and short-baseline graph."""
    paths = tuple(
        tmp_path / f"NISAR_RSLC_{date_id}.h5"
        for date_id in ("20240101", "20240113", "20240125")
    )
    handles = {path: SimpleNamespace(filename=str(path)) for path in paths}
    monkeypatch.setattr(
        "faninsar.processing.stack.nisar.NisarSensor.open_product",
        lambda _sensor, uri: handles[Path(uri)],
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "faninsar.processing.stack.nisar.NisarSensor.to_slc_product",
        lambda _sensor, handle, **kwargs: (
            calls.append(kwargs)
            or _result(
                Path(handle.filename), Path(handle.filename).stem.rsplit("_", 1)[-1]
            )
        ),
    )

    stack = NISARStack.from_rslc(paths, work_dir=tmp_path / "work")

    assert stack.master == "20240101"
    assert stack.catalog.dates == ("20240101", "20240113", "20240125")
    assert len(stack.pairs) == 3
    assert stack.channel == ("B", "HH")
    assert stack.source_lineage["20240113"] == str(paths[1])
    assert (
        stack.products["20240125"].samples.representation is ArrayRepresentation.COMPLEX
    )
    assert stack.config.extra["mission"] == "NISAR"
    assert all(
        call["frequency"] == "B" and call["polarization"] == "HH" for call in calls
    )


def test_nisar_stack_rejects_duplicate_acquisitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared catalog cannot collapse two RSLCs for one acquisition."""
    paths = (
        tmp_path / "NISAR_RSLC_20240101_a.h5",
        tmp_path / "NISAR_RSLC_20240101_b.h5",
    )
    handles = {path: SimpleNamespace(filename=str(path)) for path in paths}
    monkeypatch.setattr(
        "faninsar.processing.stack.nisar.NisarSensor.open_product",
        lambda _sensor, uri: handles[Path(uri)],
    )
    monkeypatch.setattr(
        "faninsar.processing.stack.nisar.NisarSensor.to_slc_product",
        lambda _sensor, handle, **_: _result(Path(handle.filename), "20240101"),
    )

    with pytest.raises(ValueError, match="one RSLC per acquisition"):
        NISARStack.from_rslc(paths, work_dir=tmp_path / "work")


def test_nisar_stack_fails_closed_before_shared_s1_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsupported processing stages never dispatch RSLC paths to S1 code."""
    paths = tuple(
        tmp_path / f"NISAR_RSLC_{date_id}.h5" for date_id in ("20240101", "20240113")
    )
    handles = {path: SimpleNamespace(filename=str(path)) for path in paths}
    monkeypatch.setattr(
        "faninsar.processing.stack.nisar.NisarSensor.open_product",
        lambda _sensor, uri: handles[Path(uri)],
    )
    monkeypatch.setattr(
        "faninsar.processing.stack.nisar.NisarSensor.to_slc_product",
        lambda _sensor, handle, **_: _result(
            Path(handle.filename), Path(handle.filename).stem.rsplit("_", 1)[-1]
        ),
    )
    stack = NISARStack.from_rslc(paths, work_dir=tmp_path / "work")

    with pytest.raises(NotImplementedError, match="NISAR RSLC Stack"):
        stack.coregister_scenes()
    with pytest.raises(NotImplementedError, match="NISAR RSLC Stack"):
        stack.form_interferograms()
