"""Focused NISAR RSLC Stack seam tests (PROPOSAL-0035)."""

from __future__ import annotations

from dataclasses import replace
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
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.readers import (
    DopplerCentroidPolynomial,
    SLCReadResult,
    ValidSampleMask,
)
from faninsar.processing.stack import NISARStack
from faninsar.processing.stack.provider import UnsupportedStackCapabilityError


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


def test_nisar_stack_passes_explicit_admission_metadata_to_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stack configuration cannot silently bypass trusted source admission."""
    paths = tuple(
        tmp_path / f"NISAR_RSLC_{date_id}.h5" for date_id in ("20240101", "20240113")
    )
    handles = {path: SimpleNamespace(filename=str(path)) for path in paths}
    admissions: list[object] = []

    def open_product(_sensor: object, uri: str, **kwargs: object) -> object:
        admissions.append(kwargs.get("admission"))
        return handles[Path(uri)]

    monkeypatch.setattr(
        "faninsar.processing.stack.nisar.NisarSensor.open_product", open_product
    )
    monkeypatch.setattr(
        "faninsar.processing.stack.nisar.NisarSensor.to_slc_product",
        lambda _sensor, handle, **_: _result(
            Path(handle.filename), Path(handle.filename).stem.rsplit("_", 1)[-1]
        ),
    )
    policy = {"trusted_roots": [tmp_path], "max_size_bytes": 1024}

    stack = NISARStack.from_rslc(
        paths,
        work_dir=tmp_path / "work",
        extra={"nisar_admission": policy},
    )

    assert admissions == [policy, policy]
    assert stack.config.extra["source_admission"] == {}


def test_nisar_stack_rejects_existing_paths_without_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing RSLC files cannot reach the reader without trusted admission."""
    paths = tuple(
        tmp_path / f"NISAR_RSLC_{date_id}.h5" for date_id in ("20240101", "20240113")
    )
    for path in paths:
        path.touch()
    opened: list[str] = []

    def open_product(_sensor: object, uri: str, **_kwargs: object) -> object:
        opened.append(uri)
        pytest.fail("NISAR reader opened an existing path without admission")

    monkeypatch.setattr(
        "faninsar.processing.stack.nisar.NisarSensor.open_product", open_product
    )

    with pytest.raises(InvalidProcessingStateError, match="explicit nisar_admission"):
        NISARStack.from_rslc(paths, work_dir=tmp_path / "work")

    assert opened == []


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


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("channel", "different from the requested"),
        ("lineage", "source lineage does not match"),
    ],
)
def test_nisar_stack_records_only_the_admitted_product_channel_and_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    """Reject reader results whose channel or source lineage is inconsistent."""
    paths = (
        tmp_path / "NISAR_RSLC_20240101.h5",
        tmp_path / "NISAR_RSLC_20240113.h5",
    )
    handles = {path: SimpleNamespace(filename=str(path)) for path in paths}

    monkeypatch.setattr(
        "faninsar.processing.stack.nisar.NisarSensor.open_product",
        lambda _sensor, uri: handles[Path(uri)],
    )

    def read_product(
        _sensor: object,
        handle: object,
        **_kwargs: object,
    ) -> SLCReadResult:
        path = Path(handle.filename)  # type: ignore[attr-defined]
        result = _result(path, path.stem.rsplit("_", 1)[-1])
        if field == "channel":
            return replace(
                result,
                mission_native={
                    **result.mission_native,
                    "frequency": "A",
                    "polarization": "VV",
                },
            )
        return replace(result, source_path=str(tmp_path / "other.h5"))

    monkeypatch.setattr(
        "faninsar.processing.stack.nisar.NisarSensor.to_slc_product",
        read_product,
    )

    with pytest.raises(ValueError, match=message):
        NISARStack.from_rslc(paths, work_dir=tmp_path / "work")


def test_nisar_stack_fails_closed_before_shared_s1_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsupported processing stages never dispatch RSLC paths to S1 code."""
    paths = tuple(
        tmp_path / f"NISAR_RSLC_{date_id}.h5" for date_id in ("20240101", "20240113")
    )
    for path in paths:
        path.touch()
    handles = {path: SimpleNamespace(filename=str(path)) for path in paths}
    monkeypatch.setattr(
        "faninsar.processing.stack.nisar.NisarSensor.open_product",
        lambda _sensor, uri, **_kwargs: handles[Path(uri)],
    )
    monkeypatch.setattr(
        "faninsar.processing.stack.nisar.NisarSensor.to_slc_product",
        lambda _sensor, handle, **_: _result(
            Path(handle.filename), Path(handle.filename).stem.rsplit("_", 1)[-1]
        ),
    )
    stack = NISARStack.from_rslc(
        paths,
        work_dir=tmp_path / "work",
        extra={"nisar_admission": {"trusted_roots": [tmp_path]}},
    )

    def fail_if_safe_opened(_path: object) -> object:
        pytest.fail("NISAR RSLC was routed to the SAFE opener")

    monkeypatch.setattr(
        "faninsar.processing.pipeline.production.open_safe_product",
        fail_if_safe_opened,
    )

    with pytest.raises(UnsupportedStackCapabilityError, match="NISAR RSLC Stack"):
        stack.measure_misreg()
    with pytest.raises(UnsupportedStackCapabilityError, match="NISAR RSLC Stack"):
        stack.coregister_scenes()
    with pytest.raises(InvalidProcessingStateError, match="scene generation missing"):
        stack.form_interferograms()


def test_nisar_scene_provider_exposes_named_fail_closed_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NISAR advertises unsupported scene production instead of S1 fallback."""
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

    assert stack.scene_provider is not None
    with pytest.raises(
        UnsupportedStackCapabilityError,
        match="scene-production",
    ):
        stack.scene_provider(
            paths[0],
            paths[1],
            output_dir=tmp_path / "pair",
            options={},
        )
