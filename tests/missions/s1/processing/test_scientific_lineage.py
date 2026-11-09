"""S1 scientific lineage without payload hashing (PROPOSAL-0082)."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from faninsar.missions.s1.processing import (
    ProductionPairState,
    stage_coregister,
    stage_deramp,
    stage_interferogram,
)
from faninsar.processing.coregistration.offsets import OffsetFieldResult
from faninsar.processing.coregistration.tops.deramp import TOPSCarrierModel
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry import ConstantDEM
from faninsar.processing.geometry.prepared import ResidualSolution, ResidualStatus

_PAYLOAD_KEYS = (
    "input_payload_digest",
    "output_payload_digest",
    "offset_field_digest",
    "lut_digest",
    "topographic_phase_digest",
    "range_phase_screen_digest",
)
_LEGACY_DIGEST = "cd" * 32


def _make_carrier() -> TOPSCarrierModel:
    return TOPSCarrierModel(
        radar_frequency_hz=5.405e9,
        slant_range_time0_s=0.002,
        range_sampling_rate_hz=64.348e6,
        azimuth_time_interval_s=0.002,
        doppler_centroid_hz=(0.0, 0.0),
        doppler_t0_s=0.0,
        fm_rate_hz_s=(-2300.0, 0.0),
        fm_t0_s=0.0,
        burst_sensing_time_s=0.0,
        burst_start_slant_range_time_s=0.002,
        azimuth_steering_rate_hz_s=6500.0,
    )


def _make_mock_scene(shape: tuple[int, int]) -> MagicMock:
    scene = MagicMock()
    scene.array.samples = np.ones(shape, dtype=np.complex64)
    scene.array.valid_mask = np.ones(shape, dtype=bool)
    scene.array.row0 = 0
    scene.array.col0 = 0
    scene.carrier = _make_carrier()
    scene.scene_id = "test_scene"
    scene.swath.swath = "IW1"
    scene.burst.index = 0
    scene.geometry.range_spacing_m = 2.3
    scene.geometry.wavelength_m = 0.056
    return scene


def _pair_state(
    shape: tuple[int, int] = (16, 16),
    *,
    record_scientific_lineage: bool = False,
) -> ProductionPairState:
    return ProductionPairState(
        pair_id="LINEAGE",
        primary=_make_mock_scene(shape),
        secondary=_make_mock_scene(shape),
        dem=ConstantDEM(0.0),
        record_scientific_lineage=record_scientific_lineage,
    )


def _assert_no_payload_checksums(record: dict[str, Any]) -> None:
    for key in _PAYLOAD_KEYS:
        assert key not in record
    metadata = json.loads(record["metadata"])
    for key in _PAYLOAD_KEYS:
        assert key not in metadata
        assert metadata.get(key) not in ("", "0" * 64)
    serialized = json.dumps(record)
    assert "payload_digest" not in serialized
    assert "content-bound" not in serialized.lower()


def test_opt_in_lineage_does_not_change_deramp_or_interferogram_arrays() -> None:
    """Numerical products stay identical when lightweight lineage is recorded."""
    deramp_off = stage_deramp(_pair_state(record_scientific_lineage=False))
    deramp_on = stage_deramp(_pair_state(record_scientific_lineage=True))
    np.testing.assert_array_equal(deramp_off.primary_deramped, deramp_on.primary_deramped)
    np.testing.assert_array_equal(
        deramp_off.secondary_deramped, deramp_on.secondary_deramped
    )

    def _ifg_state(*, lineage: bool) -> ProductionPairState:
        state = _pair_state(record_scientific_lineage=lineage)
        state.primary_deramped = np.full((16, 16), 1.0 + 0.25j, dtype=np.complex64)
        state.secondary_aligned = np.full((16, 16), 0.5 - 0.5j, dtype=np.complex64)
        return state

    ifg_off = stage_interferogram(
        _ifg_state(lineage=False), multilook=(2, 4), goldstein_alpha=0.0
    )
    ifg_on = stage_interferogram(
        _ifg_state(lineage=True), multilook=(2, 4), goldstein_alpha=0.0
    )
    np.testing.assert_array_equal(ifg_off.complex, ifg_on.complex)
    np.testing.assert_array_equal(ifg_off.coherence, ifg_on.coherence)
    np.testing.assert_array_equal(ifg_off.wrapped_phase, ifg_on.wrapped_phase)


def test_opt_in_lineage_serializes_operations_without_payload_checksums() -> None:
    """Enabled lineage records operations and parameters, not array hashes."""
    state = stage_deramp(_pair_state(record_scientific_lineage=True))
    assert len(state.scientific_lineage) == 1
    record = state.scientific_lineage[0]
    assert record["operation"] == "deramp"
    assert record["operation_id"]
    metadata = json.loads(record["metadata"])
    assert metadata["carrier"] == "tops"
    _assert_no_payload_checksums(dict(record))


def test_opt_in_lineage_does_not_inspect_full_array_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in recording must not SHA-scan numerical payloads."""
    import faninsar.missions.s1.processing as production

    large_updates: list[int] = []
    real_sha256 = hashlib.sha256

    class _SpyHasher:
        def __init__(self, hasher: object) -> None:
            self._hasher = hasher

        def update(self, data: object) -> None:
            size = len(data) if data is not None else 0
            if size > 1024:
                large_updates.append(size)
            self._hasher.update(data)

        def hexdigest(self) -> str:
            return self._hasher.hexdigest()

        def digest(self) -> bytes:
            return self._hasher.digest()

        def copy(self) -> _SpyHasher:
            return _SpyHasher(self._hasher.copy())

    def spy_sha256(*args: object, **kwargs: object) -> _SpyHasher:
        if args:
            data = args[0]
            size = len(data) if data is not None else 0
            if size > 1024:
                large_updates.append(size)
        return _SpyHasher(real_sha256(*args, **kwargs))

    monkeypatch.setattr(production.hashlib, "sha256", spy_sha256)
    if hasattr(production, "_lineage_payload_digest"):
        monkeypatch.setattr(
            production,
            "_lineage_payload_digest",
            lambda *_args, **_kwargs: pytest.fail(
                "scientific lineage must not hash array payloads"
            ),
        )

    deramped = stage_deramp(_pair_state((32, 48), record_scientific_lineage=True))
    deramped.primary_deramped = np.ones((32, 48), dtype=np.complex64)
    deramped.secondary_aligned = np.ones((32, 48), dtype=np.complex64)
    stage_interferogram(deramped, multilook=(2, 4), goldstein_alpha=0.0)

    assert large_updates == []
    assert deramped.scientific_lineage
    _assert_no_payload_checksums(dict(deramped.scientific_lineage[-1]))


def test_coregister_lineage_applies_solution_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coregistration records a lightweight residual application exactly once."""
    import faninsar.missions.s1.processing as production

    shape = (16, 32)
    state = _pair_state(shape, record_scientific_lineage=True)
    state.primary_deramped = np.ones(shape, dtype=np.complex64)
    state.secondary_deramped = np.ones(shape, dtype=np.complex64)
    offsets = OffsetFieldResult(
        range_offset_px=np.zeros(shape, dtype=np.float32),
        azimuth_offset_px=np.full(shape, 0.25, dtype=np.float32),
        coverage=np.ones(shape, dtype=bool),
        uncertainty_px=np.zeros(shape, dtype=np.float32),
    )
    monkeypatch.setattr(production, "dense_geometry_offsets", lambda **_: offsets)
    monkeypatch.setattr(
        production,
        "refine_shift_with_correlation",
        lambda *_args, **_kwargs: pytest.fail("empirical shift refinement was called"),
    )
    monkeypatch.setattr(
        production,
        "combine_offset_fields",
        lambda geometry_field, **_kwargs: geometry_field,
    )
    monkeypatch.setattr(
        "faninsar.processing.coregistration.offsets._resample_complex_deramped_reramp_with_validity",
        lambda samples, source_valid, **_kwargs: (
            np.asarray(samples).copy(),
            np.asarray(source_valid, dtype=bool),
        ),
    )

    result = stage_coregister(
        state,
        esd_enabled=False,
        amplitude_refinement_enabled=False,
        executor="torch",
        device="cpu",
    )
    operations = [item["operation"] for item in result.scientific_lineage]
    assert "apply_carrier_residual_range_phase" in operations
    record = next(
        item
        for item in result.scientific_lineage
        if item["operation"] == "apply_carrier_residual_range_phase"
    )
    _assert_no_payload_checksums(dict(record))
    metadata = json.loads(record["metadata"])
    assert metadata["carrier_model"] == "tops"
    assert metadata["geometric_phase"] == "range_offset_screen"
    assert result.phase_state is not None
    assert result.phase_state.residual_application_id is not None
    history = result.phase_state.transition_history
    assert history
    assert history[-1].input_payload_digest is None
    assert history[-1].output_payload_digest is None

    duplicate = ResidualSolution(
        solution_id=result.phase_state.residual_solution_id or "sol",
        kind="pair",
        parent_observation_ids=("obs",),
        payload_digest=None,
        application_id=result.phase_state.residual_application_id,
        status=ResidualStatus.VALID,
    )
    with pytest.raises(InvalidProcessingStateError, match="already applied"):
        result.phase_state.apply_solution(duplicate, operation_id="duplicate")


def test_legacy_scientific_lineage_records_remain_inert() -> None:
    """Old payload-bound lineage dicts load without being rewritten or rehashed."""
    state = _pair_state(record_scientific_lineage=True)
    state.scientific_lineage.append(
        {
            "operation": "deramp",
            "operation_id": hashlib.sha256(b"legacy-op").hexdigest(),
            "input_state_digest": "0" * 64,
            "output_state_digest": hashlib.sha256(b"legacy-state").hexdigest(),
            "input_payload_digest": _LEGACY_DIGEST,
            "output_payload_digest": _LEGACY_DIGEST,
            "metadata": json.dumps({"carrier": "tops"}, sort_keys=True),
        }
    )
    state.primary_deramped = np.ones((16, 16), dtype=np.complex64)
    state.secondary_aligned = np.ones((16, 16), dtype=np.complex64)
    result = stage_interferogram(state, multilook=(2, 4), goldstein_alpha=0.0)
    legacy = result.scientific_lineage[0]
    assert legacy["input_payload_digest"] == _LEGACY_DIGEST
    assert legacy["output_payload_digest"] == _LEGACY_DIGEST
    new_record = result.scientific_lineage[-1]
    assert new_record["operation"] == "form_interferogram"
    _assert_no_payload_checksums(dict(new_record))
