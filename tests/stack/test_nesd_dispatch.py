"""Stack NESD measure / invert / apply dispatch (PROPOSAL-0056)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from faninsar.missions.s1 import S1Stack
from faninsar.processing.coregistration.misreg_network import DateMisreg, MisregArc
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.stack.misreg import read_misreg, write_misreg
from faninsar.stack.provider import StackSceneProvider
from tests.support.scene_stores import write_test_scene


def _two_date_stack(tmp_path: Path, *, coreg_mode: str = "nesd") -> S1Stack:
    """Build a prepared two-date Stack with dummy SAFE directories.

    NESD-path tests request ``coreg_mode="nesd"`` so they stay on the
    measure/invert/apply path after the stack default became geometry.
    """
    paths: list[Path] = []
    for day in ("20160101", "20160113"):
        path = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000_{day}T000001.SAFE"
        path.mkdir()
        paths.append(path)
    stack = S1Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        activation_mode="reference",
        coreg_mode=coreg_mode,  # type: ignore[arg-type]
        pair_max_interval=2,
        pair_max_days=60,
    )
    stack = stack.prepare_scenes()
    # Dummy SAFE directories have no annotation products.
    stack._ensure_precise_orbits = lambda: None  # type: ignore[method-assign]
    return stack


def _install_fake_apply(
    stack: S1Stack,
    monkeypatch: pytest.MonkeyPatch,
    captured: list[dict[str, object]] | None = None,
) -> None:
    """Replace Reference preparation and secondary apply with scene writers."""
    calls = [] if captured is None else captured

    def fake_prepare(_reference_path: object, **kwargs: object) -> dict[str, object]:
        if (Path(kwargs["output_scene"]) / "item.json").is_file():  # type: ignore[arg-type]
            return {"prepare": kwargs}
        write_test_scene(
            stack.config.work_dir,
            date_id=stack.reference,
            reference_id=stack.reference,
            payload=np.ones((2, 2), dtype=np.complex64),
            tag="IW1_b0",
        )
        return {"prepare": kwargs}

    def fake_secondary(
        context: dict[str, object],
        secondary_path: object,
        *,
        output_scene: Path,
        misreg_az_px: float,
        misreg_rg_px: float,
        **kwargs: object,
    ) -> None:
        date_id = Path(str(secondary_path)).name.split("_")[5][:8]
        prepare = context["prepare"]
        calls.append(
            {
                "date_id": date_id,
                "misreg_az_px": misreg_az_px,
                "misreg_rg_px": misreg_rg_px,
                "roi": prepare.get("roi"),  # type: ignore[union-attr]
                "swaths": prepare.get("swaths"),  # type: ignore[union-attr]
                "bursts": prepare.get("bursts"),  # type: ignore[union-attr]
                "properties": kwargs.get("properties"),
            }
        )
        assert Path(output_scene) == stack.config.work_dir / "coreg" / date_id
        write_test_scene(
            stack.config.work_dir,
            date_id=date_id,
            reference_id=stack.reference,
            payload=np.ones((2, 2), dtype=np.complex64),
            tag="IW1_b0",
            properties=kwargs.get("properties"),
            inputs=kwargs.get("inputs"),
        )

    monkeypatch.setattr(
        "faninsar.missions.s1.coregister.prepare_s1_reference", fake_prepare
    )
    monkeypatch.setattr(
        "faninsar.missions.s1.coregister.coregister_s1_secondary", fake_secondary
    )


def _forbid_produce_pair(stack: S1Stack) -> list[str]:
    """Replace the pair engine with a callback that records and fails."""
    calls: list[str] = []

    def produce_pair(*_args: object, **_kwargs: object) -> None:
        calls.append("produce_pair")
        message = "NESD measure/apply must not call _produce_pair"
        raise AssertionError(message)

    stack.scene_provider = StackSceneProvider(name="TEST", produce_pair=produce_pair)
    return calls


def test_geometry_measure_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Geometry mode skips measure and never touches the pair engine or NESD stage."""
    stack = _two_date_stack(tmp_path, coreg_mode="geometry")
    produce_calls = _forbid_produce_pair(stack)
    measure_calls: list[str] = []

    def fake_measure(*_args: object, **_kwargs: object) -> MisregArc:
        measure_calls.append("measure")
        message = "Geometry mode must not measure NESD"
        raise AssertionError(message)

    monkeypatch.setattr(
        "faninsar.missions.s1.measure.measure_s1_pair_misreg",
        fake_measure,
    )

    stack.measure_misreg()

    assert stack.arcs == []
    assert stack.date_misreg is None
    assert measure_calls == []
    assert produce_calls == []


def test_nesd_invert_uses_optional_invert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nesd invert projects mixed float|None arcs through the optional adapter."""
    stack = _two_date_stack(tmp_path)
    _forbid_produce_pair(stack)
    invert_calls: list[object] = []

    def fake_measure(*_args: object, **_kwargs: object) -> MisregArc:
        return MisregArc(
            primary="20160101",
            secondary="20160113",
            azimuth_shift_px=None,
            range_shift_px=0.2,
            azimuth_sigma_px=0.01,
            range_sigma_px=0.01,
            method="overlap",
            n_valid=12,
        )

    def fake_optional(arcs: object, **kwargs: object) -> DateMisreg:
        invert_calls.append((arcs, kwargs))
        return DateMisreg(
            reference="20160101",
            azimuth_px={"20160101": 0.0, "20160113": 0.0},
            range_px={"20160101": 0.0, "20160113": 0.2},
            metadata={"adapter": "optional"},
        )

    def fake_joint(*_args: object, **_kwargs: object) -> DateMisreg:
        message = "Nesd invert must not call invert_pair_misregistration"
        raise AssertionError(message)

    monkeypatch.setattr(
        "faninsar.missions.s1.measure.measure_s1_pair_misreg",
        fake_measure,
    )
    monkeypatch.setattr(
        "faninsar.stack.session.invert_optional_pair_misregistration",
        fake_optional,
    )
    monkeypatch.setattr(
        "faninsar.stack.session.invert_pair_misregistration",
        fake_joint,
        raising=False,
    )

    stack.measure_misreg(overwrite=True)
    stack.invert_misreg()

    assert invert_calls, "invert_optional_pair_misregistration was not used"
    arcs, kwargs = invert_calls[0]
    assert kwargs["reference"] == "20160101"
    arc = next(iter(arcs))
    assert arc.azimuth_shift_px is None
    assert arc.range_shift_px == pytest.approx(0.2)
    assert stack.date_misreg is not None
    assert stack.date_misreg.azimuth_px["20160113"] == 0.0
    assert stack.date_misreg.range_px["20160113"] == pytest.approx(0.2)


def test_coregister_does_not_call_produce_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Apply uses the Reference-once S1 entry points, never the pair engine."""
    stack = _two_date_stack(tmp_path)
    produce_calls = _forbid_produce_pair(stack)
    stack.date_misreg = DateMisreg(
        reference="20160101",
        azimuth_px={"20160101": 0.0, "20160113": 0.05},
        range_px={"20160101": 0.0, "20160113": -0.02},
        metadata={},
    )
    captured: list[dict[str, object]] = []
    _install_fake_apply(stack, monkeypatch, captured)

    stack.coregister_scenes()

    assert produce_calls == []
    assert len(captured) == 1
    call = captured[0]
    assert call["date_id"] == "20160113"
    assert call["misreg_az_px"] == pytest.approx(0.05)
    assert call["misreg_rg_px"] == pytest.approx(-0.02)
    assert (call["roi"], call["swaths"], call["bursts"]) == (None, None, None)


def test_misreg_product_round_trips_none(tmp_path: Path) -> None:
    """Omitted components persist as JSON null and reload as None, never 0."""
    root = tmp_path / "misreg"
    arcs = [
        MisregArc(
            primary="20160101",
            secondary="20160113",
            azimuth_shift_px=None,
            range_shift_px=0.2,
            azimuth_sigma_px=0.01,
            range_sigma_px=0.05,
            method="overlap",
            n_valid=8,
        ),
        MisregArc(
            primary="20160113",
            secondary="20160125",
            azimuth_shift_px=0.03,
            range_shift_px=None,
            method="overlap",
            n_valid=3,
        ),
    ]
    constants = DateMisreg(
        reference="20160101",
        azimuth_px={"20160101": 0.0, "20160113": 0.03},
        range_px={"20160101": 0.0, "20160113": 0.2},
        metadata={"n_arcs": 2},
    )

    product = write_misreg(
        root, arcs=arcs, date_misreg=constants, skipped_no_windows=False
    )
    payload = json.loads((root / "item.json").read_text(encoding="utf-8"))
    stored = payload["properties"]["faninsar:arcs"]
    assert stored[0]["azimuth_shift_px"] is None
    assert stored[1]["range_shift_px"] is None

    record = read_misreg(root)
    assert record is not None
    assert record.write_id == product.write_id
    assert record.arcs[0].azimuth_shift_px is None
    assert record.arcs[0].range_shift_px == pytest.approx(0.2)
    assert record.arcs[1].azimuth_shift_px == pytest.approx(0.03)
    assert record.arcs[1].range_shift_px is None
    assert record.date_misreg == constants
    assert record.skipped_no_windows is False
    assert read_misreg(tmp_path / "absent") is None


def test_nesd_no_windows_warns_skips_and_zeros_dates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A selection with no overlap windows warns, skips NESD, and records zeros."""
    stack = _two_date_stack(tmp_path)
    _forbid_produce_pair(stack)

    def fake_measure(*_args: object, **_kwargs: object) -> MisregArc:
        return MisregArc(
            primary="20160101",
            secondary="20160113",
            azimuth_shift_px=None,
            range_shift_px=None,
            method="overlap",
            n_valid=0,
        )

    monkeypatch.setattr(
        "faninsar.missions.s1.measure.measure_s1_pair_misreg",
        fake_measure,
    )

    with caplog.at_level("WARNING"):
        stack.measure_misreg(overwrite=True)
    assert any("overlap" in rec.message.lower() for rec in caplog.records)
    assert stack._nesd_skipped_no_windows is True

    stack.invert_misreg()
    assert stack.date_misreg is not None
    assert stack.date_misreg.azimuth_px == {"20160101": 0.0, "20160113": 0.0}
    assert stack.date_misreg.range_px == {"20160101": 0.0, "20160113": 0.0}


def test_coregister_nesd_without_date_misreg_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nesd apply fails closed when invert has not recorded date constants."""
    stack = _two_date_stack(tmp_path)
    _forbid_produce_pair(stack)
    _install_fake_apply(stack, monkeypatch)

    with pytest.raises(InvalidProcessingStateError, match="date_misreg"):
        stack.coregister_scenes()


def test_coregister_allows_warned_no_window_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-window skip path may apply geometry zeros without invert."""
    stack = _two_date_stack(tmp_path)
    produce_calls = _forbid_produce_pair(stack)
    stack._nesd_skipped_no_windows = True
    calls: list[dict[str, object]] = []
    _install_fake_apply(stack, monkeypatch, calls)

    stack.coregister_scenes()

    assert produce_calls == []
    assert [(c["misreg_az_px"], c["misreg_rg_px"]) for c in calls] == [(0.0, 0.0)]


def test_geometry_coregister_applies_zero_constants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Geometry apply never measures and injects 0, 0 date constants."""
    stack = _two_date_stack(tmp_path, coreg_mode="geometry")
    produce_calls = _forbid_produce_pair(stack)
    calls: list[dict[str, object]] = []
    _install_fake_apply(stack, monkeypatch, calls)

    stack.coregister_scenes()

    assert produce_calls == []
    assert [(c["misreg_az_px"], c["misreg_rg_px"]) for c in calls] == [(0.0, 0.0)]
    assert stack.date_misreg is None


def test_measure_passes_thresholds_orbits_and_resolved_bursts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measure forwards Stack thresholds, extra orbits, and resolved bursts."""
    stack = _two_date_stack(tmp_path)
    _forbid_produce_pair(stack)
    stack.config.esd_coherence_threshold = 0.9
    stack.config.ampcor_snr_threshold = 7.5
    stack.config.device = "cpu"
    stack.config.extra["orbit_paths"] = {
        "20160101": Path("/orbits/ref.EOF"),
        "20160113": Path("/orbits/sec.EOF"),
    }
    monkeypatch.setattr(
        stack,
        "_resolve_s1_nesd_selection",
        lambda: (("IW1", "IW2"), {"IW1": [0, 1], "IW2": [1, 2]}),
    )
    captured: dict[str, object] = {}

    def fake_measure(*_args: object, **kwargs: object) -> MisregArc:
        captured.update(kwargs)
        return MisregArc(
            primary="20160101",
            secondary="20160113",
            azimuth_shift_px=0.01,
            range_shift_px=None,
            method="overlap",
            n_valid=4,
        )

    monkeypatch.setattr(
        "faninsar.missions.s1.measure.measure_s1_pair_misreg",
        fake_measure,
    )

    stack.measure_misreg(overwrite=True)

    assert captured["coherence_threshold"] == pytest.approx(0.9)
    assert captured["ampcor_snr_threshold"] == pytest.approx(7.5)
    assert captured["primary_orbit_path"] == Path("/orbits/ref.EOF")
    assert captured["secondary_orbit_path"] == Path("/orbits/sec.EOF")
    assert captured["device"] == "cpu"
    assert captured["swaths"] == ("IW1", "IW2")
    assert captured["bursts"] == {"IW1": [0, 1], "IW2": [1, 2]}
    assert "esd_enabled" not in captured
    assert stack.arcs[0].azimuth_shift_px == pytest.approx(0.01)
    assert stack.arcs[0].range_shift_px is None


def test_measure_lists_parent_slots_apply_keeps_roi_bursts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NESD measure lists parent slots; apply still uses the ROI burst set."""
    stack = _two_date_stack(tmp_path)
    _forbid_produce_pair(stack)
    apply_selection = (("IW1",), {"IW1": [1]})
    measure_selection = (("IW1",), {"IW1": [0, 1, 2]})
    monkeypatch.setattr(
        stack, "_resolve_s1_roi_burst_selection", lambda: apply_selection
    )
    monkeypatch.setattr(stack, "_resolve_s1_nesd_selection", lambda: measure_selection)
    stack._s1_roi_burst_selection = apply_selection
    stack._s1_nesd_selection = measure_selection
    measure_kw: dict[str, object] = {}

    def fake_measure(*_args: object, **kwargs: object) -> MisregArc:
        measure_kw.update(kwargs)
        return MisregArc(
            primary="20160101",
            secondary="20160113",
            azimuth_shift_px=None,
            range_shift_px=None,
            method="overlap",
            n_valid=0,
        )

    monkeypatch.setattr(
        "faninsar.missions.s1.measure.measure_s1_pair_misreg",
        fake_measure,
    )
    calls: list[dict[str, object]] = []
    _install_fake_apply(stack, monkeypatch, calls)

    stack.measure_misreg(overwrite=True)
    stack.coregister_scenes()

    apply_kw = calls[0]
    assert measure_kw["swaths"] == apply_kw["swaths"] == ("IW1",)
    assert measure_kw["bursts"] == {"IW1": [0, 1, 2]}
    assert apply_kw["bursts"] == {"IW1": [1]}
    assert "roi" not in measure_kw


def test_scene_item_records_coreg_mode_and_constants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The secondary Item records coreg_mode and identity, never ESD flags."""
    stack = _two_date_stack(tmp_path)
    _forbid_produce_pair(stack)
    stack.date_misreg = DateMisreg(
        reference="20160101",
        azimuth_px={"20160101": 0.0, "20160113": 0.05},
        range_px={"20160101": 0.0, "20160113": -0.02},
        metadata={},
    )
    _install_fake_apply(stack, monkeypatch)
    stack.coregister_scenes()
    scene = stack.scenes["20160113"]
    assert not (scene.root / "coreg_done.json").exists()
    assert scene.properties["faninsar:coreg_mode"] == "nesd"
    assert isinstance(scene.properties["faninsar:coreg_identity"], str)
    assert scene.inputs["reference"] == stack.reference_scene.write_id
    for key in ("esd_enabled", "amplitude_refinement_enabled"):
        assert f"faninsar:{key}" not in scene.properties


def test_coregister_reuses_current_scene_and_redoes_stale_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A current secondary is reused; a changed constant recoregisters it."""
    stack = _two_date_stack(tmp_path, coreg_mode="geometry")
    _forbid_produce_pair(stack)
    calls: list[dict[str, object]] = []
    _install_fake_apply(stack, monkeypatch, calls)
    stack.coregister_scenes()
    first = stack.scenes["20160113"].write_id
    stack.coregister_scenes()
    assert len(calls) == 1
    assert stack.scenes["20160113"].write_id == first
    monkeypatch.setattr(stack, "_apply_misreg_constants", lambda _d: (0.1, 0.0))
    stack.coregister_scenes()
    assert len(calls) == 2
    assert stack.scenes["20160113"].write_id != first


def test_rewritten_reference_invalidates_secondary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A secondary bound to an overwritten Reference is recoregistered."""
    stack = _two_date_stack(tmp_path, coreg_mode="geometry")
    _forbid_produce_pair(stack)
    calls: list[dict[str, object]] = []
    _install_fake_apply(stack, monkeypatch, calls)
    stack.coregister_scenes()
    write_test_scene(
        stack.config.work_dir,
        date_id=stack.reference,
        reference_id=stack.reference,
        payload=np.full((2, 2), 3.0, dtype=np.complex64),
        tag="IW1_b0",
    )
    stack.coregister_scenes()
    assert len(calls) == 2
    assert (
        stack.scenes["20160113"].inputs["reference"] == stack.reference_scene.write_id
    )


def test_reopened_stack_restores_nesd_skip_without_remeasuring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A skipped NESD campaign is restored from disk, never re-measured."""
    stack = _two_date_stack(tmp_path)
    _forbid_produce_pair(stack)
    calls: list[str] = []

    def fake_measure(*_args: object, **_kwargs: object) -> MisregArc:
        calls.append("measure")
        return MisregArc(
            primary="20160101",
            secondary="20160113",
            azimuth_shift_px=None,
            range_shift_px=None,
            method="overlap",
            n_valid=0,
        )

    monkeypatch.setattr(
        "faninsar.missions.s1.measure.measure_s1_pair_misreg", fake_measure
    )
    stack.measure_misreg()
    stack.invert_misreg()
    assert calls == ["measure"]

    again = tmp_path / "again"
    again.mkdir()
    reopened = _two_date_stack(again)
    reopened.config.work_dir = stack.config.work_dir
    reopened.measure_misreg()
    assert calls == ["measure"]
    assert reopened._nesd_skipped_no_windows is True
    assert reopened.date_misreg == stack.date_misreg
    assert reopened._misreg_write_id == stack._misreg_write_id


def test_nesd_secondary_binds_misreg_and_goes_stale_on_reinversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changed misreg constants make NESD secondaries stale; identical do not.

    Re-committing identical state (every process restart) keeps the write
    id so scenes and interferograms resume; only materially new constants
    replace the product and recoregister the secondaries.
    """
    stack = _two_date_stack(tmp_path)
    _forbid_produce_pair(stack)
    stack.date_misreg = DateMisreg(
        reference="20160101",
        azimuth_px={"20160101": 0.0, "20160113": 0.0},
        range_px={"20160101": 0.0, "20160113": 0.0},
        metadata={},
    )
    stack._persist_misreg()
    calls: list[dict[str, object]] = []
    _install_fake_apply(stack, monkeypatch, calls)
    stack.coregister_scenes()
    first_misreg = stack._misreg_write_id
    assert stack.scenes["20160113"].inputs["misreg"] == first_misreg
    stack.coregister_scenes()
    assert len(calls) == 1
    stack._persist_misreg()
    assert stack._misreg_write_id == first_misreg
    stack.coregister_scenes()
    assert len(calls) == 1
    stack.date_misreg = DateMisreg(
        reference="20160101",
        azimuth_px={"20160101": 0.0, "20160113": 0.0},
        range_px={"20160101": 0.0, "20160113": 0.1},
        metadata={},
    )
    stack._persist_misreg()
    assert stack._misreg_write_id != first_misreg
    stack.coregister_scenes()
    assert len(calls) == 2


def test_write_misreg_keeps_write_id_for_identical_state(tmp_path: Path) -> None:
    """Re-committing identical misreg state keeps the existing write id.

    ``invert_misreg`` re-persists restored arcs and constants in every
    process; a fresh random write id would mark every secondary scene
    stale and cascade into stale interferograms, breaking cross-process
    resume of coregister/interferogram stages.
    """
    root = tmp_path / "misreg"
    arcs = [
        MisregArc(
            primary="20160101",
            secondary="20160113",
            azimuth_shift_px=0.01,
            range_shift_px=0.2,
            azimuth_sigma_px=0.05,
            range_sigma_px=0.1,
            method="overlap",
            n_valid=8,
        )
    ]
    constants = DateMisreg(
        reference="20160101",
        azimuth_px={"20160101": 0.0, "20160113": 0.01},
        range_px={"20160101": 0.0, "20160113": 0.2},
        metadata={"n_arcs": 1},
    )

    first = write_misreg(
        root, arcs=arcs, date_misreg=constants, skipped_no_windows=False
    )
    restored = read_misreg(root)
    assert restored is not None
    second = write_misreg(
        root,
        arcs=restored.arcs,
        date_misreg=restored.date_misreg,
        skipped_no_windows=restored.skipped_no_windows,
    )
    assert second.write_id == first.write_id

    changed = DateMisreg(
        reference="20160101",
        azimuth_px={"20160101": 0.0, "20160113": 0.05},
        range_px={"20160101": 0.0, "20160113": 0.2},
        metadata={"n_arcs": 1},
    )
    third = write_misreg(
        root, arcs=restored.arcs, date_misreg=changed, skipped_no_windows=False
    )
    assert third.write_id != first.write_id
