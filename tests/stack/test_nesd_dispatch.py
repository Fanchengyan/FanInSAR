"""Stack NESD measure / invert / apply dispatch (PROPOSAL-0056)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from faninsar.io.storage.scene_store import write_scene_unit
from faninsar.missions.s1 import S1Stack
from faninsar.processing.coregistration.misreg_network import DateMisreg, MisregArc
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.stack.provider import StackSceneProvider
from faninsar.stack.session import _load_arcs, _save_arcs


def _two_date_stack(tmp_path: Path, *, coreg_mode: str = "nesd") -> S1Stack:
    """Build a prepared two-date Stack with dummy SAFE directories."""
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


def _write_fake_scene(output_scenes: Path, *, date_id: str, reference_id: str) -> None:
    """Write one radar scene unit so coregistration persistence can resume."""
    data = np.ones((2, 2), dtype=np.complex64)
    write_scene_unit(
        output_scenes,
        date_id=date_id,
        reference_id=reference_id,
        domain="radar",
        tag="IW1_b0",
        primary=data,
        secondary=data,
        row_origin=0,
        col_origin=0,
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
    """Apply uses coregister_s1_scene per date and never the pair engine."""
    stack = _two_date_stack(tmp_path)
    produce_calls = _forbid_produce_pair(stack)
    stack.date_misreg = DateMisreg(
        reference="20160101",
        azimuth_px={"20160101": 0.0, "20160113": 0.05},
        range_px={"20160101": 0.0, "20160113": -0.02},
        metadata={},
    )
    captured: list[dict[str, object]] = []

    def fake_coregister(
        _reference_path: object,
        secondary_path: object,
        *,
        output_scenes: Path,
        misreg_az_px: float,
        misreg_rg_px: float,
        **kwargs: object,
    ) -> Path:
        date_id = Path(str(secondary_path)).name.split("_")[5][:8]
        captured.append(
            {
                "date_id": date_id,
                "misreg_az_px": misreg_az_px,
                "misreg_rg_px": misreg_rg_px,
                "roi": kwargs.get("roi"),
                "swaths": kwargs.get("swaths"),
                "bursts": kwargs.get("bursts"),
            }
        )
        _write_fake_scene(
            output_scenes,
            date_id=date_id,
            reference_id=stack.reference,
        )
        return Path(output_scenes)

    monkeypatch.setattr(
        "faninsar.missions.s1.coregister.coregister_s1_scene",
        fake_coregister,
    )

    stack.coregister_scenes()

    assert produce_calls == []
    assert captured == [
        {
            "date_id": "20160113",
            "misreg_az_px": pytest.approx(0.05),
            "misreg_rg_px": pytest.approx(-0.02),
            "roi": None,
            "swaths": None,
            "bursts": None,
        }
    ]


def test_arcs_json_round_trips_none(tmp_path: Path) -> None:
    """Omitted components persist as JSON null and reload as None, never 0."""
    path = tmp_path / "misreg" / "arcs.json"
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

    _save_arcs(path, arcs)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0]["azimuth_shift_px"] is None
    assert payload[1]["range_shift_px"] is None
    assert "null" in path.read_text(encoding="utf-8")

    loaded = _load_arcs(path)
    assert loaded[0].azimuth_shift_px is None
    assert loaded[0].range_shift_px == pytest.approx(0.2)
    assert loaded[1].azimuth_shift_px == pytest.approx(0.03)
    assert loaded[1].range_shift_px is None


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
    monkeypatch.setattr(
        "faninsar.missions.s1.coregister.coregister_s1_scene",
        lambda *_args, **_kwargs: Path("unused"),
    )

    with pytest.raises(InvalidProcessingStateError, match="date_misreg"):
        stack.coregister_scenes()


def test_coregister_allows_warned_no_window_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-window skip path may apply geometry zeros without invert."""
    stack = _two_date_stack(tmp_path)
    produce_calls = _forbid_produce_pair(stack)
    stack._nesd_skipped_no_windows = True
    captured: list[tuple[float, float]] = []

    def fake_coregister(
        _reference_path: object,
        secondary_path: object,
        *,
        output_scenes: Path,
        misreg_az_px: float,
        misreg_rg_px: float,
        **_kwargs: object,
    ) -> Path:
        date_id = Path(str(secondary_path)).name.split("_")[5][:8]
        captured.append((misreg_az_px, misreg_rg_px))
        _write_fake_scene(
            output_scenes,
            date_id=date_id,
            reference_id=stack.reference,
        )
        return Path(output_scenes)

    monkeypatch.setattr(
        "faninsar.missions.s1.coregister.coregister_s1_scene",
        fake_coregister,
    )

    stack.coregister_scenes()

    assert produce_calls == []
    assert captured == [(0.0, 0.0)]


def test_geometry_coregister_applies_zero_constants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Geometry apply never measures and injects 0, 0 date constants."""
    stack = _two_date_stack(tmp_path, coreg_mode="geometry")
    produce_calls = _forbid_produce_pair(stack)
    captured: list[tuple[float, float]] = []

    def fake_coregister(
        _reference_path: object,
        secondary_path: object,
        *,
        output_scenes: Path,
        misreg_az_px: float,
        misreg_rg_px: float,
        **_kwargs: object,
    ) -> Path:
        date_id = Path(str(secondary_path)).name.split("_")[5][:8]
        captured.append((misreg_az_px, misreg_rg_px))
        _write_fake_scene(
            output_scenes,
            date_id=date_id,
            reference_id=stack.reference,
        )
        return Path(output_scenes)

    monkeypatch.setattr(
        "faninsar.missions.s1.coregister.coregister_s1_scene",
        fake_coregister,
    )

    stack.coregister_scenes()

    assert produce_calls == []
    assert captured == [(0.0, 0.0)]
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


def test_measure_and_apply_share_resolved_burst_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ROI-resolved bursts are reused; measure is not an ROI crop."""
    stack = _two_date_stack(tmp_path)
    _forbid_produce_pair(stack)
    selection = (("IW1",), {"IW1": [0, 1]})
    monkeypatch.setattr(stack, "_resolve_s1_nesd_selection", lambda: selection)
    stack._s1_nesd_selection = selection
    measure_kw: dict[str, object] = {}
    apply_kw: dict[str, object] = {}

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

    def fake_coregister(
        *_args: object,
        output_scenes: Path,
        **kwargs: object,
    ) -> Path:
        apply_kw.update(kwargs)
        _write_fake_scene(
            output_scenes,
            date_id="20160113",
            reference_id=stack.reference,
        )
        return Path(output_scenes)

    monkeypatch.setattr(
        "faninsar.missions.s1.measure.measure_s1_pair_misreg",
        fake_measure,
    )
    monkeypatch.setattr(
        "faninsar.missions.s1.coregister.coregister_s1_scene",
        fake_coregister,
    )

    stack.measure_misreg(overwrite=True)
    stack.coregister_scenes()

    assert measure_kw["swaths"] == apply_kw["swaths"] == ("IW1",)
    assert measure_kw["bursts"] == apply_kw["bursts"] == {"IW1": [0, 1]}
    assert "roi" not in measure_kw
    assert "esd_enabled" not in apply_kw
    assert "force_esd_azimuth_shift_px" not in apply_kw


def test_coreg_marker_records_coreg_mode_not_esd_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Apply markers persist coreg_mode and date constants, never esd_enabled."""
    stack = _two_date_stack(tmp_path)
    _forbid_produce_pair(stack)
    stack.date_misreg = DateMisreg(
        reference="20160101",
        azimuth_px={"20160101": 0.0, "20160113": 0.05},
        range_px={"20160101": 0.0, "20160113": -0.02},
        metadata={},
    )

    def fake_coregister(
        _reference_path: object,
        secondary_path: object,
        *,
        output_scenes: Path,
        **_kwargs: object,
    ) -> Path:
        date_id = Path(str(secondary_path)).name.split("_")[5][:8]
        _write_fake_scene(
            output_scenes,
            date_id=date_id,
            reference_id=stack.reference,
        )
        return Path(output_scenes)

    monkeypatch.setattr(
        "faninsar.missions.s1.coregister.coregister_s1_scene",
        fake_coregister,
    )
    stack.coregister_scenes()
    marker = json.loads(
        (tmp_path / "out" / "coreg" / "20160113" / "coreg_done.json").read_text(
            encoding="utf-8"
        )
    )
    assert marker["coreg_mode"] == "nesd"
    assert marker["misreg_az_px"] == pytest.approx(0.05)
    assert marker["misreg_rg_px"] == pytest.approx(-0.02)
    assert "esd_enabled" not in marker
    assert "amplitude_refinement_enabled" not in marker
    assert "esd_azimuth_shift_px" not in marker
