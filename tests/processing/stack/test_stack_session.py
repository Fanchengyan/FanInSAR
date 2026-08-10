"""Tests for Stack session scaffolding (PROPOSAL-0017)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.processing.contracts import ActivationToken, StackActivationBinding
from faninsar.processing.merge.grid import GeoGridSpec
from faninsar.processing.stack import Stack, StackConfig
from faninsar.processing.stack.catalog import SceneCatalog
from faninsar.processing.stack.scene_store import write_scene_unit

if TYPE_CHECKING:
    from pathlib import Path


def test_scene_catalog_from_paths(tmp_path: Path) -> None:
    """Catalog keys dates from SAFE-like names."""
    a = tmp_path / "S1A_IW_SLC__1SDV_20160101T000000_20160101T000001.SAFE"
    b = tmp_path / "S1A_IW_SLC__1SDV_20160113T000000_20160113T000001.SAFE"
    a.mkdir()
    b.mkdir()
    cat = SceneCatalog.from_paths([a, b])
    assert cat.dates == ("20160101", "20160113")
    assert cat.path_for("20160101") == a


def test_stack_from_safes_defaults(tmp_path: Path) -> None:
    """Stack builds default short-baseline pairs and earliest master."""
    paths = []
    for day in ("20160101", "20160113", "20160125"):
        p = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000_{day}T000001.SAFE"
        p.mkdir()
        paths.append(p)
    stack = Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        coreg_mode="geometry",
        pair_max_interval=2,
        pair_max_days=60,
    )
    assert stack.master == "20160101"
    assert len(stack.catalog) == 3
    stack.prepare_scenes()
    assert (tmp_path / "out" / "coreg").is_dir()
    # geometry mode skips measure/invert
    stack.measure_misreg()
    stack.invert_misreg()
    assert stack.date_misreg is None


def test_stack_config_multilook_normalize(tmp_path: Path) -> None:
    """StackConfig coerces multilook to int pair."""
    cfg = StackConfig(work_dir=tmp_path, multilook=(2, 10))
    assert cfg.multilook == (2, 10)


def test_stack_geo_grid_is_forwarded_to_pair_pipeline(tmp_path: Path) -> None:
    """Geo Stack configuration carries its resolved grid into Pair calls."""
    paths = []
    for day in ("20160101", "20160113"):
        path = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000_{day}T000001.SAFE"
        path.mkdir()
        paths.append(path)
    grid = GeoGridSpec(
        crs="EPSG:32647",
        transform=(0.0, 40.0, 0.0, 1000.0, 0.0, -40.0),
        width=20,
        height=20,
        resolution_m=(40.0, 40.0),
    )
    stack = Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        coregistration_grid="geo",
        geo_grid=grid,
    )
    assert stack._burst_kwargs()["geo_grid"] is grid


def test_stack_config_rejects_degrade_to_pair(tmp_path: Path) -> None:
    """Qualified Stack configuration cannot silently change coreg semantics."""
    with pytest.raises(ValueError, match="fail-closed"):
        StackConfig(work_dir=tmp_path, on_network_failure="degrade_to_pair")  # type: ignore[arg-type]


def test_stack_config_rejects_qualified_mode_without_binding(tmp_path: Path) -> None:
    """Qualified production mode cannot bypass its activation record."""
    with pytest.raises(ValueError, match="activation binding"):
        StackConfig(work_dir=tmp_path, activation_mode="qualified")


def test_stack_measure_misreg_keeps_ampcor_range_residual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Network arcs retain the production state's Ampcor range residual."""
    paths = []
    for day in ("20160101", "20160113"):
        path = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000_{day}T000001.SAFE"
        path.mkdir()
        paths.append(path)
    stack = Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        coreg_mode="network",
        pair_max_interval=2,
        pair_max_days=60,
    )
    stack.prepare_scenes()

    monkeypatch.setattr(
        "faninsar.processing.pipeline.production.run_pair",
        lambda *_args, **_kwargs: SimpleNamespace(
            esd_azimuth_shift_px=-0.3,
            amplitude_residual_rg_px=0.7,
        ),
    )
    stack.measure_misreg(overwrite=True)

    assert len(stack.arcs) == 1
    assert stack.arcs[0].azimuth_shift_px == pytest.approx(-0.3)
    assert stack.arcs[0].range_shift_px == pytest.approx(0.7)


def test_stack_forms_all_persisted_burst_units(tmp_path: Path) -> None:
    """Stack formation consumes every unit in matching scene generations."""
    dates = ("20160101", "20160113")
    safe_paths = []
    for date_id in dates:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000_{date_id}T000001.SAFE"
        path.mkdir()
        safe_paths.append(path)
    stack = Stack.from_safes(safe_paths, work_dir=tmp_path / "out")
    stack.prepare_scenes()

    reference = np.ones((2, 2), dtype=np.complex64)
    secondary = np.full((2, 2), 1.0 + 2.0j, dtype=np.complex64)
    for date_id in dates:
        root = stack.config.work_dir / "coreg" / date_id / "scenes"
        for tag in ("IW1_b0", "IW1_b1"):
            write_scene_unit(
                root,
                date_id=date_id,
                master_id=dates[0],
                domain="radar",
                tag=tag,
                reference=reference,
                secondary=secondary,
                row_origin=0,
                col_origin=0,
            )
        stack.coreg_paths[date_id] = root.parent

    stack.form_interferograms(multilook=(1, 1))
    output_root = stack.ifg_dirs[0]
    assert {path.name for path in output_root.glob("*.complex64")} == {
        "IW1_b0.complex64",
        "IW1_b1.complex64",
    }


def test_qualified_stack_form_requires_matching_activation_record(
    tmp_path: Path,
) -> None:
    """Qualified IFG formation consumes only its matching scene artifact record."""
    dates = ("20160101", "20160113")
    safe_paths = []
    for date_id in dates:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000_{date_id}T000001.SAFE"
        path.mkdir()
        safe_paths.append(path)
    token = ActivationToken(
        intent_id="intent",
        parent_id="stack-generation",
        parent_manifest_digest="c" * 64,
        root_device=1,
        root_inode=2,
        namespace="scene_artifact_v1",
        mode="qualified",
        domain="radar",
        policy_identity="policy-v1",
        code_identity="code-v1",
        schema="scene_artifact_v1",
        provider_receipt_digest="a" * 64,
        threshold_configuration_hash="d" * 64,
        qualification_evidence_digest="e" * 64,
        p19_qualified_event_ids=("p19-qualified",),
        p18_stack_gate_event_id="p18-stack",
        fence_epoch=1,
        issuer_record_digest="f" * 64,
    )
    binding = StackActivationBinding(
        provider_parent_generation_id="provider-parent",
        qualification_receipt_digest="a" * 64,
        p19_correctness_event_id="p19-correctness",
        activation_mode="qualified",
        p19_qualified_event_id="p19-qualified",
        p18_stack_gate_event_id="p18-stack",
        stack_generation_id="stack-generation",
        p19_qualified_event_ids=("p19-qualified",),
        activation_token_digest=token.digest(),
    )
    stack = Stack.from_safes(
        safe_paths,
        work_dir=tmp_path / "out",
        activation_mode="qualified",
        activation_binding=binding,
        activation_token=token,
    )
    stack.prepare_scenes()
    reference = np.ones((2, 2), dtype=np.complex64)
    secondary = np.full((2, 2), 1.0 + 2.0j, dtype=np.complex64)
    for date_id in dates:
        root = stack.config.work_dir / "coreg" / date_id / "scenes"
        write_scene_unit(
            root,
            date_id=date_id,
            master_id=dates[0],
            domain="radar",
            tag="IW1_b0",
            reference=reference,
            secondary=secondary,
            row_origin=0,
            col_origin=0,
        )
        stack.coreg_paths[date_id] = root.parent
    stack._write_qualified_activation_record()

    stack.form_interferograms(multilook=(1, 1))
    assert stack.ifg_dirs
