"""Tests for Stack session scaffolding (PROPOSAL-0017)."""

from __future__ import annotations

import weakref
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.processing.contracts import ActivationToken, StackActivationBinding
from faninsar.processing.interferometry.pair import (
    form_interferogram,
    goldstein_filter,
)
from faninsar.processing.merge.grid import GeoGridSpec
from faninsar.processing.stack import Stack, StackConfig
from faninsar.processing.stack.activation import LocalActivationAuthority
from faninsar.processing.stack.catalog import SceneCatalog
from faninsar.processing.stack.ifg_store import (
    InterferogramArtifactStore,
    write_ifg_artifact,
    write_unwrapped_artifact,
)
from faninsar.processing.stack.scene_store import write_scene_unit
from faninsar.processing.timeseries import write_timeseries_zarr

if TYPE_CHECKING:
    from pathlib import Path


def _stack_with_three_date_network(tmp_path: Path) -> Stack:
    """Create a prepared three-date Stack with an explicit triangle network."""
    from faninsar import Pairs

    dates = ("20240101", "20240113", "20240125")
    paths = []
    for date_id in dates:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000.SAFE"
        path.mkdir()
        paths.append(path)
    stack = Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        activation_mode="reference",
        pairs=Pairs.from_names(
            ["20240101_20240113", "20240113_20240125", "20240101_20240125"]
        ),
        multilook=(1, 1),
    )
    return stack.prepare_scenes()


def _write_pair_artifact(
    stack: Stack,
    pair_id: str,
    phase: np.ndarray,
) -> None:
    """Publish one synthetic common-grid pair artifact for Stack tests."""
    root = stack.config.work_dir / "ifg" / "ml_1x1" / pair_id
    complex_ifg = np.exp(1j * phase).astype(np.complex64)
    write_ifg_artifact(
        root,
        pair=tuple(pair_id.split("_")),  # type: ignore[arg-type]
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"scenes": "a" * 64},
        complex_ifg=complex_ifg,
        coherence=np.ones(phase.shape, dtype=np.float32),
        wrapped_phase=np.angle(complex_ifg).astype(np.float32),
        amplitude=np.abs(complex_ifg).astype(np.float32),
    )


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
        activation_mode="reference",
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


def test_coregister_scenes_releases_prior_pair_before_next_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-retained Pair arrays die before the next date starts processing."""
    stack = _stack_with_three_date_network(tmp_path)
    first_payload_ref: weakref.ReferenceType[np.ndarray] | None = None
    call_count = 0

    def fake_run_pair(
        _reference_path: Path,
        secondary_path: Path,
        *,
        scene_store_dir: Path,
        **_kwargs: object,
    ) -> SimpleNamespace:
        nonlocal call_count, first_payload_ref
        if call_count == 1:
            assert first_payload_ref is not None
            assert first_payload_ref() is None
        call_count += 1
        payload = np.ones((64, 64), dtype=np.complex64)
        if first_payload_ref is None:
            first_payload_ref = weakref.ref(payload)
        date_id = secondary_path.name.split("_")[5][:8]
        write_scene_unit(
            scene_store_dir,
            date_id=date_id,
            master_id=stack.master,
            domain="radar",
            tag="f0_IW1_b0",
            reference=np.ones((2, 2), dtype=np.complex64),
            secondary=np.ones((2, 2), dtype=np.complex64),
            row_origin=0,
            col_origin=0,
        )
        return SimpleNamespace(
            _payload=payload,
            esd_azimuth_shift_px=0.0,
            range_shift_px=0.0,
            azimuth_shift_px=0.0,
        )

    monkeypatch.setattr(
        "faninsar.processing.pipeline.production.run_pair", fake_run_pair
    )

    stack.coregister_scenes()

    assert call_count == 2


def test_stack_config_multilook_normalize(tmp_path: Path) -> None:
    """StackConfig coerces multilook to int pair."""
    cfg = StackConfig(
        work_dir=tmp_path,
        activation_mode="reference",
        multilook=(2, 10),
    )
    assert cfg.multilook == (2, 10)


def test_stack_requires_explicit_activation_namespace(tmp_path: Path) -> None:
    """Corrected Stack execution has no implicit reference activation path."""
    with pytest.raises(TypeError, match="activation_mode"):
        StackConfig(work_dir=tmp_path)  # type: ignore[call-arg]


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
        activation_mode="reference",
        coregistration_grid="geo",
        geo_grid=grid,
    )
    assert stack._burst_kwargs()["geo_grid"] is grid


def test_stack_config_rejects_degrade_to_pair(tmp_path: Path) -> None:
    """Qualified Stack configuration cannot silently change coreg semantics."""
    with pytest.raises(ValueError, match="fail-closed"):
        StackConfig(  # type: ignore[arg-type]
            work_dir=tmp_path,
            activation_mode="reference",
            on_network_failure="degrade_to_pair",
        )


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
        activation_mode="reference",
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
    stack = Stack.from_safes(
        safe_paths,
        work_dir=tmp_path / "out",
        activation_mode="reference",
    )
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
    from faninsar.processing.stack.ifg_store import InterferogramArtifactStore

    store = InterferogramArtifactStore.open(output_root)
    assert store.pair == dates
    assert store.shape == (2, 2)
    np.testing.assert_array_equal(store.read().complex_ifg, np.conj(secondary))


def test_stack_applies_multilook_before_publishing_ifg(tmp_path: Path) -> None:
    """Stack payload shape and bytes must match the declared multilook."""
    dates = ("20160101", "20160113")
    safe_paths = []
    for date_id in dates:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000_{date_id}T000001.SAFE"
        path.mkdir()
        safe_paths.append(path)
    stack = Stack.from_safes(
        safe_paths,
        work_dir=tmp_path / "out",
        activation_mode="reference",
    )
    stack.prepare_scenes()

    reference = (
        np.arange(64 * 64, dtype=np.float32).reshape(64, 64) + 1.0 + 1.0j
    ).astype(np.complex64)
    secondary = (np.flip(reference, axis=1) + np.complex64(0.5 + 0.25j)).astype(
        np.complex64
    )
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

    stack.form_interferograms(multilook=(2, 2), goldstein_alpha=0.5)

    from faninsar.processing.stack.ifg_store import InterferogramArtifactStore

    payload = InterferogramArtifactStore.open(stack.ifg_dirs[0]).read().complex_ifg
    expected = goldstein_filter(
        form_interferogram(
            reference,
            secondary,
            multilook=(2, 2),
        ).complex_ifg,
        alpha=0.5,
    )
    np.testing.assert_array_equal(payload, expected)


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
    authority_root = tmp_path / "authority"
    authority = LocalActivationAuthority.initialize(authority_root)
    _ = authority.issue_gate_event(
        event_id="p18-provider",
        gate_id="P18-provider-qualified",
        producer_commit="p18-provider-commit",
        predecessor_event_ids=(),
        provider_receipt_digest="a" * 64,
        parent_manifest_digest="b" * 64,
        evidence_digest="1" * 64,
        activation_mode="qualified",
    )
    _ = authority.issue_gate_event(
        event_id="p19-correctness",
        gate_id="P19-stack-correctness-verified",
        producer_commit="p19-correctness-commit",
        predecessor_event_ids=("p18-provider",),
        provider_receipt_digest="a" * 64,
        parent_manifest_digest="c" * 64,
        evidence_digest="2" * 64,
        activation_mode="reference",
    )
    _ = authority.issue_gate_event(
        event_id="p18-stack",
        gate_id="P18-stack-qualified",
        producer_commit="p18-commit",
        predecessor_event_ids=("p19-correctness",),
        provider_receipt_digest="a" * 64,
        parent_manifest_digest="c" * 64,
        evidence_digest="e" * 64,
        activation_mode="qualified",
    )
    _ = authority.issue_gate_event(
        event_id="p19-qualified",
        gate_id="P19-stack-qualified-activation",
        producer_commit="p19-commit",
        predecessor_event_ids=("p18-stack",),
        provider_receipt_digest="a" * 64,
        parent_manifest_digest="c" * 64,
        evidence_digest="e" * 64,
        activation_mode="qualified",
    )
    token_template = ActivationToken(
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
        issuer_record_digest="0" * 64,
    )
    token = authority.issue_token(token_template)
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
        activation_authority_root=authority_root,
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


def test_stack_unwrap_and_sbas_load_persisted_pair_artifacts(
    tmp_path: Path,
) -> None:
    """Persisted common-grid IFGs flow through temporal unwrap and SBAS."""
    stack = _stack_with_three_date_network(tmp_path)
    first_increment = np.full((3, 4), 0.2, dtype=np.float32)
    second_increment = np.full((3, 4), 0.35, dtype=np.float32)
    phases = {
        "20240101_20240113": first_increment,
        "20240113_20240125": second_increment,
        "20240101_20240125": first_increment + second_increment,
    }
    for pair_id, phase in phases.items():
        _write_pair_artifact(stack, pair_id, phase)

    stack.unwrap(do_spatial=False)
    result = stack.invert_timeseries()

    assert stack.unwrap_result is not None
    assert stack.unwrap_result.temporal_applied
    assert result.pair_ids == tuple(sorted(phases))
    assert result.cumulative.shape == (3, 3, 4)
    for pair_id in phases:
        store = InterferogramArtifactStore.open(
            stack.config.work_dir / "ifg" / "ml_1x1" / pair_id
        )
        unwrapped = store.read_unwrapped()
        assert unwrapped.method == "stack_irls"
        assert unwrapped.method_parameters["pair_ids"] == list(
            stack.unwrap_result.pair_ids
        )
        assert unwrapped.method_parameters["quality_report"]["passed"] is True
        assert "modulo_closure_abs_rad" in unwrapped.method_parameters["quality_report"]

    stack.unwrap_result = None
    stack.unwrap(do_spatial=False)
    assert stack.unwrap_result is not None
    assert stack.unwrap_result.temporal_applied is True
    assert stack.unwrap_result.temporal_converged_pixels == 12
    assert stack.unwrap_result.temporal_converged_fraction == 1.0
    assert stack.unwrap_result.quality_report is not None
    assert stack.unwrap_result.quality_report.passed

    from faninsar.processing.errors import InvalidProcessingStateError
    from faninsar.processing.unwrap.quality import StackQualityCriteria

    with pytest.raises(InvalidProcessingStateError, match="quality criteria"):
        stack.unwrap(
            do_spatial=False,
            quality_criteria=StackQualityCriteria(min_converged_fraction=1.0),
        )


def test_stack_generation_binds_complete_ifg_unwrap_and_timeseries_set(
    tmp_path: Path,
) -> None:
    """One parent generation binds every immutable derived child generation."""
    stack = _stack_with_three_date_network(tmp_path)
    first_increment = np.full((3, 4), 0.2, dtype=np.float32)
    second_increment = np.full((3, 4), 0.35, dtype=np.float32)
    phases = {
        "20240101_20240113": first_increment,
        "20240113_20240125": second_increment,
        "20240101_20240125": first_increment + second_increment,
    }
    for pair_id, phase in phases.items():
        _write_pair_artifact(stack, pair_id, phase)
    stack.unwrap(do_spatial=False)
    timeseries_root = write_timeseries_zarr(
        stack.invert_timeseries(),
        stack.config.work_dir / "timeseries.zarr",
    )

    published = stack.publish_generation(timeseries_root)

    assert published.pair_ids == tuple(phases)
    assert {binding.pair_id for binding in published.pairs} == set(phases)
    assert all(binding.ifg_generation_id for binding in published.pairs)
    assert all(binding.unwrap_generation_id for binding in published.pairs)
    assert published.timeseries_generation_id
    published.close()

    reopened = stack.open_generation()
    assert reopened.generation_id
    assert reopened.pair_ids == tuple(phases)
    assert reopened.manifest_digest
    reopened.close()

    from faninsar.processing.errors import InvalidProcessingStateError

    first_binding = published.pairs[0]
    unwrap_payload = (
        first_binding.artifact_root
        / ".unwrap_generations"
        / first_binding.unwrap_generation_id
        / "unwrapped_phase.npy"
    )
    unwrap_payload.write_bytes(b"corrupted")
    with pytest.raises(InvalidProcessingStateError, match="digest mismatch"):
        stack.open_generation()


def test_partial_unwrap_network_cannot_publish_stack_generation(tmp_path: Path) -> None:
    """A crash-visible subset of unwrap children is never Stack-complete."""
    from faninsar.processing.errors import InvalidProcessingStateError

    stack = _stack_with_three_date_network(tmp_path)
    phases = {
        "20240101_20240113": np.full((3, 4), 0.2, dtype=np.float32),
        "20240113_20240125": np.full((3, 4), 0.35, dtype=np.float32),
        "20240101_20240125": np.full((3, 4), 0.55, dtype=np.float32),
    }
    for pair_id, phase in phases.items():
        _write_pair_artifact(stack, pair_id, phase)
    first_store = InterferogramArtifactStore.open(
        stack.config.work_dir / "ifg/ml_1x1/20240101_20240113"
    )
    write_unwrapped_artifact(
        first_store.root,
        unwrapped_phase=phases["20240101_20240113"],
        connected_components=np.ones((3, 4), dtype=np.int32),
        method="stack_irls",
        method_parameters={"pair_ids": list(phases)},
        ifg_manifest_digest=first_store.manifest_digest,
    )
    first_store.close()
    timeseries_root = write_timeseries_zarr(
        stack.invert_timeseries(pair_phases=phases),
        stack.config.work_dir / "timeseries.zarr",
    )

    with pytest.raises(InvalidProcessingStateError):
        stack.publish_generation(timeseries_root)

    assert not (stack.config.work_dir / "STACK_CURRENT").exists()


def test_scene_artifacts_flow_through_merge_unwrap_and_sbas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three dates and overlapping bursts complete the persisted Stack chain."""
    stack = _stack_with_three_date_network(tmp_path)
    scene_phase = {
        "20240101": 0.0,
        "20240113": -0.2,
        "20240125": -0.55,
    }
    for date_id, phase in scene_phase.items():
        root = stack.config.work_dir / "coreg" / date_id / "scenes"
        aligned = np.full(
            (2, 4),
            np.exp(1j * phase),
            dtype=np.complex64,
        )
        reference = np.ones((2, 4), dtype=np.complex64)
        for tag, row_origin in (("f0_IW1_b0", 0), ("f0_IW1_b1", 2)):
            write_scene_unit(
                root,
                date_id=date_id,
                master_id="20240101",
                domain="radar",
                tag=tag,
                reference=reference,
                secondary=aligned,
                row_origin=row_origin,
                col_origin=0,
                grid_shape=(4, 4),
                wavelength_m=0.056,
            )
        stack.coreg_paths[date_id] = root.parent

    stack.form_interferograms(multilook=(1, 1))
    stack.unwrap(do_spatial=False)
    from faninsar.processing.timeseries import inversion as inversion_module

    original_invert = inversion_module.invert_unwrapped_pairs

    def assert_released_before_sbas(*args: object, **kwargs: object) -> object:
        assert stack.unwrap_result is not None
        assert stack.unwrap_result.phase_2d_unw is None
        assert stack.unwrap_result.phase_1d_unw is None
        assert stack.unwrap_result.corrections_k is None
        assert stack.unwrap_result.temporal_converged_mask is None
        return original_invert(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        inversion_module,
        "invert_unwrapped_pairs",
        assert_released_before_sbas,
    )
    result = stack.invert_timeseries()

    np.testing.assert_allclose(result.phase_cumulative_rad[1], 0.2, atol=1e-6)
    np.testing.assert_allclose(result.phase_cumulative_rad[2], 0.55, atol=1e-6)
    assert result.displacement_cumulative_m is not None
    np.testing.assert_allclose(
        result.displacement_cumulative_m,
        result.phase_cumulative_rad * (-0.056 / (4.0 * np.pi)),
    )
    assert all((directory / "manifest.json").is_file() for directory in stack.ifg_dirs)
    assert stack.unwrap_result is not None
    assert stack.unwrap_result.phase_2d_unw is None
    assert stack.unwrap_result.phase_1d_unw is None
    assert stack.unwrap_result.corrections_k is None


def test_form_interferograms_rejects_stale_scene_lineage(tmp_path: Path) -> None:
    """An existing IFG cannot be reused after a source scene generation changes."""
    from faninsar.processing.errors import InvalidProcessingStateError

    dates = ("20240101", "20240113")
    paths = []
    for date_id in dates:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000.SAFE"
        path.mkdir()
        paths.append(path)
    stack = Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        activation_mode="reference",
        multilook=(1, 1),
    ).prepare_scenes()
    data = np.ones((2, 2), dtype=np.complex64)
    for date_id in dates:
        root = stack.config.work_dir / "coreg" / date_id / "scenes"
        write_scene_unit(
            root,
            date_id=date_id,
            master_id=dates[0],
            domain="radar",
            tag="f0_IW1_b0",
            reference=data,
            secondary=data,
            row_origin=0,
            col_origin=0,
            grid_shape=(2, 2),
        )
        stack.coreg_paths[date_id] = root.parent
    stack.form_interferograms()
    changed = np.full((2, 2), 2.0 + 0.0j, dtype=np.complex64)
    write_scene_unit(
        stack.coreg_paths[dates[1]] / "scenes",
        date_id=dates[1],
        master_id=dates[0],
        domain="radar",
        tag="f0_IW1_b0",
        reference=data,
        secondary=changed,
        row_origin=0,
        col_origin=0,
        grid_shape=(2, 2),
    )

    with pytest.raises(InvalidProcessingStateError, match="lineage"):
        stack.form_interferograms()


def test_stack_unwrap_fails_closed_for_incomplete_pair_network(
    tmp_path: Path,
) -> None:
    """A missing common pair view cannot silently shrink the SBAS network."""
    from faninsar.processing.errors import InvalidProcessingStateError

    stack = _stack_with_three_date_network(tmp_path)
    phase = np.zeros((2, 2), dtype=np.float32)
    _write_pair_artifact(stack, "20240101_20240113", phase)
    _write_pair_artifact(stack, "20240113_20240125", phase)

    with pytest.raises(InvalidProcessingStateError, match="pair set"):
        stack.unwrap(do_spatial=False)


def test_stack_unwrap_fails_closed_for_mixed_common_grids(tmp_path: Path) -> None:
    """Pair artifacts on different shapes cannot enter temporal processing."""
    from faninsar.processing.errors import InvalidProcessingStateError

    stack = _stack_with_three_date_network(tmp_path)
    _write_pair_artifact(
        stack,
        "20240101_20240113",
        np.zeros((2, 2), dtype=np.float32),
    )
    _write_pair_artifact(
        stack,
        "20240113_20240125",
        np.zeros((2, 2), dtype=np.float32),
    )
    _write_pair_artifact(
        stack,
        "20240101_20240125",
        np.zeros((3, 2), dtype=np.float32),
    )

    with pytest.raises(InvalidProcessingStateError, match="common grid"):
        stack.unwrap(do_spatial=False)


def test_stack_unwrap_rejects_when_no_temporal_pixel_converges(
    tmp_path: Path,
) -> None:
    """A bounded temporal solve cannot publish an unqualified generation."""
    from faninsar.processing.errors import InvalidProcessingStateError

    stack = _stack_with_three_date_network(tmp_path)
    for pair_id, value in (
        ("20240101_20240113", 0.2),
        ("20240113_20240125", 0.3),
        ("20240101_20240125", 0.5),
    ):
        _write_pair_artifact(
            stack,
            pair_id,
            np.full((2, 2), value, dtype=np.float32),
        )

    with pytest.raises(InvalidProcessingStateError, match="no converged pixels"):
        stack.unwrap(
            do_spatial=False,
            temporal_kwargs={"max_iter": 1},
        )


def test_stack_invert_rejects_unqualified_unwrap_artifacts(tmp_path: Path) -> None:
    """Direct SBAS invocation cannot bypass temporal qualification metadata."""
    from faninsar.processing.errors import InvalidProcessingStateError

    stack = _stack_with_three_date_network(tmp_path)
    phase = np.zeros((2, 2), dtype=np.float32)
    for pair_id in (
        "20240101_20240113",
        "20240113_20240125",
        "20240101_20240125",
    ):
        _write_pair_artifact(stack, pair_id, phase)
        store = InterferogramArtifactStore.open(
            stack.config.work_dir / "ifg" / "ml_1x1" / pair_id
        )
        write_unwrapped_artifact(
            store.root,
            unwrapped_phase=phase,
            connected_components=np.ones((2, 2), dtype=np.int32),
            method="legacy",
            method_parameters={},
            ifg_manifest_digest=store.manifest_digest,
        )

    with pytest.raises(InvalidProcessingStateError, match="qualified"):
        stack.invert_timeseries()


def test_coregister_resume_rejects_scene_aligned_to_old_master(
    tmp_path: Path,
) -> None:
    """A work directory cannot resume scene artifacts from another master."""
    import json

    from faninsar.processing.errors import InvalidProcessingStateError

    stack = _stack_with_three_date_network(tmp_path)
    date_id = "20240113"
    out = stack.config.work_dir / "coreg" / date_id
    data = np.ones((2, 2), dtype=np.complex64)
    write_scene_unit(
        out / "scenes",
        date_id=date_id,
        master_id="20231220",
        domain="radar",
        tag="f0_IW1_b0",
        reference=data,
        secondary=data,
        row_origin=0,
        col_origin=0,
    )
    (out / "coreg_done.json").write_text(
        json.dumps({"master": "20231220", "date": date_id}),
        encoding="utf-8",
    )

    with pytest.raises(InvalidProcessingStateError, match="master"):
        stack.coregister_scenes(dates=[date_id])


def test_coregister_resume_rejects_changed_burst_request(tmp_path: Path) -> None:
    """A marker for burst zero cannot authorize reuse for burst one."""
    import json

    from faninsar.processing.errors import InvalidProcessingStateError

    stack = _stack_with_three_date_network(tmp_path)
    date_id = "20240113"
    out = stack.config.work_dir / "coreg" / date_id
    data = np.ones((2, 2), dtype=np.complex64)
    write_scene_unit(
        out / "scenes",
        date_id=date_id,
        master_id=stack.master,
        domain="radar",
        tag="f0_IW1_b0",
        reference=data,
        secondary=data,
        row_origin=0,
        col_origin=0,
    )
    initial_identity = stack._coreg_resume_identity(
        date_id,
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    (out / "coreg_done.json").write_text(
        json.dumps(
            {
                "master": stack.master,
                "date": date_id,
                "coreg_identity": initial_identity,
            }
        ),
        encoding="utf-8",
    )
    stack.config.bursts = {"IW1": [1]}

    with pytest.raises(InvalidProcessingStateError, match="coregistration scene"):
        stack.coregister_scenes(dates=[date_id])


def test_coreg_resume_identity_captures_nested_dem_sampling_semantics(
    tmp_path: Path,
) -> None:
    """Interpolation and nested geoid samplers must invalidate scene reuse."""
    from faninsar.processing.geometry.dem import (
        ConstantHeightDEM,
        GeoidAdjustedDEM,
        RasterDEM,
    )

    stack = _stack_with_three_date_network(tmp_path)
    date_id = "20240113"
    raster_path = tmp_path / "dem.tif"
    raster_path.write_bytes(b"identity-only-fixture")
    stack.config.dem = RasterDEM(raster_path, interpolation="bilinear", nodata=-9999)
    bilinear_identity = stack._coreg_resume_identity(
        date_id,
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    stack.config.dem = RasterDEM(raster_path, interpolation="bicubic", nodata=-9999)
    bicubic_identity = stack._coreg_resume_identity(
        date_id,
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    stack.config.dem = GeoidAdjustedDEM(
        ConstantHeightDEM(10.0),
        ConstantHeightDEM(2.0),
    )
    first_nested_identity = stack._coreg_resume_identity(
        date_id,
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    stack.config.dem = GeoidAdjustedDEM(
        ConstantHeightDEM(10.0),
        ConstantHeightDEM(3.0),
    )
    second_nested_identity = stack._coreg_resume_identity(
        date_id,
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )

    assert bilinear_identity != bicubic_identity
    assert first_nested_identity != second_nested_identity
