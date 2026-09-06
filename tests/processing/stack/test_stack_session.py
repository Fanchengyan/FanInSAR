"""Tests for Stack session scaffolding (PROPOSAL-0017)."""

from __future__ import annotations

import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.core.network import AssetKind
from faninsar.processing.contracts import ActivationToken, StackActivationBinding
from faninsar.processing.interferometry.pair import (
    form_interferogram,
    goldstein_filter,
)
from faninsar.processing.merge.grid import GeoGridSpec
from faninsar.processing.timeseries import write_timeseries_zarr
from faninsar.stack import Stack, StackConfig, StackSceneProvider
from faninsar.stack.activation import LocalActivationAuthority
from faninsar.stack.catalog import SceneCatalog
from faninsar.stack.ifg_store import (
    InterferogramArtifactStore,
    write_ifg_artifact,
    write_unwrapped_artifact,
)
from faninsar.stack.scene_store import write_scene_unit

if TYPE_CHECKING:
    from faninsar.stack.provider import SourceHandle


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
    if root not in stack.ifg_dirs:
        stack.ifg_dirs.append(root)


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
    """Stack builds default short-baseline pairs and earliest Reference."""
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
    assert stack.reference == "20160101"
    assert len(stack.catalog) == 3
    stack.prepare_scenes()
    assert (tmp_path / "out" / "coreg").is_dir()
    # geometry mode skips measure/invert
    stack.measure_misreg()
    stack.invert_misreg()
    assert stack.date_misreg is None


def test_stack_auto_grid_uses_selected_footprint_union_when_roi_omitted(
    tmp_path: Path,
) -> None:
    """Automatic Stack grid derives one deterministic extent after selection."""
    stack = _stack_with_three_date_network(tmp_path)
    stack.config.extra["selected_footprints"] = (
        (10.0, 40.0, 10.2, 40.2),
        (10.4, 40.1, 10.6, 40.3),
    )
    grid = stack.resolve_grid()
    assert grid.crs == "EPSG:32632"
    assert grid.width > 1000
    assert grid.height > 1000


def test_coreg_resume_identity_accepts_all_burst_token(tmp_path: Path) -> None:
    """Burst selection token ``all`` is part of the request identity."""
    paths = []
    for day in ("20160101", "20160113"):
        path = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000_{day}T000001.SAFE"
        path.mkdir()
        paths.append(path)
    stack = Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        activation_mode="reference",
        coreg_mode="pair",
        swaths=("IW1", "IW2"),
        bursts={"IW1": "all", "IW2": "all"},
    )
    digest = stack._coreg_resume_identity(
        "20160113",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    assert len(digest) == 64


def test_coreg_resume_identity_binds_provider_window_channel_and_source_metadata(
    tmp_path: Path,
) -> None:
    """Provider-owned window, channel, and source content invalidate reuse."""
    stack = _stack_with_three_date_network(tmp_path)
    stack.config.extra.update(
        {
            "mission": "NISAR",
            "product": "RSLC",
            "frequency": "B",
            "polarization": "HH",
            "nisar_window": (0, 16, 0, 16),
        }
    )
    stack.scene_provider = SimpleNamespace(
        name="NISAR RSLC",
        capability="scene-production",
        produce_pair=lambda *_args, **_kwargs: None,
        identity_metadata={"provider_revision": "v1"},
    )
    stack._nisar_channel = ("B", "HH")
    stack._nisar_results = {
        stack.reference: SimpleNamespace(source_id="reference", content_digest="m1"),
        "20240113": SimpleNamespace(source_id="secondary", content_digest="s1"),
    }
    initial = stack._coreg_resume_identity(
        "20240113",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )

    stack.config.extra["nisar_window"] = (0, 32, 0, 16)
    changed_window = stack._coreg_resume_identity(
        "20240113",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    stack.config.extra["nisar_window"] = (0, 16, 0, 16)
    stack._nisar_channel = ("A", "HH")
    changed_channel = stack._coreg_resume_identity(
        "20240113",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    stack._nisar_channel = ("B", "HH")
    stack._nisar_results["20240113"].content_digest = "s2"
    changed_source = stack._coreg_resume_identity(
        "20240113",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )

    assert initial != changed_window
    assert initial != changed_channel
    assert initial != changed_source


def test_s1_default_coreg_resume_identity_is_stable(tmp_path: Path) -> None:
    """The legacy S1 path remains provider-free and deterministic."""
    stack = _stack_with_three_date_network(tmp_path)
    first = stack._coreg_resume_identity(
        "20240113",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    second = stack._coreg_resume_identity(
        "20240113",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )

    assert isinstance(stack.scene_provider, StackSceneProvider)
    assert first == second


def test_coreg_resume_restores_burst_array_shape_from_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Radar resume accepts the production BurstArray shape contract."""
    from types import SimpleNamespace

    from faninsar.missions.sentinel1.io import BurstArray

    stack = _stack_with_three_date_network(tmp_path)
    shape = (5, 7)
    scene = SimpleNamespace(
        array=BurstArray(
            samples=np.ones(shape, dtype=np.complex64),
            row0=0,
            col0=0,
            burst_index=0,
            valid_mask=np.ones(shape, dtype=bool),
        ),
        geometry=object(),
    )

    def restore_scene(*_args: object, **_kwargs: object) -> object:
        """Return a production-like scene with a BurstArray payload."""
        return scene

    monkeypatch.setattr(
        "faninsar.processing.stages.load_production_scene", restore_scene
    )

    stack._restore_radar_projection_context(
        {
            "radar_projection_context": {
                "source_path": "scene.SAFE",
                "swath": "IW1",
                "burst_index": 0,
                "orbit_path": None,
                "full_range": True,
                "full_radar_shape": list(shape),
                "reference_scene": stack.reference,
                "master_grid": {},
            }
        }
    )

    assert stack._radar_projection_context is not None
    assert stack._radar_projection_context["full_radar_shape"] == shape


def test_stack_runtime_identity_mutations_invalidate_coreg_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runtime and callback identity changes cannot reuse coregistration."""
    from faninsar.stack import session as session_module

    stack = _stack_with_three_date_network(tmp_path)
    initial = stack._coreg_resume_identity(
        "20240113",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )

    monkeypatch.setattr(
        session_module,
        "_faninsar_git_revision",
        lambda: "different-revision",
    )
    changed_revision = stack._coreg_resume_identity(
        "20240113",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    assert changed_revision != initial

    monkeypatch.setattr(
        session_module,
        "_runtime_image_identity",
        lambda: {"platform": "different-runtime-image"},
    )
    changed_image = stack._coreg_resume_identity(
        "20240113",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    assert changed_image != changed_revision

    monkeypatch.setattr(
        session_module,
        "_cuda_runtime_identity",
        lambda: {
            "available": True,
            "driver": "different-driver",
            "runtime": "different-runtime",
            "devices": [{"index": 0, "uuid": "different-uuid", "name": "different"}],
        },
    )
    changed_cuda = stack._coreg_resume_identity(
        "20240113",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    assert changed_cuda != changed_image

    def callback(_reference: object, _secondary: object, **_kwargs: object) -> None:
        """Mutation-only callback fixture."""

    stack.scene_provider = SimpleNamespace(produce_pair=callback)
    changed_callback = stack._coreg_resume_identity(
        "20240113",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    assert changed_callback != changed_cuda


def test_ifg_resume_binds_runtime_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IFG source lineage includes the same runtime fingerprint as coreg."""
    from faninsar.stack import session as session_module

    stack = _stack_with_three_date_network(tmp_path)
    data = np.ones((2, 2), dtype=np.complex64)
    for date_id in stack.catalog.dates:
        root = stack.config.work_dir / "coreg" / date_id / "scenes"
        write_scene_unit(
            root,
            date_id=date_id,
            reference_id=stack.reference,
            domain="radar",
            tag="f0_IW1_b0",
            primary=data,
            secondary=data,
            row_origin=0,
            col_origin=0,
        )
        stack.coreg_paths[date_id] = root.parent

    stack.form_interferograms(multilook=(1, 1))
    store = InterferogramArtifactStore.open(stack.ifg_dirs[0])
    assert store.source_manifest_digests[
        "runtime"
    ] == session_module._runtime_fingerprint(stack.scene_provider.produce_pair)

    monkeypatch.setattr(
        session_module, "_runtime_fingerprint", lambda _callback: "f" * 64
    )
    from faninsar.processing.errors import InvalidProcessingStateError

    with pytest.raises(InvalidProcessingStateError, match="lineage"):
        stack.form_interferograms(multilook=(1, 1))


def test_coreg_resume_rejects_marker_after_provider_identity_change(
    tmp_path: Path,
) -> None:
    """An old provider marker cannot authorize a changed source window."""
    import json

    from faninsar.processing.errors import InvalidProcessingStateError

    stack = _stack_with_three_date_network(tmp_path)
    stack.config.extra.update({"mission": "NISAR", "nisar_window": (0, 16, 0, 16)})
    stack.scene_provider = SimpleNamespace(
        name="NISAR RSLC",
        capability="scene-production",
        produce_pair=lambda *_args, **_kwargs: None,
        identity_metadata={"provider_revision": "v1"},
    )
    date_id = "20240113"
    out = stack.config.work_dir / "coreg" / date_id
    data = np.ones((2, 2), dtype=np.complex64)
    write_scene_unit(
        out / "scenes",
        date_id=date_id,
        reference_id=stack.reference,
        domain="radar",
        tag="f0_IW1_b0",
        primary=data,
        secondary=data,
        row_origin=0,
        col_origin=0,
    )
    marker = {
        "reference": stack.reference,
        "date": date_id,
        "coreg_identity": stack._coreg_resume_identity(
            date_id,
            misreg_az_px=0.0,
            misreg_rg_px=0.0,
        ),
    }
    (out / "coreg_done.json").write_text(json.dumps(marker), encoding="utf-8")
    stack.config.extra["nisar_window"] = (0, 32, 0, 16)

    with pytest.raises(InvalidProcessingStateError, match="current date"):
        stack.coregister_scenes(dates=[date_id])


def test_coregister_scenes_releases_prior_pair_before_next_date(
    tmp_path: Path,
) -> None:
    """Non-retained Pair arrays die before the next date starts processing."""
    stack = _stack_with_three_date_network(tmp_path)
    first_payload_ref: weakref.ReferenceType[np.ndarray] | None = None
    call_count = 0

    def fake_produce_pair(
        _reference_path: SourceHandle,
        secondary_handle: SourceHandle,
        *,
        options: dict[str, object],
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
        secondary_path = secondary_handle._resolve()[0]
        scene_store_dir = options["scene_store_dir"]
        assert isinstance(scene_store_dir, Path)
        date_id = secondary_path.name.split("_")[5][:8]
        write_scene_unit(
            scene_store_dir,
            date_id=date_id,
            reference_id=stack.reference,
            domain="radar",
            tag="f0_IW1_b0",
            primary=np.ones((2, 2), dtype=np.complex64),
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

    stack.scene_provider = StackSceneProvider(
        produce_pair=fake_produce_pair,
        name="TEST",
    )

    stack.coregister_scenes()

    assert call_count == 2


def test_s1_default_scene_dispatch_uses_provider(
    tmp_path: Path,
) -> None:
    """SAFE stacks dispatch scene production through their provider."""
    stack = _stack_with_three_date_network(tmp_path)
    calls: list[tuple[Path, Path, Path]] = []
    expected = SimpleNamespace(esd_azimuth_shift_px=0.0)

    def fake_produce_pair(
        reference_path: SourceHandle,
        secondary_path: SourceHandle,
        *,
        output_dir: Path,
        options: dict[str, object],
    ) -> SimpleNamespace:
        del options
        calls.append(
            (
                reference_path._resolve()[0],
                secondary_path._resolve()[0],
                output_dir,
            )
        )
        return expected

    stack.scene_provider = StackSceneProvider(
        produce_pair=fake_produce_pair,
        name="TEST",
    )

    result = stack._produce_pair(
        stack.catalog.path_for(stack.reference),
        stack.catalog.path_for("20240113"),
        output_dir=tmp_path / "pair",
        multilook=(1, 1),
    )

    assert result is expected
    assert calls == [
        (
            stack.catalog.path_for(stack.reference),
            stack.catalog.path_for("20240113"),
            tmp_path / "pair",
        )
    ]


def test_stack_scene_provider_receives_normalized_callback_arguments(
    tmp_path: Path,
) -> None:
    """A mission provider receives paths, output, and one options mapping."""
    stack = _stack_with_three_date_network(tmp_path)
    calls: list[tuple[Path, Path, Path, dict[str, object]]] = []

    def produce_pair(
        reference_path: SourceHandle,
        secondary_path: SourceHandle,
        *,
        output_dir: Path,
        options: dict[str, object],
    ) -> SimpleNamespace:
        calls.append(
            (
                reference_path._resolve()[0],
                secondary_path._resolve()[0],
                output_dir,
                options,
            )
        )
        return SimpleNamespace()

    stack.scene_provider = StackSceneProvider(produce_pair=produce_pair, name="TEST")
    result = stack._produce_pair(
        stack.catalog.path_for(stack.reference),
        stack.catalog.path_for("20240113"),
        output_dir=tmp_path / "pair",
        multilook=(1, 1),
    )

    assert isinstance(result, SimpleNamespace)
    assert calls[0][:3] == (
        stack.catalog.path_for(stack.reference),
        stack.catalog.path_for("20240113"),
        tmp_path / "pair",
    )
    assert calls[0][3] == {"multilook": (1, 1)}


def test_stack_config_multilook_normalize(tmp_path: Path) -> None:
    """StackConfig coerces multilook to int pair."""
    cfg = StackConfig(
        work_dir=tmp_path,
        activation_mode="reference",
        multilook=(2, 10),
    )
    assert cfg.multilook == (2, 10)


def test_stack_config_default_multilook_is_isce2_like(tmp_path: Path) -> None:
    """Stack defaults to five azimuth by two range looks."""
    cfg = StackConfig(work_dir=tmp_path, activation_mode="reference")
    assert cfg.multilook == (5, 2)


def test_stack_formation_validates_coherence_before_scene_io(tmp_path: Path) -> None:
    """Invalid coherence support fails before formation reads any scene store."""
    paths = []
    for day in ("20160101", "20160113"):
        path = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000_{day}T000001.SAFE"
        path.mkdir()
        paths.append(path)
    stack = Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        activation_mode="reference",
    )

    with pytest.raises(ValueError, match="odd integers >= 3"):
        stack.form_interferograms(coherence_window=(2, 5))

    assert not (tmp_path / "out").exists()


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
    tmp_path: Path,
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

    stack.scene_provider = StackSceneProvider(
        name="TEST",
        produce_pair=lambda *_args, **_kwargs: SimpleNamespace(
            esd_azimuth_shift_px=-0.3, amplitude_residual_rg_px=0.7
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
                reference_id=dates[0],
                domain="radar",
                tag=tag,
                primary=reference,
                secondary=secondary,
                row_origin=0,
                col_origin=0,
            )
        stack.coreg_paths[date_id] = root.parent

    stack.form_interferograms(multilook=(1, 1))
    output_root = stack.ifg_dirs[0]
    from faninsar.stack.ifg_store import InterferogramArtifactStore

    store = InterferogramArtifactStore.open(output_root)
    assert store.pair == dates
    assert store.shape == (2, 2)
    np.testing.assert_array_equal(store.read().complex_ifg, np.conj(secondary))


def test_stack_applies_multilook_before_publishing_ifg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stack payload shape and bytes must match the declared multilook."""
    dates = ("20160101", "20160113")
    safe_paths = []
    for date_id in dates:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000_{date_id}T000001.SAFE"
        path.mkdir()
        safe_paths.append(path)
    client = object()
    observed: dict[str, object] = {}
    from faninsar.stack import session as session_module

    original_form = session_module.form_merged_scene_interferogram

    def wrapped_form(*args: object, **kwargs: object) -> object:
        observed.update(kwargs)
        kwargs["dask_client"] = None
        return original_form(*args, **kwargs)

    monkeypatch.setattr(session_module, "form_merged_scene_interferogram", wrapped_form)
    stack = Stack.from_safes(
        safe_paths,
        work_dir=tmp_path / "out",
        activation_mode="reference",
        dask_client=client,
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
            reference_id=dates[0],
            domain="radar",
            tag="IW1_b0",
            primary=reference,
            secondary=secondary,
            row_origin=0,
            col_origin=0,
        )
        stack.coreg_paths[date_id] = root.parent

    stack.form_interferograms(multilook=(2, 2), goldstein_alpha=0.5)
    assert observed["dask_client"] is client

    from faninsar.stack.ifg_store import InterferogramArtifactStore

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
            reference_id=dates[0],
            domain="radar",
            tag="IW1_b0",
            primary=reference,
            secondary=secondary,
            row_origin=0,
            col_origin=0,
        )
        stack.coreg_paths[date_id] = root.parent
    stack._write_qualified_activation_record()

    stack.form_interferograms(multilook=(1, 1))
    assert stack.ifg_dirs
    assert not stack.analysis_ready


def test_stack_unwrap_loads_persisted_pair_artifacts(
    tmp_path: Path,
) -> None:
    """Persisted common-grid IFGs flow through spatial Stack unwrapping."""
    stack = _stack_with_three_date_network(tmp_path)
    phases = {
        "20240101_20240113": np.full((3, 4), 0.2, dtype=np.float32),
        "20240113_20240125": np.full((3, 4), 0.35, dtype=np.float32),
        "20240101_20240125": np.full((3, 4), 0.55, dtype=np.float32),
    }
    for pair_id, phase in phases.items():
        _write_pair_artifact(stack, pair_id, phase)

    stack.unwrap()
    assert stack.analysis_ready
    index = stack.network_product_index
    assert index is not None
    product_kinds = tuple(product.key.product_kind for product in index.products)
    assert product_kinds.count(AssetKind.COMPLEX_INTERFEROGRAM) == len(phases)
    assert product_kinds.count(AssetKind.UNWRAPPED_PHASE) == len(phases)
    assert len(product_kinds) == 2 * len(phases)
    for product in index.products:
        assert product.content_digest
        assert product.lineage
    unwrapped = [
        product
        for product in index.products
        if product.key.product_kind is AssetKind.UNWRAPPED_PHASE
    ]
    assert len(unwrapped) == len(phases)
    assert all(product.content_digest for product in unwrapped)


def test_stack_unwrap_publishes_durable_network_generation(tmp_path: Path) -> None:
    """A spatial unwrap generation is durable and available to the Network."""
    stack = _stack_with_three_date_network(tmp_path)
    phases = {
        "20240101_20240113": np.full((3, 4), 0.2, dtype=np.float32),
        "20240113_20240125": np.full((3, 4), 0.35, dtype=np.float32),
        "20240101_20240125": np.full((3, 4), 0.55, dtype=np.float32),
    }
    for pair_id, phase in phases.items():
        _write_pair_artifact(stack, pair_id, phase)
    stack.unwrap()
    assert stack.analysis_ready
    assert stack.network_product_index is not None


def test_publish_generation_exports_canonical_network_view(tmp_path: Path) -> None:
    """A completed Stack generation can be admitted through Network.open."""
    from faninsar.network import Network

    stack = _stack_with_three_date_network(tmp_path)
    phases = {
        "20240101_20240113": np.full((3, 4), 0.2, dtype=np.float32),
        "20240113_20240125": np.full((3, 4), 0.35, dtype=np.float32),
        "20240101_20240125": np.full((3, 4), 0.55, dtype=np.float32),
    }
    for pair_id, phase in phases.items():
        _write_pair_artifact(stack, pair_id, phase)
    stack.unwrap()
    timeseries_root = write_timeseries_zarr(
        stack.invert_timeseries(), stack.config.work_dir / "timeseries.zarr"
    )

    generation = stack.publish_generation(timeseries_root)
    try:
        network = Network.open(stack.config.work_dir / "network")
        assert network.manifest["generation_id"] == generation.generation_id
        assert network.interferograms.open_stack("unw_phase").shape == (3, 3, 4)
        result = network.analyze_time_series()
        assert result.revision_id == generation.generation_id
    finally:
        generation.close()


def test_refresh_rejects_root_unwrap_bound_to_stale_ifg(tmp_path: Path) -> None:
    """Refreshing a root unwrap cannot silently mix IFG generations."""
    from faninsar.processing.errors import InvalidProcessingStateError

    stack = _stack_with_three_date_network(tmp_path)
    phase = np.full((3, 4), 0.2, dtype=np.float32)
    _write_pair_artifact(stack, "20240101_20240113", phase)
    _write_pair_artifact(stack, "20240113_20240125", phase)
    _write_pair_artifact(stack, "20240101_20240125", phase)
    stack.unwrap()
    store = InterferogramArtifactStore.open(
        stack.config.work_dir / "ifg/ml_1x1/20240101_20240113"
    )
    try:
        artifact = store.read()
        write_ifg_artifact(
            store.root,
            pair=store.pair,
            looks=store.looks,
            filter_name=store.filter_name,
            filter_parameters=store.filter_parameters,
            source_manifest_digests=store.source_manifest_digests,
            grid_identity=store.grid_identity,
            complex_ifg=artifact.complex_ifg * 1.0,
            coherence=artifact.coherence,
            wrapped_phase=artifact.wrapped_phase,
            amplitude=artifact.amplitude,
            replace_existing=True,
        )
    finally:
        store.close()
    with pytest.raises(InvalidProcessingStateError, match="stale"):
        stack.refresh_unwrap_generation()


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
    stack.ifg_dirs = [
        stack.config.work_dir / "ifg" / "ml_1x1" / pair_id for pair_id in phases
    ]
    with pytest.raises(InvalidProcessingStateError, match="partial unwrap product set"):
        stack._refresh_network_from_ifg_dirs()
    assert not stack.analysis_ready

    timeseries_root = write_timeseries_zarr(
        stack.invert_timeseries(pair_phases=phases),
        stack.config.work_dir / "timeseries.zarr",
    )

    with pytest.raises(InvalidProcessingStateError):
        stack.publish_generation(timeseries_root)

    assert not (stack.config.work_dir / "STACK_CURRENT").exists()


def test_scene_artifacts_flow_through_merge_and_spatial_unwrap(
    tmp_path: Path,
) -> None:
    """Three dates and overlapping bursts complete spatial Stack processing."""
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
                reference_id="20240101",
                domain="radar",
                tag=tag,
                primary=reference,
                secondary=aligned,
                row_origin=row_origin,
                col_origin=0,
                grid_shape=(4, 4),
                wavelength_m=0.056,
            )
        stack.coreg_paths[date_id] = root.parent

    stack.form_interferograms(multilook=(1, 1))
    stack.unwrap()
    assert all((directory / "manifest.json").is_file() for directory in stack.ifg_dirs)
    assert stack.analysis_ready


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
            reference_id=dates[0],
            domain="radar",
            tag="f0_IW1_b0",
            primary=data,
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
        reference_id=dates[0],
        domain="radar",
        tag="f0_IW1_b0",
        primary=data,
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
        stack.unwrap()


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
        stack.unwrap()


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


def test_coregister_resume_rejects_scene_aligned_to_old_reference(
    tmp_path: Path,
) -> None:
    """A work directory cannot resume scene artifacts from another Reference."""
    import json

    from faninsar.processing.errors import InvalidProcessingStateError

    stack = _stack_with_three_date_network(tmp_path)
    date_id = "20240113"
    out = stack.config.work_dir / "coreg" / date_id
    data = np.ones((2, 2), dtype=np.complex64)
    write_scene_unit(
        out / "scenes",
        date_id=date_id,
        reference_id="20231220",
        domain="radar",
        tag="f0_IW1_b0",
        primary=data,
        secondary=data,
        row_origin=0,
        col_origin=0,
    )
    (out / "coreg_done.json").write_text(
        json.dumps({"reference": "20231220", "date": date_id}),
        encoding="utf-8",
    )

    with pytest.raises(InvalidProcessingStateError, match="Reference"):
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
        reference_id=stack.reference,
        domain="radar",
        tag="f0_IW1_b0",
        primary=data,
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
                "reference": stack.reference,
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
    from affine import Affine

    from faninsar.processing.dem import DEM, GridSpec, RasterDEM
    from faninsar.processing.geometry.dem import GeoidAdjustedDEM

    stack = _stack_with_three_date_network(tmp_path)
    date_id = "20240113"
    grid = GridSpec(
        "EPSG:4326",
        Affine(0.1, 0.0, 10.0, 0.0, -0.1, 42.0),
        shape=(8, 8),
    )
    stack.config.dem = RasterDEM(
        array=np.full(grid.shape, 10.0, dtype=np.float32), grid=grid
    )
    first_identity = stack._coreg_resume_identity(
        date_id, misreg_az_px=0.0, misreg_rg_px=0.0
    )
    stack.config.dem = RasterDEM(
        array=np.full(grid.shape, 11.0, dtype=np.float32), grid=grid
    )
    second_identity = stack._coreg_resume_identity(
        date_id, misreg_az_px=0.0, misreg_rg_px=0.0
    )
    stack.config.dem = GeoidAdjustedDEM(
        DEM.from_constant(10.0),
        DEM.from_constant(2.0),
    )
    first_nested_identity = stack._coreg_resume_identity(
        date_id,
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    stack.config.dem = GeoidAdjustedDEM(
        DEM.from_constant(10.0),
        DEM.from_constant(3.0),
    )
    second_nested_identity = stack._coreg_resume_identity(
        date_id,
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )

    assert first_identity != second_identity
    assert first_nested_identity != second_nested_identity


def test_stack_config_gpu_memory_reclaim_defaults_to_adaptive(tmp_path: Path) -> None:
    """PROPOSAL-0034: StackConfig.gpu_memory_reclaim default is adaptive."""
    paths = []
    for day in ("20160101", "20160113"):
        path = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000_{day}T000001.SAFE"
        path.mkdir()
        paths.append(path)
    stack = Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        activation_mode="reference",
    )
    assert stack.config.gpu_memory_reclaim == "adaptive"
    config = StackConfig(work_dir=tmp_path / "cfg", activation_mode="reference")
    assert config.gpu_memory_reclaim == "adaptive"


def test_reclaim_checkpoint_policy_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """lazy/eager/adaptive follow the capacity and pressure table."""
    torch = pytest.importorskip("torch")
    from faninsar._core.device import reclaim_checkpoint

    calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    device = torch.device("cuda")
    gib = 1024**3
    assert (
        reclaim_checkpoint(
            device, "lazy", kind="persist", total_bytes=8 * gib, reserved_bytes=8 * gib
        )
        is False
    )
    assert calls == []
    assert reclaim_checkpoint(device, "lazy", kind="explicit") is True
    assert calls == [1]
    calls.clear()
    assert reclaim_checkpoint(device, "lazy", kind="oom") is True
    assert reclaim_checkpoint(device, "eager", kind="persist") is True
    assert reclaim_checkpoint(device, "eager", kind="stage") is True
    calls.clear()
    assert (
        reclaim_checkpoint(
            device,
            "adaptive",
            kind="persist",
            total_bytes=40 * gib,
            reserved_bytes=39 * gib,
        )
        is False
    )
    assert (
        reclaim_checkpoint(
            device,
            "adaptive",
            kind="stage",
            total_bytes=12 * gib,
            reserved_bytes=1,
        )
        is True
    )
    assert (
        reclaim_checkpoint(
            device,
            "adaptive",
            kind="persist",
            total_bytes=20 * gib,
            reserved_bytes=int(0.9 * 20 * gib),
        )
        is True
    )
    calls.clear()
    assert (
        reclaim_checkpoint(
            device,
            "adaptive",
            kind="stage",
            total_bytes=20 * gib,
            reserved_bytes=int(0.5 * 20 * gib),
        )
        is False
    )
    assert calls == []


def test_stack_dask_persist_stage_reclaim_uses_client_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dask GPU workers receive reclaim_checkpoint; tiles do not empty_cache."""
    import inspect

    from faninsar.backends import dask_gpu
    from faninsar.backends.dask_gpu import _reclaim_checkpoint_on_worker

    remote_calls: list[tuple[object, tuple[object, ...]]] = []

    class FakeClient:
        def run(
            self,
            func: object,
            *args: object,
            **_kwargs: object,
        ) -> dict[str, bool]:
            remote_calls.append((func, args))
            return {"gpu-worker": func(*args)}

    paths = []
    for day in ("20160101", "20160113"):
        path = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000_{day}T000001.SAFE"
        path.mkdir()
        paths.append(path)
    stack = Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        activation_mode="reference",
        gpu_memory_reclaim="eager",
        dask_client=FakeClient(),
    )
    monkeypatch.setattr(
        "faninsar._core.device.reclaim_checkpoint",
        lambda *_args, **kwargs: kwargs.get("kind"),
    )
    stack._reclaim_accelerator("persist")
    stack._reclaim_accelerator("stage")
    assert len(remote_calls) == 2
    assert remote_calls[0][0] is _reclaim_checkpoint_on_worker
    assert remote_calls[0][1][2] == "persist"
    assert remote_calls[1][1][2] == "stage"
    module_text = Path(dask_gpu.__file__).read_text(encoding="utf-8")
    assert "empty_cache" not in module_text
    schedule = inspect.getsource(dask_gpu._schedule_carrier_multiply)
    assert "empty_cache" not in schedule
    assert "reclaim_checkpoint" not in schedule
