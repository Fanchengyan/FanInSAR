"""Mission-unit checkpoints on the product publication kernel (PROPOSAL-0088)."""

# ruff: noqa: INP001

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.io.storage.scene_store import SceneWriter
from faninsar.io.storage.unit_checkpoint import (
    UnitCheckpointStore,
    UnitPayloads,
)
from faninsar.processing.errors import InvalidProcessingStateError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _specification(**overrides: object) -> dict[str, object]:
    """Return one small per-unit compatibility specification."""
    specification: dict[str, object] = {
        "domain": "radar",
        "date_id": "20240113",
        "burst_id": "IW1_b0",
        "reference_unit_id": "20240101-IW1_b0",
        "reference_write_id": "reference-unit-write-id",
        "recipe": "coregistration-v1",
        "support": {
            "row_origin": 0,
            "col_origin": 0,
            "rows": 3,
            "cols": 4,
        },
    }
    specification.update(overrides)
    return specification


def _payloads(
    value: float = 1.0,
    *,
    row_origin: int = 0,
    col_origin: int = 0,
) -> UnitPayloads:
    """Return one tiny real scene-unit payload set."""
    rows, cols = 3, 4
    slc = (
        np.arange(rows * cols, dtype=np.float32).reshape(rows, cols) + 1j * value
    ).astype(np.complex64)
    phase = np.linspace(0.0, 0.5, rows * cols, dtype=np.float32).reshape(rows, cols)
    valid = np.array(
        [
            [True, True, False, True],
            [True, False, True, True],
            [True, True, True, False],
        ]
    )
    return UnitPayloads(
        slc=slc,
        row_origin=row_origin,
        col_origin=col_origin,
        phase=phase,
        valid=valid,
        burst_id="IW1_b0",
        absolute_burst_id=12,
        scientific_lineage=({"faninsar:input": "s1-slc-source-1"},),
        phase_state={"flatten_stage": "coregistration", "recipe": "coreg-v1"},
    )


def test_commit_and_load_return_one_validated_unit(tmp_path: Path) -> None:
    """A committed unit is discoverable from its receipt with equal payloads."""
    store = UnitCheckpointStore(tmp_path / "units")
    specification = _specification()
    payloads = _payloads()

    committed = store.commit("20240113-IW1_b0", specification, payloads=payloads)

    assert committed.unit_id == "20240113-IW1_b0"
    assert committed.rows == 3
    assert committed.cols == 4
    assert committed.burst_id == "IW1_b0"
    assert committed.absolute_burst_id == 12
    assert committed.phase_path is not None
    assert committed.valid_path is not None
    assert len(committed.write_id) == 26
    assert len(committed.specification_id) == 64

    reopened = UnitCheckpointStore(tmp_path / "units").load(
        "20240113-IW1_b0", dict(specification)
    )

    assert reopened is not None
    assert reopened.write_id == committed.write_id
    assert reopened.specification == specification
    assert reopened.specification_id == committed.specification_id
    assert reopened.scientific_lineage == ({"faninsar:input": "s1-slc-source-1"},)
    assert reopened.phase_state == {
        "flatten_stage": "coregistration",
        "recipe": "coreg-v1",
    }
    np.testing.assert_array_equal(reopened.read_slc(), payloads.slc)
    np.testing.assert_array_equal(reopened.read_phase(), payloads.phase)
    np.testing.assert_array_equal(reopened.read_valid(), payloads.valid)


def test_interrupted_later_unit_is_absent_and_retryable(tmp_path: Path) -> None:
    """A crash before publication leaves the first unit and no second receipt."""
    store = UnitCheckpointStore(tmp_path / "units")
    first = store.commit("20240113-IW1_b0", _specification(), payloads=_payloads())

    # The second unit never reaches publication: only its staging left behind.
    staging = tmp_path / "units" / ".20240113-IW1_b1.tmp"
    staging.mkdir()
    (staging / "half.zarr").mkdir()

    assert store.load("20240113-IW1_b0", _specification()) is not None
    assert store.load("20240113-IW1_b1", _specification()) is None

    second = store.commit(
        "20240113-IW1_b1",
        _specification(burst_id="IW1_b1"),
        payloads=_payloads(2.0),
    )

    assert first.write_id != second.write_id
    assert store.load("20240113-IW1_b1", _specification(burst_id="IW1_b1"))
    assert not staging.exists()


def _rewrite_item(target: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    """Rewrite one committed item through a small mutation."""
    item_path = target / "item.json"
    item = json.loads(item_path.read_text(encoding="utf-8"))
    mutate(item)
    item_path.write_text(json.dumps(item, sort_keys=True) + "\n", encoding="utf-8")


def _first_chunk(target: Path, payload: str) -> Path:
    """Return one stored chunk file of the named committed payload."""
    return next(
        path
        for path in (target / payload).rglob("*")
        if path.is_file() and path.name != "zarr.json"
    )


def _first_slc_chunk(target: Path) -> Path:
    """Return one stored chunk file of the committed slc payload."""
    return _first_chunk(target, "slc.zarr")


def _chunk_files(target: Path, payload: str) -> list[Path]:
    """Return every stored chunk file of the named committed payload."""
    return [
        path
        for path in (target / payload).rglob("*")
        if path.is_file() and path.name != "zarr.json"
    ]


def _sparse_payloads(rows: int = 361, cols: int = 905) -> UnitPayloads:
    """Return one real Radar-sized unit whose samples are all fill values."""
    return UnitPayloads(
        slc=np.zeros((rows, cols), dtype=np.complex64),
        row_origin=0,
        col_origin=0,
        phase=np.zeros((rows, cols), dtype=np.float32),
        valid=np.zeros((rows, cols), dtype=bool),
    )


def test_stale_specification_is_not_reused_and_siblings_are_unaffected(
    tmp_path: Path,
) -> None:
    """Only an equal specification reuses a unit; replacement never touches peers."""
    store = UnitCheckpointStore(tmp_path / "units")
    first = store.commit("unit-a", _specification(), payloads=_payloads(1.0))
    second = store.commit(
        "unit-b", _specification(burst_id="IW1_b1"), payloads=_payloads(2.0)
    )

    assert store.load("unit-a", _specification(recipe="coregistration-v2")) is None

    replaced = store.commit("unit-a", _specification(), payloads=_payloads(3.0))
    reopened = store.load("unit-a", _specification())
    sibling = store.load("unit-b", _specification(burst_id="IW1_b1"))

    assert reopened is not None
    assert reopened.write_id == replaced.write_id != first.write_id
    assert sibling is not None
    assert sibling.write_id == second.write_id
    assert sibling.specification_id == second.specification_id
    np.testing.assert_array_equal(sibling.read_slc(), _payloads(2.0).slc)


def test_commit_refuses_a_foreign_kind_at_the_unit_path(tmp_path: Path) -> None:
    """A legacy product at the unit path is preserved, never relabelled."""
    store = UnitCheckpointStore(tmp_path / "units")
    foreign = store.root / "unit"
    foreign.mkdir()
    item_path = foreign / "item.json"
    item_path.write_text(
        json.dumps(
            {
                "type": "Feature",
                "properties": {
                    "faninsar:kind": "scene",
                    "faninsar:write_id": "legacy-scene-write-id",
                },
                "assets": {},
            }
        ),
        encoding="utf-8",
    )
    original = item_path.read_bytes()

    with pytest.raises(InvalidProcessingStateError, match="existing product kind"):
        store.commit("unit", _specification(), payloads=_payloads())

    assert item_path.read_bytes() == original
    assert sorted(path.name for path in foreign.iterdir()) == ["item.json"]
    with pytest.raises(InvalidProcessingStateError, match="kind"):
        store.load("unit", _specification())


def test_case_differing_unit_ids_conflict_without_overwrite(tmp_path: Path) -> None:
    """Case aliases share one directory on case-insensitive filesystems."""
    store = UnitCheckpointStore(tmp_path / "units")
    probe = store.root / "CaseProbe"
    probe.mkdir()
    try:
        case_insensitive = (store.root / "caseprobe").exists()
    finally:
        probe.rmdir()
    if not case_insensitive:
        pytest.skip("requires a case-insensitive filesystem")

    first = store.commit("Unit-A", _specification(), payloads=_payloads(1.0))
    with pytest.raises(InvalidProcessingStateError, match="differs from requested"):
        store.commit("unit-a", _specification(), payloads=_payloads(9.0))

    assert sorted(path.name for path in store.root.iterdir()) == ["Unit-A"]
    reopened = store.load("Unit-A", _specification())
    assert reopened is not None
    assert reopened.write_id == first.write_id
    np.testing.assert_array_equal(reopened.read_slc(), _payloads(1.0).slc)


def test_missing_or_unpublished_receipt_is_none(tmp_path: Path) -> None:
    """Absent and never-committed units are retryable, not errors."""
    store = UnitCheckpointStore(tmp_path / "units")
    (store.root / "unpublished").mkdir()

    assert store.load("absent", _specification()) is None
    assert store.load("unpublished", _specification()) is None


def test_committed_but_missing_asset_rejects(tmp_path: Path) -> None:
    """A committed item whose payload asset disappeared is corrupt."""
    store = UnitCheckpointStore(tmp_path / "units")
    record = store.commit("unit", _specification(), payloads=_payloads())
    assert record.phase_path is not None
    shutil.rmtree(record.phase_path)

    with pytest.raises(InvalidProcessingStateError, match="missing"):
        store.load("unit", _specification())


def test_committed_missing_chunk_rejects(tmp_path: Path) -> None:
    """A committed Zarr payload with a deleted chunk is truncated."""
    store = UnitCheckpointStore(tmp_path / "units")
    record = store.commit("unit", _specification(), payloads=_payloads())
    _first_slc_chunk(record.path).unlink()

    with pytest.raises(InvalidProcessingStateError, match="missing chunk"):
        store.load("unit", _specification())


def test_committed_truncated_chunk_rejects(tmp_path: Path) -> None:
    """A committed Zarr payload with an empty chunk is truncated."""
    store = UnitCheckpointStore(tmp_path / "units")
    record = store.commit("unit", _specification(), payloads=_payloads())
    _first_slc_chunk(record.path).write_bytes(b"")

    with pytest.raises(InvalidProcessingStateError, match="truncated chunk"):
        store.load("unit", _specification())


def test_sparse_fill_payloads_materialize_every_declared_chunk(
    tmp_path: Path,
) -> None:
    """All-fill Radar tiles commit: every declared chunk exists, none is absent."""
    store = UnitCheckpointStore(tmp_path / "units")
    payloads = _sparse_payloads()
    record = store.commit("20240113-IW1_b0", _specification(), payloads=payloads)

    for payload in ("slc.zarr", "phase.zarr", "valid.zarr"):
        chunks = _chunk_files(record.path, payload)
        assert len(chunks) == 8
        assert all(chunk.stat().st_size > 0 for chunk in chunks)

    reopened = store.load("20240113-IW1_b0", _specification())
    assert reopened is not None
    np.testing.assert_array_equal(reopened.read_slc(), payloads.slc)
    np.testing.assert_array_equal(reopened.read_phase(), payloads.phase)
    np.testing.assert_array_equal(reopened.read_valid(), payloads.valid)


def test_removed_fill_chunk_still_rejects(tmp_path: Path) -> None:
    """An absent chunk stays corruption even when its samples were all fill."""
    store = UnitCheckpointStore(tmp_path / "units")
    record = store.commit("unit", _specification(), payloads=_sparse_payloads())
    (record.path / "valid.zarr" / "c" / "0" / "3").unlink()

    with pytest.raises(InvalidProcessingStateError, match="missing chunk"):
        store.load("unit", _specification())


def test_missing_nonzero_chunk_on_large_grid_rejects(tmp_path: Path) -> None:
    """A removed chunk carrying samples is corruption on a real-sized grid."""
    store = UnitCheckpointStore(tmp_path / "units")
    rows, cols = 361, 905
    slc = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    payloads = UnitPayloads(slc=slc.astype(np.complex64), row_origin=0, col_origin=0)
    record = store.commit("large", _specification(), payloads=payloads)
    (record.path / "slc.zarr" / "c" / "1" / "2").unlink()

    with pytest.raises(InvalidProcessingStateError, match="missing chunk"):
        store.load("large", _specification())


@pytest.mark.parametrize(
    ("payload", "reader_name"),
    [
        ("slc.zarr", "read_slc"),
        ("phase.zarr", "read_phase"),
        ("valid.zarr", "read_valid"),
    ],
)
def test_corrupt_compressed_payload_reads_as_contract_error(
    tmp_path: Path, payload: str, reader_name: str
) -> None:
    """A non-empty corrupt chunk loads structurally but reads as typed state."""
    store = UnitCheckpointStore(tmp_path / "units")
    record = store.commit("unit", _specification(), payloads=_payloads())
    _first_chunk(record.path, payload).write_bytes(bytes(range(32)))

    reopened = store.load("unit", _specification())
    assert reopened is not None
    with pytest.raises(InvalidProcessingStateError, match="cannot be decoded"):
        getattr(reopened, reader_name)()


def test_corrupt_payload_fails_add_to_as_contract_error(tmp_path: Path) -> None:
    """Scene replay reports corrupt stored payloads instead of a raw crash."""
    store = UnitCheckpointStore(tmp_path / "units")
    record = store.commit("unit", _specification(), payloads=_payloads())
    _first_slc_chunk(record.path).write_bytes(bytes(range(32)))
    reopened = store.load("unit", _specification())
    assert reopened is not None

    with (
        SceneWriter(
            tmp_path / "coreg" / "20240113",
            date_id="20240113",
            reference_id="20240101",
            domain="radar",
            grid_shape=(6, 6),
        ) as writer,
        pytest.raises(InvalidProcessingStateError, match="cannot be decoded"),
    ):
        reopened.add_to(writer)


def test_corrupt_item_metadata_rejects(tmp_path: Path) -> None:
    """Unsupported schema, wrong locator, and unreadable items are rejected."""
    store = UnitCheckpointStore(tmp_path / "units")

    store.commit("bad-schema", _specification(), payloads=_payloads())
    _rewrite_item(
        store.root / "bad-schema",
        lambda item: item["properties"].update({"faninsar:unit_checkpoint_schema": 99}),
    )
    with pytest.raises(InvalidProcessingStateError, match="schema"):
        store.load("bad-schema", _specification())

    store.commit("bad-id", _specification(), payloads=_payloads())
    _rewrite_item(
        store.root / "bad-id",
        lambda item: item["properties"].update({"faninsar:unit_id": "other"}),
    )
    with pytest.raises(InvalidProcessingStateError, match="locator"):
        store.load("bad-id", _specification())

    store.commit("bad-json", _specification(), payloads=_payloads())
    (store.root / "bad-json" / "item.json").write_text("{", encoding="utf-8")
    with pytest.raises(InvalidProcessingStateError, match=r"item\.json"):
        store.load("bad-json", _specification())


@pytest.mark.parametrize(
    "unit_id",
    ["../escape", "/absolute", "nested/unit", ".hidden", ""],
)
def test_unsafe_unit_ids_are_rejected(tmp_path: Path, unit_id: str) -> None:
    """Unit ids cannot escape or hide inside the checkpoint root."""
    store = UnitCheckpointStore(tmp_path / "units")

    with pytest.raises(InvalidProcessingStateError):
        store.load(unit_id, _specification())
    with pytest.raises(InvalidProcessingStateError):
        store.commit(unit_id, _specification(), payloads=_payloads())


def test_symlinked_root_is_rejected(tmp_path: Path) -> None:
    """A symlinked checkpoint root would escape path validation."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(InvalidProcessingStateError, match="symbolic link"):
        UnitCheckpointStore(link)


def test_invalid_payload_never_replaces_the_prior_unit(tmp_path: Path) -> None:
    """Payload contract rejection happens before publication touches the unit."""
    store = UnitCheckpointStore(tmp_path / "units")
    original = store.commit("unit", _specification(), payloads=_payloads(1.0))

    with pytest.raises(InvalidProcessingStateError, match="phase"):
        UnitPayloads(
            slc=np.ones((3, 4), dtype=np.complex64),
            row_origin=0,
            col_origin=0,
            phase=np.ones((2, 2), dtype=np.float32),
        )
    with pytest.raises(InvalidProcessingStateError, match="row_origin"):
        UnitPayloads(
            slc=np.ones((3, 4), dtype=np.complex64),
            row_origin=1.5,  # type: ignore[arg-type]
            col_origin=0,
        )

    reopened = store.load("unit", _specification())
    assert reopened is not None
    assert reopened.write_id == original.write_id
    np.testing.assert_array_equal(reopened.read_slc(), _payloads(1.0).slc)


def test_failed_replacement_keeps_the_prior_good_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after staging the first payload leaves the old publication."""
    import faninsar.io.storage.unit_checkpoint as module

    store = UnitCheckpointStore(tmp_path / "units")
    original = store.commit("unit", _specification(), payloads=_payloads(1.0))
    real_write = module._write_payload
    calls = 0

    def flaky_write(path: Path, value: np.ndarray) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            message = "crash after the first payload"
            raise RuntimeError(message)
        real_write(path, value)

    monkeypatch.setattr(module, "_write_payload", flaky_write)

    with pytest.raises(RuntimeError, match="crash"):
        store.commit("unit", _specification(), payloads=_payloads(9.0))
    monkeypatch.setattr(module, "_write_payload", real_write)

    reopened = store.load("unit", _specification())
    assert reopened is not None
    assert reopened.write_id == original.write_id
    np.testing.assert_array_equal(reopened.read_slc(), _payloads(1.0).slc)
    assert sorted(path.name for path in store.root.iterdir()) == ["unit"]


def test_receipt_identity_is_specification_not_payload_bytes(tmp_path: Path) -> None:
    """New bytes under the same specification keep the specification id."""
    store = UnitCheckpointStore(tmp_path / "units")
    specification = _specification()
    first = store.commit(
        "unit",
        specification,
        payloads=_payloads(1.0),
        properties={"faninsar:mission": "s1"},
    )
    assert store.load("unit", specification).properties == {"faninsar:mission": "s1"}

    second = store.commit("unit", specification, payloads=_payloads(9.0))
    reopened = store.load("unit", specification)

    assert second.specification_id == first.specification_id
    assert second.write_id != first.write_id
    assert reopened is not None
    assert reopened.properties == {}
    np.testing.assert_array_equal(reopened.read_slc(), _payloads(9.0).slc)


def test_checkpoint_replays_into_a_scene_manifest(tmp_path: Path) -> None:
    """A restored unit keeps placement, lineage, phase, validity, and samples."""
    store = UnitCheckpointStore(tmp_path / "units")
    record = store.commit(
        "20240113-IW1_b0",
        _specification(),
        payloads=_payloads(row_origin=2, col_origin=1),
    )

    with SceneWriter(
        tmp_path / "coreg" / "20240113",
        date_id="20240113",
        reference_id="20240101",
        domain="radar",
        grid_shape=(6, 6),
    ) as writer:
        record.add_to(writer, tag="f0_IW1_b0")
        scene = writer.commit()

    unit = scene.unit_map()["f0_IW1_b0"]
    payload, _ = scene.read("f0_IW1_b0")

    assert unit.shape == (3, 4)
    assert (unit.row_origin, unit.col_origin) == (2, 1)
    assert unit.burst_id == "IW1_b0"
    assert unit.absolute_burst_id == 12
    assert unit.scientific_lineage == record.scientific_lineage
    assert unit.phase_state == dict(record.phase_state or {})
    np.testing.assert_array_equal(payload, record.read_slc())
    np.testing.assert_array_equal(
        scene.read_phase_screen("f0_IW1_b0"), record.read_phase()
    )
    np.testing.assert_array_equal(
        scene.read_valid_mask("f0_IW1_b0"), record.read_valid()
    )
