"""Product kernel: one directory of payloads plus one ``item.json`` (P0059)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from faninsar.io.storage.product import (
    Product,
    commit_product,
    discover_products,
    open_product,
    verify_product,
)
from faninsar.processing.errors import InvalidProcessingStateError


def _write_arrays(arrays: dict[str, np.ndarray]):
    """Return a payload writer that writes ``arrays`` as Zarr assets."""
    from faninsar.io.storage._zarr_stac import write_zarr_array

    def writer(staging: Path) -> dict[str, dict[str, object]]:
        assets: dict[str, dict[str, object]] = {}
        for name, value in arrays.items():
            write_zarr_array(staging / f"{name}.zarr", value)
            assets[name] = {"href": f"{name}.zarr", "faninsar:dtype": str(value.dtype)}
        return assets

    return writer


def test_commit_and_open_roundtrip(tmp_path: Path) -> None:
    """A committed product exposes its properties, assets, and a write id."""
    target = tmp_path / "coreg" / "20240101"
    payload = np.arange(6, dtype=np.float32).reshape(2, 3)
    product = commit_product(
        target,
        kind="scene",
        properties={"faninsar:date": "20240101"},
        write_payloads=_write_arrays({"a": payload}),
    )
    assert isinstance(product, Product)
    assert sorted(path.name for path in target.iterdir()) == ["a.zarr", "item.json"]
    reopened = open_product(target)
    assert reopened.kind == "scene"
    assert reopened.write_id == product.write_id
    assert reopened.properties["faninsar:date"] == "20240101"
    np.testing.assert_array_equal(reopened.read("a"), payload)
    assert len(product.write_id) == 26


def test_every_commit_gets_a_fresh_write_id(tmp_path: Path) -> None:
    """Overwriting with identical bytes still mints a new write id."""
    target = tmp_path / "p"
    writer = _write_arrays({"a": np.zeros((2, 2), dtype=np.float32)})
    first = commit_product(target, kind="x", properties={}, write_payloads=writer)
    with pytest.raises(InvalidProcessingStateError, match="exists"):
        commit_product(target, kind="x", properties={}, write_payloads=writer)
    second = commit_product(
        target, kind="x", properties={}, write_payloads=writer, overwrite=True
    )
    assert second.write_id != first.write_id
    assert open_product(target).write_id == second.write_id


def test_crash_before_item_leaves_no_product(tmp_path: Path) -> None:
    """A writer failure leaves neither a product nor a visible staging tree."""
    target = tmp_path / "p"

    def broken(staging: Path) -> dict[str, dict[str, object]]:
        (staging / "half.zarr").mkdir()
        raise RuntimeError("crash mid-payload")

    with pytest.raises(RuntimeError, match="crash"):
        commit_product(target, kind="x", properties={}, write_payloads=broken)
    assert not target.exists()
    assert discover_products(tmp_path, kind="x") == []
    # The next writer succeeds and removes the stale staging sibling.
    commit_product(
        target,
        kind="x",
        properties={},
        write_payloads=_write_arrays({"a": np.ones((1, 1), dtype=np.float32)}),
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["p"]


def test_crash_after_aside_keeps_old_product_discoverable(tmp_path: Path) -> None:
    """A crash between the aside rename and rename-in loses only the new data."""
    target = tmp_path / "p"
    writer = _write_arrays({"a": np.zeros((1, 1), dtype=np.float32)})
    first = commit_product(target, kind="x", properties={}, write_payloads=writer)
    aside = tmp_path / f".p.aside-{first.write_id}"
    os.rename(target, aside)
    assert discover_products(tmp_path, kind="x") == []
    with pytest.raises(InvalidProcessingStateError, match="missing"):
        open_product(target)
    commit_product(target, kind="x", properties={}, write_payloads=writer)
    assert not aside.exists()


def test_directory_without_item_is_not_a_product(tmp_path: Path) -> None:
    """Only ``item.json`` marks completion; partial directories are rejected."""
    (tmp_path / "p" / "a.zarr").mkdir(parents=True)
    with pytest.raises(InvalidProcessingStateError, match="missing"):
        open_product(tmp_path / "p")
    assert discover_products(tmp_path, kind="x") == []


@pytest.mark.parametrize("href", ["/abs/a.zarr", "../a.zarr", "sub/a.zarr", ".."])
def test_asset_hrefs_must_be_plain_basenames(tmp_path: Path, href: str) -> None:
    """Absolute, parent, and nested hrefs are rejected at write time."""

    def writer(staging: Path) -> dict[str, dict[str, object]]:
        return {"a": {"href": href}}

    with pytest.raises(InvalidProcessingStateError, match="basename"):
        commit_product(tmp_path / "p", kind="x", properties={}, write_payloads=writer)


def test_symlinked_asset_is_rejected_on_open(tmp_path: Path) -> None:
    """A symlinked asset inside a committed product fails to open."""
    target = tmp_path / "p"
    commit_product(
        target,
        kind="x",
        properties={},
        write_payloads=_write_arrays({"a": np.ones((1, 1), dtype=np.float32)}),
    )
    outside = tmp_path / "outside.zarr"
    os.rename(target / "a.zarr", outside)
    os.symlink(outside, target / "a.zarr")
    with pytest.raises(InvalidProcessingStateError, match="symbolic link"):
        open_product(target)


def test_cleanup_never_follows_a_symlinked_staging_entry(tmp_path: Path) -> None:
    """A symlink planted at a staging name is unlinked, never recursed."""
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("x", encoding="utf-8")
    os.symlink(victim, tmp_path / ".p.tmp")
    commit_product(
        tmp_path / "p",
        kind="x",
        properties={},
        write_payloads=_write_arrays({"a": np.ones((1, 1), dtype=np.float32)}),
    )
    assert (victim / "keep.txt").is_file()


def test_metadata_must_be_strict_json(tmp_path: Path) -> None:
    """NaN in metadata fails at write, before anything is published."""
    with pytest.raises(InvalidProcessingStateError, match="JSON"):
        commit_product(
            tmp_path / "p",
            kind="x",
            properties={"bad": float("nan")},
            write_payloads=_write_arrays({}),
        )
    assert not (tmp_path / "p").exists()


def test_inputs_record_upstream_write_ids(tmp_path: Path) -> None:
    """Dependents record upstream write ids and detect a replaced upstream."""
    writer = _write_arrays({"a": np.ones((1, 1), dtype=np.float32)})
    upstream = commit_product(
        tmp_path / "up", kind="u", properties={}, write_payloads=writer
    )
    down = commit_product(
        tmp_path / "down",
        kind="d",
        properties={},
        inputs={"reference": upstream},
        write_payloads=writer,
    )
    assert open_product(tmp_path / "down").inputs == {"reference": upstream.write_id}
    assert down.stale_inputs({"reference": upstream}) == []
    replaced = commit_product(
        tmp_path / "up", kind="u", properties={}, write_payloads=writer, overwrite=True
    )
    assert down.stale_inputs({"reference": replaced}) == ["reference"]


def test_verify_detects_modified_payload(tmp_path: Path) -> None:
    """``verify_product`` re-hashes payloads against write-time checksums."""
    target = tmp_path / "p"
    commit_product(
        target,
        kind="x",
        properties={},
        write_payloads=_write_arrays({"a": np.ones((4, 4), dtype=np.float32)}),
    )
    assert verify_product(target) == []
    chunk = next(path for path in (target / "a.zarr").rglob("*") if path.is_file()
                 and path.name not in {"zarr.json", ".zarray", ".zattrs"})
    chunk.write_bytes(chunk.read_bytes() + b"x")
    assert verify_product(target) == ["a"]


def test_item_is_stac_feature(tmp_path: Path) -> None:
    """``item.json`` is a STAC Feature with namespaced custom fields."""
    target = tmp_path / "p"
    commit_product(
        target,
        kind="scene",
        properties={"faninsar:date": "20240101"},
        write_payloads=_write_arrays({"a": np.ones((1, 1), dtype=np.float32)}),
    )
    item = json.loads((target / "item.json").read_text(encoding="utf-8"))
    assert item["type"] == "Feature"
    assert item["stac_version"] == "1.0.0"
    assert item["properties"]["faninsar:kind"] == "scene"
    assert item["assets"]["a"]["href"] == "a.zarr"
    assert "file:checksum" in item["assets"]["a"]


def test_discover_lists_committed_products_by_kind(tmp_path: Path) -> None:
    """Discovery returns committed children of one kind, sorted by name."""
    writer = _write_arrays({"a": np.ones((1, 1), dtype=np.float32)})
    for name in ("b", "a"):
        commit_product(tmp_path / name, kind="scene", properties={}, write_payloads=writer)
    commit_product(tmp_path / "c", kind="other", properties={}, write_payloads=writer)
    found = discover_products(tmp_path, kind="scene")
    assert [product.path.name for product in found] == ["a", "b"]
