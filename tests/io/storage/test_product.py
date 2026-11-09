"""Product kernel: one directory of payloads plus one ``item.json`` (P0059)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from faninsar.io.storage.product import (
    ITEM_NAME,
    Product,
    ProductWriter,
    discover_products,
    open_product,
    publish_product,
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


def _omit_asset(root: Path, name: str) -> None:
    """Drop one asset from ``item.json`` and delete its payload."""
    item_path = root / ITEM_NAME
    item = json.loads(item_path.read_text(encoding="utf-8"))
    href = item["assets"].pop(name)["href"]
    item_path.write_text(
        json.dumps(item, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    payload = root / href
    if payload.is_dir():
        shutil.rmtree(payload)
    elif payload.exists():
        payload.unlink()


def test_commit_and_open_roundtrip(tmp_path: Path) -> None:
    """A committed product exposes its properties, assets, and a write id."""
    target = tmp_path / "coreg" / "20240101"
    payload = np.arange(6, dtype=np.float32).reshape(2, 3)
    product = publish_product(
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
    first = publish_product(target, kind="x", properties={}, write_payloads=writer)
    with pytest.raises(InvalidProcessingStateError, match="exists"):
        publish_product(target, kind="x", properties={}, write_payloads=writer)
    second = publish_product(
        target, kind="x", properties={}, write_payloads=writer, overwrite=True
    )
    third = publish_product(
        target, kind="x", properties={}, write_payloads=writer, overwrite=True
    )
    assert len({first.write_id, second.write_id, third.write_id}) == 3
    assert open_product(target).write_id == third.write_id


def test_crash_before_item_leaves_no_product(tmp_path: Path) -> None:
    """A writer failure leaves neither a product nor a visible staging tree."""
    target = tmp_path / "p"

    def broken(staging: Path) -> dict[str, dict[str, object]]:
        (staging / "half.zarr").mkdir()
        raise RuntimeError("crash mid-payload")

    with pytest.raises(RuntimeError, match="crash"):
        publish_product(target, kind="x", properties={}, write_payloads=broken)
    assert not target.exists()
    assert discover_products(tmp_path, kind="x") == []
    # The next writer succeeds and removes the stale staging sibling.
    publish_product(
        target,
        kind="x",
        properties={},
        write_payloads=_write_arrays({"a": np.ones((1, 1), dtype=np.float32)}),
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["p"]


def test_crash_after_aside_keeps_old_product_discoverable(tmp_path: Path) -> None:
    """A leftover aside is restored; the next writer does not delete the only copy."""
    target = tmp_path / "p"
    writer = _write_arrays({"a": np.zeros((1, 1), dtype=np.float32)})
    first = publish_product(target, kind="x", properties={}, write_payloads=writer)
    aside = tmp_path / f".p.aside-{first.write_id}"
    os.rename(target, aside)
    assert discover_products(tmp_path, kind="x") == []
    with pytest.raises(InvalidProcessingStateError, match="exists"):
        publish_product(target, kind="x", properties={}, write_payloads=writer)
    restored = open_product(target)
    assert restored.write_id == first.write_id
    np.testing.assert_array_equal(
        restored.read("a"), np.zeros((1, 1), dtype=np.float32)
    )
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
        publish_product(tmp_path / "p", kind="x", properties={}, write_payloads=writer)


def test_symlinked_asset_is_rejected_on_open(tmp_path: Path) -> None:
    """A symlinked asset inside a committed product fails to open."""
    target = tmp_path / "p"
    publish_product(
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
    publish_product(
        tmp_path / "p",
        kind="x",
        properties={},
        write_payloads=_write_arrays({"a": np.ones((1, 1), dtype=np.float32)}),
    )
    assert (victim / "keep.txt").is_file()


def test_metadata_must_be_strict_json(tmp_path: Path) -> None:
    """NaN in metadata fails at write, before anything is published."""
    with pytest.raises(InvalidProcessingStateError, match="JSON"):
        publish_product(
            tmp_path / "p",
            kind="x",
            properties={"bad": float("nan")},
            write_payloads=_write_arrays({}),
        )
    assert not (tmp_path / "p").exists()


def test_inputs_record_upstream_write_ids(tmp_path: Path) -> None:
    """Dependents record upstream write ids and detect a replaced upstream."""
    writer = _write_arrays({"a": np.ones((1, 1), dtype=np.float32)})
    upstream = publish_product(
        tmp_path / "up", kind="u", properties={}, write_payloads=writer
    )
    down = publish_product(
        tmp_path / "down",
        kind="d",
        properties={},
        inputs={"reference": upstream},
        write_payloads=writer,
    )
    assert open_product(tmp_path / "down").inputs == {"reference": upstream.write_id}
    assert down.stale_inputs({"reference": upstream}) == []
    replaced = publish_product(
        tmp_path / "up", kind="u", properties={}, write_payloads=writer, overwrite=True
    )
    assert down.stale_inputs({"reference": replaced}) == ["reference"]


def test_new_products_omit_file_checksum(tmp_path: Path) -> None:
    """New commits keep write ids and omit a self-computed payload checksum."""
    target = tmp_path / "p"
    product = publish_product(
        target,
        kind="x",
        properties={},
        write_payloads=_write_arrays({"a": np.ones((1, 1), dtype=np.float32)}),
    )
    item = json.loads((target / ITEM_NAME).read_text(encoding="utf-8"))
    assert "file:checksum" not in item["assets"]["a"]
    assert product.write_id == item["properties"]["faninsar:write_id"]
    assert open_product(target).write_id == product.write_id


def test_verify_reports_missing_and_malformed_assets(tmp_path: Path) -> None:
    """Structural verify names missing and unreadable required assets."""
    target = tmp_path / "p"
    publish_product(
        target,
        kind="x",
        properties={},
        write_payloads=_write_arrays(
            {
                "a": np.ones((2, 2), dtype=np.float32),
                "b": np.zeros((2, 2), dtype=np.float32),
            }
        ),
    )
    assert verify_product(target) == []
    shutil.rmtree(target / "a.zarr")
    (target / "b.zarr" / "zarr.json").write_bytes(b"not zarr metadata")
    assert verify_product(target) == ["a", "b"]


def test_verify_reports_required_mask_asset_omitted_from_manifest(
    tmp_path: Path,
) -> None:
    """A mask product must name ``mask`` when that required asset is dropped."""
    target = tmp_path / "masks" / "water" / "geo"
    publish_product(
        target,
        kind="mask",
        properties={},
        write_payloads=_write_arrays({"mask": np.ones((2, 2), dtype=np.uint8)}),
    )
    assert verify_product(target) == []
    _omit_asset(target, "mask")
    assert verify_product(target) == ["mask"]


def test_verify_reports_omitted_geometry_asset_when_set_is_declared(
    tmp_path: Path,
) -> None:
    """Once a scene declares the geometry set, an omitted field is reported."""
    from faninsar.io.storage.scene_geometry import SCENE_GEOMETRY_ASSETS

    arrays = {
        name: np.zeros((2, 2), dtype=np.float32) for name in SCENE_GEOMETRY_ASSETS
    }
    arrays["geom_flags"] = np.zeros((2, 2), dtype=np.uint8)
    target = tmp_path / "coreg" / "20240101"
    publish_product(
        target,
        kind="scene",
        properties={},
        write_payloads=_write_arrays(arrays),
    )
    assert verify_product(target) == []
    _omit_asset(target, "incidence")
    assert verify_product(target) == ["incidence"]


def test_verify_allows_scene_without_geometry_assets(tmp_path: Path) -> None:
    """Geometry is lazy; a scene that never published it stays structurally healthy."""
    target = tmp_path / "coreg" / "20240101"
    publish_product(
        target,
        kind="scene",
        properties={},
        write_payloads=_write_arrays({"slc": np.ones((2, 2), dtype=np.complex64)}),
    )
    assert verify_product(target) == []


def test_verify_reports_declared_shape_and_dtype_mismatch(tmp_path: Path) -> None:
    """Declared array metadata is checked without requiring a payload hash."""
    from faninsar.io.storage._zarr_stac import write_zarr_array

    target = tmp_path / "p"
    publish_product(
        target,
        kind="x",
        properties={"faninsar:shape": [2, 2]},
        write_payloads=_write_arrays({"a": np.ones((2, 2), dtype=np.float32)}),
    )
    assert verify_product(target) == []
    write_zarr_array(target / "a.zarr", np.ones((3, 1), dtype=np.float64))
    assert verify_product(target) == ["a"]


def test_verify_ignores_legacy_checksum_and_payload_bytes(tmp_path: Path) -> None:
    """Legacy ``file:checksum`` is inert; same-size byte edits are not scanned."""
    target = tmp_path / "p"
    publish_product(
        target,
        kind="x",
        properties={},
        write_payloads=_write_arrays({"a": np.ones((4, 4), dtype=np.float32)}),
    )
    item_path = target / ITEM_NAME
    item = json.loads(item_path.read_text(encoding="utf-8"))
    item["assets"]["a"]["file:checksum"] = "0" * 64
    item_path.write_text(
        json.dumps(item, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    chunk = next(
        path
        for path in (target / "a.zarr").rglob("*")
        if path.is_file() and path.name not in {"zarr.json", ".zarray", ".zattrs"}
    )
    payload = chunk.read_bytes()
    chunk.write_bytes(bytes((payload[0] ^ 0xFF,)) + payload[1:])
    assert verify_product(target) == []


def test_failed_overwrite_rename_restores_the_old_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed staging rename after aside must put the last usable product back."""
    from faninsar.io.storage._zarr_stac import write_zarr_array

    target = tmp_path / "p"
    first = publish_product(
        target,
        kind="x",
        properties={},
        write_payloads=_write_arrays({"a": np.zeros((1, 1), dtype=np.float32)}),
    )
    original = Path.rename
    rename_error = "simulated staging rename failure"

    def flaky_rename(self: Path, dst: str | Path) -> Path:
        destination = Path(dst)
        if self.name == f".{target.name}.tmp" and destination == target:
            raise OSError(rename_error)
        return original(self, destination)

    monkeypatch.setattr(Path, "rename", flaky_rename)

    def overwrite() -> None:
        with ProductWriter(target, kind="x", overwrite=True) as writer:
            write_zarr_array(
                writer.add_asset("a", "a.zarr"), np.ones((1, 1), dtype=np.float32)
            )
            writer.commit({})

    with pytest.raises(OSError, match="staging rename"):
        overwrite()
    restored = open_product(target)
    assert restored.write_id == first.write_id
    np.testing.assert_array_equal(
        restored.read("a"), np.zeros((1, 1), dtype=np.float32)
    )
    assert list(tmp_path.glob(".p.aside-*")) == []


def test_item_is_stac_feature(tmp_path: Path) -> None:
    """``item.json`` is a STAC Feature with namespaced custom fields."""
    target = tmp_path / "p"
    publish_product(
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
    assert "file:checksum" not in item["assets"]["a"]


def test_discover_lists_committed_products_by_kind(tmp_path: Path) -> None:
    """Discovery returns committed children of one kind, sorted by name."""
    writer = _write_arrays({"a": np.ones((1, 1), dtype=np.float32)})
    for name in ("b", "a"):
        publish_product(
            tmp_path / name, kind="scene", properties={}, write_payloads=writer
        )
    publish_product(tmp_path / "c", kind="other", properties={}, write_payloads=writer)
    found = discover_products(tmp_path, kind="scene")
    assert [product.path.name for product in found] == ["a", "b"]
