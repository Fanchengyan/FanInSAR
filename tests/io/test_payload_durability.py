"""Payload-before-receipt durability on the product publication kernel (P0088)."""

# ruff: noqa: INP001

from __future__ import annotations

import contextlib
import io
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.io.storage.product import (
    ITEM_NAME,
    ProductWriter,
    open_product,
    publish_product,
)
from faninsar.processing.errors import InvalidProcessingStateError

if TYPE_CHECKING:
    from collections.abc import Callable

_REAL_RENAME = Path.rename


def _write_arrays(
    arrays: dict[str, np.ndarray],
) -> Callable[[Path], dict[str, dict[str, object]]]:
    """Return a payload writer that writes ``arrays`` as Zarr assets."""
    from faninsar.io.storage._zarr_stac import write_zarr_array

    def writer(staging: Path) -> dict[str, dict[str, object]]:
        assets: dict[str, dict[str, object]] = {}
        for name, value in arrays.items():
            write_zarr_array(staging / f"{name}.zarr", value)
            assets[name] = {"href": f"{name}.zarr"}
        return assets

    return writer


@pytest.fixture
def sync_events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Path]]:
    """Record file/directory fsyncs and renames in kernel call order."""
    events: list[tuple[str, Path]] = []
    descriptor_paths: dict[int, Path] = {}
    real_os_open = os.open
    real_fsync = os.fsync
    real_io_open = io.open

    def recording_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        descriptor = real_os_open(path, flags, *args, **kwargs)
        descriptor_paths[descriptor] = Path(path)
        return descriptor

    def recording_io_open(
        file: str | bytes | os.PathLike[str],
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ) -> io.TextIOWrapper:
        stream = real_io_open(file, mode, *args, **kwargs)
        with contextlib.suppress(AttributeError, OSError):
            descriptor_paths[stream.fileno()] = Path(file)
        return stream

    def recording_fsync(descriptor: int) -> None:
        path = descriptor_paths.get(descriptor, Path("?"))
        mode = os.fstat(descriptor).st_mode
        kind = "fsync-dir" if stat.S_ISDIR(mode) else "fsync-file"
        events.append((kind, path))
        real_fsync(descriptor)

    def recording_rename(self: Path, destination: str | Path) -> Path:
        events.append(("rename", Path(destination)))
        return _REAL_RENAME(self, Path(destination))

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(io, "open", recording_io_open)
    monkeypatch.setattr(Path, "rename", recording_rename)
    return events


def test_payloads_are_flushed_before_the_item_receipt(
    tmp_path: Path, sync_events: list[tuple[str, Path]]
) -> None:
    """Files flush first, then directories deepest-first, then ``item.json``."""
    target = tmp_path / "p"
    staging = tmp_path / f".{target.name}.tmp"

    def writer(root: Path) -> dict[str, dict[str, object]]:
        (root / "a.zarr" / "nested").mkdir(parents=True)
        (root / "a.zarr" / "nested" / "chunk.bin").write_bytes(b"payload")
        (root / "a.zarr" / "zarr.json").write_text("{}", encoding="utf-8")
        return {"a": {"href": "a.zarr"}}

    publish_product(target, kind="x", properties={}, write_payloads=writer)

    item_index = sync_events.index(("fsync-file", staging / ITEM_NAME))
    payload_files = [
        (index, path)
        for index, (kind, path) in enumerate(sync_events)
        if kind == "fsync-file" and path != staging / ITEM_NAME
    ]
    first_directory = next(
        index for index, (kind, _path) in enumerate(sync_events) if kind == "fsync-dir"
    )

    assert payload_files
    assert max(index for index, _path in payload_files) < first_directory
    assert max(index for index, _path in payload_files) < item_index

    directory_depths = [
        len(path.parts)
        for index, (kind, path) in enumerate(sync_events)
        if kind == "fsync-dir" and index < item_index
    ]
    assert directory_depths == sorted(directory_depths, reverse=True)
    assert directory_depths[0] > directory_depths[-1]
    assert sync_events[item_index - 1] == ("fsync-dir", staging)
    assert sync_events[item_index + 1] == ("fsync-dir", staging)

    target_rename = sync_events.index(("rename", target))
    assert item_index < target_rename
    assert sync_events.index(("fsync-dir", target.parent)) > target_rename


@pytest.mark.parametrize("failure", ["payload", "directory"])
def test_payload_sync_failure_preserves_the_previous_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A sync failure before completion keeps the old product and its evidence."""
    from faninsar.io.storage._zarr_stac import write_zarr_array

    target = tmp_path / "p"
    staging = tmp_path / f".{target.name}.tmp"
    first = publish_product(
        target,
        kind="x",
        properties={},
        write_payloads=_write_arrays({"a": np.zeros((1, 1), dtype=np.float32)}),
    )

    descriptor_paths: dict[int, Path] = {}
    real_os_open = os.open
    real_fsync = os.fsync
    real_io_open = io.open

    def recording_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        descriptor = real_os_open(path, flags, *args, **kwargs)
        descriptor_paths[descriptor] = Path(path)
        return descriptor

    def recording_io_open(
        file: str | bytes | os.PathLike[str],
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ) -> io.TextIOWrapper:
        stream = real_io_open(file, mode, *args, **kwargs)
        with contextlib.suppress(AttributeError, OSError):
            descriptor_paths[stream.fileno()] = Path(file)
        return stream

    def failing_fsync(descriptor: int) -> None:
        path = descriptor_paths.get(descriptor)
        staged = path is not None and (path == staging or staging in path.parents)
        if staged and path.name != ITEM_NAME:
            is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
            if (failure == "directory") == is_directory:
                message = f"simulated {failure} fsync failure"
                raise OSError(message)
        real_fsync(descriptor)

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "fsync", failing_fsync)
    monkeypatch.setattr(io, "open", recording_io_open)

    def overwrite() -> None:
        with ProductWriter(target, kind="x", overwrite=True) as writer:
            write_zarr_array(
                writer.add_asset("a", "a.zarr"), np.ones((1, 1), dtype=np.float32)
            )
            writer.commit({})

    with pytest.raises(OSError, match=f"{failure} fsync failure"):
        overwrite()

    restored = open_product(target)
    assert restored.write_id == first.write_id
    np.testing.assert_array_equal(
        restored.read("a"), np.zeros((1, 1), dtype=np.float32)
    )
    assert list(tmp_path.glob(".p.aside-*")) == []
    assert not staging.exists()


def test_commit_refuses_a_symlink_inside_staging(tmp_path: Path) -> None:
    """A staged symlink could escape the product, so commit refuses it."""
    target = tmp_path / "p"
    first = publish_product(
        target,
        kind="x",
        properties={},
        write_payloads=_write_arrays({"a": np.zeros((1, 1), dtype=np.float32)}),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.bin").write_bytes(b"untouched")

    def commit_with_symlink() -> None:
        with ProductWriter(target, kind="x", overwrite=True) as writer:
            (writer.staging / "linked.zarr").symlink_to(outside)
            writer.commit({})

    with pytest.raises(InvalidProcessingStateError, match="symbolic link"):
        commit_with_symlink()

    assert (outside / "keep.bin").read_bytes() == b"untouched"
    assert open_product(target).write_id == first.write_id
    assert list(tmp_path.glob(".p.aside-*")) == []


def test_metadata_only_receipt_commits_without_payloads(tmp_path: Path) -> None:
    """A pair receipt with no declared payloads still publishes and syncs."""
    target = tmp_path / "pair"

    def no_payloads(_staging: Path) -> dict[str, dict[str, object]]:
        return {}

    product = publish_product(
        target,
        kind="pair",
        properties={"faninsar:date": "20240101"},
        write_payloads=no_payloads,
    )
    assert sorted(path.name for path in target.iterdir()) == [ITEM_NAME]
    assert product.assets == {}
    assert open_product(target).write_id == product.write_id
