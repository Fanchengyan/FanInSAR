"""Private baseline and oracle checks for the public provenance boundary."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

from .validation import (
    _digest,
    _fail,
    _integer,
    _items,
    _mapping,
    _resolve,
    _sha256,
    _text,
    _verify_pin,
)


def repository_root(manifest: Path) -> Path:
    candidates = (
        Path.cwd(),
        *Path.cwd().parents,
        manifest.parent,
        *manifest.parent.parents,
    )
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return Path.cwd()


def _mutable(path: str, mutable_paths: tuple[str, ...]) -> bool:
    return any(
        path == entry or (entry.endswith("/") and path.startswith(entry))
        for entry in mutable_paths
    )


def _current_hashes(root: Path, mutable_paths: tuple[str, ...]) -> bytes:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        _fail(root / ".git", f"cannot enumerate worktree files: {error}")
    records: list[str] = []
    for name in result.stdout.split("\0"):
        if not name or _mutable(name, mutable_paths):
            continue
        target = root / name
        if target.is_file():
            records.append(f"{name}\t{_sha256(target)}\t{target.stat().st_size}\n")
    return "".join(sorted(records)).encode()


def verify_baseline(root: Path, manifest: Path, baseline: Mapping[str, object]) -> None:
    retired_value = baseline.get("retired_history")
    if retired_value is not None:
        retired = _mapping(retired_value, manifest, "baseline.retired_history")
        note = _text(retired, "waymark_note", manifest)
        if not note.startswith("NOTE-"):
            _fail(manifest, "retired baseline must reference a Waymark Note")
        _digest(retired, "inventory_sha256", manifest)
        _digest(retired, "archive_sha256", manifest)
        _integer(retired, "archived_entry_count", manifest)
        source_paths = {
            _text({"value": item}, "value", manifest)
            for item in _items(
                retired.get("source_paths"), manifest, "retired_history.source_paths"
            )
        }
        expected_paths = {".omo/", "plans/", "reports/", "scripts/", "workflow.md"}
        if source_paths != expected_paths:
            _fail(manifest, "retired baseline source paths differ")
        _text(baseline, "git_head", manifest)
        _integer(baseline, "out_of_scope_file_count", manifest)
        return
    mutable = tuple(
        _text({"value": item}, "value", manifest)
        for item in _items(baseline.get("mutable_paths"), manifest, "mutable_paths")
    )
    hash_path = _resolve(root, _text(baseline, "out_of_scope_hashes_path", manifest))
    expected_digest = _text(baseline, "out_of_scope_hashes_sha256", manifest)
    if not hash_path.is_file() or _sha256(hash_path) != expected_digest:
        _fail(manifest, "pre-task out-of-scope hash artifact is missing or changed")
    expected_count = _integer(baseline, "out_of_scope_file_count", manifest)
    if len(hash_path.read_bytes().splitlines()) != expected_count:
        _fail(manifest, "pre-task out-of-scope hash artifact count differs")
    if _current_hashes(root, mutable) != hash_path.read_bytes():
        _fail(manifest, "post-task out-of-scope hashes differ from the pre-task list")
    for path_key, digest_key in (
        ("status_porcelain_path", "status_porcelain_sha256"),
        ("diff_stat_path", "diff_stat_sha256"),
    ):
        target = _resolve(root, _text(baseline, path_key, manifest))
        if not target.is_file() or _sha256(target) != _text(
            baseline, digest_key, manifest
        ):
            _fail(manifest, f"baseline artifact mismatch: {path_key}")
    expected_head = _text(baseline, "git_head", manifest)
    try:
        current_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        _fail(root / ".git", f"cannot read git HEAD: {error}")
    if current_head != expected_head:
        _fail(manifest, "git HEAD differs from the frozen baseline")


def validate_oracles(
    root: Path,
    manifest: Path,
    oracles: Mapping[str, object],
    fixed_path: str,
) -> tuple[int, tuple[str, ...]]:
    primary = _items(oracles.get("primary"), manifest, "oracles.primary")
    processors: list[str] = []
    verified = 0
    for index, value in enumerate(primary):
        item = _mapping(value, manifest, f"oracle[{index}]")
        processor = _text(item, "processor", manifest)
        if _text(item, "state", manifest) != "primary":
            _fail(manifest, f"oracle is not primary: {processor}")
        oracle_path = _resolve(root, _text(item, "path", manifest))
        if not oracle_path.is_dir():
            _fail(manifest, f"oracle directory is missing: {oracle_path}")
        processors.append(processor)
        for artifact_index, artifact in enumerate(
            _items(item.get("artifacts"), manifest, f"oracle[{processor}].artifacts")
        ):
            verified += _verify_pin(
                artifact,
                root,
                manifest,
                f"oracle[{processor}].artifact[{artifact_index}]",
            )
    if processors.count("isce2") != 1:
        _fail(manifest, "exactly one primary isce2 oracle is required")
    isce = next(
        item
        for item in primary
        if _text(_mapping(item, manifest, "oracle"), "processor", manifest) == "isce2"
    )
    if _text(_mapping(isce, manifest, "oracle"), "path", manifest) != fixed_path:
        _fail(manifest, "only the fixed ISCE2 oracle path may be primary")
    for index, value in enumerate(
        _items(oracles.get("forbidden"), manifest, "oracles.forbidden")
    ):
        item = _mapping(value, manifest, f"forbidden[{index}]")
        if _text(item, "state", manifest) == "primary":
            _fail(manifest, "forbidden or diagnostic oracle marked primary")
        if "sha256" in item:
            verified += _verify_pin(item, root, manifest, f"forbidden[{index}]")
        elif "waymark_note" in item:
            note = _text(item, "waymark_note", manifest)
            if not note.startswith("NOTE-"):
                _fail(manifest, "retired diagnostic must reference a Waymark Note")
            _text(item, "source_path_last_recorded", manifest)
            _digest(item, "source_sha256", manifest)
            _integer(item, "source_size_bytes", manifest)
            if _text(item, "availability", manifest) != (
                "unavailable-before-PROPOSAL-0010"
            ):
                _fail(manifest, "retired diagnostic availability differs")
            verified += 1
    return verified, tuple(processors)
