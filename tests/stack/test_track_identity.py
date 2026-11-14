"""Stage-6 track-split ``stack.json`` persistence.

A Stack occupying more than one track group writes a split identity: the
root ``stack.json`` carries only schema/mission/the ``tracks`` list/global
science, while each track owns a complete ``tracks/<trackKey>/stack.json``
with its scene catalog, planned pairs, Reference, and DEM/mask references.
Reading the root document merges the per-track documents back into the
flat shape :meth:`Stack.open` already consumes. A single-track Stack
keeps the legacy flat document.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from faninsar.core import Pairs
from faninsar.core.sar_tools import Looks
from faninsar.missions.s1.stack import S1Stack
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.stack import StackConfig
from faninsar.stack.catalog import SceneCatalog
from faninsar.stack.identity import (
    STACK_IDENTITY_SCHEMA,
    merge_track_documents,
    read_stack_identity,
    read_stack_identity_lenient,
    save_stack_identity,
    stack_identity,
)

_DATES_88A = ("20240101", "20240113")
_DATES_12D = ("20240125",)


def _make_safe(root: Path, date_id: str) -> Path:
    safe = root / f"S1A_IW_SLC__1SDV_{date_id}T000000.SAFE"
    safe.mkdir(parents=True)
    return safe


def _metadata(track_key: str, orbit: int, direction: str) -> dict[str, object]:
    return {
        "track_key": track_key,
        "direction": direction,
        "relative_orbit": orbit,
        "mission_id": "S1A",
        "polarizations": ("VV",),
        "absolute_burst_ids": (3001,),
    }


def _tracked_stack(tmp_path: Path, *, two_tracks: bool = True) -> S1Stack:
    """Build a planned S1Stack with injected per-date track metadata.

    ``two_tracks`` puts two dates on ``88A`` and one on ``12D``; the
    single-track layout puts all three on ``88A`` for the passthrough
    case. The persisted ``track_metadata`` extra makes the mission hook
    decode the recorded map without opening the empty SAFE directories.
    """
    dates = _DATES_88A + _DATES_12D
    paths = [_make_safe(tmp_path / "inputs", date_id) for date_id in dates]
    stack = S1Stack(
        catalog=SceneCatalog.from_paths(paths),
        config=StackConfig(
            work_dir=tmp_path / "work",
            multilook=(1, 1),
            roi=None,
            dem=None,
        ),
        pairs=Pairs.from_names(
            ["20240101_20240113", "20240113_20240125", "20240101_20240125"]
        ),
        misreg_pairs=Pairs.from_names(["20240101_20240113"]),
        reference="20240101",
    )
    records: dict[str, dict[str, object]] = {
        date_id: _metadata("88A", 88, "ASCENDING") for date_id in _DATES_88A
    }
    records["20240125"] = (
        _metadata("12D", 12, "DESCENDING")
        if two_tracks
        else _metadata("88A", 88, "ASCENDING")
    )
    stack.config.extra["track_metadata"] = {"dates": records}
    stack.config.extra["orbit_paths"] = {
        date_id: tmp_path / "orbits" / f"{date_id}.EOF" for date_id in dates
    }
    for orbit_path in stack.config.extra["orbit_paths"].values():
        orbit_path.parent.mkdir(parents=True, exist_ok=True)
        orbit_path.write_text("orbit", encoding="utf-8")
    return stack


# ---------------------------------------------------------------------------
# Multi-track write
# ---------------------------------------------------------------------------


def test_multi_track_save_writes_split_documents(tmp_path: Path) -> None:
    """A two-track Stack writes tracks/<key>/stack.json plus a slim root."""
    stack = _tracked_stack(tmp_path)
    root_doc_path = save_stack_identity(stack)

    assert root_doc_path == stack.config.work_dir / "stack.json"
    sub_88a = stack.config.work_dir / "tracks" / "88A" / "stack.json"
    sub_12d = stack.config.work_dir / "tracks" / "12D" / "stack.json"
    assert sub_88a.is_file()
    assert sub_12d.is_file()

    root = json.loads(root_doc_path.read_text(encoding="utf-8"))
    assert root["schema"] == STACK_IDENTITY_SCHEMA
    assert root["mission"] == "S1Stack"
    assert sorted(root["tracks"]) == ["12D", "88A"]
    # The split root carries no per-track payload: catalogs, pairs, and
    # Reference live on the track documents so a mixed-track pair graph
    # cannot leak into the workspace identity.
    assert "inputs" not in root
    assert "pairs" not in root
    assert "misreg_pairs" not in root
    assert "reference" not in root
    # Global science configuration is still persisted at the root.
    # Look factors are per-track: a 10x2 IFG on 88A must not be overwritten
    # by a leftover root default of (5, 2).
    assert root["science"]["coreg_mode"] == "geometry"
    assert "multilook" not in root["science"]
    assert "track_metadata" in root["science_extra"]


def test_track_sub_document_is_complete_identity(tmp_path: Path) -> None:
    """tracks/<key>/stack.json records that track's catalog and pairs."""
    stack = _tracked_stack(tmp_path)
    save_stack_identity(stack)

    sub = json.loads(
        (stack.config.work_dir / "tracks" / "88A" / "stack.json").read_text(
            encoding="utf-8"
        )
    )
    assert sub["schema"] == STACK_IDENTITY_SCHEMA
    assert sub["mission"] == "S1Stack"
    # Only the two 88A dates land in the track's own catalog.
    assert sorted(sub["inputs"]["sources"]) == ["20240101", "20240113"]
    # The single-track rule drops the cross-track pairs.
    assert sub["pairs"] == ["20240101_20240113"]
    assert sub["misreg_pairs"] == ["20240101_20240113"]
    assert sub["reference"] == "20240101"
    assert sub["science"]["multilook"] == {"faninsar:tuple": [1, 1]}
    # Orbit inputs are scoped to the track's dates.
    assert sorted(sub["inputs"]["orbits"]) == ["20240101", "20240113"]
    # Persisted track metadata is scoped likewise.
    scoped = sub["science_extra"]["track_metadata"]["dates"]
    assert sorted(scoped) == ["20240101", "20240113"]

    sub_12d = json.loads(
        (stack.config.work_dir / "tracks" / "12D" / "stack.json").read_text(
            encoding="utf-8"
        )
    )
    assert sorted(sub_12d["inputs"]["sources"]) == ["20240125"]
    assert sub_12d["pairs"] == []
    # The parent's reference lives on 88A; the track doc falls back to its
    # own earliest date so the sub-stack is self-consistent.
    assert sub_12d["reference"] == "20240125"


def test_single_track_save_keeps_flat_document(tmp_path: Path) -> None:
    """A one-track Stack writes the legacy flat stack.json unchanged."""
    stack = _tracked_stack(tmp_path, two_tracks=False)
    root_doc_path = save_stack_identity(stack)

    root = json.loads(root_doc_path.read_text(encoding="utf-8"))
    assert "tracks" not in root
    assert sorted(root["inputs"]["sources"]) == [
        "20240101",
        "20240113",
        "20240125",
    ]
    # Pairs keep their planned-network order (Pairs.from_names sort=False).
    assert root["pairs"] == [
        "20240101_20240113",
        "20240113_20240125",
        "20240101_20240125",
    ]
    # No per-track documents are produced.
    assert not (stack.config.work_dir / "tracks").exists()


def test_untracked_stack_writes_flat_document(tmp_path: Path) -> None:
    """A Stack without track metadata never splits the document."""
    paths = [
        _make_safe(tmp_path / "inputs", date_id) for date_id in ("20240101", "20240113")
    ]
    stack = S1Stack(
        catalog=SceneCatalog.from_paths(paths),
        config=StackConfig(work_dir=tmp_path / "work", multilook=(1, 1)),
        pairs=Pairs.from_names(["20240101_20240113"]),
        misreg_pairs=Pairs.from_names([]),
        reference="20240101",
    )
    root_doc_path = save_stack_identity(stack)
    root = json.loads(root_doc_path.read_text(encoding="utf-8"))
    assert "tracks" not in root
    assert "inputs" in root


# ---------------------------------------------------------------------------
# Read-side merge
# ---------------------------------------------------------------------------


def test_read_stack_identity_merges_track_documents(tmp_path: Path) -> None:
    """The strict reader expands a track-split root into the flat shape."""
    stack = _tracked_stack(tmp_path)
    save_stack_identity(stack)

    merged = read_stack_identity(stack.config.work_dir)
    assert sorted(merged["tracks"]) == ["12D", "88A"]
    assert sorted(merged["inputs"]["sources"]) == [
        "20240101",
        "20240113",
        "20240125",
    ]
    # Pair networks are the per-track union; mixed-track names do not
    # reappear just because the workspace was opened at the root.
    assert merged["pairs"] == ["20240101_20240113"]
    assert merged["misreg_pairs"] == ["20240101_20240113"]
    assert merged["reference"] == "20240101"
    assert merged["science"]["multilook"] == {"faninsar:tuple": [1, 1]}
    # Orbit inputs union across tracks.
    assert sorted(merged["inputs"]["orbits"]) == [
        "20240101",
        "20240113",
        "20240125",
    ]
    # Date-keyed track metadata deep-merges.
    dates = merged["science_extra"]["track_metadata"]["dates"]
    assert {d: v["track_key"] for d, v in dates.items()} == {
        "20240101": "88A",
        "20240113": "88A",
        "20240125": "12D",
    }
    orbit_paths = merged["science_extra"]["orbit_paths"]
    assert sorted(orbit_paths) == ["20240101", "20240113", "20240125"]


def test_merged_open_restores_track_grouped_stack(tmp_path: Path) -> None:
    """Stack.open on a split workspace rebuilds the full catalog/pairs."""
    stack = _tracked_stack(tmp_path)
    save_stack_identity(stack)

    reopened = S1Stack.open(stack.config.work_dir)
    assert reopened.catalog.dates == ("20240101", "20240113", "20240125")
    assert reopened.pairs.to_names().tolist() == ["20240101_20240113"]
    assert reopened.misreg_pairs.to_names().tolist() == ["20240101_20240113"]
    assert reopened.reference == "20240101"
    # The persisted track map survives the merge so track_groups returns
    # without reopening the SAFEs.
    assert reopened.track_groups == ("12D", "88A")
    infos = reopened._catalog_track_infos()
    assert infos["20240125"].track_key == "12D"


def test_strict_read_rejects_missing_track_document(tmp_path: Path) -> None:
    """A root doc naming a missing track stack.json fails closed."""
    stack = _tracked_stack(tmp_path)
    save_stack_identity(stack)
    (stack.config.work_dir / "tracks" / "12D" / "stack.json").unlink()

    with pytest.raises(InvalidProcessingStateError, match="12D"):
        read_stack_identity(stack.config.work_dir)


def test_lenient_read_skips_missing_track_document(tmp_path: Path) -> None:
    """The lenient reader drops an unloadable track and keeps the rest."""
    stack = _tracked_stack(tmp_path)
    save_stack_identity(stack)
    (stack.config.work_dir / "tracks" / "12D" / "stack.json").unlink()

    merged = read_stack_identity_lenient(stack.config.work_dir)
    assert merged is not None
    assert sorted(merged["inputs"]["sources"]) == ["20240101", "20240113"]
    # Only the surviving track's pair network remains.
    assert merged["pairs"] == ["20240101_20240113"]


def test_merge_is_noop_without_tracks_key(tmp_path: Path) -> None:
    """merge_track_documents returns a flat document unchanged."""
    stack = _tracked_stack(tmp_path, two_tracks=False)
    save_stack_identity(stack)
    document = json.loads(
        (stack.config.work_dir / "stack.json").read_text(encoding="utf-8")
    )
    merged = merge_track_documents(stack.config.work_dir, document, strict=True)
    assert merged is document


def test_stack_identity_unchanged_shape(tmp_path: Path) -> None:
    """stack_identity still builds the flat document for a single stack."""
    stack = _tracked_stack(tmp_path)
    document = stack_identity(stack)
    assert document["schema"] == STACK_IDENTITY_SCHEMA
    assert sorted(document["inputs"]["sources"]) == [
        "20240101",
        "20240113",
        "20240125",
    ]
    assert "tracks" not in document


# ---------------------------------------------------------------------------
# Independent sub-stack open
# ---------------------------------------------------------------------------


def test_sub_track_stack_json_independently_openable(tmp_path: Path) -> None:
    """tracks/<key> opens as its own Stack with the track's catalog."""
    stack = _tracked_stack(tmp_path)
    save_stack_identity(stack)

    sub = S1Stack.open(stack.config.work_dir / "tracks" / "88A")
    assert sub.catalog.dates == ("20240101", "20240113")
    assert sub.pairs.to_names().tolist() == ["20240101_20240113"]
    assert sub.reference == "20240101"
    assert sub.track_groups == ("88A",)


def test_sub_track_open_for_off_reference_track(tmp_path: Path) -> None:
    """The track without the parent's reference opens on its own earliest."""
    stack = _tracked_stack(tmp_path)
    save_stack_identity(stack)

    sub = S1Stack.open(stack.config.work_dir / "tracks" / "12D")
    assert sub.catalog.dates == ("20240125",)
    assert sub.reference == "20240125"
    assert sub.track_groups == ("12D",)


def test_workspace_paths_are_relative_and_survive_copy(tmp_path: Path) -> None:
    """Staged SAFEs are recorded relative to stack.json and reopen after a copy."""
    work = tmp_path / "work"
    dates = ("20240101", "20240113")
    paths = [_make_safe(work / "inputs", date_id) for date_id in dates]
    stack = S1Stack(
        catalog=SceneCatalog.from_paths(paths),
        config=StackConfig(work_dir=work, multilook=(1, 1)),
        pairs=Pairs.from_names(["20240101_20240113"]),
        misreg_pairs=Pairs.from_names([]),
        reference="20240101",
    )
    save_stack_identity(stack)

    document = json.loads((work / "stack.json").read_text(encoding="utf-8"))
    recorded = document["inputs"]["sources"]["20240101"][0]["path"]
    assert recorded == f"inputs/{paths[0].name}"
    assert not Path(recorded).is_absolute()

    copied = tmp_path / "copy"
    shutil.copytree(work, copied)
    reopened = S1Stack.open(copied)
    restored = reopened.catalog.paths_for("20240101")[0]
    assert restored.exists()
    assert restored == (copied / recorded).resolve()


def test_track_multilook_is_independent_per_track(tmp_path: Path) -> None:
    """Each track document keeps its own look factors after a later rewrite."""
    stack = _tracked_stack(tmp_path)
    save_stack_identity(stack)
    path_12d = stack.config.work_dir / "tracks" / "12D" / "stack.json"
    document = json.loads(path_12d.read_text(encoding="utf-8"))
    document["science"]["multilook"] = {"faninsar:tuple": [10, 2]}
    path_12d.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    sub = S1Stack.open(stack.config.work_dir / "tracks" / "12D")
    assert sub.config.multilook == Looks(10, 2)
    other = S1Stack.open(stack.config.work_dir / "tracks" / "88A")
    assert other.config.multilook == Looks(1, 1)


def test_external_safe_paths_stay_absolute(tmp_path: Path) -> None:
    """A SAFE outside the work directory stays an absolute recorded path."""
    stack = _tracked_stack(tmp_path, two_tracks=False)
    save_stack_identity(stack)
    document = json.loads(
        (stack.config.work_dir / "stack.json").read_text(encoding="utf-8")
    )
    recorded = document["inputs"]["sources"]["20240101"][0]["path"]
    assert Path(recorded).is_absolute()
    assert Path(recorded).exists()
