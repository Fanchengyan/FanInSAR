"""Date / parent-box staging layout for track acquisitions."""

from __future__ import annotations

import json
from pathlib import Path

from faninsar.stack.layout import (
    acquisition_date_key,
    decode_identity_path,
    encode_identity_path,
    migrate_acquisition_staging,
    migrate_workspace_acquisitions,
    parent_box_stem,
    s1_transfer_root,
    staged_acquisition_path,
)

PARENT = "S1A_IW_SLC__1SDV_20260221T053526_20260221T053553_063313_07F369_7F6C"
NEIGHBOUR = "S1A_IW_SLC__1SDV_20260221T053501_20260221T053528_063313_07F369_4180"


def test_parent_box_stem_keeps_full_esa_product_id() -> None:
    """Complementary IW canvases of one ESA product share the full stem."""
    assert parent_box_stem(f"{PARENT}-IW1-VV.SAFE") == PARENT
    assert parent_box_stem(f"{PARENT}_IW2-VV.SAFE.zip") == PARENT
    assert parent_box_stem(f"{PARENT}.SAFE") == PARENT


def test_acquisition_date_key_reads_esa_start_field() -> None:
    """The date folder is the eight digits already in the product name."""
    assert acquisition_date_key(f"{PARENT}-IW2-VV.SAFE") == "20260221"
    assert acquisition_date_key("S1_FAILED.zip") == "undated"


def test_s1_transfer_root_is_outside_the_safe(tmp_path: Path) -> None:
    """CDSE range scratch lives under work_dir/tmp, keyed by canvas stem."""
    name = f"{PARENT}-IW2-VV.SAFE"
    assert s1_transfer_root(tmp_path, name) == (
        tmp_path / "tmp" / "s1-transfer" / f"{PARENT}-IW2-VV"
    )


def test_staged_acquisition_path_nests_date_then_parent(tmp_path: Path) -> None:
    """A canvas SAFE lands in acquisitions/<date>/<parent>/<canvas>.SAFE."""
    name = f"{PARENT}-IW2-VV.SAFE"
    root = tmp_path / "tracks" / "66D" / "acquisitions"
    dest = staged_acquisition_path(root, name)
    assert dest == root / "20260221" / PARENT / name


def test_migrate_acquisition_staging_is_idempotent(tmp_path: Path) -> None:
    """A flat canvas tree gains date/parent folders; a second pass is a no-op."""
    root = tmp_path / "acquisitions"
    iw1 = root / f"{PARENT}-IW1-VV.SAFE"
    iw2 = root / f"{PARENT}-IW2-VV.SAFE"
    only = root / f"{NEIGHBOUR}-IW1-VV.SAFE"
    for path in (iw1, iw2, only):
        path.mkdir(parents=True)
        (path / "manifest.safe").write_text("ok", encoding="utf-8")

    moved = migrate_acquisition_staging(root)
    assert len(moved) == 3
    assert (root / "20260221" / PARENT / iw1.name / "manifest.safe").is_file()
    assert (root / "20260221" / PARENT / iw2.name / "manifest.safe").is_file()
    assert (root / "20260221" / NEIGHBOUR / only.name / "manifest.safe").is_file()
    assert not iw1.exists()
    assert migrate_acquisition_staging(root) == []


def test_migrate_lifts_parent_box_into_date_folder(tmp_path: Path) -> None:
    """The previous parent-only layout becomes date/parent without rewriting SAFE."""
    root = tmp_path / "acquisitions"
    boxed = root / PARENT / f"{PARENT}-IW1-VV.SAFE"
    boxed.mkdir(parents=True)
    (boxed / "manifest.safe").write_text("ok", encoding="utf-8")

    moved = migrate_acquisition_staging(root)
    assert len(moved) == 1
    dest = root / "20260221" / PARENT / boxed.name
    assert dest.is_dir()
    assert not (root / PARENT).exists()
    assert migrate_acquisition_staging(root) == []


def test_migrate_workspace_rewrites_stack_json_paths(tmp_path: Path) -> None:
    """Track identity paths follow the renamed SAFE directories."""
    track = tmp_path / "tracks" / "66D"
    acq = track / "acquisitions"
    safe = acq / f"{PARENT}-IW1-VV.SAFE"
    safe.mkdir(parents=True)
    (safe / "manifest.safe").write_text("ok", encoding="utf-8")
    identity = {
        "schema": "faninsar.stack.v1",
        "inputs": {
            "sources": {
                "20260221": [{"path": str(safe), "size": 1}],
            }
        },
    }
    (track / "stack.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")

    moved = migrate_workspace_acquisitions(tmp_path)
    assert len(moved) == 1
    new_path = acq / "20260221" / PARENT / safe.name
    assert new_path.is_dir()
    document = json.loads((track / "stack.json").read_text(encoding="utf-8"))
    assert document["inputs"]["sources"]["20260221"][0]["path"] == (
        f"acquisitions/20260221/{PARENT}/{safe.name}"
    )


def test_migrate_workspace_rewrites_foreign_track_paths(tmp_path: Path) -> None:
    """A track identity that listed another track's SAFE still follows the move.

    Multi-track workspaces can record a stray source under the wrong
    ``stack.json``; the workspace mapping, not the local rename list,
    has to rewrite it — including on a second pass after the files are
    already nested.
    """
    parent = "S1C_IW_SLC__1SDV_20260417T172222_20260417T172250_007259_00EB69_8B54"
    name = f"{parent}-IW2-VV.SAFE"
    owner = tmp_path / "tracks" / "88A" / "acquisitions"
    safe = owner / name
    safe.mkdir(parents=True)
    (safe / "manifest.safe").write_text("ok", encoding="utf-8")
    stray = tmp_path / "tracks" / "66D"
    stray.mkdir(parents=True)
    (stray / "stack.json").write_text(
        json.dumps(
            {
                "schema": "faninsar.stack.v1",
                "inputs": {"sources": {"20260417": [{"path": str(safe), "size": 1}]}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert len(migrate_workspace_acquisitions(tmp_path)) == 1
    nested = owner / "20260417" / parent / name
    relative = f"../88A/acquisitions/20260417/{parent}/{name}"
    document = json.loads((stray / "stack.json").read_text(encoding="utf-8"))
    assert document["inputs"]["sources"]["20260417"][0]["path"] == relative
    assert nested.is_dir()

    assert migrate_workspace_acquisitions(tmp_path) == []
    document = json.loads((stray / "stack.json").read_text(encoding="utf-8"))
    assert document["inputs"]["sources"]["20260417"][0]["path"] == relative


def test_encode_identity_path_is_relative_inside_workspace(tmp_path: Path) -> None:
    """Staged files under the workspace become identity-relative POSIX paths."""
    track = tmp_path / "tracks" / "66D"
    safe = track / "acquisitions" / "20260221" / PARENT / f"{PARENT}-IW1-VV.SAFE"
    safe.mkdir(parents=True)
    assert encode_identity_path(safe, track) == (
        f"acquisitions/20260221/{PARENT}/{PARENT}-IW1-VV.SAFE"
    )
    dem = tmp_path / "dem.tif"
    dem.write_bytes(b"dem")
    assert encode_identity_path(dem, track) == "../../dem.tif"
    assert decode_identity_path("../../dem.tif", track) == dem.resolve()


def test_encode_identity_path_stays_absolute_outside_workspace(tmp_path: Path) -> None:
    """A SAFE outside the workspace is not rewritten as a relative path."""
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "shared" / "scene.SAFE"
    outside.mkdir(parents=True)
    recorded = encode_identity_path(outside, work)
    assert Path(recorded).is_absolute()
    assert Path(recorded) == outside
