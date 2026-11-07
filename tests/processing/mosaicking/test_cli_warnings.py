"""Tests for the midpoint merge CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from faninsar.cli.mosaicking import main as merge_cli_main
from faninsar.io.storage.merge_layout import LAYOUT_KIND, open_merge_layout
from faninsar.io.storage.product import open_product

if TYPE_CHECKING:
    from pathlib import Path


def test_cli_writes_layout_and_merged_products(tmp_path: Path) -> None:
    """The CLI publishes a merge_layout and a merged-window product."""
    spec = {
        "scope": {
            "domain": "radar",
            "track_id": "T1_A",
            "channel": "vv",
            "reference_id": "cli-reference",
            "grid_identity": "a" * 64,
            "polygon_identity": "cli-polygon",
        },
        "bursts": [
            {
                "burst_id": "a",
                "swath": "IW1",
                "phase_offset_rad": 0.0,
                "valid_slice": [[0, 4], [0, 3]],
            },
            {
                "burst_id": "b",
                "swath": "IW1",
                "phase_offset_rad": 0.5,
                "valid_slice": [[4, 8], [0, 3]],
            },
        ],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    out_dir = tmp_path / "out"
    rc = merge_cli_main(
        [
            "--spec",
            str(spec_path),
            "--output-dir",
            str(out_dir),
            "--pair-id",
            "cli_test",
        ]
    )
    assert rc == 0
    layout = open_merge_layout(out_dir / "layout")
    assert layout.scope.track_id == "T1_A"
    merged = open_product(out_dir / "cli_test", kind="merged-window")
    assert merged.kind == "merged-window"
    assert LAYOUT_KIND == "merge_layout"
    assert merged.inputs["layout"] == open_product(out_dir / "layout").write_id


def test_cli_rejects_missing_spec(tmp_path: Path) -> None:
    """The CLI exits non-zero when the spec file is missing."""
    rc = merge_cli_main(
        ["--spec", str(tmp_path / "nope.json"), "--output-dir", str(tmp_path / "out")]
    )
    assert rc != 0
