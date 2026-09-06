"""Tests for the merge CLI and failure-mode warnings."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from faninsar.cli.mosaicking import main as merge_cli_main
from faninsar.processing.mosaicking.grid import GeoGridSpec
from faninsar.processing.mosaicking.overlap import compute_feather
from faninsar.processing.mosaicking.products import BurstGeoProduct


def _grid() -> GeoGridSpec:
    return GeoGridSpec(
        crs="EPSG:32633",
        transform=(0.0, 10.0, 0.0, 1000.0, 0.0, -10.0),
        width=40,
        height=20,
        resolution_m=(10.0, 10.0),
    )


def _make_product(
    burst_id: str,
    phase_offset_rad: float,
    valid_slice: tuple[slice, slice],
    *,
    path_id: str = "T1_A",
) -> BurstGeoProduct:
    g = _grid()
    h, w = g.shape
    z = np.zeros((h, w), dtype=np.complex64)
    mask = np.zeros((h, w), dtype=bool)
    mask[valid_slice] = True
    z[mask] = np.exp(1j * phase_offset_rad, dtype=np.complex64)
    weight = compute_feather(mask, feather_width_px=0.0)
    coh = np.zeros((h, w), dtype=np.float32)
    coh[mask] = 1.0
    return BurstGeoProduct(
        burst_id=burst_id,
        path_id=path_id,
        swath="IW1",
        date=date(2024, 1, 1),
        grid=g,
        complex=z,
        weight=weight,
        coherence=coh,
    )


def _write_product_spec(tmp_path: Path, products: list[BurstGeoProduct]) -> Path:
    """Write a JSON spec file listing product parameters for the CLI."""
    spec = []
    for p in products:
        mask = p.weight > 0
        rows, cols = np.where(mask)
        valid_phase = float(np.angle(p.complex[rows[0], cols[0]]))
        spec.append(
            {
                "burst_id": p.burst_id,
                "path_id": p.path_id,
                "swath": p.swath,
                "phase_offset_rad": valid_phase,
                "valid_slice": [
                    [int(rows.min()), int(rows.max()) + 1],
                    [int(cols.min()), int(cols.max()) + 1],
                ],
            }
        )
    spec_path = tmp_path / "spec.json"
    with spec_path.open("w") as f:
        json.dump(spec, f)
    return spec_path


def test_cli_writes_mosaic_zarr(tmp_path: Path) -> None:
    """The CLI produces a mosaic Zarr from a spec file."""
    p_i = _make_product("a", 0.0, (slice(0, 20), slice(0, 16)))
    p_j = _make_product("b", 0.5, (slice(0, 20), slice(10, 26)))
    spec_path = _write_product_spec(tmp_path, [p_i, p_j])
    out_dir = tmp_path / "out"
    rc = merge_cli_main(
        [
            "--spec",
            str(spec_path),
            "--output-dir",
            str(out_dir),
            "--pair-id",
            "cli_test",
            "--min-overlap-px",
            "10",
            "--min-edge-coherence",
            "0.0",
        ]
    )
    assert rc == 0
    assert (out_dir / "cli_test_mosaic.zarr").exists()


def test_cli_rejects_missing_spec(tmp_path: Path) -> None:
    """The CLI exits non-zero when the spec file is missing."""
    rc = merge_cli_main(
        [
            "--spec",
            str(tmp_path / "nope.json"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc != 0
