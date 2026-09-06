"""Command-line interface for burst merge production."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.processing.mosaicking.products import BurstGeoProduct

logger = setup_logger(__name__)

__all__ = ["main"]


def _build_synthetic_product(
    entry: dict,
    grid_shape: tuple[int, int],
) -> BurstGeoProduct:
    """Reconstruct a synthetic BurstGeoProduct from a CLI spec entry.

    The CLI spec is a JSON list of dicts with keys ``burst_id``, ``path_id``,
    ``swath``, ``phase_offset_rad``, and ``valid_slice`` (two ``[start, stop]``
    pairs for rows and columns). This helper is intended for synthetic /
    test-driven CLI runs; production usage wires real geocoded products.
    """
    from datetime import date

    from faninsar.processing.mosaicking.grid import GeoGridSpec
    from faninsar.processing.mosaicking.overlap import compute_feather
    from faninsar.processing.mosaicking.products import BurstGeoProduct

    h, w = grid_shape
    z = np.zeros((h, w), dtype=np.complex64)
    mask = np.zeros((h, w), dtype=bool)
    rows, cols = entry["valid_slice"]
    mask[rows[0] : rows[1], cols[0] : cols[1]] = True
    phase = float(entry["phase_offset_rad"])
    z[mask] = np.exp(1j * phase, dtype=np.complex64)
    weight = compute_feather(mask, feather_width_px=0.0)
    coh = np.zeros((h, w), dtype=np.float32)
    coh[mask] = 1.0
    grid = GeoGridSpec(
        crs="EPSG:32633",
        transform=(0.0, 10.0, 0.0, 1000.0, 0.0, -10.0),
        width=w,
        height=h,
        resolution_m=(10.0, 10.0),
    )
    look = entry.get("look_direction", "ascending")
    return BurstGeoProduct(
        burst_id=entry["burst_id"],
        path_id=entry.get("path_id", "T1_A"),
        swath=entry.get("swath", "IW1"),
        date=date(2024, 1, 1),
        grid=grid,
        complex=z,
        weight=weight,
        coherence=coh,
        look_direction=look,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``faninsar-merge-bursts`` CLI.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code (0 on success).

    """
    parser = argparse.ArgumentParser(
        prog="faninsar-merge-bursts",
        description="Merge geocoded burst products into a weighted mosaic.",
    )
    parser.add_argument(
        "--spec",
        required=True,
        help="Path to a JSON spec listing burst products to merge.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for the mosaic Zarr store.",
    )
    parser.add_argument(
        "--pair-id",
        default="frame",
        help="Pair identifier used in the output Zarr store name.",
    )
    parser.add_argument(
        "--mode",
        choices=["complex_average", "phase_network"],
        default="phase_network",
        help="Merge mode (default: phase_network).",
    )
    parser.add_argument(
        "--min-overlap-px",
        type=int,
        default=500,
        help="Minimum overlap pixels to form a phase edge.",
    )
    parser.add_argument(
        "--min-edge-coherence",
        type=float,
        default=0.15,
        help="Minimum edge coherence.",
    )
    parser.add_argument(
        "--path-policy",
        choices=["same_path_only", "allow_cross_path"],
        default="allow_cross_path",
        help="Restrict phase edges to the same path or allow cross-path.",
    )
    parser.add_argument(
        "--feather-width-px",
        type=float,
        default=32.0,
        help="Feather half-width in pixels (production default: on).",
    )
    parser.add_argument(
        "--allow-unwrapped-merge",
        action="store_true",
        help="Opt in to unwrapped-phase merge (NOT recommended; not DoD).",
    )

    args = parser.parse_args(argv)
    spec_path = Path(args.spec)
    if not spec_path.exists():
        logger.error("spec file not found: %s", spec_path)
        return 2

    with spec_path.open() as f:
        spec = json.load(f)

    if not spec:
        logger.error("spec file is empty: %s", spec_path)
        return 2

    # Determine grid shape from the first entry's valid slice (synthetic CLI).
    h = max(entry["valid_slice"][0][1] for entry in spec)
    w = max(entry["valid_slice"][1][1] for entry in spec)
    grid_shape = (max(h, 1), max(w, 1))

    products = [_build_synthetic_product(entry, grid_shape) for entry in spec]

    from faninsar.processing.mosaicking.orchestration import run_network_merge

    out_path = run_network_merge(
        products,
        output_dir=Path(args.output_dir),
        merge_paths=args.path_policy == "allow_cross_path",
        pair_id=args.pair_id,
        min_overlap_px=args.min_overlap_px,
        min_edge_coherence=args.min_edge_coherence,
    )
    logger.info("merge complete: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
