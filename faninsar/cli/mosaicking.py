"""Command-line interface for midpoint burst merge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

__all__ = ["main"]


def _scope_from_spec(spec: dict) -> object:
    from faninsar.processing.mosaicking import MergeScope

    scope = spec.get("scope") or {}
    return MergeScope(
        domain=str(scope.get("domain", "radar")),
        track_id=str(scope.get("track_id", "T1_A")),
        channel=str(scope.get("channel", "vv")),
        reference_id=str(scope.get("reference_id", "cli-reference")),
        grid_identity=str(scope.get("grid_identity", "a" * 64)),
        polygon_identity=str(scope.get("polygon_identity", "cli-polygon")),
    )


def _window_from_slice(valid_slice: object) -> tuple[int, int, int, int]:
    rows, cols = valid_slice  # type: ignore[misc]
    return int(rows[0]), int(rows[1]), int(cols[0]), int(cols[1])


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
        description="Select native burst samples with a midpoint layout.",
    )
    parser.add_argument(
        "--spec",
        required=True,
        help="Path to a JSON spec listing burst windows to merge.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for the merge_layout product.",
    )
    parser.add_argument(
        "--pair-id",
        default="frame",
        help="Name used for the merged-window product directory.",
    )
    args = parser.parse_args(argv)
    spec_path = Path(args.spec)
    if not spec_path.exists():
        logger.error("spec file not found: %s", spec_path)
        return 2

    with spec_path.open() as handle:
        spec = json.load(handle)

    bursts = spec.get("bursts", spec) if isinstance(spec, dict) else spec
    if not bursts:
        logger.error("spec file is empty: %s", spec_path)
        return 2

    from faninsar.io.storage.merge_layout import publish_merge_layout
    from faninsar.io.storage.product import publish_product
    from faninsar.processing.mosaicking import (
        BurstWindow,
        LayoutSourceWindow,
        MergeSeam,
        build_merge_layout,
        merge_bursts,
    )

    scope = _scope_from_spec(spec if isinstance(spec, dict) else {})
    sources = []
    products = []
    for entry in bursts:
        window = _window_from_slice(entry["valid_slice"])
        height = window[1] - window[0]
        width = window[3] - window[2]
        valid = np.ones((height, width), dtype=bool)
        phase = float(entry.get("phase_offset_rad", 0.0))
        samples = np.full((height, width), np.exp(1j * phase), dtype=np.complex64)
        burst_id = str(entry["burst_id"])
        swath = str(entry.get("swath", "IW1"))
        sources.append(LayoutSourceWindow(burst_id, window, valid, swath))
        products.append(
            BurstWindow(
                scope=scope,
                burst_id=burst_id,
                acquisitions=tuple(entry.get("acquisitions") or ("cli-primary",)),
                write_ids=(f"{burst_id}-write",),
                phase_convention=str(entry.get("phase_convention", "reramped")),
                window=window,
                complex=samples,
                valid_mask=valid,
            )
        )
    seams = ()
    if isinstance(spec, dict):
        seams = tuple(
            MergeSeam(
                axis=item["axis"],
                boundary=int(item["boundary"]),
                earlier_id=str(item["earlier_id"]),
                later_id=str(item["later_id"]),
            )
            for item in spec.get("seams") or ()
        )
    layout = build_merge_layout(sources, scope=scope, seams=seams)
    merged = merge_bursts(products, layout=layout)
    output_dir = Path(args.output_dir)
    layout_product = publish_merge_layout(output_dir / "layout", layout)
    merged_product = publish_product(
        output_dir / args.pair_id,
        kind="merged-window",
        properties={
            "faninsar:window": list(merged.window),
            "faninsar:layout_write_id": layout_product.write_id,
        },
        write_payloads=lambda staging: _write_merged(staging, merged),
        inputs={"layout": layout_product},
    )
    logger.info("merge complete: %s", merged_product.path)
    return 0


def _write_merged(staging: Path, merged: object) -> dict[str, dict[str, object]]:
    from faninsar.io.storage._zarr_stac import write_zarr_array
    from faninsar.processing.mosaicking import MergedWindow

    if not isinstance(merged, MergedWindow):
        message = "CLI merge payload must be a MergedWindow"
        raise TypeError(message)
    write_zarr_array(staging / "complex.zarr", merged.complex)
    write_zarr_array(staging / "valid_mask.zarr", merged.valid_mask)
    write_zarr_array(staging / "source.zarr", merged.source)
    return {
        "complex": {"href": "complex.zarr", "roles": ["data"]},
        "valid_mask": {"href": "valid_mask.zarr", "roles": ["data"]},
        "source": {"href": "source.zarr", "roles": ["data"]},
    }


if __name__ == "__main__":
    sys.exit(main())
