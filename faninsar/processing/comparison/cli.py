"""CLI entry point for pair-product comparison against external references."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
from faninsar.processing.comparison.metrics import (
    PairComparisonError,
    compare_pair_products,
)

logger = setup_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faninsar-compare-pair",
        description=(
            "Compare FanInSAR pair products against a reference processor output."
        ),
    )
    parser.add_argument(
        "faninsar_zarr",
        type=str,
        help="Path to the FanInSAR pair product Zarr store",
    )
    parser.add_argument(
        "reference_zarr",
        type=str,
        help="Path to the reference pair product Zarr store",
    )
    parser.add_argument(
        "out_json",
        type=str,
        help="Output JSON report path",
    )
    parser.add_argument(
        "--pair-id",
        type=str,
        default="unknown-pair",
        help="Pair identifier string",
    )
    parser.add_argument(
        "--reference-source",
        type=str,
        default="external",
        help="Human-readable reference processor name",
    )
    parser.add_argument(
        "--candidate-source",
        type=str,
        default="faninsar",
        help="Human-readable candidate processor name",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="wrapped_phase,coherence,unwrapped_phase",
        help="Comma-separated layer names to compare",
    )
    parser.add_argument(
        "--lon-lat",
        type=str,
        default="lon,lat",
        help="Comma-separated lon/lat layer names (or 'none' to skip)",
    )
    return parser


def main(argv: "Sequence[str]" | None = None) -> int:
    """Run the comparison CLI and return an exit code.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments.  Defaults to :data:`sys.argv[1:]`.

    Returns
    -------
    int
        ``0`` on success, ``1`` on comparison error.

    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    layers = tuple(args.layers.split(","))
    lon_lat: tuple[str, str] | None
    if args.lon_lat.lower() == "none":
        lon_lat = None
    else:
        parts = args.lon_lat.split(",")
        if len(parts) != 2:
            logger.exception("--lon-lat must be two comma-separated names or 'none'")
            return 1
        lon_lat = (parts[0], parts[1])

    try:
        compare_pair_products(
            faninsar_zarr=args.faninsar_zarr,
            reference_zarr_or_arrays=args.reference_zarr,
            out_json=args.out_json,
            pair_id=args.pair_id,
            reference_source=args.reference_source,
            candidate_source=args.candidate_source,
            layers=layers,
            lon_lat_layers=lon_lat,
        )
    except PairComparisonError as exc:
        logger.exception("Comparison failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
