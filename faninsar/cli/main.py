"""faninsar console entry point."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Run the ``faninsar`` CLI.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.

    """
    parser = argparse.ArgumentParser(
        prog="faninsar",
        description="FanInSAR processing and time-series toolkit",
    )
    sub = parser.add_subparsers(dest="command")

    warm = sub.add_parser("warmup", help="Pre-compile hot Torch kernels")
    warm.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Torch device for warmup",
    )
    warm.add_argument(
        "--profile",
        default="sentinel1",
        help="Warmup profile name (e.g. sentinel1, sentinel1-cuda)",
    )

    frame = sub.add_parser(
        "frame", help="Process one or more frames, sub-swaths, or an ROI"
    )
    frame.add_argument(
        "--paths",
        default=None,
        help="All acquisition SAFE products, comma-separated",
    )
    frame.add_argument(
        "--reference",
        default=None,
        help="Removed pair-era option; use --paths",
    )
    frame.add_argument(
        "--secondary",
        default=None,
        help="Removed pair-era option; use --paths",
    )
    frame.add_argument("--output", required=True, help="Output directory")
    frame.add_argument("--dem", default=None, help="DEM GeoTIFF (optional)")
    frame.add_argument(
        "--dem-source",
        default=None,
        help=(
            "DEM source selection for bare-name/automatic builds "
            "(<product> or <product>:<provider>, e.g. glo30, glo90, "
            "nasadem:earthdata); omitted keeps env/default behavior"
        ),
    )
    frame.add_argument("--reference-orbit", default=None, help="Reference POEORB EOF")
    frame.add_argument("--secondary-orbit", default=None, help="Secondary POEORB EOF")
    frame.add_argument(
        "--swaths",
        default="IW1,IW2,IW3",
        help="Comma-separated sub-swaths in range order",
    )
    frame.add_argument(
        "--bursts",
        default=None,
        help="Burst subset per swath, e.g. IW1:0,1,2,IW2:2:5,IW3:all",
    )
    frame.add_argument("--az-looks", type=int, default=2, help="Azimuth looks")
    frame.add_argument("--rg-looks", type=int, default=10, help="Range looks")
    frame.add_argument("--goldstein", type=float, default=0.5, help="Goldstein alpha")
    frame.add_argument(
        "--roi",
        default=None,
        help="lon_min,lat_min,lon_max,lat_max in EPSG:4326 (selects bursts)",
    )
    frame.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Torch device",
    )

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "warmup":
        from faninsar.cli.warmup import run_warmup

        return run_warmup(device=args.device, profile=args.profile)
    if args.command == "frame":
        from faninsar.cli.frame import run_frame_cli
        from faninsar.processing.errors import PairConfigurationMigrationError

        if args.dem_source:
            # Fail closed BEFORE any pipeline work on unwired/unknown pairs.
            # The 'auto' alias is valid everywhere else (DEMManager resolves
            # it, not the selection grammar), so it passes the pre-gate
            # untouched.
            from faninsar.processing.geometry.dem_sources import (
                AUTO_SOURCE_NAME,
                parse_selection,
            )

            if args.dem_source != AUTO_SOURCE_NAME:
                try:
                    parse_selection(args.dem_source)
                except (ValueError, TypeError) as exc:
                    parser.exit(2, f"faninsar frame: error: {exc}\n")

        try:
            return run_frame_cli(
                paths=args.paths,
                reference=args.reference,
                secondary=args.secondary,
                output=args.output,
                dem=args.dem,
                reference_orbit=args.reference_orbit,
                secondary_orbit=args.secondary_orbit,
                swaths=args.swaths,
                bursts=args.bursts,
                roi=args.roi,
                az_looks=args.az_looks,
                rg_looks=args.rg_looks,
                goldstein=args.goldstein,
                device=args.device,
                dem_source=args.dem_source,
            )
        except PairConfigurationMigrationError as exc:
            parser.exit(2, f"faninsar frame: migration error: {exc}\n")
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
