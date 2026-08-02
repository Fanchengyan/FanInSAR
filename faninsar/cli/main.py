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

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "warmup":
        from faninsar.cli.warmup import run_warmup

        return run_warmup(device=args.device, profile=args.profile)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
