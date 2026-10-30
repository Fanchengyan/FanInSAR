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

    stack = sub.add_parser(
        "stack", help="Inspect or recover a Stack work directory (PROPOSAL-0069)"
    )
    stack_sub = stack.add_subparsers(dest="stack_command")
    stack_status = stack_sub.add_parser(
        "status", help="Print the Stack status report for a work directory"
    )
    stack_status.add_argument("work_dir", help="Stack work directory")
    stack_repair = stack_sub.add_parser(
        "repair",
        help="Rewrite stack.json from recoverable identity + committed products",
    )
    stack_repair.add_argument("work_dir", help="Stack work directory")
    stack_repair.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the RepairReport without writing stack.json",
    )
    stack_verify = stack_sub.add_parser(
        "verify", help="Check committed product payloads against checksums"
    )
    stack_verify.add_argument("work_dir", help="Stack work directory")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "warmup":
        from faninsar.cli.warmup import run_warmup

        return run_warmup(device=args.device, profile=args.profile)
    if args.command == "stack":
        return _run_stack(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


def _emit(payload: object) -> None:
    """Write one JSON report to stdout."""
    import json

    sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")


def _run_stack(args: argparse.Namespace) -> int:
    """Dispatch the ``faninsar stack`` subcommands."""
    from pathlib import Path

    from faninsar.stack.session import Stack

    if args.stack_command == "status":
        try:
            stack = Stack.open(args.work_dir)
        except Exception:
            # A fully unreadable work directory still gets a rescan report.
            _emit(Stack.rescan(args.work_dir).as_dict())
            return 0
        _emit(stack.status())
        return 0
    if args.stack_command == "repair":
        if args.dry_run:
            _emit(Stack.rescan(args.work_dir).as_dict())
            return 0
        stack = Stack.repair(args.work_dir)
        _emit(stack.repair_report.as_dict())
        return 0
    if args.stack_command == "verify":
        from faninsar.io.storage.product import verify_product

        root = Path(args.work_dir)
        failures: dict[str, list[str]] = {}
        for item in sorted(root.rglob("item.json")):
            bad = verify_product(item.parent)
            if bad:
                failures[str(item.parent)] = bad
        _emit({"failures": failures})
        return 1 if failures else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
