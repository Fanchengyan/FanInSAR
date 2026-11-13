"""faninsar console entry point."""

from __future__ import annotations

import argparse
import sys


def _port_number(value: str) -> int:
    """Parse a TCP port for argparse, rejecting out-of-range values."""
    try:
        port = int(value)
    except ValueError as error:
        message = "port must be an integer"
        raise argparse.ArgumentTypeError(message) from error
    if not 0 <= port <= 65535:
        message = "port must be between 0 and 65535"
        raise argparse.ArgumentTypeError(message)
    return port


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
        "verify",
        help=(
            "Check committed product structure, required assets, paths, "
            "and declared array metadata"
        ),
        description=(
            "Check committed product structure, required assets, paths, "
            "and declared array metadata"
        ),
    )
    stack_verify.add_argument("work_dir", help="Stack work directory")
    stack_stac_serve = stack_sub.add_parser(
        "stac-serve",
        help="Serve committed products as a read-only STAC API",
        description=(
            "Serve committed products as a read-only STAC API and "
            "Range-capable asset service. The command runs in the foreground "
            "until interrupted."
        ),
    )
    stack_stac_serve.add_argument("work_dir", help="Stack work directory")
    stack_stac_serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: loopback 127.0.0.1)",
    )
    stack_stac_serve.add_argument(
        "--port",
        type=_port_number,
        default=8000,
        help="TCP port to bind; 0 selects an ephemeral port (default: 8000)",
    )
    stack_stac_serve.add_argument(
        "--no-persist-catalog",
        dest="persist_catalog",
        action="store_false",
        help="Serve in memory only; write no catalog.json or stac/ views",
    )
    stack_stac_serve.set_defaults(persist_catalog=True)

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


def _run_stack(args: argparse.Namespace) -> int:  # noqa: PLR0911
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
    if args.stack_command == "stac-serve":
        return _run_stac_serve(args)
    return 2


def _write_error(message: str) -> None:
    """Write one CLI diagnostic to stderr."""
    sys.stderr.write(f"faninsar: {message}\n")


def _run_stac_serve(args: argparse.Namespace) -> int:  # noqa: PLR0911
    """Run the read-only STAC service in the foreground.

    The service prints its actual Catalog and API URLs once it is ready; the
    CLI adds no second address line. ``SIGINT`` stops the service cleanly.
    """
    from faninsar.stack.session import Stack

    try:
        stack = Stack.open(args.work_dir)
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        _write_error(f"cannot open Stack work directory {args.work_dir}: {error}")
        return 1
    try:
        service = stack.stac_serve(
            host=args.host,
            port=args.port,
            persist_catalog=args.persist_catalog,
        )
    except KeyboardInterrupt:
        return 0
    except ImportError as error:
        _write_error(f"STAC serving runtime is unavailable: {error}")
        return 2
    except Exception as error:
        _write_error(f"STAC service failed to start: {error}")
        return 1
    try:
        service.wait()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        _write_error(f"STAC service failed: {error}")
        return 1
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
