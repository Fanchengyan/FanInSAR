# ruff: noqa: INP001
"""CLI lifecycle tests for ``faninsar stack stac-serve`` (PROPOSAL-0089)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import faninsar.stack.session as session_module
from faninsar.cli.main import main


class _FakeService:
    """Record foreground wait/stop calls and inject a terminal outcome."""

    def __init__(self, *, wait_error: BaseException | None = None) -> None:
        self.wait_error = wait_error
        self.wait_calls = 0
        self.stop_calls = 0

    def wait(self) -> None:
        self.wait_calls += 1
        if self.wait_error is not None:
            raise self.wait_error

    def stop(self) -> None:
        self.stop_calls += 1


class _FakeStack:
    """Stand-in for the reopened Stack public seam."""

    def __init__(
        self,
        *,
        service: _FakeService | None = None,
        start_error: BaseException | None = None,
    ) -> None:
        self.service = _FakeService() if service is None else service
        self.start_error = start_error
        self.serve_calls: list[dict[str, object]] = []

    def stac_serve(self, **kwargs: object) -> _FakeService:
        self.serve_calls.append(dict(kwargs))
        if self.start_error is not None:
            raise self.start_error
        return self.service


def _install_stack(monkeypatch: pytest.MonkeyPatch, stack: _FakeStack) -> list[str]:
    """Patch ``Stack.open`` to return ``stack`` and record its arguments."""
    opened: list[str] = []

    def fake_open(work_dir: str | Path) -> _FakeStack:
        opened.append(str(work_dir))
        return stack

    monkeypatch.setattr(session_module.Stack, "open", staticmethod(fake_open))
    return opened


def test_stack_stac_serve_delegates_to_agreed_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Stack seam forwards its work directory and options lazily."""
    service = _FakeService()
    calls: list[tuple[Path, str, int, bool]] = []

    def start_stac_service(
        work_dir: Path,
        *,
        host: str,
        port: int,
        persist_catalog: bool,
    ) -> _FakeService:
        calls.append((Path(work_dir), host, port, persist_catalog))
        return service

    module = types.ModuleType("faninsar.stack._stac_service")
    module.start_stac_service = start_stac_service  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faninsar.stack._stac_service", module)

    class _Config:
        work_dir = tmp_path

    class _Stack:
        config = _Config()

    result = session_module.Stack.stac_serve(
        _Stack(), host="127.0.0.1", port=0, persist_catalog=False
    )

    assert result is service
    assert calls == [(tmp_path, "127.0.0.1", 0, False)]


def test_stac_serve_runs_foreground_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command opens locally, waits in the foreground, and stops."""
    stack = _FakeStack()
    opened = _install_stack(monkeypatch, stack)

    code = main(["stack", "stac-serve", str(tmp_path), "--port", "0"])

    assert code == 0
    assert opened == [str(tmp_path)]
    assert stack.serve_calls == [
        {"host": "127.0.0.1", "port": 0, "persist_catalog": True}
    ]
    assert stack.service.wait_calls == 1
    assert stack.service.stop_calls == 1
    # The service prints the actual addresses once; the CLI adds no line.
    assert capsys.readouterr().out == ""


def test_stac_serve_no_persist_catalog_is_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--no-persist-catalog`` selects the read-only workspace path."""
    stack = _FakeStack()
    _install_stack(monkeypatch, stack)

    code = main(
        [
            "stack",
            "stac-serve",
            str(tmp_path),
            "--host",
            "127.0.0.1",
            "--no-persist-catalog",
        ]
    )

    assert code == 0
    assert stack.serve_calls == [
        {"host": "127.0.0.1", "port": 8000, "persist_catalog": False}
    ]


def test_stac_serve_ctrl_c_stops_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupting the foreground wait performs a clean shutdown."""
    service = _FakeService(wait_error=KeyboardInterrupt())
    stack = _FakeStack(service=service)
    _install_stack(monkeypatch, stack)

    code = main(["stack", "stac-serve", str(tmp_path)])

    assert code == 0
    assert service.wait_calls == 1
    assert service.stop_calls == 1


def test_stac_serve_broken_runtime_exits_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing serving runtime is a usage/optional-runtime failure."""
    stack = _FakeStack(
        start_error=ImportError("STAC serving requires the aiohttp package")
    )
    _install_stack(monkeypatch, stack)

    code = main(["stack", "stac-serve", str(tmp_path)])

    assert code == 2
    assert "aiohttp" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error",
    [OSError("address already in use"), RuntimeError("startup failed")],
)
def test_stac_serve_startup_failure_exits_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    """Binding or startup failures return 1 and do not wait."""
    stack = _FakeStack(start_error=error)
    _install_stack(monkeypatch, stack)

    code = main(["stack", "stac-serve", str(tmp_path)])

    assert code == 1
    assert stack.service.wait_calls == 0
    assert "failed to start" in capsys.readouterr().err


def test_stac_serve_unopenable_work_dir_exits_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An invalid work directory fails before any service is started."""

    def fail_open(work_dir: str | Path) -> None:
        message = f"not a Stack work directory: {work_dir}"
        raise ValueError(message)

    monkeypatch.setattr(session_module.Stack, "open", staticmethod(fail_open))

    code = main(["stack", "stac-serve", str(tmp_path)])

    assert code == 1
    assert "cannot open Stack work directory" in capsys.readouterr().err


def test_stac_serve_wait_failure_still_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unexpected serving failure stops the service and returns 1."""
    service = _FakeService(wait_error=RuntimeError("event loop stopped"))
    stack = _FakeStack(service=service)
    _install_stack(monkeypatch, stack)

    code = main(["stack", "stac-serve", str(tmp_path)])

    assert code == 1
    assert service.stop_calls == 1
    assert "STAC service failed" in capsys.readouterr().err


def test_stac_serve_requires_work_dir() -> None:
    """Omitting the work directory is an argparse usage error."""
    with pytest.raises(SystemExit) as excinfo:
        main(["stack", "stac-serve"])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("port", ["-1", "65536", "not-a-port"])
def test_stac_serve_rejects_invalid_port(port: str) -> None:
    """Out-of-range and non-integer ports are usage errors."""
    with pytest.raises(SystemExit) as excinfo:
        main(["stack", "stac-serve", "work", "--port", port])
    assert excinfo.value.code == 2


def test_stac_serve_help_lists_public_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The documented host, port, and persistence flags are discoverable."""
    with pytest.raises(SystemExit) as excinfo:
        main(["stack", "stac-serve", "--help"])
    assert excinfo.value.code == 0
    text = capsys.readouterr().out
    assert "--host" in text
    assert "--port" in text
    assert "--no-persist-catalog" in text
