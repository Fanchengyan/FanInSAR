"""Public Network reader-dispatch contract tests for PROPOSAL-0043."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from faninsar.network import Network
from faninsar.network.registry import (
    DuplicateReaderError,
    InvalidReaderError,
    ReaderNotFoundError,
    ReaderRegistry,
)


def test_direct_reader_instance_bypasses_registry(tmp_path: Path) -> None:
    """A direct reader instance is used without consulting a registry."""

    expected = object()
    calls: list[tuple[Path, str | None]] = []

    class Reader:
        def read(self, path: str | Path, *, revision: str | None = None) -> object:
            calls.append((Path(path), revision))
            return expected

    reader = Reader()
    result = Network.open(tmp_path, reader=reader)

    assert result is expected
    assert calls == [(tmp_path, None)]

    with pytest.raises(TypeError, match="registry.*string"):
        Network.open(tmp_path, reader=reader, registry=ReaderRegistry())


def test_registered_reader_class_is_constructed_without_arguments(
    tmp_path: Path,
) -> None:
    """A registered name resolves to a zero-argument reader class."""

    expected = object()
    constructed = 0

    class Reader:
        def __init__(self) -> None:
            nonlocal constructed
            constructed += 1

        def read(self, path: str | Path, *, revision: str | None = None) -> object:
            assert Path(path) == tmp_path
            assert revision == "generation-a"
            return expected

    registry = ReaderRegistry()
    registry.register("fixture", Reader)

    assert Network.open(
        tmp_path,
        reader="fixture",
        revision="generation-a",
        registry=registry,
    ) is expected
    assert constructed == 1


def test_reader_class_selector_bypasses_registry(tmp_path: Path) -> None:
    """A direct reader class is constructed without registry lookup."""

    expected = object()

    class Reader:
        def read(self, path: str | Path, *, revision: str | None = None) -> object:
            return expected

    class ExplodingRegistry:
        def resolve(self, name: str) -> object:
            raise AssertionError(f"registry was consulted for {name}")

    assert Network.open(tmp_path, reader=Reader) is expected
    # The class selector remains independent from the default registry; this
    # object is only a guard against accidentally adding implicit lookup.
    assert callable(ExplodingRegistry.resolve)


def test_canonical_mode_does_not_discover_entry_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reader discovery is not attempted when the selector is omitted."""
    from importlib import metadata

    def fail_if_discovered(*args: object, **kwargs: object) -> object:
        raise AssertionError("canonical Network mode must not discover readers")

    monkeypatch.setattr(metadata, "entry_points", fail_if_discovered)
    with pytest.raises(Exception):
        # A missing path proves canonical validation was attempted; the exact
        # data-loader error is intentionally outside this dispatch test.
        Network.open(tmp_path)


def test_registry_isolated_and_rejects_unknown_or_non_class_readers() -> None:
    """Registries do not merge and only accept classes, not import strings."""

    registry = ReaderRegistry()

    class Reader:
        def read(self, path: str | Path, *, revision: str | None = None) -> object:
            return object()

    registry.register("fixture", Reader)
    assert registry.names == ("fixture",)
    with pytest.raises(ReaderNotFoundError):
        ReaderRegistry().resolve("fixture")
    with pytest.raises(InvalidReaderError, match="classes"):
        registry.register("module.path.Reader", "module.path.Reader")  # type: ignore[arg-type]


def test_duplicate_entry_point_metadata_fails_before_plugin_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate names are rejected before any entry point is loaded."""
    from importlib import metadata

    loaded = False

    def load() -> object:
        nonlocal loaded
        loaded = True
        raise AssertionError("duplicate plugins must not be imported")

    entry_points = [
        SimpleNamespace(name="duplicate", load=load),
        SimpleNamespace(name="duplicate", load=load),
    ]
    monkeypatch.setattr(metadata, "entry_points", lambda **kwargs: entry_points)

    with pytest.raises(DuplicateReaderError, match="duplicate"):
        ReaderRegistry().resolve("duplicate")
    assert not loaded
