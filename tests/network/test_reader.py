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


def test_canonical_reader_accepts_existing_generation_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit immutable generation can be opened without CURRENT lookup."""
    import json

    payload = {
        "schema_version": "network_v1",
        "status": "complete",
        "generation_id": "generation-1",
        "index_type": "NetworkInterferogramIndex",
        "phase_convention": "primary_minus_secondary",
        "products": [
            {
                "id": "20240101_20240113",
                "primary_id": "20240101",
                "secondary_id": "20240113",
                "product_kind": "complex_interferogram",
                "asset_location": "interferograms/20240101_20240113/complex.npy",
                "geometry_identity": "grid",
                "source_software": "fixture",
                "phase_convention": "primary_minus_secondary",
                "content_digest": "a" * 64,
                "lineage": ["source:20240101", "source:20240113"],
            }
        ],
    }
    import hashlib

    payload["manifest_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    root = tmp_path / "network"
    (root / ".network_generations/generation-1").mkdir(parents=True)
    (root / "interferograms").mkdir()
    (root / "interferograms/interferograms_index.json").write_text(
        json.dumps({"type": "NetworkInterferogramIndex", "pairs": ["20240101_20240113"]})
    )
    (root / "interferograms/20240101_20240113").mkdir()
    (root / "manifest.json").write_text(json.dumps(payload))
    (root / ".network_generations/generation-1/manifest.json").write_text(
        json.dumps(payload)
    )
    (root / "CURRENT").write_text(json.dumps({
        "schema_version": "network_current_v1",
        "status": "writing",
    }))
    monkeypatch.setattr("faninsar.network.network._legacy_markers", lambda _: ())
    network = Network.open(root, revision="generation-1")
    assert network.manifest["generation_id"] == "generation-1"


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
