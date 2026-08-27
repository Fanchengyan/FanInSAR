"""Canonical Network layout and fail-closed construction tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from faninsar.datasets.network import (
    ExternalNetworkLayoutError,
    IncompleteNetworkProductError,
    ISCE2Network,
    Network,
    NetworkGenerationError,
    NetworkManifestError,
    UnknownNetworkIndexTypeError,
)


def _write_layout(
    root: Path,
    *,
    schema_version: str = "network_v1",
    index_type: str = "NetworkInterferogramIndex",
    products: list[dict[str, str]] | None = None,
    source_software: str | None = None,
) -> None:
    """Write the smallest canonical Network root and one generation."""
    generation = "generation-1"
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "status": "complete",
        "generation_id": generation,
        "index_type": index_type,
        "products": (
            products if products is not None else [{"id": "20240101_20240113"}]
        ),
    }
    if source_software is not None:
        payload["source_software"] = source_software
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / "CURRENT").write_text(
        json.dumps({"generation_id": generation}), encoding="utf-8"
    )
    generation_root = root / ".network_generations" / generation
    generation_root.mkdir(parents=True)
    (generation_root / "manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (root / "interferograms" / "20240101_20240113").mkdir(parents=True)
    (root / "interferograms" / "interferograms_index.json").write_text(
        json.dumps({"type": "NetworkInterferogramIndex"}), encoding="utf-8"
    )


def test_network_requires_versioned_manifest_and_generation(tmp_path: Path) -> None:
    """An unmarked directory cannot be interpreted as a Network."""
    with pytest.raises(NetworkManifestError):
        Network(tmp_path)

    root = tmp_path / "network"
    _write_layout(root)
    network = Network(root)
    assert network.manifest["schema_version"] == "network_v1"
    assert network.manifest["generation_id"] == "generation-1"


def test_network_rejects_unknown_manifest_version(tmp_path: Path) -> None:
    """Manifest version changes fail before Frame discovery."""
    root = tmp_path / "network"
    _write_layout(root, schema_version="network_v2")
    with pytest.raises(NetworkManifestError, match="version"):
        Network(root)


def test_network_rejects_empty_and_geometry_only_products(tmp_path: Path) -> None:
    """Empty indexes and geometry-only roots cannot masquerade as Networks."""
    root = tmp_path / "empty"
    _write_layout(root, products=[])
    with pytest.raises(IncompleteNetworkProductError):
        Network(root)

    geometry_only = tmp_path / "geometry-only"
    _write_layout(geometry_only)
    (geometry_only / "interferograms").rename(geometry_only / "removed")
    with pytest.raises(IncompleteNetworkProductError):
        Network(geometry_only)


def test_network_rejects_unknown_index_type(tmp_path: Path) -> None:
    """Only the canonical Network index type is admitted."""
    root = tmp_path / "network"
    _write_layout(root, index_type="FrameInterferogramIndex")
    with pytest.raises(UnknownNetworkIndexTypeError, match="unknown"):
        Network(root)


def test_network_rejects_missing_generation(tmp_path: Path) -> None:
    """A manifest without its immutable generation fails closed."""
    root = tmp_path / "network"
    _write_layout(root)
    generation_root = root / ".network_generations" / "generation-1"
    for path in generation_root.iterdir():
        path.unlink()
    generation_root.rmdir()
    with pytest.raises(NetworkGenerationError):
        Network(root)


def test_external_processor_network_requires_declared_marker(tmp_path: Path) -> None:
    """External processor adapters never discover unrelated layouts."""
    root = tmp_path / "network"
    _write_layout(root)
    with pytest.raises(ExternalNetworkLayoutError, match="source_software"):
        ISCE2Network(root)

    marked = tmp_path / "marked"
    _write_layout(marked, source_software="isce2")
    assert isinstance(ISCE2Network(marked), Network)
