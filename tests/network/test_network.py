"""Canonical Network layout and fail-closed construction tests."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from faninsar.network import (
    IncompleteNetworkProductError,
    LegacyNetworkLayoutError,
    Network,
    NetworkCurrentError,
    NetworkGenerationError,
    NetworkManifestError,
    UnknownNetworkIndexTypeError,
)


def _refresh_manifest_digest(payload: dict[str, Any]) -> None:
    """Update the test manifest's canonical self-digest after a mutation."""
    payload["manifest_digest"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "manifest_digest"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _write_layout(
    root: Path,
    *,
    schema_version: str = "network_v1",
    index_type: str = "NetworkInterferogramIndex",
    products: list[dict[str, Any]] | None = None,
    source_software: str | None = None,
) -> None:
    """Write the smallest canonical Network root and one generation."""
    generation = "generation-1"
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "status": "complete",
        "generation_id": generation,
        "index_type": index_type,
        "phase_convention": "primary_minus_secondary",
        "products": (
            products
            if products is not None
            else [
                {
                    "id": "20240101_20240113",
                    "primary_id": "20240101",
                    "secondary_id": "20240113",
                    "product_kind": "complex_interferogram",
                    "asset_location": "interferograms/20240101_20240113/complex.npy",
                    "geometry_identity": "grid-1",
                    "source_software": "faninsar",
                    "phase_convention": "primary_minus_secondary",
                    "content_digest": "a" * 64,
                    "lineage": ["source:20240101", "source:20240113"],
                }
            ]
        ),
    }
    if source_software is not None:
        payload["source_software"] = source_software
    _refresh_manifest_digest(payload)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / "CURRENT").write_text(
        json.dumps(
            {
                "schema_version": "network_current_v1",
                "status": "complete",
                "generation_id": generation,
                "manifest_digest": payload["manifest_digest"],
            }
        ),
        encoding="utf-8",
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
    assert all(base.__name__ != "Frame" for base in Network.__mro__)


def test_network_rejects_legacy_layout_with_canonical_error(tmp_path: Path) -> None:
    """Legacy interferogram markers fail with the canonical error type."""
    root = tmp_path / "legacy"
    (root / "ifg").mkdir(parents=True)
    with pytest.raises(LegacyNetworkLayoutError):
        Network(root)


def test_network_public_surface_excludes_retired_aliases_and_adapters() -> None:
    """Retired aliases and unimplemented processor adapters are not exported."""
    import faninsar.network as public

    retired = {
        "ExternalNetworkLayoutError",
        "GAMMANetwork",
        "ISCE2Network",
        "ISCE3Network",
        "IncompleteNetworkError",
        "LegacyLayoutError",
        "GMTSARNetwork",
        "NetworkLayoutError",
        "SNAPNetwork",
    }
    assert retired.isdisjoint(public.__all__)
    assert all(not hasattr(public, name) for name in retired)


def test_network_rejects_unknown_manifest_version(tmp_path: Path) -> None:
    """Manifest version changes fail before product discovery."""
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


def test_network_revision_requires_generation_local_interferograms(
    tmp_path: Path,
) -> None:
    """An explicit revision cannot fall back to mutable root assets."""
    root = tmp_path / "network"
    _write_layout(root)
    generation_interferograms = (
        root / ".network_generations" / "generation-1" / "interferograms"
    )
    generation_interferograms.mkdir()
    (generation_interferograms / "interferograms_index.json").write_text(
        json.dumps({"type": "NetworkInterferogramIndex"}), encoding="utf-8"
    )
    for path in generation_interferograms.iterdir():
        path.unlink()
    generation_interferograms.rmdir()
    with pytest.raises(IncompleteNetworkProductError, match="selected Network"):
        Network.open(root, revision="generation-1")


@pytest.mark.parametrize(
    "current_update",
    [
        {"schema_version": "network_v0"},
        {"schema_version": "network_current_v1", "status": "writing"},
        {"schema_version": "network_current_v1", "status": "complete"},
    ],
)
def test_network_rejects_unversioned_or_incomplete_current(
    tmp_path: Path, current_update: dict[str, str]
) -> None:
    """CURRENT is a versioned complete pointer, not a bare generation ID."""
    root = tmp_path / "network"
    _write_layout(root)
    (root / "CURRENT").write_text(json.dumps(current_update), encoding="utf-8")
    with pytest.raises(NetworkGenerationError, match="CURRENT"):
        Network(root)


def test_network_rejects_incomplete_product_record(tmp_path: Path) -> None:
    """A product record must carry all stable identity and asset fields."""
    root = tmp_path / "network"
    _write_layout(root, products=[{"id": "20240101_20240113"}])
    with pytest.raises(NetworkManifestError, match="primary_id"):
        Network(root)


def test_network_rejects_generation_product_set_mismatch(tmp_path: Path) -> None:
    """The root index and immutable generation must describe one product set."""
    root = tmp_path / "network"
    _write_layout(root)
    generation_manifest = (
        root / ".network_generations" / "generation-1" / "manifest.json"
    )
    payload = json.loads(generation_manifest.read_text())
    payload["products"][0]["id"] = "20240201_20240213"
    _refresh_manifest_digest(payload)
    generation_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NetworkGenerationError, match="product set"):
        Network(root)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("content_digest", None, "content_digest"),
        ("lineage", [], "lineage"),
        ("asset_location", "/outside/network.npy", "relative path"),
        ("asset_location", "interferograms/../outside.npy", "relative path"),
        ("asset_location", "C:\\outside\\network.npy", "separators"),
    ],
)
def test_network_rejects_invalid_content_lineage_and_asset_location(
    tmp_path: Path, field_name: str, value: object, message: str
) -> None:
    """Content identity is structured and asset paths stay inside the Network."""
    root = tmp_path / "network"
    _write_layout(root)
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["products"][0][field_name] = value
    _refresh_manifest_digest(payload)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NetworkManifestError, match=message):
        Network(root)


def test_network_rejects_manifest_digest_tampering(tmp_path: Path) -> None:
    """Changing a manifest without republishing its digest fails closed."""
    root = tmp_path / "network"
    _write_layout(root)
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["products"][0]["geometry_identity"] = "forged-grid"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NetworkManifestError, match="manifest_digest"):
        Network(root)


def test_network_rejects_forged_current_digest(tmp_path: Path) -> None:
    """CURRENT cannot point at a different manifest generation digest."""
    root = tmp_path / "network"
    _write_layout(root)
    current_path = root / "CURRENT"
    payload = json.loads(current_path.read_text())
    payload["manifest_digest"] = "b" * 64
    current_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NetworkCurrentError, match="manifest_digest"):
        Network(root)
