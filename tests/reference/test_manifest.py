"""Frozen scientific reference manifest tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from tests.reference.manifest import ManifestFormatError, load_manifest

ROOT = Path(__file__).parents[2]
MANIFEST_PATH = Path(__file__).with_name("manifest.yaml")


def test_manifest_registers_frozen_corpus_tiers() -> None:
    """Register the committed fixture, regimes, and local execution scenes."""
    manifest = load_manifest(MANIFEST_PATH)

    assert manifest.schema_version == 1
    assert len(manifest.downloadable_regimes) >= 4
    assert len(manifest.local_execution.scenes) == 3
    assert len(manifest.local_execution.pairs) == 3


def test_local_execution_scene_identities_match_available_zip_names() -> None:
    """Keep local scene filenames aligned with the three-scene execution set."""
    manifest = load_manifest(MANIFEST_PATH)

    assert [scene.filename for scene in manifest.local_execution.scenes] == [
        "S1A_IW_SLC__1SSV_20161207T111852_20161207T111919_014273_01716D_67BC.zip",
        "S1A_IW_SLC__1SSV_20161231T111851_20161231T111918_014623_017C5B_9218.zip",
        "S1A_IW_SLC__1SSV_20170124T111849_20170124T111916_014973_018715_4CD9.zip",
    ]


def test_every_downloadable_regime_has_frozen_integrity_metadata() -> None:
    """Require unique integrity and strata fields for every downloadable regime."""
    manifest = load_manifest(MANIFEST_PATH)

    for regime in manifest.downloadable_regimes:
        assert regime.url.startswith("https://")
        assert len(regime.sha256) == 64
        int(regime.sha256, 16)
        assert regime.size_bytes > 0
        assert regime.terrain in {"flat", "steep"}
        assert regime.coherence in {"low", "high"}
        assert regime.baseline in {"short", "long"}

    assert len({regime.identifier for regime in manifest.downloadable_regimes}) == 4
    assert len({regime.url for regime in manifest.downloadable_regimes}) == 4
    assert len({regime.sha256 for regime in manifest.downloadable_regimes}) == 4


def test_tiny_committed_fixture_matches_manifest_checksum() -> None:
    """Verify the committed fixture payload against the frozen checksum."""
    manifest = load_manifest(MANIFEST_PATH)
    fixture_path = ROOT / manifest.committed_fixture.path

    assert fixture_path.stat().st_size == manifest.committed_fixture.size_bytes
    assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == (
        manifest.committed_fixture.sha256
    )
    payload = json.loads(fixture_path.read_text())
    assert payload["coordinate_system"] == "radar"
    assert len(payload["wrapped_phase_radians"]) == 4


def test_malformed_manifest_is_rejected_at_parse_boundary(tmp_path: Path) -> None:
    """Reject malformed YAML shapes at the parse boundary."""
    malformed_path = tmp_path / "manifest.yaml"
    malformed_path.write_text("schema_version: 1\ndownloadable_regimes: nope\n")

    with pytest.raises(ManifestFormatError, match="manifest"):
        load_manifest(malformed_path)


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("duplicate_scene", "duplicate scene"),
        ("duplicate_pair", "duplicate pair"),
        ("duplicate_regime", "duplicate regime"),
        ("dangling_pair", "unknown scene"),
    ],
)
def test_manifest_rejects_duplicate_ids_and_dangling_pairs(
    tmp_path: Path,
    mutation: str,
    expected_message: str,
) -> None:
    """Reject duplicate identifiers and dangling pair references."""
    payload = yaml.safe_load(MANIFEST_PATH.read_text())
    match mutation:
        case "duplicate_scene":
            payload["local_execution"]["scenes"][1]["identifier"] = payload[
                "local_execution"
            ]["scenes"][0]["identifier"]
        case "duplicate_pair":
            payload["local_execution"]["pairs"][1]["identifier"] = payload[
                "local_execution"
            ]["pairs"][0]["identifier"]
        case "duplicate_regime":
            payload["downloadable_regimes"][1]["identifier"] = payload[
                "downloadable_regimes"
            ][0]["identifier"]
        case "dangling_pair":
            payload["local_execution"]["pairs"][0]["secondary"] = "missing-scene"
        case unreachable:
            raise AssertionError(unreachable)
    invalid_path = tmp_path / "manifest.yaml"
    invalid_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ManifestFormatError, match=expected_message):
        load_manifest(invalid_path)


def test_manifest_declares_velocity_bias_and_uncertainty_coverage_gates() -> None:
    """Freeze velocity-bias and 95% uncertainty-coverage gates in the manifest."""
    gates = load_manifest(MANIFEST_PATH).metric_gates

    assert gates["velocity_bias_millimeters_per_year_max"] >= 0.0
    assert 0.0 < gates["uncertainty_95_coverage_fraction_min"] <= 1.0
