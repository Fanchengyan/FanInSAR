"""Strict original-source qualification tests (PROPOSAL-0088 section 8)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.stack.config import StackConfig
from faninsar.stack.integrity import establish_integrity
from faninsar.stack.preparation import (
    DataPreparationError,
    PreparedInputs,
)
from faninsar.stack.session import Stack
from faninsar.stack.source_integrity import (
    SOURCE_BASELINE_SCHEMA,
    establish_source_integrity,
    source_identity,
    verify_source_integrity,
)


class PairMission(Stack):
    """Local two-date mission whose sources are ordinary files."""

    def _prepare_inputs(self, context: PreparedInputs) -> PreparedInputs:
        sources = {
            target: (Path(path),)
            for target, path in self.request["sources"].items()
            if Path(path).exists()
        }
        return PreparedInputs(sources=sources)


def _safe_tree(root: Path) -> Path:
    """Create a tiny real SAFE directory with non-scientific sidecars."""
    root.mkdir()
    (root / "manifest.safe").write_bytes(b"<safe>manifest</safe>")
    for name in ("annotation", "measurement", "calibration"):
        (root / name).mkdir()
    (root / "annotation" / "a.xml").write_bytes(b"<annotation/>")
    (root / "measurement" / "a.tiff").write_bytes(b"measurement-bytes")
    (root / "calibration" / "a.xml").write_bytes(b"<calibration/>")
    (root / "report.json").write_text("{}", encoding="utf-8")
    (root / "a.SAFE.faninsar.remote.identity.json").write_text(
        "{}", encoding="utf-8"
    )
    return root


def _mutate_same_size(path: Path) -> None:
    """Flip one byte and restore mtime so only a payload reread can detect it."""
    stat = path.stat()
    data = bytearray(path.read_bytes())
    assert data
    data[0] ^= 0xFF
    path.write_bytes(data)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def _work_stack(tmp_path: Path, sources: dict[str, Path]) -> PairMission:
    return PairMission.configure(
        config=StackConfig(work_dir=tmp_path / "work"),
        request={"kind": "fixture", "sources": sources},
    )


def test_upright_file_and_safe_tree_verify_from_a_preestablished_baseline(
    tmp_path: Path,
) -> None:
    root = tmp_path / "integrity" / "sources"
    slc = tmp_path / "scene.h5"
    slc.write_bytes(b"rslc-bytes")
    safe = _safe_tree(tmp_path / "source.SAFE")

    file_baseline = establish_source_integrity(root, slc)
    safe_baseline = establish_source_integrity(root, safe)

    assert file_baseline.write_id == source_identity(slc)
    assert safe_baseline.write_id == source_identity(safe)
    assert "report.json" not in safe_baseline.files
    assert "a.SAFE.faninsar.remote.identity.json" not in safe_baseline.files
    assert set(safe_baseline.files) == {
        "manifest.safe",
        "annotation/a.xml",
        "measurement/a.tiff",
        "calibration/a.xml",
    }

    file_verdict = verify_source_integrity(root, slc, strict=True)
    safe_verdict = verify_source_integrity(root, safe, strict=True)

    assert file_verdict.status == "verified"
    assert safe_verdict.status == "verified"
    assert safe_verdict.checked_files == 4
    assert SOURCE_BASELINE_SCHEMA == (
        "faninsar.integrity.baseline.v1"
    )


def test_strict_without_a_baseline_is_unverifiable_and_never_hashes_current_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "integrity" / "sources"
    slc = tmp_path / "scene.h5"
    slc.write_bytes(b"rslc-bytes")
    safe = _safe_tree(tmp_path / "source.SAFE")

    def unexpected_hash(path: Path) -> tuple[str, int]:
        raise AssertionError(f"strict verification hashed current bytes: {path}")

    monkeypatch.setattr("faninsar.stack.integrity._hash_file", unexpected_hash)
    monkeypatch.setattr(
        "faninsar.stack.source_integrity._fingerprint", unexpected_hash
    )

    for source in (slc, safe):
        verdict = verify_source_integrity(root, source, strict=True)
        assert verdict.status == "integrity_unverifiable"
        assert verdict.checked_files == 0
        assert not list(root.glob("baseline-*.json"))


def test_fast_verification_reads_only_the_cheap_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "integrity" / "sources"
    safe = _safe_tree(tmp_path / "source.SAFE")
    establish_source_integrity(root, safe)

    def unexpected_walk(source: Path) -> dict[str, Path]:
        raise AssertionError(f"fast verification walked {source}")

    def unexpected_hash(path: Path) -> tuple[str, int]:
        raise AssertionError(f"fast verification hashed {path}")

    monkeypatch.setattr("faninsar.stack.source_integrity._scientific_files", unexpected_walk)
    monkeypatch.setattr("faninsar.stack.integrity._fingerprint", unexpected_hash)
    monkeypatch.setattr("faninsar.stack.integrity._hash_file", unexpected_hash)

    verdict = verify_source_integrity(root, safe)

    assert verdict.status == "metadata_verified"
    assert verdict.checked_files == 0
    assert verdict.baseline_write_id == source_identity(safe)


def test_strict_rejects_same_size_local_mutation(tmp_path: Path) -> None:
    root = tmp_path / "integrity" / "sources"
    slc = tmp_path / "scene.h5"
    slc.write_bytes(b"rslc-bytes")
    safe = _safe_tree(tmp_path / "source.SAFE")
    for source in (slc, safe):
        establish_source_integrity(root, source)

    _mutate_same_size(slc)
    _mutate_same_size(safe / "measurement" / "a.tiff")

    file_verdict = verify_source_integrity(root, slc, strict=True)
    safe_verdict = verify_source_integrity(root, safe, strict=True)

    assert file_verdict.status == "mismatch"
    assert safe_verdict.status == "mismatch"
    assert safe_verdict.mismatched == ("measurement/a.tiff",)


def test_mutated_bytes_cannot_be_laundered_into_a_new_same_identity_baseline(
    tmp_path: Path,
) -> None:
    root = tmp_path / "integrity" / "sources"
    safe = _safe_tree(tmp_path / "source.SAFE")
    baseline = establish_source_integrity(root, safe)
    _mutate_same_size(safe / "annotation" / "a.xml")

    with pytest.raises(InvalidProcessingStateError, match="already exists"):
        establish_source_integrity(root, safe)

    assert baseline.baseline_path.is_file()
    assert verify_source_integrity(root, safe, strict=True).status == "mismatch"


def test_safe_tree_rejects_symbolic_links(tmp_path: Path) -> None:
    root = tmp_path / "integrity" / "sources"
    safe = _safe_tree(tmp_path / "source.SAFE")
    outside = tmp_path / "outside.xml"
    outside.write_bytes(b"outside")
    (safe / "annotation" / "escape.xml").symlink_to(outside)

    with pytest.raises(InvalidProcessingStateError, match="symbolic links"):
        establish_source_integrity(root, safe)


def test_provider_digest_is_trusted_for_one_file_but_not_a_manifest_tree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "integrity" / "sources"
    slc = tmp_path / "scene.h5"
    slc.write_bytes(b"provider-verified bytes")
    digest = __import__("hashlib").sha256(slc.read_bytes()).hexdigest()

    verdict = verify_source_integrity(
        root, slc, strict=True, provider_digest=f"sha256:{digest}"
    )
    assert verdict.status == "verified"
    assert verdict.basis == "provider"
    assert not list(root.glob("baseline-*.json"))

    wrong = verify_source_integrity(
        root, slc, strict=True, provider_digest="sha256:" + "0" * 64
    )
    assert wrong.status == "mismatch"

    safe = _safe_tree(tmp_path / "source.SAFE")
    with pytest.raises(ValueError, match="single-file"):
        verify_source_integrity(
            root, safe, strict=True, provider_digest=f"sha256:{digest}"
        )


def test_output_product_baseline_cannot_substitute_for_a_source_baseline(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    source = tmp_path / "scene.h5"
    source.write_bytes(b"source-bytes")
    output = tmp_path / "scene-product"
    output.mkdir()
    (output / "payload.bin").write_bytes(b"output-bytes")
    establish_integrity(
        work / "integrity", output, write_id="01HZZZZZZZZZZZZZZZZZZZZZZZ"
    )

    verdict = verify_source_integrity(
        work / "integrity" / "sources", source, strict=True
    )

    assert verdict.status == "integrity_unverifiable"
    assert not (work / "integrity" / "sources").exists() or not list(
        (work / "integrity" / "sources").glob("baseline-*.json")
    )


def test_fast_preparation_never_runs_strict_source_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "20200101.h5"
    source.write_bytes(b"source")
    stack = _work_stack(tmp_path, {"20200101": source})
    calls: list[Path] = []

    def unexpected(*args: object, **kwargs: object) -> None:
        calls.append(Path(str(args[1])))
        raise AssertionError("fast preparation ran strict source verification")

    monkeypatch.setattr(
        "faninsar.stack.source_integrity.verify_source_integrity", unexpected
    )

    report = stack.prepare_data()

    assert report.complete
    assert calls == []


def test_strict_prepare_and_reopen_require_a_real_source_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {
        "20200101": tmp_path / "20200101.h5",
        "20200113": tmp_path / "20200113.h5",
    }
    for path in sources.values():
        path.write_bytes(b"source")
    stack = _work_stack(tmp_path, sources)
    assert stack.prepare_data().complete

    stack.verification = "strict"
    with pytest.raises(InvalidProcessingStateError, match="strict source"):
        stack._ensure_prepared()

    root = stack.config.work_dir / "integrity" / "sources"
    for path in sources.values():
        establish_source_integrity(root, path)
    stack._ensure_prepared()

    monkeypatch.setattr(
        "faninsar.stack.session._mission_class", lambda mission: PairMission
    )
    opened = PairMission.open(stack.config.work_dir, verification="strict")
    assert opened.data_ready
    assert opened.preparation_report is not None
    assert opened.preparation_report.complete


def test_strict_warn_continues_verified_dates_and_raise_reports_truthfully(
    tmp_path: Path,
) -> None:
    sources = {
        "20200101": tmp_path / "20200101.h5",
        "20200113": tmp_path / "20200113.h5",
    }
    for path in sources.values():
        path.write_bytes(b"source")
    stack = _work_stack(tmp_path, sources)
    assert stack.prepare_data().complete
    root = stack.config.work_dir / "integrity" / "sources"
    for path in sources.values():
        establish_source_integrity(root, path)
    stack.verification = "strict"

    _mutate_same_size(sources["20200113"])
    report = stack.prepare_data()

    assert report.outcome == "partial"
    assert report.ready == ("20200101",)
    assert [failure.target for failure in report.failed] == ["20200113"]
    assert report.failed[0].phase == "source_integrity"
    assert stack.data_ready
    stack._ensure_prepared()

    with pytest.raises(DataPreparationError) as raised:
        stack.prepare_data(on_error="raise")
    assert raised.value.report.ready == ("20200101",)


def test_unverified_reference_blocks_consumers_with_a_useful_reason(
    tmp_path: Path,
) -> None:
    sources = {
        "20200101": tmp_path / "20200101.h5",
        "20200113": tmp_path / "20200113.h5",
    }
    for path in sources.values():
        path.write_bytes(b"source")
    stack = _work_stack(tmp_path, sources)
    assert stack.prepare_data().complete
    root = stack.config.work_dir / "integrity" / "sources"
    for path in sources.values():
        establish_source_integrity(root, path)
    stack.verification = "strict"
    for baseline in root.glob("baseline-*.json"):
        if source_identity(sources["20200101"]) in baseline.read_text():
            baseline.unlink()

    report = stack.prepare_data()
    assert report.outcome == "partial"
    assert any("20200101" in item for item in report.prerequisites)
    with pytest.raises(InvalidProcessingStateError, match="reference 20200101"):
        stack._ensure_prepared()


def test_strict_reopen_marks_unbaselined_sources_unready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {"20200101": tmp_path / "20200101.h5"}
    sources["20200101"].write_bytes(b"source")
    stack = _work_stack(tmp_path, sources)
    assert stack.prepare_data().complete

    monkeypatch.setattr(
        "faninsar.stack.session._mission_class", lambda mission: PairMission
    )
    opened = PairMission.open(stack.config.work_dir, verification="strict")

    assert not opened.data_ready
    assert opened.preparation_report is not None
    assert opened.preparation_report.outcome == "failed"
    assert opened.preparation_report.failed[0].phase == "source_integrity"


def test_reuse_scene_checks_the_source_before_any_cached_output(
    tmp_path: Path,
) -> None:
    sources = {"20200101": tmp_path / "20200101.h5"}
    sources["20200101"].write_bytes(b"source")
    stack = _work_stack(tmp_path, sources)
    assert stack.prepare_data().complete
    stack.verification = "strict"

    with pytest.raises(InvalidProcessingStateError, match="cached scene reuse"):
        stack._reuse_scene("20200101", "coreg-identity")


def test_fresh_provider_receipt_establishes_one_strict_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fresh.h5"
    source.write_bytes(b"fresh-provider-verified-bytes")
    digest = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    receipt = {
        "identity": "provider-identity",
        "sha256": digest,
        "size": source.stat().st_size,
        "verification": {"status": "verified", "size": source.stat().st_size},
    }
    monkeypatch.setattr(
        "faninsar.remote.fetch.complete.qualified_download_receipt",
        lambda path: receipt,
    )
    stack = _work_stack(tmp_path, {"20200101": source})
    stack.verification = "strict"

    report = stack.prepare_data()

    assert report.complete
    assert (
        verify_source_integrity(
            stack.config.work_dir / "integrity" / "sources",
            source,
            strict=True,
        ).status
        == "verified"
    )
