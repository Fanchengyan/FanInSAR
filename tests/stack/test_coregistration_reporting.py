"""Durable coregistration diagnostics (PROPOSAL-0088)."""

import json

import pytest

from faninsar.stack import reporting
from faninsar.stack.reporting import ReportWriteError, ReportWriter


def _writer(tmp_path, **overrides):
    values = {
        "intent_revision": 7,
        "mode": "nesd",
        "grid": "radar",
        "reference": "20200101",
        "verification": "fast",
    }
    values.update(overrides)
    return ReportWriter(tmp_path / "work", **values)


def test_success_reuse_failure_and_blocked_records_finish_partial(tmp_path):
    writer = _writer(tmp_path)
    completed = writer.record(
        stage="prepare",
        unit_id="acq:20200101",
        status="completed",
        acquisition="20200101",
        output="inputs/safe/20200101",
    )
    reused = writer.record(
        stage="geometry",
        unit_id="unit:20200101:iw1",
        status="reused",
        date="20200101",
        burst=1,
        output="units/20200101-iw1",
    )
    failed = writer.record(
        stage="coregistration",
        unit_id="pair:20200101_20200113",
        status="failed",
        pair=("20200101", "20200113"),
        reason_code="no_overlap",
        message="no overlap window",
        dependency="unit:20200113",
        retry="add support or exclude the pair",
    )
    blocked = writer.record(
        stage="coregistration",
        unit_id="unit:20200125",
        status="blocked",
        date="20200125",
        reason_code="missing_prerequisite",
        dependency="safe:20200125",
    )

    report = writer.finish()

    assert report.outcome == "partial"
    assert [record.status for record in report.records] == [
        "completed",
        "reused",
        "failed",
        "blocked",
    ]
    assert report.records == (completed, reused, failed, blocked)
    assert report.counts == {
        "completed": 1,
        "reused": 1,
        "failed": 1,
        "blocked": 1,
    }
    assert report.stage_counts["coregistration"] == {"failed": 1, "blocked": 1}
    assert writer.report == report
    assert writer.path.is_file()
    assert writer.markdown_path.is_file()

    document = json.loads(writer.path.read_text(encoding="utf-8"))
    assert document["schema"] == reporting.REPORT_SCHEMA
    assert document["attempt_id"] == report.attempt_id
    assert document["outcome"] == "partial"
    assert document["counts"]["failed"] == 1
    assert document["records"][2]["reason_code"] == "no_overlap"
    assert document["records"][2]["pair"] == ["20200101", "20200113"]
    assert document["finished_at"] is not None

    markdown = writer.markdown_path.read_text(encoding="utf-8")
    assert "partial" in markdown
    assert "pair:20200101_20200113" in markdown
    assert "no_overlap" in markdown
    assert "coregistration" in markdown


def test_blocked_only_attempt_finishes_blocked(tmp_path):
    writer = _writer(tmp_path)
    writer.record(
        stage="coregistration",
        unit_id="unit:20200101",
        status="blocked",
        reason_code="missing",
    )
    assert writer.finish().outcome == "blocked"


def test_latest_roundtrip_returns_the_finished_report(tmp_path):
    writer = _writer(tmp_path, reference=None)
    writer.record(stage="coregistration", unit_id="unit:a", status="completed")
    finished = writer.finish(message="done")

    loaded = ReportWriter.load_latest(tmp_path / "work")

    assert loaded is not None
    assert loaded == finished
    assert loaded.to_dict() == finished.to_dict()
    assert loaded.reference is None
    assert loaded.current_outcome == "complete"
    assert loaded.interrupted is False
    assert ReportWriter.load_latest(tmp_path / "empty-work") is None


def test_failed_then_repaired_attempt_keeps_historic_failure(tmp_path):
    first = _writer(tmp_path)
    first.record(
        stage="coregistration",
        unit_id="unit:a",
        status="failed",
        reason_code="broken",
    )
    failed_report = first.finish()

    second = _writer(tmp_path)
    second.record(stage="coregistration", unit_id="unit:a", status="completed")
    repaired = second.finish()

    loaded = ReportWriter.load_latest(tmp_path / "work")
    assert loaded is not None
    assert loaded.attempt_id == repaired.attempt_id
    assert loaded.outcome == "complete"
    assert loaded.current_outcome == "complete"

    historic = json.loads(first.path.read_text(encoding="utf-8"))
    assert historic["attempt_id"] == failed_report.attempt_id
    assert historic["outcome"] == "failed"
    assert first.path.is_file()


def test_unfinished_attempt_stays_running_without_fabricating_a_lease(tmp_path):
    writer = _writer(tmp_path)
    writer.record(stage="coregistration", unit_id="unit:a", status="completed")

    assert writer.report.outcome == "running"
    assert writer.report.unfinished is True
    assert writer.report.interrupted is False
    assert writer.report.current_outcome == "running"

    loaded = ReportWriter.load_latest(tmp_path / "work")
    assert loaded is not None
    assert loaded.outcome == "running"
    assert loaded.finished_at is None
    assert loaded.unfinished is True
    assert loaded.interrupted is False
    assert loaded.current_outcome == "running"

    reconciled = ReportWriter.load_latest(tmp_path / "work", reconcile_interrupted=True)
    assert reconciled is not None
    assert reconciled.outcome == "running"
    assert reconciled.finished_at is None
    assert reconciled.interrupted is True
    assert reconciled.current_outcome == "interrupted"

    attempt_files = sorted(item.name for item in writer.path.parent.iterdir())
    assert attempt_files == ["report.json", "report.md"]
    assert not (tmp_path / "work" / "reports" / "coregistration" / ".lock").exists()


def test_redacts_credentials_signed_urls_and_exception_text(tmp_path):
    writer = _writer(tmp_path)
    writer.record(
        stage="coregistration",
        unit_id="unit:a",
        status="failed",
        message=RuntimeError(
            "GET https://user:password@example.invalid/scene?token=alpha"
            "&X-Amz-Signature=deadbeef failed with Authorization: Bearer topsecret"
        ),
        quality={
            "coherence": 0.5,
            "password": "hunter2",
            "nested": {"api_key": "key-123", "href": "https://h/x?access_token=beta"},
        },
        output="https://user:pass@example.invalid/out?token=gamma",
        log="logs/coregistration.log",
    )

    text = writer.path.read_text(encoding="utf-8")
    for secret in (
        "password",
        "alpha",
        "deadbeef",
        "topsecret",
        "hunter2",
        "key-123",
        "beta",
        "user:pass",
        "token=gamma",
        "Authorization",
        "Bearer",
    ):
        assert secret not in text
    assert "0.5" in text
    assert "logs/coregistration.log" in text
    assert "[REDACTED]" in text


def test_json_write_failure_preserves_authority_snapshot_and_memory(
    tmp_path, monkeypatch
):
    writer = _writer(tmp_path)
    writer.record(stage="geometry", unit_id="unit:a", status="completed")
    prior_json = writer.path.read_bytes()
    prior_markdown = writer.markdown_path.read_bytes()

    def broken_json(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(reporting, "_write_json_snapshot", broken_json)
    with pytest.raises(ReportWriteError):
        writer.record(stage="geometry", unit_id="unit:b", status="completed")

    assert writer.path.read_bytes() == prior_json
    assert writer.markdown_path.read_bytes() == prior_markdown
    assert [record.unit_id for record in writer.report.records] == ["unit:a"]
    assert writer.markdown_error is None


def test_markdown_failure_is_truthful_after_json_authority_is_saved(
    tmp_path, monkeypatch
):
    writer = _writer(tmp_path)
    writer.record(stage="geometry", unit_id="unit:a", status="completed")
    prior_markdown = writer.markdown_path.read_bytes()

    def broken_markdown(*args, **kwargs):
        raise OSError("markdown disk full token=leaked-secret")

    monkeypatch.setattr(reporting, "_write_markdown_snapshot", broken_markdown)
    writer.record(stage="geometry", unit_id="unit:b", status="completed")

    document = json.loads(writer.path.read_text(encoding="utf-8"))
    assert [record["unit_id"] for record in document["records"]] == ["unit:a", "unit:b"]
    assert [record.unit_id for record in writer.report.records] == ["unit:a", "unit:b"]
    assert writer.markdown_path.read_bytes() == prior_markdown
    assert writer.markdown_error is not None
    assert "markdown disk full" in writer.markdown_error
    assert "leaked-secret" not in writer.markdown_error


def test_report_io_failure_at_start_is_truthful(tmp_path):
    blocked_root = tmp_path / "work" / "reports"
    blocked_root.parent.mkdir(parents=True)
    blocked_root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ReportWriteError):
        _writer(tmp_path)


def test_record_message_is_frozen_and_history_cannot_be_rewritten(tmp_path):
    writer = _writer(tmp_path)
    record = writer.record(
        stage="coregistration",
        unit_id="unit:a",
        status="failed",
        message={"detail": "original"},
    )
    published = json.loads(writer.path.read_text(encoding="utf-8"))
    assert published["records"][0]["message"] == {"detail": "original"}

    with pytest.raises(TypeError):
        record.message["detail"] = "MUTATED-AFTER-PUBLISH"

    writer.record(stage="coregistration", unit_id="unit:b", status="completed")
    republished = json.loads(writer.path.read_text(encoding="utf-8"))
    assert republished["records"][0]["message"] == {"detail": "original"}
    assert record.message == {"detail": "original"}


def test_finish_message_is_frozen_for_the_report_and_disk_snapshot(tmp_path):
    writer = _writer(tmp_path)
    writer.record(stage="coregistration", unit_id="unit:a", status="completed")
    report = writer.finish(message={"note": "original"})

    with pytest.raises(TypeError):
        report.message["note"] = "MUTATED-AFTER-FINISH"

    assert json.loads(writer.path.read_text(encoding="utf-8"))["message"] == {
        "note": "original"
    }
    assert report.message == {"note": "original"}


def test_load_latest_ignores_invalid_and_mismatched_attempt_directories(tmp_path):
    writer = _writer(tmp_path)
    writer.record(stage="coregistration", unit_id="unit:a", status="completed")
    finished = writer.finish()
    root = writer.path.parent.parent

    junk = root / "zzzz-junk"
    junk.mkdir(parents=True)
    (junk / "report.json").write_text(
        json.dumps(
            reporting.CoregistrationReport(
                attempt_id="zzzz-junk",
                intent_revision=9,
                mode="forged",
                grid="forged",
                reference=None,
                verification="fast",
                created_at="2020-01-01T00:00:00Z",
                updated_at="2020-01-01T00:00:00Z",
                outcome="failed",
                finished_at="2020-01-01T00:00:01Z",
            ).to_dict()
        ),
        encoding="utf-8",
    )

    misplaced = root / "29991231T235959999999Z-deadbeef"
    misplaced.mkdir(parents=True)
    (misplaced / "report.json").write_text(
        json.dumps(
            reporting.CoregistrationReport(
                attempt_id="20200101T000000000000Z-cafebabe",
                intent_revision=9,
                mode="forged",
                grid="forged",
                reference=None,
                verification="fast",
                created_at="2020-01-01T00:00:00Z",
                updated_at="2020-01-01T00:00:00Z",
                outcome="failed",
                finished_at="2020-01-01T00:00:01Z",
            ).to_dict()
        ),
        encoding="utf-8",
    )

    loaded = ReportWriter.load_latest(tmp_path / "work")

    assert loaded is not None
    assert loaded.attempt_id == finished.attempt_id


def test_empty_attempt_finishes_without_claiming_success(tmp_path):
    writer = _writer(tmp_path)

    report = writer.finish()

    assert report.outcome == "no_work"
    assert json.loads(writer.path.read_text(encoding="utf-8"))["outcome"] == "no_work"
    assert "no_work" in writer.markdown_path.read_text(encoding="utf-8")


def test_all_skipped_attempt_finishes_skipped(tmp_path):
    writer = _writer(tmp_path)
    writer.record(
        stage="coregistration",
        unit_id="unit:a",
        status="skipped",
        reason_code="roi",
    )
    writer.record(
        stage="coregistration",
        unit_id="unit:b",
        status="skipped",
        reason_code="roi",
    )

    assert writer.finish().outcome == "skipped"


def test_mixed_skipped_disposition_is_reported_as_partial(tmp_path):
    writer = _writer(tmp_path)
    writer.record(stage="coregistration", unit_id="unit:a", status="completed")
    writer.record(
        stage="coregistration",
        unit_id="unit:b",
        status="skipped",
        reason_code="roi",
    )

    report = writer.finish()

    assert report.outcome == "partial"
    assert report.counts == {"completed": 1, "skipped": 1}


def test_context_fields_are_sanitized_before_publication(tmp_path):
    writer = _writer(
        tmp_path,
        mode="nesd?token=mode-secret",
        grid="https://user:pass@example.invalid/grid?api_key=grid-secret",
        reference="https://user:pass@example.invalid/ref?token=ref-secret",
        verification="fast & signature=verify-secret",
    )
    writer.record(stage="coregistration", unit_id="unit:a", status="completed")

    document = writer.path.read_text(encoding="utf-8")
    markdown = writer.markdown_path.read_text(encoding="utf-8")
    for rendered in (document, markdown):
        for secret in (
            "mode-secret",
            "user:pass",
            "grid-secret",
            "ref-secret",
            "verify-secret",
            "api_key",
            "token=",
        ):
            assert secret not in rendered

    assert writer.report.mode == "nesd?[REDACTED]"
    assert writer.report.reference == "https://example.invalid/ref"
