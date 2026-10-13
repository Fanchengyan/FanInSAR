"""Tests for the mission-neutral Stack registration values."""

from __future__ import annotations

import json

import pytest

from faninsar.stack.registration import (
    CampaignSelection,
    CoregistrationReport,
    RegistrationCapability,
    RegistrationCorrection,
    identity_digest,
)


def test_campaign_selection_is_one_reusable_identity() -> None:
    """Selection identity includes scenes, grid, frame, units, provenance."""
    selection = CampaignSelection(
        scenes=("20240101", "20240113"),
        reference="20240101",
        grid="radar-grid-v1",
        frame="slant-range",
        units="pixel",
        provenance={"orbit": "precise"},
    )

    assert selection.as_dict() == {
        "scenes": ["20240101", "20240113"],
        "reference": "20240101",
        "grid": "radar-grid-v1",
        "frame": "slant-range",
        "units": "pixel",
        "provenance": {"orbit": "precise"},
    }
    with pytest.raises(ValueError, match="reference"):
        CampaignSelection(("20240101",), "20240113", "radar")


def test_run_identity_changes_when_grid_frame_units_or_provenance_change() -> None:
    """A correction cannot be reused across a changed scientific context."""
    base = {
        "scenes": ["20240101", "20240113"],
        "workflow": "sentinel-1/burst/radar/nesd",
        "grid": "grid-a",
        "frame": "slant-range",
        "units": "pixel",
        "provenance": {"orbit": "precise"},
    }
    baseline = identity_digest(base)
    for key, value in (
        ("grid", "grid-b"),
        ("frame", "geographic"),
        ("units", "metre"),
        ("provenance", {"orbit": "restituted"}),
    ):
        changed = dict(base)
        changed[key] = value
        assert identity_digest(changed) != baseline


def test_report_is_data_only_and_preserves_omitted_axes() -> None:
    """Report details retain an omitted axis instead of inventing numeric zero."""
    capability = RegistrationCapability("sentinel-1", "burst", "radar", "nesd")
    correction = RegistrationCorrection(
        kind="date-constant",
        units="pixel",
        azimuth={"20240113": 0.2},
        range={},
    )
    report = CoregistrationReport(
        capability=capability,
        scenes=("20240101", "20240113"),
        reference="20240101",
        grid="radar",
        frame="slant-range",
        units="pixel",
        status="applied",
        run_identity="abc",
        correction_kind=correction.kind,
        details={"azimuth": dict(correction.azimuth), "range": None},
    )

    payload = report.as_dict()
    assert payload["details"]["range"] is None
    json.dumps(payload, allow_nan=False)
