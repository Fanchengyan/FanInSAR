# ruff: noqa: INP001
"""Durable NESD pair recovery and qualified network solve (PROPOSAL-0088)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from faninsar.processing.coregistration.misreg_network import MisregArc
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.stack.misreg import (
    misreg_arc_from_properties,
    misreg_arc_to_properties,
    read_misreg,
    write_misreg,
)
from faninsar.stack.nesd_recovery import (
    NESD_RECOVERY_REVISION,
    NESDQualityPolicy,
    PairMeasurementSpec,
    PairMeasurementStore,
    measurement_spec,
    solve_misreg,
)

if TYPE_CHECKING:
    from pathlib import Path

REFERENCE = "20160101"
SECOND = "20160113"
THIRD = "20160125"
FOURTH = "20160206"
DATES = (REFERENCE, SECOND, THIRD)


def _arc(
    primary: str,
    secondary: str,
    azimuth: float | None,
    ranges: float | None,
    *,
    azimuth_sigma: float = 0.05,
    range_sigma: float = 0.1,
    n_valid: int = 12,
) -> MisregArc:
    return MisregArc(
        primary=primary,
        secondary=secondary,
        azimuth_shift_px=azimuth,
        range_shift_px=ranges,
        azimuth_sigma_px=azimuth_sigma,
        range_sigma_px=range_sigma,
        method="overlap",
        n_valid=n_valid,
    )


def _triangle() -> list[MisregArc]:
    """Exact synthetic truth: A=0, B=(2.0, 0.5), C=(-1.0, 0.25)."""
    return [
        _arc(REFERENCE, SECOND, 2.0, 0.5),
        _arc(REFERENCE, THIRD, -1.0, 0.25),
        _arc(SECOND, THIRD, -3.0, -0.25),
    ]


def _spec(primary: str, secondary: str, **kwargs: object) -> PairMeasurementSpec:
    kwargs.setdefault("support", {"bursts": ["IW1_b0"]})
    return measurement_spec(primary, secondary, **kwargs)  # type: ignore[arg-type]


def test_pair_store_reuses_arc_when_catalog_gains_a_date(tmp_path: Path) -> None:
    """A-B survives adding D; only endpoint recipe/support key the arc."""
    store = PairMeasurementStore(tmp_path / "pairs")
    pair_id = f"{REFERENCE}_{SECOND}"
    before = _spec(REFERENCE, SECOND)
    arc = _arc(REFERENCE, SECOND, 2.0, 0.5)

    assert store.load(pair_id, before) is None
    receipt = store.commit(pair_id, before, arc)
    assert receipt.arc == arc
    assert (receipt.path / "item.json").is_file()

    after = _spec(REFERENCE, SECOND)
    assert after.signature == before.signature
    assert "dates" not in before.document()
    assert "reference" not in before.document()
    assert store.load(pair_id, after) == arc

    changed = _spec(REFERENCE, SECOND, support={"bursts": ["IW2_b1"]})
    assert store.load(pair_id, changed) is None
    assert NESD_RECOVERY_REVISION == "nesd-recovery-v2"
    old_revision = _spec(REFERENCE, SECOND, stage_revision="nesd-recovery-v1")
    assert store.load(pair_id, old_revision) is None


def test_pair_store_does_not_rewrite_identical_arc(tmp_path: Path) -> None:
    """Re-committing the same observation keeps its write id and product."""
    store = PairMeasurementStore(tmp_path / "pairs")
    spec = _spec(REFERENCE, SECOND)
    pair_id = spec.pair_id
    arc = _arc(REFERENCE, SECOND, 2.0, 0.5)

    first = store.commit(pair_id, spec, arc)
    again = store.commit(pair_id, spec, arc)
    assert again.write_id == first.write_id

    changed_arc = replace(arc, range_shift_px=0.6)
    changed = store.commit(pair_id, spec, changed_arc)
    assert changed.write_id != first.write_id
    assert store.load(pair_id, spec) == changed_arc


def test_pair_store_rejects_endpoint_mismatch_and_unsafe_pair_id(
    tmp_path: Path,
) -> None:
    """A saved arc can only live under its own canonical pair id."""
    store = PairMeasurementStore(tmp_path / "pairs")
    spec = _spec(REFERENCE, SECOND)
    arc = _arc(REFERENCE, SECOND, 2.0, 0.5)

    with pytest.raises(InvalidProcessingStateError, match="pair_id"):
        store.commit(f"{REFERENCE}_{THIRD}", spec, arc)
    with pytest.raises(InvalidProcessingStateError):
        store.load(f"../{REFERENCE}_{SECOND}", spec)


def test_pair_store_ignores_a_corrupt_product(tmp_path: Path) -> None:
    """Recovery treats an unreadable pair product as absent and remeasures."""
    store = PairMeasurementStore(tmp_path / "pairs")
    spec = _spec(REFERENCE, SECOND)
    receipt = store.commit(spec.pair_id, spec, _arc(REFERENCE, SECOND, 2.0, 0.5))
    (receipt.path / "item.json").write_text("{}", encoding="utf-8")

    assert store.load(spec.pair_id, spec) is None


def test_solve_warns_missing_redundant_edge(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing redundant edge warns while the reference component solves."""
    arcs = [_arc(REFERENCE, SECOND, 2.0, 0.5), _arc(SECOND, THIRD, -3.0, -0.25)]
    planned = [
        (REFERENCE, SECOND),
        (SECOND, THIRD),
        (REFERENCE, THIRD),
    ]

    with caplog.at_level("WARNING"):
        solution = solve_misreg(
            arcs, reference=REFERENCE, dates=DATES, planned_pairs=planned
        )

    assert solution.missing_pairs == (f"{REFERENCE}_{THIRD}",)
    assert any("missing" in record.message.lower() for record in caplog.records)
    assert solution.blocked_dates == ()
    assert solution.date_misreg.azimuth_px[SECOND] == pytest.approx(2.0)
    assert solution.date_misreg.range_px[SECOND] == pytest.approx(0.5)
    assert solution.date_misreg.azimuth_px[THIRD] == pytest.approx(-1.0)
    assert solution.date_misreg.range_px[THIRD] == pytest.approx(0.25)


def test_solve_strict_completeness_rejects_missing_edge() -> None:
    """Strict completeness rejects a planned edge that was never measured."""
    arcs = [_arc(REFERENCE, SECOND, 2.0, 0.5), _arc(SECOND, THIRD, -3.0, -0.25)]
    planned = [
        (REFERENCE, SECOND),
        (SECOND, THIRD),
        (REFERENCE, THIRD),
    ]

    with pytest.raises(InvalidProcessingStateError, match="strict"):
        solve_misreg(
            arcs,
            reference=REFERENCE,
            dates=DATES,
            planned_pairs=planned,
            strict_completeness=True,
        )


def test_solve_salvages_reference_connected_subset_and_blocks_date() -> None:
    """A disconnected published date is blocked, not zeroed or fatal."""
    dates = (*DATES, FOURTH)
    solution = solve_misreg(_triangle(), reference=REFERENCE, dates=dates)

    assert solution.blocked_dates == (FOURTH,)
    assert solution.date_misreg.azimuth_px[SECOND] == pytest.approx(2.0)
    assert solution.date_misreg.range_px[THIRD] == pytest.approx(0.25)
    assert FOURTH not in solution.date_misreg.azimuth_px
    assert FOURTH not in solution.date_misreg.range_px
    assert (FOURTH, "azimuth", "disconnected") in solution.blocked_components
    assert (FOURTH, "range", "disconnected") in solution.blocked_components

    with pytest.raises(InvalidProcessingStateError, match="strict"):
        solve_misreg(
            _triangle(),
            reference=REFERENCE,
            dates=dates,
            strict_completeness=True,
        )


def test_solve_raises_when_reference_has_no_component() -> None:
    """Without a reference-connected axis there is no honest NESD solve."""
    arcs = [_arc(SECOND, THIRD, -3.0, -0.25)]

    with pytest.raises(InvalidProcessingStateError, match="reference-connected"):
        solve_misreg(arcs, reference=REFERENCE, dates=DATES)


def test_solve_requires_explicit_no_overlap_for_omitted_campaign() -> None:
    """All-omitted components need an explicit known no-overlap reason."""
    arcs = [
        _arc(REFERENCE, SECOND, None, None),
        _arc(SECOND, THIRD, None, None),
        _arc(REFERENCE, THIRD, None, None),
    ]

    with pytest.raises(InvalidProcessingStateError, match="no_overlap"):
        solve_misreg(arcs, reference=REFERENCE, dates=DATES)

    solution = solve_misreg(arcs, reference=REFERENCE, dates=DATES, no_overlap=True)
    assert solution.skipped_no_windows is True
    assert solution.date_misreg.azimuth_px == dict.fromkeys(DATES, 0.0)
    assert solution.date_misreg.range_px == dict.fromkeys(DATES, 0.0)
    assert solution.date_misreg.metadata["nesd_recovery"]["no_overlap"] is True


def test_solve_rejects_only_low_valid_evidence() -> None:
    """A single quality-rejected edge cannot become a silent geometry solve."""
    arcs = [_arc(REFERENCE, SECOND, 2.0, 0.5, n_valid=1)]

    with pytest.raises(InvalidProcessingStateError, match="no usable"):
        solve_misreg(
            arcs,
            reference=REFERENCE,
            dates=(REFERENCE, SECOND),
            quality=NESDQualityPolicy(min_n_valid=5),
        )


def test_solve_rejects_only_high_sigma_evidence() -> None:
    """A single over-sigma edge cannot become a silent geometry solve."""
    arcs = [
        _arc(
            REFERENCE,
            SECOND,
            2.0,
            0.5,
            azimuth_sigma=10.0,
            range_sigma=10.0,
        )
    ]

    with pytest.raises(InvalidProcessingStateError, match="no usable"):
        solve_misreg(
            arcs,
            reference=REFERENCE,
            dates=(REFERENCE, SECOND),
            quality=NESDQualityPolicy(
                max_azimuth_sigma_px=1.0,
                max_range_sigma_px=1.0,
            ),
        )


def test_solve_applies_per_axis_quality_and_keeps_other_axis() -> None:
    """A noisy range component is excluded while azimuth stays qualified."""
    arcs = [
        _arc(REFERENCE, SECOND, 2.0, 0.5),
        _arc(REFERENCE, THIRD, -1.0, 0.25),
        _arc(SECOND, THIRD, -3.0, -0.25, range_sigma=10.0),
    ]
    quality = NESDQualityPolicy(max_range_sigma_px=1.0)
    planned = [
        (REFERENCE, SECOND),
        (REFERENCE, THIRD),
        (SECOND, THIRD),
    ]

    solution = solve_misreg(
        arcs,
        reference=REFERENCE,
        dates=DATES,
        planned_pairs=planned,
        quality=quality,
    )

    assert solution.date_misreg.azimuth_px[SECOND] == pytest.approx(2.0)
    assert solution.date_misreg.azimuth_px[THIRD] == pytest.approx(-1.0)
    assert solution.date_misreg.range_px[SECOND] == pytest.approx(0.5)
    assert solution.date_misreg.range_px[THIRD] == pytest.approx(0.25)
    assert (
        f"{SECOND}_{THIRD}",
        "range",
        "sigma_above_max",
    ) in {(item.pair_id, item.component, item.reason) for item in solution.exclusions}

    with pytest.raises(InvalidProcessingStateError, match="strict"):
        solve_misreg(
            arcs,
            reference=REFERENCE,
            dates=DATES,
            planned_pairs=planned,
            quality=quality,
            strict_completeness=True,
        )


def test_solve_keeps_existing_omitted_axis_policy() -> None:
    """One wholly omitted axis keeps the optional solver's geometry zeros."""
    arcs = [
        _arc(REFERENCE, SECOND, 2.0, None),
        _arc(REFERENCE, THIRD, -1.0, None),
        _arc(SECOND, THIRD, -3.0, None),
    ]

    solution = solve_misreg(arcs, reference=REFERENCE, dates=DATES)

    assert solution.date_misreg.azimuth_px[SECOND] == pytest.approx(2.0)
    assert solution.date_misreg.azimuth_px[THIRD] == pytest.approx(-1.0)
    assert solution.date_misreg.range_px == dict.fromkeys(DATES, 0.0)
    assert solution.date_misreg.metadata["nesd_recovery"]["range"]["status"] == (
        "absent"
    )


def test_solve_quality_rejected_azimuth_blocks_dates_despite_valid_range() -> None:
    """A measured quality-rejected azimuth is blocked, never certified as zero."""
    arcs = [
        replace(
            _arc(REFERENCE, SECOND, None, 0.5),
            azimuth_omission="quality_rejected",
        ),
        replace(
            _arc(REFERENCE, THIRD, None, 0.25),
            azimuth_omission="quality_rejected",
        ),
        replace(
            _arc(SECOND, THIRD, None, -0.25),
            azimuth_omission="quality_rejected",
        ),
    ]

    solution = solve_misreg(arcs, reference=REFERENCE, dates=DATES)

    assert solution.date_misreg.azimuth_px == {}
    assert solution.date_misreg.range_px[SECOND] == pytest.approx(0.5)
    assert solution.date_misreg.range_px[THIRD] == pytest.approx(0.25)
    assert solution.blocked_dates == (SECOND, THIRD)
    assert (
        SECOND,
        "azimuth",
        "no_usable_observations",
    ) in solution.blocked_components
    assert (
        f"{REFERENCE}_{SECOND}",
        "azimuth",
        "quality_rejected",
    ) in {(item.pair_id, item.component, item.reason) for item in solution.exclusions}
    recovery = solution.date_misreg.metadata["nesd_recovery"]
    assert recovery["azimuth"]["status"] == "blocked"
    assert recovery["range"]["status"] == "solved"

    with pytest.raises(InvalidProcessingStateError, match="strict"):
        solve_misreg(
            arcs,
            reference=REFERENCE,
            dates=DATES,
            strict_completeness=True,
        )


def test_solve_method_omitted_axis_is_documented_geometry_fallback() -> None:
    """A legitimate method omission stays the optional solver's zero fallback."""
    arcs = [
        replace(
            _arc(REFERENCE, SECOND, 2.0, None),
            range_omission="method_omitted",
        ),
        replace(
            _arc(REFERENCE, THIRD, -1.0, None),
            range_omission="method_omitted",
        ),
        replace(
            _arc(SECOND, THIRD, -3.0, None),
            range_omission="method_omitted",
        ),
    ]

    solution = solve_misreg(arcs, reference=REFERENCE, dates=DATES)

    assert solution.date_misreg.azimuth_px[SECOND] == pytest.approx(2.0)
    assert solution.date_misreg.range_px == dict.fromkeys(DATES, 0.0)
    assert solution.date_misreg.metadata["nesd_recovery"]["range"]["status"] == (
        "absent"
    )
    assert solution.blocked_dates == ()


def test_pair_store_and_aggregate_round_trip_omission_reasons(
    tmp_path: Path,
) -> None:
    """Measured omission reasons survive both persisted product shapes."""
    arc = replace(
        _arc(REFERENCE, SECOND, None, 0.5),
        azimuth_omission="quality_rejected",
    )
    row = misreg_arc_to_properties(arc)
    assert row["azimuth_omission"] == "quality_rejected"
    assert row["range_omission"] is None
    assert misreg_arc_from_properties(row) == arc

    store = PairMeasurementStore(tmp_path / "pairs")
    spec = _spec(REFERENCE, SECOND)
    store.commit(spec.pair_id, spec, arc)
    assert store.load(spec.pair_id, spec) == arc

    root = tmp_path / "misreg"
    write_misreg(
        root,
        arcs=[arc],
        date_misreg=None,
        skipped_no_windows=False,
    )
    record = read_misreg(root)
    assert record is not None
    assert record.arcs == [arc]


def test_solution_signature_binds_omission_reason() -> None:
    """Equal corrections still carry distinct omission provenance."""

    def omitted(reason: str) -> list[MisregArc]:
        return [
            replace(_arc(REFERENCE, SECOND, 2.0, None), range_omission=reason),
            replace(_arc(REFERENCE, THIRD, -1.0, None), range_omission=reason),
            replace(_arc(SECOND, THIRD, -3.0, None), range_omission=reason),
        ]

    method = solve_misreg(omitted("method_omitted"), reference=REFERENCE, dates=DATES)
    no_overlap = solve_misreg(omitted("no_overlap"), reference=REFERENCE, dates=DATES)

    assert method.correction_signature == no_overlap.correction_signature
    assert method.solution_signature != no_overlap.solution_signature


def test_later_measurement_failure_keeps_committed_arcs_and_reuses_them(
    tmp_path: Path,
) -> None:
    """A failed later edge does not discard A-B; retry measures only B-C."""
    store = PairMeasurementStore(tmp_path / "pairs")
    arc_ab = _arc(REFERENCE, SECOND, 2.0, 0.5)
    arc_bc = _arc(SECOND, THIRD, -3.0, -0.25)
    spec_ab = _spec(REFERENCE, SECOND)
    spec_bc = _spec(SECOND, THIRD)
    receipt_ab = store.commit(spec_ab.pair_id, spec_ab, arc_ab)
    measured: list[str] = []

    def measure(primary: str, secondary: str, *, fail: bool) -> MisregArc:
        measured.append(f"{primary}_{secondary}")
        if fail:
            message = "pair measurement failed"
            raise RuntimeError(message)
        return arc_bc

    with pytest.raises(RuntimeError, match="failed"):
        measure(SECOND, THIRD, fail=True)

    assert store.load(spec_ab.pair_id, spec_ab) == arc_ab
    assert store.load(spec_bc.pair_id, spec_bc) is None

    for primary, secondary, fail in (
        (REFERENCE, SECOND, False),
        (SECOND, THIRD, False),
    ):
        spec = _spec(primary, secondary)
        cached = store.load(spec.pair_id, spec)
        if cached is None:
            cached = measure(primary, secondary, fail=fail)
            store.commit(spec.pair_id, spec, cached)
    assert measured == [f"{SECOND}_{THIRD}", f"{SECOND}_{THIRD}"]
    assert store.commit(spec_ab.pair_id, spec_ab, arc_ab).write_id == (
        receipt_ab.write_id
    )

    solution = solve_misreg([arc_ab, arc_bc], reference=REFERENCE, dates=DATES)
    assert solution.date_misreg.azimuth_px[THIRD] == pytest.approx(-1.0)


def test_solution_signature_binds_numeric_arcs_and_weights() -> None:
    """Changed offsets or weights change provenance even with equal pairs."""
    base = solve_misreg(_triangle(), reference=REFERENCE, dates=DATES)
    changed_offset = solve_misreg(
        [
            _arc(REFERENCE, SECOND, 2.0, 0.5),
            _arc(REFERENCE, THIRD, -1.0, 0.25),
            _arc(SECOND, THIRD, -2.5, -0.25),
        ],
        reference=REFERENCE,
        dates=DATES,
    )
    changed_weight = solve_misreg(
        [
            _arc(REFERENCE, SECOND, 2.0, 0.5, azimuth_sigma=0.5),
            _arc(REFERENCE, THIRD, -1.0, 0.25),
            _arc(SECOND, THIRD, -3.0, -0.25),
        ],
        reference=REFERENCE,
        dates=DATES,
    )

    assert base.used_pairs == changed_offset.used_pairs == changed_weight.used_pairs
    assert base.solution_signature != changed_offset.solution_signature
    assert base.solution_signature != changed_weight.solution_signature
    assert base.correction_signature != changed_offset.correction_signature
    assert base.correction_signature == changed_weight.correction_signature


def test_correction_signature_separates_effective_correction_from_provenance() -> None:
    """Equal corrections reuse consumers; changed eligibility changes signature."""
    base = solve_misreg(_triangle(), reference=REFERENCE, dates=DATES)
    strict = solve_misreg(
        _triangle(), reference=REFERENCE, dates=DATES, strict_completeness=True
    )
    noisy = [
        _arc(REFERENCE, SECOND, 2.0, 0.5),
        _arc(REFERENCE, THIRD, -1.0, 0.25),
        _arc(SECOND, THIRD, -2.5, -0.25, n_valid=1),
    ]
    broad = solve_misreg(noisy, reference=REFERENCE, dates=DATES)
    gated = solve_misreg(
        noisy,
        reference=REFERENCE,
        dates=DATES,
        quality=NESDQualityPolicy(min_n_valid=5),
    )

    assert base.correction_signature == strict.correction_signature
    assert base.solution_signature != strict.solution_signature
    assert broad.correction_signature != gated.correction_signature


def test_recovered_solution_round_trips_through_aggregate_product(
    tmp_path: Path,
) -> None:
    """The existing misreg product accepts recovered arcs and constants."""
    arcs = _triangle()
    solution = solve_misreg(arcs, reference=REFERENCE, dates=DATES)
    root = tmp_path / "misreg"

    product = write_misreg(
        root,
        arcs=arcs,
        date_misreg=solution.date_misreg,
        skipped_no_windows=solution.skipped_no_windows,
    )
    record = read_misreg(root)

    assert record is not None
    assert record.write_id == product.write_id
    assert record.date_misreg is not None
    assert record.date_misreg.azimuth_px == solution.date_misreg.azimuth_px
    assert record.date_misreg.range_px == solution.date_misreg.range_px
