"""Durable NESD pair measurements and qualified network recovery (PROPOSAL-0088).

One independently committed product stores each measured residual pair arc at
``<root>/<primary>_<secondary>``. The product metadata binds a small measurement
specification (endpoints, method, stage recipe, support identity) and the arc;
it never hashes pixel payloads (PROPOSAL-0082).

:func:`solve_misreg` applies :class:`NESDQualityPolicy` per azimuth/range
component, keeps the reference-connected identifiable component of a planned
network, and reports blocked dates/components instead of inventing geometry
zeros. It reuses :func:`invert_optional_pair_misregistration`; this module adds
recovery and qualification, not a second numerical solver.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np

from faninsar.io.storage.product import open_product, publish_product
from faninsar.logging import setup_logger
from faninsar.processing.coregistration.misreg_network import (
    DateMisreg,
    MisregArc,
    OmissionReason,
    _axis_unusable_reason,
    invert_optional_pair_misregistration,
)
from faninsar.processing.errors import (
    InvalidProcessingStateError,
    reject_invalid_state,
)
from faninsar.stack.misreg import (
    misreg_arc_from_properties,
    misreg_arc_to_properties,
)

if TYPE_CHECKING:
    from faninsar.io.storage.product import Product

logger = setup_logger(__name__)

PAIR_MEASUREMENT_KIND = "misreg-pair"
"""Product kind of one committed pair measurement."""

NESD_RECOVERY_REVISION = "nesd-recovery-v2"
"""Scientific recipe revision of the recovery/qualification seam."""

_NO_OVERLAP_REASON = "no_overlap_windows"

_Component = Literal["azimuth", "range"]
_AxisStatus = Literal["solved", "absent", "blocked"]
_ShiftAttr = Literal["azimuth_shift_px", "range_shift_px"]
_SigmaAttr = Literal["azimuth_sigma_px", "range_sigma_px"]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")
_PAIR_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*_[A-Za-z0-9][A-Za-z0-9.+-]*$")
_SPEC_KEYS = frozenset({"primary", "secondary", "method", "stage_revision", "support"})

__all__ = [
    "NESD_RECOVERY_REVISION",
    "PAIR_MEASUREMENT_KIND",
    "EdgeExclusion",
    "EligibleArcs",
    "MisregSolution",
    "NESDQualityPolicy",
    "OmissionReason",
    "PairMeasurement",
    "PairMeasurementSpec",
    "PairMeasurementStore",
    "effective_correction_signature",
    "measurement_spec",
    "qualify_arcs",
    "solve_misreg",
]


def _require_identifier(value: object, name: str) -> str:
    """Return ``value`` when it is a safe date/endpoint identifier."""
    if not isinstance(value, str) or not value or not _IDENTIFIER.fullmatch(value):
        reject_invalid_state(
            f"{name} must be a non-empty plain identifier, got {value!r}"
        )
    return value


def _require_text(value: object, name: str) -> str:
    """Return ``value`` when it is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        reject_invalid_state(f"{name} must be a non-empty string, got {value!r}")
    return value


def _canonical_json(payload: Mapping[str, object]) -> str:
    """Serialize recovery metadata deterministically as strict JSON."""
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        reject_invalid_state(f"NESD recovery metadata is not strict JSON: {error}")


def _signature(payload: Mapping[str, object]) -> str:
    """Return a SHA-256 digest of deterministic metadata (never payload bytes)."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _freeze_mapping(value: object, name: str) -> Mapping[str, object]:
    """Return a read-only copy of a mapping used in a specification."""
    if not isinstance(value, Mapping):
        reject_invalid_state(f"{name} must be a mapping, got {type(value).__name__}")
    return MappingProxyType({str(key): item for key, item in value.items()})


def _pair_id(primary: str, secondary: str) -> str:
    """Return the canonical ``primary_secondary`` pair id."""
    return f"{primary}_{secondary}"


def _validate_pair_id(pair_id: object) -> str:
    """Return ``pair_id`` when it is one safe ``primary_secondary`` name."""
    if not isinstance(pair_id, str) or not _PAIR_ID.fullmatch(pair_id):
        reject_invalid_state(
            f"pair_id must be a safe primary_secondary identifier: {pair_id!r}"
        )
    return pair_id


def _pair_path(root: Path, pair_id: str) -> Path:
    """Return the product directory for ``pair_id`` under ``root``."""
    return root / _validate_pair_id(pair_id)


@dataclass(frozen=True, slots=True)
class NESDQualityPolicy:
    """Per-component quality limits applied before network inversion.

    Defaults are the declared thresholds of the existing network solver: no
    minimum valid samples and a 1e3 px sigma ceiling. The limits are enforced,
    not only recorded.

    Attributes
    ----------
    min_n_valid : int, optional
        Minimum valid overlap samples per arc.
    max_azimuth_sigma_px : float, optional
        Maximum accepted azimuth shift sigma in pixels.
    max_range_sigma_px : float, optional
        Maximum accepted range shift sigma in pixels.

    """

    min_n_valid: int = 0
    max_azimuth_sigma_px: float = 1e3
    max_range_sigma_px: float = 1e3

    def __post_init__(self) -> None:
        """Validate the declared thresholds."""
        if self.min_n_valid < 0:
            reject_invalid_state("min_n_valid must be non-negative")
        for name, value in (
            ("max_azimuth_sigma_px", self.max_azimuth_sigma_px),
            ("max_range_sigma_px", self.max_range_sigma_px),
        ):
            if not math.isfinite(value) or value <= 0.0:
                reject_invalid_state(f"{name} must be finite and positive")

    @classmethod
    def uniform(
        cls, *, min_n_valid: int = 0, max_sigma_px: float = 1e3
    ) -> NESDQualityPolicy:
        """Return one sigma ceiling shared by both axes.

        Parameters
        ----------
        min_n_valid : int, optional
            Minimum valid overlap samples per arc.
        max_sigma_px : float, optional
            Maximum accepted sigma for both components.

        Returns
        -------
        NESDQualityPolicy
            Policy with equal azimuth and range ceilings.

        """
        return cls(
            min_n_valid=min_n_valid,
            max_azimuth_sigma_px=max_sigma_px,
            max_range_sigma_px=max_sigma_px,
        )

    def document(self) -> dict[str, object]:
        """Return the strict-JSON policy fields recorded in a solution."""
        return {
            "min_n_valid": int(self.min_n_valid),
            "max_azimuth_sigma_px": float(self.max_azimuth_sigma_px),
            "max_range_sigma_px": float(self.max_range_sigma_px),
        }


@dataclass(frozen=True, slots=True)
class PairMeasurementSpec:
    """Stable identity of one measured pair residual.

    The fingerprint intentionally excludes the Stack date catalog, global
    reference, solve quality policy and every payload byte. Adding unrelated
    dates or changing a network solve must reuse an already measured edge.
    ``support`` carries only cheap endpoint source/geometry/support identity
    supplied by the mission.

    Attributes
    ----------
    primary : str
        Design-matrix primary date id.
    secondary : str
        Design-matrix secondary date id.
    method : str, optional
        Measurement method, for example ``"overlap"``.
    stage_revision : str, optional
        Declared result-affecting measurement recipe revision.
    support : mapping of str to object, optional
        Cheap endpoint source/geometry/support identity.

    """

    primary: str
    secondary: str
    method: str = "overlap"
    stage_revision: str = NESD_RECOVERY_REVISION
    support: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate endpoints and freeze the support mapping."""
        _require_identifier(self.primary, "primary")
        _require_identifier(self.secondary, "secondary")
        if self.primary == self.secondary:
            reject_invalid_state("a misreg pair needs two distinct dates")
        _require_text(self.method, "method")
        _require_text(self.stage_revision, "stage_revision")
        object.__setattr__(
            self,
            "support",
            _freeze_mapping(self.support, "support"),
        )
        _canonical_json(self.document())

    @property
    def pair_id(self) -> str:
        """Return the canonical ``primary_secondary`` id."""
        return _pair_id(self.primary, self.secondary)

    @property
    def signature(self) -> str:
        """Return the per-edge measurement fingerprint."""
        return _signature(self.document())

    def document(self) -> dict[str, object]:
        """Return the strict-JSON measurement specification."""
        return {
            "schema": 1,
            "primary": self.primary,
            "secondary": self.secondary,
            "method": self.method,
            "stage_revision": self.stage_revision,
            "support": dict(self.support),
        }


def measurement_spec(
    primary: str,
    secondary: str,
    *,
    method: str = "overlap",
    stage_revision: str = NESD_RECOVERY_REVISION,
    support: Mapping[str, object] | None = None,
) -> PairMeasurementSpec:
    """Build the stable identity of one measured pair residual.

    Parameters
    ----------
    primary : str
        Design-matrix primary date id.
    secondary : str
        Design-matrix secondary date id.
    method : str, optional
        Measurement method.
    stage_revision : str, optional
        Declared result-affecting measurement recipe revision.
    support : mapping of str to object, optional
        Cheap endpoint source/geometry/support identity.

    Returns
    -------
    PairMeasurementSpec
        Validated per-edge specification.

    """
    return PairMeasurementSpec(
        primary=primary,
        secondary=secondary,
        method=method,
        stage_revision=stage_revision,
        support={} if support is None else support,
    )


def _coerce_spec(
    specification: PairMeasurementSpec | Mapping[str, object],
    pair_id: str,
) -> PairMeasurementSpec:
    """Return ``specification`` as a validated spec for ``pair_id``."""
    if isinstance(specification, PairMeasurementSpec):
        spec = specification
    elif isinstance(specification, Mapping):
        unknown = sorted(set(specification) - _SPEC_KEYS)
        if unknown:
            reject_invalid_state(f"unsupported pair measurement spec keys: {unknown}")
        try:
            spec = PairMeasurementSpec(
                primary=str(specification["primary"]),
                secondary=str(specification["secondary"]),
                method=str(specification.get("method", "overlap")),
                stage_revision=str(
                    specification.get("stage_revision", NESD_RECOVERY_REVISION)
                ),
                support=specification.get("support") or {},
            )
        except KeyError as error:
            reject_invalid_state(f"pair measurement spec is missing {error.args[0]!r}")
        except (TypeError, ValueError) as error:
            reject_invalid_state(f"invalid pair measurement spec: {error}")
    else:
        reject_invalid_state(
            "pair measurement spec must be PairMeasurementSpec or a mapping"
        )
    if spec.pair_id != pair_id:
        reject_invalid_state(
            f"pair_id {pair_id!r} does not match specification {spec.pair_id!r}"
        )
    return spec


def _arc_signature(arc: MisregArc) -> str:
    """Return a digest of the persisted numerical arc fields."""
    return _signature({"arc": misreg_arc_to_properties(arc)})


@dataclass(frozen=True, slots=True)
class PairMeasurement:
    """Receipt of one committed pair measurement product.

    Attributes
    ----------
    pair_id : str
        Canonical ``primary_secondary`` id.
    arc : MisregArc
        Committed residual arc.
    write_id : str
        Product write id of the committed measurement.
    spec_signature : str
        Measurement specification signature that owns the product.
    path : pathlib.Path
        Product directory.

    """

    pair_id: str
    arc: MisregArc
    write_id: str
    spec_signature: str
    path: Path


def _matching_measurement(
    product: Product,
    pair_id: str,
    spec: PairMeasurementSpec,
    path: Path,
) -> PairMeasurement | None:
    """Return the measurement when product identity and arc still match."""
    properties = product.properties
    row = properties.get("faninsar:arc")
    if (
        properties.get("faninsar:pair_id") != pair_id
        or properties.get("faninsar:spec_signature") != spec.signature
        or not isinstance(row, Mapping)
    ):
        return None
    arc = misreg_arc_from_properties(row)
    if arc.pair_id != pair_id or properties.get(
        "faninsar:arc_signature"
    ) != _arc_signature(arc):
        return None
    return PairMeasurement(
        pair_id=pair_id,
        arc=arc,
        write_id=product.write_id,
        spec_signature=spec.signature,
        path=path,
    )


class PairMeasurementStore:
    """Commit and reopen residual pair arcs independently.

    Each edge is one committed product under ``root/<pair_id>``. A saved arc
    is reused only when the requested per-edge specification signature matches;
    a changed method, recipe or support identity invalidates that edge only.

    Parameters
    ----------
    root : path-like
        Directory holding pair measurement products, for example
        ``<work_dir>/misreg/measurements``.

    """

    def __init__(self, root: str | Path) -> None:
        """Create a store rooted at ``root``."""
        self.root = Path(root)

    def load(
        self,
        pair_id: str,
        specification: PairMeasurementSpec | Mapping[str, object],
    ) -> MisregArc | None:
        """Return the committed arc for ``pair_id`` and specification.

        Parameters
        ----------
        pair_id : str
            Canonical ``primary_secondary`` id.
        specification : PairMeasurementSpec or mapping
            Current per-edge measurement identity.

        Returns
        -------
        MisregArc or None
            The committed arc, or ``None`` when absent, unreadable, or built
            from a different measurement specification.

        """
        spec = _coerce_spec(specification, pair_id)
        measurement = self._read(pair_id, spec)
        return None if measurement is None else measurement.arc

    def commit(
        self,
        pair_id: str,
        specification: PairMeasurementSpec | Mapping[str, object],
        arc: MisregArc,
    ) -> PairMeasurement:
        """Commit one pair arc durably and independently.

        Re-committing an identical arc under the same specification keeps the
        existing product and write id instead of rewriting it.

        Parameters
        ----------
        pair_id : str
            Canonical ``primary_secondary`` id.
        specification : PairMeasurementSpec or mapping
            Per-edge measurement identity.
        arc : MisregArc
            Measured residual arc.

        Returns
        -------
        PairMeasurement
            Receipt of the committed product.

        """
        spec = _coerce_spec(specification, pair_id)
        if arc.pair_id != pair_id:
            reject_invalid_state(
                f"pair_id {pair_id!r} does not match measured arc {arc.pair_id!r}"
            )
        if (arc.primary, arc.secondary) != (spec.primary, spec.secondary):
            reject_invalid_state(
                f"measured arc endpoints do not match specification {pair_id!r}"
            )
        existing = self._read(pair_id, spec)
        if existing is not None and (
            _arc_signature(existing.arc) == _arc_signature(arc)
        ):
            return existing

        path = _pair_path(self.root, pair_id)
        properties = {
            "faninsar:pair_id": pair_id,
            "faninsar:spec": spec.document(),
            "faninsar:spec_signature": spec.signature,
            "faninsar:arc": misreg_arc_to_properties(arc),
            "faninsar:arc_signature": _arc_signature(arc),
            "faninsar:stage_revision": spec.stage_revision,
        }
        product = publish_product(
            path,
            kind=PAIR_MEASUREMENT_KIND,
            properties=properties,
            write_payloads=lambda _staging: {},
            overwrite=True,
        )
        return PairMeasurement(
            pair_id=pair_id,
            arc=arc,
            write_id=product.write_id,
            spec_signature=spec.signature,
            path=path,
        )

    def _read(self, pair_id: str, spec: PairMeasurementSpec) -> PairMeasurement | None:
        """Return the committed measurement when its spec and arc match."""
        path = _pair_path(self.root, pair_id)
        if not (path / "item.json").is_file():
            return None
        try:
            product = open_product(path, kind=PAIR_MEASUREMENT_KIND)
            return _matching_measurement(product, pair_id, spec, path)
        except (InvalidProcessingStateError, KeyError, TypeError, ValueError) as error:
            logger.warning(
                "Ignoring unreadable NESD pair measurement %s: %s", path, error
            )
            return None


@dataclass(frozen=True, slots=True)
class EdgeExclusion:
    """One omitted or quality-rejected pair component.

    Attributes
    ----------
    pair_id : str
        Canonical ``primary_secondary`` id.
    component : {"azimuth", "range"}
        Rejected axis.
    reason : str
        ``shift_omitted``, ``shift_non_finite``, ``sigma_invalid``,
        ``sigma_above_max``, or ``n_valid_below_min``.

    """

    pair_id: str
    component: _Component
    reason: str


@dataclass(frozen=True, slots=True)
class EligibleArcs:
    """Quality-qualified arcs and the exact edge decisions of one solve.

    Attributes
    ----------
    arcs : tuple of MisregArc
        Saved arcs with rejected components replaced by ``None``.
    exclusions : tuple of EdgeExclusion
        Omitted or rejected components.
    missing_pairs : tuple of str
        Planned pair ids with no saved measurement.
    reference : str
        Reference date id.
    dates : tuple of str
        Sorted published target dates.

    """

    arcs: tuple[MisregArc, ...]
    exclusions: tuple[EdgeExclusion, ...]
    missing_pairs: tuple[str, ...]
    reference: str
    dates: tuple[str, ...]


def _normalize_dates(dates: Sequence[str], reference: str) -> tuple[str, ...]:
    """Return validated, sorted, unique target dates including ``reference``."""
    _require_identifier(reference, "reference")
    normalized = tuple(sorted({_require_identifier(date, "date") for date in dates}))
    if not normalized:
        reject_invalid_state("NESD solve requires at least one date")
    if reference not in normalized:
        reject_invalid_state("Reference must be included in dates")
    return normalized


def _normalize_planned_pairs(
    planned_pairs: Iterable[tuple[str, str]],
    dates: tuple[str, ...],
) -> tuple[str, ...]:
    """Return validated, sorted, unique planned pair ids."""
    known = set(dates)
    pair_ids: set[str] = set()
    for primary, secondary in planned_pairs:
        _require_identifier(primary, "planned primary")
        _require_identifier(secondary, "planned secondary")
        if primary == secondary:
            reject_invalid_state("a planned misreg pair needs two distinct dates")
        if primary not in known or secondary not in known:
            reject_invalid_state(
                f"planned pair {primary}_{secondary} references unknown dates"
            )
        pair_ids.add(_pair_id(primary, secondary))
    return tuple(sorted(pair_ids))


def _validate_arc(arc: MisregArc, dates: set[str]) -> None:
    """Reject an arc whose endpoints are not published target dates."""
    _require_identifier(arc.primary, "arc primary")
    _require_identifier(arc.secondary, "arc secondary")
    if arc.primary == arc.secondary:
        reject_invalid_state(f"arc {arc.pair_id} joins one date to itself")
    if arc.primary not in dates or arc.secondary not in dates:
        reject_invalid_state(
            f"arc {arc.pair_id} references dates outside the NESD target"
        )


def _component_reason(
    arc: MisregArc,
    component: _Component,
    policy: NESDQualityPolicy,
) -> str | None:
    """Return why one axis component is unusable, or ``None``."""
    if component == "azimuth":
        shift = arc.azimuth_shift_px
        sigma = arc.azimuth_sigma_px
        limit = policy.max_azimuth_sigma_px
        omission = arc.azimuth_omission
    else:
        shift = arc.range_shift_px
        sigma = arc.range_sigma_px
        limit = policy.max_range_sigma_px
        omission = arc.range_omission
    reason: str | None = None
    if shift is None:
        if omission == "quality_rejected":
            reason = "quality_rejected"
        elif omission == "no_overlap":
            reason = "no_overlap"
        else:
            reason = "shift_omitted"
    elif not math.isfinite(float(shift)):
        reason = "shift_non_finite"
    elif not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
        reason = "sigma_invalid"
    elif float(sigma) > limit:
        reason = "sigma_above_max"
    elif int(arc.n_valid) < policy.min_n_valid:
        reason = "n_valid_below_min"
    return reason


def _qualified_omission(
    existing: OmissionReason | None,
    reason: str | None,
) -> OmissionReason | None:
    """Keep a scoped reason, or record a policy rejection as quality."""
    if reason is None or reason in {"shift_omitted", "no_overlap"}:
        return existing
    return "quality_rejected"


def _exclusion_payload(
    exclusions: Sequence[EdgeExclusion],
) -> list[dict[str, object]]:
    """Return strict-JSON exclusion rows."""
    return [
        {
            "pair_id": item.pair_id,
            "component": item.component,
            "reason": item.reason,
        }
        for item in exclusions
    ]


def qualify_arcs(
    arcs: Sequence[MisregArc],
    *,
    reference: str,
    dates: Sequence[str],
    planned_pairs: Iterable[tuple[str, str]] = (),
    quality: NESDQualityPolicy | None = None,
    strict_completeness: bool = False,
) -> EligibleArcs:
    """Apply per-axis quality limits and the planned-edge completeness policy.

    A rejected component becomes ``None`` on the returned arc so the other axis
    survives. Missing planned edges and quality-rejected planned components
    warn by default; ``strict_completeness`` rejects them.

    Parameters
    ----------
    arcs : sequence of MisregArc
        Saved residual arcs.
    reference : str
        Reference date id.
    dates : sequence of str
        Published target dates.
    planned_pairs : iterable of tuple of str, optional
        Planned measurement edges. Empty means no completeness expectation.
    quality : NESDQualityPolicy, optional
        Per-axis quality limits.
    strict_completeness : bool, optional
        Reject any missing planned edge or quality-rejected planned component.

    Returns
    -------
    EligibleArcs
        Sanitized arcs, exact exclusions, and missing planned pair ids.

    """
    policy = quality or NESDQualityPolicy()
    date_list = _normalize_dates(dates, reference)
    planned = _normalize_planned_pairs(planned_pairs, date_list)
    known = set(date_list)
    saved: dict[str, MisregArc] = {}
    for arc in arcs:
        _validate_arc(arc, known)
        if arc.pair_id in saved:
            reject_invalid_state(f"duplicate saved NESD pair arc: {arc.pair_id}")
        saved[arc.pair_id] = arc
    missing = tuple(pid for pid in planned if pid not in saved)
    qualified: list[MisregArc] = []
    exclusions: list[EdgeExclusion] = []
    planned_set = set(planned)
    for pair_id in sorted(saved):
        arc = saved[pair_id]
        azimuth_reason = _component_reason(arc, "azimuth", policy)
        range_reason = _component_reason(arc, "range", policy)
        if azimuth_reason is not None:
            exclusions.append(EdgeExclusion(pair_id, "azimuth", azimuth_reason))
        if range_reason is not None:
            exclusions.append(EdgeExclusion(pair_id, "range", range_reason))
        qualified.append(
            replace(
                arc,
                azimuth_shift_px=(
                    None if azimuth_reason is not None else arc.azimuth_shift_px
                ),
                range_shift_px=(
                    None if range_reason is not None else arc.range_shift_px
                ),
                azimuth_omission=_qualified_omission(
                    arc.azimuth_omission, azimuth_reason
                ),
                range_omission=_qualified_omission(arc.range_omission, range_reason),
            )
        )
    failed = tuple(
        item
        for item in exclusions
        if item.pair_id in planned_set
        and item.reason not in {"shift_omitted", "no_overlap"}
    )
    if strict_completeness and (missing or failed):
        details = [*missing, *(item.pair_id for item in failed)]
        reject_invalid_state(
            "strict NESD completeness requires every planned edge: "
            f"{sorted(set(details))}"
        )
    if missing:
        logger.warning(
            "NESD planned pair edges are missing from saved measurements: %s",
            list(missing),
        )
    for item in failed:
        logger.warning(
            "NESD pair %s %s component rejected: %s",
            item.pair_id,
            item.component,
            item.reason,
        )
    return EligibleArcs(
        arcs=tuple(qualified),
        exclusions=tuple(exclusions),
        missing_pairs=missing,
        reference=reference,
        dates=date_list,
    )


@dataclass(frozen=True, slots=True)
class _AxisResult:
    """Qualified result of one invertible component."""

    status: _AxisStatus
    values: Mapping[str, float]
    observations: tuple[tuple[MisregArc, float, float], ...]
    blocked: tuple[tuple[str, str], ...]
    reason: str | None
    metadata: Mapping[str, object]


def _reference_component(arcs: Sequence[MisregArc], reference: str) -> set[str]:
    """Return dates reachable from ``reference`` through ``arcs``."""
    adjacency: dict[str, set[str]] = {}
    for arc in arcs:
        adjacency.setdefault(arc.primary, set()).add(arc.secondary)
        adjacency.setdefault(arc.secondary, set()).add(arc.primary)
    seen = {reference}
    pending = [reference]
    while pending:
        node = pending.pop()
        for neighbor in adjacency.get(node, ()):
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return seen


def _axis_observations(
    arcs: Sequence[MisregArc],
    shift_attr: _ShiftAttr,
    sigma_attr: _SigmaAttr,
) -> tuple[tuple[MisregArc, float, float], ...]:
    """Return numerical observations and weights actually used by an axis."""
    return tuple(
        (
            arc,
            float(getattr(arc, shift_attr)),
            float(getattr(arc, sigma_attr)),
        )
        for arc in arcs
        if getattr(arc, shift_attr) is not None
    )


def _blocked_all(
    dates: Sequence[str], reference: str, reason: str
) -> tuple[tuple[str, str], ...]:
    """Return every non-reference date with one blocked reason."""
    return tuple((date, reason) for date in dates if date != reference)


def _solve_axis(
    eligible: EligibleArcs,
    raw_arcs: Sequence[MisregArc],
    *,
    component: _Component,
    shift_attr: _ShiftAttr,
    sigma_attr: _SigmaAttr,
    no_overlap: bool,
    strict_completeness: bool,
) -> _AxisResult:
    """Solve one axis of the reference-connected qualified component."""
    dates = eligible.dates
    reference = eligible.reference
    omission_attr = "azimuth_omission" if component == "azimuth" else "range_omission"
    raw_has_component = any(getattr(arc, shift_attr) is not None for arc in raw_arcs)
    raw_quality_rejected = any(
        getattr(arc, omission_attr) == "quality_rejected" for arc in raw_arcs
    )
    axis_arcs = [arc for arc in eligible.arcs if getattr(arc, shift_attr) is not None]
    if not axis_arcs:
        if raw_has_component or raw_quality_rejected:
            if strict_completeness:
                reject_invalid_state(
                    f"strict NESD completeness has no usable {component} observations"
                )
            return _AxisResult(
                status="blocked",
                values={},
                observations=(),
                blocked=_blocked_all(dates, reference, "no_usable_observations"),
                reason="no_usable_observations",
                metadata={"n_arcs": 0, "unusable": "no_usable_observations"},
            )
        _ = no_overlap
        return _AxisResult(
            status="absent",
            values=dict.fromkeys(dates, 0.0),
            observations=(),
            blocked=(),
            reason=None,
            metadata={"n_arcs": 0, "empty": True, "unusable": "empty"},
        )

    component_dates = _reference_component(axis_arcs, reference)
    if component_dates == {reference}:
        if strict_completeness:
            reject_invalid_state(
                f"strict NESD completeness has no reference-connected {component} "
                "component"
            )
        return _AxisResult(
            status="blocked",
            values={},
            observations=_axis_observations(axis_arcs, shift_attr, sigma_attr),
            blocked=_blocked_all(dates, reference, "no_reference_connection"),
            reason="no_reference_connection",
            metadata={"n_arcs": 0, "unusable": "no_reference_connection"},
        )

    solved_dates = tuple(date for date in dates if date in component_dates)
    disconnected = tuple(
        (date, "disconnected") for date in dates if date not in component_dates
    )
    if disconnected and strict_completeness:
        reject_invalid_state(
            f"strict NESD completeness blocks disconnected {component} dates: "
            f"{[date for date, _reason in disconnected]}"
        )
    component_set = set(solved_dates)
    component_arcs = [
        arc
        for arc in axis_arcs
        if arc.primary in component_set and arc.secondary in component_set
    ]
    sigmas = np.array(
        [float(getattr(arc, sigma_attr)) for arc in component_arcs],
        dtype=np.float64,
    )
    unusable = _axis_unusable_reason(
        component_arcs,
        reference=reference,
        date_list=list(solved_dates),
        sigmas=sigmas,
    )
    if unusable is not None:
        if strict_completeness:
            reject_invalid_state(
                f"strict NESD completeness rejects {component} network: {unusable}"
            )
        blocked = disconnected + tuple(
            (date, unusable) for date in solved_dates if date != reference
        )
        return _AxisResult(
            status="blocked",
            values={},
            observations=_axis_observations(component_arcs, shift_attr, sigma_attr),
            blocked=blocked,
            reason=unusable,
            metadata={"n_arcs": len(component_arcs), "unusable": unusable},
        )

    inverted = invert_optional_pair_misregistration(
        component_arcs,
        reference=reference,
        dates=list(solved_dates),
    )
    values = {
        date: float(getattr(inverted, f"{component}_px")[date]) for date in solved_dates
    }
    metadata = inverted.metadata.get(component)
    return _AxisResult(
        status="solved",
        values=values,
        observations=_axis_observations(component_arcs, shift_attr, sigma_attr),
        blocked=disconnected,
        reason=None,
        metadata=metadata if isinstance(metadata, Mapping) else {},
    )


def _axis_values(result: _AxisResult) -> dict[str, float]:
    """Return the persisted per-date constants of one axis."""
    if result.status in {"solved", "absent"}:
        return {date: float(value) for date, value in result.values.items()}
    return {}


def _axis_signature_payload(result: _AxisResult) -> dict[str, object]:
    """Return the numerical and aggregate identity of one axis."""
    return {
        "status": result.status,
        "reason": result.reason,
        "dates": sorted(result.values) if result.status != "blocked" else [],
        "observations": [
            {
                "primary": arc.primary,
                "secondary": arc.secondary,
                "value": value,
                "sigma": sigma,
                "n_valid": int(arc.n_valid),
                "method": arc.method,
                "azimuth_omission": arc.azimuth_omission,
                "range_omission": arc.range_omission,
            }
            for arc, value, sigma in result.observations
        ],
        "blocked": [
            {"date": date, "reason": reason} for date, reason in result.blocked
        ],
    }


def _correction_signature(
    azimuth_px: Mapping[str, float],
    range_px: Mapping[str, float],
    dates: Sequence[str],
) -> str:
    """Return the effective per-date correction signature."""
    payload: dict[str, list[float | None]] = {}
    for date in dates:
        payload[date] = [
            _stable_float(azimuth_px.get(date)),
            _stable_float(range_px.get(date)),
        ]
    return _signature(payload)


def _stable_float(value: float | None) -> float | None:
    """Round an effective correction so solver roundoff is not a change."""
    if value is None:
        return None
    rounded = round(float(value), 12)
    return 0.0 if rounded == 0.0 else rounded


def effective_correction_signature(
    date_misreg: DateMisreg,
    *,
    dates: Sequence[str] | None = None,
) -> str:
    """Return the effective per-date correction signature.

    Two solutions with different provenance but equal per-date azimuth/range
    corrections produce the same signature, so their consumers stay reusable.

    Parameters
    ----------
    date_misreg : DateMisreg
        Inverted per-date constants.
    dates : sequence of str, optional
        Dates to include. Default: every date present on either axis.

    Returns
    -------
    str
        SHA-256 digest of the effective corrections.

    """
    date_list = (
        tuple(sorted(dates))
        if dates is not None
        else tuple(sorted(set(date_misreg.azimuth_px) | set(date_misreg.range_px)))
    )
    return _correction_signature(
        date_misreg.azimuth_px,
        date_misreg.range_px,
        date_list,
    )


@dataclass(frozen=True, slots=True)
class MisregSolution:
    """Recovered network solution with explicit edge and blocking evidence.

    Attributes
    ----------
    date_misreg : DateMisreg
        Recovered per-date constants. Blocked dates are omitted, never zeroed.
    reference : str
        Reference date id.
    dates : tuple of str
        Published target dates.
    used_pairs : tuple of str
        Pair ids contributing at least one solved component.
    exclusions : tuple of EdgeExclusion
        Omitted or quality-rejected components.
    missing_pairs : tuple of str
        Planned pair ids with no saved measurement.
    strict_completeness : bool
        Whether strict planned-edge completion was required.
    skipped_no_windows : bool
        Whether all components were explicitly known to have no overlap.
    blocked_dates : tuple of str
        Dates outside at least one axis solution.
    blocked_components : tuple of tuple
        ``(date, component, reason)`` blocked evidence.
    solution_signature : str
        Digest of actual numerical arcs, weights, eligibility and blocking.
    correction_signature : str
        Digest of effective per-date corrections only.

    """

    date_misreg: DateMisreg
    reference: str
    dates: tuple[str, ...]
    used_pairs: tuple[str, ...]
    exclusions: tuple[EdgeExclusion, ...]
    missing_pairs: tuple[str, ...]
    strict_completeness: bool
    skipped_no_windows: bool
    blocked_dates: tuple[str, ...]
    blocked_components: tuple[tuple[str, str, str], ...]
    solution_signature: str
    correction_signature: str


def solve_misreg(
    arcs: Sequence[MisregArc],
    *,
    reference: str,
    dates: Sequence[str],
    planned_pairs: Iterable[tuple[str, str]] = (),
    quality: NESDQualityPolicy | None = None,
    strict_completeness: bool = False,
    stage_revision: str = NESD_RECOVERY_REVISION,
    no_overlap: bool = False,
) -> MisregSolution:
    """Invert quality-qualified saved arcs and report blocked targets.

    By default the reference-connected, identifiable component of each axis is
    solved and published dates outside it are reported in ``blocked_dates``.
    Strict completeness raises instead. An all-omitted campaign returns
    geometry zeros only when ``no_overlap=True`` explicitly records a known
    no-overlap selection.

    Parameters
    ----------
    arcs : sequence of MisregArc
        Saved residual arcs for this measurement attempt.
    reference : str
        Reference date id fixed at zero.
    dates : sequence of str
        Published target dates.
    planned_pairs : iterable of tuple of str, optional
        Planned measurement edges.
    quality : NESDQualityPolicy, optional
        Per-axis quality limits.
    strict_completeness : bool, optional
        Require every planned edge and every published target to solve.
    stage_revision : str, optional
        Declared recovery recipe revision.
    no_overlap : bool, optional
        Explicit known no-overlap selection when every component is omitted.

    Returns
    -------
    MisregSolution
        Recovered constants, exact edge decisions and blocked evidence.

    Raises
    ------
    InvalidProcessingStateError
        If strict completeness fails, no reference-connected axis solves, or
        all components are omitted without ``no_overlap=True``.

    """
    policy = quality or NESDQualityPolicy()
    _require_text(stage_revision, "stage_revision")
    eligible = qualify_arcs(
        arcs,
        reference=reference,
        dates=dates,
        planned_pairs=planned_pairs,
        quality=policy,
        strict_completeness=strict_completeness,
    )
    raw_has_azimuth = any(arc.azimuth_shift_px is not None for arc in arcs)
    raw_has_range = any(arc.range_shift_px is not None for arc in arcs)
    raw_quality_rejected = any(
        arc.azimuth_omission == "quality_rejected"
        or arc.range_omission == "quality_rejected"
        for arc in arcs
    )
    raw_no_overlap = any(
        arc.azimuth_omission == "no_overlap" or arc.range_omission == "no_overlap"
        for arc in arcs
    )
    known_no_overlap = bool(no_overlap or raw_no_overlap)
    skipped_no_windows = (
        not raw_has_azimuth and not raw_has_range and not raw_quality_rejected
    )

    azimuth = _solve_axis(
        eligible,
        arcs,
        component="azimuth",
        shift_attr="azimuth_shift_px",
        sigma_attr="azimuth_sigma_px",
        no_overlap=no_overlap,
        strict_completeness=strict_completeness,
    )
    ranges = _solve_axis(
        eligible,
        arcs,
        component="range",
        shift_attr="range_shift_px",
        sigma_attr="range_sigma_px",
        no_overlap=no_overlap,
        strict_completeness=strict_completeness,
    )

    if skipped_no_windows and not known_no_overlap:
        reject_invalid_state(
            "all NESD components are omitted; pass no_overlap=True for a "
            "known no-overlap selection or record the measured omission reason"
        )
    if (
        azimuth.status != "solved"
        and ranges.status != "solved"
        and not (skipped_no_windows and known_no_overlap)
    ):
        reject_invalid_state(
            "no usable reference-connected, identifiable NESD solve remains"
        )

    azimuth_px = _axis_values(azimuth)
    range_px = _axis_values(ranges)
    blocked_components = tuple(
        sorted(
            (date, component, reason)
            for component, result in (
                ("azimuth", azimuth),
                ("range", ranges),
            )
            for date, reason in result.blocked
        )
    )
    blocked_dates = tuple(sorted({row[0] for row in blocked_components}))
    used_pairs = tuple(
        sorted(
            {
                arc.pair_id
                for result in (azimuth, ranges)
                if result.status == "solved"
                for arc, _value, _sigma in result.observations
            }
        )
    )
    correction_signature = _correction_signature(azimuth_px, range_px, eligible.dates)
    signature_payload: dict[str, object] = {
        "reference": reference,
        "dates": list(eligible.dates),
        "stage_revision": stage_revision,
        "quality": policy.document(),
        "strict_completeness": bool(strict_completeness),
        "no_overlap": bool(no_overlap),
        "known_no_overlap": known_no_overlap,
        "used_pairs": list(used_pairs),
        "missing_pairs": list(eligible.missing_pairs),
        "exclusions": _exclusion_payload(eligible.exclusions),
        "blocked_components": [list(row) for row in blocked_components],
        "azimuth": _axis_signature_payload(azimuth),
        "range": _axis_signature_payload(ranges),
    }
    solution_signature = _signature(signature_payload)
    metadata = {
        "n_arcs_az": (len(azimuth.observations) if azimuth.status == "solved" else 0),
        "n_arcs_rg": (len(ranges.observations) if ranges.status == "solved" else 0),
        "n_dates": len(eligible.dates),
        "azimuth": dict(azimuth.metadata),
        "range": dict(ranges.metadata),
        "nesd_recovery": {
            "schema": 1,
            "stage_revision": stage_revision,
            "quality": policy.document(),
            "strict_completeness": bool(strict_completeness),
            "no_overlap": bool(no_overlap),
            "known_no_overlap": known_no_overlap,
            "no_overlap_reason": _NO_OVERLAP_REASON if known_no_overlap else None,
            "used_pairs": list(used_pairs),
            "missing_pairs": list(eligible.missing_pairs),
            "exclusions": _exclusion_payload(eligible.exclusions),
            "blocked_components": [list(row) for row in blocked_components],
            "azimuth": {
                "status": azimuth.status,
                "reason": azimuth.reason,
            },
            "range": {
                "status": ranges.status,
                "reason": ranges.reason,
            },
            "solution_signature": solution_signature,
            "correction_signature": correction_signature,
        },
    }
    date_misreg = DateMisreg(
        reference=reference,
        azimuth_px=azimuth_px,
        range_px=range_px,
        metadata=metadata,
    )
    return MisregSolution(
        date_misreg=date_misreg,
        reference=reference,
        dates=eligible.dates,
        used_pairs=used_pairs,
        exclusions=eligible.exclusions,
        missing_pairs=eligible.missing_pairs,
        strict_completeness=bool(strict_completeness),
        skipped_no_windows=skipped_no_windows,
        blocked_dates=blocked_dates,
        blocked_components=blocked_components,
        solution_signature=solution_signature,
        correction_signature=correction_signature,
    )
