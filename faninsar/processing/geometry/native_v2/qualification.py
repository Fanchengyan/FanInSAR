"""Exact-artifact OpenMP qualification telemetry and checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OpenMPTelemetry:
    """Telemetry emitted by the exact extension geometry symbol.

    Parameters
    ----------
    extension_name : str
        Loaded extension identity.
    geometry_symbol : str
        Exported operation symbol that executed the fixture.
    openmp_defined : bool
        Value of the extension's compile-time ``_OPENMP`` assertion.
    runtime_name : str
        Runtime library observed while loading the extension.
    observed_thread_ids : tuple[int, ...]
        Distinct worker IDs observed in the geometry loop.
    fixture_point_count : int
        Number of points submitted to qualification.
    processed_point_count : int
        Number of points processed by the exact geometry loop.
    visit_counts : tuple[int, ...]
        Per-index visit counts from the exact geometry loop.
    observed_affinity : tuple[int, ...]
        CPUs observed by the loop, normalized as sorted IDs.

    """

    extension_name: str
    geometry_symbol: str
    openmp_defined: bool
    runtime_name: str
    observed_thread_ids: tuple[int, ...]
    fixture_point_count: int
    processed_point_count: int
    visit_counts: tuple[int, ...]
    observed_affinity: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class QualificationResult:
    """Fail-closed qualification result for one exact candidate."""

    qualified: bool
    reason: str
    extension_name: str
    geometry_symbol: str
    observed_threads: int


def qualify_openmp_telemetry(
    telemetry: OpenMPTelemetry,
    *,
    expected_extension: str,
    expected_geometry_symbol: str,
    expected_runtime: str,
    requested_threads: int,
    expected_affinity: tuple[int, ...] | None = None,
) -> QualificationResult:
    """Validate OpenMP evidence from the exact extension and symbol.

    Parameters
    ----------
    telemetry : OpenMPTelemetry
        Observation produced by the operation's real geometry loop.
    expected_extension : str
        Exact extension selected for runtime execution.
    expected_geometry_symbol : str
        Exact operation symbol selected for runtime execution.
    expected_runtime : str
        Runtime identity supplied by the platform provider.
    requested_threads : int
        Requested OpenMP thread count.  Counts greater than one must observe
        multiple threads.
    expected_affinity : tuple[int, ...], optional
        Exact normalized affinity policy, when one is configured.

    Returns
    -------
    QualificationResult
        Qualified only when all identity, coverage, macro, runtime, thread,
        and affinity checks pass.

    """
    observed_threads = len(set(telemetry.observed_thread_ids))
    checks = (
        (telemetry.extension_name == expected_extension, "extension identity mismatch"),
        (
            telemetry.geometry_symbol == expected_geometry_symbol,
            "geometry symbol mismatch",
        ),
        (telemetry.openmp_defined, "_OPENMP is not defined in the exact extension"),
        (telemetry.runtime_name == expected_runtime, "OpenMP runtime mismatch"),
        (
            requested_threads >= 1,
            "requested OpenMP thread count must be positive",
        ),
        (
            telemetry.fixture_point_count > 0,
            "qualification fixture must contain points",
        ),
        (
            telemetry.processed_point_count == telemetry.fixture_point_count,
            "geometry loop did not process exactly the fixture point count",
        ),
        (
            len(telemetry.visit_counts) == telemetry.fixture_point_count
            and all(count == 1 for count in telemetry.visit_counts),
            "geometry loop did not visit every fixture index exactly once",
        ),
        (
            observed_threads == requested_threads,
            "observed thread count differs from the requested profile",
        ),
        (
            expected_affinity is None
            or tuple(sorted(set(telemetry.observed_affinity)))
            == tuple(sorted(set(expected_affinity))),
            "observed affinity differs from the execution profile",
        ),
    )
    for passed, reason in checks:
        if not passed:
            return QualificationResult(
                False,
                reason,
                telemetry.extension_name,
                telemetry.geometry_symbol,
                observed_threads,
            )
    return QualificationResult(
        True,
        "qualified",
        telemetry.extension_name,
        telemetry.geometry_symbol,
        observed_threads,
    )
