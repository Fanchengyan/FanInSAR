"""Validate FanInSAR's machine-readable source provenance ledger."""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias, TypeGuard

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

ImplementationMethod: TypeAlias = Literal[
    "audited_bsd_adaptation",
    "clean_room",
    "external_optional_dependency",
    "oracle_only",
]
SourceStatus: TypeAlias = Literal["permitted", "prohibited", "quarantined"]
TomlScalar: TypeAlias = str | int | float | bool
TomlValue: TypeAlias = TomlScalar | list["TomlValue"] | dict[str, "TomlValue"]

_REQUIRED_STRING_FIELDS: Final = (
    "name",
    "source",
    "license",
    "implementation_method",
    "status",
    "attribution",
    "test_oracle",
)
_IMPLEMENTATION_METHODS: Final = {
    "audited_bsd_adaptation",
    "clean_room",
    "external_optional_dependency",
    "oracle_only",
}
_PERMITTED_METHODS: Final = {
    "audited_bsd_adaptation",
    "clean_room",
    "external_optional_dependency",
}
_SOURCE_STATUSES: Final = {"permitted", "prohibited", "quarantined"}
_INCOMPATIBLE_LICENSE_MARKERS: Final = (
    "AGPL-",
    "GPL-",
    "LGPL-",
    "NON-COMMERCIAL",
    "NONCOMMERCIAL",
    "SOURCE-AVAILABLE",
)


@dataclass(frozen=True, slots=True)
class _PolicyInvariant:
    """Define a boundary record that every release ledger must retain."""

    policy_id: str
    name: str
    status: SourceStatus
    implementation_method: ImplementationMethod
    required_source_files: tuple[str, ...]


_REQUIRED_POLICIES: Final = (
    _PolicyInvariant(
        "insardev-pygmtsar-gmtsar-derived",
        "GMTSAR-marked insardev_pygmtsar orbit, Doppler, burst, geometry, "
        "baseline, tide, and phase bodies",
        "quarantined",
        "oracle_only",
        ("utils_s1.py", "utils_satellite.py", "utils_tidal.py"),
    ),
    _PolicyInvariant(
        "insardev-source-available",
        "InSAR.dev source-available processing algorithms",
        "prohibited",
        "oracle_only",
        ("core/insardev/",),
    ),
    _PolicyInvariant(
        "snaphu-separate-license",
        "snaphu-py optional unwrapping backend",
        "quarantined",
        "external_optional_dependency",
        ("snaphu-py/ext/snaphu/README",),
    ),
    _PolicyInvariant(
        "gmtsar-oracle",
        "GMTSAR executable and source oracle",
        "quarantined",
        "oracle_only",
        ("gmtsar/LICENSE.TXT",),
    ),
)


class ProvenanceLedgerError(Exception):
    """Report one or more invalid provenance ledger fields.

    Parameters
    ----------
    path : pathlib.Path
        Ledger that failed validation.
    issues : tuple[str, ...]
        Human-readable validation failures.

    """

    def __init__(self, path: Path, issues: tuple[str, ...]) -> None:
        """Initialize the error with the invalid ledger and its issues."""
        self.path = path
        self.issues = issues
        super().__init__(f"invalid provenance ledger {path}: {'; '.join(issues)}")


def _raise_invalid(path: Path, issues: list[str]) -> None:
    error = ProvenanceLedgerError(path, tuple(issues))
    logger.error("Provenance ledger validation failed: %s", error)
    raise error


def _is_nonempty_string(value: TomlValue | None) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value.strip())


def _is_nonempty_string_array(value: TomlValue | None) -> TypeGuard[list[str]]:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_is_nonempty_string(item) for item in value)
    )


def _is_array(value: TomlValue | None) -> TypeGuard[list[TomlValue]]:
    return isinstance(value, list)


def _is_table(value: TomlValue) -> TypeGuard[dict[str, TomlValue]]:
    return isinstance(value, dict)


def _has_incompatible_license(value: TomlValue | None) -> bool:
    return _is_nonempty_string(value) and any(
        marker in value.upper() for marker in _INCOMPATIBLE_LICENSE_MARKERS
    )


def _validate_entry(entry: dict[str, TomlValue], index: int) -> list[str]:
    prefix = f"sources[{index}]"
    issues = [
        f"{prefix}.{field} must be a non-empty string"
        for field in _REQUIRED_STRING_FIELDS
        if not _is_nonempty_string(entry.get(field))
    ]

    if not _is_nonempty_string_array(entry.get("source_files")):
        issues.append(f"{prefix}.source_files must be a non-empty string array")

    method = entry.get("implementation_method")
    status = entry.get("status")
    license_identifier = entry.get("license")
    if method not in _IMPLEMENTATION_METHODS:
        issues.append(f"{prefix}.implementation_method is unsupported: {method!r}")
    if status not in _SOURCE_STATUSES:
        issues.append(f"{prefix}.status is unsupported: {status!r}")
    if status == "permitted" and method not in _PERMITTED_METHODS:
        issues.append(f"{prefix}: {method!r} sources cannot be permitted")
    if (
        status == "permitted"
        and method == "audited_bsd_adaptation"
        and (
            not _is_nonempty_string(license_identifier)
            or "BSD-3-Clause" not in license_identifier
        )
    ):
        issues.append(f"{prefix}: permitted audited adaptations require BSD-3-Clause")
    if (
        status == "permitted"
        and method == "clean_room"
        and _has_incompatible_license(license_identifier)
    ):
        issues.append(f"{prefix}: permitted source has an incompatible license")
    return issues


def _validate_required_policies(
    entries: list[dict[str, TomlValue]],
) -> list[str]:
    issues: list[str] = []
    for policy in _REQUIRED_POLICIES:
        matches = [entry for entry in entries if entry.get("name") == policy.name]
        if not matches:
            issues.append(f"required policy missing: {policy.policy_id}")
            continue
        if len(matches) > 1:
            issues.append(f"required policy duplicated: {policy.policy_id}")
            continue
        entry = matches[0]
        if entry.get("status") != policy.status:
            issues.append(f"required policy has wrong status: {policy.policy_id}")
        if entry.get("implementation_method") != policy.implementation_method:
            issues.append(f"required policy has wrong method: {policy.policy_id}")
        source_files = entry.get("source_files")
        if not _is_nonempty_string_array(source_files) or any(
            not any(marker in source_file for source_file in source_files)
            for marker in policy.required_source_files
        ):
            issues.append(f"required policy paths incomplete: {policy.policy_id}")
    return issues


def check_provenance_ledger(path: Path) -> None:
    """Check a source provenance ledger.

    Parameters
    ----------
    path : pathlib.Path
        Path to the TOML ledger.

    Raises
    ------
    ProvenanceLedgerError
        If the ledger cannot be read or violates the provenance policy.

    """
    try:
        with path.open("rb") as ledger_file:
            raw_ledger: dict[str, TomlValue] = tomllib.load(ledger_file)
    except FileNotFoundError:
        _raise_invalid(path, ["file does not exist"])
    except tomllib.TOMLDecodeError as error:
        _raise_invalid(path, [f"invalid TOML: {error}"])

    issues: list[str] = []
    if raw_ledger.get("schema_version") != 1:
        issues.append("schema_version must equal 1")

    raw_entries = raw_ledger.get("sources")
    if not _is_array(raw_entries) or not raw_entries:
        issues.append("sources must be a non-empty array of tables")
    else:
        entries: list[dict[str, TomlValue]] = []
        for index, entry in enumerate(raw_entries):
            if _is_table(entry):
                entries.append(entry)
                issues.extend(_validate_entry(entry, index))
            else:
                issues.append(f"sources[{index}] must be a table")
        issues.extend(_validate_required_policies(entries))

    if issues:
        _raise_invalid(path, issues)


def main(arguments: list[str] | None = None) -> int:
    """Run the provenance ledger checker command.

    Parameters
    ----------
    arguments : list[str] | None, optional
        Command arguments, excluding the executable name.

    Returns
    -------
    int
        Process exit status.

    """
    cli_arguments = sys.argv[1:] if arguments is None else arguments
    if cli_arguments in (["--help"], ["-h"]):
        sys.stdout.write("usage: python -m faninsar.network.provenance LEDGER\n")
        return 0
    if len(cli_arguments) != 1:
        logger.error("Expected exactly one provenance ledger path")
        return 2
    try:
        check_provenance_ledger(Path(cli_arguments[0]))
    except ProvenanceLedgerError:
        return 1
    logger.info("Provenance ledger is valid: %s", cli_arguments[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
