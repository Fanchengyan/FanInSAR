"""Tests for the source provenance ledger checker."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from faninsar.network.provenance import (
    ProvenanceLedgerError,
    check_provenance_ledger,
)

if TYPE_CHECKING:
    from pathlib import Path

ATTRIBUTION_LINE = (
    'attribution = "Specification cited in the implementation documentation."\n'
)
BOUNDARY_RECORDS = '''

[[sources]]
name = """\
GMTSAR-marked insardev_pygmtsar orbit, Doppler, burst, geometry, \
baseline, tide, and phase bodies"""
source = "core/insardev_pygmtsar GMTSAR-derived sections"
license = "BSD-3-Clause distribution with unresolved GPL-derived provenance markers"
implementation_method = "oracle_only"
status = "quarantined"
attribution = "GMTSAR and insardev_pygmtsar identified in oracle metadata."
test_oracle = "Black-box numeric outputs only."
source_files = [
  "core/insardev_pygmtsar/insardev_pygmtsar/utils_s1.py:11-12",
  "core/insardev_pygmtsar/insardev_pygmtsar/utils_satellite.py:1115",
  "core/insardev_pygmtsar/insardev_pygmtsar/utils_tidal.py:11-15",
]

[[sources]]
name = "InSAR.dev source-available processing algorithms"
source = "core/insardev"
license = "InSAR.dev Source-Available License 1.0"
implementation_method = "oracle_only"
status = "prohibited"
attribution = "InSAR.dev recorded only in isolated oracle metadata."
test_oracle = "Black-box outputs in a separate environment only."
source_files = ["core/insardev/insardev/utils_unwrap2d.py"]

[[sources]]
name = "snaphu-py optional unwrapping backend"
source = "isce-framework/snaphu-py"
license = "Separate mixed terms including non-commercial restrictions"
implementation_method = "external_optional_dependency"
status = "quarantined"
attribution = "Show the separate-license caveat."
test_oracle = "Optional clean-environment backend tests."
source_files = ["snaphu-py/ext/snaphu/README"]

[[sources]]
name = "GMTSAR executable and source oracle"
source = "GMTSAR"
license = "GPL-3.0"
implementation_method = "oracle_only"
status = "quarantined"
attribution = "Record GMTSAR version and license with oracle artifacts."
test_oracle = "Isolated development-only outputs."
source_files = ["gmtsar/LICENSE.TXT"]
'''

VALID_LEDGER = (
    """\
schema_version = 1

[[sources]]
name = "range interpolation"
source = "public algorithm specification"
license = "N/A"
implementation_method = "clean_room"
status = "permitted"
attribution = "Specification cited in the implementation documentation."
test_oracle = "Synthetic analytical result."
source_files = ["faninsar/processing/range.py"]
"""
    + BOUNDARY_RECORDS
)


def _write_ledger(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "source_provenance.toml"
    path.write_text(content, encoding="utf-8")
    return path


def _remove_record(content: str, name: str) -> str:
    records = content.split("[[sources]]")
    return "[[sources]]".join(
        record for record in records if f'name = "{name}"' not in record
    )


def test_valid_ledger_passes_when_clean_room_source_is_complete(
    tmp_path: Path,
) -> None:
    """Accept a complete clean-room ledger with required boundaries."""
    # Given: a complete clean-room ledger entry.
    ledger_path = _write_ledger(tmp_path, VALID_LEDGER)

    # When: the ledger is checked.
    check_provenance_ledger(ledger_path)

    # Then: no validation error is raised.


def test_missing_field_is_rejected_when_entry_omits_attribution(
    tmp_path: Path,
) -> None:
    """Reject an entry that omits required attribution."""
    # Given: a ledger entry without the required attribution field.
    ledger_path = _write_ledger(
        tmp_path,
        VALID_LEDGER.replace(ATTRIBUTION_LINE, "", 1),
    )

    # When: the ledger is checked.
    with pytest.raises(ProvenanceLedgerError, match="attribution"):
        check_provenance_ledger(ledger_path)

    # Then: the missing field is named by the typed validation error.


def test_incompatible_license_is_rejected_when_audited_adaptation_is_permitted(
    tmp_path: Path,
) -> None:
    """Reject GPL-only licensing for an audited adaptation."""
    # Given: a permitted audited adaptation carrying an incompatible license.
    ledger_path = _write_ledger(
        tmp_path,
        VALID_LEDGER.replace('license = "N/A"', 'license = "GPL-3.0-only"').replace(
            'implementation_method = "clean_room"',
            'implementation_method = "audited_bsd_adaptation"',
        ),
    )

    # When: the ledger is checked.
    with pytest.raises(ProvenanceLedgerError, match="BSD-3-Clause"):
        check_provenance_ledger(ledger_path)

    # Then: the compatibility requirement is named by the validation error.


def test_audited_adaptation_passes_when_license_field_identifies_bsd(
    tmp_path: Path,
) -> None:
    """Accept an audited adaptation that identifies BSD-3-Clause."""
    # Given: an audited adaptation whose license note identifies BSD-3-Clause.
    ledger_path = _write_ledger(
        tmp_path,
        VALID_LEDGER.replace(
            'license = "N/A"',
            'license = "Public specification; BSD-3-Clause adaptation"',
        ).replace(
            'implementation_method = "clean_room"',
            'implementation_method = "audited_bsd_adaptation"',
        ),
    )

    # When: the ledger is checked.
    check_provenance_ledger(ledger_path)

    # Then: the SPDX identifier within the license note satisfies the policy.


def test_quarantined_source_is_rejected_when_marked_as_permitted(
    tmp_path: Path,
) -> None:
    """Reject oracle-only material that is incorrectly permitted."""
    # Given: an oracle-only source incorrectly marked as permitted.
    ledger_path = _write_ledger(
        tmp_path,
        VALID_LEDGER.replace(
            'implementation_method = "clean_room"',
            'implementation_method = "oracle_only"',
        ),
    )

    # When: the ledger is checked.
    with pytest.raises(ProvenanceLedgerError, match="oracle_only"):
        check_provenance_ledger(ledger_path)

    # Then: incompatible method/status pairing is rejected.


def test_required_prohibited_boundary_is_rejected_when_record_is_removed(
    tmp_path: Path,
) -> None:
    """Reject removal of the source-available InSAR.dev boundary."""
    # Given: a ledger with its required prohibited boundary removed.
    ledger_path = _write_ledger(
        tmp_path,
        _remove_record(
            VALID_LEDGER,
            "InSAR.dev source-available processing algorithms",
        ),
    )

    # When: the ledger is checked.
    with pytest.raises(ProvenanceLedgerError, match="insardev-source-available"):
        check_provenance_ledger(ledger_path)

    # Then: the stable missing policy identity is reported.


def test_required_quarantined_boundary_is_rejected_when_record_is_removed(
    tmp_path: Path,
) -> None:
    """Reject removal of the quarantined GMTSAR oracle boundary."""
    # Given: a ledger with its required GMTSAR oracle boundary removed.
    ledger_path = _write_ledger(
        tmp_path,
        _remove_record(VALID_LEDGER, "GMTSAR executable and source oracle"),
    )

    # When: the ledger is checked.
    with pytest.raises(ProvenanceLedgerError, match="gmtsar-oracle"):
        check_provenance_ledger(ledger_path)

    # Then: the stable missing policy identity is reported.


def test_gpl_license_is_rejected_when_clean_room_source_is_permitted(
    tmp_path: Path,
) -> None:
    """Reject incompatible licensing hidden behind clean-room classification."""
    # Given: a permitted clean-room entry that declares GPL-only licensing.
    ledger_path = _write_ledger(
        tmp_path,
        VALID_LEDGER.replace('license = "N/A"', 'license = "GPL-3.0-only"', 1),
    )

    # When: the ledger is checked.
    with pytest.raises(ProvenanceLedgerError, match="incompatible license"):
        check_provenance_ledger(ledger_path)

    # Then: implementation method does not bypass license compatibility.
