"""Installation and import boundaries for the remote MVP."""

from __future__ import annotations

import subprocess
import sys


def test_base_remote_import_does_not_load_optional_provider_sdks() -> None:
    """The base namespace remains importable without provider extras."""
    script = """
import sys
import faninsar.remote
import faninsar.remote.providers

blocked = ("asf_search", "planetary_computer", "pystac_client")
assert not any(
    module == name or module.startswith(name + ".")
    for module in sys.modules
    for name in blocked
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_provider_registration_exports_share_catalog_registry() -> None:
    """Provider convenience exports register adapters for remote.search."""
    from faninsar import remote

    assert remote.register_cmr_catalog is remote.register_cmr_collection
    assert remote.register_asf_catalog is remote.register_asf_search_catalog
    assert remote.register_pc_catalog is remote.register_planetary_computer

    from faninsar.remote import providers

    assert providers.register_asf_catalog is remote.register_asf_catalog
    assert providers.register_pc_catalog is remote.register_pc_catalog


def test_optional_provider_construction_does_not_import_sdk() -> None:
    """Adapter construction is offline; SDK imports belong to I/O methods."""
    from faninsar import remote

    remote.register_planetary_computer("install-boundary-pc")
    assert "planetary_computer" not in sys.modules
    assert "pystac_client" not in sys.modules
