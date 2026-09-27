"""Contract tests for the anonymous s1-orbits catalog (PROPOSAL-0054)."""

from __future__ import annotations

from typing import Any

import pytest

from faninsar import remote
from faninsar.missions.s1.search import ensure_asf_catalog
from faninsar.remote.catalog import _ADAPTERS
from faninsar.remote.providers.s1_orbits import (
    S1_ORBITS_CATALOG,
    S1_ORBITS_ORIGIN,
    ensure_s1_orbits_catalog,
)


def test_s1_orbits_catalog_is_anonymous_only() -> None:
    """The dedicated origin admits anonymous and rejects scene Earthdata."""
    name = ensure_s1_orbits_catalog()
    adapter = _ADAPTERS[name]
    assert name == S1_ORBITS_CATALOG
    assert adapter.profiles == ("anonymous",)
    assert S1_ORBITS_ORIGIN in adapter.origins
    assert getattr(adapter, "_allow_anonymous_delivery", False) is False
    asset = remote.RemoteAsset(
        provider="s1-orbits",
        catalog=name,
        collection="AUX_POEORB",
        item_id="eof",
        key="eof",
        href=f"{S1_ORBITS_ORIGIN}/AUX_POEORB/S1A_OPER_AUX_POEORB_TEST.EOF",
        auth_profile="earthdata-asf",
    )
    with pytest.raises(remote.RemoteAccessError, match="unknown_auth_profile"):
        remote.download(asset, destination=__import__("pathlib").Path("unused.EOF"))


def test_s1_orbits_origin_is_not_added_to_sentinel1_asf() -> None:
    """Scene ASF registration must not inherit the public orbit bucket."""
    ensure_asf_catalog(product_type="burst")
    ensure_s1_orbits_catalog()
    asf = _ADAPTERS["sentinel1-asf"]
    origins = {str(origin).rstrip("/") for origin in asf.origins}
    redirect = {str(origin).rstrip("/") for origin in asf.redirect_origins}
    assert S1_ORBITS_ORIGIN not in origins
    assert S1_ORBITS_ORIGIN not in redirect
    assert "s1qc.asf.alaska.edu" not in " ".join(origins | redirect)


def test_s1_orbits_unknown_profile_fails_closed() -> None:
    """Copying the Stack scene profile onto this catalog is rejected."""
    name = ensure_s1_orbits_catalog()
    adapter = _ADAPTERS[name]
    assert "earthdata-asf" not in adapter.profiles
    assert "sentinel1-asf" not in {adapter.provider, name}
