"""Checksum grammar widening (PROPOSAL-0055 slice 1)."""

from __future__ import annotations

import pytest

from faninsar.remote.records import RemoteAsset

MD5 = "d41d8cd98f00b204e9800998ecf8427e"
SHA3 = "ab" * 32


def _asset(checksum: object) -> RemoteAsset:
    return RemoteAsset(
        provider="asf",
        catalog="cmr",
        collection="SENTINEL-1A_SLC",
        item_id="item",
        key="key",
        href="https://example.invalid/a.zip",
        size_bytes=1,
        checksum=checksum,
    )


def test_md5_and_sha3_256_are_qualified() -> None:
    """Provider algorithms recorded at download time qualify for the cache."""
    assert _asset(f"md5:{MD5}").checksum == f"md5:{MD5}"
    assert _asset(f"sha3_256:{SHA3}").checksum == f"sha3_256:{SHA3}"


def test_cmr_dict_shaped_checksum_normalizes() -> None:
    """CMR dict-shaped entries normalize through their Value key."""
    assert _asset({"Value": f"MD5:{MD5.upper()}"}).checksum == f"md5:{MD5}"


def test_unknown_algorithm_and_bad_length_stay_unqualified() -> None:
    """Unknown algorithms and wrong digest lengths count as no checksum."""
    assert _asset("crc32:deadbeef").checksum is None
    assert _asset(f"md5:{MD5[:-1]}").checksum is None
    assert _asset("not-a-checksum").checksum is None


def test_sha256_grammar_unchanged() -> None:
    """The pre-existing sha256 spelling is unaffected by the widening."""
    sha = "ab" * 32
    assert _asset(f"sha256:{sha}").checksum == f"sha256:{sha}"
