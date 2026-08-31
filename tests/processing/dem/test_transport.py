# ruff: noqa: D103, INP001, TC003, E501
"""Public boundary tests for the DEM provider transport."""

from __future__ import annotations

from pathlib import Path

import pytest

from faninsar.processing.dem.transport import (
    BoundedTransferError,
    stream_to_cache,
    validate_https_origin,
)


def test_stream_to_cache_accepts_exact_http_payload(tmp_path: Path) -> None:
    destination = tmp_path / "tile.tif"
    written = stream_to_cache(
        [b"abc", b"def"], destination, max_bytes=6, expected_length=6
    )
    assert written == 6
    assert destination.read_bytes() == b"abcdef"


def test_stream_to_cache_rejects_unknown_length_over_budget(tmp_path: Path) -> None:
    with pytest.raises(BoundedTransferError, match="maximum"):
        stream_to_cache([b"1234", b"56"], tmp_path / "tile", max_bytes=5)


def test_validate_https_origin_rejects_redirect_host() -> None:
    with pytest.raises(ValueError, match="origin"):
        validate_https_origin("https://evil.example/a", {"planetarycomputer.microsoft.com"})


def test_http_range_requires_exact_protocol_length(tmp_path: Path) -> None:
    with pytest.raises(BoundedTransferError, match="Content-Range"):
        stream_to_cache(
            [b"abcd"],
            tmp_path / "tile",
            max_bytes=10,
            expected_length=4,
            status_code=206,
            content_range="bytes 0-4/5",
        )
