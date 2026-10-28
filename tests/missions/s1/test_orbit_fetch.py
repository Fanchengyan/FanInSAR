"""Tests for automatic covering Sentinel-1 EOF acquisition (PROPOSAL-0054)."""

from __future__ import annotations

import io
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import pytest

from faninsar.missions.s1.orbit_fetch import (
    S1_ORBITS_ORIGIN,
    S1OrbitDownloadError,
    ensure_covering_orbit_paths,
    parse_eof_filename,
    select_covering_eof_keys,
)
from faninsar.missions.s1.search import select_orbit_bindings

PRE_2017_POEORB = (
    "AUX_POEORB/"
    "S1A_OPER_AUX_POEORB_OPOD_20210314T232123_V20161207T225943_20161209T005943.EOF"
)
CURRENT_POEORB = (
    "AUX_POEORB/"
    "S1A_OPER_AUX_POEORB_OPOD_20260101T070407_V20251211T225942_20251213T005942.EOF"
)
COVERING_POEORB = (
    "AUX_POEORB/"
    "S1A_OPER_AUX_POEORB_OPOD_20240102T000000_V20231231T230000_20240101T010000.EOF"
)
NEIGHBOR_POEORB = (
    "AUX_POEORB/"
    "S1A_OPER_AUX_POEORB_OPOD_20240103T000000_V20240101T230000_20240102T010000.EOF"
)
COVERING_RESORB = (
    "AUX_RESORB/"
    "S1A_OPER_AUX_RESORB_OPOD_20240101T120000_V20231231T230000_20240101T010000.EOF"
)
COVERING_S1B_POEORB = (
    "AUX_POEORB/"
    "S1B_OPER_AUX_POEORB_OPOD_20240102T000000_V20231231T230000_20240101T010000.EOF"
)

_SCENE_INTERVAL = (
    datetime(2024, 1, 1, tzinfo=UTC),
    datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
)


def _eof_xml(start: str, stop: str) -> bytes:
    """Return a minimal ESA EOF body covering ``start`` through ``stop``."""
    return (
        '<?xml version="1.0"?>'
        "<Earth_Explorer_File><Data_Block><List_of_OSVs>"
        f"<OSV><UTC>UTC={start}</UTC><X>1</X><Y>2</Y><Z>3</Z>"
        "<VX>4</VX><VY>5</VY><VZ>6</VZ></OSV>"
        f"<OSV><UTC>UTC={stop}</UTC><X>1</X><Y>2</Y><Z>3</Z>"
        "<VX>4</VX><VY>5</VY><VZ>6</VZ></OSV>"
        "</List_of_OSVs></Data_Block></Earth_Explorer_File>"
    ).encode()


def _list_xml(
    keys: list[str], *, truncated: bool = False, token: str | None = None
) -> bytes:
    """Return a ListObjects v2 payload for the given keys."""
    contents = "".join(f"<Contents><Key>{key}</Key></Contents>" for key in keys)
    token_xml = (
        f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    )
    truncated_xml = "true" if truncated else "false"
    return (
        '<?xml version="1.0"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<IsTruncated>{truncated_xml}</IsTruncated>{token_xml}{contents}"
        "</ListBucketResult>"
    ).encode()


class _FakeResponse(io.BytesIO):
    """Minimal urllib response exposing status and headers."""

    def __init__(self, payload: bytes, url: str, status: int = 200) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Encoding": "identity"}
        self._url = url

    def geturl(self) -> str:
        """Return the request URL."""
        return self._url

    def getcode(self) -> int:
        """Return the HTTP status."""
        return self.status


class _FakeS3Opener:
    """In-memory s1-orbits S3 listing and object store."""

    def __init__(
        self,
        objects: dict[str, bytes],
        *,
        get_status: dict[str, int] | None = None,
        list_status: int = 200,
    ) -> None:
        self.objects = objects
        self.get_status = get_status or {}
        self.list_status = list_status
        self.requests: list[urllib.request.Request] = []

    def open(
        self, request: urllib.request.Request, timeout: float | None = None
    ) -> _FakeResponse:
        """Serve one listing or object GET without leaving the test process."""
        del timeout
        self.requests.append(request)
        url = request.full_url
        parsed = urllib.parse.urlsplit(url)
        host = (parsed.hostname or "").lower()
        if "s1qc" in host:
            raise urllib.error.HTTPError(
                url, 401, "Unauthorized", hdrs={}, fp=io.BytesIO()
            )
        if host not in {
            "s1-orbits.s3.amazonaws.com",
            "s1-orbits.s3.us-west-2.amazonaws.com",
        }:
            raise urllib.error.HTTPError(
                url, 404, "Not Found", hdrs={}, fp=io.BytesIO()
            )
        query = dict(urllib.parse.parse_qsl(parsed.query))
        if query.get("list-type") == "2":
            if self.list_status != 200:
                raise urllib.error.HTTPError(
                    url, self.list_status, "list failed", hdrs={}, fp=io.BytesIO()
                )
            prefix = query.get("prefix", "")
            keys = [key for key in self.objects if key.startswith(prefix)]
            return _FakeResponse(_list_xml(keys), url)
        key = parsed.path.lstrip("/")
        status = self.get_status.get(key, 200)
        if status != 200 or key not in self.objects:
            raise urllib.error.HTTPError(
                url,
                status if status != 200 else 404,
                "object failed",
                hdrs={},
                fp=io.BytesIO(),
            )
        return _FakeResponse(self.objects[key], url)


def _install_fake_s3(
    monkeypatch: pytest.MonkeyPatch, opener: _FakeS3Opener
) -> _FakeS3Opener:
    """Route both listing and complete-file GET through the in-memory opener."""
    monkeypatch.setattr("urllib.request.build_opener", lambda *_handlers: opener)
    monkeypatch.setattr("urllib.request.urlopen", opener.open)
    return opener


def test_parse_eof_filename_reads_mission_family_and_validity_window() -> None:
    """Basename grammar exposes the SAFE mission prefix and validity interval."""
    parsed = parse_eof_filename(PRE_2017_POEORB)
    assert parsed is not None
    assert parsed.mission_id == "S1A"
    assert parsed.family == "POEORB"
    assert parsed.valid_start == datetime(2016, 12, 7, 22, 59, 43, tzinfo=UTC)
    assert parsed.valid_stop == datetime(2016, 12, 9, 0, 59, 43, tzinfo=UTC)


def test_parse_eof_filename_accepts_s1d_prefix() -> None:
    """Sentinel-1D covering EOFs share the ESA AUX_*ORB grammar."""
    parsed = parse_eof_filename(
        "S1D_OPER_AUX_POEORB_OPOD_20240102T000000_V20231231T230000_20240101T010000.EOF"
    )
    assert parsed is not None
    assert parsed.mission_id == "S1D"
    assert parsed.family == "POEORB"


def test_select_covering_keys_keeps_only_the_scene_interval() -> None:
    """Neighboring validity windows are not selected for download."""
    selected = select_covering_eof_keys(
        [COVERING_POEORB, NEIGHBOR_POEORB, COVERING_RESORB, COVERING_S1B_POEORB],
        interval=_SCENE_INTERVAL,
        mission_id="S1A",
    )
    assert selected[0] == COVERING_POEORB
    assert NEIGHBOR_POEORB not in selected
    assert COVERING_S1B_POEORB not in selected


def test_select_covering_keys_ranks_poeorb_before_resorb() -> None:
    """Precise files win when both families cover the same sensing window."""
    selected = select_covering_eof_keys(
        [COVERING_RESORB, COVERING_POEORB],
        interval=_SCENE_INTERVAL,
        mission_id="S1A",
    )
    assert selected[0] == COVERING_POEORB


def test_select_covering_keys_rejects_other_satellites() -> None:
    """Overlapping S1B windows must not bind an S1A acquisition."""
    selected = select_covering_eof_keys(
        [COVERING_S1B_POEORB],
        interval=_SCENE_INTERVAL,
        mission_id="S1A",
    )
    assert selected == []


def test_recorded_s1_orbits_fixture_keys_parse() -> None:
    """Feasibility-probe keys remain the recorded covering-file fixtures."""
    pre2017 = parse_eof_filename(PRE_2017_POEORB)
    current = parse_eof_filename(CURRENT_POEORB)
    assert pre2017 is not None and current is not None
    assert pre2017.valid_start.year == 2016
    assert current.valid_start == datetime(2025, 12, 11, 22, 59, 42, tzinfo=UTC)


def test_orbit_selection_rejects_wrong_mission_prefix(tmp_path: Any) -> None:
    """Admission matches SAFE mission_id, not just vector coverage."""
    body = _eof_xml("2023-12-31T23:00:00", "2024-01-01T01:00:00")
    s1b = tmp_path / "S1B_OPER_AUX_POEORB_TEST.EOF"
    s1b.write_bytes(body)
    with pytest.raises(ValueError, match="do not cover"):
        select_orbit_bindings(
            [s1b],
            {"20240101": _SCENE_INTERVAL},
            mission_ids={"20240101": "S1A"},
        )


def test_orbit_selection_allows_uncovered_dates_for_embedded_fallback(
    tmp_path: Any,
) -> None:
    """Auto-fetch may bind only the dates that have a covering EOF."""
    body = _eof_xml("2023-12-31T23:00:00", "2024-01-01T01:00:00")
    precise = tmp_path / "S1A_OPER_AUX_POEORB_TEST.EOF"
    precise.write_bytes(body)
    selected = select_orbit_bindings(
        {"20240101": precise},
        {
            "20240101": _SCENE_INTERVAL,
            "20240102": (
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 2, 0, 1, tzinfo=UTC),
            ),
        },
        mission_ids={"20240101": "S1A", "20240102": "S1A"},
        require_coverage=False,
    )
    assert selected == {"20240101": precise}


def test_download_covering_orbits_gets_s1_orbits_anonymously(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Missing EOF files are listed and GET from s1-orbits without auth headers."""
    payload = _eof_xml("2023-12-31T23:00:00", "2024-01-01T01:00:00")
    opener = _install_fake_s3(
        monkeypatch,
        _FakeS3Opener(
            {
                COVERING_POEORB: payload,
                NEIGHBOR_POEORB: payload,
                COVERING_RESORB: payload,
                COVERING_S1B_POEORB: payload,
            }
        ),
    )
    cache = tmp_path / "orbits"
    selected = ensure_covering_orbit_paths(
        intervals={"20240101": _SCENE_INTERVAL},
        mission_ids={"20240101": "S1A"},
        cache_dir=cache,
    )
    hosts = {
        urllib.parse.urlsplit(request.full_url).hostname for request in opener.requests
    }
    paths = [
        urllib.parse.urlsplit(request.full_url).path for request in opener.requests
    ]
    headers = [dict(request.header_items()) for request in opener.requests]
    assert hosts == {"s1-orbits.s3.amazonaws.com"}
    assert all("s1qc" not in (request.full_url or "") for request in opener.requests)
    assert any(
        path.endswith(COVERING_POEORB.rsplit("/", 1)[1]) or COVERING_POEORB in path
        for path in paths
    )
    assert not any(
        NEIGHBOR_POEORB.rsplit("/", 1)[1] in path for path in paths if path != "/"
    )
    assert all("Authorization" not in header for header in headers)
    bound = selected["20240101"]
    assert bound.name.startswith("S1A_OPER_AUX_POEORB_")
    assert bound.is_file()


def test_download_does_not_get_eof_bytes_from_s1qc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """s1qc file GET is 401 and must not be attempted."""
    payload = _eof_xml("2023-12-31T23:00:00", "2024-01-01T01:00:00")
    opener = _install_fake_s3(monkeypatch, _FakeS3Opener({COVERING_POEORB: payload}))
    ensure_covering_orbit_paths(
        intervals={"20240101": _SCENE_INTERVAL},
        mission_ids={"20240101": "S1A"},
        cache_dir=tmp_path / "orbits",
    )
    assert opener.requests
    assert all(
        "s1qc.asf.alaska.edu" not in request.full_url for request in opener.requests
    )


def test_http_404_is_a_download_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A failed EOF GET is a download error naming dates, host, and status."""
    opener = _install_fake_s3(
        monkeypatch,
        _FakeS3Opener(
            {COVERING_POEORB: _eof_xml("2023-12-31T23:00:00", "2024-01-01T01:00:00")},
            get_status={COVERING_POEORB: 404},
        ),
    )
    with pytest.raises(S1OrbitDownloadError, match="20240101") as caught:
        ensure_covering_orbit_paths(
            intervals={"20240101": _SCENE_INTERVAL},
            mission_ids={"20240101": "S1A"},
            cache_dir=tmp_path / "orbits",
        )
    error = caught.value
    assert error.status == 404
    assert "s1-orbits" in error.host
    assert "20240101" in error.dates
    assert opener.requests


def test_http_401_is_a_download_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Unauthorized listing or GET is reported as a download failure."""
    _install_fake_s3(
        monkeypatch,
        _FakeS3Opener(
            {COVERING_POEORB: _eof_xml("2023-12-31T23:00:00", "2024-01-01T01:00:00")},
            list_status=401,
        ),
    )
    with pytest.raises(S1OrbitDownloadError, match="HTTP 401") as caught:
        ensure_covering_orbit_paths(
            intervals={"20240101": _SCENE_INTERVAL},
            mission_ids={"20240101": "S1A"},
            cache_dir=tmp_path / "orbits",
        )
    assert caught.value.status == 401


def test_catalog_miss_leaves_embedded_vectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """No covering key on s1-orbits keeps SAFE-embedded restituted vectors."""
    _install_fake_s3(monkeypatch, _FakeS3Opener({NEIGHBOR_POEORB: b"unused"}))
    selected = ensure_covering_orbit_paths(
        intervals={"20240101": _SCENE_INTERVAL},
        mission_ids={"20240101": "S1A"},
        cache_dir=tmp_path / "orbits",
    )
    assert selected == {}


def test_caller_covering_resorb_is_not_upgraded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Explicit local covering files are a pin, not a hint for POEORB upgrade."""
    restituted = tmp_path / "S1A_OPER_AUX_RESORB_PIN.EOF"
    restituted.write_bytes(_eof_xml("2023-12-31T23:00:00", "2024-01-01T01:00:00"))
    opener = _install_fake_s3(
        monkeypatch,
        _FakeS3Opener(
            {COVERING_POEORB: _eof_xml("2023-12-31T23:00:00", "2024-01-01T01:00:00")}
        ),
    )
    selected = ensure_covering_orbit_paths(
        intervals={"20240101": _SCENE_INTERVAL},
        mission_ids={"20240101": "S1A"},
        cache_dir=tmp_path / "orbits",
        existing={"20240101": restituted},
    )
    assert selected == {"20240101": restituted}
    object_gets = [
        request
        for request in opener.requests
        if urllib.parse.urlsplit(request.full_url).path.endswith(".EOF")
    ]
    assert object_gets == []


def test_skip_if_present_reuses_cached_covering_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A cached covering EOF is reused without another object GET."""
    cache = tmp_path / "orbits"
    cache.mkdir()
    cached = cache / COVERING_POEORB.rsplit("/", 1)[1]
    cached.write_bytes(_eof_xml("2023-12-31T23:00:00", "2024-01-01T01:00:00"))
    opener = _install_fake_s3(monkeypatch, _FakeS3Opener({COVERING_POEORB: b"stale"}))
    selected = ensure_covering_orbit_paths(
        intervals={"20240101": _SCENE_INTERVAL},
        mission_ids={"20240101": "S1A"},
        cache_dir=cache,
    )
    assert selected["20240101"] == cached
    object_gets = [
        request
        for request in opener.requests
        if urllib.parse.urlsplit(request.full_url).path.endswith(".EOF")
    ]
    assert object_gets == []


def test_manifest_reuse_skips_auto_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Published-stack reuse does not download unless refresh is requested."""
    opener = _install_fake_s3(
        monkeypatch,
        _FakeS3Opener(
            {COVERING_POEORB: _eof_xml("2023-12-31T23:00:00", "2024-01-01T01:00:00")}
        ),
    )
    selected = ensure_covering_orbit_paths(
        intervals={"20240101": _SCENE_INTERVAL},
        mission_ids={"20240101": "S1A"},
        cache_dir=tmp_path / "orbits",
        auto_fetch=False,
    )
    assert selected == {}
    assert opener.requests == []


def _dummy_safe(tmp_path: Any) -> Path:
    """Create a ZIP path that catalogs as a 2024-01-01 Sentinel-1 scene."""
    import zipfile

    safe = (
        tmp_path
        / "S1A_IW_SLC__1SDV_20240101T000000_20240101T000002_000001_000000_0000.zip"
    )
    with zipfile.ZipFile(safe, "w") as archive:
        archive.writestr("placeholder.txt", "SAFE")
    return safe


def _fake_product(*, mission_id: str = "S1A") -> Any:
    """Return a SAFE handle with a covering sensing window and no footprints."""
    swath = type(
        "Swath",
        (),
        {
            "sensing_start": _SCENE_INTERVAL[0],
            "sensing_stop": _SCENE_INTERVAL[1],
            "bursts": (),
        },
    )()
    return type(
        "Product",
        (),
        {"mission_id": mission_id, "swaths": (swath,)},
    )()


def test_from_safes_prepare_does_not_fetch_orbits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Footprint preparation may use embedded vectors; fetch waits for coregistration."""
    from faninsar.missions.s1.stack import S1Stack

    opener = _install_fake_s3(
        monkeypatch,
        _FakeS3Opener(
            {COVERING_POEORB: _eof_xml("2023-12-31T23:00:00", "2024-01-01T01:00:00")}
        ),
    )
    monkeypatch.setattr(
        "faninsar.missions.s1.safe.open_safe_product",
        lambda *_args, **_kwargs: _fake_product(),
    )
    stack = S1Stack.from_safes([_dummy_safe(tmp_path)], work_dir=tmp_path / "work")
    stack._prepare_mission()
    assert opener.requests == []
    assert not stack.config.extra.get("orbit_paths")


def test_precise_orbit_stage_downloads_and_binds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """coregister_scenes downloads covering EOFs and binds config.extra orbit_paths."""
    from faninsar.missions.s1.stack import S1Stack

    payload = _eof_xml("2023-12-31T23:00:00", "2024-01-01T01:00:00")
    opener = _install_fake_s3(monkeypatch, _FakeS3Opener({COVERING_POEORB: payload}))
    monkeypatch.setattr(
        "faninsar.missions.s1.safe.open_safe_product",
        lambda *_args, **_kwargs: _fake_product(),
    )
    stack = S1Stack.from_safes([_dummy_safe(tmp_path)], work_dir=tmp_path / "work")
    stack._ensure_precise_orbits()
    bound = stack.config.extra["orbit_paths"]
    assert "20240101" in bound
    assert Path(bound["20240101"]).name.startswith("S1A_OPER_AUX_POEORB_")
    assert opener.requests
    assert all("s1qc" not in request.full_url for request in opener.requests)


def test_coregister_scenes_invokes_precise_orbit_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The first precise-orbit stage is coregister_scenes."""
    from faninsar.missions.s1.stack import S1Stack

    monkeypatch.setattr(
        "faninsar.missions.s1.safe.open_safe_product",
        lambda *_args, **_kwargs: _fake_product(),
    )
    stack = S1Stack.from_safes([_dummy_safe(tmp_path)], work_dir=tmp_path / "work")
    called: list[str] = []
    monkeypatch.setattr(
        stack,
        "_ensure_precise_orbits",
        lambda: called.append("orbits"),
    )
    monkeypatch.setattr(stack, "_ensure_prepared", lambda: None)
    stack.coregister_scenes(dates=[])
    assert called == ["orbits"]


def test_open_manifest_reuse_does_not_auto_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Reopening a published stack skips orbit download unless refresh=True."""
    from faninsar.missions.s1.stack import S1Stack

    opener = _install_fake_s3(
        monkeypatch,
        _FakeS3Opener(
            {COVERING_POEORB: _eof_xml("2023-12-31T23:00:00", "2024-01-01T01:00:00")}
        ),
    )
    monkeypatch.setattr(
        "faninsar.missions.s1.safe.open_safe_product",
        lambda *_args, **_kwargs: _fake_product(),
    )
    stack = S1Stack.from_safes([_dummy_safe(tmp_path)], work_dir=tmp_path / "work")
    stack.config.extra["orbit_auto_fetch"] = False
    stack._ensure_precise_orbits()
    assert not stack.config.extra.get("orbit_paths")
    assert opener.requests == []


def test_s1qc_origin_is_rejected_for_object_get(tmp_path: Any) -> None:
    """The dedicated catalog does not admit s1qc EOF hrefs."""
    from faninsar import remote
    from faninsar.missions.s1.orbit_fetch import ensure_s1_orbits_catalog

    ensure_s1_orbits_catalog()
    asset = remote.RemoteAsset(
        provider="s1-orbits",
        catalog="s1-orbits",
        collection="AUX_POEORB",
        item_id="eof",
        key="eof",
        href="https://s1qc.asf.alaska.edu/aux_poeorb/S1A_OPER_AUX_POEORB_TEST.EOF",
        auth_profile="anonymous",
    )
    with pytest.raises(remote.RemoteAccessError, match="unregistered_endpoint"):
        remote.download(asset, tmp_path / "orbit.EOF")
