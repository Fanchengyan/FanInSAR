"""CDSE STAC adapter seams (PROPOSAL-0061).

The adapter is the internal ``sentinel1-cdse`` catalog: anonymous STAC search
rewrites ``s3://eodata`` measurement/annotation hrefs to HTTPS OData
``Nodes(...)/$value`` URLs and ``file:size``.  Byte-range transfer is owned by
``RangeFetcher``, not ``adapter.fetch``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from faninsar import remote
from faninsar.data.query import BoundingBox
from faninsar.remote.auth import _no_auth
from faninsar.remote.catalog import _ADAPTERS, _REGISTRY_LOCK
from faninsar.remote.errors import RemoteAccessError
from faninsar.remote.fetch import RangeSpan
from faninsar.remote.providers.cdse import (
    CDSE_CATALOG,
    CDSE_COLLECTION,
    CDSE_STAC_ENDPOINT,
    CDSESearchAdapter,
    ensure_cdse_catalog,
    fetch_cdse_ranges,
    request_cdse_token,
)

ROI = BoundingBox(100.0, 44.0, 100.4, 44.4, crs="EPSG:4326")

_PRODUCT_ID = "S1C_IW_SLC__1SDV_20251120T111823_20251120T111850_005097_00A19E_DDCB"
_ODATA_MEASUREMENT = (
    "https://zipper.dataspace.copernicus.eu/odata/v1/"
    "Products(078a8ae1-47f1-40ed-87f6-99a3e62ca665)/"
    f"Nodes({_PRODUCT_ID}.SAFE)/Nodes(measurement)/"
    "Nodes(s1c-iw1-slc-vv-20251120t111823-20251120t111850-005097-00a19e-004.tiff)/$value"
)
_ODATA_ANNOTATION = (
    "https://zipper.dataspace.copernicus.eu/odata/v1/"
    "Products(078a8ae1-47f1-40ed-87f6-99a3e62ca665)/"
    f"Nodes({_PRODUCT_ID}.SAFE)/Nodes(annotation)/"
    "Nodes(s1c-iw1-slc-vv-20251120t111823-20251120t111850-005097-00a19e-004.xml)/$value"
)
_S3_MEASUREMENT = (
    "s3://eodata/Sentinel-1/SAR/IW_SLC__1S/2025/11/20/"
    f"{_PRODUCT_ID}.SAFE/measurement/"
    "s1c-iw1-slc-vv-20251120t111823-20251120t111850-005097-00a19e-004.tiff"
)
_S3_ANNOTATION = (
    "s3://eodata/Sentinel-1/SAR/IW_SLC__1S/2025/11/20/"
    f"{_PRODUCT_ID}.SAFE/annotation/"
    "s1c-iw1-slc-vv-20251120t111823-20251120t111850-005097-00a19e-004.xml"
)


def _odata_pair(swath: str, polarization: str) -> tuple[str, str]:
    """Return OData measurement/annotation hrefs for one swath and polarization."""
    stem = (
        f"s1c-{swath}-slc-{polarization}-"
        "20251120t111823-20251120t111850-005097-00a19e-004"
    )
    product = (
        "https://zipper.dataspace.copernicus.eu/odata/v1/"
        "Products(078a8ae1-47f1-40ed-87f6-99a3e62ca665)/"
        f"Nodes({_PRODUCT_ID}.SAFE)"
    )
    return (
        f"{product}/Nodes(measurement)/Nodes({stem}.tiff)/$value",
        f"{product}/Nodes(annotation)/Nodes({stem}.xml)/$value",
    )


def _stac_feature(
    *,
    with_https_alternate: bool = True,
    relative_orbit: int = 26,
    extra_swaths: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return one CDSE ``sentinel-1-slc`` feature covering the test ROI."""
    assets: dict[str, Any] = {}
    for swath in ("iw1", *extra_swaths):
        measurement_href, annotation_href = (
            (_ODATA_MEASUREMENT, _ODATA_ANNOTATION)
            if swath == "iw1"
            else _odata_pair(swath, "vv")
        )
        measurement: dict[str, Any] = {
            "href": _S3_MEASUREMENT.replace("iw1", swath),
            "type": "image/tiff",
            "file:size": 1_135_276_112,
        }
        annotation: dict[str, Any] = {
            "href": _S3_ANNOTATION.replace("iw1", swath),
            "type": "application/xml",
            "file:size": 845_602,
        }
        if with_https_alternate:
            measurement["alternate"] = {"https": {"href": measurement_href}}
            annotation["alternate"] = {"https": {"href": annotation_href}}
        assets[f"{swath}-vv"] = measurement
        assets[f"schema-product-{swath}-vv"] = annotation
    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": _PRODUCT_ID,
        "collection": CDSE_COLLECTION,
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[100.0, 44.0], [100.4, 44.0], [100.4, 44.4], [100.0, 44.0]]
            ],
        },
        "properties": {
            "datetime": "2025-11-20T11:18:23Z",
            "platform": "sentinel-1c",
            "sar:instrument_mode": "IW",
            "sar:polarizations": ["VV", "VH"],
            "sat:relative_orbit": relative_orbit,
        },
        "assets": assets,
    }


class _FakeCDSEClient:
    """Injected STAC search used by adapter tests."""

    def __init__(self, features: list[dict[str, Any]]) -> None:
        self.features = features
        self.calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return list(self.features)


@pytest.fixture
def isolated_cdse_catalog() -> Any:
    """Remove ``sentinel1-cdse`` for the test and restore afterwards."""
    with _REGISTRY_LOCK:
        previous = _ADAPTERS.pop(CDSE_CATALOG, None)
    try:
        yield
    finally:
        with _REGISTRY_LOCK:
            _ADAPTERS.pop(CDSE_CATALOG, None)
            if previous is not None:
                _ADAPTERS[CDSE_CATALOG] = previous


def test_adapter_rewrites_s3_eodata_to_odata_value_hrefs() -> None:
    """Primary ``s3://eodata`` hrefs become HTTPS OData ``$value`` URLs."""
    client = _FakeCDSEClient([_stac_feature()])
    adapter = CDSESearchAdapter(client=client)
    records = list(adapter.items(spatial=ROI))

    assert len(records) == 1
    assets = records[0]["assets"]
    assert assets["measurement-iw1-vv"]["href"] == _ODATA_MEASUREMENT
    assert assets["annotation-iw1-vv"]["href"] == _ODATA_ANNOTATION
    assert assets["measurement-iw1-vv"]["size"] == 1_135_276_112
    assert assets["annotation-iw1-vv"]["size"] == 845_602
    assert assets["measurement-iw1-vv"]["properties"]["swath"] == "IW1"
    assert "s3://" not in assets["measurement-iw1-vv"]["href"]
    assert "eodata" not in assets["measurement-iw1-vv"]["href"]


def test_adapter_does_not_emit_s3_eodata_as_range_href() -> None:
    """A feature without ``alternate.https`` is dropped, not rewritten to s3."""
    client = _FakeCDSEClient([_stac_feature(with_https_alternate=False)])
    adapter = CDSESearchAdapter(client=client)
    records = list(adapter.items(spatial=ROI))
    assert records == []


def test_ensure_cdse_catalog_is_idempotent(isolated_cdse_catalog: None) -> None:
    """The stable catalog name registers once and can be reused."""
    del isolated_cdse_catalog
    first = ensure_cdse_catalog()
    second = ensure_cdse_catalog()
    assert first == second == CDSE_CATALOG
    adapter = _ADAPTERS[CDSE_CATALOG]
    assert adapter.provider == "cdse"
    assert "cdse" in adapter.profiles
    assert "anonymous" in adapter.profiles
    assert CDSE_STAC_ENDPOINT.startswith("https://stac.dataspace.copernicus.eu")
    origins = {str(origin).rstrip("/") for origin in adapter.origins}
    assert "https://stac.dataspace.copernicus.eu" in origins
    assert "https://zipper.dataspace.copernicus.eu" in origins
    assert "https://identity.dataspace.copernicus.eu" in origins


def test_remote_search_returns_odata_assets(isolated_cdse_catalog: None) -> None:
    """``remote.search`` on the registered catalog exposes OData measurement hrefs."""
    del isolated_cdse_catalog
    adapter = CDSESearchAdapter(client=_FakeCDSEClient([_stac_feature()]))
    with _REGISTRY_LOCK:
        _ADAPTERS[CDSE_CATALOG] = adapter
    found = remote.search(ROI, catalog=CDSE_CATALOG, auth_profile="cdse")
    assert len(found) == 1
    assert found[0].assets["measurement-iw1-vv"].href == _ODATA_MEASUREMENT
    assert found[0].assets["annotation-iw1-vv"].href == _ODATA_ANNOTATION
    assert found[0].assets["measurement-iw1-vv"].size_bytes == 1_135_276_112


def test_complete_object_fetch_is_not_the_roi_path(
    isolated_cdse_catalog: None,
) -> None:
    """ROI canvases use RangeFetcher; complete-object fetch stays closed."""
    del isolated_cdse_catalog
    adapter = CDSESearchAdapter(client=_FakeCDSEClient([_stac_feature()]))
    with _REGISTRY_LOCK:
        _ADAPTERS[CDSE_CATALOG] = adapter
    found = remote.search(ROI, catalog=CDSE_CATALOG, auth_profile="cdse")
    with pytest.raises(RemoteAccessError, match="byte range"):
        adapter.fetch(
            found[0].assets["measurement-iw1-vv"], remote.RemoteResourceBudget()
        )


def test_request_cdse_token_uses_netrc_password_grant(
    isolated_cdse_catalog: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keycloak password grant uses netrc and never attaches Basic to the POST."""
    del isolated_cdse_catalog
    captured: dict[str, Any] = {}

    class _Response:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"access_token": "cdse-access-token", "expires_in": 1800}

    def fake_credentials(host: str) -> tuple[str, str] | None:
        assert host == "identity.dataspace.copernicus.eu"
        return "user@example.test", "secret-password"

    def fake_post(url: str, **kwargs: Any) -> _Response:
        captured["url"] = url
        captured["data"] = dict(kwargs.get("data") or {})
        captured["auth"] = kwargs.get("auth")
        return _Response()

    monkeypatch.setattr(
        "faninsar.remote.providers.cdse._netrc_credentials",
        fake_credentials,
    )
    monkeypatch.setattr("faninsar.remote.providers.cdse.requests.post", fake_post)
    monkeypatch.setattr(
        "faninsar.remote.providers.cdse._token_cache",
        {"token": None, "expires_at": 0.0},
    )

    token = request_cdse_token(force=True)
    assert token == "cdse-access-token"
    assert captured["data"]["grant_type"] == "password"
    assert captured["data"]["client_id"] == "cdse-public"
    assert captured["data"]["username"] == "user@example.test"
    assert captured["data"]["password"] == "secret-password"
    assert captured["auth"] is _no_auth
    assert "Authorization" not in (captured.get("headers") or {})


def test_adapter_filters_relative_orbit() -> None:
    """``relativeOrbit`` keeps matching STAC items and drops the rest."""
    features = [
        _stac_feature(relative_orbit=26),
        _stac_feature(relative_orbit=99),
    ]
    features[1]["id"] = features[1]["id"].replace("005097", "005098")
    client = _FakeCDSEClient(features)
    adapter = CDSESearchAdapter(client=client)
    start = datetime(2025, 11, 1, tzinfo=UTC)
    end = datetime(2025, 11, 30, tzinfo=UTC)
    records = list(
        adapter.items(
            spatial=ROI,
            datetime_range=(start, end),
            filters={"relativeOrbit": 26},
        )
    )
    assert len(records) == 1
    assert records[0]["id"] == _PRODUCT_ID
    assert records[0]["assets"]["measurement-iw1-vv"]["properties"]["swath"] == "IW1"


def test_adapter_emits_every_iw_vv_pair() -> None:
    """Each IW measurement/annotation pair becomes its own OData asset."""
    client = _FakeCDSEClient([_stac_feature(extra_swaths=("iw2", "iw3"))])
    adapter = CDSESearchAdapter(client=client)
    records = list(adapter.items(spatial=ROI))
    assets = records[0]["assets"]
    assert set(assets) == {
        "measurement-iw1-vv",
        "annotation-iw1-vv",
        "measurement-iw2-vv",
        "annotation-iw2-vv",
        "measurement-iw3-vv",
        "annotation-iw3-vv",
    }
    assert assets["measurement-iw2-vv"]["properties"]["swath"] == "IW2"


def test_adapter_rejects_foreign_collection() -> None:
    """Collection is adapter-owned; callers cannot punch a different catalog."""
    adapter = CDSESearchAdapter(client=_FakeCDSEClient([_stac_feature()]))
    with pytest.raises(RemoteAccessError, match="registry-owned"):
        list(adapter.items(spatial=ROI, collections=("sentinel-2-l1c",)))


def test_fetch_cdse_ranges_validates_span_url_before_token(
    isolated_cdse_catalog: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credentialed range GETs validate the span URL before requesting a token."""
    import asyncio

    del isolated_cdse_catalog
    ensure_cdse_catalog()
    called: list[str] = []

    def fake_token(*, force: bool = False) -> str:
        del force
        called.append("token")
        return "unused"

    monkeypatch.setattr(
        "faninsar.remote.providers.cdse.request_cdse_token",
        fake_token,
    )
    span = RangeSpan("https://example.invalid/Nodes(x)/$value", 0, 15, "probe")
    with pytest.raises(RemoteAccessError):
        asyncio.run(fetch_cdse_ranges((span,)))
    assert called == []
