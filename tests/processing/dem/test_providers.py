# ruff: noqa: D103, INP001, PT011
"""Provider registry and Planetary Computer contract tests."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from faninsar.processing.dem.providers import (
    GLO30_PC,
    GLO90_PC,
    PC_STAC_URL,
    PcStacSource,
    parse_selection,
)


def test_glo_pc_registry_uses_authoritative_collections() -> None:
    assert GLO30_PC.collection_id == "cop-dem-glo-30"
    assert GLO30_PC.asset_key == "data"
    assert GLO30_PC.tile_shape == (3600, 3600)
    assert GLO90_PC.collection_id == "cop-dem-glo-90"
    assert GLO90_PC.asset_key == "data"
    assert GLO90_PC.tile_shape == (1200, 1200)


def test_parse_selection_preserves_canonical_product_provider_grammar() -> None:
    assert parse_selection("glo30:pc") == ("glo30", "pc")
    assert parse_selection("glo30") == ("glo30", None)
    with pytest.raises(ValueError):
        parse_selection("glo30:")


def test_pc_construction_does_not_open_network() -> None:
    source = PcStacSource("glo30", "pc")
    assert source.collection_id == "cop-dem-glo-30"


def test_pc_plan_is_immutable_and_zero_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planning records STAC identity but defers discovery to materialization."""
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("PC planning must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    plan = GLO30_PC.plan((170.0, -1.0, -170.0, 1.0))
    assert plan.collection == "cop-dem-glo-30"
    assert plan.asset == "data"
    assert plan.bounds == (170.0, -1.0, -170.0, 1.0)
    assert len(plan.windows) == 2
    assert plan.endpoint_identity == PC_STAC_URL
    with pytest.raises(AttributeError):
        plan.collection = "other"  # type: ignore[misc]


@dataclass
class _Asset:
    href: str
    extra_fields: dict[str, int]


@dataclass
class _Item:
    id: str
    assets: dict[str, _Asset]


class _Search:
    def __init__(self, item: _Item) -> None:
        self.item = item

    def items(self) -> tuple[_Item, ...]:
        return (self.item,)


class _Client:
    def __init__(self, item: _Item) -> None:
        self.item = item
        self.bboxes: list[list[float]] = []

    def search(self, *, collections: list[str], bbox: list[float]) -> _Search:
        assert collections == ["cop-dem-glo-30"]
        self.bboxes.append(bbox)
        return _Search(self.item)


def test_pc_discovery_queries_both_windows_and_redacts_sas_identity() -> None:
    item = _Item(
        "tile-1",
        {
            "data": _Asset(
                "https://elevationeuwest.blob.core.windows.net/tile.tif?sig=secret",
                {},
            )
        },
    )
    client = _Client(item)
    resources = GLO30_PC.discover((170.0, -1.0, -170.0, 1.0), client=client)
    assert len(client.bboxes) == 2
    assert len(resources) == 1
    assert "secret" not in resources[0].identity


@pytest.mark.parametrize(
    ("product", "collection", "shape"),
    [
        ("glo30", "cop-dem-glo-30", (3600, 3600)),
        ("glo90", "cop-dem-glo-90", (1200, 1200)),
    ],
)
def test_pc_materializes_signed_cog_directly_to_projected_grid(
    product: str,
    collection: str,
    shape: tuple[int, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signed PC COG is fetched and sampled once onto a UTM target."""
    import rasterio
    from affine import Affine
    from pyproj import Transformer

    asset_path = tmp_path / "source.tif"
    with rasterio.open(
        asset_path,
        "w",
        driver="GTiff",
        height=12,
        width=12,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=Affine(1.0 / 6.0, 0.0, 9.0, 0.0, -1.0 / 6.0, 41.0),
    ) as dataset:
        dataset.write(np.arange(144, dtype=np.float32).reshape(12, 12), 1)

    @dataclass
    class _SignedAsset:
        href: str
        extra_fields: dict[str, object]

    @dataclass
    class _SignedItem:
        id: str
        assets: dict[str, _SignedAsset]

    item = _SignedItem(
        "pc-tile",
        {"data": _SignedAsset("https://elevationeuwest.blob.core.windows.net/tile.tif?sig=x", {})},
    )

    class _Search:
        def items(self) -> tuple[_SignedItem, ...]:
            return (item,)

    class _Client:
        def search(self, *, collections: list[str], bbox: list[float]) -> _Search:
            assert collections == [collection]
            assert len(bbox) == 4
            return _Search()

    def fake_sign(value: _SignedItem) -> _SignedItem:
        return value

    def fake_fetch(resource: object, *, cache_dir: Path, max_bytes: int) -> Path:
        assert max_bytes > 0
        target = cache_dir / resource.cache_path  # type: ignore[attr-defined]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(asset_path.read_bytes())
        return target

    monkeypatch.setattr(
        "faninsar.processing.dem.providers.fetch_asset", fake_fetch
    )
    from faninsar.processing.dem import GridSpec, materialize_source

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    center_x, center_y = transformer.transform(10.0, 40.0)
    grid = GridSpec(
        "EPSG:32632",
        Affine(30.0, 0.0, center_x - 30.0, 0.0, -30.0, center_y + 30.0),
        shape=(2, 2),
    )
    source = PcStacSource(product)
    raster = materialize_source(
        source,
        grid,
        cache_dir=tmp_path / "cache",
        client=_Client(),
        signer=fake_sign,
    )
    assert raster.grid == grid
    assert raster.provenance["collection"] == collection
    assert raster.provenance["asset"] == "data"
    assert raster.provenance["resampling"] == "direct-source-target"
    assert (tmp_path / "cache" / collection).is_dir()
    assert not (tmp_path / "cache" / "EPSG:4326").exists()
