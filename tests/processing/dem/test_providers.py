# ruff: noqa: D103, INP001, PT011
"""Provider registry and Planetary Computer contract tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from faninsar.processing.dem.providers import (
    GLO30_PC,
    GLO90_PC,
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
