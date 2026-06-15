"""Tests for the discovery registry and built-in processor discoverers (M4).

All processor discoverers except ``hyp3`` are stubs whose real-data
behaviour has not been verified against sample products. The tests here
check the **registry contract** and **pattern-matching logic** using
synthetic directory layouts. Real-data validation is tracked in the plan
as a follow-up.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from faninsar.datasets.frame import discovery


class TestRegistry:
    def test_all_builtins_registered(self) -> None:
        names = discovery.available()
        assert "hyp3" in names
        assert "isce" in names
        assert "gamma" in names
        assert "gmtsar" in names
        assert "mintpy" in names
        assert "stamps" in names

    def test_get_returns_instance(self) -> None:
        d = discovery.get("hyp3")
        assert d.name == "hyp3"

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            discovery.get("nope")

    def test_register_overwrites(self) -> None:
        class FakeDiscoverer:
            name = "_test_fake"

            def discover_geometry_product(self, root_dir):
                return Path(root_dir)

        discovery.register(FakeDiscoverer())
        assert "_test_fake" in discovery.available()
        discovery.register(FakeDiscoverer())  # idempotent
        assert discovery.available().count("_test_fake") == 1

    def test_register_requires_name(self) -> None:
        class NoName:
            pass

        with pytest.raises(TypeError):
            discovery.register(NoName())


class TestHyP3Discoverer:
    def test_finds_inc_map(self, tmp_path: Path) -> None:
        prod = tmp_path / "S1_product"
        prod.mkdir()
        (prod / "foo_inc_map_ell.tif").write_bytes(b"")
        d = discovery.get("hyp3")
        assert d.discover_geometry_product(tmp_path) == prod

    def test_no_match_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discovery.get("hyp3").discover_geometry_product(tmp_path)

    def test_discovers_pairs(self, tmp_path: Path) -> None:
        for name in ("a", "b"):
            p = tmp_path / name
            p.mkdir()
            (p / "x_unw_phase.tif").write_bytes(b"")
        pairs = discovery.get("hyp3").discover_pairs(tmp_path)
        assert len(pairs) == 2


class TestISCEDiscoverer:
    def test_finds_geometry_dir(self, tmp_path: Path) -> None:
        geom = tmp_path / "geometry"
        geom.mkdir()
        (geom / "lat.rdr").write_bytes(b"")
        d = discovery.get("isce")
        assert d.discover_geometry_product(tmp_path) == geom

    def test_no_match_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discovery.get("isce").discover_geometry_product(tmp_path)

    def test_discovers_pairs(self, tmp_path: Path) -> None:
        pair = tmp_path / "20191115_20200314"
        pair.mkdir()
        (pair / "filt_topophase.unw").write_bytes(b"")
        pairs = discovery.get("isce").discover_pairs(tmp_path)
        assert len(pairs) == 1


class TestGAMMADiscoverer:
    def test_finds_geometry_dir(self, tmp_path: Path) -> None:
        prod = tmp_path / "gamma_prod"
        prod.mkdir()
        (prod / "dem_seg").write_bytes(b"")
        d = discovery.get("gamma")
        assert d.discover_geometry_product(tmp_path) == prod

    def test_no_match_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discovery.get("gamma").discover_geometry_product(tmp_path)

    def test_discovers_pairs(self, tmp_path: Path) -> None:
        pair = tmp_path / "pair1"
        pair.mkdir()
        (pair / "phase.unw").write_bytes(b"")
        pairs = discovery.get("gamma").discover_pairs(tmp_path)
        assert len(pairs) == 1


class TestGMTSARDISCOVERER:
    def test_finds_geometry(self, tmp_path: Path) -> None:
        (tmp_path / "inc_geom.grd").write_bytes(b"")
        d = discovery.get("gmtsar")
        result = d.discover_geometry_product(tmp_path)
        assert result == tmp_path

    def test_no_match_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discovery.get("gmtsar").discover_geometry_product(tmp_path)

    def test_discovers_pairs(self, tmp_path: Path) -> None:
        pair = tmp_path / "pair1"
        pair.mkdir()
        (pair / "unwrap.grd").write_bytes(b"")
        pairs = discovery.get("gmtsar").discover_pairs(tmp_path)
        assert len(pairs) == 1


class TestMintPyDiscoverer:
    def test_finds_geometry(self, tmp_path: Path) -> None:
        (tmp_path / "geometryGeo.tif").write_bytes(b"")  # doesn't match pattern
        (tmp_path / "lat.geo.tif").write_bytes(b"")
        d = discovery.get("mintpy")
        result = d.discover_geometry_product(tmp_path)
        assert result == tmp_path

    def test_no_match_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discovery.get("mintpy").discover_geometry_product(tmp_path)

    def test_discovers_pairs(self, tmp_path: Path) -> None:
        (tmp_path / "ifgramStack.h5").write_bytes(b"")
        pairs = discovery.get("mintpy").discover_pairs(tmp_path)
        assert pairs == [tmp_path]

    def test_no_pairs_when_no_h5(self, tmp_path: Path) -> None:
        assert discovery.get("mintpy").discover_pairs(tmp_path) == []


class TestStaMPSDiscoverer:
    def test_finds_geometry(self, tmp_path: Path) -> None:
        (tmp_path / "los.geo.tif").write_bytes(b"")
        d = discovery.get("stamps")
        result = d.discover_geometry_product(tmp_path)
        assert result == tmp_path

    def test_no_match_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discovery.get("stamps").discover_geometry_product(tmp_path)

    def test_discovers_pairs(self, tmp_path: Path) -> None:
        (tmp_path / "PATCH_1").mkdir()
        pairs = discovery.get("stamps").discover_pairs(tmp_path)
        assert pairs == [tmp_path]
