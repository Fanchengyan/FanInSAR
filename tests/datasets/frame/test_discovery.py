"""Tests for the discovery registry and built-in processor discoverers (M4).

Patterns match the canonical MintPy directory conventions documented at
https://mintpy.readthedocs.io/en/latest/dir_structure/. Each test creates
a minimal synthetic directory layout that mirrors the real product structure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from faninsar.datasets.frame import discovery

# ---------------------------------------------------------------------------
# Registry contract tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_builtins_registered(self) -> None:
        names = discovery.available()
        for expected in ("hyp3", "isce", "gamma", "gmtsar", "mintpy", "stamps", "aria"):
            assert expected in names, f"{expected!r} not in {names}"

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


# ---------------------------------------------------------------------------
# HyP3
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ISCE2 (topsStack / stripmapStack / alosStack)
# ---------------------------------------------------------------------------


class TestISCEDiscoverer:
    """ISCE2 discovery matches MintPy topsStack/stripmapStack/alosStack layouts."""

    # -- topsStack (canonical Sentinel-1 TOPS) --

    def test_topsstack_geometry(self, tmp_path: Path) -> None:
        geom = tmp_path / "merged" / "geom_reference"
        geom.mkdir(parents=True)
        (geom / "lat.rdr").write_bytes(b"")
        d = discovery.get("isce")
        assert d.discover_geometry_product(tmp_path) == geom

    def test_topsstack_pairs(self, tmp_path: Path) -> None:
        pair = tmp_path / "merged" / "interferograms" / "20191115_20200314"
        pair.mkdir(parents=True)
        (pair / "filt_topophase.unw").write_bytes(b"")
        pairs = discovery.get("isce").discover_pairs(tmp_path)
        assert len(pairs) == 1
        assert pairs[0] == pair

    # -- stripmapStack --

    def test_stripmapstack_geometry(self, tmp_path: Path) -> None:
        geom = tmp_path / "geom_reference"
        geom.mkdir()
        (geom / "lat.rdr").write_bytes(b"")
        d = discovery.get("isce")
        assert d.discover_geometry_product(tmp_path) == geom

    def test_stripmapstack_pairs(self, tmp_path: Path) -> None:
        pair = tmp_path / "Igrams" / "20191115_20200314"
        pair.mkdir(parents=True)
        (pair / "filt_topophase_snaphu.unw").write_bytes(b"")
        pairs = discovery.get("isce").discover_pairs(tmp_path)
        assert len(pairs) == 1
        assert pairs[0] == pair

    # -- alosStack --

    def test_alosstack_geometry(self, tmp_path: Path) -> None:
        insar = tmp_path / "dates_res1" / "20150101" / "insar"
        insar.mkdir(parents=True)
        (insar / "20150101.los").write_bytes(b"")
        d = discovery.get("isce")
        assert d.discover_geometry_product(tmp_path) == insar

    def test_alosstack_pairs(self, tmp_path: Path) -> None:
        insar = tmp_path / "pairs" / "20150101-20150801" / "insar"
        insar.mkdir(parents=True)
        (insar / "filt_20150101_20150801.unw").write_bytes(b"")
        pairs = discovery.get("isce").discover_pairs(tmp_path)
        assert len(pairs) == 1

    # -- error cases --

    def test_no_geometry_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discovery.get("isce").discover_geometry_product(tmp_path)

    def test_no_pairs_returns_empty(self, tmp_path: Path) -> None:
        assert discovery.get("isce").discover_pairs(tmp_path) == []


# ---------------------------------------------------------------------------
# GAMMA
# ---------------------------------------------------------------------------


class TestGAMMADiscoverer:
    """GAMMA discovery matches MintPy GAMMA layout (geometry/ + interferograms/)."""

    def test_geometry_dir(self, tmp_path: Path) -> None:
        geom = tmp_path / "geometry"
        geom.mkdir()
        (geom / "20050619_20070809.sim_01.rdc.dem").write_bytes(b"")
        d = discovery.get("gamma")
        assert d.discover_geometry_product(tmp_path) == geom

    def test_geometry_geo_lat(self, tmp_path: Path) -> None:
        geom = tmp_path / "geometry"
        geom.mkdir()
        (geom / "20050619.geo.lat").write_bytes(b"")
        d = discovery.get("gamma")
        assert d.discover_geometry_product(tmp_path) == geom

    def test_fallback_flat_dem_seg(self, tmp_path: Path) -> None:
        """Fallback: flat layout with dem_seg* in a subdirectory."""
        prod = tmp_path / "gamma_prod"
        prod.mkdir()
        (prod / "dem_seg").write_bytes(b"")
        d = discovery.get("gamma")
        assert d.discover_geometry_product(tmp_path) == prod

    def test_no_match_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discovery.get("gamma").discover_geometry_product(tmp_path)

    def test_discovers_pairs(self, tmp_path: Path) -> None:
        pair = tmp_path / "interferograms" / "20050619_20070809"
        pair.mkdir(parents=True)
        (pair / "diff_20050619_20070809.lat_20050619_20070809.unw").write_bytes(b"")
        pairs = discovery.get("gamma").discover_pairs(tmp_path)
        assert len(pairs) == 1
        assert pairs[0] == pair

    def test_fallback_pairs_unw(self, tmp_path: Path) -> None:
        """Fallback: any subdir with *.unw."""
        pair = tmp_path / "pair1"
        pair.mkdir()
        (pair / "phase.unw").write_bytes(b"")
        pairs = discovery.get("gamma").discover_pairs(tmp_path)
        assert len(pairs) == 1


# ---------------------------------------------------------------------------
# GMTSAR
# ---------------------------------------------------------------------------


class TestGMTSARDiscoverer:
    """GMTSAR discovery matches MintPy GMTSAR layout (merged/)."""

    def test_merged_geom_reference(self, tmp_path: Path) -> None:
        geom = tmp_path / "merged" / "geom_reference"
        geom.mkdir(parents=True)
        (geom / "lat.rdr").write_bytes(b"")
        d = discovery.get("gmtsar")
        assert d.discover_geometry_product(tmp_path) == geom

    def test_merged_dem_grd(self, tmp_path: Path) -> None:
        merged = tmp_path / "merged"
        merged.mkdir()
        (merged / "dem.grd").write_bytes(b"")
        d = discovery.get("gmtsar")
        assert d.discover_geometry_product(tmp_path) == merged

    def test_toplevel_dem_grd(self, tmp_path: Path) -> None:
        (tmp_path / "dem.grd").write_bytes(b"")
        d = discovery.get("gmtsar")
        assert d.discover_geometry_product(tmp_path) == tmp_path

    def test_no_match_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discovery.get("gmtsar").discover_geometry_product(tmp_path)

    def test_discovers_pairs(self, tmp_path: Path) -> None:
        pair = tmp_path / "merged" / "interferograms" / "20191115_20200314"
        pair.mkdir(parents=True)
        (pair / "unwrap_ll.grd").write_bytes(b"")
        pairs = discovery.get("gmtsar").discover_pairs(tmp_path)
        assert len(pairs) == 1
        assert pairs[0] == pair

    def test_fallback_pairs_unwrap_grd(self, tmp_path: Path) -> None:
        """Fallback: any subdir with unwrap.grd / unwrap_ll.grd."""
        pair = tmp_path / "pair1"
        pair.mkdir()
        (pair / "unwrap.grd").write_bytes(b"")
        pairs = discovery.get("gmtsar").discover_pairs(tmp_path)
        assert len(pairs) == 1


# ---------------------------------------------------------------------------
# MintPy
# ---------------------------------------------------------------------------


class TestMintPyDiscoverer:
    """MintPy discovery matches canonical inputs/ layout."""

    def test_geometry_h5(self, tmp_path: Path) -> None:
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        (inputs / "geometryGeo.h5").write_bytes(b"")
        d = discovery.get("mintpy")
        assert d.discover_geometry_product(tmp_path) == inputs

    def test_geometry_radar_h5(self, tmp_path: Path) -> None:
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        (inputs / "geometryRadar.h5").write_bytes(b"")
        d = discovery.get("mintpy")
        assert d.discover_geometry_product(tmp_path) == inputs

    def test_fallback_geo_tif(self, tmp_path: Path) -> None:
        """Fallback: top-level geometryGeo.tif."""
        (tmp_path / "geometryGeo.tif").write_bytes(b"")
        d = discovery.get("mintpy")
        assert d.discover_geometry_product(tmp_path) == tmp_path

    def test_no_match_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discovery.get("mintpy").discover_geometry_product(tmp_path)

    def test_discovers_pairs_ifgram_stack(self, tmp_path: Path) -> None:
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        (inputs / "ifgramStack.h5").write_bytes(b"")
        pairs = discovery.get("mintpy").discover_pairs(tmp_path)
        assert pairs == [inputs]

    def test_fallback_toplevel_ifgram_stack(self, tmp_path: Path) -> None:
        """Fallback: top-level ifgramStack.h5."""
        (tmp_path / "ifgramStack.h5").write_bytes(b"")
        pairs = discovery.get("mintpy").discover_pairs(tmp_path)
        assert pairs == [tmp_path]

    def test_no_pairs_when_no_h5(self, tmp_path: Path) -> None:
        assert discovery.get("mintpy").discover_pairs(tmp_path) == []


# ---------------------------------------------------------------------------
# StaMPS
# ---------------------------------------------------------------------------


class TestStaMPSDiscoverer:
    """StaMPS discovery matches PATCH_* + ps_plot exports convention."""

    def test_geometry_geo_tif(self, tmp_path: Path) -> None:
        (tmp_path / "los.geo.tif").write_bytes(b"")
        d = discovery.get("stamps")
        assert d.discover_geometry_product(tmp_path) == tmp_path

    def test_geometry_subdir(self, tmp_path: Path) -> None:
        geom = tmp_path / "geometry"
        geom.mkdir()
        (geom / "lat.rdr").write_bytes(b"")
        d = discovery.get("stamps")
        assert d.discover_geometry_product(tmp_path) == geom

    def test_no_match_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discovery.get("stamps").discover_geometry_product(tmp_path)

    def test_discovers_patches(self, tmp_path: Path) -> None:
        (tmp_path / "PATCH_1").mkdir()
        (tmp_path / "PATCH_2").mkdir()
        pairs = discovery.get("stamps").discover_pairs(tmp_path)
        assert pairs == [tmp_path]

    def test_discovers_ps_plot_exports(self, tmp_path: Path) -> None:
        (tmp_path / "ps_plot_v-d.h5").write_bytes(b"")
        pairs = discovery.get("stamps").discover_pairs(tmp_path)
        assert pairs == [tmp_path]

    def test_no_pairs_returns_empty(self, tmp_path: Path) -> None:
        assert discovery.get("stamps").discover_pairs(tmp_path) == []


# ---------------------------------------------------------------------------
# ARIA
# ---------------------------------------------------------------------------


class TestARIADiscoverer:
    """ARIA discovery matches ARIA-tools stack/ + DEM/incidence/azimuth/ layout."""

    def test_geometry_incidence(self, tmp_path: Path) -> None:
        inc = tmp_path / "incidence"
        inc.mkdir()
        (inc / "S1_20191115_20200314_incidence.vrt").write_bytes(b"")
        d = discovery.get("aria")
        assert d.discover_geometry_product(tmp_path) == inc

    def test_geometry_dem(self, tmp_path: Path) -> None:
        dem = tmp_path / "DEM"
        dem.mkdir()
        (dem / "dem.vrt").write_bytes(b"")
        d = discovery.get("aria")
        assert d.discover_geometry_product(tmp_path) == dem

    def test_geometry_gunw_nc(self, tmp_path: Path) -> None:
        (tmp_path / "S1-GUNW-D-R-022-tops-20191115_20200314-s1-012.nc").write_bytes(
            b""
        )
        d = discovery.get("aria")
        assert d.discover_geometry_product(tmp_path) == tmp_path

    def test_no_match_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discovery.get("aria").discover_geometry_product(tmp_path)

    def test_discovers_stack_layers(self, tmp_path: Path) -> None:
        unw = tmp_path / "stack" / "unwrappedPhase"
        unw.mkdir(parents=True)
        (unw / "S1_20191115_20200314_unwrappedPhase.vrt").write_bytes(b"")
        coh = tmp_path / "stack" / "coh"
        coh.mkdir(parents=True)
        (coh / "S1_20191115_20200314_coh.vrt").write_bytes(b"")
        pairs = discovery.get("aria").discover_pairs(tmp_path)
        assert len(pairs) == 2

    def test_fallback_gunw_pairs(self, tmp_path: Path) -> None:
        """Fallback: top-level GUNW .nc products."""
        (tmp_path / "S1-GUNW-A-R-001-tops-20191115_20200314-s1-001.nc").write_bytes(
            b""
        )
        pairs = discovery.get("aria").discover_pairs(tmp_path)
        assert pairs == [tmp_path]

    def test_no_pairs_returns_empty(self, tmp_path: Path) -> None:
        assert discovery.get("aria").discover_pairs(tmp_path) == []
