"""Tests for the PROPOSAL-0030 v2 DEM source registry.

Zero-network by contract: import, list, get, catalog, and plan() perform no
socket I/O (pinned by test_dem_catalog_zero_network).
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from faninsar.processing.geometry import dem_sources as ds
from faninsar.processing.geometry.dem_sources import (
    MosaicRecipe,
    dem_catalog,
    get_dem_source,
    list_dem_sources,
    parse_selection,
)
from faninsar.query import BoundingBox

Bounds = BoundingBox | tuple[float, float, float, float]


def _bounds(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> BoundingBox:
    return BoundingBox(min_lon, min_lat, max_lon, max_lat)


# ---------------------------------------------------------------------------
# 1. Shape classes: abstract surface = plan() only, zero fetching
# ---------------------------------------------------------------------------


class TestShapeClasses:
    def test_seven_shape_classes_exist(self) -> None:
        for cls in (
            ds.LatLonGridSource,
            ds.PgcQuadSource,
            ds.TerrainPyramidSource,
            ds.FtpZipSource,
            ds.AuthenticatedGranuleSource,
            ds.RoiClipSource,
            ds.PcStacSource,
        ):
            assert hasattr(cls, "plan")

    def test_dem_source_is_abstract_plan_only(self) -> None:
        with pytest.raises(TypeError):
            ds.DemSource(name="x", description="y")  # type: ignore[abstract]

        # Subclass implementing only plan() is concrete.
        class _Minimal(ds.DemSource):
            def plan(self, bounds):  # noqa: ANN001, ANN202
                raise NotImplementedError

        entry = _Minimal(name="m", description="d", product="glo30", provider="aws")
        assert entry is not None

    def test_planning_does_no_socket_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object, **kwargs: object) -> None:
            raise AssertionError("registry planning must not touch the network")

        monkeypatch.setattr(socket.socket, "connect", boom)
        for name in list_dem_sources():
            source = get_dem_source(name)
            if name == ds.AUTO_SOURCE_NAME:
                continue
            try:
                source.plan(_bounds(10.0, 40.0, 10.5, 40.5))
            except ds.DemSourceUnavailableError:
                continue  # fail-closed coverage/auth gating is fine


# ---------------------------------------------------------------------------
# 2. Selection grammar
# ---------------------------------------------------------------------------


class TestSelectionGrammar:
    def test_plain_product_resolves(self) -> None:
        source = parse_selection("glo30")
        assert source.name == "glo30"

    def test_compound_resolves(self) -> None:
        source = parse_selection("glo30:aws")
        assert source.name == "glo30"

    def test_unknown_product_fails_closed_listing_valid(self) -> None:
        with pytest.raises(ValueError, match="glo30") as excinfo:
            parse_selection("does-not-exist")
        message = str(excinfo.value)
        for name in ("glo90", "nasadem", "alos-dem"):
            assert name in message

    def test_unknown_provider_fails_closed_listing_valid(self) -> None:
        with pytest.raises(ValueError, match="aws") as excinfo:
            parse_selection("glo30:not-a-provider")
        message = str(excinfo.value)
        assert "aws" in message and "pc" in message

    def test_unwired_pair_fails_closed_citing_wired_status(self) -> None:
        with pytest.raises(ValueError, match="[Uu]nwired|not wired") as excinfo:
            parse_selection("glo30:ot")
        assert "ot" in str(excinfo.value)

    @pytest.mark.parametrize(
        "payload",
        [
            "glo30:../..",
            "glo30:",
            ":aws",
            "glο30",  # homoglyph omicron
            " glo30",
            "glo30 ",
            "glo30:aws:extra",
            "GLO30",
        ],
    )
    def test_hostile_payloads_fail_exact_match_closed(self, payload: str) -> None:
        with pytest.raises(ValueError):
            parse_selection(payload)

    def test_raw_dem_source_name_charset_validated(self) -> None:
        with pytest.raises(ValueError, match="charset|name"):
            ds.LatLonGridSource(
                name="bad name with spaces",
                description="d",
                product="glo30",
                provider="aws",
                base_url="https://x.test",
                suffix=".tif",
                min_bytes=1024,
                remote_layout="skadi",
            )


# ---------------------------------------------------------------------------
# 3. Closed valid-pair matrix + catalog
# ---------------------------------------------------------------------------


class TestPairMatrix:
    def test_valid_pairs_resolve_exactly(self) -> None:
        expected = {
            "glo30": ("glo30", "aws", "none", True, "egm2008", 1 / 3600),
            "glo90": ("glo90", "aws", "none", True, "egm2008", 1 / 1200),
            "auto": ("glo30", "aws", "none", True, "egm2008", 1 / 3600),
            "nasadem": ("nasadem", "pc", "none", True, "egm96", 1 / 3600),
            "alos-dem": ("alos-dem", "pc", "none", True, "egm96", 1 / 3600),
            "srtm-skadi": ("srtm-skadi", "aws", "none", True, "egm96", 1 / 3600),
            "terrain-tiles": (
                "terrain-tiles",
                "aws",
                "none",
                True,
                "mixed-derived",
                360.0 / (256 * 2**12),
            ),
            "arcticdem-32": ("arcticdem-32", "aws", "none", True, "ellipsoidal", 32.0),
            "rema-32": ("rema-32", "aws", "none", True, "ellipsoidal", 32.0),
            "nisar-glo30": (
                "nisar-glo30",
                "earthdata",
                "token",
                True,
                "ellipsoidal",
                1 / 3600,
            ),
        }
        for name, (product, provider, auth, wired, datum, res) in expected.items():
            source = get_dem_source(name)
            assert source.product == product, name
            assert source.provider == provider, name
            assert source.auth == auth, name
            assert source.wired is wired, name
            assert source.vertical_datum == datum, name
            assert source.resolution_m == pytest.approx(res), name

    def test_list_dem_sources_returns_fourteen_names(self) -> None:
        names = set(list_dem_sources())
        for name in (
            "glo30",
            "glo90",
            "auto",
            "nasadem",
            "alos-dem",
            "srtm-skadi",
            "terrain-tiles",
            "arcticdem-10",
            "arcticdem-32",
            "arcticdem-2",
            "rema-10",
            "rema-32",
            "rema-2",
            "nisar-glo30",
        ):
            assert name in names
        assert len(names) == 14

    def test_dem_catalog_structure(self) -> None:
        catalog = dem_catalog()
        assert "glo30" in catalog
        entry = catalog["glo30"]
        assert entry["default"] == "aws"
        assert "aws" in entry["providers"]
        assert "pc" in entry["providers"]
        assert entry["providers"]["ot"]["wired"] is False
        assert entry["vertical_datum"] == "egm2008"

    def test_dem_catalog_zero_network(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object, **kwargs: object) -> None:
            raise AssertionError("catalog must not touch the network")

        monkeypatch.setattr(socket.socket, "connect", boom)
        monkeypatch.setattr(socket, "create_connection", boom)
        monkeypatch.setattr(socket.socket, "connect_ex", boom)
        list_dem_sources()
        dem_catalog()
        get_dem_source("glo30")


# ---------------------------------------------------------------------------
# 4. LatLonGridSource layouts
# ---------------------------------------------------------------------------


class TestLatLonGridSource:
    def test_glo30_stem_layout(self) -> None:
        plan = get_dem_source("glo30").plan(_bounds(100.2, 38.2, 100.4, 38.4))
        assert isinstance(plan, ds.TileSet)
        assert plan.allowed_hosts == ("copernicus-dem-30m.s3.amazonaws.com",)
        assert len(plan.tiles) == 1
        tile = plan.tiles[0]
        assert tile.url == (
            "https://copernicus-dem-30m.s3.amazonaws.com/"
            "Copernicus_DSM_COG_10_N38_00_E100_00_DEM/"
            "Copernicus_DSM_COG_10_N38_00_E100_00_DEM.tif"
        )
        assert tile.min_bytes >= 1 << 20

    def test_glo90_stem_layout_no_directory_prefix(self) -> None:
        plan = get_dem_source("glo90").plan(_bounds(45.2, 38.2, 45.4, 38.4))
        tile = plan.tiles[0]
        assert tile.url == (
            "https://copernicus-dem-90m.s3.amazonaws.com/"
            "Copernicus_DSM_COG_30_N38_00_E045_00_DEM/"
            "Copernicus_DSM_COG_30_N38_00_E045_00_DEM.tif"
        )
        assert "COG_30/" not in tile.url

    def test_skadi_layout_small_floor_ocean_skip(self) -> None:
        source = get_dem_source("srtm-skadi")
        plan = source.plan(_bounds(94.2, 34.2, 94.4, 34.4))
        tile = plan.tiles[0]
        assert tile.url.endswith("/skadi/N34/N34E094.hgt.gz")
        assert tile.min_bytes < 1 << 20
        assert tile.expected_decompressed_bytes == 3601 * 3601 * 2
        # ocean-404 skip flag scoped to skadi only
        assert tile.ocean_404_skip is True
        glo30 = get_dem_source("glo30").plan(_bounds(10.2, 40.2, 10.4, 40.4))
        assert all(t.ocean_404_skip is False for t in glo30.tiles)


# ---------------------------------------------------------------------------
# 5. PgcQuadSource
# ---------------------------------------------------------------------------


class _ScriptedEnumerator:
    """QuadEnumerator double: returns scripted quad ids for a bounds."""

    def __init__(self, quads: list[str], pages: int = 1) -> None:
        self.quads = quads
        self.pages = pages

    def quads_for_bounds(self, bounds: Bounds) -> list[str]:
        return list(self.quads)

    def list_quads(self, prefix: str) -> tuple[list[str], bool]:
        return list(self.quads), False


class _ListingEnumerator:
    """QuadEnumerator double exposing only the raw paged-listing API."""

    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    def list_quads(self, prefix: str) -> tuple[list[str], bool]:
        hits = [k for k in self.keys if k.startswith(prefix)]
        return hits, False


class TestPgcQuadSource:
    def test_quad_grid_mapping_pinned(self) -> None:
        source = get_dem_source("arcticdem-32")
        assert isinstance(source, ds.PgcQuadSource)
        quads = source.bounds_to_quads(_bounds(-70.0, 75.0, -60.0, 80.0))
        assert quads, "expected quads inside the arctic band"
        for quad in quads:
            col, row = quad.split("_")
            assert len(col) == 2 and len(row) == 2
            assert col.isdigit() and row.isdigit()

    def test_coverage_fail_closed_outside_polar_bands(self) -> None:
        message = get_dem_source("arcticdem-32").coverage(
            _bounds(10.0, 40.0, 11.0, 41.0)
        )
        assert message is not None
        assert "polar" in message.lower() or "latitude" in message.lower()
        assert (
            get_dem_source("rema-32").coverage(_bounds(0.0, -80.0, 1.0, -79.0)) is None
        )

    def test_plan_uses_enumerator_and_reconciles_gaps(self) -> None:
        source = get_dem_source("arcticdem-32")
        assert isinstance(source, ds.PgcQuadSource)
        source.quad_enumerator = _ScriptedEnumerator(["07_40", "07_41"])
        plan = source.plan(_bounds(-70.0, 75.0, -60.0, 80.0))
        urls = [tile.url for tile in plan.tiles]
        assert any(
            "arcticdem/mosaics/v4.1/32m/07_40/07_40_32m_v4.1_dem.tif" in u for u in urls
        )

    def test_listing_gap_is_hard_failure(self) -> None:
        source = get_dem_source("arcticdem-32")
        assert isinstance(source, ds.PgcQuadSource)
        # Enumerator returns a partial page set for a multi-quad ROI: the
        # completeness reconciliation must fail loudly on the missing quad.
        source.quad_enumerator = _ListingEnumerator(
            [
                "arcticdem/mosaics/v4.1/32m/07_40/07_40_32m_v4.1_dem.tif",
                # 07_41 deliberately missing -> gap
                "arcticdem/mosaics/v4.1/32m/07_42/07_42_32m_v4.1_dem.tif",
            ]
        )
        with pytest.raises(ds.DemSourceUnavailableError, match="gap|missing|quad"):
            source._expected_quads(_bounds(-70.0, 75.0, -60.0, 80.0))

    def test_2m_tier_uses_row_col_subtiles(self) -> None:
        source = get_dem_source("arcticdem-2")
        assert isinstance(source, ds.PgcQuadSource)
        source.quad_enumerator = _ScriptedEnumerator(["07_40"])
        plan = source.plan(_bounds(-70.0, 75.0, -60.0, 80.0))
        assert any("_2m_v4.1_dem.tif" in tile.url for tile in plan.tiles)

    def test_2m_tier_plan_covers_all_four_subtiles(self) -> None:
        """BLOCKER-0030-B2: a 2m-tier plan entry fan-outs into 4 sub-tiles.

        The registry's ``_MultiTileTile`` must carry every
        ``{quad}_{r}_{c}_2m_v4.1_dem.tif`` part (r/c in 1..2), so planning
        alone never silently drops three quarters of a quad.
        """
        source = get_dem_source("arcticdem-2")
        assert isinstance(source, ds.PgcQuadSource)
        source.quad_enumerator = _ScriptedEnumerator(["07_40"])
        plan = source.plan(_bounds(-70.0, 75.0, -60.0, 80.0))
        assert len(plan.tiles) == 1
        entry = plan.tiles[0]
        assert isinstance(entry, ds._MultiTileTile)
        parts = entry.parts
        assert len(parts) == 4
        expected = {
            f"07_40_{row}_{col}_2m_v4.1_dem.tif"
            for row in (1, 2)
            for col in (1, 2)
        }
        assert {part.cache_path.name for part in parts} == expected
        # The fan-out entry mirrors the first sub-tile's URL but every part
        # URL is distinct and planned.
        assert len({part.url for part in parts}) == 4
        assert entry.url == parts[0].url


# ---------------------------------------------------------------------------
# 6. TerrainPyramidSource
# ---------------------------------------------------------------------------


class TestTerrainPyramid:
    def test_z12_xyz_orientation_pin(self) -> None:
        plan = get_dem_source("terrain-tiles").plan(_bounds(-3.51, 40.49, -3.50, 40.50))
        tile = plan.tiles[0]
        assert tile.url.endswith("/geotiff/12/2008/1543.tif")

    def test_derived_warning_registry_wide(self) -> None:
        assert get_dem_source("terrain-tiles").derived is True
        assert get_dem_source("nisar-glo30").derived is True
        assert get_dem_source("glo30").derived is False


# ---------------------------------------------------------------------------
# 7. FtpZipSource
# ---------------------------------------------------------------------------


class TestFtpZipSource:
    def test_alos_jaxa_ftp_artifact(self) -> None:
        source = parse_selection("alos-dem:jaxa-ftp")
        assert isinstance(source, ds.FtpZipSource)
        plan = source.plan(_bounds(6.2, 0.2, 6.4, 0.4))
        assert isinstance(plan, ds.Artifact)
        assert plan.scheme == "ftp"
        assert plan.expand == "zip"
        assert plan.url == (
            "ftp://ftp.eorc.jaxa.jp/pub/ALOS/ext1/AW3D30/release_v2303/"
            "N000E005_N005E010.zip"
        )
        assert plan.member_pattern == "ALPSMLC30_*_DSM.tif"

    def test_multi_block_bounds_plan_without_type_error(self) -> None:
        """BLOCKER-0030-B1: multi-block plans construct without TypeError.

        Bounds spanning more than one 5-degree JAXA block must produce a
        ``_MultiArtifactPlan`` whose artifacts list carries every block zip
        (the missing ``@dataclass`` used to crash with ``FetchPlan.__init__
        got an unexpected keyword argument 'artifacts'``).
        """
        source = parse_selection("alos-dem:jaxa-ftp")
        assert isinstance(source, ds.FtpZipSource)
        plan = source.plan(_bounds(2.0, -2.0, 12.0, 5.0))
        assert isinstance(plan, ds._MultiArtifactPlan)
        assert isinstance(plan.artifacts, tuple)
        assert len(plan.artifacts) >= 2
        assert all(isinstance(artifact, ds.Artifact) for artifact in plan.artifacts)
        assert plan.allowed_hosts == ("ftp.eorc.jaxa.jp",)
        for artifact in plan.artifacts:
            assert artifact.scheme == "ftp"
            assert artifact.expand == "zip"
            assert artifact.url.startswith("ftp://ftp.eorc.jaxa.jp/")
            assert artifact.url.endswith(".zip")


# ---------------------------------------------------------------------------
# 8. AuthenticatedGranuleSource
# ---------------------------------------------------------------------------


class TestAuthenticatedGranule:
    def test_nasadem_earthdata_entry(self) -> None:
        source = parse_selection("nasadem:earthdata")
        assert isinstance(source, ds.AuthenticatedGranuleSource)
        assert source.cmr_collection == "C2763264762-LPCLOUD"
        assert source.vertical_datum == "egm96"
        message = source.coverage(_bounds(10.0, 61.0, 11.0, 62.0))
        assert message is not None
        assert source.coverage(_bounds(10.0, 40.0, 11.0, 41.0)) is None

    def test_nisar_glo30_earthdata_entry(self) -> None:
        source = get_dem_source("nisar-glo30")
        assert isinstance(source, ds.AuthenticatedGranuleSource)
        assert source.cmr_collection == "C3803703055-ASF"
        assert source.vertical_datum == "ellipsoidal"
        assert source.derived is True
        assert source.coverage(_bounds(80.0, -70.0, 81.0, -69.0)) is None

    def test_urs_302_to_200_html_is_hard_error(self, tmp_path: Path) -> None:
        from faninsar.processing.geometry.dem_transport import (
            CredentialProvider,
            Tile,
            TileSet,
            fetch_plan,
        )

        class _FakeResponse:
            def __init__(self, *, status: int, headers: dict, body: bytes) -> None:
                self.status_code = status
                self.headers = headers
                self._body = body

            def iter_content(self, _size: int):
                yield self._body

            def close(self) -> None:
                pass

        class _Session:
            def request(self, method: str, url: str, **kw: object):
                assert method == "GET"
                return _FakeResponse(
                    status=200,
                    headers={"Content-Type": "text/html"},
                    body=b"<html><body>Earthdata Login</body></html>",
                )

        monkey_session = _Session()
        monkeypatch_obj = pytest.MonkeyPatch()
        monkeypatch_obj.setattr(
            transport_module(), "thread_local_session", lambda: monkey_session
        )
        try:
            plan = TileSet(
                allowed_hosts=("data.lpdaac.earthdatacloud.nasa.gov",),
                tiles=(
                    Tile(
                        url="https://data.lpdaac.earthdatacloud.nasa.gov/g.zip",
                        cache_path=Path("g.zip"),
                        min_bytes=16,
                        ranged=False,
                    ),
                ),
            )
            with pytest.raises(Exception, match="HTML|login"):
                fetch_plan(plan, tmp_path)
        finally:
            monkeypatch_obj.undo()
        del CredentialProvider


def transport_module():
    from faninsar.processing.geometry import dem_transport as mod

    return mod


# ---------------------------------------------------------------------------
# 9. PcStacSource
# ---------------------------------------------------------------------------


class TestPcStacSource:
    def test_pc_extra_absent_fails_closed_with_guidance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ds, "_import_pc_stack", lambda: None)
        source = get_dem_source("nasadem")
        assert isinstance(source, ds.PcStacSource)
        plan = source.plan(_bounds(10.0, 40.0, 11.0, 41.0))
        with pytest.raises(ds.DemSourceUnavailableError, match="pip install|\\[pc\\]"):
            source.discover(plan)

    def test_pc_sign_inplace_modifier_flow(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, object] = {}

        class _FakeAsset:
            href = "https://pc.blob.core.windows.net/dem/tile.tif?sig=abc"

        class _FakeItem:
            id = "nasadem-item-0"
            assets = {"elevation": _FakeAsset()}

        class _FakeSearch:
            def __init__(self, **kwargs: object) -> None:
                calls["search_kwargs"] = kwargs

            def items(self):
                return [_FakeItem()]

        class _FakeCollection:
            id = "cop-dem-glo-30"

        class _FakeClient:
            def get_collection(self, collection_id: str):
                calls["collection"] = collection_id
                return _FakeCollection()

            def search(self, **kwargs: object):
                return _FakeSearch(**kwargs)

        def fake_import():
            class _PC:
                @staticmethod
                def sign_inplace(item):
                    calls["signed"] = True
                    return item

            class _Client:
                def get_collection(self, collection_id):
                    calls["collection"] = collection_id

                def search(self, **kwargs):
                    calls["search_kwargs"] = kwargs
                    return self

                def items(self):
                    return [_FakeItem()]

            class _Stac:
                Client = None

            stac_client = type(
                "_StacClientModule",
                (),
                {
                    "Client": type(
                        "Client",
                        (),
                        {"open": staticmethod(lambda url, modifier=None: _Client())},
                    )
                },
            )
            return _PC(), stac_client

        monkeypatch.setattr(ds, "_import_pc_stack", fake_import)
        source = get_dem_source("nasadem")
        assert isinstance(source, ds.PcStacSource)
        plan = source.plan(_bounds(10.0, 40.0, 11.0, 41.0))
        assert isinstance(plan, ds.DeferredStacPlan)
        resolved = source.discover(plan)
        assert resolved.tiles
        assert calls.get("signed") is True
        assert isinstance(resolved, ds.TileSet)
        assert resolved.tiles
        # SAS signature must never leak into the plan's cache identity
        assert all("sig=" not in str(tile.cache_path) for tile in resolved.tiles)


# ---------------------------------------------------------------------------
# 10. MosaicRecipe drives mosaic behavior
# ---------------------------------------------------------------------------


class TestMosaicRecipe:
    def test_recipe_is_frozen_dataclass(self) -> None:
        recipe = MosaicRecipe(
            gdal_open="{path}",
            source_crs=None,
            warp_target="none",
            resampling="bilinear",
            nodata=None,
            mask_to_nan=True,
        )
        with pytest.raises(Exception):
            recipe.nodata = 1.0  # type: ignore[misc]

    def test_per_entry_recipes(self) -> None:
        expected = {
            "glo30": ("{path}", "none", "bilinear", None),
            "glo90": ("{path}", "none", "bilinear", None),
            "srtm-skadi": ("/vsigzip/{path}", "none", "bilinear", None),
            "terrain-tiles": ("/vsicurl/{path}", "epsg4326", "bilinear", -32768.0),
            "arcticdem-32": ("/vsicurl/{path}", "epsg4326", "bilinear", -9999.0),
            "rema-32": ("/vsicurl/{path}", "epsg4326", "bilinear", -9999.0),
            "alos-dem": ("{path}", "none", "bilinear", None),
            "nasadem": ("{path}", "none", "bilinear", None),
            "nasadem:earthdata": ("/vsizip/{path}/{member}", "none", "bilinear", None),
            "nisar-glo30": ("/vsizip/{path}/{member}", "none", "bilinear", None),
        }
        for name, (gdal_open, warp, resampling, nodata) in expected.items():
            recipe = get_dem_source(name).mosaic_recipe()
            assert recipe.gdal_open == gdal_open, name
            assert recipe.warp_target == warp, name
            assert recipe.resampling == resampling, name
            assert recipe.nodata == nodata, name


class TestMatrixRegression:
    """Regression pins for defects found in the 2026-08-23 live matrix run.

    Evidence: /Volumes/DATA2/TEST_sentinel-1/proposal-0030-impl-v2/MATRIX.md
    (F1: PC asset keys; F2: PGC meter-to-degree resolution conversion).
    """

    def test_pc_entries_use_live_asset_keys(self) -> None:
        """F1: nasadem/alos-dem PC assets match the live collections."""
        assert get_dem_source("nasadem").asset_key == "elevation"
        assert get_dem_source("alos-dem").asset_key == "data"

    def test_pgc_resolution_m_stays_meters(self) -> None:
        """F2 part 1: registry keeps resolution_m in meters (pair-matrix pin).

        The conversion to degrees happens at the mosaic call site.
        """
        import math

        src = get_dem_source("arcticdem-32")
        assert src.resolution_m == 32.0

    def test_pgc_mosaic_grid_conversion_is_deterministic(self) -> None:
        """F2 part 2: the meter-to-degree conversion is deterministic.

        32 m at the documented cos(75 deg) mid-band factor must land in the
        GLO-30-class range on the EPSG:4326 grid (not 32 degrees).
        """
        import math

        from faninsar.processing.geometry.dem_manager import (
            resolution_m_to_degrees,
        )

        deg = resolution_m_to_degrees(32.0)
        expected = 32.0 / (111_320.0 * math.cos(math.radians(75.0)))
        assert math.isclose(deg, expected, rel_tol=1e-9)
        assert 1 / 1200 / 2 < deg < 1 / 900, deg

    def test_pgc_default_entry_grid_not_degenerate(self) -> None:
        """F2 end-to-end: a 1-degree PGC mosaic plan sizes a real grid."""
        import numpy as np

        from faninsar.processing.geometry.dem_manager import _mosaic_arrays
        from faninsar.processing.geometry.dem_sources import get_dem_source

        source = get_dem_source("arcticdem-32")
        recipe = source.mosaic_recipe()
        # Synthetic polar-stereo tile covering ~1 degree square near 75N.
        import io

        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_origin

        profile = {
            "driver": "GTiff",
            "width": 100,
            "height": 100,
            "count": 1,
            "dtype": "float32",
            "crs": CRS.from_epsg(3413),
            "transform": from_origin(-500000, -500000, 3200, 3200),
        }
        buf = io.BytesIO()
        with rasterio.open(buf, "w", **profile) as dst:
            dst.write(np.full((100, 100), 500.0, dtype=np.float32), 1)
        buf.seek(0)
        tmp = Path("/tmp/p0030_regression_tile.tif")
        tmp.write_bytes(buf.read())

        try:
            # The manager converts meters→degrees before calling the mosaic
            # (polar-stereo sources reproject to EPSG:4326); replicate it.
            from faninsar.processing.geometry.dem_manager import (
                resolution_m_to_degrees,
            )

            deg = resolution_m_to_degrees(source.resolution_m)
            mosaic, transform = _mosaic_arrays([tmp], recipe, deg)
            assert mosaic.shape[0] > 50 and mosaic.shape[1] > 50
            finite = np.isfinite(mosaic)
            assert finite.any()
        finally:
            tmp.unlink(missing_ok=True)
