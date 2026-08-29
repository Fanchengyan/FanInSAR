"""Tests for the PROPOSAL-0039 mask source registry (offline; zero network).

Covers TDD-plan items 9-11 for :mod:`faninsar.processing.masking.mask_sources`:

- Registry completeness and the ``water`` / ``water:<provider>`` selection
  grammar, including fail-closed rejection of invalid selections.
- Live-verified URL-template fixtures for the JRC GSW and ESA WorldCover tile
  layouts; the spike's known-bad patterns are pinned as negatives.
- Zero-network planning (no socket I/O at import or plan time).
- Name and cache-path traversal guards reused from ``dem_sources``.
- The one shared tile-snap function used by both the padded fetch band and
  the antimeridian seam guard (PROPOSAL-0039 round-5 executed semantics).
"""

from __future__ import annotations

import re
import socket
from pathlib import Path

import pytest

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry import dem_sources as dem_ds
from faninsar.processing.geometry.dem_transport import FetchPlan, TileSet
from faninsar.processing.masking import mask_sources as ms
from faninsar.processing.masking.mask_sources import (
    AUTO_SOURCE_NAME,
    DEFAULT_PRODUCT,
    MaskSource,
    MaskSourceUnavailableError,
    OsmOverpassSource,
    get_mask_source,
    list_mask_sources,
    mask_cache_relative_path,
    parse_mask_selection,
    tile_snap,
    validate_mask_source_name,
)
from faninsar.query import BoundingBox

#: Live-verified flat GSW filename shape: ``occurrence_{lon}{E|W}_{lat}{N|S}``
#: with hemisphere letters and no zero padding (e.g. ``occurrence_10W_0N.tif``).
GSW_FLAT_NAME_RE = re.compile(r"occurrence_\d+[EW]_\d+[NS]\.tif$")

#: Live-verified WorldCover filename shape: zero-padded ``{N|S}{lat:02d}``
#: followed by ``{E|W}{lon:03d}`` (e.g. ``..._N00E009_Map.tif``).
WORLDCOVER_NAME_RE = re.compile(
    r"ESA_WorldCover_10m_2021_v200_[NS]\d{2}[EW]\d{3}_Map\.tif$"
)


def _bounds(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> tuple[float, float, float, float]:
    """Build a plain ``(lon_min, lat_min, lon_max, lat_max)`` bounds tuple."""
    return (min_lon, min_lat, max_lon, max_lat)


# ---------------------------------------------------------------------------
# 9. Registry completeness + selection grammar
# ---------------------------------------------------------------------------


class TestMaskSourceRegistry:
    """Registry completeness and the water selection grammar."""

    def test_registry_completeness(self) -> None:
        """Check the registry lists the water product and its two providers."""
        assert AUTO_SOURCE_NAME == "water"
        assert DEFAULT_PRODUCT == "water"
        assert list_mask_sources() == ["gsw", "water", "worldcover"]

    def test_auto_selection_resolves_to_gsw_default(self) -> None:
        """Check the ``water`` auto alias resolves to the default provider."""
        source = get_mask_source(AUTO_SOURCE_NAME)
        assert source is get_mask_source("gsw")
        assert source is parse_mask_selection("water")
        assert source.product == "water"
        assert source.provider == "gsw"

    def test_compound_selection_resolves(self) -> None:
        """Check ``water:<provider>`` selections resolve exactly."""
        assert parse_mask_selection("water:gsw") is get_mask_source("gsw")
        assert parse_mask_selection("water:worldcover") is get_mask_source("worldcover")
        assert get_mask_source("water:worldcover") is get_mask_source("worldcover")

    def test_unknown_product_fails_closed_listing_valid(self) -> None:
        """Check unknown products fail closed while listing valid products."""
        with pytest.raises(ValueError, match="water") as excinfo:
            parse_mask_selection("not-a-product")
        assert "valid products" in str(excinfo.value)

    def test_unknown_provider_fails_closed_listing_valid(self) -> None:
        """Check unknown providers fail closed while listing valid providers."""
        with pytest.raises(ValueError, match="gsw") as excinfo:
            parse_mask_selection("water:not-a-provider")
        message = str(excinfo.value)
        assert "worldcover" in message
        assert "osm-overpass" in message

    def test_unwired_osm_overpass_fails_closed(self) -> None:
        """Check the opt-in OSM provider is registered but not wired in v1."""
        providers = ms.MASK_PRODUCT_DEFAULTS["water"]["providers"]
        assert set(providers) == {"gsw", "worldcover", "osm-overpass"}
        assert providers["osm-overpass"]["wired"] is False
        assert ms.MASK_PRODUCT_DEFAULTS["water"]["default"] == "gsw"
        with pytest.raises(ValueError, match="not wired") as excinfo:
            parse_mask_selection("water:osm-overpass")
        assert "osm-overpass" in str(excinfo.value)

    @pytest.mark.parametrize(
        "payload",
        [
            "water:../..",
            "water:",
            ":gsw",
            "wäter",  # homoglyph a-umlaut
            " water",
            "water ",
            "water:gsw:extra",
            "WATER",
        ],
    )
    def test_hostile_payloads_fail_exact_match_closed(self, payload: str) -> None:
        """Check hostile selection payloads fail closed on exact match."""
        with pytest.raises(ValueError, match=r"mask|wired"):
            parse_mask_selection(payload)

    def test_non_string_selection_raises_type_error(self) -> None:
        """Check non-string selections raise TypeError, not ValueError."""
        with pytest.raises(TypeError, match="string"):
            parse_mask_selection(123)  # type: ignore[arg-type]

    def test_source_metadata_polarity(self) -> None:
        """Check extraction/polarity metadata lives with the source."""
        gsw = get_mask_source("gsw")
        assert gsw.product == "water"
        assert gsw.provider == "gsw"
        assert gsw.tile_size_deg == 10.0
        assert gsw.resolution_m == 30.0
        assert gsw.extraction == "threshold"
        assert gsw.threshold == 50

        worldcover = get_mask_source("worldcover")
        assert worldcover.tile_size_deg == 3.0
        assert worldcover.resolution_m == 10.0
        assert worldcover.extraction == "categorical"
        assert worldcover.excluded_values == (80,)
        assert worldcover.threshold is None

        osm = OsmOverpassSource(name="osm-overpass", description="OSM water")
        assert osm.extraction == "vector"
        assert osm.provider == "osm-overpass"
        assert osm.wired is False

    def test_mask_source_unavailable_error_is_structured(self) -> None:
        """Check the unavailable error stays a structured processing error."""
        assert issubclass(MaskSourceUnavailableError, InvalidProcessingStateError)

    def test_mask_source_is_abstract_plan_only(self) -> None:
        """Check MaskSource is abstract with plan() as the only abstractmethod."""

        class _Minimal(MaskSource):
            """Minimal concrete probe."""

            def plan(self, bounds):  # noqa: ANN001, ANN202
                raise NotImplementedError

        with pytest.raises(TypeError, match="abstract"):
            MaskSource(name="x", description="y")  # type: ignore[abstract]
        entry = _Minimal(name="m", description="d", extraction="vector")
        assert entry.name == "m"

    def test_source_name_charset_validated_at_construction(self) -> None:
        """Check the name charset guard fires at construction (fail closed)."""

        class _Minimal(MaskSource):
            """Minimal concrete probe."""

            def plan(self, bounds):  # noqa: ANN001, ANN202
                raise NotImplementedError

        with pytest.raises(ValueError, match=r"charset|name"):
            _Minimal(
                name="bad name with spaces",
                description="d",
                extraction="vector",
            )


# ---------------------------------------------------------------------------
# 9. Live-verified URL templates (known-bad spike patterns as negatives)
# ---------------------------------------------------------------------------


class TestGswUrlTemplates:
    """Live-verified GSW URL-template fixtures (known-bad as negatives)."""

    def test_flat_url_template_positive(self) -> None:
        """Check a ROI inside one tile plans the exact live-verified URL."""
        plan = get_mask_source("gsw").plan(_bounds(10.0, 0.0, 10.5, 0.5))
        assert isinstance(plan, TileSet)
        assert isinstance(plan, FetchPlan)
        assert len(plan.tiles) == 1
        tile = plan.tiles[0]
        assert tile.url == f"{ms.GSW_BASE_URL}/occurrence_10E_0N.tif"
        assert tile.cache_path == Path("water-gsw/occurrence_10E_0N.tif")
        assert tile.ranged is True
        assert tile.ocean_404_skip is False
        assert plan.allowed_hosts == ("storage.googleapis.com",)

    def test_hemisphere_letter_naming_for_negative_origins(self) -> None:
        """Check negative tile origins use W/S letters, never literal minus."""
        plan = get_mask_source("gsw").plan(_bounds(-11.0, -1.0, -9.0, 1.0))
        names = {tile.url.rsplit("/", 1)[-1] for tile in plan.tiles}
        assert names == {
            "occurrence_20W_10S.tif",
            "occurrence_20W_0N.tif",
            "occurrence_10W_10S.tif",
            "occurrence_10W_0N.tif",
        }

    def test_negative_latitude_uses_s_letter(self) -> None:
        """Check negative latitudes render with the S letter (no minus)."""
        plan = get_mask_source("gsw").plan(_bounds(9.0, -1.0, 11.0, 1.0))
        names = {tile.url.rsplit("/", 1)[-1] for tile in plan.tiles}
        assert names == {
            "occurrence_0E_10S.tif",
            "occurrence_10E_10S.tif",
            "occurrence_0E_0N.tif",
            "occurrence_10E_0N.tif",
        }

    def test_known_bad_patterns_not_produced(self) -> None:
        """Check the spike's 404 patterns are never rendered.

        Known-bad shapes (live-probed 404 on the GSW bucket):

        - the nested guess ``.../occurrence/{lon}E/W_{lat}N/S.tif`` (path
          segments instead of one flat filename);
        - literal-minus naming (``occurrence_-10E_0N.tif``).
        """
        bounds_list = [
            (10.0, 0.0, 10.5, 0.5),
            (-11.0, -1.0, -9.0, 1.0),
            (9.0, -1.0, 11.0, 1.0),
            (170.0, 0.0, 180.0, 5.0),
        ]
        for bounds in bounds_list:
            plan = get_mask_source("gsw").plan(bounds)
            for tile in plan.tiles:
                url = tile.url
                assert GSW_FLAT_NAME_RE.search(url) is not None, url
                assert "-" not in url.rsplit("/", 1)[-1], url
                # nested guess: exactly one flat segment after /occurrence/
                assert url.count("/") == 6, url
                assert "/W_" not in url, url
                assert "/S.tif" not in url, url
                assert url.startswith(f"{ms.GSW_BASE_URL}/"), url

    def test_domain_clamp_no_phantom_seam_tile(self) -> None:
        """Check bounds ending at +180 never plan the nonexistent 180E tile."""
        plan = get_mask_source("gsw").plan(_bounds(170.0, 0.0, 180.0, 5.0))
        names = [tile.url.rsplit("/", 1)[-1] for tile in plan.tiles]
        assert names == ["occurrence_170E_0N.tif"]

    def test_domain_clamp_no_phantom_pole_tile(self) -> None:
        """Check bounds ending at +90 never plan the nonexistent 90N tile."""
        plan = get_mask_source("gsw").plan(_bounds(0.0, 85.0, 5.0, 90.0))
        names = [tile.url.rsplit("/", 1)[-1] for tile in plan.tiles]
        assert names == ["occurrence_0E_80N.tif"]


class TestWorldCoverUrlTemplates:
    """Live-verified WorldCover URL-template fixtures (known-bad negatives)."""

    def test_zero_padded_url_template_positive(self) -> None:
        """Check a ROI inside one tile plans the exact zero-padded URL."""
        plan = get_mask_source("worldcover").plan(_bounds(9.0, 0.0, 9.5, 0.5))
        assert isinstance(plan, TileSet)
        assert len(plan.tiles) == 1
        tile = plan.tiles[0]
        assert tile.url == (
            f"{ms.WORLDCOVER_BASE_URL}/ESA_WorldCover_10m_2021_v200_N00E009_Map.tif"
        )
        assert tile.cache_path == Path(
            "water-worldcover/ESA_WorldCover_10m_2021_v200_N00E009_Map.tif"
        )
        assert tile.ranged is True
        assert plan.allowed_hosts == ("esa-worldcover.s3.amazonaws.com",)

    def test_ocean_tiles_tolerated_absent(self) -> None:
        """Check absent pure-ocean tiles skip with 404 (dem skadi precedent).

        Live bucket listing shows pure-ocean WorldCover tiles are absent;
        GSW ships the complete grid and must not enable the skip flag.
        """
        plan = get_mask_source("worldcover").plan(_bounds(9.0, 0.0, 9.5, 0.5))
        assert all(tile.ocean_404_skip is True for tile in plan.tiles)
        gsw_plan = get_mask_source("gsw").plan(_bounds(9.0, 0.0, 9.5, 0.5))
        assert all(tile.ocean_404_skip is False for tile in gsw_plan.tiles)

    def test_negative_hemisphere_naming(self) -> None:
        """Check S/W letters with zero padding (live-verified HTTP 200)."""
        plan = get_mask_source("worldcover").plan(_bounds(-71.0, -13.0, -70.5, -12.5))
        names = [tile.url.rsplit("/", 1)[-1] for tile in plan.tiles]
        assert names == ["ESA_WorldCover_10m_2021_v200_S15W072_Map.tif"]

    def test_unpadded_bad_pattern_not_produced(self) -> None:
        """Check the unpadded ``N{lat}E{lon}`` spike guess is never rendered."""
        for bounds in (
            (9.0, 0.0, 9.5, 0.5),
            (-77.0, -13.0, -76.5, -12.5),
            (8.0, 0.0, 10.0, 2.0),
        ):
            plan = get_mask_source("worldcover").plan(bounds)
            for tile in plan.tiles:
                url = tile.url
                assert WORLDCOVER_NAME_RE.search(url) is not None, url
                assert "N0E9" not in url, url
                assert "E9_" not in url, url
                assert "_E9_M" not in url, url

    def test_tile_grid_3deg(self) -> None:
        """Check the WorldCover plan enumerates 3-degree tile origins."""
        plan = get_mask_source("worldcover").plan(_bounds(8.0, 0.0, 10.0, 2.0))
        names = {tile.url.rsplit("/", 1)[-1] for tile in plan.tiles}
        assert names == {
            "ESA_WorldCover_10m_2021_v200_N00E006_Map.tif",
            "ESA_WorldCover_10m_2021_v200_N00E009_Map.tif",
        }

    def test_domain_clamp_no_phantom_seam_tile(self) -> None:
        """Check bounds ending at +180 never plan a phantom 180-origin tile."""
        plan = get_mask_source("worldcover").plan(_bounds(175.0, 0.0, 180.0, 5.0))
        names = {tile.url.rsplit("/", 1)[-1] for tile in plan.tiles}
        assert names == {
            "ESA_WorldCover_10m_2021_v200_N00E174_Map.tif",
            "ESA_WorldCover_10m_2021_v200_N00E177_Map.tif",
            "ESA_WorldCover_10m_2021_v200_N03E174_Map.tif",
            "ESA_WorldCover_10m_2021_v200_N03E177_Map.tif",
        }


class TestOsmOverpassSource:
    """The opt-in OSM provider stays unwired until the response cap binds."""

    def test_plan_fails_closed_until_cap_binds(self) -> None:
        """Check plan() fails closed with the pending-cap reason."""
        source = OsmOverpassSource(name="osm-overpass", description="OSM water")
        with pytest.raises(MaskSourceUnavailableError, match="response-size cap"):
            source.plan(_bounds(0.0, 0.0, 0.5, 0.5))

    def test_endpoint_is_https_enforced(self) -> None:
        """Check plain-http endpoints are rejected at construction."""
        with pytest.raises(ValueError, match="https"):
            OsmOverpassSource(
                name="osm-overpass",
                description="OSM water",
                endpoint="http://overpass-api.de/api/interpreter",
            )

    def test_default_endpoint_is_https(self) -> None:
        """Check the default Overpass endpoint is https."""
        source = OsmOverpassSource(name="osm-overpass", description="OSM water")
        assert source.endpoint.startswith("https://")


# ---------------------------------------------------------------------------
# 9. Zero-network invariant
# ---------------------------------------------------------------------------


class TestZeroNetwork:
    """Zero network at plan time (import-time purity is the module contract)."""

    def test_planning_does_no_socket_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Check plan() performs no socket I/O for every registered source."""
        message = "mask registry planning must not touch the network"

        def boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError(message)

        monkeypatch.setattr(socket.socket, "connect", boom)
        monkeypatch.setattr(socket, "create_connection", boom)
        for name in list_mask_sources():
            source = get_mask_source(name)
            source.plan(_bounds(10.0, 0.0, 11.0, 1.0))
            source.plan(_bounds(-11.0, -1.0, -9.0, 1.0))

    def test_plan_is_deterministic_pure_function(self) -> None:
        """Check identical bounds produce identical plans (pure function)."""
        source = get_mask_source("gsw")
        first = source.plan(_bounds(9.0, -1.0, 11.0, 1.0))
        second = source.plan(_bounds(9.0, -1.0, 11.0, 1.0))
        assert first == second

    def test_plan_accepts_bounding_box(self) -> None:
        """Check BoundingBox and tuple bounds plan identically."""
        source = get_mask_source("gsw")
        from_box = source.plan(BoundingBox(9.0, -1.0, 11.0, 1.0))
        from_tuple = source.plan((9.0, -1.0, 11.0, 1.0))
        assert from_box == from_tuple


# ---------------------------------------------------------------------------
# 10. Name / cache-path guards reused from dem_sources
# ---------------------------------------------------------------------------


class TestSourceNameAndCacheGuards:
    """Name and cache-path traversal guards reused from ``dem_sources``."""

    @pytest.mark.parametrize(
        "name", ["gsw", "water", "worldcover", "osm-overpass", "a1"]
    )
    def test_valid_names_pass(self, name: str) -> None:
        """Check registry-charset names pass the mask name guard."""
        validate_mask_source_name(name)

    @pytest.mark.parametrize(
        "name",
        ["bad name", "", "GSW", "../gsw", "gsw:aws", "wàter", "-gsw", "gsw-"],
    )
    def test_invalid_names_rejected(self, name: str) -> None:
        """Check names outside the dem guard charset fail closed."""
        with pytest.raises(ValueError, match=r"name|charset"):
            validate_mask_source_name(name)

    def test_name_guard_matches_dem_guard(self) -> None:
        """Check behavioral delegation: identical accept/reject as the DEM guard."""
        for name in ("gsw", "water", "Bad Name", "", "../gsw", "gsw-"):
            try:
                dem_ds._validate_source_name(name)
                dem_ok = True
            except ValueError:
                dem_ok = False
            if dem_ok:
                validate_mask_source_name(name)
            else:
                with pytest.raises(ValueError, match=r"name|charset"):
                    validate_mask_source_name(name)

    def test_name_validation_delegates_to_dem_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Check the wrapper routes through the dem_sources guard at call time."""
        seen: list[str] = []
        message = "spy"

        def spy(name: str) -> None:
            seen.append(name)
            raise ValueError(message)

        monkeypatch.setattr(dem_ds, "_validate_source_name", spy)
        with pytest.raises(ValueError, match="spy"):
            validate_mask_source_name("gsw")
        assert seen == ["gsw"]

    @pytest.mark.parametrize(
        "relative",
        [
            "../escape.tif",
            # double backslashes normalize to separators exactly like the
            # delegated dem guard, then reject as traversal
            "..\\\\../escape.tif",
            "/abs/path.tif",
            "a/../../b.tif",
        ],
    )
    def test_cache_relative_path_rejects_traversal(self, relative: str) -> None:
        """Check traversal and absolute cache paths fail closed."""
        with pytest.raises(ValueError, match="cache-relative"):
            mask_cache_relative_path(relative, None)

    @pytest.mark.parametrize(
        "relative",
        [
            "water-gsw/occurrence_0E_0N.tif",
            "N38_E100.tif.123-abc.part",
            "water-worldcover/vectors/abc123/layer.geojson",
        ],
    )
    def test_cache_relative_path_accepts_safe(self, relative: str) -> None:
        """Check safe cache-relative paths pass the delegated guard."""
        mask_cache_relative_path(relative, None)

    def test_cache_path_delegates_to_dem_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Check the wrapper routes through the dem_sources guard at call time."""
        seen: list[str] = []
        message = "spy"

        def spy(relative_path: str, _cache_dir: object) -> None:
            seen.append(relative_path)
            raise ValueError(message)

        monkeypatch.setattr(dem_ds, "validate_cache_relative_path", spy)
        with pytest.raises(ValueError, match="spy"):
            mask_cache_relative_path("water-gsw/x.tif", None)
        assert seen == ["water-gsw/x.tif"]

    def test_planned_cache_paths_are_guarded(self) -> None:
        """Check every planned tile cache path passes the traversal guard."""
        for bounds in (
            (9.0, -1.0, 11.0, 1.0),
            (-11.0, -1.0, -9.0, 1.0),
            (-77.0, -13.0, -76.5, -12.5),
        ):
            for name in ("gsw", "worldcover"):
                plan = get_mask_source(name).plan(bounds)
                for tile in plan.tiles:
                    relative = str(tile.cache_path)
                    mask_cache_relative_path(relative, None)
                    assert not relative.startswith("/")
                    assert ".." not in relative.split("/")
                    partition = f"{DEFAULT_PRODUCT}-{name}/"
                    assert relative.startswith(partition)


# ---------------------------------------------------------------------------
# 11. One shared tile-snap function
# ---------------------------------------------------------------------------


class TestTileSnapping:
    """The one shared tile-snap function (fetch band + seam guard)."""

    def test_gsw_band_snaps(self) -> None:
        """Check the pinned 10-degree GSW band examples."""
        assert tile_snap((165.0, 175.0), 10.0) == (160.0, 180.0)
        assert tile_snap((155.0, 169.0), 10.0) == (150.0, 170.0)

    def test_worldcover_band_snaps(self) -> None:
        """Check the pinned 3-degree WorldCover band examples."""
        assert tile_snap((0.5, 2.0), 3.0) == (0.0, 3.0)
        assert tile_snap((177.5, 179.0), 3.0) == (177.0, 180.0)

    def test_round5_guard_band_semantics(self) -> None:
        """Check the snap reproduces the executed round-5 seam-guard edges."""
        # in-tile band 165..175 reaches the seam -> guard fails closed
        assert tile_snap((165.0, 175.0), 10.0)[1] == 180.0
        # tile band short of the seam 155..169 stays at 170 -> guard passes
        assert tile_snap((155.0, 169.0), 10.0)[1] == 170.0
        # exact-180-span hemisphere (-90..90) stays far from the seam
        assert tile_snap((-90.0, 90.0), 10.0) == (-90.0, 100.0)
        # near-seam west-only band snaps onto -180 -> guard fails closed
        assert tile_snap((-179.9, -178.0), 10.0)[0] == -180.0

    def test_snap_exact_and_negative_bands(self) -> None:
        """Check exact multiples and negative bands snap outward correctly."""
        assert tile_snap((160.0, 170.0), 10.0) == (160.0, 180.0)
        assert tile_snap((-165.0, -155.0), 10.0) == (-170.0, -150.0)
        assert tile_snap((0.0, 0.5), 10.0) == (0.0, 10.0)

    def test_snap_covers_the_band(self) -> None:
        """Check the snapped band always contains the input band."""
        for lo, hi in ((165.0, 175.0), (155.0, 169.0), (-11.0, -9.0), (0.0, 0.5)):
            for size in (10.0, 3.0):
                snapped_lo, snapped_hi = tile_snap((lo, hi), size)
                assert snapped_lo <= lo
                assert snapped_hi >= hi

    def test_snap_rejects_nonpositive_size(self) -> None:
        """Check non-positive tile sizes fail closed."""
        with pytest.raises(ValueError, match="positive"):
            tile_snap((0.0, 1.0), 0.0)
        with pytest.raises(ValueError, match="positive"):
            tile_snap((0.0, 1.0), -3.0)

    @pytest.mark.parametrize(
        "bounds",
        [
            (9.0, -1.0, 11.0, 1.0),
            (100.0, 30.0, 120.0, 40.0),
            (8.0, 0.0, 10.0, 2.0),
            (100.0, 30.0, 102.0, 32.0),
            (0.0, 0.0, 0.5, 0.5),
        ],
    )
    def test_plan_tile_set_matches_shared_snap(
        self, bounds: tuple[float, float, float, float]
    ) -> None:
        """Check registry enumeration and the shared snap agree on tiles.

        The padded fetch band (mask.py) and the seam guard both use
        ``tile_snap``; the number of planned tiles must equal the snapped
        band's tile count so the guarded band is exactly what gets fetched.
        """
        for provider in ("gsw", "worldcover"):
            source = get_mask_source(provider)
            size = source.tile_size_deg
            lon_band = tile_snap((bounds[0], bounds[2]), size)
            lat_band = tile_snap((bounds[1], bounds[3]), size)
            n_lon = round((lon_band[1] - lon_band[0]) / size)
            n_lat = round((lat_band[1] - lat_band[0]) / size)
            plan = source.plan(bounds)
            assert len(plan.tiles) == n_lon * n_lat
