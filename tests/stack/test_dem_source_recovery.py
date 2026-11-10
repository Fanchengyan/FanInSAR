"""Recover managed DEM sources from bound records, never cache contents."""

# Tests intentionally use a namespace package.
# ruff: noqa: INP001

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from affine import Affine

from faninsar.processing.geometry import GridSpec, RasterDEM, SourceDEM
from faninsar.processing.geometry.managed_dem import DEMCoverageInsufficientError
from faninsar.stack.identity import save_stack_identity
from faninsar.stack.session import Stack
from tests.stack.test_dem_handoff import _dummy_safes, _fake_source_to_raster
from tests.support.local_stack import LocalStack

if TYPE_CHECKING:
    from pathlib import Path


def _prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalStack:
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(SourceDEM, "to_raster", _fake_source_to_raster([]))
    grid = GridSpec(
        crs="EPSG:4326",
        transform=Affine(0.001, 0, 10, 0, -0.001, 45),
        height=8,
        width=8,
    )
    stack = LocalStack.from_paths(
        _dummy_safes(tmp_path),
        work_dir=tmp_path / "work",
        dem=SourceDEM("glo30:pc", cache_dir=cache),
        dem_cache_dir=cache,
        grid=grid,
    )
    stack.prepare_data()
    # Simulate older identities which persisted the materialized raster only.
    stack._dem_source = None
    save_stack_identity(stack)
    return stack


def _record(stack: LocalStack) -> tuple[Path, dict[str, object]]:
    path = stack._stack_dem_sidecar_path()
    return path, json.loads(path.read_text(encoding="utf-8"))


def _no_fetch(monkeypatch: pytest.MonkeyPatch) -> Mock:
    fetch = Mock(side_effect=AssertionError("reopen must not materialize a DEM"))
    monkeypatch.setattr(SourceDEM, "to_raster", fetch)
    return fetch


def test_reopen_restores_recorded_source_amid_unrelated_cache_products(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound source, not directory ordering, supplies future coverage."""
    stack = _prepared(tmp_path, monkeypatch)
    cache = stack.config.dem_cache_dir
    (cache / "aaa-unrelated").mkdir()
    (cache / "glo90-copernicus").mkdir()
    path, record = _record(stack)
    record.pop("kind", None)  # Existing flat source records remain readable.
    path.write_text(json.dumps(record), encoding="utf-8")
    fetch = _no_fetch(monkeypatch)
    reopened = Stack.open(stack.config.work_dir)
    assert reopened._dem_source is not None
    assert reopened._dem_source.product == "glo30"
    assert reopened._dem_source.provider == "pc"
    assert reopened._dem_source.cache_dir == cache
    fetch.assert_not_called()


def test_reopened_source_expands_with_the_same_product_and_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later coverage request retains the persisted acquisition recipe."""
    stack = _prepared(tmp_path, monkeypatch)
    reopened = Stack.open(stack.config.work_dir)
    calls: list[tuple[str, str | None]] = []
    fake = _fake_source_to_raster([])

    def materialize(source: SourceDEM, grid: GridSpec, **kwargs: object) -> RasterDEM:
        calls.append((source.product, source.provider))
        return fake(source, grid, **kwargs)

    monkeypatch.setattr(SourceDEM, "to_raster", materialize)
    west, south, east, north = reopened.dem.grid.bounds
    reopened._expand_managed_dem((west - 0.001, south, east, north))
    assert calls == [("glo30", "pc")]


@pytest.mark.parametrize("missing", [True, False])
def test_no_source_record_never_guesses_from_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: bool
) -> None:
    """Missing records and explicit raster records cannot enable downloads."""
    stack = _prepared(tmp_path, monkeypatch)
    (stack.config.dem_cache_dir / "glo30-pc").mkdir()
    path, record = _record(stack)
    if missing:
        path.unlink()
    else:
        record.update(kind="raster")
        path.write_text(json.dumps(record), encoding="utf-8")
    fetch = _no_fetch(monkeypatch)
    reopened = Stack.open(stack.config.work_dir)
    assert reopened._dem_source is None
    assert isinstance(reopened.dem, RasterDEM)
    west, south, east, north = reopened.dem.grid.bounds
    with pytest.raises(DEMCoverageInsufficientError):
        reopened._expand_managed_dem((west - 0.001, south, east, north))
    fetch.assert_not_called()


@pytest.mark.parametrize("changed", ["path", "identity", "vertical_datum", "provider"])
def test_mismatched_source_record_does_not_authorize_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    """Corrupt source bindings are ignored without changing raster interpretation."""
    stack = _prepared(tmp_path, monkeypatch)
    path, record = _record(stack)
    record[changed] = {
        "path": "other-dem.tif",
        "identity": "not-the-raster-identity",
        "vertical_datum": "egm96",
        "provider": "not-a-provider",
    }[changed]
    path.write_text(json.dumps(record), encoding="utf-8")
    fetch = _no_fetch(monkeypatch)
    reopened = Stack.open(stack.config.work_dir)
    assert reopened._dem_source is None
    assert reopened.dem.vertical_datum == "ellipsoidal"
    fetch.assert_not_called()


def test_external_user_raster_does_not_inherit_managed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A separately supplied raster must not gain a stale workspace recipe."""
    stack = _prepared(tmp_path, monkeypatch)
    external_path = tmp_path / "user-dem.tif"
    stack.dem.save(external_path)
    external = RasterDEM(path=external_path)
    stack.config.dem = external
    stack._dem_source = None
    _no_fetch(monkeypatch)
    stack._restore_local_inputs()
    assert stack._dem_source is None
    assert stack.config.dem is external


def test_missing_managed_raster_restores_recipe_without_fetching_on_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inspection can reopen before an explicit operation rematerializes terrain."""
    stack = _prepared(tmp_path, monkeypatch)
    stack._stack_dem_raster_path().unlink()
    fetch = _no_fetch(monkeypatch)
    reopened = Stack.open(stack.config.work_dir)
    assert reopened._dem_source is not None
    assert (reopened._dem_source.product, reopened._dem_source.provider) == (
        "glo30",
        "pc",
    )
    fetch.assert_not_called()
    calls: list[GridSpec] = []
    monkeypatch.setattr(SourceDEM, "to_raster", _fake_source_to_raster(calls))
    reopened._ensure_raster_dem()
    assert len(calls) == 1
    assert reopened._stack_dem_raster_path().is_file()


def test_explicit_source_takes_precedence_over_stale_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A captured source is not overwritten by recovery metadata."""
    stack = _prepared(tmp_path, monkeypatch)
    source = SourceDEM("glo90:pc", cache_dir=stack.config.dem_cache_dir)
    stack.config.dem = source
    stack._capture_dem_source()
    stack._restore_local_inputs()
    assert stack._dem_source is source


@pytest.mark.parametrize("corrupt", [False, True])
def test_missing_raster_without_trusted_record_cannot_select_auto_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: bool
) -> None:
    """A missing recorded raster is not permission to pick a default provider."""
    stack = _prepared(tmp_path, monkeypatch)
    stack._stack_dem_raster_path().unlink()
    record_path, _ = _record(stack)
    if corrupt:
        record_path.write_text("not JSON", encoding="utf-8")
    else:
        record_path.unlink()
    fetch = _no_fetch(monkeypatch)
    reopened = Stack.open(stack.config.work_dir)
    assert reopened._dem_source is None
    with pytest.raises(DEMCoverageInsufficientError):
        reopened.materialize_dem()
    fetch.assert_not_called()


@pytest.mark.parametrize("content", ["[]", "null", "not JSON"])
def test_malformed_record_keeps_covered_raster_inspectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    """Malformed metadata cannot authorize a source or break covered local reads."""
    stack = _prepared(tmp_path, monkeypatch)
    record_path, _ = _record(stack)
    record_path.write_text(content, encoding="utf-8")
    fetch = _no_fetch(monkeypatch)
    reopened = Stack.open(stack.config.work_dir)
    assert reopened._dem_source is None
    assert reopened.dem.vertical_datum == "ellipsoidal"
    fetch.assert_not_called()


def test_coverage_failure_after_one_expansion_is_propagated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated shortages do not trigger an unbounded or eight-attempt loop."""
    stack = _prepared(tmp_path, monkeypatch)
    bounds = stack.dem.grid.bounds
    expand = Mock(return_value=stack.dem)
    monkeypatch.setattr(stack, "_expand_managed_dem", expand)
    operation = Mock(
        side_effect=DEMCoverageInsufficientError(
            "still short", required_bounds=bounds, actual_bounds=None
        )
    )
    with pytest.raises(DEMCoverageInsufficientError):
        stack._call_with_dem_coverage(None, operation)
    assert expand.call_count == 1
    assert operation.call_count == 2
