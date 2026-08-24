"""Executable bounded NISAR provider tests (PROPOSAL-0035)."""

# Test modules intentionally import runtime fixtures directly.
# ruff: noqa: TC001, TC002

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from faninsar.missions.nisar import NisarSensor
from faninsar.processing.coordinates import GeoGrid
from faninsar.processing.merge.grid import GeoGridSpec
from faninsar.processing.slc import GeoSLC, RadarSLC
from faninsar.processing.stack import NISARStack
from faninsar.processing.stack.scene_store import CoregisteredSceneStore

from .test_nisar_stack import _result


class _Dataset:
    """Small reader-like dataset that records bounded selections."""

    shape = (4, 5)
    dtype = np.dtype("complex64")

    def __init__(self, value: complex) -> None:
        self.samples = np.full(self.shape, value, dtype=np.complex64)
        self.selections: list[tuple[slice, slice]] = []

    def __getitem__(self, selection: tuple[slice, slice]) -> np.ndarray:
        self.selections.append(selection)
        return self.samples[selection]


def _stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    domain: str = "radar",
) -> tuple[NISARStack, tuple[Path, Path], dict[Path, SimpleNamespace]]:
    """Construct a Stack with two fake lazily-read RSLCs."""
    paths = tuple(
        tmp_path / f"NISAR_RSLC_{date_id}.h5"
        for date_id in ("20240101", "20240113")
    )
    for path in paths:
        path.touch()
    handles = {
        path: SimpleNamespace(
            filename=str(path),
            dataset=_Dataset(index + 1),
        )
        for index, path in enumerate(paths)
    }
    monkeypatch.setattr(
        NisarSensor,
        "open_product",
        lambda _sensor, uri: handles[Path(uri)],
    )
    monkeypatch.setattr(
        NisarSensor,
        "to_slc_product",
        lambda _sensor, handle, **_kwargs: _result(
            Path(handle.filename),
            Path(handle.filename).stem.rsplit("_", 1)[-1],
        ),
    )
    monkeypatch.setattr(
        NisarSensor,
        "read_slc_window",
        lambda _sensor, handle, window, **_kwargs: handle.dataset[window],
    )
    config: dict[str, object] = {
        "extra": {"nisar_window": (1, 4, 1, 5)},
        "multilook": (1, 1),
        "goldstein_alpha": 0.0,
        "coregistration_grid": domain,
    }
    if domain == "geo":
        config["geo_grid"] = GeoGridSpec(
            crs="EPSG:32633",
            transform=(100.0, 10.0, 0.0, 200.0, 0.0, -10.0),
            width=3,
            height=2,
            resolution_m=(10.0, 10.0),
        )
    stack = NISARStack.from_rslc(paths, work_dir=tmp_path / "work", **config)
    return stack, paths, handles


def test_nisar_radar_provider_runs_shared_stack_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Radar RSLC windows publish scenes consumed by shared IFG formation."""
    stack, paths, handles = _stack(tmp_path, monkeypatch)

    stack.prepare_scenes().coregister_scenes().form_interferograms(multilook=(1, 1))

    store = CoregisteredSceneStore.open(
        tmp_path / "work" / "coreg" / "20240113" / "scenes"
    )
    assert store.domain == "radar"
    assert store.grid_shape == (3, 4)
    assert stack.ifg_dirs == [
        tmp_path / "work" / "ifg" / "ml_1x1" / "20240101_20240113"
    ]
    expected = (slice(1, 4), slice(1, 5))
    assert handles[paths[0]].dataset.selections == [expected]
    assert handles[paths[1]].dataset.selections == [expected]


def test_nisar_geo_provider_uses_one_shared_geo_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Geo RSLC windows are converted through the common RadarSLC API."""
    calls: list[tuple[tuple[int, int], str]] = []

    def fake_rdr2geo(
        slc: RadarSLC,
        *,
        geo_grid: GeoGrid,
        **_kwargs: object,
    ) -> GeoSLC:
        calls.append((geo_grid.shape, geo_grid.crs))
        product = replace(
            slc.product,
            grid=geo_grid,
            samples=replace(slc.product.samples, shape=geo_grid.shape),
        )
        return GeoSLC(product=product, samples=np.ones(geo_grid.shape, np.complex64))

    monkeypatch.setattr(RadarSLC, "rdr2geo", fake_rdr2geo)
    stack, _paths, _handles = _stack(tmp_path, monkeypatch, domain="geo")

    stack.prepare_scenes().coregister_scenes().form_interferograms(multilook=(1, 1))

    store = CoregisteredSceneStore.open(
        tmp_path / "work" / "coreg" / "20240113" / "scenes"
    )
    assert store.domain == "geo"
    assert store.grid_shape == (2, 3)
    assert calls == [((2, 3), "EPSG:32633"), ((2, 3), "EPSG:32633")]
