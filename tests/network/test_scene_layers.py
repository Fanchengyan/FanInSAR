"""Scene-layer lazy read view tests (PROPOSAL-0063)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.io.storage.scene_store import write_scene

if TYPE_CHECKING:
    from pathlib import Path


def _scene(
    root: Path,
    *,
    units: list[dict],
    grid_shape: tuple[int, int] = (16, 16),
    domain: str = "radar",
    **kwargs,
):
    return write_scene(
        root,
        date_id="20260706",
        reference_id="20260706",
        domain=domain,
        units=units,
        grid_shape=grid_shape,
        overwrite=True,
        **kwargs,
    )


def _payload(seed: int, shape: tuple[int, int] = (8, 8)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex64
    )


def test_unit_kind_burst_and_swaths(tmp_path: Path) -> None:
    store = _scene(
        tmp_path / "s",
        units=[
            {"tag": "f1_iw1_b0", "payload": _payload(0), "row_origin": 0, "col_origin": 0},
            {"tag": "f1_iw2_b0", "payload": _payload(1), "row_origin": 0, "col_origin": 8},
        ],
    )
    layers = store.layers
    assert layers.unit_kind == "burst"
    assert layers.bursts == ("f1_iw1_b0", "f1_iw2_b0")
    assert layers.swaths == ("iw1", "iw2")
    with pytest.raises(Exception, match="not tiles"):
        _ = layers.tiles


def test_unit_kind_two_part_tags(tmp_path: Path) -> None:
    store = _scene(
        tmp_path / "s",
        units=[
            {"tag": "iw1_b0", "payload": _payload(0), "row_origin": 0, "col_origin": 0},
        ],
        grid_shape=(8, 8),
    )
    assert store.layers.unit_kind == "burst"
    assert store.layers.swaths == ("iw1",)


def test_unit_kind_tile_nisar(tmp_path: Path) -> None:
    store = _scene(
        tmp_path / "s",
        units=[
            {
                "tag": "NISAR_b000001",
                "payload": _payload(0),
                "row_origin": 0,
                "col_origin": 0,
            }
        ],
        grid_shape=(8, 8),
    )
    layers = store.layers
    assert layers.unit_kind == "tile"
    assert layers.tiles == ("NISAR_b000001",)
    assert layers.swaths == ()
    with pytest.raises(Exception, match="not bursts"):
        _ = layers.bursts


def test_select_bounding_box_and_full(tmp_path: Path) -> None:
    store = _scene(
        tmp_path / "s",
        units=[
            {"tag": "f1_iw1_b0", "payload": _payload(0), "row_origin": 0, "col_origin": 0},
            {"tag": "f1_iw2_b0", "payload": _payload(1), "row_origin": 0, "col_origin": 8},
        ],
    )
    sel = store.layers.sel(bursts=("f1_iw1_b0",))
    assert sel.slc.extent == (8, 8)
    assert sel.slc.shape == (8, 8)
    assert sel.slc.values.shape == (8, 8)
    full = store.layers.full()
    assert full.slc.extent is None
    assert full.slc.shape == (16, 16)


def test_select_swaths_only(tmp_path: Path) -> None:
    store = _scene(
        tmp_path / "s",
        units=[
            {"tag": "f1_iw1_b0", "payload": _payload(0), "row_origin": 0, "col_origin": 0},
            {"tag": "f1_iw2_b0", "payload": _payload(1), "row_origin": 0, "col_origin": 8},
        ],
    )
    sel = store.layers.sel(swaths=("iw1",))
    assert set(sel._selection) == {"f1_iw1_b0"}


def test_select_wrong_mission_alias(tmp_path: Path) -> None:
    store = _scene(
        tmp_path / "s",
        units=[
            {
                "tag": "NISAR_b000001",
                "payload": _payload(0),
                "row_origin": 0,
                "col_origin": 0,
            }
        ],
        grid_shape=(8, 8),
    )
    with pytest.raises(Exception, match="use tiles="):
        store.layers.sel(bursts=("NISAR_b000001",))
    with pytest.raises(Exception, match="swaths="):
        store.layers.sel(swaths=("iw1",))


def test_virtual_layer_derivations(tmp_path: Path) -> None:
    store = _scene(
        tmp_path / "s",
        units=[
            {"tag": "f1_iw1_b0", "payload": _payload(0), "row_origin": 0, "col_origin": 0},
        ],
        grid_shape=(8, 8),
    )
    slc = store.layers.slc
    np.testing.assert_allclose(
        store.layers.amplitude.values, np.abs(slc.values), equal_nan=True
    )
    np.testing.assert_allclose(
        store.layers.intensity.values, np.abs(slc.values) ** 2, equal_nan=True
    )
    np.testing.assert_allclose(
        store.layers.phase.values, np.angle(slc.values), equal_nan=True
    )
    assert store.layers.slc.path is None


def test_mosaic_overlap_first_valid_wins(tmp_path: Path) -> None:
    u0 = np.ones((8, 8), np.complex64)
    u0[6:8, :] = 0.0
    u1 = np.full((8, 8), 2.0 + 0j, np.complex64)
    store = _scene(
        tmp_path / "s",
        units=[
            {"tag": "f1_iw1_b0", "payload": u0, "row_origin": 0, "col_origin": 0},
            {"tag": "f1_iw1_b1", "payload": u1, "row_origin": 4, "col_origin": 0},
        ],
        grid_shape=(12, 8),
    )
    mosaic = store.layers.slc.values
    assert np.all(mosaic[4:6, :] == 1.0 + 0j)
    assert np.all(mosaic[6:8, :] == 2.0 + 0j)


def test_valid_mask_computed(tmp_path: Path) -> None:
    u0 = np.ones((8, 8), np.complex64)
    u0[0, 0] = 0.0
    store = _scene(
        tmp_path / "s",
        units=[
            {"tag": "f1_iw1_b0", "payload": u0, "row_origin": 0, "col_origin": 0},
        ],
        grid_shape=(8, 8),
    )
    mask = store.layers.slc.valid_mask
    assert mask.dtype == np.bool_
    assert not mask[0, 0]
    assert mask[1, 1]


def test_to_xarray_lazy(tmp_path: Path) -> None:
    store = _scene(
        tmp_path / "s",
        units=[
            {"tag": "f1_iw1_b0", "payload": _payload(0), "row_origin": 0, "col_origin": 0},
        ],
        grid_shape=(8, 8),
    )
    layer = store.layers.slc
    da = layer.to_xarray()
    assert da.dims == ("y", "x")
    assert da.shape == layer.shape
    assert da.attrs["asset_kind"] == "slc"
    assert da.attrs["source_write_id"] == store.write_id
    np.testing.assert_allclose(da.compute().values, layer.values, equal_nan=True)
