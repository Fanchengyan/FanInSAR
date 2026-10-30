"""Reference-bound geometry and LOS layers on Stack scenes (PROPOSAL-0066)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.missions.s1 import S1Stack
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry.coordinates import RadarGrid
from faninsar.processing.geometry.record import radar_geometry_record

if TYPE_CHECKING:
    from pathlib import Path

_GRID_SHAPE = (8, 16)


def _geometry_record() -> dict[str, object]:
    """Return a strict-JSON radar-geometry record on the test scene grid."""
    epoch = datetime(2024, 1, 1, tzinfo=UTC)
    vectors = tuple(
        OrbitStateVector(
            time=epoch + timedelta(seconds=float(k)),
            position_m=(7000_000.0, 0.0, 100.0 * k),
            velocity_m_s=(0.0, 7500.0, 0.0),
        )
        for k in range(40)
    )
    orbit = OrbitMetadata(reference_frame="ECFT", source="test", vectors=vectors)
    rgrid = RadarGrid(
        shape=_GRID_SHAPE,
        starting_slant_range_m=800_000.0,
        range_spacing_m=2.3,
        sensing_start=epoch,
        azimuth_time_interval_s=0.002,
        wavelength_m=0.0555,
        look_direction="right",
    )
    return radar_geometry_record(rgrid, orbit)


def _units() -> list[dict[str, object]]:
    """Two bursts side by side covering the (8, 16) common grid."""
    return [
        {
            "tag": "f0_IW1_b0",
            "payload": np.ones((8, 8), np.complex64),
            "row_origin": 0,
            "col_origin": 0,
        },
        {
            "tag": "f0_IW2_b0",
            "payload": np.ones((8, 8), np.complex64),
            "row_origin": 0,
            "col_origin": 8,
        },
    ]


def _stack_with_geometry(tmp_path: Path) -> S1Stack:
    """Build a two-date Stack whose Reference carries a radar-geometry record."""
    from faninsar import Pairs
    from faninsar.io.storage.scene_store import write_scene
    from tests.support.scene_stores import coreg_dir

    dates = ("20240101", "20240113")
    paths = []
    for date_id in dates:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000.SAFE"
        path.mkdir()
        paths.append(path)
    stack = S1Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        pairs=Pairs.from_names(["20240101_20240113"]),
        multilook=(1, 1),
    )
    reference_root = coreg_dir(stack.config.work_dir, stack.reference)
    reference = write_scene(
        reference_root,
        date_id=stack.reference,
        reference_id=stack.reference,
        domain="radar",
        units=_units(),
        grid_shape=_GRID_SHAPE,
        properties={"faninsar:radar_geometry": _geometry_record()},
    )
    write_scene(
        coreg_dir(stack.config.work_dir, "20240113"),
        date_id="20240113",
        reference_id=stack.reference,
        domain="radar",
        units=_units(),
        grid_shape=_GRID_SHAPE,
        inputs={"reference": reference.write_id},
    )
    return stack


def _item(root: Path) -> dict[str, object]:
    """Return the parsed item.json of one scene product."""
    return json.loads((root / "item.json").read_text(encoding="utf-8"))


def test_geometry_layers_materialize_lazily_with_same_write_id(
    tmp_path: Path,
) -> None:
    """First .layers read commits the six assets on the Reference item."""
    stack = _stack_with_geometry(tmp_path)
    reference_root = stack.config.work_dir / "coreg" / stack.reference
    before = _item(reference_root)
    write_id = before["properties"]["faninsar:write_id"]
    assert "latitude" not in before["assets"]

    layer = stack.scenes.latitude
    assert layer.shape == _GRID_SHAPE
    values = layer.values
    assert values.shape == _GRID_SHAPE
    assert np.isfinite(values).all()

    after = _item(reference_root)
    for name in (
        "latitude",
        "longitude",
        "height",
        "incidence",
        "azimuth",
        "geom_flags",
    ):
        assert name in after["assets"]
    # The augmentation preserves the scene write_id so bound secondaries do
    # not go stale.
    assert after["properties"]["faninsar:write_id"] == write_id


def test_geometry_fields_read_committed_zarr(tmp_path: Path) -> None:
    """All six committed fields read dense float64/uint8 rasters."""
    stack = _stack_with_geometry(tmp_path)
    scenes = stack.scenes
    assert scenes.longitude.values.shape == _GRID_SHAPE
    assert scenes.height.values.shape == _GRID_SHAPE
    assert scenes.incidence.values.dtype == np.float64
    assert scenes.azimuth.values.dtype == np.float64
    flags = scenes.geom_flags.values
    assert flags.dtype == np.uint8
    # The toy flat DEM yields no layover or shadow lanes.
    assert np.unique(flags).tolist() == [0]


def test_los_layers_are_virtual_unit_vectors(tmp_path: Path) -> None:
    """los_* derive ground-to-satellite unit vectors; no asset is persisted."""
    stack = _stack_with_geometry(tmp_path)
    reference_root = stack.config.work_dir / "coreg" / stack.reference
    east = stack.scenes.los_east
    north = stack.scenes.los_north
    up = stack.scenes.los_up
    assert east.path is None  # virtual layer: no payload
    e, n, u = east.values, north.values, up.values
    norm = np.sqrt(e**2 + n**2 + u**2)
    finite = np.isfinite(norm)
    assert np.allclose(norm[finite], 1.0, atol=1e-6)
    # No los_* asset is written on the item.
    assets = _item(reference_root)["assets"]
    assert not any(name.startswith("los_") for name in assets)


def test_secondary_geometry_layer_raises(tmp_path: Path) -> None:
    """A secondary's geometry layer fails fast: fields are Reference-bound."""
    stack = _stack_with_geometry(tmp_path)
    secondary = stack.scenes["20240113"]
    assert not secondary.is_reference
    with pytest.raises(InvalidProcessingStateError, match="Reference"):
        secondary.layers.latitude
    with pytest.raises(InvalidProcessingStateError, match="Reference"):
        secondary.layers.los_up


def test_los_cache_drops_on_new_collection(tmp_path: Path) -> None:
    """A fresh stack.scenes collection does not share the LOS cache."""
    stack = _stack_with_geometry(tmp_path)
    first = stack.scenes
    first_layer = first.los_up
    assert first._los_cache  # populated
    second = stack.scenes
    assert second._los_cache == {}  # fresh dict per collection
    assert second.los_up is not first_layer


def test_verify_catches_corrupted_geometry_asset(tmp_path: Path) -> None:
    """verify_product flags a damaged committed geometry payload."""
    stack = _stack_with_geometry(tmp_path)
    reference_root = stack.config.work_dir / "coreg" / stack.reference
    _ = stack.scenes.incidence.values  # materialize the six assets

    from faninsar.io.storage.product import verify_product

    assert verify_product(reference_root) == []
    import zarr

    array = zarr.open_array(store=str(reference_root / "incidence.zarr"), mode="a")
    array[0, 0] = -999.0
    assert "incidence" in verify_product(reference_root)
