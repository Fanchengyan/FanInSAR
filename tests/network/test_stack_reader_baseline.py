"""StackInterferogramCollection.baseline over committed scene products.

The collection reads the per-scene ``faninsar:baseline_perpendicular_m``
persisted at coregistration (PROPOSAL-0065): the Reference is its own
origin (``0.0``), each secondary carries its signed scene baseline, and
:class:`~faninsar.core.baseline.Baselines` is populated per date — no
``from_pair_wise`` inversion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar import Pairs
from faninsar.core.baseline import Baselines
from faninsar.io.storage.scene_store import write_scene
from faninsar.missions.s1 import S1Stack
from faninsar.network.readers.stack import StackInterferogramCollection
from tests.support.scene_stores import coreg_dir

if TYPE_CHECKING:
    from pathlib import Path

REFERENCE_ID = "20240101"
SECONDARY_ID = "20240113"
THIRD_ID = "20240125"
PLANNED = ("20240101_20240113", "20240113_20240125", "20240101_20240125")


def _stack_with_scene_baselines(
    tmp_path: Path,
    perp: dict[str, float | None] | None = None,
) -> S1Stack:
    """Build a three-date Stack with scenes carrying known baselines.

    ``perp`` maps a date id to its persisted signed perpendicular baseline;
    ``None`` commits the scene without the field (pre-change product).
    Defaults: Reference 0.0 (no field needed), 87.0, -42.5.
    """
    values = {SECONDARY_ID: 87.0, THIRD_ID: -42.5}
    if perp is not None:
        values.update(perp)

    paths = []
    for date_id in (REFERENCE_ID, SECONDARY_ID, THIRD_ID):
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000.SAFE"
        path.mkdir()
        paths.append(path)
    stack = S1Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        pairs=Pairs.from_names(list(PLANNED)),
        multilook=(1, 1),
    )

    units = [
        {
            "tag": "f0_IW1_b0",
            "payload": np.ones((8, 8), np.complex64),
            "row_origin": 0,
            "col_origin": 0,
        }
    ]
    work_dir = stack.config.work_dir
    reference = write_scene(
        coreg_dir(work_dir, REFERENCE_ID),
        date_id=REFERENCE_ID,
        reference_id=REFERENCE_ID,
        domain="radar",
        units=units,
        grid_shape=(8, 8),
    )
    for date_id in (SECONDARY_ID, THIRD_ID):
        value = values[date_id]
        properties = None
        if value is not None:
            properties = {
                "faninsar:baseline_parallel_m": 0.0,
                "faninsar:baseline_perpendicular_m": value,
            }
        write_scene(
            coreg_dir(work_dir, date_id),
            date_id=date_id,
            reference_id=REFERENCE_ID,
            domain="radar",
            units=units,
            grid_shape=(8, 8),
            inputs={"reference": reference.write_id},
            properties=properties,
        )
    return stack


def test_baseline_returns_per_date_signed_perpendicular(tmp_path: Path) -> None:
    """Baselines carries each scene's signed value keyed on its date."""
    stack = _stack_with_scene_baselines(tmp_path)
    baseline = stack.interferograms.baseline
    assert isinstance(baseline, Baselines)
    assert list(baseline.dates.strftime("%Y%m%d")) == [
        REFERENCE_ID,
        SECONDARY_ID,
        THIRD_ID,
    ]
    np.testing.assert_allclose(baseline.values, [0.0, 87.0, -42.5])


def test_baseline_to_pair_wise_recovers_pair_differences(tmp_path: Path) -> None:
    """Per-pair baselines are secondary − primary scene values."""
    stack = _stack_with_scene_baselines(tmp_path)
    collection = stack.interferograms
    pair_wise = collection.baseline.to_pair_wise(collection.pairs)
    np.testing.assert_allclose(
        pair_wise.to_numpy(),
        [87.0 - 0.0, -42.5 - 87.0, -42.5 - 0.0],
        rtol=1e-6,
    )


def test_baseline_sel_pairs_subset(tmp_path: Path) -> None:
    """A sel(pairs=) view reports the subset network's baselines."""
    stack = _stack_with_scene_baselines(tmp_path)
    subset = Pairs.from_names(["20240113_20240125"])
    view = stack.interferograms.sel(pairs=subset)
    baseline = view.baseline
    assert list(baseline.dates.strftime("%Y%m%d")) == [SECONDARY_ID, THIRD_ID]
    np.testing.assert_allclose(baseline.values, [87.0, -42.5])


def test_baseline_missing_scene_field_reads_nan(tmp_path: Path) -> None:
    """A pre-change secondary without the field contributes NaN."""
    stack = _stack_with_scene_baselines(tmp_path, perp={THIRD_ID: None})
    baseline = stack.interferograms.baseline
    np.testing.assert_allclose(baseline.values[:2], [0.0, 87.0])
    assert np.isnan(baseline.values[2])
