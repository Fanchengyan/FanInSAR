"""Persisted scene masks and IFG intersection for P0078."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from faninsar.io.storage.scene_store import (
    form_burst_scene_interferogram,
    write_scene,
)

if TYPE_CHECKING:
    from pathlib import Path

REFERENCE_ID = "20200101"
DATE_B = "20200113"
DATE_C = "20200125"
TAG = "f0_IW1_b0"
SHAPE = (8, 8)
COMPLEX_NAN = np.complex64(np.nan + 1j * np.nan)
_REF_HOLE = (3, 2)
_B_HOLE = (3, 5)
_C_HOLE = (6, 3)


def _payload(hole: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return a complex payload and mask with one echo hole and a valid zero."""
    samples = np.ones(SHAPE, dtype=np.complex64)
    valid = np.ones(SHAPE, dtype=bool)
    samples[4, 4] = np.complex64(0.0)
    valid[hole] = False
    samples[hole] = COMPLEX_NAN
    return samples, valid


def _write_date(
    root: Path,
    date_id: str,
    hole: tuple[int, int],
    *,
    reference_write_id: str | None = None,
):
    """Persist one scene product with an explicit valid_mask."""
    payload, valid = _payload(hole)
    inputs = None if reference_write_id is None else {"reference": reference_write_id}
    return write_scene(
        root / date_id,
        date_id=date_id,
        reference_id=REFERENCE_ID,
        domain="radar",
        units=[
            {
                "tag": TAG,
                "payload": payload,
                "row_origin": 0,
                "col_origin": 0,
                "valid_mask": valid,
                "burst_id": "IW1_b0",
            }
        ],
        inputs=inputs,
    )


def test_reopened_scene_mask_matches_nan_payload(tmp_path: Path) -> None:
    """Stored valid_mask and NaN payload stay aligned after reopen."""
    store = _write_date(tmp_path, REFERENCE_ID, _REF_HOLE)
    write_id = store.write_id
    reopened = store.__class__.open(store.root)
    assert reopened.write_id == write_id
    mask = np.asarray(reopened.read_valid_mask(TAG), dtype=bool)
    payload, _ = reopened.read(TAG)
    assert not mask[_REF_HOLE]
    assert bool(mask[4, 4])
    assert np.isnan(payload.real[~mask]).all()
    assert payload[4, 4] == np.complex64(0.0)


def test_ifg_intersects_actual_endpoint_masks_not_reference_only(
    tmp_path: Path,
) -> None:
    """A B-C interferogram is invalid only where either endpoint is invalid."""
    reference = _write_date(tmp_path, REFERENCE_ID, _REF_HOLE)
    date_b = _write_date(
        tmp_path, DATE_B, _B_HOLE, reference_write_id=reference.write_id
    )
    date_c = _write_date(
        tmp_path, DATE_C, _C_HOLE, reference_write_id=reference.write_id
    )
    product = form_burst_scene_interferogram(date_b, date_c, TAG)
    mask = np.asarray(product.valid_mask, dtype=bool)
    assert mask[_REF_HOLE]
    assert not mask[_B_HOLE]
    assert not mask[_C_HOLE]
    assert bool(mask[0, 0])
    assert np.isnan(product.complex.real[~mask]).all()
