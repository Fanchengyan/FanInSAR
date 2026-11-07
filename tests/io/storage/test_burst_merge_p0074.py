"""P0074 storage seams: windowed reads, explicit support, burst/merged IFGs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.io.storage._zarr_stac import read_zarr_array, write_zarr_array
from faninsar.io.storage.ifg_store import (
    InterferogramArtifactStore,
    ifg_artifact_dir,
    unwrap_root,
    write_ifg_artifact,
    write_unwrapped_artifact,
)
from faninsar.io.storage.product import open_product, publish_product
from faninsar.io.storage.scene_store import (
    CoregisteredSceneStore,
    form_merged_scene_interferogram,
    write_scene,
)
from faninsar.processing.errors import InvalidProcessingStateError

if TYPE_CHECKING:
    from pathlib import Path

REFERENCE_ID = "20240101"
SECONDARY_ID = "20240113"
_SOURCES = {"primary": "a" * 64, "secondary": "b" * 64}


def _write_arrays(arrays: dict[str, np.ndarray]):
    """Return a product payload writer for ``arrays``."""

    def writer(staging: Path) -> dict[str, dict[str, object]]:
        assets: dict[str, dict[str, object]] = {}
        for name, value in arrays.items():
            write_zarr_array(staging / f"{name}.zarr", value)
            assets[name] = {"href": f"{name}.zarr"}
        return assets

    return writer


def _scene_units(
    payload: np.ndarray,
    *,
    tag: str = "IW1_b0",
    row_origin: int = 0,
    col_origin: int = 0,
    valid_mask: np.ndarray | None = None,
    burst_id: str | None = None,
    phase_screen: np.ndarray | None = None,
) -> list[dict[str, object]]:
    unit: dict[str, object] = {
        "tag": tag,
        "payload": payload,
        "row_origin": row_origin,
        "col_origin": col_origin,
    }
    if valid_mask is not None:
        unit["valid_mask"] = valid_mask
    if burst_id is not None:
        unit["burst_id"] = burst_id
    if phase_screen is not None:
        unit["phase_screen"] = phase_screen
    return [unit]


def _write_pair(
    root: Path,
    reference: np.ndarray,
    secondary: np.ndarray,
    *,
    reference_units: list[dict[str, object]] | None = None,
    secondary_units: list[dict[str, object]] | None = None,
    grid_shape: tuple[int, int] | None = None,
) -> tuple[CoregisteredSceneStore, CoregisteredSceneStore]:
    ref_units = reference_units or _scene_units(reference)
    sec_units = secondary_units or _scene_units(secondary)
    header: dict[str, object] = {}
    if grid_shape is not None:
        header["grid_shape"] = grid_shape
    reference_store = write_scene(
        root / "reference",
        date_id=REFERENCE_ID,
        reference_id=REFERENCE_ID,
        domain="radar",
        units=ref_units,
        **header,
    )
    secondary_store = write_scene(
        root / "secondary",
        date_id=SECONDARY_ID,
        reference_id=REFERENCE_ID,
        domain="radar",
        units=sec_units,
        inputs={"reference": reference_store.write_id},
        **header,
    )
    return reference_store, secondary_store


def test_read_zarr_array_returns_local_window_without_full_copy(tmp_path: Path) -> None:
    """A half-open local window is the returned 2-D slice."""
    payload = np.arange(20, dtype=np.float32).reshape(4, 5)
    path = tmp_path / "block.zarr"
    write_zarr_array(path, payload)
    windowed = read_zarr_array(path, window=(1, 3, 2, 5))
    np.testing.assert_array_equal(windowed, payload[1:3, 2:5])
    np.testing.assert_array_equal(read_zarr_array(path), payload)


def test_read_zarr_array_rejects_out_of_bounds_window(tmp_path: Path) -> None:
    """Windows must lie strictly inside the array."""
    path = tmp_path / "block.zarr"
    write_zarr_array(path, np.ones((2, 3), dtype=np.float32))
    with pytest.raises(InvalidProcessingStateError, match="window"):
        read_zarr_array(path, window=(0, 2, 0, 4))
    with pytest.raises(InvalidProcessingStateError, match="window"):
        read_zarr_array(path, window=(1, 1, 0, 1))


def test_product_read_window_is_local_to_the_named_asset(tmp_path: Path) -> None:
    """Product.read slices one asset in local array coordinates."""
    payload = np.arange(12, dtype=np.float32).reshape(3, 4)
    product = publish_product(
        tmp_path / "p",
        kind="scene",
        properties={"faninsar:date": REFERENCE_ID},
        write_payloads=_write_arrays({"a": payload}),
    )
    np.testing.assert_array_equal(product.read("a", window=(1, 3, 1, 3)), payload[1:3, 1:3])
    reopened = open_product(tmp_path / "p")
    np.testing.assert_array_equal(reopened.read("a"), payload)


def test_scene_persists_explicit_valid_mask_and_physical_burst_id(
    tmp_path: Path,
) -> None:
    """Zero-valued valid samples survive only with an explicit mask asset."""
    payload = np.zeros((2, 3), dtype=np.complex64)
    payload[0, 0] = 1.0 + 0.0j
    valid_mask = np.array([[True, True, False], [False, True, True]])
    store = write_scene(
        tmp_path / "scene",
        date_id=REFERENCE_ID,
        reference_id=REFERENCE_ID,
        domain="radar",
        units=_scene_units(
            payload,
            valid_mask=valid_mask,
            burst_id="IW1_b0",
        ),
    )
    unit = store.unit_map()["IW1_b0"]
    assert unit.burst_id == "IW1_b0"
    np.testing.assert_array_equal(store.read_valid_mask("IW1_b0"), valid_mask)
    slc, _ = store.read("IW1_b0")
    np.testing.assert_array_equal(slc, payload)
    item = json.loads((store.root / "item.json").read_text(encoding="utf-8"))
    assert item["assets"]["IW1_b0"]["faninsar:burst_id"] == "IW1_b0"
    assert item["assets"]["IW1_b0.valid"]["faninsar:role"] == "valid_mask"


def test_scene_without_valid_mask_still_opens_for_no_roi_products(
    tmp_path: Path,
) -> None:
    """No-ROI products omit support metadata and still reopen."""
    payload = np.ones((2, 2), dtype=np.complex64)
    store = write_scene(
        tmp_path / "scene",
        date_id=REFERENCE_ID,
        reference_id=REFERENCE_ID,
        domain="radar",
        units=_scene_units(payload),
    )
    assert store.read_valid_mask("IW1_b0") is None
    slc, unit = store.read("IW1_b0")
    np.testing.assert_array_equal(slc, payload)
    assert unit.burst_id is None


def test_scene_read_and_phase_screen_honour_local_unit_window(tmp_path: Path) -> None:
    """Scene payload and phase-screen windows are local to the unit origin."""
    payload = (
        np.arange(20, dtype=np.float32).reshape(4, 5) + 1.0j * np.arange(20).reshape(4, 5)
    ).astype(np.complex64)
    screen = np.arange(20, dtype=np.float32).reshape(4, 5)
    store = write_scene(
        tmp_path / "scene",
        date_id=SECONDARY_ID,
        reference_id=REFERENCE_ID,
        domain="radar",
        units=_scene_units(
            payload,
            row_origin=4,
            col_origin=2,
            phase_screen=screen,
        ),
        grid_shape=(10, 10),
    )
    window = (1, 3, 2, 4)
    slc, unit = store.read("IW1_b0", window=window)
    np.testing.assert_array_equal(slc, payload[1:3, 2:4])
    assert unit.row_origin == 4
    assert unit.col_origin == 2
    np.testing.assert_array_equal(store.read_phase_screen("IW1_b0", window=window), screen[1:3, 2:4])


def test_supported_zero_look_is_not_erased_as_missing_support(tmp_path: Path) -> None:
    """A two-sample look of +1 and -1 stays supported 0j, not empty."""
    primary = np.ones((2, 1), dtype=np.complex64)
    secondary = np.array([[1.0 + 0.0j], [-1.0 + 0.0j]], dtype=np.complex64)
    support = np.ones((1, 1), dtype=bool)
    reference_store, secondary_store = _write_pair(
        tmp_path,
        primary[:1],
        secondary[:1],
        reference_units=[
            *_scene_units(primary[:1], tag="IW1_b0", valid_mask=support),
            *_scene_units(
                primary[1:], tag="IW1_b1", row_origin=1, valid_mask=support
            ),
        ],
        secondary_units=[
            *_scene_units(secondary[:1], tag="IW1_b0", valid_mask=support),
            *_scene_units(
                secondary[1:], tag="IW1_b1", row_origin=1, valid_mask=support
            ),
        ],
        grid_shape=(2, 1),
    )
    product = form_merged_scene_interferogram(
        reference_store,
        secondary_store,
        multilook=(2, 1),
    )
    assert product.complex.shape == (1, 1)
    np.testing.assert_allclose(product.complex[0, 0], 0.0 + 0.0j, atol=1e-6)
    assert product.valid_mask is not None
    assert bool(product.valid_mask[0, 0])
    np.testing.assert_allclose(product.coherence[0, 0], 0.0, atol=1e-6)


def test_single_input_roi_hole_is_excluded_before_look_statistics(
    tmp_path: Path,
) -> None:
    """A full-grid single unit still honours holes inside the bbox."""
    primary = np.array(
        [[1.0 + 0.0j, 1.0 + 0.0j], [1000.0j, -1000.0j]],
        dtype=np.complex64,
    )
    secondary = np.ones((2, 2), dtype=np.complex64)
    valid_mask = np.array([[True, True], [False, False]])
    reference_store, secondary_store = _write_pair(
        tmp_path,
        primary,
        secondary,
        reference_units=_scene_units(primary, valid_mask=valid_mask),
        secondary_units=_scene_units(secondary, valid_mask=valid_mask),
    )
    product = form_merged_scene_interferogram(
        reference_store,
        secondary_store,
        multilook=(2, 2),
    )
    np.testing.assert_allclose(product.complex[0, 0], 1.0 + 0.0j, atol=1e-6)
    assert product.valid_mask is not None
    assert bool(product.valid_mask[0, 0])


def test_ifg_merged_path_stays_under_ml_and_defaults_representation(
    tmp_path: Path,
) -> None:
    """Default merged IFGs keep the existing ``ifg/ml_AxR/pair`` location."""
    root = ifg_artifact_dir(tmp_path / "work", (2, 3), (REFERENCE_ID, SECONDARY_ID))
    assert root == tmp_path / "work" / "ifg" / "ml_2x3" / f"{REFERENCE_ID}_{SECONDARY_ID}"
    store = write_ifg_artifact(
        root,
        pair=(REFERENCE_ID, SECONDARY_ID),
        looks=(2, 3),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests=_SOURCES,
        complex_ifg=np.ones((2, 3), dtype=np.complex64),
        coherence=np.full((2, 3), 0.5, dtype=np.float32),
        valid_mask=np.ones((2, 3), dtype=bool),
    )
    assert store.representation == "merged"
    assert store.burst_id is None
    assert store.look_origin == (0, 0)
    np.testing.assert_array_equal(
        store.read_layer("complex", window=(0, 1, 1, 3)),
        np.ones((1, 2), dtype=np.complex64),
    )
    assert unwrap_root(root) == tmp_path / "work" / "unwrap" / "ml_2x3" / (
        f"{REFERENCE_ID}_{SECONDARY_ID}"
    )


def test_ifg_burst_products_coexist_with_merged_and_reject_unwrap(
    tmp_path: Path,
) -> None:
    """Burst IFGs live under ``ifg/bursts/...`` and cannot unwrap by default."""
    work = tmp_path / "work"
    merged_root = ifg_artifact_dir(work, (1, 1), (REFERENCE_ID, SECONDARY_ID))
    burst_root = ifg_artifact_dir(
        work,
        (1, 1),
        (REFERENCE_ID, SECONDARY_ID),
        representation="burst",
        burst_id="IW1_b0",
    )
    complex_merged = np.full((2, 2), 2.0 + 0.0j, dtype=np.complex64)
    complex_burst = np.full((1, 2), 3.0 + 0.0j, dtype=np.complex64)
    merged = write_ifg_artifact(
        merged_root,
        pair=(REFERENCE_ID, SECONDARY_ID),
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests=_SOURCES,
        complex_ifg=complex_merged,
        coherence=np.ones((2, 2), dtype=np.float32),
        valid_mask=np.ones((2, 2), dtype=bool),
        layout_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
    )
    burst = write_ifg_artifact(
        burst_root,
        pair=(REFERENCE_ID, SECONDARY_ID),
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests=_SOURCES,
        complex_ifg=complex_burst,
        coherence=np.ones((1, 2), dtype=np.float32),
        valid_mask=np.ones((1, 2), dtype=bool),
        representation="burst",
        burst_id="IW1_b0",
        row_origin=1,
        col_origin=0,
        grid_shape=(2, 2),
        look_origin=(0, 0),
        layout_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
    )
    reopened_merged = InterferogramArtifactStore.open(merged_root)
    reopened_burst = InterferogramArtifactStore.open(burst_root)
    assert reopened_merged.representation == "merged"
    assert reopened_burst.representation == "burst"
    assert reopened_burst.burst_id == "IW1_b0"
    assert reopened_burst.row_origin == 1
    assert reopened_burst.grid_shape == (2, 2)
    assert reopened_burst.layout_id == merged.layout_id
    np.testing.assert_array_equal(reopened_merged.read().complex, complex_merged)
    np.testing.assert_array_equal(reopened_burst.read().complex, complex_burst)
    with pytest.raises(InvalidProcessingStateError, match="merged"):
        unwrap_root(burst_root)
    with pytest.raises(InvalidProcessingStateError, match="merged"):
        write_unwrapped_artifact(
            burst.root,
            unwrapped_phase=np.zeros((1, 2), dtype=np.float32),
            connected_components=np.ones((1, 2), dtype=np.int32),
            method="snaphu",
            method_parameters={},
            ifg_manifest_digest=burst.write_id,
        )
    write_unwrapped_artifact(
        merged.root,
        unwrapped_phase=np.zeros((2, 2), dtype=np.float32),
        connected_components=np.ones((2, 2), dtype=np.int32),
        method="snaphu",
        method_parameters={},
        ifg_manifest_digest=merged.write_id,
    )
    assert merged.has_unwrapped()


def test_ifg_burst_id_must_be_a_basename(tmp_path: Path) -> None:
    """Burst identifiers cannot escape the product directory."""
    with pytest.raises(InvalidProcessingStateError, match="basename"):
        write_ifg_artifact(
            tmp_path / "ifg" / "bursts" / "ml_1x1" / "a_b" / "bad",
            pair=(REFERENCE_ID, SECONDARY_ID),
            looks=(1, 1),
            filter_name="none",
            filter_parameters={},
            source_manifest_digests=_SOURCES,
            complex_ifg=np.ones((1, 1), dtype=np.complex64),
            representation="burst",
            burst_id="../IW1_b0",
        )
