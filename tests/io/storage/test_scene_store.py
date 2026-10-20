"""Scene products on the storage kernel (PROPOSAL-0059)."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from faninsar.io.storage.scene_store import (
    CoregisteredSceneStore,
    SceneWriter,
    form_merged_scene_interferogram,
    form_scene_interferograms,
    validate_reference_binding,
    write_scene,
)
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.interferometry import PhaseFilter, PhaseFilterResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class _InvalidPhaseFilter(PhaseFilter):
    """Test double that violates one formation-boundary output postcondition."""

    def __init__(self, violation: str) -> None:
        self.violation = violation

    def apply(
        self,
        interferogram: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> PhaseFilterResult:
        """Return a deliberately malformed filter result."""
        assert valid_mask is not None
        if self.violation == "shape":
            return PhaseFilterResult(interferogram[:, :-1], valid_mask[:, :-1])
        if self.violation == "dtype":
            return PhaseFilterResult(interferogram.real, valid_mask)
        if self.violation == "mask_dtype":
            return PhaseFilterResult(interferogram, valid_mask.to(torch.uint8))
        if self.violation == "mask_subset":
            return PhaseFilterResult(interferogram, torch.ones_like(valid_mask))
        if self.violation == "finite":
            output = interferogram.clone()
            output[0, 1] = torch.nan + 0j
            return PhaseFilterResult(output, valid_mask)
        raise AssertionError


REFERENCE_ID = "20240101"
SECONDARY_ID = "20240113"


def _units(
    payload: np.ndarray,
    tags: Sequence[tuple[str, int, int]],
) -> list[dict[str, object]]:
    return [
        {"tag": tag, "payload": payload, "row_origin": row, "col_origin": col}
        for tag, row, col in tags
    ]


def _write_pair(
    root: Path,
    reference: np.ndarray,
    secondary: np.ndarray,
    *,
    tag: str | Sequence[tuple[str, int, int]] = "IW1_b0",
    row_origin: int = 0,
    col_origin: int = 0,
    reference_id: str = REFERENCE_ID,
    secondary_id: str = SECONDARY_ID,
    secondary_tags: Sequence[tuple[str, int, int]] | None = None,
    **header: object,
) -> tuple[Path, Path]:
    """Commit one Reference product and one secondary bound to it."""
    tags = [(tag, row_origin, col_origin)] if isinstance(tag, str) else list(tag)
    domain = str(header.pop("domain", "radar"))
    reference_store = write_scene(
        root / "reference",
        date_id=reference_id,
        reference_id=reference_id,
        domain=domain,
        units=_units(reference, tags),
        **header,
    )
    write_scene(
        root / "secondary",
        date_id=secondary_id,
        reference_id=reference_id,
        domain=domain,
        units=_units(secondary, secondary_tags or tags),
        inputs={"reference": reference_store.write_id},
        **header,
    )
    return root / "reference", root / "secondary"


def _write_secondary(
    root: Path,
    reference_root: Path,
    payload: np.ndarray,
    *,
    date_id: str,
    tags: Sequence[tuple[str, int, int]] = (("IW1_b0", 0, 0),),
    **header: object,
) -> Path:
    reference = CoregisteredSceneStore.open(reference_root)
    write_scene(
        root,
        date_id=date_id,
        reference_id=reference.reference_id,
        domain=reference.domain,
        units=_units(payload, tags),
        inputs={"reference": reference.write_id},
        **header,
    )
    return root


def test_scene_store_round_trip_and_ifg_uses_aligned_payloads(tmp_path: Path) -> None:
    """Persisted Reference-aligned arrays round-trip into one exact IFG."""
    reference = np.ones((2, 3), dtype=np.complex64)
    secondary = np.full((2, 3), 1.0 + 2.0j, dtype=np.complex64)
    first, second = _write_pair(tmp_path, reference, secondary)
    output = form_scene_interferograms(
        CoregisteredSceneStore.open(first), CoregisteredSceneStore.open(second)
    )
    np.testing.assert_array_equal(output["IW1_b0"], np.conj(secondary))


def test_reference_and_secondary_products_hold_one_payload_each(
    tmp_path: Path,
) -> None:
    """The Reference is stored once; a secondary holds only its own SLC."""
    reference = np.ones((2, 3), dtype=np.complex64)
    secondary = np.full((2, 3), 1.0 + 2.0j, dtype=np.complex64)
    first, second = _write_pair(tmp_path, reference, secondary)

    reference_store = CoregisteredSceneStore.open(first)
    secondary_store = CoregisteredSceneStore.open(second)
    assert reference_store.role == "reference"
    assert reference_store.is_reference
    assert secondary_store.role == "secondary"
    assert sorted(path.name for path in first.glob("*.zarr")) == ["IW1_b0.slc.zarr"]
    assert sorted(path.name for path in second.glob("*.zarr")) == ["IW1_b0.slc.zarr"]
    payload, _ = secondary_store.read("IW1_b0")
    np.testing.assert_array_equal(payload, secondary)
    assert secondary_store.inputs == {"reference": reference_store.write_id}
    assert reference_store.inputs == {}
    validate_reference_binding(reference_store, secondary_store)


def test_secondary_without_reference_binding_is_rejected(tmp_path: Path) -> None:
    """A secondary that binds no Reference write id cannot be consumed."""
    data = np.ones((2, 2), dtype=np.complex64)
    first, _ = _write_pair(tmp_path, data, data)
    unbound = write_scene(
        tmp_path / "unbound",
        date_id=SECONDARY_ID,
        reference_id=REFERENCE_ID,
        domain="radar",
        units=_units(data, [("IW1_b0", 0, 0)]),
    )
    reference = CoregisteredSceneStore.open(first)
    with pytest.raises(InvalidProcessingStateError, match="another Reference"):
        validate_reference_binding(reference, unbound)
    with pytest.raises(InvalidProcessingStateError, match="different Reference"):
        form_merged_scene_interferogram(reference, unbound)


def test_overwritten_reference_invalidates_bound_secondaries(
    tmp_path: Path,
) -> None:
    """A rewritten Reference gets a new write id; old secondaries go stale."""
    data = np.ones((2, 2), dtype=np.complex64)
    first, second = _write_pair(tmp_path, data, data)
    old = CoregisteredSceneStore.open(first).write_id
    write_scene(
        first,
        date_id=REFERENCE_ID,
        reference_id=REFERENCE_ID,
        domain="radar",
        units=_units(np.full((2, 2), 2.0, dtype=np.complex64), [("IW1_b0", 0, 0)]),
        overwrite=True,
    )
    reference_store = CoregisteredSceneStore.open(first)
    secondary_store = CoregisteredSceneStore.open(second)
    assert reference_store.write_id != old
    assert secondary_store.stale_inputs({"reference": reference_store}) == ["reference"]
    assert secondary_store.stale_inputs({"reference": old}) == []
    with pytest.raises(InvalidProcessingStateError, match="another Reference"):
        validate_reference_binding(reference_store, secondary_store)
    with pytest.raises(InvalidProcessingStateError, match="different Reference"):
        form_merged_scene_interferogram(reference_store, secondary_store)


def test_secondary_pair_ifg_uses_both_secondary_payloads(tmp_path: Path) -> None:
    """A pair of two secondaries never reads the Reference payload."""
    reference = np.ones((2, 2), dtype=np.complex64)
    first, second = _write_pair(
        tmp_path, reference, np.full((2, 2), 1.0j, dtype=np.complex64)
    )
    third = _write_secondary(
        tmp_path / "third",
        first,
        np.full((2, 2), -1.0, dtype=np.complex64),
        date_id="20240125",
    )
    product = form_merged_scene_interferogram(
        CoregisteredSceneStore.open(second),
        CoregisteredSceneStore.open(third),
    )
    np.testing.assert_array_equal(
        product.complex_ifg, np.full((2, 2), -1.0j, dtype=np.complex64)
    )


def test_ifg_leaves_a_unit_missing_from_one_date_blank(tmp_path: Path) -> None:
    """Units are formed on the common tags; an uncovered burst stays blank."""
    reference = np.ones((2, 4), dtype=np.complex64)
    tags = [("f0_IW1_b0", 0, 0), ("f0_IW1_b1", 2, 0)]
    first, second = _write_pair(
        tmp_path,
        reference,
        np.full((2, 4), 1.0j, dtype=np.complex64),
        tag=tags,
        grid_shape=(4, 4),
    )
    partial = _write_secondary(
        tmp_path / "partial",
        first,
        np.full((2, 4), -1.0j, dtype=np.complex64),
        date_id="20240125",
        tags=[("f0_IW1_b0", 0, 0)],
        grid_shape=(4, 4),
    )
    product = form_merged_scene_interferogram(
        CoregisteredSceneStore.open(second),
        CoregisteredSceneStore.open(partial),
    )
    assert product.valid_mask[:2].all()
    assert not product.valid_mask[2:].any()
    np.testing.assert_array_equal(
        product.complex_ifg[:2], np.full((2, 4), -1.0, dtype=np.complex64)
    )
    by_tag = form_scene_interferograms(
        CoregisteredSceneStore.open(second), CoregisteredSceneStore.open(partial)
    )
    assert set(by_tag) == {"f0_IW1_b0"}


def test_pair_without_common_units_is_rejected(tmp_path: Path) -> None:
    """Two dates that share no unit cannot form an interferogram."""
    reference = np.ones((2, 4), dtype=np.complex64)
    tags = [("f0_IW1_b0", 0, 0), ("f0_IW1_b1", 2, 0)]
    first, _ = _write_pair(tmp_path, reference, reference, tag=tags, grid_shape=(4, 4))
    upper = _write_secondary(
        tmp_path / "upper",
        first,
        reference,
        date_id="20240125",
        tags=[tags[0]],
        grid_shape=(4, 4),
    )
    lower = _write_secondary(
        tmp_path / "lower",
        first,
        reference,
        date_id="20240206",
        tags=[tags[1]],
        grid_shape=(4, 4),
    )
    with pytest.raises(InvalidProcessingStateError, match="share no unit"):
        form_merged_scene_interferogram(
            CoregisteredSceneStore.open(upper), CoregisteredSceneStore.open(lower)
        )


def test_scene_product_publishes_zarr_assets_in_one_stac_item(tmp_path: Path) -> None:
    """A scene product is one item.json with Zarr assets and no sidecars."""
    data = np.ones((2, 3), dtype=np.complex64)
    _, root = _write_pair(tmp_path, data, data)

    assert sorted(path.name for path in root.iterdir()) == [
        "IW1_b0.slc.zarr",
        "item.json",
    ]
    item = json.loads((root / "item.json").read_text())
    assert item["type"] == "Feature"
    properties = item["properties"]
    assert properties["faninsar:kind"] == "scene"
    assert properties["faninsar:domain"] == "radar"
    assert properties["faninsar:role"] == "secondary"
    assert properties["faninsar:shape"] == [2, 3]
    assert len(properties["faninsar:grid_identity"]) == 64
    asset = item["assets"]["IW1_b0"]
    assert asset["href"] == "IW1_b0.slc.zarr"
    assert asset["faninsar:role"] == "slc"
    assert not list(root.rglob("*.npy"))
    assert CoregisteredSceneStore.open(root).domain == "radar"


def test_geo_scene_product_requires_and_preserves_projection_metadata(
    tmp_path: Path,
) -> None:
    """Geo scenes persist CRS/transform while radar scenes do not invent them."""
    data = np.ones((2, 3), dtype=np.complex64)
    _, root = _write_pair(
        tmp_path,
        data,
        data,
        domain="geo",
        grid_identity="a" * 64,
        crs="EPSG:4326",
        transform=(1, 0, 10, 0, -1, 20),
    )
    opened = CoregisteredSceneStore.open(root)
    assert opened.crs == "EPSG:4326"
    assert opened.transform == (1.0, 0.0, 10.0, 0.0, -1.0, 20.0)
    item = json.loads((root / "item.json").read_text())
    assert item["properties"]["proj:code"] == "EPSG:4326"


@pytest.mark.parametrize(
    ("violation", "match"),
    [
        ("shape", "shape"),
        ("dtype", "complex dtype"),
        ("mask_dtype", "bool dtype"),
        ("mask_subset", "cannot expand"),
        ("finite", "must be finite"),
    ],
)
def test_merged_scene_filter_rejects_invalid_custom_result(
    tmp_path: Path,
    violation: str,
    match: str,
) -> None:
    """The durable Stack boundary rejects malformed custom filter outputs."""
    primary = np.ones((3, 3), dtype=np.complex64)
    secondary = np.full((3, 3), 1.0 + 1.0j, dtype=np.complex64)
    primary[0, 0] = np.nan + 1j * np.nan
    secondary[0, 0] = np.nan + 1j * np.nan
    first, second = _write_pair(tmp_path, primary, secondary)

    with pytest.raises(InvalidProcessingStateError, match=match):
        form_merged_scene_interferogram(
            CoregisteredSceneStore.open(first),
            CoregisteredSceneStore.open(second),
            phase_filter=_InvalidPhaseFilter(violation),
        )


def _rewrite_item(root: Path, mutate: Callable[[dict], object]) -> None:
    """Apply ``mutate`` to a committed item.json in place."""
    item_path = root / "item.json"
    item = json.loads(item_path.read_text(encoding="utf-8"))
    mutate(item)
    item_path.write_text(json.dumps(item), encoding="utf-8")


def test_scene_product_rejects_foreign_kind(tmp_path: Path) -> None:
    """An item of another kind cannot be opened as a scene product."""
    data = np.ones((2, 2), dtype=np.complex64)
    _, root = _write_pair(tmp_path, data, data)
    _rewrite_item(
        root, lambda item: item["properties"].update({"faninsar:kind": "ifg"})
    )
    with pytest.raises(InvalidProcessingStateError, match="kind"):
        CoregisteredSceneStore.open(root)


def test_scene_product_rejects_role_that_contradicts_dates(tmp_path: Path) -> None:
    """A secondary product cannot masquerade as the Reference product."""
    data = np.ones((2, 2), dtype=np.complex64)
    _, root = _write_pair(tmp_path, data, data)
    _rewrite_item(
        root, lambda item: item["properties"].update({"faninsar:role": "reference"})
    )
    with pytest.raises(InvalidProcessingStateError, match="role"):
        CoregisteredSceneStore.open(root)


def test_scene_product_rejects_symlinked_root(tmp_path: Path) -> None:
    """A symlinked product root fails closed before metadata access."""
    data = np.ones((2, 2), dtype=np.complex64)
    _, real_root = _write_pair(tmp_path, data, data)
    link_root = tmp_path / "link"
    link_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(InvalidProcessingStateError):
        CoregisteredSceneStore.open(link_root)


def test_form_interferograms_supports_multiple_units(tmp_path: Path) -> None:
    """All complete burst units are formed without a single-unit shortcut."""
    reference = np.ones((2, 2), dtype=np.complex64)
    secondary = np.full((2, 2), 1.0 + 2.0j, dtype=np.complex64)
    first, second = _write_pair(
        tmp_path,
        reference,
        secondary,
        tag=[("IW1_b0", 0, 0), ("IW1_b1", 0, 0)],
    )
    output = form_scene_interferograms(
        CoregisteredSceneStore.open(first), CoregisteredSceneStore.open(second)
    )
    assert set(output) == {"IW1_b0", "IW1_b1"}


def test_form_interferograms_has_no_legacy_pair_call() -> None:
    """Stack formation must not import or invoke the Pair engine."""
    source = Path("faninsar/stack/session.py").read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "form_interferograms"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == "run_pair"
        for node in ast.walk(method)
    )


def test_merged_scene_interferogram_uses_global_origins_before_looks(
    tmp_path: Path,
) -> None:
    """Overlapping bursts are merged in complex space on the common grid."""
    reference = np.ones((2, 4), dtype=np.complex64)
    upper_secondary = np.full((2, 4), 1.0j, dtype=np.complex64)
    lower_secondary = np.full((2, 4), -1.0j, dtype=np.complex64)
    reference_store = write_scene(
        tmp_path / "reference",
        date_id=REFERENCE_ID,
        reference_id=REFERENCE_ID,
        domain="radar",
        units=_units(reference, [("f0_IW1_b0", 0, 0), ("f0_IW1_b1", 2, 0)]),
        grid_shape=(4, 4),
    )
    secondary_store = write_scene(
        tmp_path / "secondary",
        date_id=SECONDARY_ID,
        reference_id=REFERENCE_ID,
        domain="radar",
        units=[
            *_units(upper_secondary, [("f0_IW1_b0", 0, 0)]),
            *_units(lower_secondary, [("f0_IW1_b1", 2, 0)]),
        ],
        grid_shape=(4, 4),
        inputs={"reference": reference_store.write_id},
    )

    product = form_merged_scene_interferogram(
        reference_store,
        secondary_store,
        multilook=(2, 2),
    )

    expected = np.vstack(
        (
            np.full((1, 2), -1.0j, dtype=np.complex64),
            np.full((1, 2), 1.0j, dtype=np.complex64),
        )
    )
    np.testing.assert_array_equal(product.complex_ifg, expected)
    np.testing.assert_array_equal(product.coherence, np.ones((2, 2), np.float32))


def test_merged_scene_interferogram_rejects_mixed_grid_placement(
    tmp_path: Path,
) -> None:
    """The same burst tag cannot move between acquisition products."""
    data = np.ones((2, 2), dtype=np.complex64)
    first, _ = _write_pair(tmp_path, data, data, tag="f0_IW1_b0", grid_shape=(4, 2))
    moved = _write_secondary(
        tmp_path / "moved",
        first,
        data,
        date_id=SECONDARY_ID,
        tags=[("f0_IW1_b0", 1, 0)],
        grid_shape=(4, 2),
    )
    reference_store = CoregisteredSceneStore.open(first)
    moved_store = CoregisteredSceneStore.open(moved)
    with pytest.raises(InvalidProcessingStateError, match="placement"):
        form_merged_scene_interferogram(reference_store, moved_store)
    with pytest.raises(InvalidProcessingStateError, match="placement"):
        validate_reference_binding(reference_store, moved_store)


def test_merged_scene_interferogram_rejects_equal_shape_shifted_grid(
    tmp_path: Path,
) -> None:
    """Equal array shapes cannot hide different coordinate-grid identities."""
    data = np.ones((2, 2), dtype=np.complex64)
    geo = {"domain": "geo", "crs": "EPSG:4326", "transform": (1, 0, 0, 0, -1, 0)}
    first, _ = _write_pair(
        tmp_path / "a", data, data, grid_shape=(2, 2), grid_identity="a" * 64, **geo
    )
    _, second = _write_pair(
        tmp_path / "b", data, data, grid_shape=(2, 2), grid_identity="b" * 64, **geo
    )
    with pytest.raises(InvalidProcessingStateError, match="coordinate grids"):
        form_merged_scene_interferogram(
            CoregisteredSceneStore.open(first),
            CoregisteredSceneStore.open(second),
        )


def test_scene_product_rejects_geo_item_without_grid_identity(
    tmp_path: Path,
) -> None:
    """A Geo item cannot substitute array shape for coordinates."""
    data = np.ones((2, 2), dtype=np.complex64)
    root, _ = _write_pair(
        tmp_path,
        data,
        data,
        domain="geo",
        grid_shape=(2, 2),
        grid_identity="a" * 64,
        crs="EPSG:4326",
        transform=(1, 0, 0, 0, -1, 0),
    )
    _rewrite_item(root, lambda item: item["properties"].pop("faninsar:grid_identity"))

    with pytest.raises(InvalidProcessingStateError, match="grid"):
        CoregisteredSceneStore.open(root)


def test_scene_writer_requires_explicit_geo_grid_identity(tmp_path: Path) -> None:
    """Geo publication cannot derive coordinates from array shape alone."""
    with pytest.raises(InvalidProcessingStateError, match="explicit grid_identity"):
        SceneWriter(
            tmp_path / "geo",
            date_id=REFERENCE_ID,
            reference_id=REFERENCE_ID,
            domain="geo",
            grid_shape=(2, 2),
            crs="EPSG:4326",
            transform=(1, 0, 0, 0, -1, 0),
        )
    assert not (tmp_path / "geo").exists()


@pytest.mark.parametrize(
    ("payload", "row_origin", "phase_screen", "match"),
    [
        (np.ones((2, 2), dtype=np.complex128), 0, None, "complex64"),
        (np.ones(4, dtype=np.complex64), 0, None, "complex64"),
        (np.ones((2, 2), dtype=np.complex64), 3, None, "outside"),
        (
            np.ones((2, 2), dtype=np.complex64),
            0,
            np.full((2, 2), np.nan, dtype=np.float32),
            "phase screen",
        ),
    ],
)
def test_scene_writer_rejects_invalid_units(
    tmp_path: Path,
    payload: np.ndarray,
    row_origin: int,
    phase_screen: np.ndarray | None,
    match: str,
) -> None:
    """Dtype, rank, placement, and phase screen are validated per unit."""
    with SceneWriter(
        tmp_path / "scene",
        date_id=REFERENCE_ID,
        reference_id=REFERENCE_ID,
        domain="radar",
        grid_shape=(4, 2),
    ) as writer:
        with pytest.raises(InvalidProcessingStateError, match=match):
            writer.add_unit(
                tag="f0_IW1_b0",
                payload=payload,
                row_origin=row_origin,
                col_origin=0,
                phase_screen=phase_screen,
            )
        writer.abort()
    assert not (tmp_path / "scene").exists()


def test_scene_writer_rejects_mixed_flattening_stages(tmp_path: Path) -> None:
    """All units of one product must share one flattening stage."""
    data = np.ones((2, 2), dtype=np.complex64)
    writer = SceneWriter(
        tmp_path / "scene",
        date_id=REFERENCE_ID,
        reference_id=REFERENCE_ID,
        domain="radar",
        grid_shape=(4, 2),
    )
    writer.add_unit(
        tag="f0_IW1_b0",
        payload=data,
        row_origin=0,
        col_origin=0,
        phase_state={"flatten_stage": "coregistration"},
    )
    writer.add_unit(
        tag="f0_IW1_b1",
        payload=data,
        row_origin=2,
        col_origin=0,
        phase_state={"flatten_stage": "interferogram"},
    )
    with pytest.raises(InvalidProcessingStateError, match="flattening"):
        writer.commit()
    assert not (tmp_path / "scene").exists()


def _write_then_crash(root: Path) -> None:
    with SceneWriter(
        root,
        date_id=REFERENCE_ID,
        reference_id=REFERENCE_ID,
        domain="radar",
        grid_shape=(2, 2),
    ) as writer:
        writer.add_unit(
            tag="f0_IW1_b0",
            payload=np.ones((2, 2), dtype=np.complex64),
            row_origin=0,
            col_origin=0,
        )
        message = "boom"
        raise RuntimeError(message)


def test_crash_inside_scene_writer_leaves_no_product_or_staging(
    tmp_path: Path,
) -> None:
    """An exception before commit publishes nothing and cleans its staging."""
    root = tmp_path / "coreg" / SECONDARY_ID
    root.parent.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="boom"):
        _write_then_crash(root)
    assert list(root.parent.iterdir()) == []


def test_existing_product_requires_overwrite(tmp_path: Path) -> None:
    """A committed product is never replaced without ``overwrite=True``."""
    data = np.ones((2, 2), dtype=np.complex64)
    first, _ = _write_pair(tmp_path, data, data)
    with pytest.raises(InvalidProcessingStateError, match="already exists"):
        write_scene(
            first,
            date_id=REFERENCE_ID,
            reference_id=REFERENCE_ID,
            domain="radar",
            units=_units(data, [("IW1_b0", 0, 0)]),
        )


def test_merged_scene_interferogram_rejects_different_references(
    tmp_path: Path,
) -> None:
    """Equal grids cannot hide products aligned to different References."""
    data = np.ones((2, 2), dtype=np.complex64)
    first, _ = _write_pair(tmp_path / "a", data, data, grid_shape=(2, 2))
    _, second = _write_pair(
        tmp_path / "b", data, data, reference_id="20231220", grid_shape=(2, 2)
    )

    with pytest.raises(InvalidProcessingStateError, match="References"):
        form_merged_scene_interferogram(
            CoregisteredSceneStore.open(first),
            CoregisteredSceneStore.open(second),
        )
