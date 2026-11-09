"""Stack ion correction, unwrap source, and inversion phase (PROPOSAL-0080)."""

from __future__ import annotations

from tests.support.local_stack import LocalStack

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from faninsar import Pairs
from faninsar.io.storage.ifg_store import write_ifg_artifact, write_unwrapped_artifact
from faninsar.io.storage.ion_store import ion_dir, write_ionosphere_artifact
from faninsar.io.storage.product import open_product
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.unwrapping.common import SpatialUnwrapper, SpatialUnwrapResult
from faninsar.stack import Stack

if TYPE_CHECKING:
    from pathlib import Path

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_PAIR = ("20240101", "20240113")
_PAIR_ID = "20240101_20240113"


class _IdentityUnwrapper(SpatialUnwrapper):
    """Return supported wrapped phase as a successful spatial result."""

    def unwrap(
        self,
        wrapped_phase: torch.Tensor,
        *,
        coherence: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> SpatialUnwrapResult:
        del coherence
        support = torch.isfinite(wrapped_phase)
        if valid_mask is not None:
            support &= valid_mask
        labels = torch.where(
            support,
            torch.zeros_like(wrapped_phase, dtype=torch.int64),
            torch.full_like(wrapped_phase, -1, dtype=torch.int64),
        )
        return SpatialUnwrapResult(
            phase=torch.where(support, wrapped_phase, torch.nan),
            valid_mask=support,
            component_labels=labels,
            reference_values=torch.zeros(1, device=wrapped_phase.device),
            converged=True,
            iterations=0,
            pcg_iterations=0,
            residual_norm=0.0,
            failure_reason=None,
        )


def _stack_with_ifg(
    tmp_path: Path,
    *,
    looks: tuple[int, int] = (1, 1),
    shape: tuple[int, int] = (2, 2),
    phase: float = 0.3,
    amplitude: float = 2.0,
) -> Stack:
    dates = _PAIR
    safe_paths = []
    for date in dates:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date}T000000.SAFE"
        path.mkdir()
        safe_paths.append(path)
    stack = LocalStack.from_paths(
        safe_paths,
        work_dir=tmp_path / "out",
        pairs=Pairs.from_names([_PAIR_ID]),
        multilook=looks,
    )
    stack.prepare_data()
    complex_ifg = np.full(shape, amplitude * np.exp(1j * phase), dtype=np.complex64)
    write_ifg_artifact(
        stack.config.work_dir / "ifg" / f"ml_{looks[0]}x{looks[1]}" / _PAIR_ID,
        pair=dates,
        looks=looks,
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"primary": _DIGEST_A, "secondary": _DIGEST_B},
        complex_ifg=complex_ifg,
        coherence=np.ones(shape, dtype=np.float32),
        valid_mask=np.ones(shape, dtype=bool),
        grid_identity=_DIGEST_A,
    )
    return stack


def _write_ion(
    stack: Stack,
    *,
    looks: tuple[int, int],
    shape: tuple[int, int],
    screen: float = 0.1,
    native_origin: tuple[int, int] = (0, 0),
) -> None:
    write_ionosphere_artifact(
        ion_dir(stack.config.work_dir, _PAIR_ID, looks),
        pair=_PAIR,
        looks=looks,
        domain="radar",
        wavelength_m=0.236,
        grid_identity=_DIGEST_A,
        method_name="ionosphere_split_spectrum",
        method_parameters={"looks": list(looks)},
        degraded=False,
        degradation_reason=None,
        runtime_fingerprint=_DIGEST_A,
        devices=("cpu",),
        source_manifest_digests={"primary": _DIGEST_A, "secondary": _DIGEST_B},
        ionosphere_phase=np.full(shape, screen, dtype=np.float32),
        nondispersive_phase=np.zeros(shape, dtype=np.float32),
        weight=np.ones(shape, dtype=np.float32),
        native_origin=native_origin,
    )


def test_apply_ionosphere_correction_writes_complex_ion_without_unwrap(
    tmp_path: Path,
) -> None:
    """Correction maps ion onto the IFG grid and commits complex_ion only."""
    stack = _stack_with_ifg(tmp_path, phase=0.3, amplitude=2.0)
    _write_ion(stack, looks=(1, 1), shape=(2, 2), screen=0.1)
    published = stack.apply_ionosphere_correction(ion_multilook=(1, 1))
    pair_root = stack.config.work_dir / "ifg" / "ml_1x1" / _PAIR_ID
    assert published[_PAIR_ID] == pair_root / "complex_ion"
    assert (pair_root / "complex_ion" / "complex.zarr").exists()
    assert not (pair_root / "unwrap").exists()
    corrected = stack.interferograms.open(_PAIR_ID, "complex_ion").values
    np.testing.assert_allclose(np.abs(corrected), 2.0, atol=1e-5)
    np.testing.assert_allclose(np.angle(corrected), 0.2, atol=1e-5)


def test_one_ion_estimate_corrects_two_target_looks(tmp_path: Path) -> None:
    """A native ion product is reused across compatible consumer multilooks."""
    stack = _stack_with_ifg(tmp_path, looks=(1, 1), shape=(4, 4), phase=0.0, amplitude=1.0)
    write_ifg_artifact(
        stack.config.work_dir / "ifg" / "ml_2x2" / _PAIR_ID,
        pair=_PAIR,
        looks=(2, 2),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"primary": _DIGEST_A, "secondary": _DIGEST_B},
        complex_ifg=np.ones((2, 2), dtype=np.complex64),
        coherence=np.ones((2, 2), dtype=np.float32),
        valid_mask=np.ones((2, 2), dtype=bool),
        grid_identity=_DIGEST_A,
    )
    _write_ion(stack, looks=(2, 2), shape=(2, 2), screen=0.5, native_origin=(0, 0))
    first = stack.apply_ionosphere_correction(ion_multilook=(2, 2), multilook=(1, 1))
    second = stack.apply_ionosphere_correction(ion_multilook=(2, 2), multilook=(2, 2))
    assert first[_PAIR_ID].parent.name == _PAIR_ID
    assert second[_PAIR_ID].parent.name == _PAIR_ID
    assert first[_PAIR_ID] != second[_PAIR_ID]
    ion_root = ion_dir(stack.config.work_dir, _PAIR_ID, (2, 2))
    assert open_product(first[_PAIR_ID]).inputs["ion"] == open_product(ion_root).write_id
    assert open_product(second[_PAIR_ID]).inputs["ion"] == open_product(ion_root).write_id


def test_unwrap_source_complex_ion_does_not_require_ordinary_unwrap(
    tmp_path: Path,
) -> None:
    """Corrected unwrapping consumes complex_ion and writes unwrap_ion."""
    stack = _stack_with_ifg(tmp_path, phase=0.3)
    _write_ion(stack, looks=(1, 1), shape=(2, 2), screen=0.3)
    stack.apply_ionosphere_correction(ion_multilook=(1, 1))
    stack.unwrap(_IdentityUnwrapper(), source="complex_ion")
    pair_root = stack.config.work_dir / "ifg" / "ml_1x1" / _PAIR_ID
    assert (pair_root / "unwrap_ion" / "item.json").is_file()
    assert not (pair_root / "unwrap" / "item.json").is_file()
    phase = stack.interferograms.open_stack("unwrapped_ion").values[0]
    np.testing.assert_allclose(phase, 0.0, atol=1e-5)


def test_invert_phase_selector_records_consumed_branch(tmp_path: Path) -> None:
    """Inversion phase= selects unwrap_ion and records its write ids."""
    stack = _stack_with_ifg(tmp_path, phase=0.4)
    _write_ion(stack, looks=(1, 1), shape=(2, 2), screen=0.1)
    stack.apply_ionosphere_correction(ion_multilook=(1, 1))
    stack.unwrap(_IdentityUnwrapper(), source="complex")
    stack.unwrap(_IdentityUnwrapper(), source="complex_ion")
    ordinary = stack.invert_timeseries(phase="unwrapped")
    corrected = stack.invert_timeseries(phase="unwrapped_ion", overwrite=True)
    item = open_product(stack.config.work_dir / "timeseries")
    assert item.properties["faninsar:phase_source"] == "unwrapped_ion"
    unwrap_ion = open_product(
        stack.config.work_dir / "ifg" / "ml_1x1" / _PAIR_ID / "unwrap_ion"
    )
    assert item.inputs[_PAIR_ID] == unwrap_ion.write_id
    network_result = stack.network.invert(phase="unwrapped_ion")
    assert open_product(stack.config.work_dir / "timeseries").inputs[
        _PAIR_ID
    ] == unwrap_ion.write_id
    assert not np.allclose(
        np.nan_to_num(ordinary.phase_cumulative_rad),
        np.nan_to_num(corrected.phase_cumulative_rad),
    )
    np.testing.assert_allclose(
        np.nan_to_num(network_result.phase_cumulative_rad),
        np.nan_to_num(corrected.phase_cumulative_rad),
    )


def test_named_branch_does_not_fall_back_to_ordinary(tmp_path: Path) -> None:
    """Missing or stale corrected products raise instead of switching branches."""
    stack = _stack_with_ifg(tmp_path)
    with pytest.raises(InvalidProcessingStateError, match="complex_ion"):
        stack.unwrap(_IdentityUnwrapper(), source="complex_ion")
    stack.unwrap(_IdentityUnwrapper())
    with pytest.raises(InvalidProcessingStateError, match="unwrapped_ion"):
        stack.invert_timeseries(phase="unwrapped_ion")
    with pytest.raises(TypeError, match="ionosphere_corrected"):
        stack.invert_timeseries(ionosphere_corrected=True)
    with pytest.raises(TypeError, match="ionosphere_corrected"):
        stack.network.invert(ionosphere_corrected=True)


def test_status_reports_pair_owned_ion_branches(tmp_path: Path) -> None:
    """status() discovers nested unwrap/ion siblings, not top-level trees."""
    stack = _stack_with_ifg(tmp_path)
    _write_ion(stack, looks=(1, 1), shape=(2, 2))
    stack.apply_ionosphere_correction(ion_multilook=(1, 1))
    stack.unwrap(_IdentityUnwrapper(), source="complex_ion")
    report = stack.status()
    assert _PAIR_ID in report["complex_ion"]["committed"]
    assert _PAIR_ID in report["unwrap_ion"]["committed"]
    assert _PAIR_ID in report["ionosphere"]["committed"]
    assert _PAIR_ID in report["unwrap"]["missing"]
