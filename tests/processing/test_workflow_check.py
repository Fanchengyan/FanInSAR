"""Workflow lattice check tests."""

from __future__ import annotations

import pytest

from faninsar.core.physical import PhysicalType
from faninsar.processing.contracts.stage import StageNode
from faninsar.processing.errors import StageError
from faninsar.processing.workflow import Workflow, get_default_pair_stages


def _identity(state):
    return state


def test_build_accepts_compatible_seq() -> None:
    nodes = [
        StageNode(_identity, "a", None, PhysicalType.SLC_RAW),
        StageNode(_identity, "b", PhysicalType.SLC_RAW, PhysicalType.SLC_DERAMPED),
        StageNode(
            _identity, "c", PhysicalType.SLC_DERAMPED, PhysicalType.SLC_COREG
        ),
    ]
    wf = Workflow.from_stages(nodes).build()
    assert wf._built


def test_build_rejects_type_mismatch() -> None:
    nodes = [
        StageNode(_identity, "a", None, PhysicalType.SLC_RAW),
        StageNode(
            _identity, "bad", PhysicalType.IFG_COMPLEX, PhysicalType.IFG_FLATTENED
        ),
    ]
    with pytest.raises(StageError):
        Workflow.from_stages(nodes).build()


def test_default_pair_stages_are_callable() -> None:
    stages = get_default_pair_stages()
    assert all(callable(s) for s in stages)


def test_eager_run_without_build() -> None:
    def add_one(x):
        return x + 1

    wf = Workflow.from_stages([add_one, add_one])
    assert wf.run(0) == 2
