"""Stage callables must be importable and typed for Workflow wiring."""

from __future__ import annotations

from faninsar.core.physical import PhysicalType
from faninsar.processing.stages import (
    stage_coreg,
    stage_deramp,
    stage_flatten,
    stage_geocode,
    stage_ifg,
    stage_read,
    stage_unwrap,
    stage_write,
)
from faninsar.processing.workflow import Workflow


def test_stage_callables_have_physical_types() -> None:
    chain = (
        (stage_read, stage_deramp),
        (stage_deramp, stage_coreg),
        (stage_coreg, stage_ifg),
        (stage_ifg, stage_flatten),
        (stage_flatten, stage_unwrap),
        (stage_unwrap, stage_geocode),
        (stage_geocode, stage_write),
    )
    for producer, consumer in chain:
        out = getattr(producer, "output_type", None)
        inp = getattr(consumer, "input_type", None)
        assert out is not None or producer is stage_read
        if out is not None and inp is not None:
            assert out == inp or (
                out == PhysicalType.PHASE_UNWRAPPED
                and inp == PhysicalType.PHASE_UNWRAPPED
            )


def test_workflow_build_accepts_stage_callables() -> None:
    stages = (
        stage_read,
        stage_deramp,
        stage_coreg,
        stage_ifg,
        stage_flatten,
        stage_unwrap,
        stage_write,
    )
    # build may only check lattice connectivity
    wf = Workflow.from_stages(stages)
    built = wf.build()
    assert built is not None
