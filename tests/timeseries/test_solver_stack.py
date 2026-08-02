"""NSBASSolver accepts InterferogramStack only at the public seam."""

from __future__ import annotations

import numpy as np

from faninsar import Pairs
from faninsar.processing.contracts.ifg import InterferogramStack
from faninsar.timeseries.solver import NSBASSolver


def test_solver_from_stack() -> None:
    names = [
        "20170111_20170204",
        "20170111_20170222",
        "20170204_20170222",
    ]
    pairs = Pairs.from_names(names)
    # Simple unwrapped values: 3 pairs x 5 pixels
    rng = np.random.default_rng(0)
    unw = rng.normal(size=(len(pairs), 5)).astype(np.float64)
    stack = InterferogramStack.from_unwrapped("synth", pairs, unw)
    solver = NSBASSolver(stack, model=None, verbose=False, dtype=__import__("torch").float64)
    assert solver is not None
    # d shape should match
    assert solver._d.shape[0] >= len(pairs)  # SBAS rows may equal n_pairs


def test_solver_rejects_frame_like() -> None:
    class FakeFrame:
        pass

    try:
        NSBASSolver(FakeFrame())  # type: ignore[arg-type]
        raised = False
    except TypeError:
        raised = True
    assert raised
