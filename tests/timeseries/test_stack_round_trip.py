"""Synthetic InterferogramStack to time-series solver seam tests."""

from __future__ import annotations

import numpy as np

from faninsar import Pairs
from faninsar.processing.interferometry.contracts import InterferogramStack
from faninsar.timeseries.solver import NSBASSolver


def test_stack_round_trip_inverse() -> None:
    """Synthetic InterferogramStack inverts without legacy product coupling."""
    names = [
        "20170111_20170204",
        "20170111_20170222",
        "20170204_20170222",
    ]
    pairs = Pairs.from_names(names)
    rng = np.random.default_rng(1)
    unw = rng.normal(size=(len(pairs), 8)).astype(np.float64)
    stack = InterferogramStack.from_unwrapped("frame-synth", pairs, unw)
    assert isinstance(stack, InterferogramStack)
    solver = NSBASSolver(stack, model=None, verbose=False)
    result = solver.inverse(return_numpy=True)
    assert result is not None


def test_solver_source_has_no_frame_import() -> None:
    import inspect

    import faninsar.timeseries.solver as s

    src = inspect.getsource(s)
    # Ensure the solver has no dependency on the retired Frame façade.
    assert "from faninsar.datasets.frame" not in src
    assert "import Frame" not in src
