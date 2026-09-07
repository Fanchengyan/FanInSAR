"""Interferogram / InterferogramStack seam tests."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar import Pairs
from faninsar.core.physical import PhysicalType
from faninsar.network.products import Interferogram
from faninsar.processing.interferometry.contracts import InterferogramStack
from faninsar.processing.errors import InvalidProcessingStateError


def test_pair_id_law() -> None:
    ifg = Interferogram.from_pair_name(
        "20161207_20161231",
        samples=np.ones((4, 4), dtype=np.complex64),
    )
    assert ifg.primary_id == "20161207"
    assert ifg.secondary_id == "20161231"
    assert ifg.physical is PhysicalType.IFG_COMPLEX


def test_invalid_pair_id() -> None:
    with pytest.raises(InvalidProcessingStateError):
        Interferogram.from_pair_name("bad", samples=None)


def test_stack_alignment_and_unwrapped_matrix() -> None:
    names = ["20170101_20170113", "20170101_20170125", "20170113_20170125"]
    pairs = Pairs.from_names(names)
    unw = np.arange(3 * 10, dtype=np.float64).reshape(3, 10)
    stack = InterferogramStack.from_unwrapped("test", pairs, unw)
    assert len(stack.interferograms) == 3
    mat = stack.unwrapped_matrix()
    assert mat.shape == (3, 10)
    np.testing.assert_array_equal(mat, unw)


def test_stack_length_mismatch() -> None:
    pairs = Pairs.from_names(["20170101_20170113"])
    ifg = Interferogram.from_pair_name("20170101_20170113", samples=None)
    with pytest.raises(InvalidProcessingStateError):
        InterferogramStack(
            stack_id="x",
            pairs=pairs,
            interferograms=(ifg, ifg),  # wrong length
        )
