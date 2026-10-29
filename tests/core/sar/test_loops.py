"""Tests for array-backed Loops and the vectorized build_loops."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.core import Loop, Loops, Pairs
from faninsar.core import loops as loops_module

if TYPE_CHECKING:
    from collections.abc import Sequence


def make_network(n_dates: int, interval: int = 12, span: int = 4) -> Pairs:
    """Build a network pairing acquisitions within ``span`` intervals."""
    start = datetime(2022, 1, 1, tzinfo=UTC)
    dates = [start + timedelta(days=i * interval) for i in range(n_dates)]
    names = [
        dates[i].strftime("%Y%m%d") + "_" + dates[j].strftime("%Y%m%d")
        for i in range(n_dates)
        for j in range(i + 1, min(i + span + 1, n_dates))
    ]
    return Pairs.from_names(names)


def _date_index(pairs: Pairs) -> tuple[np.ndarray, dict]:
    """Return the sorted date array and a date-to-index mapping."""
    dates = np.asarray(pairs.dates, dtype="M8[D]")
    return dates, {d: i for i, d in enumerate(dates)}


def _dfs_paths(adjacency: dict, start: int, end: int, max_acq: int) -> list[list[int]]:
    """Enumerate all bounded-length paths from start to end (reference)."""
    paths: list[list[int]] = []

    def dfs(current: int, path: list[int]) -> None:
        for nxt in adjacency.get(current, ()):
            if nxt > end:
                break
            if nxt == end:
                paths.append([*path, end])
            elif len(path) + 1 < max_acq:
                dfs(nxt, [*path, nxt])

    dfs(start, [start])
    return paths


def reference_loop_names(pairs: Pairs, edge_days: int, max_acq: int = 5) -> list:
    """Independent DFS reference returning sorted loop names."""
    dates, index = _date_index(pairs)
    edge_mask = np.asarray(pairs.days <= edge_days)
    adjacency: dict = {}
    for a, b in zip(pairs.primary[edge_mask], pairs.secondary[edge_mask], strict=True):
        u = index[np.datetime64(a, "D")]
        v = index[np.datetime64(b, "D")]
        adjacency.setdefault(u, []).append(v)

    def to_name(indices: Sequence[int]) -> str:
        parts = [dates[i].astype(datetime).strftime("%Y%m%d") for i in indices]
        return "_".join(parts)

    names = []
    for a, b in zip(
        pairs.primary[~edge_mask], pairs.secondary[~edge_mask], strict=True
    ):
        start = index[np.datetime64(a, "D")]
        end = index[np.datetime64(b, "D")]
        names.extend(
            to_name(path) for path in _dfs_paths(adjacency, start, end, max_acq)
        )
    return sorted(names)


@pytest.mark.parametrize(
    ("n_dates", "span", "edge_days"),
    [(8, 4, 24), (12, 4, 36), (61, 4, 12), (61, 4, 24), (61, 4, 36), (40, 8, 36)],
)
def test_build_loops_matches_reference_dfs(
    n_dates: int, span: int, edge_days: int
) -> None:
    """build_loops must enumerate exactly the reference DFS loop set."""
    pairs = make_network(n_dates, span=span)
    loops = pairs.build_loops(edge_days=edge_days)
    expected = reference_loop_names(pairs, edge_days)
    assert sorted(loops.names.tolist()) == expected


def test_build_loops_requires_edge_selection() -> None:
    """Building loops needs either edge_days or edge_pairs."""
    pairs = make_network(6)
    with pytest.raises(ValueError, match="edge_days"):
        pairs.build_loops()


def test_loop_matrix_matches_loop_objects() -> None:
    """The scattered matrix must equal one built from the Loop objects."""
    pairs = make_network(25)
    loops = pairs.build_loops(edge_days=24)
    matrix = loops.loop_matrix()
    all_names = loops.pairs.names
    brute = np.zeros_like(matrix)
    for i in range(len(loops)):
        loop_pairs = loops[i].pairs.names
        brute[i][np.isin(all_names, loop_pairs[:-1])] = 1
        brute[i][np.isin(all_names, loop_pairs[-1:])] = -1
    np.testing.assert_array_equal(matrix, brute)


def test_pairs_metadata_consistency() -> None:
    """Edge and diagonal pairs partition all pairs with correct day spans."""
    pairs = make_network(30)
    loops = pairs.build_loops(edge_days=36)
    all_names = set(loops.pairs.names)
    edge_names = set(loops.edge_pairs.names)
    diag_names = set(loops.diagonal_pairs.names)
    assert edge_names | diag_names == all_names
    assert not edge_names & diag_names
    assert loops.edge_pairs.days.max() <= 36
    assert loops.diagonal_pairs.days.min() > 36


def test_loop_length_bounds() -> None:
    """Loops have 3..max_acquisition acquisitions."""
    pairs = make_network(30)
    loops = pairs.build_loops(edge_days=24, max_acquisition=5)
    lengths = [len(loop.values) for loop in loops]
    assert min(lengths) >= 3
    assert max(lengths) <= 5
    assert len(set(lengths)) > 1  # variable-length padding is exercised

    loops4 = pairs.build_loops(edge_days=24, max_acquisition=4)
    assert max(len(loop.values) for loop in loops4) <= 4
    assert len(loops4) < len(loops)


def test_lazy_loop_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Loop object is built until one is accessed."""
    pairs = make_network(20)
    built = {"count": 0}

    class CountingLoop(Loop):
        def __init__(self, loop: Sequence[datetime]) -> None:
            built["count"] += 1
            super().__init__(loop)

    monkeypatch.setattr(loops_module, "Loop", CountingLoop)
    loops = pairs.build_loops(edge_days=24)
    assert built["count"] == 0
    assert len(loops) > 0
    first = loops[3]
    assert built["count"] == 1
    assert first.name == loops.names[3]
    total = len(loops)
    assert len(list(loops)) == total
    assert built["count"] == 1 + total


def test_list_constructor_matches_fast_path() -> None:
    """Loops(list-of-Loop) must equal the array-backed construction."""
    pairs = make_network(20)
    loops = pairs.build_loops(edge_days=24)
    rebuilt = Loops(list(loops))
    assert rebuilt.names.tolist() == loops.names.tolist()
    np.testing.assert_array_equal(rebuilt.loop_matrix(), loops.loop_matrix())
    assert rebuilt.shape == loops.shape


def test_sort_and_contains() -> None:
    """Sorting reorders metadata and lookups work by name."""
    pairs = make_network(20)
    loops = pairs.build_loops(edge_days=24)
    assert loops.names.tolist() == sorted(loops.names.tolist())

    descending = loops.sort(ascending=False, inplace=False)
    assert isinstance(descending, Loops)
    assert descending.names.tolist() == sorted(loops.names.tolist(), reverse=True)

    loops.sort(ascending=False)
    assert loops.names.tolist() == sorted(loops.names.tolist(), reverse=True)
    assert loops[0] in loops


def test_empty_loops() -> None:
    """A network without diagonal pairs yields an empty Loops."""
    pairs = make_network(6)
    loops = pairs.build_loops(edge_days=400)
    assert len(loops) == 0
    assert loops.loop_matrix().shape == (0, 0)
    assert list(loops) == []
    assert loops.shape == (0, 0)


def test_edge_pairs_argument_matches_edge_days() -> None:
    """Selecting edges by Pairs or by days gives identical loops."""
    pairs = make_network(25)
    edges = pairs[pairs.days <= 24]
    loops_via_pairs = pairs.build_loops(edge_pairs=edges)
    loops_via_days = pairs.build_loops(edge_days=24)
    assert loops_via_pairs.names.tolist() == loops_via_days.names.tolist()


def test_max_days_filters_diagonals() -> None:
    """max_days bounds the diagonal pairs considered."""
    pairs = make_network(40, span=4)
    loops = pairs.build_loops(edge_days=24, max_days=24)
    assert len(loops) == 0  # no diagonal pair survives max_days == edge_days
    loops2 = pairs.build_loops(edge_days=24, max_days=36)
    assert len(loops2) > 0
    assert loops2.diagonal_pairs.days.max() <= 36


def test_calculate_u_integration() -> None:
    """The calculate_u consumer works on the array-backed loops."""
    from faninsar.timeseries import calculate_u

    pairs = make_network(15)
    loops = pairs.build_loops(edge_days=24)
    unw = np.arange(2 * len(pairs), dtype=np.float64).reshape(len(pairs), 2)
    u = calculate_u(loops, unw, device="cpu")
    assert u.shape == (len(pairs), 2)
    raw_closure = np.abs(loops.loop_matrix() @ unw).max()
    corrected = np.abs(loops.loop_matrix() @ (unw - 2 * np.pi * u)).max()
    assert corrected <= raw_closure
