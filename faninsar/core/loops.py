"""Pair and loop tools for InSAR time series analysis."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import pandas as pd

from faninsar.core.dates import DateManager
from faninsar.core.pair import Pair, Pairs
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

    from faninsar.core.types import TripletLoopLike

logger = setup_logger(__name__)


class TripletLoop:
    """TripletLoop class containing three pairs/acquisitions."""

    _values: np.ndarray
    _name: str
    _pairs: list[Pair]

    __slots__ = ["_days12", "_days13", "_days23", "_name", "_pairs", "_values"]

    def __init__(self, loop: Sequence[datetime, datetime, datetime]) -> None:
        """Initialize the TripletLoop class.

        Parameters
        ----------
        loop: Sequence
            Sequence object of three dates. Each date is a datetime object.
            For example, (date1, date2, date3).

        """
        self._values = np.sort(loop).astype("M8[D]")
        loop_dt = self._values.astype(datetime)
        self._name = "_".join([i.strftime("%Y%m%d") for i in loop_dt])
        self._pairs = [
            Pair([loop_dt[0], loop_dt[1]]),
            Pair([loop_dt[1], loop_dt[2]]),
            Pair([loop_dt[0], loop_dt[2]]),
        ]

    def __str__(self) -> str:
        """Return the name of the loop."""
        return self._name

    def __repr__(self) -> str:
        """Return the representation of the loop."""
        return f"TripletLoop({self._name})"

    def __eq__(self, other: TripletLoop) -> bool:
        """Compare whether the loop is same to another loop."""
        return self.name == other.name

    def __hash__(self) -> int:
        """Return the hash of the loop."""
        return hash(self.name)

    def __array__(self, dtype: DTypeLike = None) -> NDArray:
        """Return the loop as a numpy array."""
        return np.asarray(self._values, dtype=dtype)

    @property
    def values(self) -> NDArray[np.datetime64]:
        """Three dates of the loop with format of np.datetime64[D]."""
        return self._values

    @property
    def pairs(self) -> list[Pair]:
        """All three pairs of the loop."""
        return self._pairs

    @property
    def days12(self) -> int:
        """Return the time span of the first pair in days.

        Returns
        -------
        days12: int
            Time span of the first pair in days.

        """
        return (self._values[1] - self._values[0]).astype(int)

    @property
    def days23(self) -> int:
        """Return the time span of the second pair in days.

        Returns
        -------
        days23: int
            Time span of the second pair in days.

        """
        return (self._values[2] - self._values[1]).astype(int)

    @property
    def days13(self) -> int:
        """Return the time span of the third pair in days.

        Returns
        -------
        days13: int
            Time span of the third pair in days.

        """
        return (self._values[2] - self._values[0]).astype(int)

    @property
    def name(self) -> str:
        """Return the string format the loop.

        Returns
        -------
        name: str
            String of the loop.

        """
        return self._name

    @classmethod
    def from_name(
        cls,
        name: str,
        parse_function: Callable | None = None,
        date_args: dict | None = None,
    ) -> TripletLoop:
        """Initialize the loop class from a loop name.

        Parameters
        ----------
        name: str
            TripletLoop name.
        parse_function: Callable, optional
            Function to parse the date strings from the loop name.
            If None, the loop name will be split by '_' and the
            last 3 items will be used. Default is None.
        date_args: dict, optional
            Keyword arguments for pd.to_datetime() to convert the
            date strings to datetime objects. For example,
            {'format': '%Y%m%d'}. Default is {}.

        Returns
        -------
        loop: TripletLoop
            TripletLoop object.

        """
        if date_args is None:
            date_args = {}
        dates = DateManager.str_to_dates(name, 3, parse_function, date_args)
        return cls(dates)

    def to_numpy(self, dtype: DTypeLike = None) -> NDArray:
        """Return the loop as a numpy array."""
        return np.asarray(self._values, dtype=dtype)


class TripletLoops:
    """TripletLoops class to handle loops with three pairs/acquisitions."""

    _values: np.ndarray
    _dates: np.ndarray
    _length: int

    __slots__ = ["_dates", "_length", "_values"]

    def __init__(
        self,
        loops: Sequence[Sequence[datetime, datetime, datetime]] | Sequence[TripletLoop],
        sort: bool = True,
    ) -> None:
        """Initialize the triplet loops class.

        Parameters
        ----------
        loops: Sequence
            Sequence object of triplet loops. Each loop is an Sequence object
            of three dates with format of datetime or TripletLoop object.
            For example, [(date1, date2, date3), ...].
        sort: bool, optional
            Whether to sort the loops. Default is True.

        """
        if loops is None or len(loops) == 0:
            msg = "loops cannot be None."
            raise ValueError(msg)

        _values = np.array(loops, dtype="M8[D]")
        self._values = _values
        self._parse_loop_meta()

        if sort:
            self.sort(inplace=True)

    def _parse_loop_meta(self) -> None:
        self._dates = np.unique(self._values)
        self._length = self._values.shape[0]
        self._names = self.to_names()

    def __str__(self) -> str:
        """Return the string representation of the loops."""
        return f"TripletLoops({self._length})"

    def __repr__(self) -> str:
        """Return the representation of the loops."""
        return self.to_dataframe("dates").__repr__()

    def __len__(self) -> int:
        """Return the number of loops."""
        return self._length

    def __eq__(self, other: TripletLoops) -> bool:
        """Compare whether the loops are same to another loops."""
        return np.array_equal(self.values, other.values)

    def __add__(self, other: TripletLoops) -> TripletLoops:
        """Return the union of the loops."""
        _loops = np.union1d(self.names, other.names)
        return TripletLoops.from_names(_loops)

    def __sub__(self, other: TripletLoops) -> TripletLoops | None:
        """Return the difference of the loops."""
        _loops = np.setdiff1d(self.names, other.names)
        if len(_loops) > 0:
            return TripletLoops.from_names(_loops)
        return None

    @overload
    def __getitem__(self, index: int) -> TripletLoop: ...

    @overload
    def __getitem__(self, index: slice) -> TripletLoops: ...

    def __getitem__(  # noqa: PLR0911
        self,
        index: int | slice | datetime | Iterable[datetime],
    ) -> TripletLoop | TripletLoops:
        """Return the loop or loops by index or slice."""
        if isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            if isinstance(start, (int, np.integer, type(None))) and isinstance(
                stop,
                (int, np.integer, type(None)),
            ):
                if start is None:
                    start = 0
                if stop is None:
                    stop = self._length
                return TripletLoops(self._values[start:stop:step])
            if isinstance(
                start,
                (datetime, np.datetime64, pd.Timestamp, str, type(None)),
            ) and isinstance(
                stop,
                (datetime, np.datetime64, pd.Timestamp, str, type(None)),
            ):
                if isinstance(start, str):
                    start = DateManager.ensure_datetime(start)
                if isinstance(stop, str):
                    stop = DateManager.ensure_datetime(stop)
                if start is None:
                    start = self._dates[0]
                if stop is None:
                    stop = self._dates[-1]

                start, stop = (np.datetime64(start, "s"), np.datetime64(stop, "s"))

                if start > stop:
                    msg = (
                        f"Index start {start} should be earlier than index stop {stop}."
                    )
                    raise ValueError(
                        msg,
                    )
                _loops = []
                for loop in self._values:
                    loop = loop.astype("M8[s]")  # noqa: PLW2901
                    if np.all((start <= loop) & (loop <= stop)):
                        _loops.append(loop)
                if len(_loops) > 0:
                    return TripletLoops(_loops)
                return None
            return None
        if isinstance(index, (int, np.integer)):
            if index >= self._length:
                msg = (
                    f"Index {index} out of range. TripletLoops number "
                    f"is {self._length}."
                )
                raise IndexError(msg)
            return TripletLoop(self._values[index])
        if isinstance(index, (datetime, np.datetime64, pd.Timestamp, str)):
            if isinstance(index, str):
                try:
                    index = pd.to_datetime(index)
                except Exception as e:
                    msg = f"String {index} cannot be converted to datetime."
                    raise ValueError(msg) from e
            loops = [loop for loop in self._values if index in loop]
            if len(loops) > 0:
                return TripletLoops(loops)
            return None
        if isinstance(index, Iterable):
            index = np.array(index)
            return TripletLoops(self._values[index])
        msg = (
            f"Index should be int, slice, datetime, str, or bool or int array"
            f"indexing, but got {type(index)}."
        )
        raise TypeError(msg)

    def __hash__(self) -> int:
        """Return the hash of the loops."""
        return hash("".join(self.names))

    def __iter__(self) -> iter[TripletLoop]:
        """Iterate the loops."""
        return iter(self.values)

    def __contains__(self, item: TripletLoopLike) -> bool:
        """Check if the item is in the loops."""
        if isinstance(item, TripletLoop):
            item = item.to_numpy()
        elif isinstance(item, str):
            item = TripletLoop.from_name(item).to_numpy()
        elif isinstance(item, Sequence):
            item = np.sort(item)
        else:
            msg = f"item should be TripletLoop, str, or Sequence, but got {type(item)}."
            raise TypeError(
                msg,
            )

        return np.any(np.all(item == self.values, axis=1))

    def __array__(self, dtype: DTypeLike = None) -> NDArray:
        """Return the loops as a numpy array."""
        return np.asarray(self._values, dtype=dtype)

    @property
    def values(self) -> NDArray[np.datetime64]:
        """Return the values of the loops.

        Returns
        -------
        values: np.ndarray
            Values of the loops with format of datetime.

        """
        return self._values

    @property
    def names(self) -> NDArray[np.str_]:
        """The names (sting format) of the loops."""
        return self._names

    @property
    def dates(self) -> NDArray[np.datetime64]:
        """Sorted dates of the loops with format of datetime."""
        return self._dates

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the loop array."""
        return self._values.shape

    @property
    def pairs(self) -> Pairs:
        """All sorted pairs of the loops."""
        pairs = np.unique(
            np.vstack(
                [self._values[:, :2], self._values[:, 1:], self._values[:, [0, 2]]],
            ),
            axis=0,
        )
        return Pairs(pairs, sort=False)

    @property
    def pairs12(self) -> Pairs:
        """The first pairs of the loops."""
        return Pairs(self._values[:, :2], sort=False)

    @property
    def pairs23(self) -> Pairs:
        """The second pairs of the loops."""
        return Pairs(self._values[:, 1:], sort=False)

    @property
    def pairs13(self) -> Pairs:
        """The third pairs of the loops."""
        return Pairs(self._values[:, [0, 2]], sort=False)

    @property
    def days12(self) -> NDArray[np.int64]:
        """The time span of the first pair in days."""
        return (self._values[:, 1] - self._values[:, 0]).astype(int)

    @property
    def days23(self) -> NDArray[np.int64]:
        """The time span of the second pair in days."""
        return (self._values[:, 2] - self._values[:, 1]).astype(int)

    @property
    def days13(self) -> NDArray[np.int64]:
        """The time span of the third pair in days."""
        return (self._values[:, 2] - self._values[:, 0]).astype(int)

    @property
    def index(self) -> NDArray[np.int64]:
        """The index of the loops in dates coordinates."""
        return np.searchsorted(self._dates, self._values)

    @classmethod
    def from_names(
        cls,
        names: list[str],
        parse_function: Callable | None = None,
        date_args: dict | None = None,
    ) -> TripletLoops:
        """Initialize the loops class from a list of loop file names.

        Parameters
        ----------
        names: list
            list of loop file names.
        parse_function: Callable, optional
            Function to parse the date strings from the loop file name.
            If None, the loop file name will be split by '_' and
            the last 3 items will be used. Default is None.
        date_args: dict, optional
            Keyword arguments for pd.to_datetime() to convert the date strings
            to datetime objects. For example, {'format': '%Y%m%d'}.
            Default is {}.

        Returns
        -------
        loops: TripletLoops
            unsorted TripletLoops object.

        """
        if date_args is None:
            date_args = {}
        loops = []
        for name in names:
            loop = TripletLoop.from_name(name, parse_function, date_args)
            loops.append(loop.values)
        return cls(loops, sort=False)

    def to_names(self, prefix: str | None = None) -> NDArray[np.str_]:
        """Return the string name of each loop.

        Parameters
        ----------
        prefix: str, optional
            Prefix of the output loop names. Default is None.

        Returns
        -------
        names: np.ndarray
            String names of the loops.

        """
        names = (
            pd.DatetimeIndex(self.values[:, 0]).strftime("%Y%m%d")
            + "_"
            + pd.DatetimeIndex(self.values[:, 1]).strftime("%Y%m%d")
            + "_"
            + pd.DatetimeIndex(self.values[:, 1]).strftime("%Y%m%d")
        )
        if prefix:
            names = prefix + "_" + names

        return names.to_numpy(dtype="S")

    def to_dataframe(self, target: Literal["pairs", "dates"] = "pairs") -> pd.DataFrame:
        """Return the loops as a DataFrame.

        Parameters
        ----------
        target: str, one of ['pairs', 'dates']
            Target of the DataFrame. Default is 'pairs'.

        """
        if target == "pairs":
            return pd.DataFrame(
                zip(
                    self.pairs12.values,
                    self.pairs23.values,
                    self.pairs13.values,
                    strict=False,
                ),
                columns=["pair12", "pair23", "pair13"],
            )
        if target == "dates":
            return pd.DataFrame(self.values, columns=["date1", "date2", "date3"])
        msg = f"target should be 'pairs' or 'dates', but got {target}."
        raise ValueError(msg)

    def loop_matrix(self, dtype: DTypeLike = None) -> NDArray[np.int8]:
        """Return loop matrix (containing 1, -1, 0) from pairs.

        Returns
        -------
        matrix: np.ndarray
            TripletLoop matrix with the shape of (n_loop, n_pair). The values of each
            loop/row in matrix are:

            - 1: pair12 and pair23
            - -1: pair13
            - 0: otherwise

        """
        n_loop = len(self)
        n_pair = len(self.pairs)
        matrix = np.zeros((n_loop, n_pair), dtype=dtype)
        pairs_ls = self.pairs.values.tolist()
        for i, loop in enumerate(self.values):
            matrix[i, pairs_ls.index(loop[:2].tolist())] = 1
            matrix[i, pairs_ls.index(loop[1:].tolist())] = 1
            matrix[i, pairs_ls.index(loop[[0, 2]].tolist())] = -1
        return matrix

    def where(
        self,
        loop: str | TripletLoop,
        return_type: Literal["index", "mask"] = "index",  # noqa: ARG002
    ) -> int | None:
        """Return the index of the loop.

        Parameters
        ----------
        loop: str or TripletLoop
            TripletLoop name or TripletLoop object.
        return_type: str, optional
            Whether to return the index or mask of the loop. Default is 'index'.

        """
        if isinstance(loop, str):
            loop = TripletLoop.from_name(loop)
        elif not isinstance(loop, TripletLoop):
            msg = f"loop should be str or TripletLoop, but got {type(loop)}."
            raise TypeError(msg)

        # TODO: finish this function

    def sort(
        self,
        order: str | list = "pairs",
        ascending: bool = True,
        inplace: bool = True,
    ) -> tuple[TripletLoops, NDArray[np.int64]] | None:
        """Sort the loops.

        Parameters
        ----------
        order: str or list of str, optional
            By which fields to sort the loops. this argument specifies
            which fields to compare first, second, etc. Default is 'pairs'.

            The available options are one or a list of:

            - **date:**: 'date1', 'date2', 'date3'
            - **pairs:** 'pairs12', 'pairs23', 'pairs13'
            - **days:** 'days12', 'days23', 'days13'
            - **short name:** 'date', 'pairs', 'days'. short name will be
                treated as a combination of the above options. For example,
                'date' is equivalent to ['date1', 'date2', 'date3'].
        ascending: bool, optional
            Whether to sort ascending. Default is True.
        inplace: bool, optional
            Whether to sort the loops inplace. Default is True.

        Returns
        -------
        sorted: (TripletLoops, np.ndarray) | None
            if inplace is True, return the sorted loops and the index of the
            sorted loops in the original loops. Otherwise, return None.

        """
        item_map = {
            "date1": self._values[:, 0],
            "date2": self._values[:, 1],
            "date3": self._values[:, 2],
            "pairs12": self.pairs12.values,
            "pairs23": self.pairs23.values,
            "pairs13": self.pairs13.values,
            "days12": self.days12,
            "days23": self.days23,
            "days13": self.days13,
            "date": self._values,
            "pairs": np.hstack(
                [self.pairs12.values, self.pairs23.values, self.pairs13.values],
            ),
            "days": np.hstack([self.days12, self.days23, self.days13]),
        }
        if isinstance(order, str):
            order = [order]
        _values = []
        for i in order:
            if i not in item_map:
                msg = (
                    f"order should be one of {list(item_map.keys())}, but got {order}."
                )
                raise ValueError(
                    msg,
                )
            _values.append(item_map[i])
        _values = np.hstack(_values)
        _, _index = np.unique(_values, axis=0, return_index=True)
        if not ascending:
            _index = _index[::-1]
        if inplace:
            self._values = self._values[_index]
            self._parse_loop_meta()
            return None
        return TripletLoops(self._values[_index]), _index

    def to_seasons(self) -> NDArray[np.int8]:
        """Return the season of each loop.

        Returns
        -------
        seasons: list
            list of seasons of each loop.
                0: not the same season
                1: spring
                2: summer
                3: fall
                4: winter

        """
        seasons = []
        for loop in self.values.astype("O"):
            season1 = DateManager.season_of_month(loop[0].month)
            season2 = DateManager.season_of_month(loop[1].month)
            season3 = DateManager.season_of_month(loop[2].month)
            if season1 == season2 == season3 and loop.days13 < 180:
                seasons.append(season1)
            else:
                seasons.append(0)
        return np.asarray(seasons, dtype=np.int8)


class Loop:
    """Loop class containing multiple pairs/acquisitions."""

    def __init__(self, loop: Sequence[datetime]) -> None:
        """Initialize the Loop class.

        loop: Sequence
            Sequence object of dates. Each date is a datetime object.
            For example, (date1, ..., date_n).
        """
        self._values = np.asarray(loop).astype("M8[D]")
        loop_dt = self._values.astype(datetime)
        self._name = "_".join([i.strftime("%Y%m%d") for i in loop_dt])

        num = len(self._values)
        self._length = num

        _pairs = []
        for i in range(num - 1):
            _pair = Pair([loop_dt[i], loop_dt[i + 1]])
            _pairs.append(_pair)
        _pairs.append(Pair([loop_dt[0], loop_dt[-1]]))
        self._pairs = Pairs(_pairs, sort=False)

    def __len__(self) -> int:
        """Return the number of pairs in the loop."""
        return self._length

    def __str__(self) -> str:
        """Return the name of the loop."""
        return self._name

    def __repr__(self) -> str:
        """Return the representation of the loop."""
        return f"Loop({self._name})"

    def __eq__(self, other: Loop) -> bool:
        """Compare whether the loop is same to another loop."""
        return self.name == other.name

    def __hash__(self) -> int:
        """Return the hash of the loop."""
        return hash(self.name)

    @property
    def values(self) -> NDArray[np.datetime64]:
        """Return the values array of the loop.

        Returns
        -------
        values: np.ndarray
            dates of the loop with format of np.datetime64[D].

        """
        return self._values

    @property
    def pairs(self) -> Pairs:
        """All pairs of the loop."""
        return self._pairs

    @property
    def name(self) -> str:
        """The string format the loop.

        Returns
        -------
        name: str
            String of the loop.

        """
        return self._name

    def from_name(
        self,
        name: str,
        parse_function: Callable | None = None,
        date_args: dict | None = None,
    ) -> Loop:
        """Initialize the loop class from a loop name.

        Parameters
        ----------
        name: str
            Loop name.
        parse_function: Callable, optional
            Function to parse the date strings from the loop name.
            If None, the loop name will be split by '_' and the
            last 3 items will be used. Default is None.
        date_args: dict, optional
            Keyword arguments for pd.to_datetime() to convert the date strings
            to datetime objects. For example, {'format': '%Y%m%d'}.
            Default is None.

        Returns
        -------
        loop: Loop
            Loop object.

        """
        if date_args is None:
            date_args = {}
        dates = DateManager.str_to_dates(name, 0, parse_function, date_args)
        return self(dates)


class Loops:
    """Loops class to handle loops with multiple acquisitions.

    The loops are stored as a padded date-index matrix over a shared
    sorted date array. All metadata (loop names, pairs, edge pairs,
    diagonal pairs, and the design-matrix entry locations) is built once
    with vectorized operations. Individual :class:`Loop` objects are
    only constructed when they are actually accessed through indexing,
    iteration, or the :attr:`loops` property.
    """

    def __init__(self, loops: list, sort: bool = True) -> None:
        """Initialize the loops class.

        Parameters
        ----------
        loops: list
            a list containing Loop objects.
        sort: bool, optional
            Whether to sort the loops. Default is True.

        """
        loops = list(loops)
        if not loops:
            self._dates = np.array([], dtype="M8[D]")
            self._date_idx = np.zeros((0, 3), dtype=np.int64)
        else:
            self._dates = np.unique(
                np.concatenate([np.asarray(loop.values) for loop in loops])
            )
            max_len = max(len(loop) for loop in loops)
            self._date_idx = np.full((len(loops), max_len), -1, dtype=np.int64)
            for i, loop in enumerate(loops):
                idx = np.searchsorted(self._dates, np.asarray(loop.values))
                self._date_idx[i, : len(loop)] = idx
        self._init_meta(sort=sort)

    @classmethod
    def from_date_indices(
        cls,
        date_idx: NDArray[np.int64],
        dates: NDArray[np.datetime64],
        sort: bool = True,
    ) -> Loops:
        """Build Loops from a padded date-index matrix without Loop objects.

        Parameters
        ----------
        date_idx : np.ndarray
            ``(n_loops, max_len)`` integer matrix of date indices into
            ``dates``. Variable-length loops are padded at the end with
            ``-1``.
        dates : np.ndarray
            Sorted date array (``datetime64``) that the indices refer to.
        sort : bool, optional
            Whether to sort the loops by name. Default is True.

        Returns
        -------
        Loops
            The Loops object.

        """
        loops = cls.__new__(cls)
        loops._dates = np.asarray(dates, dtype="M8[D]")
        loops._date_idx = np.asarray(date_idx, dtype=np.int64)
        loops._init_meta(sort=sort)
        return loops

    def _init_meta(self, sort: bool = True) -> None:
        """Compute names, pair sets, and matrix entry locations vectorized."""
        rows = self._date_idx
        self._length = len(rows)
        self._loops_cache = None

        if self._length == 0:
            self._names = np.array([], dtype="<U17")
            self._pairs = Pairs.from_names([])
            self._edge_pairs = Pairs.from_names([])
            self._diagonal_pairs = Pairs.from_names([])
            self._edge_rows = np.array([], dtype=np.int64)
            self._edge_cols = np.array([], dtype=np.int64)
            self._diag_rows = np.array([], dtype=np.int64)
            self._diag_cols = np.array([], dtype=np.int64)
            return

        date_str = np.char.replace(
            np.datetime_as_string(self._dates, unit="D"), "-", ""
        )
        lengths = (rows >= 0).sum(axis=1)
        self._names = np.array(
            ["_".join(date_str[row[row >= 0]].tolist()) for row in rows]
        )

        edge_u = rows[:, :-1]
        edge_v = rows[:, 1:]
        edge_valid = (edge_u >= 0) & (edge_v >= 0)
        self._edge_rows = np.repeat(np.arange(self._length), rows.shape[1] - 1)[
            edge_valid.ravel()
        ]
        edge_names = np.char.add(
            date_str[edge_u[edge_valid]],
            np.char.add("_", date_str[edge_v[edge_valid]]),
        )
        last_dates = rows[np.arange(self._length), lengths - 1]
        diag_names = np.char.add(
            date_str[rows[:, 0]], np.char.add("_", date_str[last_dates])
        )

        pair_names = np.concatenate([edge_names, diag_names])
        unique_pairs = np.unique(pair_names)
        self._pairs = Pairs.from_names(unique_pairs.tolist())
        self._edge_pairs = Pairs.from_names(np.unique(edge_names).tolist())
        self._diagonal_pairs = Pairs.from_names(np.unique(diag_names).tolist())
        self._edge_cols = np.searchsorted(unique_pairs, edge_names)
        self._diag_cols = np.searchsorted(unique_pairs, diag_names)
        self._diag_rows = np.arange(self._length)

        if sort:
            self.sort()

    def __str__(self) -> str:
        """Return the string representation of the loops."""
        return (
            f"Loops(loops={len(self)}, pairs={len(self.pairs)}, "
            f"edge_pairs={len(self.edge_pairs)}, "
            f"diagonal_pairs={len(self.diagonal_pairs)})"
        )

    def __repr__(self) -> str:
        """Return the representation of the loops."""
        return (
            "Loops("
            f"\n    loops={len(self)},"
            f"\n    pairs={len(self.pairs)},"
            f"\n    edge_pairs={len(self.edge_pairs)},"
            f"\n    diagonal_pairs={len(self.diagonal_pairs)}"
            f"\n)"
        )

    def __len__(self) -> int:
        """Return the number of loops."""
        return self._length

    def __getitem__(self, index: int) -> Loop:
        """Return the loop by index, constructing it on demand."""
        return self._make_loop(index)

    def __iter__(self) -> iter[Loop]:
        """Iterate the loops, constructing each Loop on demand."""
        return iter(self.loops)

    def __contains__(self, item: Loop) -> bool:
        """Check if the item is in the loops by name."""
        return item.name in self._names

    def _make_loop(self, index: int) -> Loop:
        """Build a single Loop object from one row of the date matrix."""
        row = self._date_idx[index]
        return Loop(self._dates[row[row >= 0]])

    @property
    def loops(self) -> NDArray[np.object_]:
        """The loops in the numpy array format.

        Every :class:`Loop` object is constructed here, once, and cached.
        """
        if self._loops_cache is None:
            self._loops_cache = np.array(
                [self._make_loop(i) for i in range(self._length)],
                dtype=object,
            )
        return self._loops_cache

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the loops array with format of (n_loops, n_pairs)."""
        return (len(self), len(self.pairs))

    @property
    def names(self) -> NDArray[np.str_]:
        """The names (str format) of the loops."""
        return self._names

    @property
    def pairs(self) -> Pairs:
        """All pairs in the loops."""
        return self._pairs

    @property
    def edge_pairs(self) -> Pairs:
        """All edge pairs in the loops."""
        return self._edge_pairs

    @property
    def diagonal_pairs(self) -> Pairs:
        """All diagonal pairs in the loops."""
        return self._diagonal_pairs

    def sort(self, ascending: bool = True, inplace: bool = True) -> Loops | None:
        """Sort the loops by name.

        Parameters
        ----------
        ascending: bool, optional
            Whether to sort the loops ascending. Default is True.
        inplace: bool, optional
            Whether to sort the loops in place. if False, return the sorted loops.
            Default is True.

        """
        if self._length == 0:
            if inplace:
                return None
            return Loops.from_date_indices(self._date_idx, self._dates, sort=False)
        _, index = np.unique(self._names, return_index=True)
        if not ascending:
            index = index[::-1]
        if inplace:
            self._reindex(index)
            return None
        return Loops.from_date_indices(self._date_idx[index], self._dates, sort=False)

    def _reindex(self, index: NDArray[np.int64]) -> None:
        """Reorder all array-backed metadata by the given row index.

        ``index`` may be shorter than ``_length`` when duplicate-named
        loops are dropped; scatter entries that point at dropped rows are
        removed so the matrix locations stay in bounds.
        """
        keep = np.zeros(self._length, dtype=bool)
        keep[index] = True
        position = np.zeros(self._length, dtype=np.int64)
        position[index] = np.arange(len(index))
        self._date_idx = self._date_idx[index]
        self._names = self._names[index]
        edge_keep = keep[self._edge_rows]
        self._edge_rows = position[self._edge_rows[edge_keep]]
        self._edge_cols = self._edge_cols[edge_keep]
        diag_keep = keep[self._diag_rows]
        self._diag_rows = position[self._diag_rows[diag_keep]]
        self._diag_cols = self._diag_cols[diag_keep]
        self._length = len(index)
        self._loops_cache = None

    def loop_matrix(self, dtype: DTypeLike = None) -> NDArray[np.number]:
        """Return a design matrix describes the relationship between loops and pairs.

        The rows and columns of this matrix are loops and pairs respectively.
        The values of the matrix are 1 for the edge pairs, -1 for the diagonal
        pairs, and 0 otherwise. The matrix is filled by a vectorized scatter
        of the precomputed entry locations.
        """
        matrix = np.zeros((self._length, len(self.pairs)), dtype=dtype)
        matrix[self._edge_rows, self._edge_cols] = 1
        matrix[self._diag_rows, self._diag_cols] = -1
        return matrix
