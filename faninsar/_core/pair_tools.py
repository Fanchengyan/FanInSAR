from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray


class Pair:
    """Pair class for one pair."""

    _values: np.ndarray
    _name: str
    _days: int

    __slots__ = ["_values", "_name", "_days"]

    def __init__(
        self,
        pair: Iterable[datetime, datetime],
    ) -> None:
        """
        Parameters
        ----------
        pair: Iterable
            Iterable object of two dates. Each date is a datetime object.
            For example, (date1, date2).
        """
        self._values = np.sort(pair).astype("M8[D]")
        self._name = "_".join([i.strftime("%Y%m%d") for i in self._values.astype("O")])
        self._days = (self._values[1] - self._values[0]).astype(int)

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"Pair({self._name})"

    def __eq__(self, other: "Pair") -> bool:
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self._name)

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self._values, dtype=dtype)

    @property
    def values(self) -> NDArray[np.datetime64]:
        """the values of the pair with format of datetime."""
        return self._values

    @property
    def name(self) -> str:
        """String of the pair with format of '%Y%m%d_%Y%m%d'."""
        return self._name

    @property
    def days(self) -> int:
        """the time span of the pair in days."""
        return self._days

    @property
    def primary(self) -> np.datetime64:
        """the primary dates of all pairs"""
        return self.values[0]

    @property
    def secondary(self) -> np.datetime64:
        """the secondary dates of all pairs"""
        return self.values[1]

    def primary_string(self, date_format="%Y%m%d") -> str:
        """return the primary dates of all pairs in string format

        Parameters
        ----------
        date_format: str
            Format of the date string. Default is '%Y%m%d'. See more at
            `strftime Format Codes <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_.
        """
        return pd.to_datetime(self.values[0]).strftime(date_format)

    def secondary_string(self, date_format="%Y%m%d") -> str:
        """return the secondary dates of all pairs in string format

        Parameters
        ----------
        date_format: str
            Format of the date string. Default is '%Y%m%d'. See more at
            `strftime Format Codes <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_.
        """
        return pd.to_datetime(self.values[1]).strftime(date_format)

    @classmethod
    def from_name(
        cls,
        name: str,
        parse_function: Optional[Callable] = None,
        date_args: dict = {},
    ) -> "Pair":
        """initialize the pair class from a pair name

        Parameters
        ----------
        name: str
            Pair name.
        parse_function: Callable, optional
            Function to parse the date strings from the pair name.
            If None, the pair name will be split by '_' and
            the last 2 items will be used. Default is None.
        date_args: dict, optional
            Keyword arguments for pd.to_datetime() to convert the date strings
            to datetime objects. For example, {'format': '%Y%m%d'}.
            Default is {}.

        Returns
        -------
        pair: Pair
            Pair class.
        """
        dates = DateManager.str_to_dates(name, 2, parse_function, date_args)
        return cls(dates)


class Pairs:
    """Pairs class to handle pairs

    Examples
    --------
    prepare dates and pairs for examples:

    >>> dates = pd.date_range('20130101', '20231231').values
    >>> n = len(dates)
    >>> pair_ls = []
    >>> loop_ls = []
    >>> for i in range(5):
    ...    np.random.seed(i)
    ...    pair_ls.append(dates[np.random.randint(0, n, 2)])

    initialize pairs from a list of pairs

    >>> pairs = Pairs(pair_ls)
    >>> print(pairs)
        primary  secondary
    0 2013-06-24 2016-02-21
    1 2013-08-24 2015-11-28
    2 2017-08-16 2018-03-14
    3 2020-01-20 2021-11-15
    4 2020-02-21 2020-06-25

    select pairs by date slice

    >>> pairs1 = pairs['2018-03-09':]
    >>> print(pairs1)
        primary  secondary
    0 2020-01-20 2021-11-15
    1 2020-02-21 2020-06-25

    pairs can be added (union)  and subtracted (difference)

    >>> pairs2 = pairs - pairs1
    >>> pairs3 = pairs1 + pairs2
    >>> print(pairs2)
        primary  secondary
    0 2013-06-24 2016-02-21
    1 2013-08-24 2015-11-28
    2 2017-08-16 2018-03-14
    >>> print(pairs3)
        primary  secondary
    0 2013-06-24 2016-02-21
    1 2013-08-24 2015-11-28
    2 2017-08-16 2018-03-14
    3 2020-01-20 2021-11-15
    4 2020-02-21 2020-06-25

    pairs can be compared with `==`and `!=`

    >>> print(pairs3 == pairs)
    >>> print(pairs3 != pairs)
    True
    False
    """

    _values: np.ndarray
    _dates: np.ndarray
    _length: int

    __slots__ = ["_values", "_dates", "_length", "_edge_index", "_names"]

    def __init__(
        self,
        pairs: Iterable[Iterable[datetime, datetime]] | Iterable[Pair],
        sort: bool = True,
    ) -> None:
        """initialize the pairs class


        Parameters
        ----------
        pairs: Iterable
            Iterable object of pairs. Each pair is an Iterable or Pair
            object of two dates with format of datetime. For example,
            [(date1, date2), ...].
        sort: bool, optional
            Whether to sort the pairs. Default is True.
        """
        if pairs is None or len(pairs) == 0:
            raise ValueError("pairs cannot be None.")

        _values = np.array(pairs, dtype="M8[D]")

        self._values = _values
        self._parse_pair_meta()

        if sort:
            self.sort(inplace=True)

    def _parse_pair_meta(self):
        self._dates = np.unique(self._values.flatten())
        self._length = self._values.shape[0]
        self._edge_index = np.searchsorted(self._dates, self._values)
        self._names = self.to_names()

    def __len__(self) -> int:
        return self._length

    def __str__(self) -> str:
        return f"Pairs({self._length})"

    def __repr__(self) -> str:
        return self.to_frame().__repr__()

    def __eq__(self, other: "Pairs") -> bool:
        return np.array_equal(self.values, other.values)

    def __add__(self, other: "Pairs") -> "Pairs":
        _pairs = np.union1d(self.names, other.names)
        if len(_pairs) > 0:
            return Pairs.from_names(_pairs)

    def __sub__(self, other: "Pairs") -> "Pairs":
        _pairs = np.setdiff1d(self.names, other.names)
        if len(_pairs) > 0:
            return Pairs.from_names(_pairs)

    def __getitem__(self, index: int) -> "Pair" | "Pairs":
        if isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            if isinstance(start, (int, np.integer, type(None))) and isinstance(
                stop, (int, np.integer, type(None))
            ):
                if start is None:
                    start = 0
                if stop is None:
                    stop = self._length
                return Pairs(self._values[start:stop:step])
            elif isinstance(
                start, (datetime, np.datetime64, pd.Timestamp, str, type(None))
            ) and isinstance(
                stop, (datetime, np.datetime64, pd.Timestamp, str, type(None))
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
                    raise ValueError(
                        f"Index start {start} should be earlier than index stop {stop}."
                    )
                _pairs = []
                for pair in self._values:
                    pair = pair.astype("M8[s]")
                    if start <= pair[0] <= stop and start <= pair[1] <= stop:
                        _pairs.append(pair)
                if len(_pairs) > 0:
                    return Pairs(_pairs)
                else:
                    return None
        elif isinstance(index, (int, np.integer)):
            if index >= self._length:
                raise IndexError(
                    f"Index {index} out of range. Pairs number is {self._length}."
                )
            return Pair(self._values[index])
        elif isinstance(index, (datetime, np.datetime64, pd.Timestamp, str)):
            if isinstance(index, str):
                try:
                    index = pd.to_datetime(index)
                except:
                    raise ValueError(f"String {index} cannot be converted to datetime.")
            pairs = []
            for pair in self._values:
                if index in pair:
                    pairs.append(pair)
            if len(pairs) > 0:
                return Pairs(pairs)
            else:
                return None
        elif isinstance(index, Iterable):
            index = np.array(index)
            return Pairs(self._values[index])
        else:
            raise TypeError(
                "Index should be int, slice, datetime, str, or bool or int array"
                f" indexing, but got {type(index)}."
            )

    def __hash__(self) -> int:
        return hash("".join(self.names))

    def __iter__(self):
        pairs_ls = [Pair(i) for i in self._values]
        return iter(pairs_ls)

    def __contains__(self, item):
        if isinstance(item, Pair):
            item = item.values
        elif isinstance(item, str):
            item = Pair.from_name(item).values
        elif isinstance(item, Iterable):
            item = np.sort(item)
        else:
            raise TypeError(
                f"item should be Pair, str, or Iterable, but got {type(item)}."
            )

        return np.any(np.all(item == self.values, axis=1))

    def __array__(self, dtype=None) -> NDArray[np.datetime64]:
        return np.asarray(self._values, dtype=dtype)

    def _ensure_pairs(
        self, pairs: str | Pair | "Pairs" | Iterable[str] | Iterable[Pair]
    ) -> "Pairs":
        """ensure the pairs are in the Pairs object"""
        if isinstance(pairs, str):
            pairs = Pairs.from_names([pairs])
        elif isinstance(pairs, Pair):
            pairs = Pairs([pairs])
        elif isinstance(pairs, Pairs):
            return pairs
        elif isinstance(pairs, Iterable):
            pairs = np.asarray(pairs)
            if pairs.ndim == 1 and pairs.dtype == "O":
                pairs = Pairs.from_names(pairs)
            elif pairs.ndim == 2 and pairs.shape[1] == 2:
                pairs = Pairs(pairs)
            else:
                raise ValueError(
                    f"pairs should be 1D array of str, 2D array of datetime, or Pairs, but got {pairs}."
                )
        else:
            raise TypeError(
                f"pairs should be str, Pair, list of str, list of Pair, or Pairs, but got {type(pairs)}."
            )
        return pairs

    @property
    def values(self) -> NDArray[np.datetime64]:
        """return the numpy array of the pairs"""
        return self._values

    @property
    def names(self) -> NDArray[np.str_]:
        """return the names (string format) of the pairs"""
        return self._names

    @property
    def dates(self) -> pd.DatetimeIndex:
        """return the sorted dates array of all pairs in type of np.datetime64[D]"""
        return pd.to_datetime(self._dates)

    @property
    def days(self) -> NDArray[np.int64]:
        """return the time span of all pairs in days"""
        return (self._values[:, 1] - self._values[:, 0]).astype(int)

    @property
    def primary(self) -> pd.DatetimeIndex:
        """return the primary dates of all pairs"""
        return pd.to_datetime(self._values[:, 0])

    @property
    def secondary(self) -> pd.DatetimeIndex:
        """return the secondary dates of all pairs"""
        return pd.to_datetime(self._values[:, 1])

    def primary_string(self, date_format="%Y%m%d") -> pd.Index:
        """return the primary dates of all pairs in string format

        Parameters
        ----------
        date_format: str
            Format of the date string. Default is '%Y%m%d'. See more at
            `strftime Format Codes <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_.
        """
        return self.primary.strftime(date_format)

    def secondary_string(self, date_format="%Y%m%d") -> pd.Index:
        """return the secondary dates of all pairs in string format

        Parameters
        ----------
        date_format: str
            Format of the date string. Default is '%Y%m%d'. See more at
            `strftime Format Codes <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_.
        """
        return self.secondary.strftime(date_format)

    @property
    def edge_index(self) -> NDArray[np.int64]:
        """return the index of the pairs in the dates coordinate (edge index in
        graph theory)"""
        return self._edge_index

    @property
    def shape(self) -> tuple[int, int]:
        """return the shape of the pairs array"""
        return self._values.shape

    @classmethod
    def from_names(
        cls,
        names: Iterable[str],
        parse_function: Optional[Callable] = None,
        date_args: dict = {},
    ) -> "Pairs":
        """initialize the pair class from a pair name

        .. note::
            The pairs will be in the order of the input names. If you want to
            sort the pairs, you can call the :meth:`Pairs.sort()` method to
            achieve it.

        Parameters
        ----------
        name: str
            Pair name.
        parse_function: Callable, optional
            Function to parse the date strings from the pair name.
            If None, the pair name will be split by '_' and
            the last 2 items will be used. Default is None.
        date_args: dict, optional
            Keyword arguments for pd.to_datetime() to convert the date strings
            to datetime objects. For example, {'format': '%Y%m%d'}.
            Default is {}.

        Returns
        -------
        pairs: Pairs
            unsorted Pairs object.
        """
        pairs = []
        for name in names:
            pair = Pair.from_name(name, parse_function, date_args)
            pairs.append(pair.values)
        return cls(pairs, sort=False)

    def where(
        self,
        pairs: list[str] | list[Pair] | "Pairs",
        return_type: Literal["index", "mask"] = "index",
    ) -> Optional[NDArray[np.int64 | np.bool_]]:
        """return the index of the pairs

        Parameters
        ----------
        pairs: list of str or Pair, or Pairs
            Pair names or Pair objects, or Pairs object.
        return_type: str, optional
            Whether to return the index or mask of the pairs. Default is 'index'.
        """
        pairs = self._ensure_pairs(pairs)
        con = np.isin(self.names, pairs.names)
        if return_type == "mask":
            return con
        elif return_type == "index":
            if np.any(con):
                return np.where(con)[0]
        else:
            raise ValueError(
                f"return_type should be one of ['index', 'mask'], but got {return_type}."
            )

    def intersect(self, pairs: list[str] | list[Pair] | "Pairs") -> Optional["Pairs"]:
        """return the intersection of the pairs. The pairs both in self and
        input pairs.

        Parameters
        ----------
        pairs: list of str or Pair, or Pairs
            Pair names or Pair objects, or Pairs object.
        """
        pairs = self._ensure_pairs(pairs)
        return self[self.where(pairs)]

    def union(self, pairs: list[str] | list[Pair] | "Pairs") -> "Pairs":
        """return the union of the pairs. All pairs that in self and input pairs.
        A more robust operation than addition.

        Parameters
        ----------
        pairs: list of str or Pair, or Pairs
            Pair names or Pair objects, or Pairs object.
        """
        pairs = self._ensure_pairs(pairs)
        return self + pairs

    def difference(self, pairs: list[str] | list[Pair] | "Pairs") -> Optional["Pairs"]:
        """return the difference of the pairs. The pairs in self but not in pairs.
        A more robust operation than subtraction.

        Parameters
        ----------
        pairs: list of str or Pair, or Pairs
            Pair names or Pair objects, or Pairs object.
        """
        pairs = self._ensure_pairs(pairs)
        return self - pairs

    def copy(self) -> "Pairs":
        """return a copy of the pairs"""
        return Pairs(self._values.copy())

    def sort(
        self,
        order: list | Literal["pairs", "primary", "secondary", "days"] = "pairs",
        ascending: bool = True,
        inplace: bool = True,
    ) -> Optional[tuple["Pairs", NDArray[np.int64]]]:
        """sort the pairs

        Parameters
        ----------
        order: str or list of str, optional
            By which fields to sort the pairs. This argument specifies
            which fields to compare first, second, etc. Default is 'pairs'.

            The available options are one or a list of:
            ['pairs', 'primary', 'secondary', 'days'].
        ascending: bool, optional
            Whether to sort ascending. Default is True.
        inplace: bool, optional
            Whether to sort the pairs inplace. Default is True.

        Returns
        -------
        None or (Pairs, np.ndarray). if inplace is True, return the sorted pairs
        and the index of the sorted pairs in the original pairs. Otherwise,
        return None.
        """
        item_map = {
            "pairs": self._values,
            "primary": self._values[:, 0],
            "secondary": self._values[:, 1],
            "days": self.days,
        }
        if isinstance(order, str):
            order = [order]
        _values_ = []
        for i in order:
            if i not in item_map.keys():
                raise ValueError(
                    f"order should be one of {list(item_map.keys())}, but got {order}."
                )
            _values_.append(item_map[i].reshape(self._length, -1))
        _values_ = np.hstack(_values_)
        _values, _index = np.unique(_values_, axis=0, return_index=True)
        if not ascending:
            _index = _index[::-1]
        if inplace:
            self._values = _values
            self._parse_pair_meta()
        else:
            return Pairs(_values), _index

    def to_names(self, prefix: Optional[str] = None) -> NDArray[np.str_]:
        """generate pairs names string with prefix

        Parameters
        ----------
        prefix: str
            Prefix of the pair file names. Default is None.

        Returns
        -------
        names: np.ndarray
            Pairs names string with format of '%Y%m%d_%Y%m%d'.
        """
        names = (
            pd.DatetimeIndex(self.primary).strftime("%Y%m%d")
            + "_"
            + pd.DatetimeIndex(self.secondary).strftime("%Y%m%d")
        )
        if prefix:
            names = prefix + "_" + names

        return names.values

    def to_frame(self) -> pd.DataFrame:
        """return the pairs as a DataFrame"""
        return pd.DataFrame(self._values, columns=["primary", "secondary"])

    def to_triplet_loops(self) -> "TripletLoops":
        """return all possible triplet loops from the pairs"""
        loops = []
        for i, pair12 in enumerate(self._values):
            for pair23 in self._values[i + 1 :]:
                if pair12[1] == pair23[0] and Pair([pair12[0], pair23[1]]) in self:
                    loops.append([pair12[0], pair12[1], pair23[1]])
        return TripletLoops(loops)

    def to_loops(
        self,
        max_acquisition: int = 5,
        max_days: int | None = None,
        edge_pairs: "Pairs" | None = None,
        edge_days: int | None = None,
    ) -> "Loops":
        """Return all possible loops from the pairs.

        .. important::
            The pairs in the loops may be fewer than the input pairs. You can use
            the :meth:`Pairs.where` method to get the index/mask of the pairs
            in the loops from the input pairs.

            **Example**:

            >>> loops = pairs.to_loops()
            >>> mask = pairs.where(loops.pairs, return_type='mask')

        Parameters
        ----------
        max_acquisition: int
            The maximum number of acquisitions in the loops. It should be at least 3.

            .. note::
                The number of acquisitions is equal to the number of intervals + 1
                :math:`n (acquisition) = n (edge\ pairs) + 1 (diagonal\ pair) = n (intervals) + 1`.
        max_days: int, optional
            The maximum number of days for the pairs in the loops. If None, all available pairs
            will be used. Default is None.
        edge_pairs: Pairs, optional
            The edge pairs to form loops. If None, ``edge_days`` must be provided.
        edge_days: int, optional
            The maximum number of days used to identify the edge pairs. If None,
            ``edge_pairs`` must be provided. Default is None.

            .. note::
                This parameter will be ignored if ``edge_pairs`` is provided.
        """
        # a list containing all loops
        loops = []
        if edge_pairs is None:
            if edge_days is None:
                raise ValueError("Either edge_days or edge_pairs should be provided.")
            edge_pairs = self[self.days <= edge_days]
        # find valid diagonal pairs
        for i in self - edge_pairs:
            if max_days is not None and i.days > max_days:
                continue
            if not valid_diagonal_pair(i, self, edge_pairs):
                continue
            start_date, end_date = i.values[0], i.values[1]

            # find valid primary pairs
            mask_primaries = (self.primary == start_date) & (self.secondary < end_date)
            if not mask_primaries.any():
                continue
            pairs_primary = self[mask_primaries]

            # initialize a loop with the primary acquisition
            loop = [start_date]

            # find all loops for all primary acquisitions
            find_loops(
                self, loops, loop, pairs_primary, end_date, edge_pairs, max_acquisition
            )

        return Loops(loops)

    def to_matrix(self, dtype=None) -> NDArray[np.number]:
        """return the SBAS matrix

        Parameters
        ----------
        matrix: np.ndarray
            SBAS matrix in shape of (n_pairs, n_dates-1). The dates between
            pairs are set to 1, otherwise 0.
        """
        matrix = np.zeros((len(self), len(self.dates) - 1), dtype=dtype)
        col_idxs = self.edge_index.copy()
        for row_idx, col_idx in enumerate(col_idxs):
            matrix[row_idx, col_idx[0] : col_idx[1]] = 1

        return matrix


class TripletLoop:
    """TripletLoop class containing three pairs/acquisitions."""

    _values: np.ndarray
    _name: str
    _pairs: list[Pair]

    __slots__ = ["_values", "_pairs", "_name", "_days12", "_days23", "_days13"]

    def __init__(self, loop: Iterable[datetime, datetime, datetime]) -> None:
        """initialize the TripletLoop class

        Parameters
        ----------
        loop: Iterable
            Iterable object of three dates. Each date is a datetime object.
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
        return self._name

    def __repr__(self) -> str:
        return f"TripletLoop({self._name})"

    def __eq__(self, other) -> bool:
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self._values, dtype=dtype)

    @property
    def values(self) -> NDArray[np.datetime64]:
        """Three dates of the loop with format of np.datetime64[D]."""
        return self._values

    @property
    def pairs(self) -> list[Pair]:
        """all three pairs of the loop."""
        return self._pairs

    @property
    def days12(self) -> int:
        """return the time span of the first pair in days.

        Returns
        -------
        days12: int
            Time span of the first pair in days.
        """
        return (self._values[1] - self._values[0]).astype(int)

    @property
    def days23(self) -> int:
        """return the time span of the second pair in days.

        Returns
        -------
        days23: int
            Time span of the second pair in days.
        """
        return (self._values[2] - self._values[1]).astype(int)

    @property
    def days13(self) -> int:
        """return the time span of the third pair in days.

        Returns
        -------
        days13: int
            Time span of the third pair in days.
        """
        return (self._values[2] - self._values[0]).astype(int)

    @property
    def name(self) -> str:
        """return the string format the loop.

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
        parse_function: Optional[Callable] = None,
        date_args: dict = {},
    ) -> "TripletLoop":
        """initialize the loop class from a loop name

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
        dates = DateManager.str_to_dates(name, 3, parse_function, date_args)
        return cls(dates)


class TripletLoops:
    """TripletLoops class to handle loops with three pairs/acquisitions."""

    _values: np.ndarray
    _dates: np.ndarray
    _length: int

    __slots__ = ["_values", "_dates", "_length"]

    def __init__(
        self,
        loops: Iterable[Iterable[datetime, datetime, datetime]] | Iterable[TripletLoop],
        sort: bool = True,
    ) -> None:
        """initialize the triplet loops class

        Parameters
        ----------
        loops: Iterable
            Iterable object of triplet loops. Each loop is an Iterable object
            of three dates with format of datetime or TripletLoop object.
            For example, [(date1, date2, date3), ...].
        """
        if loops is None or len(loops) == 0:
            raise ValueError("loops cannot be None.")

        _values = np.array(loops, dtype="M8[D]")
        self._values = _values
        self._parse_loop_meta()

        if sort:
            self.sort(inplace=True)

    def _parse_loop_meta(self):
        self._dates = np.unique(self._values)
        self._length = self._values.shape[0]
        self._names = self.to_names()

    def __str__(self) -> str:
        return f"TripletLoops({self._length})"

    def __repr__(self) -> str:
        return self.to_frame("dates").__repr__()

    def __len__(self) -> int:
        return self._length

    def __eq__(self, other: "TripletLoops") -> bool:
        return np.array_equal(self.values, other.values)

    def __add__(self, other: "TripletLoops") -> "TripletLoops":
        _loops = np.union1d(self.names, other.names)
        return TripletLoops.from_names(_loops)

    def __sub__(self, other: "TripletLoops") -> Optional["TripletLoops"]:
        _loops = np.setdiff1d(self.names, other.names)
        if len(_loops) > 0:
            return TripletLoops.from_names(_loops)

    def __getitem__(self, index: int) -> "TripletLoop" | "TripletLoops":
        if isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            if isinstance(start, (int, np.integer, type(None))) and isinstance(
                stop, (int, np.integer, type(None))
            ):
                if start is None:
                    start = 0
                if stop is None:
                    stop = self._length
                return TripletLoops(self._values[start:stop:step])
            elif isinstance(
                start, (datetime, np.datetime64, pd.Timestamp, str, type(None))
            ) and isinstance(
                stop, (datetime, np.datetime64, pd.Timestamp, str, type(None))
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
                    raise ValueError(
                        f"Index start {start} should be earlier than index stop {stop}."
                    )
                _loops = []
                for loop in self._values:
                    loop = loop.astype("M8[s]")
                    if np.all((start <= loop) & (loop <= stop)):
                        _loops.append(loop)
                if len(_loops) > 0:
                    return TripletLoops(_loops)
                else:
                    return None
        elif isinstance(index, (int, np.integer)):
            if index >= self._length:
                raise IndexError(
                    f"Index {index} out of range. TripletLoops number is {self._length}."
                )
            return TripletLoop(self._values[index])
        elif isinstance(index, (datetime, np.datetime64, pd.Timestamp, str)):
            if isinstance(index, str):
                try:
                    index = pd.to_datetime(index)
                except:
                    raise ValueError(f"String {index} cannot be converted to datetime.")
            loops = []
            for loop in self._values:
                if index in loop:
                    loops.append(loop)
            if len(loops) > 0:
                return TripletLoops(loops)
            else:
                return None
        elif isinstance(index, Iterable):
            index = np.array(index)
            return TripletLoops(self._values[index])
        else:
            raise TypeError(
                f"Index should be int, slice, datetime, str, or bool or int array"
                f"indexing, but got {type(index)}."
            )

    def __hash__(self) -> int:
        return hash("".join(self.names))

    def __iter__(self):
        return iter(self.values)

    def __contains__(self, item):
        if isinstance(item, TripletLoop):
            item = item.values
        elif isinstance(item, str):
            item = TripletLoop.from_name(item).values
        elif isinstance(item, Iterable):
            item = np.sort(item)
        else:
            raise TypeError(
                f"item should be TripletLoop, str, or Iterable, but got {type(item)}."
            )

        return np.any(np.all(item == self.values, axis=1))

    def __array__(self, dtype=None) -> NDArray[np.datetime64]:
        return np.asarray(self._values, dtype=dtype)

    @property
    def values(self) -> NDArray[np.datetime64]:
        """return the values of the loops.

        Returns
        -------
        values: np.ndarray
            Values of the loops with format of datetime.
        """
        return self._values

    @property
    def names(self) -> NDArray[np.str_]:
        """the names (sting format) of the loops."""
        return self._names

    @property
    def dates(self) -> NDArray[np.datetime64]:
        """Sorted dates of the loops with format of datetime."""
        return self._dates

    @property
    def shape(self) -> tuple[int, int]:
        """the shape of the loop array"""
        return self._values.shape

    @property
    def pairs(self) -> Pairs:
        """all sorted pairs of the loops."""
        pairs = np.unique(
            np.vstack(
                [self._values[:, :2], self._values[:, 1:], self._values[:, [0, 2]]]
            ),
            axis=0,
        )
        return Pairs(pairs, sort=False)

    @property
    def pairs12(self) -> Pairs:
        """the first pairs of the loops."""
        return Pairs(self._values[:, :2], sort=False)

    @property
    def pairs23(self) -> Pairs:
        """the second pairs of the loops."""
        return Pairs(self._values[:, 1:], sort=False)

    @property
    def pairs13(self) -> Pairs:
        """the third pairs of the loops."""
        return Pairs(self._values[:, [0, 2]], sort=False)

    @property
    def days12(self) -> NDArray[np.int64]:
        """the time span of the first pair in days."""
        return (self._values[:, 1] - self._values[:, 0]).astype(int)

    @property
    def days23(self) -> NDArray[np.int64]:
        """the time span of the second pair in days."""
        return (self._values[:, 2] - self._values[:, 1]).astype(int)

    @property
    def days13(self) -> NDArray[np.int64]:
        """the time span of the third pair in days."""
        return (self._values[:, 2] - self._values[:, 0]).astype(int)

    @property
    def index(self) -> NDArray[np.int64]:
        """the index of the loops in dates coordinates."""
        return np.searchsorted(self._dates, self._values)

    @classmethod
    def from_names(
        cls,
        names: list[str],
        parse_function: Optional[Callable] = None,
        date_args: dict = {},
    ) -> "TripletLoops":
        """initialize the loops class from a list of loop file names.

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
        loops = []
        for name in names:
            loop = TripletLoop.from_name(name, parse_function, date_args)
            loops.append(loop.values)
        return cls(loops, sort=False)

    def to_names(self, prefix: Optional[str] = None) -> NDArray[np.str_]:
        """return the string name of each loop.

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

        return names.values

    def to_frame(self, target: Literal["pairs", "dates"] = "pairs") -> pd.DataFrame:
        """return the loops as a DataFrame

        Parameters
        ----------
        target: str, one of ['pairs', 'dates']
            Target of the DataFrame. Default is 'pairs'.
        """
        if target == "pairs":
            return pd.DataFrame(
                zip(self.pairs12.values, self.pairs23.values, self.pairs13.values),
                columns=["pair12", "pair23", "pair13"],
            )
        elif target == "dates":
            return pd.DataFrame(self.values, columns=["date1", "date2", "date3"])
        else:
            raise ValueError(f"target should be 'pairs' or 'dates', but got {target}.")

    def to_matrix(self) -> NDArray[np.int8]:
        """
        return loop matrix (containing 1, -1, 0) from pairs.

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
        matrix = np.zeros((n_loop, n_pair))
        pairs_ls = self.pairs.values.tolist()
        for i, loop in enumerate(self.values):
            matrix[i, pairs_ls.index(loop[:2].tolist())] = 1
            matrix[i, pairs_ls.index(loop[1:].tolist())] = 1
            matrix[i, pairs_ls.index(loop[[0, 2]].tolist())] = -1
        return matrix

    def where(
        self,
        loop: str | TripletLoop,
        return_type: Literal["index", "mask"] = "index",
    ) -> Optional[int]:
        """return the index of the loop

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
            raise TypeError(f"loop should be str or TripletLoop, but got {type(loop)}.")

        # TODO: finish this function

    def sort(
        self,
        order: str | list = "pairs",
        ascending: bool = True,
        inplace: bool = True,
    ) -> Optional[tuple["TripletLoops", NDArray[np.int64]]]:
        """sort the loops

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
        return_index: bool, optional
            Whether to return the index of the sorted loops. Default is False.

        Returns
        -------
        None or (TripletLoops, np.ndarray). if inplace is True, return the sorted loops
        and the index of the sorted loops in the original loops. Otherwise,
        return None.
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
                [self.pairs12.values, self.pairs23.values, self.pairs13.values]
            ),
            "days": np.hstack([self.days12, self.days23, self.days13]),
        }
        if isinstance(order, str):
            order = [order]
        _values = []
        for i in order:
            if i not in item_map.keys():
                raise ValueError(
                    f"order should be one of {list(item_map.keys())}, but got {order}."
                )
            _values.append(item_map[i])
        _values = np.hstack(_values)
        _, _index = np.unique(_values, axis=0, return_index=True)
        if not ascending:
            _index = _index[::-1]
        if inplace:
            self._values = self._values[_index]
            self._parse_loop_meta()
        else:
            return TripletLoops(self._values[_index]), _index

    def to_seasons(self) -> NDArray[np.int8]:
        """return the season of each loop.

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

    def __init__(self, loop: Iterable[datetime]) -> None:
        """Initialize the Loop class

        loop: Iterable
            Iterable object of dates. Each date is a datetime object.
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
        return self._length

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"Loop({self._name})"

    def __eq__(self, other) -> bool:
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def values(self) -> NDArray[np.datetime64]:
        """return the values array of the loop.

        Returns
        -------
        values: np.ndarray
            dates of the loop with format of np.datetime64[D].
        """
        return self._values

    @property
    def pairs(self) -> Pairs:
        """all pairs of the loop."""
        return self._pairs

    @property
    def name(self) -> str:
        """the string format the loop.

        Returns
        -------
        name: str
            String of the loop.
        """
        return self._name

    def from_name(
        cls,
        name: str,
        parse_function: Optional[Callable] = None,
        date_args: dict = {},
    ) -> "Loop":
        """initialize the loop class from a loop name

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
        dates = DateManager.str_to_dates(name, 0, parse_function, date_args)
        return cls(dates)


class Loops:
    """Loops class to handle loops with multiple acquisitions."""

    def __init__(self, loops: list, sort=True) -> None:
        """initialize the loops class

        Parameters
        ----------
        loops: list
            a list containing Loop objects.
        sort: bool, optional
            Whether to sort the loops. Default is True.
        """
        self._loops = np.array(loops, dtype=object)
        self._parse_loop_meta()

        if sort:
            self.sort()

    def _parse_loop_meta(self) -> None:
        self._length = len(self._loops)
        self._names = np.array([i.name for i in self.loops])
        self._pairs, self._edge_pairs, self._diagonal_pairs = self._parse_pairs()

    def __str__(self) -> str:
        return f"Loops(loops={len(self)}, pairs={len(self.pairs)}, edge_pairs={len(self.edge_pairs)}, diagonal_pairs={len(self.diagonal_pairs)})"

    def __repr__(self) -> str:
        return (
            "Loops("
            f"\n    loops={len(self)},"
            f"\n    pairs={len(self.pairs)},"
            f"\n    edge_pairs={len(self.edge_pairs)},"
            f"\n    diagonal_pairs={len(self.diagonal_pairs)}"
            f"\n)"
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Loop:
        return self.loops[index]

    def __iter__(self):
        return iter(self.loops)

    def __contains__(self, item: Loop) -> bool:
        return item in self.loops

    def _parse_pairs(self) -> tuple[Pairs, Pairs, Pairs]:
        """parse the pairs in the loops

        Returns
        -------
        (Pairs, Pairs, Pairs) : all pairs, edge pairs, and diagonal pairs
        """
        pairs = []
        edge_pairs = []
        diagonal_pairs = []
        for loop in self.loops:
            pairs.extend(loop.pairs)
            edge_pairs.extend(loop.pairs[:-1])
            diagonal_pairs.extend(loop.pairs[-1:])
        return (
            Pairs(pairs, sort=True),
            Pairs(edge_pairs, sort=True),
            Pairs(diagonal_pairs, sort=True),
        )

    @property
    def loops(self) -> NDArray[np.object_]:
        """the loops in the numpy array format."""
        return self._loops

    @property
    def shape(self) -> tuple[int, int]:
        """the shape of the loops array with format of (n_loops, n_pairs)."""
        return (len(self), len(self.pairs))

    @property
    def names(self) -> NDArray[np.str_]:
        """the names (str format) of the loops."""
        return self._names

    @property
    def pairs(self) -> Pairs:
        """all pairs in the loops."""
        return self._pairs

    @property
    def edge_pairs(self) -> Pairs:
        """all edge pairs in the loops."""
        return self._edge_pairs

    @property
    def diagonal_pairs(self) -> Pairs:
        """all diagonal pairs in the loops."""
        return self._diagonal_pairs

    def sort(self, ascending: bool = True, inplace: bool = True) -> "Loops" | None:
        """sort the loops

        Parameters
        ----------
        ascending: bool, optional
            Whether to sort the loops ascending. Default is True.
        inplace: bool, optional
            Whether to sort the loops in place. if False, return the sorted loops. Default is True.
        """
        names = self.names
        _, _index = np.unique(names, return_index=True)
        if not ascending:
            _index = _index[::-1]
        if inplace:
            self._loops = self._loops[_index]
            self._parse_loop_meta()
        else:
            return Loops(self._loops[_index])

    def to_matrix(self, dtype=None) -> NDArray[np.number]:
        """return a design matrix which rows and columns are loops and pairs
        respectively. The values of the matrix are 1 for the edge pairs, -1 for
        the diagonal pairs, and 0 otherwise.
        """
        loop_pairs_ls = [i.pairs.names for i in self.loops]
        all_pairs = self.pairs.names
        matrix = np.zeros((len(self), len(self.pairs)), dtype=dtype)
        for i, loop_pairs in enumerate(loop_pairs_ls):
            matrix[i][np.in1d(all_pairs, loop_pairs[:-1])] = 1
            matrix[i][np.in1d(all_pairs, loop_pairs[-1])] = -1
        return matrix


class SBASNetwork:
    """SBAS network class to handle pairs and loops.

    Examples
    --------

    initialize SBASNetwork class from Pairs object named pairs

    >>> sbas = SBASNetwork(pairs)

    get pairs that are not in the loops

    >>> pairs_alone = sbas.pairs - sbas.loops.pairs
    """

    _pairs: Pairs
    _loops: TripletLoops
    _baseline: np.ndarray

    __slots__ = ["_pairs", "_loops", "_baseline"]

    def __init__(
        self,
        pairs: Iterable[Iterable[datetime, datetime]] | Iterable[Pair],
        Pairs,
        baseline: Optional[Iterable[float]] = None,
        sort: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        pairs : Iterable or Pairs
            Iterable object of pairs. Each pair is an Iterable object of two dates
            or a Pair object, or Pairs object.
        baseline : Iterable, optional
            Baseline of each pair in meters. Default is None.
        sort : bool, optional
            Whether to sort the pairs and baseline. Default is True.

        Examples
        --------
        >>> from fancy_sar import SBASNetwork

        """

        if isinstance(pairs, Pairs):
            self._pairs = pairs
        elif isinstance(pairs, Iterable):
            self._pairs = Pairs(pairs, sort=False)
        else:
            raise TypeError(
                f"pairs should be an Iterable or Pairs object, but got {type(pairs)}."
            )
        self.baseline = baseline
        self._loops = self._pairs.to_loops()
        if sort:
            self.sort(inplace=True)

    @property
    def pairs(self) -> Pairs:
        """return the pairs of the network

        Returns
        -------
        pairs: Pairs
            Pairs of the network.
        """
        return self._pairs

    @property
    def loops(self) -> TripletLoops:
        """return the loops of the network

        Returns
        -------
        loops: TripletLoops
            TripletLoops of the network.
        """
        return self._loops

    @property
    def baseline(self) -> Optional[np.ndarray]:
        """return the baseline of the network

        Returns
        -------
        baseline: np.ndarray or None
            Baseline of the network.
        """
        return self._baseline

    @baseline.setter
    def baseline(self, value: Optional[Iterable]):
        """set the baseline of the network

        Parameters
        ----------
        value: Iterable or None
            Baseline of the network.
        """
        if value is None:
            self._baseline = None
            return
        elif isinstance(value, Iterable):
            value = np.asarray(value)
        else:
            raise TypeError(f"baseline should be an Iterable, but got {type(value)}.")

        if len(self._pairs) != len(value):
            raise ValueError(
                f"Length of pairs {len(self._pairs)} should be equal to length of baseline {len(value)}."
            )

        self._baseline = value

    @property
    def dates(self) -> list[str]:
        return self._pairs.dates

    @classmethod
    def from_names(
        cls,
        names: list[str],
        parse_function: Optional[Callable] = None,
        date_args: dict = None,
    ) -> "SBASNetwork":
        """initialize SBASNetwork class from a list of pair names

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
            Default is None.

        Returns
        -------
        SBASNetwork: SBASNetwork
            SBASNetwork object.
        """
        pairs = Pairs.from_names(names, parse_function, date_args)
        return cls(pairs)

    def sort(
        self,
        order: list | Literal["pairs", "primary", "secondary", "days"] = "pairs",
        ascending: bool = True,
        inplace: bool = True,
    ):
        """sort the pairs and corresponding baseline of the network

        Parameters
        ----------
        order: str or list of str, optional
            By which fields to sort the loops. This argument specifies
            which fields to compare first, second, etc. Default is 'pairs'.

            The available options are one or a list of:
            ['pairs', 'primary', 'secondary', 'days'].
        ascending: bool, optional
            Whether to sort ascending. Default is True.
        inplace: bool, optional
            Whether to sort the pairs and loops inplace. Default is True.

        Returns
        -------
        None or (SBASNetwork, np.ndarray). if inplace is True, return the sorted
        SBASNetwork and the index of the sorted pairs in the original pairs.
        Otherwise, return None.
        """
        pairs, _index = self._pairs.sort(order, ascending, inplace=False)
        if self.baseline is not None:
            self.baseline = self.baseline[_index]
        if inplace:
            self._pairs = pairs
            self._loops = pairs.to_loops()
        else:
            return SBASNetwork(pairs, self.baseline), _index

    def plot(
        self,
        fname: Optional[str | Path] = None,
        ax: Optional[plt.Axes] = None,
        dpi: Optional[int] = None,
    ):
        """plot the network"""
        # TODO: plot network
        pass


class PairsFactory:
    """This class is used to generate interferometric pairs for InSAR processing."""

    def __init__(self, dates: Iterable, **kwargs) -> None:
        """initialize the PairGenerator class

        Parameters
        ----------
        dates: Iterable
            Iterable object that contains the dates. Can be any object that
            accepted by :func:`pd.to_datetime`
        date_args: dict, optional
            Keyword arguments passed to :func:`pd.to_datetime`
        """
        self.dates = pd.to_datetime(dates, **kwargs).unique().sort_values()

    def from_interval(self, max_interval: int = 2, max_day: int = 180) -> Pairs:
        """generate interferometric pairs by SAR acquisition interval. SAR
        acquisition interval is defined as the number of SAR acquisitions
        between two SAR acquisitions.

        **For example:**

        if the SAR acquisition interval is 2, then the interferometric pairs
        will be generated between SAR acquisitions with interval of 1 and 2.
        This will be useful to generate interferometric pairs with different
        temporal baselines.

        Parameters
        ----------
        max_interval: int
            max interval between two SAR acquisitions for interferometric pair.
            interval is defined as the number of SAR acquisitions between two SAR acquisitions.
        max_day:int
            max day between two SAR acquisitions for interferometric pair

        Returns
        -------
        pairs: Pairs object
        """
        num = len(self.dates)
        _pairs = []
        for i, date in enumerate(self.dates):
            n_interval = 1
            while n_interval <= max_interval:
                if i + n_interval < num:
                    if (self.dates[i + n_interval] - date).days < max_day:
                        pair = (date, self.dates[i + n_interval])
                        _pairs.append(pair)
                        n_interval += 1
                    else:
                        break
                else:
                    break

        return Pairs(_pairs)

    def linking_winter(
        self,
        winter_start: str = "0101",
        winter_end: str = "0331",
        n_per_winter: int = 5,
        max_winter_interval: int = 1,
    ) -> Pairs:
        """generate interferometric pairs linking winter in each year. winter
        is defined by month and day for each year. For example, winter_start='0101',
        winter_end='0331' means the winter is from Jan 1 to Mar 31 for each year in
        the time series. This will be useful to add pairs for completely frozen
        period across years in permafrost region.

        Parameters
        ----------
        winter_start, winter_end:  str
            start and end date for the winter which expressed as month and day
            with format '%m%d'
        n_per_winter: int
            how many dates will be used for each winter. Those dates will be
            selected randomly in each winter. Default is 5
        max_winter_interval: int
            max interval between winters for interferometric pair. If
            max_winter_interval=1, hen the interferometric pairs will be generated
            between neighboring winters.

        Returns
        -------
        pairs: Pairs object
        """
        years = sorted(set(self.dates.year))
        df_dates = pd.Series(self.dates, index=self.dates)

        # check if period_start and period_end are in the same year. If not,
        # the period_end should be in the next year
        same_year = True if int(winter_start) < int(winter_end) else False

        # randomly select n_per_period dates in each period/year
        date_years = []
        for year in years:
            start = pd.to_datetime(f"{year}{winter_start}", format="%Y%m%d")
            if same_year:
                end = pd.to_datetime(f"{year}{winter_end}", format="%Y%m%d")
            else:
                end = pd.to_datetime(f"{year+1}{winter_end}", format="%Y%m%d")

            dt_year = df_dates[start:end]
            if len(dt_year) > 0:
                np.random.shuffle(dt_year)
                date_years.append(dt_year[:n_per_winter].to_list())

        n_years = len(date_years)

        _pairs = []
        for i, date_year in enumerate(date_years):
            # primary/reference dates
            for date_primary in date_year:
                # secondary dates
                for j in range(1, max_winter_interval + 1):
                    if i + j < n_years:
                        for date_secondary in date_years[i + j]:
                            _pairs.append((date_primary, date_secondary))
        return Pairs(_pairs)

    def from_period(
        self,
        period_start: str = "0101",
        period_end: str = "0331",
        n_per_period: int | None = None,
        n_primary_period: Optional[str] = None,
        primary_years: Optional[list[int]] = None,
    ) -> Pairs:
        """generate interferometric pairs between periods for all years.
        period is defined by month and day for each year. For example,
        period_start='0101', period_end='0331' means the period is from Dec 1
        to Mar 31 for each year in the time series. This function will randomly
        select n_per_period dates in each period and generate interferometric
        pairs between those dates. This will be useful to mitigate the temporal
        cumulative bias.

        Parameters
        ----------
        period_start, period_end:  str
            start and end date for the period which expressed as month and day
            with format '%m%d'
        n_per_period: int | None
            how many dates will be used for each period. Those dates will be
            selected randomly in each period. If None, all dates in the period
            will be used. Default is None.
        n_primary_period: int, optional
            how many periods/years used as primary date of ifg. For example, if
            n_primary_period=2, then the interferometric pairs will be generated
            between the first two periods and the rest periods. If None, all
            periods will be used. Default is None.
        primary_years: list, optional
            years used as primary date of ifg. If None, all years in the time
            series will be used. Default is None.

        Returns
        -------
        pairs: Pairs object
        """
        years = sorted(set(self.dates.year))
        df_dates = pd.Series(self.dates, index=self.dates)

        # check if period_start and period_end are in the same year. If not,
        # the period_end should be in the next year
        same_year = True if int(period_start) < int(period_end) else False

        # randomly select n_per_period dates in each period/year
        date_years = []
        for year in years:
            start = pd.to_datetime(f"{year}{period_start}", format="%Y%m%d")
            if same_year:
                end = pd.to_datetime(f"{year}{period_end}", format="%Y%m%d")
            else:
                end = pd.to_datetime(f"{year+1}{period_end}", format="%Y%m%d")

            dt_year = df_dates[start:end]
            if (n_year := len(dt_year)) > 0:
                n = n_year if n_per_period is None else n_per_period
                np.random.shuffle(dt_year)
                date_years.append(dt_year[:n].to_list())

        # generate interferometric pairs between primary period and the rest periods
        _pairs = []
        for i, date_year in enumerate(date_years):
            # only generate pairs for n_primary_period
            if n_primary_period is not None and i + 1 > n_primary_period:
                break
            if primary_years is not None and years[i] not in primary_years:
                continue
            for date_primary in date_year:
                # all rest periods
                for date_year1 in date_years[i + 1 :]:
                    for date_secondary in date_year1:
                        pair = (date_primary, date_secondary)
                        _pairs.append(pair)

        return Pairs(_pairs)

    def from_summer_winter(
        self,
        summer_start: str = "0801",
        summer_end: str = "1001",
        winter_start: str = "1201",
        winter_end: str = "0331",
    ) -> Pairs:
        """generate interferometric pairs between summer and winter in each yea. summer and winter
        are defined by month and day for each year. For example, summer_start='0801', summer_end='1001'
        means the summer is from Aug 1 to Oct 1 for each year in the time series. This will be useful
        to add pairs for whole thawing and freezing process.

        Parameters
        ---------
        summer_start, summer_end:  str
            start and end date for the summer which expressed as month and day with format '%m%d'
        winter_start, winter_end:  str
            start and end date for the winter which expressed as month and day with format '%m%d'

        Returns
        -------
        Pairs object
        """
        years = sorted(set(self.dates.year))
        df_dates = pd.Series(self.dates.strftime("%Y%m%d"), index=self.dates)

        _pairs = []
        for year in years:
            s_start = pd.to_datetime(f"{year}{summer_start}", format="%Y%m%d")
            s_end = pd.to_datetime(f"{year}{summer_end}", format="%Y%m%d")

            if int(winter_start) > int(summer_end):
                w_start1 = pd.to_datetime(f"{year-1}{winter_start}", format="%Y%m%d")
                w_start2 = pd.to_datetime(f"{year}{winter_start}", format="%Y%m%d")
                if int(winter_end) > int(summer_end):
                    w_end1 = pd.to_datetime(f"{year-1}{winter_end}", format="%Y%m%d")
                    w_end2 = pd.to_datetime(f"{year}{winter_end}", format="%Y%m%d")
                else:
                    w_end1 = pd.to_datetime(f"{year}{winter_end}", format="%Y%m%d")
                    w_end2 = pd.to_datetime(f"{year+1}{winter_end}", format="%Y%m%d")
            else:
                w_start1 = pd.to_datetime(f"{year}{winter_start}", format="%Y%m%d")
                w_start2 = pd.to_datetime(f"{year+1}{winter_start}", format="%Y%m%d")

                w_end1 = pd.to_datetime(f"{year}{winter_end}", format="%Y%m%d")
                w_end2 = pd.to_datetime(f"{year+1}{winter_end}", format="%Y%m%d")

            dt_winter1 = df_dates[w_start1:w_end1].to_list()
            dt_summer = df_dates[s_start:s_end].to_list()
            dt_winter2 = df_dates[w_start2:w_end2].to_list()

            # thawing process
            if len(dt_winter1) > 0 and len(dt_summer) > 0:
                for dt_w1 in dt_winter1:
                    for dt_s in dt_summer:
                        pair = (dt_w1, dt_s)
                        _pairs.append(pair)
            # freezing process
            if len(dt_winter2) > 0 and len(dt_summer) > 0:
                for dt_w2 in dt_winter2:
                    for dt_s in dt_summer:
                        pair = (dt_s, dt_w2)
                        _pairs.append(pair)

        return Pairs(_pairs)


class DateManager:
    # TODO: user defined period (using day of year, wrap into new year and cut into periods)
    def __init__(self) -> None:
        pass

    @staticmethod
    def season_of_month(month: int) -> int:
        """return the season of a given month

        Parameters
        ----------
        month: int
            Month of the year.

        Returns
        -------
        season: int
            Season of corresponding month:
                1 for spring,
                2 for summer,
                3 for fall,
                4 for winter.
        """
        month = int(month)
        if month not in list(range(1, 13)):
            raise ValueError("Month should be in range 1-12." f" But got '{month}'.")
        season = (month - 3) % 12 // 3 + 1
        return season

    @staticmethod
    def ensure_datetime(date: Any) -> datetime:
        """ensure the date is a datetime object

        Parameters
        ----------
        date: datetime or str
            Date to be ensured.

        Returns
        -------
        date: datetime
            Date with format of datetime.
        """
        if isinstance(date, datetime):
            pass
        elif isinstance(date, str):
            date = pd.to_datetime(date)
        else:
            raise TypeError(f"Date should be datetime or str, but got {type(date)}")
        return date

    @staticmethod
    def str_to_dates(
        date_str: str,
        length: Optional[int] = 2,
        parse_function: Optional[Callable] = None,
        date_args: dict = {},
    ):
        """convert date string to dates

        Parameters
        ----------
        date_str: str
            Date string containing dates.
        length: int, optional
            Length/number of dates in the date string. if 0, all dates in the
            date string will be used. Default is 2.
        parse_function: Callable, optional
            Function to parse the date strings from the date string.
            If None, the date string will be split by '_' and the
            last 2 items will be used. Default is None.
        date_args: dict, optional
            Keyword arguments for :func:`pd.to_datetime` to convert the date strings
            to datetime objects. For example, {'format': '%Y%m%d'}.
            Default is {}.
        """
        if parse_function is not None:
            dates = parse_function(date_str)
        else:
            items = date_str.split("_")
            if len(items) >= length:
                dates_ls = items[-length:]
            else:
                raise ValueError(
                    f"The number of dates in {date_str} is less than {length}."
                )

        date_args.update({"errors": "raise"})
        try:
            dates = [pd.to_datetime(i, **date_args) for i in dates_ls]
        except:
            raise ValueError(f"Dates in {date_str} not recognized.")

        return tuple(dates)


def find_loops(
    loops_pairs: Pairs,
    loops: list[Loop],
    loop: Loop,
    pairs_left: Pairs,
    end_date: datetime,
    edge_pairs: Pairs,
    max_acquisition=5,
) -> None:
    """recursively find all available loops within pairs_left and end_date."""
    for pair_left in pairs_left:
        if pair_left not in edge_pairs:
            continue
        m_candidate = (
            (loops_pairs.primary == pair_left.secondary)
            & (loops_pairs.secondary <= end_date)
            & (loops_pairs.where(edge_pairs, return_type="mask"))
        )
        if not m_candidate.any():
            continue
        pairs_candidate = loops_pairs[m_candidate]
        for pair_middle in pairs_candidate:
            loop_i = loop.copy()
            loop_i.append(pair_middle.primary)
            if pair_middle.secondary == end_date:
                loop_i.append(pair_middle.secondary)
                loops.append(Loop(loop_i))
            else:
                # +1 for diagonal pair
                if len(loop_i) + 1 < max_acquisition:
                    find_loops(
                        loops_pairs,
                        loops,
                        loop_i,
                        pairs_candidate,
                        end_date,
                        edge_pairs,
                        max_acquisition,
                    )


def valid_diagonal_pair(
    pair: Pair, pairs: Pairs, edge_pairs: Optional[int] = None
) -> bool:
    """check if the pair is a valid diagonal pair that can be used to form a loop
    using the edge pairs. This function is just a simple check to see if days of
    diagonal pair is less than the sum of edge pairs.

    Parameters
    ----------
    pair: Pair
        Pair to be checked if it is a valid diagonal pair.
    pairs: Pairs
        All pairs of the loop.
    edge_pairs: int, optional
        The edge pairs to form loops.

    Returns
    -------
    valid: bool
        Whether the pair is a valid diagonal pair.
    """
    valid = False
    start_date, end_date = pair.values
    mask_edge = (
        (pairs.primary >= start_date)
        & (pairs.secondary <= end_date)
        & (pairs.where(edge_pairs, return_type="mask"))
    )

    if not mask_edge.any():
        return False

    _edge_pairs = pairs[mask_edge]
    if _edge_pairs.days.sum() >= pair.days:
        valid = True
    return valid


if __name__ == "__main__":
    names = [
        "20170111_20170204",
        "20170111_20170222",
        "20170111_20170318",
        "20170204_20170222",
        "20170204_20170318",
        "20170204_20170330",
        "20170222_20170318",
        "20170222_20170330",
        "20170222_20170411",
        "20170318_20170330",
    ]

    pairs = Pairs.from_names(names)
    loops = pairs.to_loops()
    # pairs1 = loops.pairs
