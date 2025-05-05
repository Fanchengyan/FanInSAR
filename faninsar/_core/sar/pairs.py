"""Pair and loop tools for InSAR time series analysis."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Literal, overload

import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.indexes import Indexes

from faninsar._core.render import formatting_html
from faninsar.logging import setup_logger

from .acquisition import Acquisition, DateManager, DaySpan

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

    from faninsar import Loop, Loops, TripletLoops
    from faninsar.typing import PairLike, PairsLike, PairsOrder

logger = setup_logger(log_name=__name__)


class Pair:
    """Pair class for one pair."""

    _values: np.ndarray
    _name: str
    _days: int

    __slots__ = ["_days", "_name", "_values"]

    def __init__(
        self,
        pair: Sequence[datetime, datetime],
    ) -> None:
        """Initialize the Pair class.

        Parameters
        ----------
        pair: Sequence[datetime, datetime]
            Sequence object of two dates. Each date is a datetime object.
            For example, (date1, date2).

        """
        self._values = np.sort(pair).astype("M8[D]")
        self._name = "_".join(
            [i.strftime("%Y%m%d") for i in self._values.astype("O")],
        )
        self._days = (self._values[1] - self._values[0]).astype(int)

    def __str__(self) -> str:
        """Return the name of the pair."""
        return self._name

    def __repr__(self) -> str:
        """Return the representation of the pair."""
        return f"Pair({self._name})"

    def __eq__(self, other: Pair) -> bool:
        """Compare the pair with another pair."""
        return self.name == other.name

    def __hash__(self) -> int:
        """Return the hash of the pair."""
        return hash(self._name)

    def __array__(self, dtype: DTypeLike = None) -> np.ndarray:
        """Return the pair as a numpy array."""
        return np.asarray(self._values, dtype=dtype)

    @property
    def values(self) -> NDArray[np.datetime64]:
        """The values of the pair with format of datetime."""
        return self._values

    @property
    def name(self) -> str:
        """String of the pair with format of '%Y%m%d_%Y%m%d'."""
        return self._name

    @property
    def days(self) -> int:
        """The time span of the pair in days."""
        return self._days

    @property
    def primary(self) -> pd.Timestamp:
        """The primary dates of all pairs."""
        return pd.Timestamp(self.values[0])

    @property
    def secondary(self) -> pd.Timestamp:
        """The secondary dates of all pairs."""
        return pd.Timestamp(self.values[1])

    def primary_string(self, date_format: str = "%Y%m%d") -> str:
        """Return the primary dates of all pairs in string format.

        Parameters
        ----------
        date_format: str
            Format of the date string. Default is '%Y%m%d'. See more at
            `strftime Format Codes <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_.

        """
        return pd.to_datetime(self.values[0]).strftime(date_format)

    def secondary_string(self, date_format: str = "%Y%m%d") -> str:
        """Return the secondary dates of all pairs in string format.

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
        parse_function: Callable | None = None,
        date_args: dict | None = None,
    ) -> Pair:
        """Initialize the pair class from a pair name.

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
        if date_args is None:
            date_args = {}
        dates = DateManager.str_to_dates(name, 2, parse_function, date_args)
        return cls(dates)

    def to_numpy(self, dtype: DTypeLike = None) -> NDArray[np.datetime64]:
        """Return the pair as a numpy array."""
        return np.asarray(self._values, dtype=dtype)


class Pairs:
    """Pairs class to handle pairs.

    .. note::
        - Each pair will be sorted in the acquisition order. The primary date
          will be always earlier than the secondary date.
        - The pairs will be sorted by the primary date.



    Examples
    --------
    prepare dates and pairs for examples:

    >>> dates = pd.date_range("20130101", "20231231").values
    >>> n = len(dates)
    >>> pair_ls = []
    >>> loop_ls = []
    >>> for i in range(5):
    ...     np.random.seed(i)
    ...     pair_ls.append(dates[np.random.randint(0, n, 2)])

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

    >>> pairs1 = pairs["2018-03-09":]
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
    _edge_index: np.ndarray
    _names: np.ndarray

    __slots__ = ["_dates", "_edge_index", "_length", "_names", "_values"]

    def __init__(
        self,
        pairs: Sequence[Sequence[datetime, datetime]] | Sequence[Pair],
        sort: bool = True,
    ) -> None:
        """Initialize the pairs class.

        Parameters
        ----------
        pairs: Sequence
            Sequence object of pairs. Each pair is an Sequence or Pair
            object of two dates with format of datetime. For example,
            [(date1, date2), ...].
        sort: bool, optional
            Whether to sort the pairs. Default is True.

        """
        if pairs is None:
            pairs = []

        _values = np.array(pairs, dtype="M8[D]")

        self._values = self._format_pair_values(_values)
        self._parse_pair_meta()

        if sort:
            self.sort(inplace=True)

    def _format_pair_values(
        self,
        values: NDArray[np.datetime64],
    ) -> NDArray[np.datetime64]:
        """Format the pair values.

        This function is used to make sure the primary date is earlier than the
        secondary date.
        """
        if values.ndim == 1:
            values = values.reshape(-1, 2)
        idx = values[:, 0] > values[:, 1]
        values[idx] = values[idx, ::-1]
        return values

    def _parse_pair_meta(self) -> None:
        self._dates = np.unique(self._values.flatten())
        self._length = self._values.shape[0]
        self._edge_index = np.searchsorted(self._dates, self._values)
        self._names = self.to_names()

    def __len__(self) -> int:
        """Return the number of pairs."""
        return self._length

    def __str__(self) -> str:
        """Return the string representation of the pairs."""
        return f"Pairs({self._length})"

    def __repr__(self) -> str:
        """Return the representation of the pairs."""
        return self._to_frame().__repr__()

    def _repr_html_(self) -> str:
        """Return the HTML representation of the pairs."""
        return formatting_html.pairs_repr(self)

    def __eq__(self, other: Pairs) -> bool:
        """Compare the pairs with another pairs."""
        return np.array_equal(self.values, other.values)

    def __add__(self, other: Pairs) -> Pairs:
        """Return the unique, sorted union of the pairs."""
        _pairs = np.union1d(self.names, other.names)
        return Pairs.from_names(_pairs)

    def __sub__(self, other: Pairs) -> Pairs:
        """Return the difference of the pairs."""
        _pairs = np.setdiff1d(self.names, other.names)

        return Pairs.from_names(_pairs)

    @overload
    def __getitem__(self, index: int | np.integer) -> Pair: ...

    @overload
    def __getitem__(self, index: slice) -> Pairs | None: ...

    def __getitem__(  # noqa: PLR0911, PLR0912
        self,
        index: int | slice | datetime | str | PairLike | Iterable,
    ) -> Pair | Pairs | None:
        """Return the pair or pairs by index or slice."""
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
                return Pairs(self._values[start:stop:step])
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
                    raise ValueError(msg)
                _pairs = []
                for pair in self._values:
                    pair = pair.astype("M8[s]")  # noqa: PLW2901
                    if start <= pair[0] <= stop and start <= pair[1] <= stop:
                        _pairs.append(pair)
                if len(_pairs) > 0:
                    return Pairs(_pairs)
                return None
            msg = f"Unsupported slice index {index} for Pairs."
            logger.warning(msg)
            return None
        if isinstance(index, (int, np.integer)):
            if index >= self._length:
                msg = f"Index {index} out of range. Pairs number is {self._length}."
                raise IndexError(
                    msg,
                )
            return Pair(self._values[index])
        if isinstance(index, (datetime, np.datetime64, pd.Timestamp, str)):
            if isinstance(index, str):
                try:
                    index = pd.to_datetime(index)
                except Exception as e:
                    msg = f"String {index} cannot be converted to datetime."
                    raise ValueError(msg) from e
            pairs = [pair for pair in self._values if index in pair]
            if len(pairs) > 0:
                return Pairs(pairs)
            return None
        if isinstance(index, Iterable):
            index = np.array(index)
            return Pairs(self._values[index])
        msg = (
            "Index should be int, slice, datetime, str, or bool or int array"
            f" indexing, but got {type(index)}."
        )
        raise TypeError(
            msg,
        )

    def __hash__(self) -> int:
        """Return the hash of the pairs."""
        return hash("".join(self.names))

    def __iter__(self) -> iter[Pair]:
        """Iterate the pairs."""
        pairs_ls = [Pair(i) for i in self._values]
        return iter(pairs_ls)

    def __array__(self, dtype: DTypeLike = None) -> NDArray:
        """Return the pairs as a numpy array."""
        return np.asarray(self._values, dtype=dtype)

    def _to_frame(self) -> pd.DataFrame:
        """Return the pairs as a DataFrame."""
        pairs_str = "Pairs"
        columns = pd.MultiIndex.from_tuples(
            [(pairs_str, "primary"), (pairs_str, "secondary")],
        )
        return pd.DataFrame(self._values, columns=columns)

    def _ensure_pairs(
        self,
        pairs: str | Pair | Pairs | Sequence[str] | Sequence[Pair],
    ) -> Pairs:
        """Ensure the pairs are in the Pairs object."""
        if isinstance(pairs, str):
            pairs = Pairs.from_names([pairs])
        elif isinstance(pairs, Pair):
            pairs = Pairs([pairs])
        elif isinstance(pairs, Pairs):
            return pairs
        elif isinstance(pairs, Sequence):
            pairs = np.asarray(pairs)
            if pairs.ndim == 1:
                pairs = Pairs.from_names(pairs)
            elif pairs.ndim == 2 and pairs.shape[1] == 2:
                pairs = Pairs(pairs)
            else:
                msg = (
                    "pairs should be 1D array of str, 2D array of datetime, "
                    f"or Pairs, but got {pairs}."
                )
                raise ValueError(msg)
        else:
            msg = (
                "pairs should be str, Pair, list of str, list of Pair, "
                f"or Pairs, but got {type(pairs)}."
            )
            raise TypeError(msg)
        return pairs

    @property
    def values(self) -> NDArray[np.datetime64]:
        """The numpy array of the pairs."""
        return self._values

    @property
    def names(self) -> NDArray[np.str_]:
        """The names (string format) of the pairs."""
        return self._names

    @property
    def dates(self) -> Acquisition:
        """The sorted dates array of all pairs in type of np.datetime64[D]."""
        return Acquisition(pd.to_datetime(self._dates))

    @property
    def days(self) -> DaySpan:
        """The time span of all pairs in days."""
        days = (self._values[:, 1] - self._values[:, 0]).astype(int)
        return DaySpan(days, dtype=np.int32)

    @property
    def primary(self) -> Acquisition:
        """The primary dates of all pairs."""
        if len(self._values) == 0:
            return Acquisition([])
        return Acquisition(pd.to_datetime(self._values[:, 0]))

    @property
    def secondary(self) -> Acquisition:
        """The secondary dates of all pairs."""
        if len(self._values) == 0:
            return Acquisition([])
        return Acquisition(pd.to_datetime(self._values[:, 1]))

    @property
    def xindexes(self) -> Indexes:
        """The xarray indexes of the pairs."""
        indexes = {
            "primary": self.primary,
            "secondary": self.secondary,
            "days": self.days,
            "dates": self.dates,
        }
        variables = {
            "primary": self.primary.to_xarray(),
            "secondary": self.secondary.to_xarray(),
            "days": self.days.to_xarray(),
            "dates": self.dates.to_xarray(),
        }
        return Indexes(indexes=indexes, variables=variables, index_type=pd.Index)

    def primary_string(self, date_format: str = "%Y%m%d") -> pd.Index:
        """Return the primary dates of all pairs in string format.

        Parameters
        ----------
        date_format: str
            Format of the date string. Default is '%Y%m%d'. See more at
            `strftime Format Codes <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_.

        """
        return self.primary.strftime(date_format)

    def secondary_string(self, date_format: str = "%Y%m%d") -> pd.Index:
        """Return the secondary dates of all pairs in string format.

        Parameters
        ----------
        date_format: str
            Format of the date string. Default is '%Y%m%d'. See more at
            `strftime Format Codes <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_.

        """
        return self.secondary.strftime(date_format)

    @property
    def edge_index(self) -> NDArray[np.int64]:
        """The index of the pairs in the dates coordinate.

        This is useful to construct the edge index in graph theory.
        """
        return self._edge_index

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the pairs array."""
        return self._values.shape

    @classmethod
    def from_names(
        cls,
        names: Sequence[str | None],
        parse_function: Callable | None = None,
        date_args: dict | None = None,
    ) -> Pairs:
        """Initialize the pair class from a pair name.

        .. note::
            The pairs will be in the order of the input names. If you want to
            sort the pairs, you can call the :meth:`Pairs.sort()` method to
            achieve it.

        Parameters
        ----------
        names: str
            Pair names.
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
        if date_args is None:
            date_args = {}
        pairs = []
        # No names in list
        if len(names) == 0:
            return cls(pairs)
        # Parse the names
        for name in names:
            pair = Pair.from_name(name, parse_function, date_args)
            pairs.append(pair.values)
        return cls(pairs, sort=False)

    def where(
        self,
        pairs: list[str] | list[Pair] | Pairs,
        return_type: Literal["index", "mask"] = "index",
    ) -> NDArray[np.int64 | np.bool_]:
        """Return the index of the pairs.

        Parameters
        ----------
        pairs: list of str or Pair, or Pairs
            Pair names or Pair objects, or Pairs object.
        return_type: str, optional
            Whether to return the index or mask of the pairs. Default is 'index'.

        """
        if return_type not in ["index", "mask"]:
            msg = (
                "return_type should be one of ['index', 'mask'], "
                f"but got {return_type}."
            )
            raise ValueError(msg)

        pairs = self._ensure_pairs(pairs)
        con = np.isin(self.names, pairs.names)
        if return_type == "index":
            con = np.where(con)[0]
        return con

    def intersect(self, pairs: PairLike) -> Pairs | None:
        """Return the intersection of the pairs.

        The pairs both in self and input pairs.

        Parameters
        ----------
        pairs: list of str or Pair, or Pairs
            Pair names or Pair objects, or Pairs object.

        """
        pairs = self._ensure_pairs(pairs)
        return self[self.where(pairs)]

    def union(self, pairs: list[str] | list[Pair] | Pairs) -> Pairs:
        """Return the unique, sorted union of the pairs.

        All pairs that in self and input pairs. Same as addition.

        Parameters
        ----------
        pairs: list of str or Pair, or Pairs
            Pair names or Pair objects, or Pairs object.

        """
        pairs = self._ensure_pairs(pairs)
        return self + pairs

    def difference(self, pairs: PairsLike) -> Pairs | None:
        """Return the difference of the pairs.

        The pairs in self but not in pairs. Same as subtraction.

        Parameters
        ----------
        pairs: list of str or Pair, or Pairs
            Pair names or Pair objects, or Pairs object.

        """
        pairs = self._ensure_pairs(pairs)
        return self - pairs

    def copy(self) -> Pairs:
        """Return a copy of the pairs."""
        return Pairs(self._values.copy())

    def sort(
        self,
        order: PairsOrder = "pairs",
        ascending: bool = True,
        inplace: bool = True,
    ) -> tuple[Pairs, NDArray[np.int64]] | None:
        """Sort the pairs.

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
        # for null pairs
        if len(self) == 0:
            if inplace:
                return None
            return self

        # for valid pairs
        item_map = {
            "pairs": self._values,
            "primary": self._values[:, 0],
            "secondary": self._values[:, 1],
            "days": self.days.data,
        }
        if isinstance(order, str):
            order = [order]
        _values_ = []
        for i in order:
            if i not in item_map:
                msg = (
                    f"order should be one of {list(item_map.keys())}, but got {order}."
                )
                raise ValueError(
                    msg,
                )
            _values_.append(item_map[i].reshape(self._length, -1))
        _values_ = np.hstack(_values_)
        _values, _index = np.unique(_values_, axis=0, return_index=True)
        if not ascending:
            _index = _index[::-1]
        if inplace:
            self._values = _values
            self._parse_pair_meta()
            return None
        return Pairs(_values), _index

    def to_names(self, prefix: str | None = None) -> NDArray[np.str_]:
        """Generate pairs names string with prefix.

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

        return names.to_numpy(dtype=np.str_)

    def to_numpy(self, dtype: DTypeLike = None) -> NDArray[np.datetime64]:
        """Return the pairs as a numpy array."""
        return np.asarray(self._values, dtype=dtype)

    def to_frame(self) -> pd.DataFrame:
        """Return the pairs as a DataFrame."""
        frame = pd.DataFrame()
        frame["primary"] = self.primary
        frame["secondary"] = self.secondary
        frame["days"] = self.days
        return frame

    def to_xarray(self) -> pd.DataFrame:
        """Return the pairs as a xarray DataArray."""
        return xr.Variable(dims=["pairs", "primary-secondary"], data=self._values)

    def to_triplet_loops(self) -> TripletLoops:
        """Return all possible triplet loops from the pairs."""
        from faninsar import TripletLoops

        loops = []
        for i, pair12 in enumerate(self._values):
            for pair23 in self._values[i + 1 :]:
                if pair12[1] == pair23[0] and Pair([pair12[0], pair23[1]]) in self:
                    loops.append([pair12[0], pair12[1], pair23[1]])  # noqa: PERF401
        return TripletLoops(loops)

    def to_loops(
        self,
        max_acquisition: int = 5,
        max_days: int | None = None,
        edge_pairs: Pairs | None = None,
        edge_days: int | None = None,
    ) -> Loops:
        r"""Return all possible loops from the pairs.

        .. important::
            The pairs in the loops may be fewer than the input pairs. You can use
            the :meth:`Pairs.where` method to get the index/mask of the pairs
            in the loops from the input pairs.

            **Example**:

            >>> loops = pairs.to_loops()
            >>> mask = pairs.where(loops.pairs, return_type="mask")

        Parameters
        ----------
        max_acquisition: int
            The maximum number of acquisitions in the loops. It should be at least 3.

            .. note::
                The number of acquisitions is equal to the number of intervals + 1
                :math:`n (acquisition) = n (edge\ pairs) + 1 (diagonal\ pair) =
                n (intervals) + 1`.
        max_days: int, optional
            The maximum number of days for the pairs in the loops. If None, all
            available pairs will be used. Default is None.
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
                msg = "Either edge_days or edge_pairs should be provided."
                raise ValueError(msg)
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
                self,
                loops,
                loop,
                pairs_primary,
                end_date,
                edge_pairs,
                max_acquisition,
            )
        from faninsar import Loops

        return Loops(loops)

    def to_matrix(self, dtype: DTypeLike = None) -> NDArray[np.number]:
        """Return the SBAS matrix.

        Parameters
        ----------
        matrix: np.ndarray
            SBAS matrix in shape of (n_pairs, n_dates-1). The dates between
            pairs are set to 1, otherwise 0.
        dtype: np.dtype, optional
            Data type of the matrix. Default is None.

        """
        matrix = np.zeros((len(self), len(self.dates) - 1), dtype=dtype)
        col_idxs = self.edge_index.copy()
        for row_idx, col_idx in enumerate(col_idxs):
            matrix[row_idx, col_idx[0] : col_idx[1]] = 1

        return matrix

    def parse_gaps(self, pairs_removed: Pairs | None = None) -> pd.DatetimeIndex:
        """Parse network gaps where the acquisitions are not connected by pairs.

        The gaps are detected by the dates that are not present in the secondary
        acquisition of the pairs.

        .. note::
            Theoretically, the gaps should be the temporal spans  (or intervals)
            between the consecutive acquisitions. For simplicity, the end dates
            of the gaps are returned here.

        Parameters
        ----------
        pairs_removed: Pairs, optional
            Pairs that are removed from the original pairs. Default is None,
            which means all pairs are used.

        Returns
        -------
        gaps : pd.DatetimeIndex
            Acquisition/date gaps that are not covered by any pairs.

        """
        dates = self.dates[1:]
        pairs_valid = self - pairs_removed if pairs_removed is not None else self

        dates_secondary = np.unique(pairs_valid.secondary)
        return np.setdiff1d(dates, dates_secondary)


class PairsFactory:
    """A class used to generate interferometric pairs for InSAR processing."""

    def __init__(self, dates: Sequence, **kwargs) -> None:
        """Initialize the PairGenerator class.

        Parameters
        ----------
        dates: Sequence
            Sequence object that contains the dates. Can be any object that
            accepted by :func:`pd.to_datetime`
        kwargs: dict, optional
            Keyword arguments passed to :func:`pd.to_datetime`

        """
        self.dates = pd.to_datetime(dates, **kwargs).unique().sort_values()

    def from_interval(self, max_interval: int = 2, max_day: int = 180) -> Pairs:
        """Generate interferometric pairs by SAR acquisition interval.

        SAR acquisition interval is defined as the number of SAR acquisitions
        between two SAR acquisitions.

        .. admonition:: Example

            If the SAR acquisition interval is 2, then the interferometric pairs
            will be generated between SAR acquisitions with interval of 1 and 2.
            This will be useful to generate interferometric pairs with different
            temporal baselines.

        Parameters
        ----------
        max_interval: int
            max interval between two SAR acquisitions for interferometric pair.
            interval is defined as the number of SAR acquisitions between two SAR
            acquisitions.
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
        """Generate interferometric pairs linking winter in each year.

        winter is defined by month and day for each year. For instance,
        winter_start='0101', winter_end='0331' means the winter is from Jan 1 to
        Mar 31 for each year in the time series. This will be useful to add pairs
        for completely frozen period across years in permafrost region.

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
        same_year = int(winter_start) < int(winter_end)

        # randomly select n_per_period dates in each period/year
        date_years = []
        for year in years:
            start = pd.to_datetime(f"{year}{winter_start}", format="%Y%m%d")
            if same_year:
                end = pd.to_datetime(f"{year}{winter_end}", format="%Y%m%d")
            else:
                end = pd.to_datetime(f"{year + 1}{winter_end}", format="%Y%m%d")

            dt_year = df_dates[start:end]
            if len(dt_year) > 0:
                np.random.default_rng().shuffle(dt_year)
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
                            _pairs.append((date_primary, date_secondary))  # noqa: PERF401
        return Pairs(_pairs)

    def from_period(
        self,
        period_start: str = "0101",
        period_end: str = "0331",
        n_per_period: int | None = None,
        n_primary_period: str | None = None,
        primary_years: list[int] | None = None,
    ) -> Pairs:
        """Generate interferometric pairs between periods for all years.

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
        same_year = int(period_start) < int(period_end)

        # randomly select n_per_period dates in each period/year
        date_years = []
        for year in years:
            start = pd.to_datetime(f"{year}{period_start}", format="%Y%m%d")
            if same_year:
                end = pd.to_datetime(f"{year}{period_end}", format="%Y%m%d")
            else:
                end = pd.to_datetime(f"{year + 1}{period_end}", format="%Y%m%d")

            dt_year = df_dates[start:end]
            if (n_year := len(dt_year)) > 0:
                n = n_year if n_per_period is None else n_per_period
                np.random.default_rng().shuffle(dt_year)
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
        """Generate interferometric pairs between summer and winter in each year.

        summer and winter are defined by month and day for each year. For example,
        summer_start='0801', summer_end='1001' means the summer is from Aug 1to
        Oct 1 for each year in the time series. This will be useful to add pairs
        for whole thawing and freezing process.

        Parameters
        ----------
        summer_start, summer_end:  str
            start and end date for the summer which expressed as month and day
            with format '%m%d'
        winter_start, winter_end:  str
            start and end date for the winter which expressed as month and day
            with format '%m%d'

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
                w_start1 = pd.to_datetime(f"{year - 1}{winter_start}", format="%Y%m%d")
                w_start2 = pd.to_datetime(f"{year}{winter_start}", format="%Y%m%d")
                if int(winter_end) > int(summer_end):
                    w_end1 = pd.to_datetime(f"{year - 1}{winter_end}", format="%Y%m%d")
                    w_end2 = pd.to_datetime(f"{year}{winter_end}", format="%Y%m%d")
                else:
                    w_end1 = pd.to_datetime(f"{year}{winter_end}", format="%Y%m%d")
                    w_end2 = pd.to_datetime(f"{year + 1}{winter_end}", format="%Y%m%d")
            else:
                w_start1 = pd.to_datetime(f"{year}{winter_start}", format="%Y%m%d")
                w_start2 = pd.to_datetime(f"{year + 1}{winter_start}", format="%Y%m%d")

                w_end1 = pd.to_datetime(f"{year}{winter_end}", format="%Y%m%d")
                w_end2 = pd.to_datetime(f"{year + 1}{winter_end}", format="%Y%m%d")

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


# TODO: accelerate the process using numba or cython
def find_loops(
    loops_pairs: Pairs,
    loops: list[Loop],
    loop: Loop,
    pairs_left: Pairs,
    end_date: datetime,
    edge_pairs: Pairs,
    max_acquisition: int = 5,
) -> None:
    """Recursively find all available loops within pairs_left and end_date."""
    from faninsar import Loop

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
            elif len(loop_i) + 1 < max_acquisition:
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
    pair: Pair,
    pairs: Pairs,
    edge_pairs: int | None = None,
) -> bool:
    """Check if the pair is a valid diagonal pair.

    A valid diagonal pair can be used to form a loop using the edge pairs. This
    function is just a simple check to see if days of diagonal pair is less than
    the sum of edge pairs.

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
    if _edge_pairs.days.values.sum() >= pair.days:
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
