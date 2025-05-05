"""A module for SAR acquisition related operations."""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import pandas as pd
import pandas.core.common as com
import xarray as xr
from pandas._libs.internals import BlockValuesRefs
from pandas.core.arrays import ExtensionArray
from pandas.core.construction import ensure_wrapped_if_datetimelike, sanitize_array
from pandas.core.dtypes.common import is_iterator, is_list_like, is_scalar, pandas_dtype
from pandas.core.dtypes.generic import ABCMultiIndex, ABCSeries
from pandas.core.indexes.base import maybe_extract_name
from typing_extensions import Self

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import ArrayLike

logger = setup_logger(log_name=__name__)
_dtype_obj = np.dtype("object")


class Acquisition(pd.DatetimeIndex):
    """A class to handle SAR acquisition dates in FanInSAR.

    This class is a wrapper around :class:`pandas.DatetimeIndex` to handle
    SAR acquisition dates in FanInSAR.

    """

    dims = ("dates",)
    _in_memory = True

    def __new__(cls, *args, **kwargs) -> Self:
        """Create a new instance of Acquisition."""
        return super(Acquisition, cls).__new__(cls, *args, **kwargs)

    def _repr_html_(self) -> str:
        """Return the HTML representation of the class."""
        from faninsar._core.render import array_repr

        return array_repr(self)

    def to_xarray(self) -> pd.DatetimeIndex:
        """Convert the acquisition dates to xarray format."""
        return xr.Variable("Acquisition", self)

    @property
    def stats(self) -> dict:
        """Return the statistical attributes of the acquisition dates."""
        return {
            "start": self.min().strftime("%F"),
            "end": self.max().strftime("%F"),
            "unique": len(self.unique()),
            "total": len(self),
        }

    @property
    def data(self) -> np.ndarray:
        """Return the internal data of the index."""
        return self._data


class DaySpan(pd.Index):
    """A class to handle day span between SAR acquisitions in FanInSAR.

    This class is a wrapper around :class:`pandas.Index` to add some additional
    functionalities.

    Parameters
    ----------
    data: Sequence
        Days span between SAR acquisitions.
    dtype: np.dtype, optional
        Data type of the days span. Default is None.
    copy: bool, optional
        Whether to copy the data. Default is False.

    """

    dims = ("days",)
    _in_memory = True

    def __new__(  # noqa: PLR0912
        cls,
        data: Sequence,
        dtype: np.dtype = None,
        copy: bool = False,
    ) -> Self:
        """Create a new instance of Index."""
        name = maybe_extract_name(None, data, cls)
        # if name is None:
        #     name = "DaySpan"
        if dtype is not None:
            dtype = pandas_dtype(dtype)
        data_dtype = getattr(data, "dtype", None)
        refs = None
        if not copy and isinstance(data, (ABCSeries, pd.Index)):
            refs = data._references

        is_pandas_object = isinstance(data, (ABCSeries, pd.Index, ExtensionArray))
        # ra
        if isinstance(data, (np.ndarray, pd.Index, ABCSeries)):
            if isinstance(data, ABCMultiIndex):
                data = data._values
            if data.dtype.kind not in "iufcbmM":
                # GH#11836 we need to avoid having numpy coerce
                # things that look like ints/floats to ints unless
                # they are actually ints, e.g. '0' and 0.0
                # should not be coerced
                data = com.asarray_tuplesafe(data, dtype=_dtype_obj)
        elif is_scalar(data):
            raise cls._raise_scalar_data_error(data)
        elif hasattr(data, "__array__"):
            return cls(np.asarray(data), dtype=dtype, copy=copy, name=name)
        elif not is_list_like(data) and not isinstance(data, memoryview):
            # 2022-11-16 the memoryview check is only necessary on some CI
            #  builds, not clear why
            raise cls._raise_scalar_data_error(data)
        else:
            # GH21470: convert iterable to list before determining if empty
            if is_iterator(data):
                data = list(data)

            if not isinstance(data, (list, tuple)):
                # we allow set/frozenset, which Series/sanitize_array does not, so
                #  cast to list here
                data = list(data)
            if len(data) == 0:
                # unlike Series, we default to object dtype:
                data = np.array(data, dtype=object)

            if len(data) and isinstance(data[0], tuple):
                # Ensure we get 1-D array of tuples instead of 2D array.
                data = com.asarray_tuplesafe(data, dtype=_dtype_obj)

        try:
            arr = sanitize_array(data, None, dtype=dtype, copy=copy)
        except ValueError as err:
            if "index must be specified when data is not list-like" in str(err):
                raise cls._raise_scalar_data_error(data) from err
            if "Data must be 1-dimensional" in str(err):
                msg = "Index data must be 1-dimensional"
                raise ValueError(msg) from err
            raise
        arr = ensure_wrapped_if_datetimelike(arr)

        arr = cls._ensure_array(arr, arr.dtype, copy=False)
        result = cls._simple_new(arr, name, refs=refs)
        if (
            dtype is None
            and is_pandas_object
            and data_dtype == np.object_
            and result.dtype != data_dtype
        ):
            msg = (
                "Dtype inference on a pandas object "
                "(Series, Index, ExtensionArray) is deprecated. The Index "
                "constructor will keep the original dtype in the future. "
                "Call `infer_objects` on the result to get the old "
                "behavior."
            )
            warnings.warn(
                msg,
                FutureWarning,
                stacklevel=2,
            )
        return result  # type: ignore[return-value]

    @classmethod
    def _simple_new(
        cls,
        values: ArrayLike,
        name: Hashable | None = None,
        refs: object = None,
    ) -> Self:
        """We require that we have a dtype compat for the values. If we are passed
        a non-dtype compat, then coerce using the constructor.

        Must be careful not to recurse.
        """  # noqa: D205
        assert isinstance(values, cls._data_cls), type(values)

        result = object.__new__(cls)
        result._data = values
        result._name = name
        result._cache = {}
        result._reset_identity()
        if refs is not None:
            result._references = refs
        else:
            result._references = BlockValuesRefs()
        result._references.add_index_reference(result)

        return result

    @classmethod
    def _ensure_array(cls, data: ArrayLike, dtype: np.dtype, copy: bool) -> ArrayLike:
        """Ensure we have a valid array to pass to _simple_new."""
        if data.ndim > 1:
            # GH#13601, GH#20285, GH#27125
            msg = "Index data must be 1-dimensional"
            raise ValueError(msg)
        if dtype == np.float16:
            # float16 not supported (no indexing engine)
            msg = "float16 indexes are not supported"
            raise NotImplementedError(msg)

        if copy:
            # asarray_tuplesafe does not always copy underlying data,
            #  so need to make sure that this happens
            data = data.copy()
        return data

    def _repr_html_(self) -> str:
        """Return the HTML representation of the class."""
        from faninsar._core.render import array_repr

        return array_repr(self)

    def to_xarray(self) -> pd.Index:
        """Convert the acquisition dates to xarray format."""
        return xr.Variable("days", self)

    @property
    def stats(self) -> dict:
        """Return the statistical attributes of the acquisition dates."""
        return {
            "min": self.min(),
            "max": self.max(),
            "unique": len(self.unique()),
            "total": len(self),
        }

    @property
    def data(self) -> np.ndarray:
        """Return the internal data of the index."""
        return self._data


class DateManager:
    """Date manager class to handle date operations."""

    # TODO: user defined period (using day of year, wrap into new year and cut into
    # periods)

    @staticmethod
    def season_of_month(month: int) -> int:
        """Return the season of a given month.

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
            msg = "Month should be in range 1-12. But got '{month}'."
            raise ValueError(msg)
        return (month - 3) % 12 // 3 + 1

    @staticmethod
    def ensure_datetime(date: str | datetime) -> datetime:
        """Ensure the date is a datetime object.

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
            msg = f"Date should be datetime or str, but got {type(date)}"
            raise TypeError(msg)
        return date

    @staticmethod
    def str_to_dates(
        date_str: str,
        length: int | None = 2,
        parse_function: Callable | None = None,
        date_args: dict | None = None,
    ) -> tuple[datetime, ...]:
        """Convert date string to dates.

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
        date_str = str(date_str)
        if date_args is None:
            date_args = {}
        if parse_function is not None:
            dates = parse_function(date_str)
        else:
            items = date_str.split("_")
            if len(items) >= length:
                dates_ls = items[-length:]
            else:
                msg = f"The number of dates in {date_str} is less than {length}."
                raise ValueError(msg)

        date_args.update({"errors": "raise"})
        try:
            dates = [pd.to_datetime(i, **date_args) for i in dates_ls]
        except Exception as e:
            msg = f"Dates in {date_str} not recognized."
            raise ValueError(msg) from e

        return tuple(dates)
