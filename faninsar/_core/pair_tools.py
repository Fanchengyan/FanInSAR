from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Tuple, Union
import functools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Pair:
    '''Pair class for one pair.
    '''
    _values: np.ndarray
    _name: str
    _days: int

    __slots__ = ['_values', '_name', '_days']

    def __init__(
        self,
        pair: Iterable[datetime, datetime],
    ) -> None:
        '''
        Parameters
        ----------
        pair: Iterable
            Iterable object of two dates. Each date is a datetime object.
            For example, (date1, date2).
        '''
        self._values = np.sort(pair).astype('M8[D]')
        self._name = '_'.join([i.strftime('%Y%m%d')
                              for i in self._values.astype('O')])
        self._days = (self._values[1] - self._values[0]).astype(int)

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"Pair({self._name})"

    def __eq__(self, other: 'Pair') -> bool:
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self._name)

    @property
    def values(self):
        '''return the values of the pair.

        Returns
        -------
        values: np.ndarray
            Values of the pair with format of datetime.
        '''
        return self._values

    @property
    def name(self):
        '''return the string of the pair.

        Returns
        -------
        name: str
            String of the pair with format of '%Y%m%d_%Y%m%d'.
        '''
        return self._name

    @property
    def days(self):
        '''return the time span of the pair in days.

        Returns
        -------
        days: int
            Time span of the pair in days.
        '''
        return self._days

    @classmethod
    def from_name(
        cls,
        name: str,
        parse_function: Optional[Callable] = None,
        date_args: Optional[dict] = None
    ) -> 'Pair':
        '''initialize the pair class from a pair name

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
            Default is None.

        Returns
        -------
        pair: Pair
            Pair class.
        '''
        dates = DateManager.str_to_dates(name, 2, parse_function, date_args)
        return cls(dates)


class Pairs:
    """Pairs class to handle pairs

    Examples
    --------
    ::

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

    __slots__ = ['_values', '_dates', '_length', '_edge_index']

    def __init__(
        self,
        pairs: Union[Iterable[Iterable[datetime, datetime]],
                     Iterable[Pair]],
        sort: bool = True,
    ) -> None:
        '''initialize the pairs class


        Parameters
        ----------
        pairs: Iterable
            Iterable object of pairs. Each pair is an Iterable or Pair
            object of two dates with format of datetime. For example, 
            [(date1, date2), ...].
        sort: bool, optional
            Whether to sort the pairs. Default is True.
        '''
        if pairs is None or len(pairs) == 0:
            raise ValueError("pairs cannot be None.")
        pairs_ls = []
        for pair in pairs:
            if isinstance(pair, Pair):
                _pair = pair
            elif isinstance(pair, Iterable):
                _pair = Pair(pair)
            else:
                raise TypeError(
                    f"pairs should be an Iterable containing Iterable or Pair object, but got {type(pair)}.")
            pairs_ls.append(_pair.values)

        _values = np.array(pairs_ls)

        self._values = _values
        self._dates = np.unique(pairs_ls)
        self._length = self._values.shape[0]
        
        self._edge_index = np.searchsorted(self._dates, self._values)
        
        if sort:
            self.sort()

    def __len__(self) -> int:
        return self._length

    def __str__(self) -> str:
        return f"Pairs({self._length})"

    def __repr__(self) -> str:
        return self.to_frame().__repr__()

    def __eq__(self, other: 'Pairs') -> bool:
        return np.array_equal(self.values, other.values)

    def __add__(self, other: 'Pairs') -> 'Pairs':
        _pairs = np.union1d(self.to_names(), other.to_names())
        if len(_pairs) > 0:
            return Pairs.from_names(_pairs)

    def __sub__(self, other: 'Pairs') -> 'Pairs':
        _pairs = np.setdiff1d(self.to_names(), other.to_names())
        if len(_pairs) > 0:
            return Pairs.from_names(_pairs)

    def __getitem__(self, index: int) -> Union['Pair', 'Pairs']:
        if isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            if (isinstance(start, (int, np.integer, type(None)))
                    and isinstance(stop, (int, np.integer, type(None)))):
                if start is None:
                    start = 0
                if stop is None:
                    stop = self._length
                return Pairs(self._values[start:stop:step])
            elif (isinstance(start, (datetime, np.datetime64, pd.Timestamp, str, type(None)))
                  and isinstance(stop, (datetime, np.datetime64, pd.Timestamp, str, type(None)))):
                if isinstance(start, str):
                    start = DateManager.ensure_datetime(start)
                if isinstance(stop, str):
                    stop = DateManager.ensure_datetime(stop)
                if start is None:
                    start = self._dates[0]
                if stop is None:
                    stop = self._dates[-1]

                start, stop = (np.datetime64(start, "s"),
                               np.datetime64(stop, "s"))

                if start > stop:
                    raise ValueError(
                        f"Index start {start} should be earlier than index stop {stop}.")
                _pairs = []
                for pair in self._values:
                    pair = pair.astype('M8[s]')
                    if start <= pair[0] <= stop and start <= pair[1] <= stop:
                        _pairs.append(pair)
                if len(_pairs) > 0:
                    return Pairs(_pairs)
                else:
                    return None
        elif isinstance(index, (int, np.integer)):
            if index >= self._length:
                raise IndexError(
                    f"Index {index} out of range. Pairs number is {self._length}.")
            return Pair(self._values[index])
        elif isinstance(index, (datetime, np.datetime64, pd.Timestamp, str)):
            if isinstance(index, str):
                try:
                    index = pd.to_datetime(index)
                except:
                    raise ValueError(
                        f"String {index} cannot be converted to datetime.")
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
            if not index.ndim == 1:
                raise IndexError(
                    f"Index should be 1D array, but got {index.ndim}D array.")
            if len(index) > self._length:
                raise IndexError(
                    f"Index length should be less than pairs length {self._length},"
                    f" but got {len(index)}.")
            return Pairs(self._values[index])

        else:
            raise TypeError(
                f"Index should be int, slice, datetime, str, or 1D bool array, but got {type(index)}.")

    def __hash__(self) -> int:
        return hash(''.join(self.to_names()))

    def __iter__(self):
        return iter(self.values)

    def __contains__(self, item):
        if isinstance(item, Pair):
            item = item.values
        elif isinstance(item, str):
            item = Pair.from_name(item).values
        elif isinstance(item, Iterable):
            item = np.sort(item)
        else:
            raise TypeError(
                f"item should be Pair, str, or Iterable, but got {type(item)}.")

        return np.any(np.all(item == self.values, axis=1))

    @property
    def values(self) -> np.ndarray:
        '''return the pairs array in type of np.datetime64[D]'''
        return self._values

    @property
    def dates(self) -> np.ndarray:
        '''return the sorted dates array of all pairs in type of np.datetime64[D]'''
        return self._dates

    @property
    def days(self) -> np.ndarray:
        '''return the time span of all pairs in days'''
        return (self._values[:, 1] - self._values[:, 0]).astype(int)

    @property
    def edge_index(self) -> np.ndarray:
        '''return the index of the pairs in the dates coordinate (edge index in 
        graph theory)'''
        return self._edge_index

    @property
    def shape(self) -> Tuple[int, int]:
        '''return the shape of the pairs array'''
        return self._values.shape

    @classmethod
    def from_names(
        cls,
        names: Iterable[str],
        parse_function: Optional[Callable] = None,
        date_args: Optional[dict] = None
    ) -> 'Pairs':
        '''initialize the pair class from a pair name

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
            Default is None.

        Returns
        -------
        pairs: Pairs
            unsorted Pairs object.
        '''
        pairs = []
        for name in names:
            pair = Pair.from_name(name, parse_function, date_args)
            pairs.append(pair.values)
        return cls(pairs, sort=False)

    def where(self, pair: Union[str, Pair]) -> Optional[int]:
        '''return the index of the pair

        Parameters
        ----------
        pair: str or Pair
            Pair name or Pair object.
        '''
        if isinstance(pair, str):
            pair = Pair.from_name(pair)
        elif not isinstance(pair, Pair):
            raise TypeError(
                f"pair should be str or Pair, but got {type(pair)}.")
        if pair in self:
            return np.where(np.all(self._values == pair.values, axis=1))[0][0]
        else:
            return None

    # TODO: 1. not duplicated pairs 2. add duplicate function
    def sort(
        self,
        order: Union[str, list] = 'pairs',
        ascending: bool = True,
        return_index: bool = False
    ) -> Optional[np.ndarray]:
        '''sort the pairs

        Parameters
        ----------
        order: str or list of str, optional
            By which fields to sort the pairs. This argument specifies
            which fields to compare first, second, etc. Default is 'pairs'.

            The available options are one or a list of:
            ['pairs', 'primary', 'secondary', 'days'].           
        ascending: bool, optional
            Whether to sort ascending. Default is True.
        return_index: bool, optional
            Whether to return the index of the sorted pairs. Default is False.

        Returns
        -------
        None or np.ndarray. if return_index is True, return the index of the 
        sorted pairs.
        '''
        item_map = {
            'pairs': self._values,
            'primary': self._values[:, 0],
            'secondary': self._values[:, 1],
            'days': self.days,
        }
        if isinstance(order, str):
            order = [order]
        _values = []
        for i in order:
            if i not in item_map.keys():
                raise ValueError(
                    f"order should be one of {list(item_map.keys())}, but got {order}.")
            _values.append(item_map[i].reshape(self._length, -1))
        _values = np.hstack(_values)
        _, _index = np.unique(
            _values, axis=0, return_index=True)
        if not ascending:
            _index = _index[::-1]
        self._values = self._values[_index]

        if return_index:
            return _index

    def to_names(self, prefix: Optional[str] = None) -> List[str]:
        '''generate pair names string with prefix

        Parameters
        ----------
        prefix: str
            Prefix of the pair file names. Default is ''.
        '''
        if prefix:
            return [f"{prefix}_{Pair(i).name}" for i in self._values]
        else:
            return [Pair(i).name for i in self._values]

    def to_frame(self) -> pd.DataFrame:
        '''return the pairs as a DataFrame'''
        return pd.DataFrame(self._values, columns=['primary', 'secondary'])

    def to_loops(self) -> 'Loops':
        '''return all possible loops from the pairs'''
        loops = []
        for i, pair12 in enumerate(self._values):
            for pair23 in self._values[i+1:]:
                if pair12[1] == pair23[0] and Pair([pair12[0], pair23[1]]) in self:
                    loops.append([pair12[0], pair12[1], pair23[1]])
        return Loops(loops)

    # def to_matrix(self) -> np.ndarray:
    #     '''return the SBAS matrix

    #     Parameters
    #     ----------
    #     matrix: np.ndarray
    #         SBAS matrix in shape of (n_pairs, n_dates-1). The dates between
    #         pairs are set to 1, otherwise 0.
    #     '''
    #     matrix = np.zeros((len(self), len(self.dates)-1))
    #     dates = self.dates.tolist()
    #     for i, pair in enumerate(self.values):
    #         index1 = dates.index(pair[0])
    #         index2 = dates.index(pair[1])
    #         matrix[i, index1:index2] = 1

    #     return matrix
    
    def to_matrix(self) -> np.ndarray:
        '''return the SBAS matrix

        Parameters
        ----------
        matrix: np.ndarray
            SBAS matrix in shape of (n_pairs, n_dates-1). The dates between
            pairs are set to 1, otherwise 0.
        '''
        matrix = np.zeros((len(self), len(self.dates)-1))
        col_idx = self.edge_index.copy()
        for idx, i in enumerate(col_idx):
            matrix[idx, i[0]:i[1]] = 1

        return matrix

    def dates_string(self, format='%Y%m%d') -> List[str]:
        '''return the dates of the pairs with format of str

        Parameters
        ----------
        format: str
            Format of the date string. Default is '%Y%m%d'.
        '''
        return [i.strftime(format) for i in self._dates.astype(datetime)]


class Loop:
    '''Loop class containing three pairs/acquisitions.

    Examples
    --------

        prepare dates and loops for examples:

        Get pairs of loops and then convert back to loops to check if they are equal

        >>> loops == loops.pairs.to_loops()
        True
    '''
    _values: np.ndarray
    _name: str
    _pairs: List[Pair]

    __slots__ = ['_values', '_pairs', '_name',
                 '_days12', '_days23', '_days13']

    def __init__(
        self,
        loop: Iterable[datetime, datetime, datetime]
    ) -> None:
        '''initialize the Loop class

        Parameters
        ----------
        loop: Iterable
            Iterable object of three dates. Each date is a datetime object.
            For example, (date1, date2, date3).
        '''
        self._values = np.sort(loop).astype('M8[D]')
        loop_dt = self._values.astype(datetime)
        self._name = '_'.join([i.strftime('%Y%m%d') for i in loop_dt])
        self._pairs = [
            Pair([loop_dt[0], loop_dt[1]]),
            Pair([loop_dt[1], loop_dt[2]]),
            Pair([loop_dt[0], loop_dt[2]])
        ]

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"Loop({self._name})"

    def __eq__(self, other) -> bool:
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def values(self) -> np.ndarray:
        '''return the values array of the loop.

        Returns
        -------
        values: np.ndarray
            Three dates of the loop with format of np.datetime64[D].
        '''
        return self._values

    @property
    def pairs(self) -> List[Pair]:
        '''return all three pairs of the loop.

        Returns
        -------
        pairs: list
            List containing three pairs. Each pair is a Pair class.
        '''
        return self._pairs

    @property
    def days12(self) -> int:
        '''return the time span of the first pair in days.

        Returns
        -------
        days12: int
            Time span of the first pair in days.
        '''
        return (self._values[1] - self._values[0]).astype(int)

    @property
    def days23(self) -> int:
        '''return the time span of the second pair in days.

        Returns
        -------
        days23: int
            Time span of the second pair in days.
        '''
        return (self._values[2] - self._values[1]).astype(int)

    @property
    def days13(self) -> int:
        '''return the time span of the third pair in days.

        Returns
        -------
        days13: int
            Time span of the third pair in days.
        '''
        return (self._values[2] - self._values[0]).astype(int)

    @property
    def name(self) -> str:
        '''return the string format the loop.

        Returns
        -------
        name: str
            String of the loop.
        '''
        return self._name

    @classmethod
    def from_name(
        cls,
        name: str,
        parse_function: Optional[Callable] = None,
        date_args: Optional[dict] = None
    ) -> 'Loop':
        '''initialize the loop class from a loop name

        Parameters
        ----------
        name: str
            Loop name.
        parse_function: Callable, optional
            Function to parse the date strings from the loop name.
            If None, the loop name will be split by '_' and the 
            last 3 items will be used. Default is None.
        date_args: dict, optional
            Keyword arguments for pd.to_datetime() to convert the 
            date strings to datetime objects. For example, 
            {'format': '%Y%m%d'}. Default is None.

        Returns
        -------
        loop: Loop
            Loop object.
        '''
        dates = DateManager.str_to_dates(name, 3, parse_function, date_args)
        return cls(dates)


class Loops:
    '''Loops class to handle loops'''

    _values: np.ndarray
    _dates: np.ndarray
    _length: int

    __slots__ = ['_values', '_dates', '_length']

    def __init__(
        self,
        loops: Union[Iterable[Iterable[datetime, datetime, datetime]],
                     Iterable[Loop]],
        sort: bool = True
    ) -> None:
        '''initialize the loops class

        Parameters
        ----------
        loops: Iterable
            Iterable object of loops. Each loop is an Iterable object
            of three dates with format of datetime or Loop object. 
            For example, [(date1, date2, date3), ...].
        '''
        if loops is None or len(loops) == 0:
            raise ValueError("loops cannot be None.")
        loops_ls = []
        for loop in loops:
            if isinstance(loop, Loop):
                _loop = loop
            elif isinstance(loop, Iterable):
                _loop = Loop(loop)
            else:
                raise TypeError(
                    f"loops should be an Iterable containing Iterable or Poop object, but got {type(loop)}.")
            loops_ls.append(_loop.values)

        _values, _index = np.unique(loops_ls, axis=0, return_index=True)
        if not sort:
            _values = _values[_index]

        self._values = _values
        self._dates = np.unique(loops_ls)
        self._length = self._values.shape[0]

    def __str__(self) -> str:
        return f"Loops({self._length})"

    def __repr__(self) -> str:
        return self.to_frame('dates').__repr__()

    def __len__(self) -> int:
        return self._length

    def __eq__(self, other: 'Loops') -> bool:
        return np.array_equal(self.values, other.values)

    def __add__(self, other: 'Loops') -> 'Loops':
        _loops = np.union1d(self.to_names(), other.to_names())
        return Loops.from_names(_loops)

    def __sub__(self, other: 'Loops') -> Optional['Loops']:
        _loops = np.setdiff1d(self.to_names(), other.to_names())
        if len(_loops) > 0:
            return Loops.from_names(_loops)

    def __getitem__(self, index: int) -> Union['Loop', 'Loops']:
        if isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            if (isinstance(start, (int, np.integer, type(None)))
                    and isinstance(stop, (int, np.integer, type(None)))):
                if start is None:
                    start = 0
                if stop is None:
                    stop = self._length
                return Loops(self._values[start:stop:step])
            elif (isinstance(start, (datetime, np.datetime64, pd.Timestamp, str, type(None)))
                  and isinstance(stop, (datetime, np.datetime64, pd.Timestamp, str, type(None)))):
                if isinstance(start, str):
                    start = DateManager.ensure_datetime(start)
                if isinstance(stop, str):
                    stop = DateManager.ensure_datetime(stop)
                if start is None:
                    start = self._dates[0]
                if stop is None:
                    stop = self._dates[-1]

                start, stop = (np.datetime64(start, "s"),
                               np.datetime64(stop, "s"))

                if start > stop:
                    raise ValueError(
                        f"Index start {start} should be earlier than index stop {stop}.")
                _loops = []
                for loop in self._values:
                    loop = loop.astype('M8[s]')
                    if np.all((start <= loop) & (loop <= stop)):
                        _loops.append(loop)
                if len(_loops) > 0:
                    return Loops(_loops)
                else:
                    return None
        elif isinstance(index, (int, np.integer)):
            if index >= self._length:
                raise IndexError(
                    f"Index {index} out of range. Loops number is {self._length}.")
            return Loop(self._values[index])
        elif isinstance(index, (datetime, np.datetime64, pd.Timestamp, str)):
            if isinstance(index, str):
                try:
                    index = pd.to_datetime(index)
                except:
                    raise ValueError(
                        f"String {index} cannot be converted to datetime.")
            loops = []
            for loop in self._values:
                if index in loop:
                    loops.append(loop)
            if len(loops) > 0:
                return Loops(loops)
            else:
                return None
        elif isinstance(index, Iterable):
            index = np.array(index)
            if not index.ndim == 1:
                raise IndexError(
                    f"Index should be 1D array, but got {index.ndim}D array.")
            if len(index) > self._length:
                raise IndexError(
                    f"Index length should be less than pairs length {self._length},"
                    f" but got {len(index)}.")
            return Loops(self._values[index])
        else:
            raise TypeError(
                f"Index should be int, slice, datetime, str, or 1D bool array, but got {type(index)}.")

    def __hash__(self) -> int:
        return hash(''.join(self.to_names()))

    def __iter__(self):
        return iter(self.values)

    def __contains__(self, item):
        if isinstance(item, Loop):
            item = item.values
        elif isinstance(item, str):
            item = Loop.from_name(item).values
        elif isinstance(item, Iterable):
            item = np.sort(item)
        else:
            raise TypeError(
                f"item should be Loop, str, or Iterable, but got {type(item)}.")

        return np.any(np.all(item == self.values, axis=1))

    @property
    def values(self) -> np.ndarray:
        '''return the values of the loops.

        Returns
        -------
        values: np.ndarray
            Values of the loops with format of datetime.
        '''
        return self._values

    @property
    def dates(self) -> np.ndarray:
        '''return the sorted dates of the loops.

        Returns
        -------
        dates: np.ndarray
            Sorted dates of the loops with format of datetime.
        '''
        return self._dates

    @property
    def shape(self) -> Tuple[int, int]:
        '''return the shape of the loop array'''
        return self._values.shape

    @property
    def pairs(self) -> Pairs:
        '''return the sorted all pairs of the loops.

        Returns
        -------
        pairs: Pairs
            Pairs of the loops.
        '''
        pairs = np.unique(np.vstack([
            self._values[:, :2],
            self._values[:, 1:],
            self._values[:, [0, 2]]
        ]), axis=0)
        return Pairs(pairs, sort=False)

    @property
    def pairs12(self) -> Pairs:
        '''return the first pairs of the loops.

        Returns
        -------
        pairs12: Pairs
            First pairs of the loops.
        '''
        return Pairs(self._values[:, :2], sort=False)

    @property
    def pairs23(self) -> Pairs:
        '''return the second pairs of the loops.

        Returns
        -------
        pairs23: Pairs
            Second pairs of the loops.
        '''
        return Pairs(self._values[:, 1:], sort=False)

    @property
    def pairs13(self) -> Pairs:
        '''return the third pairs of the loops.

        Returns
        -------
        pairs13: Pairs
            Third pairs of the loops.
        '''
        return Pairs(self._values[:, [0, 2]], sort=False)

    @property
    def days12(self) -> np.ndarray:
        '''return the time span of the first pair in days.

        Returns
        -------
        days12: np.ndarray
            Time span of the first pair in days.
        '''
        return (self._values[:, 1] - self._values[:, 0]).astype(int)

    @property
    def days23(self) -> np.ndarray:
        '''return the time span of the second pair in days.

        Returns
        -------
        days23: np.ndarray
            Time span of the second pair in days.
        '''
        return (self._values[:, 2] - self._values[:, 1]).astype(int)

    @property
    def days13(self) -> np.ndarray:
        '''return the time span of the third pair in days.

        Returns
        -------
        days13: np.ndarray
            Time span of the third pair in days.
        '''
        return (self._values[:, 2] - self._values[:, 0]).astype(int)

    @property
    def index(self) -> np.ndarray:
        '''return the index of the loops in dates coordinates.

        Returns
        -------
        index: np.ndarray
            Index of the loops in dates coordinates.
        '''
        return np.searchsorted(self._dates, self._values)

    @classmethod
    def from_names(
        cls,
        names: List[str],
        parse_function: Optional[Callable] = None,
        date_args: Optional[dict] = None,
    ) -> 'Loops':
        '''initialize the loops class from a list of loop file names.

        Parameters
        ----------
        names: list
            List of loop file names.
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
        loops: Loops
            unsorted Loops object.
        '''
        loops = []
        for name in names:
            loop = Loop.from_name(name, parse_function, date_args)
            loops.append(loop.values)
        return cls(loops, sort=False)

    def to_names(self, prefix: Optional[str] = None) -> List[str]:
        '''return the string name of each loop.

        Returns
        -------
        prefix: str, optional
            Prefix of the output loop names. Default is None.
        '''
        if prefix:
            return [f"{prefix}_{Loop(i).name}" for i in self._values]
        else:
            return [Loop(i).name for i in self._values]

    def to_frame(
        self,
        target: Literal['pairs', 'dates'] = 'pairs'
    ) -> pd.DataFrame:
        '''return the loops as a DataFrame

        Parameters
        ----------
        target: str, one of ['pairs', 'dates']
            Target of the DataFrame. Default is 'pairs'.
        '''
        if target == 'pairs':
            return pd.DataFrame(
                zip(self.pairs12.values, self.pairs23.values, self.pairs13.values),
                columns=['pair12', 'pair23', 'pair13']
            )
        elif target == 'dates':
            return pd.DataFrame(
                self.values,
                columns=['date1', 'date2', 'date3']
            )
        else:
            raise ValueError(
                f"target should be 'pairs' or 'dates', but got {target}.")

    def to_matrix(self) -> np.ndarray:
        """
        return loop matrix (containing 1, -1, 0) from pairs.

        Returns
        -------
        matrix: np.ndarray
            Loop matrix with the shape of (n_loop, n_pair). The values of each
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

    def where(self, loop: Union[str, Loop]) -> Optional[int]:
        '''return the index of the loop

        Parameters
        ----------
        loop: str or Loop
            Loop name or Loop object.
        '''
        if isinstance(loop, str):
            loop = Loop.from_name(loop)
        elif not isinstance(loop, Loop):
            raise TypeError(
                f"loop should be str or Loop, but got {type(loop)}.")
        if loop in self:
            return np.where(np.all(self._values == loop.values, axis=1))[0][0]
        else:
            return None

    def sort(
        self,
        order: Union[str, List] = 'pairs',
        ascending: bool = True,
        return_index: bool = False
    ) -> Optional[np.ndarray]:
        '''sort the loops

        Parameters
        ----------
        order: str or list of str, optional
            By which fields to sort the loops. this argument specifies
            which fields to compare first, second, etc. Default is 'pairs'.

            The available options are one or a list of:

            - **date:**: 'date1', 'date2', 'date3'
            - **pairs:*: 'pairs12', 'pairs23', 'pairs13'
            - **days:**: 'days12', 'days23', 'days13'
            - **short name:** 'date', 'pairs', 'days'. short name will be
                treated as a combination of the above options. For example,
                'date' is equivalent to ['date1', 'date2', 'date3']. 
        ascending: bool, optional
            Whether to sort ascending. Default is True.
        return_index: bool, optional
            Whether to return the index of the sorted loops. Default is False.

        Returns
        -------
        None or np.ndarray. if return_index is True, return the index of the
        sorted loops.
        '''
        item_map = {
            'date1': self._values[:, 0],
            'date2': self._values[:, 1],
            'date3': self._values[:, 2],
            'pairs12': self.pairs12.values,
            'pairs23': self.pairs23.values,
            'pairs13': self.pairs13.values,
            'days12': self.days12,
            'days23': self.days23,
            'days13': self.days13,
            'date': self._values,
            'pairs': zip([self.pairs12.values,
                          self.pairs23.values,
                          self.pairs13.values]),
            'days': zip([self.days12, self.days23, self.days13])
        }
        if isinstance(order, str):
            order = [order]
        _values = []
        for i in order:
            if i not in item_map.keys():
                raise ValueError(
                    f"order should be one of {list(item_map.keys())}, but got {order}.")
            _values.append(item_map[i].reshape(self._length, -1))
        _values = np.hstack(_values)
        _, _index = np.unique(
            _values, axis=0, return_index=True)
        if not ascending:
            _index = _index[::-1]
        self._values = self._values[_index]

        if return_index:
            return _index

    def to_seasons(self):
        '''return the season of each loop.

        Returns
        -------
        seasons: list
            List of seasons of each loop.
                0: not the same season
                1: spring
                2: summer
                3: fall
                4: winter
        '''
        seasons = []
        for loop in self.values.astype("O"):
            season1 = DateManager.season_of_month(loop[0].month)
            season2 = DateManager.season_of_month(loop[1].month)
            season3 = DateManager.season_of_month(loop[2].month)
            if season1 == season2 == season3 and loop.days13 < 180:
                seasons.append(season1)
            else:
                seasons.append(0)
        return seasons


class SBASNetwork:
    '''SBAS network class to handle pairs and loops.

    Examples
    --------
    ::
        initialize SBASNetwork class from Pairs object named pairs

        >>> sbas = SBASNetwork(pairs)

        get pairs that are not in the loops

        >>> pairs_alone = sbas.pairs - sbas.loops.pairs
    '''

    _pairs: Pairs
    _loops: Loops
    _baseline: np.ndarray

    __slots__ = ['_pairs', '_loops', '_baseline']

    def __init__(
        self,
        pairs: Union[Iterable[Iterable[datetime, datetime]],
                     Iterable[Pair],
                     Pairs],
        baseline: Optional[Iterable[float]] = None,
        sort: bool = True
    ) -> None:
        '''    
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

        '''

        if isinstance(pairs, Pairs):
            self._pairs = pairs
        elif isinstance(pairs, Iterable):
            self._pairs = Pairs(pairs, sort=False)
        else:
            raise TypeError(
                f"pairs should be an Iterable or Pairs object, but got {type(pairs)}.")
        self.baseline = baseline
        self._loops = self._pairs.to_loops()
        if sort:
            self.sort()

    @property
    def pairs(self) -> Pairs:
        '''return the pairs of the network

        Returns
        -------
        pairs: Pairs
            Pairs of the network.
        '''
        return self._pairs

    @property
    def loops(self) -> Loops:
        '''return the loops of the network

        Returns
        -------
        loops: Loops
            Loops of the network.
        '''
        return self._loops

    @property
    def baseline(self) -> Optional[np.ndarray]:
        '''return the baseline of the network

        Returns
        -------
        baseline: np.ndarray or None
            Baseline of the network.
        '''
        return self._baseline

    @baseline.setter
    def baseline(self, value: Optional[Iterable]):
        '''set the baseline of the network

        Parameters
        ----------
        value: Iterable or None
            Baseline of the network.
        '''
        if value is None:
            self._baseline = None
            return
        elif isinstance(value, Iterable):
            value = np.asarray(value)
        else:
            raise TypeError(
                f"baseline should be an Iterable, but got {type(value)}.")

        if len(self._pairs) != len(value):
            raise ValueError(
                f"Length of pairs {len(self._pairs)} should be equal to length of baseline {len(value)}.")

        self._baseline = value

    @property
    def dates(self) -> List[str]:
        return self._pairs.dates

    @classmethod
    def from_names(
        cls,
        names: List[str],
        parse_function: Optional[Callable] = None,
        date_args: Optional[dict] = None,
    ) -> 'SBASNetwork':
        '''initialize SBASNetwork class from a list of pair names

        Parameters
        ----------
        names: list
            List of loop file names.
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
        '''
        pairs = Pairs.from_names(names, parse_function, date_args)
        return cls(pairs)

    def sort(
        self,
        order: Union[str, list] = 'pairs',
        ascending: bool = True,
        return_index: bool = False
    ):
        '''sort the pairs and corresponding baseline of the network

        Parameters
        ----------
        order: str or list of str, optional
            By which fields to sort the loops. This argument specifies
            which fields to compare first, second, etc. Default is 'pairs'.

            The available options are one or a list of:
            ['pairs', 'primary', 'secondary', 'days'].           
        ascending: bool, optional
            Whether to sort ascending. Default is True.
        return_index: bool, optional
            Whether to return the index of the sorted loops. Default is False.
        '''
        _index = self._pairs.sort(order, ascending)
        if self.baseline is not None:
            self.baseline = self.baseline[_index]
        if return_index:
            return _index

    def plot(
        self,
        fname: Optional[Union[str, Path]] = None,
        ax: Optional[plt.Axes] = None,
        dpi: Optional[int] = None,
    ):
        '''plot the network'''
        # TODO: plot network
        pass


class PairsFactory:
    '''This class is used to generate interferometric pairs for InSAR  
    processing. 

    .. note::

        Functions in this class only generate interferometric pairs and
        do not check if the interferometric pairs are valid. 

        For example, if the interferometric pairs are generated by 
        ``from_period`` function, the interferometric pairs may not be 
        valid if the temporal baseline of the interferometric pairs 
        are too large. The users should check the temporal baseline 
        of the interferometric pairs and remove the invalid interferometric 
        pairs.
    '''

    def __init__(
        self,
        dates: Iterable,
        **kwargs
    ) -> None:
        '''initialize the PairGenerator class

        Parameters
        ----------
        dates: Iterable
            Iterable object that contains the dates. Can be any object that
            can be passed to pd.to_datetime(). For example, ['20190101', '20190201'].
        date_args: dict, optional
            Keyword arguments for pd.to_datetime().
        '''
        self.dates = pd.to_datetime(dates, **kwargs)

    def from_interval(
        self,
        max_interval: int = 2,
        max_day: int = 180
    ) -> Pairs:
        '''generate interferometric pairs by SAR acquisition interval. SAR 
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
        '''
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

    def from_period(
        self,
        period_start: str = '1201',
        period_end: str = '0331',
        n_per_period: str = 3,
        n_primary_period: str = 2
    ) -> Pairs:
        '''generate interferometric pairs between periods for all years. period is defined by month 
        and day for each year. For example, period_start='1201', period_end='0331' means the period
        is from Dec 1 to Mar 31 for each year in the time series. This function will randomly select
        n_per_period dates in each period and generate interferometric pairs between those dates. This
        will be useful to mitigate the temporal cumulative bias.

        Parameters
        ----------
        period_start, period_end:  str
            start and end date for the period which expressed as month and day with format '%m%d'
        n_per_period: int
            how many dates will be used for each period. Those dates will be selected randomly 
            in each period. Default is 3
        n_primary_period: int
            how many periods used as primary date of ifg. Default is 2. For example, if n_primary_period=2,
            then the interferometric pairs will be generated between the first two periods and the rest
            periods. 

        Returns
        -------
        pairs: Pairs object
        '''

        years = sorted(set(self.dates.year))
        df_dates = pd.Series(self.dates.strftime('%Y%m%d'), index=self.dates)

        # check if period_start and period_end are in the same year. If not, the period_end should be
        # in the next year
        same_year = False
        if int(period_start) < int(period_end):
            same_year = True

        # randomly select n_per_period dates in each period/year
        date_years = []
        for year in years:
            start = pd.to_datetime(f'{year}{period_start}', format='%Y%m%d')
            if same_year:
                end = pd.to_datetime(f'{year}{period_end}', format='%Y%m%d')
            else:
                end = pd.to_datetime(f'{year+1}{period_end}', format='%Y%m%d')

            dt_year = df_dates[start:end]
            if len(dt_year) > 0:
                np.random.shuffle(dt_year)
                date_years.append(dt_year[:n_per_period].to_list())

        # generate interferometric pairs between primary period and the rest periods
        _pairs = []
        for i, date_year in enumerate(date_years):
            # only generate pairs for n_primary_period
            if i+1 > n_primary_period:
                break
            for date_primary in date_year:
                # all rest periods
                for date_year1 in date_years[i+1:]:
                    for date_secondary in date_year1:
                        pair = (date_primary, date_secondary)
                        _pairs.append(pair)

        return Pairs(_pairs)

    def from_summer_winter(
        self,
        dates_str: str,
        date_format: str = '%Y%m%d',
        summer_start: str = '0801',
        summer_end: str = '1001',
        winter_start: str = '1201',
        winter_end: str = '0331'
    ) -> Pairs:
        '''generate interferometric pairs between summer and winter in each yea. summer and winter
        are defined by month and day for each year. For example, summer_start='0801', summer_end='1001'
        means the summer is from Aug 1 to Oct 1 for each year in the time series. This will be useful
        to add pairs for whole thawing and freezing process.

        Parameters
        ---------
        dates_str: str 
            date string with format of '%Y%m%d'
        date_format : str
            format of dates string, which is used to convert string to datetime object. 
            Default is '%Y%m%d'. More information can be found at:
            https://docs.python.org/3.11/library/datetime.html#strftime-strptime-behavior
        summer_start, summer_end:  str
            start and end date for the summer which expressed as month and day with format '%m%d'
        winter_start, winter_end:  str
            start and end date for the winter which expressed as month and day with format '%m%d'

        Returns
        -------
        _pairs: a list containing interferometric pairs with each pair as a tuple of two dates
        '''
        years = sorted(set(self.dates.year))
        df_dates = pd.Series(self.dates.strftime(date_format),
                             index=self.dates)

        _pairs = []
        for year in years:
            s_start = pd.to_datetime(f'{year}{summer_start}', format='%Y%m%d')
            s_end = pd.to_datetime(f'{year}{summer_end}', format='%Y%m%d')

            if int(winter_start) > int(summer_end):
                w_start1 = pd.to_datetime(
                    f'{year-1}{winter_start}', format='%Y%m%d')
                w_start2 = pd.to_datetime(
                    f'{year}{winter_start}', format='%Y%m%d')
                if int(winter_end) > int(summer_end):
                    w_end1 = pd.to_datetime(
                        f'{year-1}{winter_end}', format='%Y%m%d')
                    w_end2 = pd.to_datetime(
                        f'{year}{winter_end}', format='%Y%m%d')
                else:
                    w_end1 = pd.to_datetime(
                        f'{year}{winter_end}', format='%Y%m%d')
                    w_end2 = pd.to_datetime(
                        f'{year+1}{winter_end}', format='%Y%m%d')
            else:
                w_start1 = pd.to_datetime(
                    f'{year}{winter_start}', format='%Y%m%d')
                w_start2 = pd.to_datetime(
                    f'{year+1}{winter_start}', format='%Y%m%d')

                w_end1 = pd.to_datetime(f'{year}{winter_end}', format='%Y%m%d')
                w_end2 = pd.to_datetime(
                    f'{year+1}{winter_end}', format='%Y%m%d')

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
    def season_of_month(month):
        '''return the season of a given month

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
        '''
        month = int(month)
        if month not in list(range(1, 13)):
            raise ValueError('Month should be in range 1-12.'
                             f" But got '{month}'.")
        season = (month-3) % 12 // 3 + 1
        return season

    @staticmethod
    def ensure_datetime(date: Any) -> datetime:
        '''ensure the date is a datetime object

        Parameters
        ----------
        date: datetime or str
            Date to be ensured.

        Returns
        -------
        date: datetime
            Date with format of datetime.
        '''
        if isinstance(date, datetime):
            pass
        elif isinstance(date, str):
            date = pd.to_datetime(date)
        else:
            raise TypeError(
                f"Date should be datetime or str, but got {type(date)}")
        return date

    @staticmethod
    def str_to_dates(
        date_str: str,
        length: int = 2,
        parse_function: Optional[Callable] = None,
        date_args: Optional[dict] = None
    ):
        '''convert date string to dates

        Parameters
        ----------
        date_str: str
            Date string containing dates.
        length: int, optional
            Length/number of dates in the date string. Default is 2.
        parse_function: Callable, optional
            Function to parse the date strings from the date string.
            If None, the date string will be split by '_' and the
            last 2 items will be used. Default is None.
        date_args: dict, optional
            Keyword arguments for pd.to_datetime() to convert the date strings
            to datetime objects. For example, {'format': '%Y%m%d'}.
            Default is None.        
        '''
        if parse_function is not None:
            dates = parse_function(date_str)
        else:
            items = date_str.split('_')
            if len(items) >= length:
                dates_ls = items[-length:]
            else:
                raise ValueError(
                    f'The number of dates in {date_str} is less than {length}.')

        if date_args is None:
            date_args = {}
        date_args.update({"errors": 'raise'})

        try:
            dates = [pd.to_datetime(i, **date_args) for i in dates_ls]
        except:
            raise ValueError(f'Dates in {date_str} not recognized.')

        return tuple(dates)


if __name__ == '__main__':
    names = ['20170111_20170204',
             '20170111_20170222',
             '20170111_20170318',
             '20170204_20170222',
             '20170204_20170318',
             '20170204_20170330',
             '20170222_20170318',
             '20170222_20170330',
             '20170222_20170411',
             '20170318_20170330']

    pairs = Pairs.from_names(names)
    loops = pairs.to_loops()
    pairs1 = loops.pairs
