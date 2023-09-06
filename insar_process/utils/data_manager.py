from rasterio import Affine, dtypes, transform
from typing import Union
import pprint
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
from collections.abc import Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio import Affine
from rasterio.crs import CRS


class SBASNetwork:
    '''SBAS network class to handle interferograms and loops for a given directory.

    Parameters
    ----------
    ifg_dir : Path or str
        Path to directory containing interferograms.
    type : str, optional
        Type of interferogram. The default is 'hyp3'.
    '''

    def __init__(self, pairs) -> None:
        self.pairs = pairs

    @property
    def dates(self) -> List[str]:
        dates = set()
        for pair in self.pairs:
            dates.update(pair)
        return sorted(dates)

    @property
    def loop_matrix(self) -> np.ndarray:
        """
        Make loop matrix (containing 1, -1, 0) from ifg_dates.

        Returns
        -------
        Loops : Loop matrix with 1 for pair12/pair23 and -1 for pair13
                with the shape of (n_loop, n_ifg)
        """
        n_ifg = len(self.pairs)
        Loops = []
        for idx_pair12, pair12 in enumerate(self.pairs):
            pairs23 = [
                pair for pair in self.pairs if pair[0] == pair12[1]
            ]  # all candidates of ifg23

            for pair23 in pairs23:  # for each candidate of ifg23
                try:
                    idx_pair13 = self.pairs.index((pair12[0], pair23[1]))
                except:  # no loop for this ifg23. Next.
                    continue

                # Loop found
                idx_pair23 = self.pairs.index(pair23)

                loop = np.zeros(n_ifg)
                loop[idx_pair12] = 1
                loop[idx_pair23] = 1
                loop[idx_pair13] = -1
                Loops.append(loop)

        return np.array(Loops)

    @classmethod
    def from_ifg_dir(cls, ifg_dir, ifg_type='hyp3') -> 'SBASNetwork':
        '''initialize SBASNetwork class from a directory of interferograms

        Parameters
        ----------
        ifg_dir : Path or str
            Path to directory containing interferograms.
        ifg_type : str, optional
            Type of interferogram. The default is 'hyp3'.
        '''
        ifg_dir = Path(ifg_dir)
        if not ifg_dir.exists():
            raise FileNotFoundError(f"{ifg_dir} not found")

        if ifg_type == 'hyp3':
            ifg_names = [i.name for i in ifg_dir.iterdir()]

            def _name2pair(name):
                pair = name.split("_")[1:3]
                pair = pair[0][:8], pair[1][:8]
                return pair
            pairs = [_name2pair(i) for i in ifg_names]
        else:
            # TODO: add other types
            pass

        return cls(pairs)

    @classmethod
    def from_names(cls, ifg_names) -> 'SBASNetwork':
        '''initialize SBASNetwork class from a list of pair names

        Parameters
        ----------
        ifg_names: list
            List of pair names. Each pair name should be in the format 
            of 'yyyymmdd_yyyymmdd'.
        '''
        if not isinstance(ifg_names, list):
            raise TypeError('ifg_names should be a list.')
        pairs = pairs = [tuple(name.split("_")) for name in ifg_names]
        return cls(pairs)

    @property
    def loop_info(self):
        '''print loop information'''
        ns_loop4ifg = np.abs(self.values_matrix).sum(axis=0)
        idx_pairs_no_loop = np.where(ns_loop4ifg == 0)[0]
        no_loop_pair = [self.pairs[ix] for ix in idx_pairs_no_loop]
        print(f"Number of interferograms: {len(self.pairs)}")
        print(f"Number of loops: {self.values_matrix.shape[0]}")
        print(f"Number of dates: {len(self.dates)}")
        print(f"Number of loops per date: {len(self.pairs)/len(self.dates)}")
        print(f"Number of interferograms without loop: {len(no_loop_pair)}")
        print(f"Interferograms without loop: {no_loop_pair}")

    def dir_of_pair(self, pair) -> Path:
        '''return path to pair directory for a given pair'''
        name = self.ifg_names[self.pairs.index(pair)]
        return self.ifg_dir / name

    def unw_file_of_pair(self, pair, pattern="*unw_phase.tif") -> Path:
        '''return path to unw file for a given pair

        Parameters
        ----------
        pair : tuple
            Pair of dates.
        pattern : str, optional
            Pattern of unwrapped phase file. The default is "*unw_phase.tif" (hyp3).
            This is used to find the unwrapped phase file in the pair directory.
        '''
        dir_of_pair = self.dir_of_pair(pair)
        try:
            unw_file = list(dir_of_pair.glob(pattern))[0]
        except:
            print(f"Unwrapped phase file not found in {dir_of_pair}")
            unw_file = None
        return unw_file

    def pairs_of_loop(self, loop) -> List[Tuple[str, str]]:
        '''return the 3 pairs of the given loop'''
        idx_pair12 = np.where(loop == 1)[0][0]
        idx_pair23 = np.where(loop == 1)[0][1]
        idx_pair13 = np.where(loop == -1)[0][0]
        pair12 = self.pairs[idx_pair12]
        pair23 = self.pairs[idx_pair23]
        pair13 = self.pairs[idx_pair13]
        return [pair12, pair23, pair13]

    def pairs_of_date(self, date) -> List[Tuple[str, str]]:
        '''return all pairs of a given date'''
        pairs = [pair for pair in self.pairs if date in pair]
        return pairs


class Pair:
    '''Pair class for one pair.

    Parameters
    ----------
    pair: Iterable
        Iterable object of two dates. Each date is a datetime object.
        For example, (date1, date2).
    '''
    _values: np.ndarray
    _name: str

    __slots__ = ['_values', '_name']

    def __init__(
        self,
        pair: Iterable[datetime, datetime]
    ) -> None:
        self._values = np.array(sorted(pair), dtype='M8[D]')
        self._name = '_'.join([i.strftime('%Y%m%d')
                              for i in self._values.astype('O')])

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Pair({self.name})"

    def __eq__(self, other: 'Pair') -> bool:
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

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
        dates = str_to_dates(name, 2, parse_function, date_args)
        return cls(dates)


class Pairs:
    """Pairs class to handle pairs

    Parameters
    ----------
    pairs: Iterable
        Iterable object of pairs. Each pair is an Iterable or Pair
        object of two dates with format of datetime. For example, 
        [(date1, date2), ...].
    """

    _values: np.ndarray
    _dates: np.ndarray
    _length: int

    __slots__ = ['_values', '_dates', '_length']

    def __init__(
        self,
        pairs: Union[Iterable[Iterable[datetime, datetime]], Iterable[Pair]]
    ) -> None:
        pairs_set = set()
        for pair in pairs:
            if isinstance(pair, Pair):
                _pair = pair
            elif isinstance(pair, Iterable):
                _pair = Pair(pair)
            else:
                raise TypeError(
                    f"pairs should be an Iterable of Pair or Iterable of Iterable, but got {type(pair)}")
            pairs_set.add(_pair)

        dates_set = set()
        pair_ls = []
        for pair in pairs_set:
            pair_ls.append(pair.values)
            dates_set.update(pair.values.astype('O').tolist())

        self._values = np.sort(pair_ls, axis=0)
        self._length = self._values.shape[0]
        self._dates = np.array(sorted(dates_set), dtype='M8[D]')

    def __len__(self) -> int:
        return self._length

    def __str__(self) -> str:
        return self.to_frame().__repr__()

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: 'Pairs') -> bool:
        return np.array_equal(self.values, other.values)

    def __getitem__(self, index: int) -> Union['Pair', 'Pairs']:
        if isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            if (isinstance(index.start, (int, np.integer, type(None)))
                    and isinstance(index.stop, (int, np.integer, type(None)))):
                if index.start is None:
                    start = 0
                if index.stop is None:
                    stop = self._length
                return Pairs(self._values[start:stop:step])
            elif (isinstance(index.start, (datetime, np.datetime64, pd.Timestamp, str, type(None)))
                  and isinstance(index.stop, (datetime, np.datetime64, pd.Timestamp, str, type(None)))):
                if isinstance(index.start, str):
                    start = DateManager.ensure_datetime(index.start)
                if isinstance(index.stop, str):
                    stop = DateManager.ensure_datetime(index.stop)
                if index.start is None:
                    start = self._dates[0]
                if index.stop is None:
                    stop = self._dates[-1]
                    
                start, stop = np.datetime64(start), np.datetime64(stop)
                    
                if start > stop:
                    raise ValueError(
                        f"Index start {start} should be earlier than index stop {stop}.")
                pairs = []
                for pair in self._values:
                    pair = pair.astype('M8[s]')
                    if start <= pair[0] <= stop and start <= pair[1] <= stop:
                        pairs.append(pair)
                if len(pairs) > 0:
                    return Pairs(pairs)
                else:
                    return None
        elif isinstance(index, int):
            if index >= self._length:
                raise IndexError(
                    f"Index {index} out of range. Pairs number is {self._length}.")
            return Pair(self._values[index])
        elif isinstance(index, Iterable) and not isinstance(index, str):
            index = np.array(index)
            if not index.ndim == 1:
                raise IndexError(
                    f"Index should be 1D array, but got {index.ndim}D array.")
            if not index.dtype == bool:
                raise TypeError(
                    f"Index should be bool array, but got {index.dtype}.")
            if not len(index) == self._length:
                raise IndexError(
                    f"Index length should be equal to pairs length {self._length}"
                    f" for boolean indexing, but got {len(index)}.")
            return Pairs(self._values[index])
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
        else:
            raise TypeError(
                f"Index should be int, slice, datetime, str, or 1D bool array, but got {type(index)}.")

    def __hash__(self) -> int:
        return hash(self.values)

    def __iter__(self):
        return iter(self.values)

    def __contains__(self, item):
        return item in self.values

    @property
    def values(self) -> np.ndarray:
        '''return the values of the pairs with format of np.datetime64[D]'''
        return self._values

    @property
    def dates(self) -> np.ndarray:
        '''return the dates of the pairs with format of np.datetime64[D]'''
        return self._dates

    @property
    def shape(self) -> Tuple[int, int]:
        '''return the shape of the pairs'''
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
            Pairs class.
        '''
        pairs = []
        for name in names:
            pair = Pair.from_name(name, parse_function, date_args)
            pairs.append(pair.values)
        return cls(pairs)

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

    def dates_string(self, format='%Y%m%d') -> List[str]:
        '''return the dates of the pairs with format of str

        Parameters
        ----------
        format: str
            Format of the date string. Default is '%Y%m%d'.
        '''
        return [i.strftime(format) for i in self._dates.astype(datetime)]


class Loop:
    '''Loop class containing three pairs.

    Parameters
    ----------
    loop: Iterable
        Iterable object of three dates. Each date is a datetime object.
        For example, (date1, date2, date3).
    '''
    _values: np.ndarray
    _name: str
    _pairs: List[Pair]

    __slots__ = ['_values', '_pairs', '_name']

    def __init__(
        self,
        loop: Iterable[datetime, datetime, datetime]
    ) -> None:
        self._values = np.array(sorted(loop), dtype='M8[D]')
        loop_dt = self._values.astype(datetime)
        self._name = '_'.join([i.strftime('%Y%m%d') for i in loop_dt])
        self._pairs = [
            Pair([loop_dt[0], loop_dt[1]]),
            Pair([loop_dt[1], loop_dt[2]]),
            Pair([loop_dt[0], loop_dt[2]])
        ]

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Loop({self.name})"

    def __eq__(self, other) -> bool:
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def values(self):
        '''return the values of the loop.

        Returns
        -------
        values: np.ndarray
            Values of the loop with format of datetime.
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
    def name(self) -> str:
        '''return the string of the loop.

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
        '''initialize the loop class from a loop file name

        Parameters
        ----------
        name: str
            Loop file name.
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
        loop: Loop
            Loop class.
        '''
        dates = str_to_dates(name, 3, parse_function, date_args)
        return cls(dates)


class Loops:
    '''Loops class to handle loops'''

    def __init__(
        self,
        loops: Iterable[Iterable[datetime, datetime, datetime]],
    ) -> None:
        '''initialize the loops class

        Parameters
        ----------
        loops: Iterable
            Iterable object of loops. Each loop is an Iterable object of three dates
            with format of datetime. For example, [(date1, date2, date3), ...].
        '''
        loops_sorted = []
        dates = set()
        dates_str = set()
        for _loop in loops:
            _loop_sorted = sorted(_loop)
            loops_sorted.append(_loop_sorted)
            for _date in _loop_sorted:
                dates.add(_date)
                dates_str.add(_date.strftime('%Y%m%d'))

        self._values = np.array(loops_sorted, dtype='M8[D]')
        self._dates = np.array(sorted(dates), dtype='M8[D]')
        self._dates_str = np.array(sorted(dates_str), dtype=np.string_)

    @property
    def values(self):
        '''return the values of the loops.

        Returns
        -------
        values: np.ndarray
            Values of the loops with format of datetime.
        '''
        return self._values

    @classmethod
    def from_files(
        cls,
        loop_dir: Union[str, Path],
        parse_function: Optional[Callable] = None,
        date_args: Optional[dict] = None
    ) -> 'Loops':
        '''initialize the loops class from a directory of loop files 

        Parameters
        ----------
        loops_dir: str or Path
            Path to the directory containing all the loop files.
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
            Loops class.
        '''
        loop_dir = Path(loop_dir)
        loop_files = list(loop_dir.iterdir())
        loops = [cls.dates_of_loop(i, parse_function, date_args)
                 for i in loop_files]
        return cls(loops)

    @classmethod
    def from_names(
        cls,
        names: List[str],
        parse_function: Optional[Callable] = None,
        date_args: Optional[dict] = None,
    ) -> 'Loops':
        '''initialize the loops class from a list of loop file names

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
            Loops class.
        '''
        loops = [cls.dates_of_loop(i, parse_function, date_args)
                 for i in names]
        return cls(loops)

    @staticmethod
    def dates_of_loop(
        loop: Union[str, Path],
        parse_function: Optional[Callable] = None,
        date_args: Optional[dict] = None,
    ) -> Tuple[datetime, datetime, datetime]:
        '''parse three Dates from loop file name

        Parameters
        ----------
        loop: str or Path
            loop name or Path to the loop file .
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
        date1, date2, date3: datetime
            Three Dates of the loop file with format of datetime.
        '''
        name = Path(loop).stem
        if parse_function is not None:
            date1, date2, date3 = parse_function(name)
        else:
            items = name.split('_')
            if len(items) >= 3:
                date1, date2, date3 = items[-3:]
                if not (len(date1) == len(date2) == len(date3)):
                    raise ValueError(
                        f'Loop file name {loop} not recognized.')
            else:
                raise ValueError(f'Loop file name {loop} not recognized.')

        if date_args is None:
            date_args = {}
        date_args.update({"errors": 'raise'})

        date1 = pd.to_datetime(date1, **date_args)
        date2 = pd.to_datetime(date2, **date_args)
        date3 = pd.to_datetime(date3, **date_args)
        return date1, date2, date3

    @property
    def loops_str(self):
        '''return the string of each loop.

        Returns
        -------
        loops_str: list
            List of strings of each loop.
        '''
        loops_str = []
        for loop in self.values:
            loops_str.append('_'.join([i.strftime('%Y%m%d') for i in loop]))
        return loops_str

    def generate_loops_str(self, prefix='loop_'):
        '''generate loop file names with prefix

        Parameters
        ----------
        prefix: str
            Prefix of the loop file names. Default is 'loop_'.
        '''
        return [prefix+i for i in self.valuess_str]

    @property
    def dates(self) -> np.ndarray:
        '''return all dates in the loops.

        Returns
        -------
        dates: np.ndarray
            Dates of the loops with format of datetime.
        '''
        return self._dates

    @property
    def dates_str(self):
        '''return all  dates in the loops with format of "%Y%m%d".

        Returns
        -------
        dates: np.ndarray
            Dates of the loops with format of "%Y%m%d".
        '''
        return self._dates_str

    @property
    def pairs(self):
        # TODO: Using Pairs class
        '''return all pairs in the loops.

        Returns
        -------
        pairs: list
            List of pairs with format of ('%Y%m%d', '%Y%m%d').
        '''
        pairs = set()
        for loop in self.valuess:
            loop = [i.strftime('%Y%m%d') for i in loop]
            pairs.update(
                [(loop[0], loop[1]),
                 (loop[1], loop[2]),
                 (loop[0], loop[2])]
            )
        return sorted(pairs)

    @property
    def loop_matrix(self):
        """
        Make loop matrix (containing 1, -1, 0) from ifg_dates.

        Returns
        -------
        Loops : Loop matrix with 1 for pair12/pair23 and -1 for pair13
                with the shape of (n_loop, n_ifg)
        """
        n_ifg = len(self.pairs)
        Loops = []
        for idx_pair12, pair12 in enumerate(self.pairs):
            pairs23 = [
                pair for pair in self.pairs if pair[0] == pair12[1]
            ]  # all candidates of ifg23

            for pair23 in pairs23:  # for each candidate of ifg23
                try:
                    idx_pair13 = self.pairs.index((pair12[0], pair23[1]))
                except:  # no loop for this ifg23. Next.
                    continue

                # Loop found
                idx_pair23 = self.pairs.index(pair23)

                loop = np.zeros(n_ifg)
                loop[idx_pair12] = 1
                loop[idx_pair23] = 1
                loop[idx_pair13] = -1
                Loops.append(loop)

        return np.array(Loops)

    @property
    def seasons(self):
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
        for loop in self.valuess:
            season1 = DateManager.season_of_month(loop[0].month)
            season2 = DateManager.season_of_month(loop[1].month)
            season3 = DateManager.season_of_month(loop[2].month)
            if season1 == season2 == season3:
                seasons.append(season1)
            else:
                seasons.append(0)
        return seasons

    @property
    def day_range_max(self):
        '''return the day range of each loop.

        Returns
        -------
        day_range: list
            List of day range of each loop.
        '''
        day_range = []
        for loop in self.valuess:
            range1 = abs((loop[0]-loop[1]).days)
            range2 = abs((loop[1]-loop[2]).days)
            range3 = abs((loop[2]-loop[0]).days)
            day_range.append(max(range1, range2, range3))
        return day_range

    @property
    def day_range_min(self):
        '''return the day range of each loop.

        Returns
        -------
        day_range: list
            List of day range of each loop.
        '''
        day_range = []
        for loop in self.valuess:
            range1 = abs((loop[0]-loop[1]).days)
            range2 = abs((loop[1]-loop[2]).days)
            range3 = abs((loop[2]-loop[0]).days)
            day_range.append(min(range1, range2, range3))
        return day_range


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
    def ensure_datetime(date) -> datetime:
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


class GeoDataFormatConverter:
    '''A class to convert data format between raster and binary.

    Examples:
    --------
    >>> from pathlib import Path
    >>> from data_tool import GeoDataFormatConverter
    >>> phase_file = Path("phase.tif")
    >>> amplitude_file = Path("amplitude.tif")
    >>> binary_file = Path("phase.int")


    ### load/add raster and convert to binary
    >>> gfc = GeoDataFormatConverter()
    >>> gfc.load_raster(phase_file)
    >>> gfc.add_band_from_raster(amplitude_file)
    >>> gfc.to_binary(binary_file)

    ### load binary file
    >>> gfc.load_binary(binary_file)
    >>> print(gfc.arr.shape)
    '''

    def __init__(self) -> None:
        self.arr: np.ndarray = None
        self.profile: dict = None

    @property
    def _profile_str(self):
        return pprint.pformat(self.profile, sort_dicts=False)

    def __str__(self) -> str:
        return f"DataConverter: \n{self._profile_str}"

    def __repr__(self) -> str:
        return str(self)

    def _load_raster(self, raster_file: Union[str, Path]):
        '''Load a raster file into the data array.'''
        with rasterio.open(raster_file) as ds:
            arr = ds.read()
            profile = ds.profile.copy()
        return arr, profile

    def load_binary(self, binary_file: Union[str, Path], order='BSQ', dtype='auto'):
        '''Load a binary file into the data array.

        Parameters
        ----------
        binary_file : str or Path
            The binary file to be loaded. the binary file should be with a profile file with the same name.
        order : str, one of ['BSQ', 'BIP', 'BIL']
            The order of the data array. 'BSQ' for band sequential, 'BIP' for band interleaved by pixel, 
            'BIL' for band interleaved by line. Default is 'BSQ'. 
            See: https://desktop.arcgis.com/zh-cn/arcmap/latest/manage-data/raster-and-images/bil-bip-and-bsq-raster-files.htm
        '''
        binary_profile_file = str(binary_file) + '.profile'
        if not Path(binary_profile_file).exists():
            raise FileNotFoundError(f"{binary_profile_file} not found")

        with open(binary_profile_file, 'r') as f:
            profile = eval(f.read())

        # todo: auto detect dtype by shape
        if dtype == 'auto':
            dtype = np.float32

        arr = np.fromfile(binary_file, dtype=dtype)
        if order == 'BSQ':
            arr = arr.reshape(profile['count'],
                              profile['height'],
                              profile['width'])
        elif order == 'BIP':
            arr = (arr.reshape(profile['height'],
                               profile['width'],
                               profile['count'])
                   .transpose(2, 0, 1))
        elif order == 'BIL':
            arr = (arr.reshape(profile['height'],
                               profile['count'],
                               profile['width'])
                   .transpose(1, 0, 2))
        else:
            raise ValueError(
                "order should be one of ['BSQ', 'BIP', 'BIL'],"
                f" but got {order}"
            )

        if 'dtype' not in profile:
            profile['dtype'] = dtypes.get_minimum_dtype(arr)

        self.arr = arr
        self.profile = profile

    def load_raster(self, raster_file: Union[str, Path]):
        '''Load a raster file into the data array.

        Parameters
        ----------
        raster_file : str or Path
            The raster file to be loaded. raster format should be supported by gdal. 
            See: https://gdal.org/drivers/raster/index.html
        '''
        self.arr, self.profile = self._load_raster(raster_file)

    def to_binary(self, out_file: Union[str, Path], order='BSQ'):
        '''Write the data array into a binary file.

        Parameters
        ----------
        out_file : str or Path
            The binary file to be written. the binary file will be with a profile file with the same name.
        order : str, one of ['BSQ', 'BIP', 'BIL']
            The order of the data array. 'BSQ' for band sequential, 'BIP' for band interleaved by pixel, 
            'BIL' for band interleaved by line. Default is 'BSQ'. 
            See: https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/bil-bip-and-bsq-raster-files.htm
        '''
        if order == 'BSQ':
            arr = self.arr
        elif order == 'BIL':
            arr = np.transpose(self.arr, (1, 2, 0))
        elif order == 'BIP':
            arr = np.transpose(self.arr, (1, 0, 2))

        # write data into a binary file
        (arr.astype(np.float32)
         .tofile(out_file))

        # write profile into a file with the same name
        out_profile_file = str(out_file) + '.profile'
        with open(out_profile_file, 'w') as f:
            f.write(self._profile_str)

    def to_raster(self, out_file: Union[str, Path], driver='GTiff'):
        '''Write the data array into a raster file.

        Parameters
        ----------
        out_file : str or Path
            The raster file to be written. 
        driver : str
            The driver to be used to write the raster file. See: https://gdal.org/drivers/raster/index.html
        '''
        self.profile.update({'driver': driver})
        with rasterio.open(out_file, 'w', **self.profile) as ds:
            bands = range(1, self.profile['count']+1)
            ds.write(self.arr, bands)

    def add_band(self, arr: np.ndarray):
        '''Add a band to the data array.

        Parameters
        ----------
        arr : 2D or 3D numpy.ndarray
            The array to be added. The shape of the array should be (height, width) or (band, height, width).
        '''
        if not isinstance(arr, np.ndarray):
            try:
                arr = np.array(arr)
            except:
                raise TypeError("arr can not be converted to numpy array")

        if len(arr.shape) == 2:
            arr = np.concatenate((self.arr, arr[None, :, :]), axis=0)
        if len(arr.shape) == 3:
            arr = np.concatenate((self.arr, arr), axis=0)

        self.update_arr(arr)

    def add_band_from_raster(self, raster_file: Union[str, Path]):
        '''Add band to the data array from a raster file.

        Parameters
        ----------
        raster_file : str or Path
            The raster file to be added. raster format should be supported by gdal. See: https://gdal.org/drivers/raster/index.html
        '''
        arr, profile = self._load_raster(raster_file)
        self.add_band(arr)

    def add_band_from_binary(self, binary_file: Union[str, Path]):
        '''Add band to the data array from a binary file.

        Parameters
        ----------
        binary_file : str or Path
            The binary file to be added. the binary file should be with a profile file with the same name.
        '''
        arr, profile = self._load_binary(binary_file)
        self.add_band(arr)

    def update_arr(
        self,
        arr: np.ndarray,
        dtype: str = 'auto',
        nodata: Union[int, float, None, str] = 'auto',
        error_if_nodata_invalid: bool = True,
    ):
        '''update the data array.

        Parameters
        ----------
        arr : numpy.ndarray
            The array to be updated. The profile will be updated accordingly.
        dtype : str or numpy.dtype
            The dtype of the array. If 'auto', the minimum dtype will be used. Default is 'auto'.
        nodata : int, float, None or 'auto'
            The nodata value of the array. If 'auto', the nodata value will be set to the nodata value of the profile if valid, otherwise None. Default is 'auto'.
        error_if_nodata_invalid : bool
            Whether to raise error if nodata is out of dtype range. Default is True.
        '''
        self.arr = arr
        if not hasattr(self, 'profile'):
            raise AttributeError("profile is not set yet")

        # update profile info
        self.profile['count'] = arr.shape[0]
        self.profile['height'] = arr.shape[1]
        self.profile['width'] = arr.shape[2]

        if dtype == 'auto':
            self.profile['dtype'] = dtypes.get_minimum_dtype(arr)
        else:
            if not dtypes.check_dtype(dtype):
                raise ValueError(f"dtype {dtype} is not supported")
            self.profile['dtype'] = dtype

        if nodata == 'auto':
            nodata = self.profile['nodata']
            error_if_nodata_invalid = False

        if nodata is None:
            self.profile['nodata'] = None
        else:
            dtype_ranges = dtypes.dtype_ranges[self.profile['dtype']]
            if dtypes.in_dtype_range(nodata, self.profile['dtype']):
                self.profile['nodata'] = nodata
            else:
                if error_if_nodata_invalid:
                    raise ValueError(
                        f"nodata {nodata} is out of dtype range {dtype_ranges}")
                else:
                    print('Warning: nodata is out of dtype range, '
                          'nodata will be set to None')
                    self.profile['nodata'] = None


class PhaseDeformationConverter:
    '''A class to convert between phase and deformation(LOS) for SAR interferometry.'''

    def __init__(self, wavelength: float = None, frequency: float = None) -> None:
        '''Initialize the converter. Either wavelength or frequency should be provided. If both are provided, wavelength will be recalculated by frequency.

        Parameters
        ----------
        wavelength : float
            The wavelength of the radar signal. Unit: meter.
        frequency : float
            The frequency of the radar signal. Unit: Hz.
        '''
        speed_of_light = 299792458
        if wavelength is None and frequency is None:
            raise ValueError(
                "Either wavelength or frequency should be provided.")
        elif frequency is not None:
            self.wavelength = speed_of_light/frequency  # meter
            self.frequency = frequency
        elif wavelength is not None and frequency is None:
            self.frequency = speed_of_light/wavelength
            self.wavelength = wavelength
        else:
            raise ValueError(
                "Either wavelength or frequency should be provided.")

        # convert radian to mm
        self.coef_rd2mm = - wavelength/4/np.pi*1000

    def __str__(self) -> str:
        return f"PhaseDeformationConverter(wavelength={self.wavelength})"

    def __repr__(self) -> str:
        return str(self)

    def phase2deformation(self, phase: np.ndarray):
        return phase * self.coef_rd2mm

    def deformation2phase(self, deformation: np.ndarray):
        return deformation / self.coef_rd2mm

    def wrap_phase(self, phase: np.ndarray):
        return np.mod(phase, 2*np.pi)


class Profile:
    '''a class to manage the profile of a raster file. 
    The profile is the metadata of the raster file and 
    can be recognized by rasterio package'''

    def __init__(self, profile: dict = None) -> None:
        self.profile = profile

    def __str__(self) -> str:
        return pprint.pformat(self.profile, sort_dicts=False)

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, key):
        return self.profile[key]

    def __setitem__(self, key, value):
        self.profile[key] = value

    def __contains__(self, key):
        return key in self.profile

    def __iter__(self):
        return iter(self.profile)

    def __len__(self):
        return len(self.profile)

    def __delitem__(self, key):
        del self.profile[key]

    def __eq__(self, other):
        return self.profile == other.profile

    def __ne__(self, other):
        return self.profile != other.profile

    @classmethod
    def from_raster_file(cls, raster_file: Union[str, Path]):
        '''Create a Profile object from a raster file.'''
        with rasterio.open(raster_file) as ds:
            profile = ds.profile.copy()
        return cls(profile)

    @classmethod
    def from_ascii_header_file(cls, ascii_file: Union[str, Path]):
        '''Create a Profile object from an ascii header file. 
        The ascii header file is the metadata of a binary.
        More information can be found at: https://desktop.arcgis.com/zh-cn/arcmap/latest/manage-data/raster-and-images/esri-ascii-raster-format.htm

        Example of an ascii header file
        -------------------------------
        ncols         43200
        nrows         18000
        xllcorner     -180.000000
        yllcorner     -60.000000
        cellsize      0.008333
        NODATA_value  -9999
        '''
        df = pd.read_csv(ascii_file, sep='\s+', header=None, index_col=0)
        df.index = df.index.str.lower()

        width = int(df.loc['ncols', 1])
        height = int(df.loc['nrows', 1])
        cell_size = float(df.loc['cellsize', 1])
        try:
            left = float(df.loc['xllcorner', 1])
            bottom = float(df.loc['yllcorner', 1])
        except:
            left = float(df.loc['xllcenter', 1]) - cell_size/2
            bottom = float(df.loc['yllcenter', 1]) - cell_size/2

        # pixel left lower corner to pixel left upper corner (rasterio transform)
        top = bottom + (height+1) * cell_size

        tf = transform.from_origin(left, top, cell_size, cell_size)

        nodata = None
        if 'nodata_value' in df.index:
            nodata = float(df.loc['nodata_value', 1])

        profile = {
            'width': width,
            'height': height,
            'transform': tf,
            'count': 1,
            'nodata': nodata,
        }

        return cls(profile)

    @classmethod
    def from_dict(cls, profile: dict):
        '''Create a Profile object from a dict.'''
        return cls(profile)

    @classmethod
    def from_profile_file(cls, profile_file: Union[str, Path]):
        '''Create a Profile object from a profile file.'''
        with open(profile_file, 'r') as f:
            profile = eval(f.read())
        return cls(profile)

    def to_file(self, file: Union[str, Path]):
        '''Write the profile into a file.'''
        file = Path(file)
        if file.suffix != '.profile':
            file = file.parent / (file.name + '.profile')
        with open(file, 'w') as f:
            f.write(str(self))


def str_to_dates(
    date_str: str,
    length: int = 2,
    parse_function: Optional[Callable] = None,
    date_args: Optional[dict] = None
):
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
    dates = pd.date_range('20130101', '20231231').values
    n = len(dates)
    pair_ls = []
    loop_ls = []
    for i in range(5):
        pair_ls.append(dates[np.random.randint(0, n, 2)])
        loop_ls.append(dates[np.random.randint(0, n, 3)])

    pairs = Pairs(pair_ls)
    print(pairs['2015-03-09':])
    # loops = Loops(loop_ls)
