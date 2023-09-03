from typing import Union
import pprint
from rasterio.crs import CRS
from rasterio import Affine
import rasterio
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional, Callable
from datetime import datetime


class SBASNetwork:
    '''SBAS network class to handle interferograms and loops for a given directory.

    Parameters
    ----------
    ifg_dir : Path or str
        Path to directory containing interferograms.
    type : str, optional
        Type of interferogram. The default is 'hyp3'.
    '''

    def __init__(self, ifg_dir, type='hyp3') -> None:

        self.ifg_dir = ifg_dir
        self.ifg_names = [i.name for i in ifg_dir.iterdir()]
        if type == 'hyp3':
            self.pairs = self._get_hyp3_ifg_pairs()
        else:
            # TODO: add support for other types
            raise ValueError("Only 'hyp3' is supported for now.")

        self.dates = self._get_dates()
        self.loop_matrix = self._make_loop_matrix()

    def _get_hyp3_ifg_pairs(self):
        def _name2pair(name):
            pair = name.split("_")[1:3]
            pair = pair[0][:8], pair[1][:8]
            return pair

        pairs = [_name2pair(i) for i in self.ifg_names]
        return pairs

    def _get_dates(self):
        dates = set()
        for pair in self.pairs:
            dates.update(pair)
        return dates

    def _make_loop_matrix(self):
        """
        Make loop matrix (containing 1, -1, 0) from ifg_dates.

        Returns:
        Loops : Loop matrix with 1 for pair12/pair23 and -1 for pair13
                (n_loop, n_ifg)

        """
        n_ifg = len(self.pairs)
        Loops = []

        for idx_pair12, pair12 in enumerate(self.pairs):
            pairs23 = [pair for pair in self.pairs if pair[0]
                       == pair12[1]]  # all candidates of ifg23

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
    def loop_info(self):
        '''print loop information'''
        ns_loop4ifg = np.abs(self.loop_matrix).sum(axis=0)
        idx_pairs_no_loop = np.where(ns_loop4ifg == 0)[0]
        no_loop_pair = [self.pairs[ix] for ix in idx_pairs_no_loop]
        print(f"Number of interferograms: {len(self.pairs)}")
        print(f"Number of loops: {self.loop_matrix.shape[0]}")
        print(f"Number of dates: {len(self.dates)}")
        print(f"Number of loops per date: {len(self.pairs)/len(self.dates)}")
        print(f"Number of interferograms without loop: {len(no_loop_pair)}")
        print(f"Interferograms without loop: {no_loop_pair}")

    def dir_of_pair(self, pair):
        '''return path to pair directory for a given pair'''
        name = self.ifg_names[self.pairs.index(pair)]
        return self.ifg_dir / name

    def unw_file_of_pair(self, pair, pattern="*unw_phase.tif"):
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

    def pairs_of_loop(self, loop):
        '''return the 3 pairs of the given loop'''
        idx_pair12 = np.where(loop == 1)[0][0]
        idx_pair23 = np.where(loop == 1)[0][1]
        idx_pair13 = np.where(loop == -1)[0][0]
        pair12 = self.pairs[idx_pair12]
        pair23 = self.pairs[idx_pair23]
        pair13 = self.pairs[idx_pair13]
        return pair12, pair23, pair13

    def pairs_of_date(self, date):
        '''return all pairs of a given date'''
        pairs = [pair for pair in self.pairs if date in pair]
        return pairs


class Loops:
    '''Loops class to handle loops for a given directory or loop list.'''

    def __init__(
        self,
        loops: List[Tuple[datetime, datetime, datetime]],
    ) -> None:
        '''initialize the loops class

        Parameters:
        ----------
        loops: list
            List of loops. Each loop is a tuple of three dates.
        '''
        if not isinstance(loops, list) or not isinstance(loops[0], tuple):
            raise TypeError(
                'loops should be a list. Each loop is a tuple of three dates.')
        self.loops = loops

    @classmethod
    def from_files(
        cls,
        loop_dir: Union[str, Path],
        parse_function: Optional[Callable] = None,
        date_args: Optional[dict] = None
    ) -> 'Loops':
        '''initialize the loops class from a directory of loop files 

        Parameters:
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

        Returns:
        --------
        loops: Loops
            Loops class.
        '''
        loop_dir = Path(loop_dir)
        loop_files = list(loop_dir.iterdir())
        loops = [cls.dates_of_loop(i, parse_function, date_args)
                 for i in loop_files]
        return cls(loops)

    @classmethod
    def from_name_list(
        cls,
        loop_names: List[str],
        parse_function: Optional[Callable] = None,
        date_args: Optional[dict] = None,
    ) -> 'Loops':
        '''initialize the loops class from a list of loop file names

        Parameters:
        ----------
        loop_names: list
            List of loop file names.
        parse_function: Callable, optional
            Function to parse the date strings from the loop file name.
            If None, the loop file name will be split by '_' and
            the last 3 items will be used. Default is None.
        date_args: dict, optional
            Keyword arguments for pd.to_datetime() to convert the date strings
            to datetime objects. For example, {'format': '%Y%m%d'}.
            Default is None.

        Returns:
        --------
        loops: Loops
            Loops class.
        '''
        loops = [cls.dates_of_loop(i, parse_function, date_args)
                 for i in loop_names]
        return cls(loops)

    @staticmethod
    def dates_of_loop(
        loop: Union[str, Path],
        parse_function: Optional[Callable] = None,
        date_args: Optional[dict] = None,
    ) -> Tuple[datetime, datetime, datetime]:
        '''parse three Dates from loop file name

        Parameters:
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

        Returns:
        --------
        date1, date2, date3: datetime
            Three Dates of the loop file with format of datetime.
        '''
        loop_str = Path(loop).stem
        if parse_function is not None:
            date1, date2, date3 = parse_function(loop_str)
        else:
            items = loop_str.split('_')
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
    def seasons(self):
        '''return the season of each loop.

        Returns:
        --------
        seasons: list
            List of seasons of each loop.
                0: not the same season
                1: spring
                2: summer
                3: fall
                4: winter
        '''
        seasons = []
        for loop in self.loops:
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

        Returns:
        --------
        day_range: list
            List of day range of each loop.
        '''
        day_range = []
        for loop in self.loops:
            range1 = abs((loop[0]-loop[1]).days)
            range2 = abs((loop[1]-loop[2]).days)
            range3 = abs((loop[2]-loop[0]).days)
            day_range.append(max(range1, range2, range3))
        return day_range

    @property
    def day_range_min(self):
        '''return the day range of each loop.

        Returns:
        --------
        day_range: list
            List of day range of each loop.
        '''
        day_range = []
        for loop in self.loops:
            range1 = abs((loop[0]-loop[1]).days)
            range2 = abs((loop[1]-loop[2]).days)
            range3 = abs((loop[2]-loop[0]).days)
            day_range.append(min(range1, range2, range3))
        return day_range


class DateManager:
    def __init__(self) -> None:
        pass

    @staticmethod
    def season_of_month(month):
        '''return the season of a given month

        Parameters:
        -----------
        month: int
            Month of the year.

        Returns:
        --------
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
    >>> gdf = GeoDataFormatConverter()
    >>> gdf.load_raster(phase_file)
    >>> gdf.add_band_from_raster(amplitude_file)
    >>> gdf.to_binary(binary_file)

    ### load binary file
    >>> gdf.load_binary(binary_file)
    >>> print(gdf.arr.shape)
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

    def _load_binary(self, binary_file: Union[str, Path], order: str = 'BSQ'):
        '''Load a binary file into the data array.'''
        binary_profile_file = str(binary_file) + '.profile'
        if not Path(binary_profile_file).exists():
            raise FileNotFoundError(f"{binary_profile_file} not found")

        with open(binary_profile_file, 'r') as f:
            profile = eval(f.read())

        arr = np.fromfile(binary_file, dtype=np.float32)
        if order == 'BSQ':
            arr = arr.reshape(profile['count'],
                              profile['height'], profile['width'])
        elif order == 'BIP':
            arr = arr.reshape(profile['height'],
                              profile['width'], profile['count']).transpose(2, 0, 1)
        elif order == 'BIL':
            arr = arr.reshape(profile['height'],
                              profile['count'], profile['width']).transpose(1, 0, 2)
        else:
            raise ValueError(
                "order should be one of ['BSQ', 'BIP', 'BIL'],"
                f" but got {order}"
            )

        return arr, profile

    def load_binary(self, binary_file: Union[str, Path], order='BSQ'):
        '''Load a binary file into the data array.

        Parameters:
        ----------
        binary_file : str or Path
            The binary file to be loaded. the binary file should be with a profile file with the same name.
        order : str, one of ['BSQ', 'BIP', 'BIL']
            The order of the data array. 'BSQ' for band sequential, 'BIP' for band interleaved by pixel, 
            'BIL' for band interleaved by line. Default is 'BSQ'. 
            See: https://desktop.arcgis.com/zh-cn/arcmap/latest/manage-data/raster-and-images/bil-bip-and-bsq-raster-files.htm
        '''
        self.arr, self.profile = self._load_binary(binary_file, order)

    def load_raster(self, raster_file: Union[str, Path]):
        '''Load a raster file into the data array.

        Parameters:
        ----------
        raster_file : str or Path
            The raster file to be loaded. raster format should be supported by gdal. 
            See: https://gdal.org/drivers/raster/index.html
        '''
        self.arr, self.profile = self._load_raster(raster_file)

    def to_binary(self, out_file: Union[str, Path], order='BSQ'):
        '''Write the data array into a binary file.

        Parameters:
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

        Parameters:
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

        Parameters:
        ----------
        raster_file : str or Path
            The raster file to be added. raster format should be supported by gdal. See: https://gdal.org/drivers/raster/index.html
        '''
        arr, profile = self._load_raster(raster_file)
        self.add_band(arr)

    def add_band_from_binary(self, binary_file: Union[str, Path]):
        '''Add band to the data array from a binary file.

        Parameters:
        ----------
        binary_file : str or Path
            The binary file to be added. the binary file should be with a profile file with the same name.
        '''
        arr, profile = self._load_binary(binary_file)
        self.add_band(arr)

    def update_arr(self, arr: np.ndarray):
        '''update the data array.'''
        self.arr = arr
        if self.profile is not None:
            self.profile['count'] = arr.shape[0]
            self.profile['height'] = arr.shape[1]
            self.profile['width'] = arr.shape[2]


class PhaseDeformationConverter:
    '''A class to convert between phase and deformation(LOS) for SAR interferometry.'''

    def __init__(self, wavelength: float = None, frequency: float = None) -> None:
        '''Initialize the converter. Either wavelength or frequency should be provided. If both are provided, wavelength will be recalculated by frequency.

        Parameters:
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
