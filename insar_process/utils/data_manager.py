from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional, Callable
from datetime import datetime


class SBAS_Network:
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
    '''Loops class to handle loop files for a given directory.'''

    def __init__(
        self,
        home_dir: Union[str, Path],
        file_type: str = 'tif',
    ) -> None:
        '''initialize the loops class

        Parameters:
        ----------
        home_dir: Path
            Path to the home directory containing all the loops.
        file_type: str, one of ['tif', 'nc']
            File type of the loops. Default is 'tif'.
        '''
        self.home_dir = Path(home_dir)
        self.loop_files = sorted(self.home_dir.glob(f'*.{file_type}'))

    @staticmethod
    def get_loop_file_dates(
        loop_file: Union[str, Path],
        parse_function: Optional[Callable] = None,
        date_args: Optional[dict] = None,
    ) -> Tuple[datetime, datetime, datetime]:
        '''parse three Dates from loop file name

        Parameters:
        ----------
        loop_file: str or Path
            Path to the loop file.
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
        loop_str = Path(loop_file).stem
        if parse_function is not None:
            date1, date2, date3 = parse_function(loop_str)
        else:
            items = loop_str.split('_')
            if len(items) >= 3:
                date1, date2, date3 = items[-3:]
                if not (len(date1) == len(date2) == len(date3)):
                    raise ValueError(
                        f'Loop file name {loop_file} not recognized.')
            else:
                raise ValueError(f'Loop file name {loop_file} not recognized.')

        if date_args is None:
            date_args = {}
        date_args.update({"errors": 'raise'})

        date1 = pd.to_datetime(date1, **date_args)
        date2 = pd.to_datetime(date2, **date_args)
        date3 = pd.to_datetime(date3, **date_args)
        return date1, date2, date3
