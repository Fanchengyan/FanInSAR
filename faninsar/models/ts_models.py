#!/usr/bin/env python3
from datetime import datetime
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

from faninsar import Pair, Pairs


# TODO: divide into multiple ts classes and base ts_model
class TimeSeriesModels:
    '''Base class for time series models'''

    def __init__(self, dates, unit='day'):
        if unit not in ['day', 'year']:
            raise ValueError('unit must be either day or year')

        dates = pd.to_datetime(dates)
        date_diff = (dates-dates[0]).days.values
        if unit == 'year':
            date_diff = date_diff / 365.25

        self.unit = unit
        self.dates = dates
        self.date_diff = date_diff


class LinearModel(TimeSeriesModels):
    '''Linear model'''

    def __init__(self, dates, unit='day'):
        super().__init__(dates, unit=unit)

        self.G_br = np.array([self.date_diff,
                              np.ones_like(self.date_diff)]).T
        self.param_names = ['velocity', 'constant']


class QuadraticModel(TimeSeriesModels):
    '''Quadratic model'''

    def __init__(self, dates, unit='day'):
        super().__init__(dates, unit=unit)
        self.G_br = np.array([self.date_diff**2,
                              self.date_diff,
                              np.ones_like(self.date_diff)]).T
        self.param_names = ['1/2_acceleration', 'initial_velocity', 'constant']


class CubicModel(TimeSeriesModels):
    '''Cubic model'''

    def __init__(self, dates, unit='day'):
        super().__init__(dates, unit=unit)
        self.G_br = np.array([self.date_diff**3,
                              self.date_diff**2,
                              self.date_diff,
                              np.ones_like(self.date_diff)]).T
        self.param_names = ['Rate of Change',
                            'acceleration', 'velocity', 'constant']


class AnnualSinusoidalModel(TimeSeriesModels):
    '''A sinusoidal model with annual period'''

    def __init__(self, dates, unit='day'):
        super().__init__(dates, unit=unit)
        if self.unit == 'day':
            coeff = 2*np.pi/365.25
        else:
            coeff = 2*np.pi

        self.G_br = np.array([np.sin(self.date_diff*coeff),
                              np.cos(self.date_diff*coeff),
                              self.date_diff,
                              np.ones_like(self.date_diff)]).T
        self.param_names = ['sin(T)', 'cos(T)',
                            'velocity', 'constant']


class AnnualSemiannualSinusoidal(TimeSeriesModels):
    '''A compose sinusoidal model that contains annual and semi-annual periods'''

    def __init__(self, dates, unit='day'):
        super().__init__(dates, unit=unit)

        if self.unit == 'day':
            coeff = 2*np.pi/365.25
        else:
            coeff = 2*np.pi

        G_br = np.array([np.sin(self.date_diff*coeff),
                        np.cos(self.date_diff*coeff),
                        np.sin(self.date_diff*coeff*2),
                        np.cos(self.date_diff*coeff*2),
                        self.date_diff,
                        np.ones_like(self.date_diff)]).T
        param_names = ['sin(T)', 'cos(T)', 'sin(T/2)',
                       'cos(T/2)', 'velocity', 'constant']
        return G_br, param_names


class FreezeThawCycleModel(TimeSeriesModels):
    '''A pure Freeze-thaw cycle model without velocity'''

    def __init__(
            self,
            t1s,
            t2s,
            t3s,
            years,
            ftc,
            dates,
            unit='day'

    ):
        super().__init__(dates, unit=unit)

        self.G_br = np.array([np.sqrt(self.date_diff),
                              np.sqrt(self.date_diff),
                              np.ones_like(self.date_diff)]).T
        self.param_names = ['E_t', 'E_f', 'constant']

        df_br = pd.DataFrame(np.full(
            (len(self.dates), 2),
            np.nan,
            dtype=np.float32
        ),
            index=self.dates)
        bias = np.zeros((1, 2), dtype=np.float32)

        for year in years:
            start = ftc.get_year_start(t1s, t2s, year)
            end = ftc.get_year_end(t1s, year)

            if not start:
                continue

            m = np.logical_and(self.dates >= start, self.dates <= end)
            imdates = self.dates[m]

            DDT = ftc.DDT[start:end]
            DDF = ftc.DDF[start:end]

            if pd.isna(DDT[0]):
                DDT[0] = 0
            DDF[:f'{year}-07-01'] = 0

            DDT = DDT.fillna(method='ffill')
            DDF = DDF.fillna(method='ffill')

            t3 = t3s[year]
            if not pd.isnull(t3):
                if t3 in DDT.index:
                    DDT[t3:] = DDT[t3]
                if t3 in DDF.index:
                    DDF[t3:] = DDF[t3]

            DDT_A1 = np.sqrt(DDT[imdates].values)
            DDF_A4 = np.sqrt(DDF[imdates].values)

            try:
                df_br[start:end] = np.array(
                    [DDT_A1, DDF_A4]).T + bias
            except:
                raise ValueError(f'{bias}\n\n{df_br[start:end]}')
            # df_br[start:end] = np.array(
            #     [DDT_A1, DDF_A4]).T + bias

            DDT_A1_end = np.sqrt(DDT[-1])
            DDF_A4_end = np.sqrt(DDF[-1])

            bias = bias + np.asarray([[DDT_A1_end, DDF_A4_end]])

        self.G_br = df_br.values
        self.param_names = ['E_t', 'E_f']


class FreezeThawCycleModelWithVelocity(TimeSeriesModels):
    '''A Freeze-thaw cycle model with velocity'''

    def __init__(
            self,
            t1s,
            t2s,
            t3s,
            years,
            ftc,
            dates,
            unit='day'
    ):
        super().__init__(dates, unit=unit)

        df_br = pd.DataFrame(
            np.full((len(self.dates), 3), np.nan, dtype=np.float32),
            index=self.dates)
        bias = 0
        for year in years:
            start = ftc.get_year_start(t1s, t2s, year)
            end = ftc.get_year_end(t1s, year)

            m = np.logical_and(self.dates >= start, self.dates <= end)
            imdates = self.dates[m]

            DDT = ftc.DDT[start:end]
            DDF = ftc.DDF[start:end]

            if pd.isna(DDT[0]):
                DDT[0] = 0
            DDF[:f'{year}-07-01'] = 0

            DDT = DDT.fillna(method='ffill')
            DDF = DDF.fillna(method='ffill')

            t3 = t3s[year]
            if not pd.isnull(t3):
                if t3 in DDT.index:
                    DDT[t3:] = DDT[t3]
                if t3 in DDF.index:
                    DDF[t3:] = DDF[t3]

            DDT_A1 = np.sqrt(DDT[imdates].values)
            DDF_A4 = np.sqrt(DDF[imdates].values)

            df_br[start:end] = np.array(
                [DDT_A1, DDF_A4, np.full_like(DDF_A4, bias)]).T

            bias = bias + 1

        self.G_br = df_br.values
        self.param_names = ['E_t', 'E_f', 'V']


class NSBASMatrixFactory:
    '''Factory class for NSBAS matrix. The NSBAS matrix is usually expressed as:
    ``d = Gm``, where ``d`` is the unwrapped interferograms matrix, ``G`` is the
    NSBAS matrix, and ``m`` is the model parameters, which is the combination of
    the deformation increment and the model parameters. see paper: TODO for more
    details.
    '''
    _pairs: Pairs = []
    _model: TimeSeriesModels = None
    _gamma: float = 0.0001
    _G: np.ndarray = None
    _d: np.ndarray = None

    slots = ['_pairs', '_model', '_gamma', '_G', '_d']

    def __init__(
        self,
        unw: np.ndarray,
        pairs: Union[Pairs, Iterable[str]],
        model: TimeSeriesModels,
        gamma: float = 0.0001
    ):
        '''Initialize NSBASMatrixFactory

        Parameters
        ----------
        unw : np.ndarray (n_pairs, n_pixels)
            Unwrapped interferograms matrix
        pairs : Union[Pairs, Iterable[str]]
            Pairs or iterable of pair names
        model : TimeSeriesModels
            Time series model
        gamma : float, optional
            weight for the model component, by default 0.0001
        '''
        self.pairs = pairs
        self.d = unw
        self.model = model
        self.gamma = gamma

    @property
    def pairs(self):
        '''Return pairs'''
        return self._pairs

    @pairs.setter
    def pairs(self, pairs):
        '''Update pairs by input pairs'''
        if isinstance(pairs, Pairs):
            self._pairs = pairs
        elif isinstance(pairs, Iterable):
            self._pairs = Pairs.from_names(pairs)
        else:
            raise TypeError('pairs must be either Pairs or Iterable')

        if self.d is not None:
            if self.d.shape[0] != len(self.pairs) + len(self.pairs.dates):
                raise ValueError(
                    'Incorrect pairs shape, pairs number plus dates number should be equal to rows number of d')

        self._pairs = pairs

    @property
    def model(self):
        '''Return model'''
        return self._model

    @model.setter
    def model(self, model):
        '''Update model and G by input model'''
        if not isinstance(model, TimeSeriesModels):
            raise TypeError('model must be a TimeSeriesModels instance')
        if self._model == model:
            return

        self._model = model
        if hasattr(self, 'gamma'):
            self.G = self._make_nsbas_matrix(model.G_br, self.gamma)

    @property
    def gamma(self):
        '''Return gamma'''
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        '''Update gamma and G by input gamma'''
        if not isinstance(gamma, (float, int)):
            raise TypeError('gamma must be either float or int')
        if gamma == self._gamma:
            return
        if gamma <= 0:
            raise ValueError('gamma must be positive')

        self._gamma = gamma
        if hasattr(self, 'model'):
            self.G = self._make_nsbas_matrix(self.model.G_br, gamma)

    @property
    def d(self):
        '''Return d for NSBAS: d = Gm'''
        return self._d

    @d.setter
    def d(self, unw):
        '''Update d: restructure unw by appending model matrix part'''
        if not isinstance(unw, np.ndarray):
            raise TypeError('d must be a numpy array')
        if len(unw.shape) != 2:
            raise ValueError('d must be a 2D array')
        if unw.shape[0] != len(self.pairs):
            raise ValueError(
                'input unw must have the same rows number as pairs number')

        self._d = self._restructure_unw(unw)

    @property
    def G(self):
        '''Return G for NSBAS: d = Gm'''
        return self._G

    @G.setter
    def G(self, G):
        '''Update G by input G'''
        if not isinstance(G, np.ndarray):
            raise TypeError('G must be a numpy array')
        if G.shape[0] != (len(self.pairs) + len(self.pairs.dates)):
            raise ValueError(
                'G must have the same number of rows as (pairs number + dates number)')

        self._G = G

    def _make_nsbas_matrix(self, G_br, gamma):
        G_br = np.asarray(G_br)
        G_tl = self.pairs.to_matrix()

        if len(G_br.shape) == 1:
            G_br = G_br.reshape(-1, 1)
        n_param = G_br.shape[1]

        n_date = len(self.pairs.dates)
        G_bl = np.tril(np.ones((n_date, n_date-1),
                               dtype=np.float32), k=-1)
        G_b = np.hstack((G_bl, G_br))*gamma
        G_t = np.hstack((G_tl, np.zeros((len(self._pairs), n_param))))
        G = np.vstack((G_t, G_b))

        return G

    def _restructure_unw(self, unw):
        unw = np.vstack((unw, np.zeros((len(self.pairs.dates), unw.shape[1]))))
        return unw
