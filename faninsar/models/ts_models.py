#!/usr/bin/env python3
from datetime import datetime
from typing import Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd


class TimeSeriesModels:
    '''Base class for time series models'''

    _unit: Literal['year', 'day'] = 'day'
    _dates: pd.DatetimeIndex = None
    _date_diff: np.ndarray = None
    _G_br: np.ndarray = None
    _param_names: List[str] = []

    def __init__(
        self,
        dates,
        unit='day'
    ):

        dates = pd.to_datetime(dates)
        date_diff = (dates-dates[0]).days.values
        if unit == 'year':
            date_diff = date_diff / 365.25

        self.unit = unit
        self.dates = dates
        self.date_diff = date_diff

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.unit})'

    def __repr__(self) -> str:
        return str(self)

    @property
    def unit(self) -> str:
        '''Return unit of datetime in time series model'''
        return self._unit

    @unit.setter
    def unit(self, unit) -> None:
        '''Update unit'''
        if unit not in ['day', 'year']:
            raise ValueError('unit must be either day or year')
        self._unit = unit

    @property
    def dates(self) -> pd.DatetimeIndex:
        '''Return dates'''
        return self._dates

    @dates.setter
    def dates(self, dates) -> None:
        '''Update dates'''
        if not isinstance(dates, pd.DatetimeIndex):
            try:
                dates = pd.to_datetime(dates)
            except:
                raise TypeError(
                    'dates must be either pd.DatetimeIndex or iterable of datetime')
        self._dates = dates

    @property
    def date_diff(self) -> np.ndarray:
        '''Return date_diff'''
        return self._date_diff

    @property
    def G_br(self) -> np.ndarray:
        '''Return G_br'''
        return self._G_br

    @property
    def param_names(self) -> List[str]:
        '''Return param_names'''
        return self._param_names


class LinearModel(TimeSeriesModels):
    '''Linear model'''

    def __init__(self, dates, unit='day'):
        super().__init__(dates, unit=unit)

        self.G_br = np.array(
            [self.date_diff,
             np.ones_like(self.date_diff)]
        ).T
        self.param_names = ['velocity', 'constant']


class QuadraticModel(TimeSeriesModels):
    '''Quadratic model'''

    def __init__(self, dates, unit='day'):
        super().__init__(dates, unit=unit)
        self.G_br = np.array(
            [self.date_diff**2,
             self.date_diff,
             np.ones_like(self.date_diff)]
        ).T
        self.param_names = ['1/2_acceleration', 'initial_velocity', 'constant']


class CubicModel(TimeSeriesModels):
    '''Cubic model'''

    def __init__(self, dates, unit='day'):
        super().__init__(dates, unit=unit)
        self.G_br = np.array(
            [self.date_diff**3,
             self.date_diff**2,
             self.date_diff,
             np.ones_like(self.date_diff)]
        ).T
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

        self.G_br = np.array(
            [np.sin(self.date_diff*coeff),
             np.cos(self.date_diff*coeff),
             self.date_diff,
             np.ones_like(self.date_diff)]
        ).T
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

        G_br = np.array(
            [np.sin(self.date_diff*coeff),
             np.cos(self.date_diff*coeff),
             np.sin(self.date_diff*coeff*2),
             np.cos(self.date_diff*coeff*2),
             self.date_diff,
             np.ones_like(self.date_diff)]
        ).T
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
