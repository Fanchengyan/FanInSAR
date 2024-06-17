#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .freeze_thaw_process import FreezeThawCycle


class TimeSeriesModels:
    """Base class for time series models"""

    _unit: Literal["year", "day"]
    _dates: pd.DatetimeIndex
    _date_spans: np.ndarray

    # Following attributes should be set in subclasses
    _G_br: np.ndarray
    _param_names: list[str]

    __slots__ = ["_unit", "_dates", "_date_spans", "_G_br", "_param_names"]

    def __init__(
        self,
        dates: pd.DatetimeIndex | Sequence[datetime],
        unit: Literal["year", "day"] = "day",
    ):
        """Initialize TimeSeriesModels

        Parameters
        ----------
        dates : pd.DatetimeIndex | Sequence[datetime]
            Dates of SAR acquisitions. This can be easily obtained by accessing
            :attr:`Pairs.dates <faninsar.Pairs.dates>`.
        unit : Literal["year", "day"], optional
            Unit of day spans in time series model, by default "day".
        """
        self._unit = None
        self._dates = None
        self._date_spans = None

        self.unit = unit
        self.dates = dates

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(dates: {len(self.dates)}, unit: {self.unit})"

    def __repr__(self) -> str:
        _str = (
            f"{self.__class__.__name__}(\n"
            f"    dates: {len(self.dates)}\n"
            f"    unit: {self.unit}\n"
            f"    param_names: {self.param_names}\n"
            f"    G_br shape: {self.G_br.shape})\n"
            ")"
        )
        return _str

    @property
    def unit(self) -> str:
        """unit of date_spans in time series model"""
        return self._unit

    @unit.setter
    def unit(self, unit) -> None:
        """Update unit"""
        if unit not in ["day", "year"]:
            raise ValueError("unit must be either day or year")
        if unit != self._unit:
            if self._unit is not None:
                self._date_spans = self._date_spans * (
                    1 / 365.25 if unit == "year" else 365.25
                )
        self._unit = unit

    @property
    def dates(self) -> pd.DatetimeIndex:
        """dates of SAR acquisitions"""
        return self._dates

    @dates.setter
    def dates(self, dates) -> None:
        """Update dates"""
        if not isinstance(dates, pd.DatetimeIndex):
            try:
                dates = pd.to_datetime(dates)
            except:
                raise TypeError(
                    "dates must be either pd.DatetimeIndex or iterable of datetime"
                )
        self._dates = dates
        date_spans = (dates - dates[0]).days.values
        if self.unit == "year":
            date_spans = date_spans / 365.25
        self._date_spans = date_spans

    @property
    def date_spans(self) -> np.ndarray:
        """date spans of SAR acquisitions in unit of year or day"""
        return self._date_spans

    @property
    def G_br(self) -> np.ndarray:
        """bottom right block of the design matrix G in NSBAS inversion"""
        return self._G_br

    @property
    def param_names(self) -> list[str]:
        """parameter names in time series model"""
        return self._param_names


class LinearModel(TimeSeriesModels):
    """Linear model"""

    def __init__(
        self,
        dates: pd.DatetimeIndex | Sequence[datetime],
        unit: Literal["year", "day"] = "day",
    ):
        """Initialize LinearModel

        Parameters
        ----------
        dates : pd.DatetimeIndex | Sequence[datetime]
            Dates of SAR acquisitions. This can be easily obtained by accessing
            :attr:`Pairs.dates <faninsar.Pairs.dates>`.
        unit : Literal["year", "day"], optional
            Unit of day spans in time series model, by default "day".
        """
        super().__init__(dates, unit=unit)

        self._G_br = np.array([self.date_spans, np.ones_like(self.date_spans)]).T
        self._param_names = ["velocity", "constant"]


class QuadraticModel(TimeSeriesModels):
    """Quadratic model"""

    def __init__(
        self,
        dates: pd.DatetimeIndex | Sequence[datetime],
        unit: Literal["year", "day"] = "day",
    ):
        """Initialize QuadraticModel

        Parameters
        ----------
        dates : pd.DatetimeIndex | Sequence[datetime]
            Dates of SAR acquisitions. This can be easily obtained by accessing
            :attr:`Pairs.dates <faninsar.Pairs.dates>`.
        unit : Literal["year", "day"], optional
            Unit of day spans in time series model, by default "day".
        """
        super().__init__(dates, unit=unit)
        self._G_br = np.array(
            [self.date_spans**2, self.date_spans, np.ones_like(self.date_spans)]
        ).T
        self._param_names = ["1/2_acceleration", "initial_velocity", "constant"]


class CubicModel(TimeSeriesModels):
    """Cubic model"""

    def __init__(
        self,
        dates: pd.DatetimeIndex | Sequence[datetime],
        unit: Literal["year", "day"] = "day",
    ):
        """Initialize CubicModel

        Parameters
        ----------
        dates : pd.DatetimeIndex | Sequence[datetime]
            Dates of SAR acquisitions. This can be easily obtained by accessing
            :attr:`Pairs.dates <faninsar.Pairs.dates>`.
        unit : Literal["year", "day"], optional
            Unit of day spans in time series model, by default "day".
        """
        super().__init__(dates, unit=unit)
        self._G_br = np.array(
            [
                self.date_spans**3,
                self.date_spans**2,
                self.date_spans,
                np.ones_like(self.date_spans),
            ]
        ).T
        self._param_names = ["Rate of Change", "acceleration", "velocity", "constant"]


class AnnualSinusoidalModel(TimeSeriesModels):
    """A sinusoidal model with annual period"""

    def __init__(
        self,
        dates: pd.DatetimeIndex | Sequence[datetime],
        unit: Literal["year", "day"] = "day",
    ):
        """Initialize AnnualSinusoidalModel

        Parameters
        ----------
        dates : pd.DatetimeIndex | Sequence[datetime]
            Dates of SAR acquisitions. This can be easily obtained by accessing
            :attr:`Pairs.dates <faninsar.Pairs.dates>`.
        unit : Literal["year", "day"], optional
            Unit of day spans in time series model, by default "day".
        """
        super().__init__(dates, unit=unit)
        if self.unit == "day":
            coeff = 2 * np.pi / 365.25
        else:
            coeff = 2 * np.pi

        self._G_br = np.array(
            [
                np.sin(self.date_spans * coeff),
                np.cos(self.date_spans * coeff),
                self.date_spans,
                np.ones_like(self.date_spans),
            ]
        ).T
        self._param_names = ["sin(T)", "cos(T)", "velocity", "constant"]


class AnnualSemiannualSinusoidal(TimeSeriesModels):
    """A compose sinusoidal model that contains annual and semi-annual periods"""

    def __init__(
        self,
        dates: pd.DatetimeIndex | Sequence[datetime],
        unit: Literal["year", "day"] = "day",
    ):
        """Initialize AnnualSemiannualSinusoidal

        Parameters
        ----------
        dates : pd.DatetimeIndex | Sequence[datetime]
            Dates of SAR acquisitions. This can be easily obtained by accessing
            :attr:`Pairs.dates <faninsar.Pairs.dates>`.
        unit : Literal["year", "day"], optional
            Unit of day spans in time series model, by default "day".
        """
        super().__init__(dates, unit=unit)

        if self.unit == "day":
            coeff = 2 * np.pi / 365.25
        else:
            coeff = 2 * np.pi

        self._G_br = np.array(
            [
                np.sin(self.date_spans * coeff),
                np.cos(self.date_spans * coeff),
                np.sin(self.date_spans * coeff * 2),
                np.cos(self.date_spans * coeff * 2),
                self.date_spans,
                np.ones_like(self.date_spans),
            ]
        ).T
        self._param_names = [
            "sin(T)",
            "cos(T)",
            "sin(T/2)",
            "cos(T/2)",
            "velocity",
            "constant",
        ]


class FreezeThawCycleModel(TimeSeriesModels):
    """A pure Freeze-thaw cycle model without velocity"""

    def __init__(
        self,
        ftc: FreezeThawCycle,
        dates: pd.DatetimeIndex | Sequence[datetime],
        unit: Literal["year", "day"] = "day",
    ):
        """Initialize FreezeThawCycleModel

        Parameters
        ----------
        ftc : FreezeThawCycle
            Freeze-thaw cycle instance. The dates in ftc should cover the dates
            of SAR acquisitions.

            .. warning::
                The first date in ftc should be earlier than the thawing onset
                of the first year in the time series model.
        dates : pd.DatetimeIndex | Sequence[datetime]
            Dates of SAR acquisitions. This can be easily obtained by accessing
            :attr:`Pairs.dates <faninsar.Pairs.dates>`.
        unit : Literal["year", "day"], optional
            Unit of day spans in time series model, by default "day".
        """
        super().__init__(dates, unit=unit)

        df_br = pd.DataFrame(
            np.full((len(self.dates), 2), np.nan, dtype=np.float32), index=self.dates
        )
        bias = np.zeros((1, 2), dtype=np.float32)

        years = self.dates.year.unique()
        for year in years:
            start = ftc.get_year_start(year)
            end = ftc.get_year_end(year)

            if not start:
                continue

            m = np.logical_and(self.dates >= start, self.dates <= end)
            img_dates = self.dates[m]

            DDT = ftc.DDT[start:end].copy()
            DDF = ftc.DDF[start:end].copy()

            if year == years[0]:
                # add missing dates before 07-01 for DDF in the first year
                if DDF.index[0] > start:
                    dt_missing = pd.date_range(start, DDF.index[0], freq="1D")[:-1]
                    DDF_missing = pd.Series(np.nan, index=dt_missing)
                    DDF = pd.concat([DDF_missing, DDF])
                # set coefficients to zero before the thawing onset for the first year
                if start > self.dates[0]:
                    df_br.loc[self.dates[0] : start, :] = 0
                # set the DDF to 0 during the thawing onset

            if pd.isna(DDT[0]):
                DDT[0] = 0
            DDF[:f"{year}-07-01"] = 0

            DDT = DDT.ffill()
            DDF = DDF.ffill()

            t3 = ftc.t3s[year]
            if not pd.isnull(t3):
                if t3 in DDT.index:
                    DDT[t3:] = DDT[t3]
                if t3 in DDF.index:
                    DDF[t3:] = DDF[t3]

            DDT_A1 = np.sqrt(DDT[img_dates].values)
            DDF_A4 = np.sqrt(DDF[img_dates].values)

            try:
                df_br[start:end] = np.array([DDT_A1, DDF_A4]).T + bias
            except:
                raise ValueError(f"{bias}\n\n{df_br[start:end]}")
            # df_br[start:end] = np.array(
            #     [DDT_A1, DDF_A4]).T + bias

            DDT_A1_end = np.sqrt(DDT[-1])
            DDF_A4_end = np.sqrt(DDF[-1])

            bias = bias + np.asarray([[DDT_A1_end, DDF_A4_end]])
        df_br.loc[:, "constant"] = 1
        self._G_br = df_br.values
        self._param_names = ["E_t", "E_f", "constant"]


class FreezeThawCycleModelWithVelocity(TimeSeriesModels):
    """A Freeze-thaw cycle model with velocity"""

    def __init__(
        self,
        ftc: FreezeThawCycle,
        dates: pd.DatetimeIndex | Sequence[datetime],
        unit: Literal["year", "day"] = "day",
    ):
        """Initialize FreezeThawCycleModelWithVelocity

        Parameters
        ----------
        ftc : FreezeThawCycle
            Freeze-thaw cycle instance. The dates in ftc should cover the dates
            of SAR acquisitions.
        dates : pd.DatetimeIndex | Sequence[datetime]
            Dates of SAR acquisitions. This can be easily obtained by accessing
            :attr:`Pairs.dates <faninsar.Pairs.dates>`.
        unit : Literal["year", "day"], optional
            Unit of day spans in time series model, by default "day".
        """

        super().__init__(dates, unit=unit)

        df_br = pd.DataFrame(
            np.full((len(self.dates), 3), np.nan, dtype=np.float32), index=self.dates
        )
        bias = 0

        years = self.dates.year.unique()
        for year in years:
            start = ftc.get_year_start(ftc.t1s, ftc.t2s, year)
            end = ftc.get_year_end(ftc.t1s, year)

            m = np.logical_and(self.dates >= start, self.dates <= end)
            img_dates = self.dates[m]

            DDT = ftc.DDT[start:end].copy()
            DDF = ftc.DDF[start:end].copy()

            if pd.isna(DDT[0]):
                DDT[0] = 0
            DDF[:f"{year}-07-01"] = 0

            DDT = DDT.fillna(method="ffill")
            DDF = DDF.fillna(method="ffill")

            t3 = ftc.t3s[year]
            if not pd.isnull(t3):
                if t3 in DDT.index:
                    DDT[t3:] = DDT[t3]
                if t3 in DDF.index:
                    DDF[t3:] = DDF[t3]

            DDT_A1 = np.sqrt(DDT[img_dates].values)
            DDF_A4 = np.sqrt(DDF[img_dates].values)

            df_br[start:end] = np.array([DDT_A1, DDF_A4, np.full_like(DDF_A4, bias)]).T

            bias = bias + 1
        df_br.loc[:, "constant"] = 1
        self._G_br = df_br.values
        self._param_names = ["E_t", "E_f", "V", "constant"]
