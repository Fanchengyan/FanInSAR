"""Time series models for NSBAS inversion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from datetime import datetime

    from .freeze_thaw_process import FreezeThawCycle


class TimeSeriesModels:
    """Base class for time series models."""

    _unit: Literal["year", "day"]
    _dates: pd.DatetimeIndex
    _date_spans: np.ndarray

    # Following attributes should be set in subclasses
    _G_br: np.ndarray
    _param_names: list[str]

    __slots__ = ["_G_br", "_date_spans", "_dates", "_param_names", "_unit"]

    def __init__(
        self,
        dates: pd.DatetimeIndex | Sequence[datetime],
        unit: Literal["year", "day"] = "day",
    ) -> None:
        """Initialize TimeSeriesModels.

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
        """Return a string representation of the object."""
        return f"{self.__class__.__name__}(dates: {len(self.dates)}, unit: {self.unit})"

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return (
            f"{self.__class__.__name__}(\n"
            f"    dates: {len(self.dates)}\n"
            f"    unit: {self.unit}\n"
            f"    param_names: {self.param_names}\n"
            f"    G_br shape: {self.G_br.shape})\n"
            ")"
        )

    @property
    def unit(self) -> str:
        """Unit of date_spans in time series model."""
        return self._unit

    @unit.setter
    def unit(
        self,
        unit: Literal["year", "day"],
    ) -> None:
        """Update unit."""
        if unit not in ["day", "year"]:
            msg = "unit must be either day or year"
            raise ValueError(msg)
        if unit != self._unit and self._unit is not None:
            self._date_spans = self._date_spans * (
                1 / 365.25 if unit == "year" else 365.25
            )
        self._unit = unit

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Dates of SAR acquisitions."""
        return self._dates

    @dates.setter
    def dates(self, dates: pd.DatetimeIndex) -> None:
        """Update dates."""
        if not isinstance(dates, pd.DatetimeIndex):
            try:
                dates = pd.to_datetime(dates)
            except Exception as e:
                msg = "dates must be either pd.DatetimeIndex or iterable of datetime"
                raise TypeError(msg) from e
        self._dates = dates
        date_spans = (dates - dates[0]).days.values
        if self.unit == "year":
            date_spans = date_spans / 365.25
        self._date_spans = date_spans

    @property
    def date_spans(self) -> np.ndarray:
        """Date spans of SAR acquisitions in unit of year or day."""
        return self._date_spans

    @property
    def G_br(self) -> np.ndarray:  # noqa: N802
        """Bottom right block of the design matrix G in NSBAS inversion."""
        return self._G_br

    @property
    def param_names(self) -> list[str]:
        """Parameter names in time series model."""
        return self._param_names


class LinearModel(TimeSeriesModels):
    """Linear model."""

    def __init__(
        self,
        dates: pd.DatetimeIndex | Sequence[datetime],
        unit: Literal["year", "day"] = "day",
    ) -> None:
        """Initialize LinearModel.

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
    """Quadratic model."""

    def __init__(
        self,
        dates: pd.DatetimeIndex | Sequence[datetime],
        unit: Literal["year", "day"] = "day",
    ) -> None:
        """Initialize QuadraticModel.

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
            [self.date_spans**2, self.date_spans, np.ones_like(self.date_spans)],
        ).T
        self._param_names = ["1/2_acceleration", "initial_velocity", "constant"]


class CubicModel(TimeSeriesModels):
    """Cubic model."""

    def __init__(
        self,
        dates: pd.DatetimeIndex | Sequence[datetime],
        unit: Literal["year", "day"] = "day",
    ) -> None:
        """Initialize CubicModel.

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
            ],
        ).T
        self._param_names = ["Rate of Change", "acceleration", "velocity", "constant"]


class AnnualSinusoidalModel(TimeSeriesModels):
    """A sinusoidal model with annual period."""

    def __init__(
        self,
        dates: pd.DatetimeIndex | Sequence[datetime],
        unit: Literal["year", "day"] = "day",
    ) -> None:
        """Initialize AnnualSinusoidalModel.

        Parameters
        ----------
        dates : pd.DatetimeIndex | Sequence[datetime]
            Dates of SAR acquisitions. This can be easily obtained by accessing
            :attr:`Pairs.dates <faninsar.Pairs.dates>`.
        unit : Literal["year", "day"], optional
            Unit of day spans in time series model, by default "day".

        """
        super().__init__(dates, unit=unit)
        coeff = 2 * np.pi / 365.25 if self.unit == "day" else 2 * np.pi

        self._G_br = np.array(
            [
                np.sin(self.date_spans * coeff),
                np.cos(self.date_spans * coeff),
                self.date_spans,
                np.ones_like(self.date_spans),
            ],
        ).T
        self._param_names = ["sin(T)", "cos(T)", "velocity", "constant"]


class AnnualSemiannualSinusoidal(TimeSeriesModels):
    """A compose sinusoidal model that contains annual and semi-annual periods."""

    def __init__(
        self,
        dates: pd.DatetimeIndex | Sequence[datetime],
        unit: Literal["year", "day"] = "day",
    ) -> None:
        """Initialize AnnualSemiannualSinusoidal.

        Parameters
        ----------
        dates : pd.DatetimeIndex | Sequence[datetime]
            Dates of SAR acquisitions. This can be easily obtained by accessing
            :attr:`Pairs.dates <faninsar.Pairs.dates>`.
        unit : Literal["year", "day"], optional
            Unit of day spans in time series model, by default "day".

        """
        super().__init__(dates, unit=unit)

        coeff = 2 * np.pi / 365.25 if self.unit == "day" else 2 * np.pi

        self._G_br = np.array(
            [
                np.sin(self.date_spans * coeff),
                np.cos(self.date_spans * coeff),
                np.sin(self.date_spans * coeff * 2),
                np.cos(self.date_spans * coeff * 2),
                self.date_spans,
                np.ones_like(self.date_spans),
            ],
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
    """A pure Freeze-thaw cycle model without velocity."""

    def __init__(
        self,
        ftc: FreezeThawCycle,
        dates: pd.DatetimeIndex | Sequence[datetime],
        unit: Literal["year", "day"] = "day",
    ) -> None:
        """Initialize FreezeThawCycleModel.

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
            np.full((len(self.dates), 2), np.nan, dtype=np.float32),
            index=self.dates,
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

            DDT = ftc.DDT[start:end].copy()  # noqa: N806
            DDF = ftc.DDF[start:end].copy()  # noqa: N806

            if year == years[0]:
                # add missing dates before 07-01 for DDF in the first year
                if DDF.index[0] > start:
                    dt_missing = pd.date_range(start, DDF.index[0], freq="1D")[:-1]
                    DDF_missing = pd.Series(np.nan, index=dt_missing)  # noqa: N806
                    DDF = pd.concat([DDF_missing, DDF])  # noqa: N806
                # set coefficients to zero before the thawing onset for the first year
                if start > self.dates[0]:
                    df_br.loc[self.dates[0] : start, :] = 0
                # set the DDF to 0 during the thawing onset

            if pd.isna(DDT[0]):
                DDT[0] = 0
            DDF[: f"{year}-07-01"] = 0

            DDT = DDT.ffill()  # noqa: N806
            DDF = DDF.ffill()  # noqa: N806

            t3 = ftc.t3s[year]
            if not pd.isna(t3):
                if t3 in DDT.index:
                    DDT[t3:] = DDT[t3]
                if t3 in DDF.index:
                    DDF[t3:] = DDF[t3]

            DDT_A1 = np.sqrt(DDT[img_dates].values)  # noqa: N806
            DDF_A4 = np.sqrt(DDF[img_dates].values)  # noqa: N806

            try:
                df_br[start:end] = np.array([DDT_A1, DDF_A4]).T + bias
            except Exception as e:
                msg = f"{bias}\n\n{df_br[start:end]}"
                raise ValueError(msg) from e
            # df_br[start:end] = np.array(
            #     [DDT_A1, DDF_A4]).T + bias

            DDT_A1_end = np.sqrt(DDT[-1])  # noqa: N806
            DDF_A4_end = np.sqrt(DDF[-1])  # noqa: N806

            bias = bias + np.asarray([[DDT_A1_end, DDF_A4_end]])
        df_br.loc[:, "constant"] = 1
        self._G_br = df_br.values
        self._param_names = ["E_t", "E_f", "constant"]


class FreezeThawCycleModelWithVelocity(TimeSeriesModels):
    """A Freeze-thaw cycle model with velocity."""

    def __init__(
        self,
        ftc: FreezeThawCycle,
        dates: pd.DatetimeIndex | Sequence[datetime],
        unit: Literal["year", "day"] = "day",
    ) -> None:
        """Initialize FreezeThawCycleModelWithVelocity.

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
            np.full((len(self.dates), 3), np.nan, dtype=np.float32),
            index=self.dates,
        )
        bias = 0

        years = self.dates.year.unique()
        for year in years:
            start = ftc.get_year_start(year)
            end = ftc.get_year_end(year)

            if not start:
                continue

            m = np.logical_and(self.dates >= start, self.dates <= end)
            img_dates = self.dates[m]

            DDT = ftc.DDT[start:end].copy()  # noqa: N806
            DDF = ftc.DDF[start:end].copy()  # noqa: N806

            if year == years[0]:
                # add missing dates before 07-01 for DDF in the first year
                if DDF.index[0] > start:
                    dt_missing = pd.date_range(start, DDF.index[0], freq="1D")[:-1]
                    DDF_missing = pd.Series(np.nan, index=dt_missing)  # noqa: N806
                    DDF = pd.concat([DDF_missing, DDF])  # noqa: N806
                # set coefficients to zero before the thawing onset for the first year
                if start > self.dates[0]:
                    df_br.loc[self.dates[0] : start, :] = 0
                # set the DDF to 0 during the thawing onset

            if pd.isna(DDT[0]):
                DDT[0] = 0
            DDF[: f"{year}-07-01"] = 0

            DDT = DDT.ffill()  # noqa: N806
            DDF = DDF.ffill()  # noqa: N806

            t3 = ftc.t3s[year]
            if not pd.isna(t3):
                if t3 in DDT.index:
                    DDT[t3:] = DDT[t3]
                if t3 in DDF.index:
                    DDF[t3:] = DDF[t3]

            DDT_A1 = np.sqrt(DDT[img_dates].values)  # noqa: N806
            DDF_A4 = np.sqrt(DDF[img_dates].values)  # noqa: N806

            df_br[start:end] = np.array([DDT_A1, DDF_A4, np.full_like(DDF_A4, bias)]).T

            bias = bias + 1
        df_br.loc[:, "constant"] = 1
        self._G_br = df_br.values
        self._param_names = ["E_t", "E_f", "V", "constant"]
