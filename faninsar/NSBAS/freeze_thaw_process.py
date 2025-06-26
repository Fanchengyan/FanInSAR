"""A module to calculate the freeze-thaw cycle from temperature data."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


class FreezeThawCycle:
    """A class to calculate the freeze-thaw cycle from temperature data."""

    def __init__(
        self,
        dates: Sequence,
        temperature: Sequence,
        date_args: dict | None = None,
        day_duration: int = 5,
        ER: float = 1,
        no_gap: bool = True,
        thaw_start: str = "01-01",
        thaw_end: str = "12-31",
        freeze_start: str = "07-01",
        freeze_end: str = "06-30",
    ) -> None:
        """Initialize the FreezeThawCycle class.

        Parameters
        ----------
        dates : Sequence
            dates corresponding to temperature. date can be any format that
            pd.to_datetime acceptable
        temperature: Sequence
            air/surface temperature. Must be the same length as dates
        date_args: dict, optional
            Keyword arguments for :func:`pandas.to_datetime` to convert the date
            strings to datetime objects. For example, {'format': '%Y%m%d'}.
            Default is {}.
        day_duration : int, optional
            the duration in days used to calculate thawing or freezing onsets.
            The first day for continuous n-day period where the air temperature
            exceeds/drops 0°C is considered as the onset of thawing/freezing.
        ER : float, optional
            E factor ratio, expressed numerically as A1/A4, used to calculate
            the onset of winter-stable period. Default is 1.
        no_gap : bool, optional
            if True, the temperature data should not have any gap in time series.
            Default is True.
        thaw_start, thaw_end : str, optional
            the start and end date that cover whole thawing period with
            format of "month-day". Default is '01-01' and '12-31'.
        freeze_start, freeze_end : str, optional
            the start and end date that cover whole freezing period with
            format of "month-day". Default is '07-01' and '06-30'.

        """
        if date_args is None:
            date_args = {}
        self._day_duration = day_duration
        self._ER = ER
        self._dates = pd.DatetimeIndex(pd.to_datetime(dates, **date_args).date)
        self._data = pd.Series(temperature, index=self.dates, name="temperature")

        if no_gap:
            self._ensure_dates_nogap(self.data)
        self.years = sorted(set(self.data.index.year))

        self.thaw_start = thaw_start
        self.thaw_end = thaw_end
        self.freeze_start = freeze_start
        self.freeze_end = freeze_end

        self._calculate_DDT()
        self._calculate_DDF()
        self.update_FTI()
        self.update_ts()

    @property
    def day_duration(self) -> int:
        """The duration in days used to calculate thawing or freezing onsets."""
        return self._day_duration

    @day_duration.setter
    def day_duration(self, value: int) -> None:
        self._day_duration = value
        self.update_ts(day_duration=value)

    @property
    def ER(self) -> float:  # noqa: N802
        """E factor ratio, used to calculate the onset of winter-stable period."""
        return self._ER

    @ER.setter
    def ER(self, value: float) -> None:  # noqa: N802
        self._ER = value
        self.update_ts(ER=value)

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Dates of temperature in pandas DatetimeIndex format."""
        return self._dates

    @property
    def data(self) -> pd.Series:
        """Temperature data in pandas Series format."""
        return self._data

    @property
    def DDT(self) -> pd.Series:  # noqa: N802
        """Cumulative Degree Days of Thawing period for every day."""
        return self._DDT

    @property
    def DDF(self) -> pd.Series:  # noqa: N802
        """Cumulative Degree Days of Freezing period for every day."""
        return self._DDF

    @property
    def TI(self) -> pd.Series:  # noqa: N802
        """Thawing Index for every year."""
        return self._TI

    @property
    def FI(self) -> pd.Series:  # noqa: N802
        """Freezing Index for every year."""
        return self._FI

    @property
    def t1s(self) -> pd.Series:
        """Onset of thawing for every year."""
        return self._t1s

    @property
    def t2s(self) -> pd.Series:
        """Onset of freezing for every year."""
        return self._t2s

    @property
    def t3s(self) -> pd.Series:
        """Onset of winter-stable period for every year."""
        return self._t3s

    def _ensure_dates_nogap(self, df: pd.DatetimeIndex) -> None:
        df = df.resample("D").mean()
        dates_nan = df[pd.isna(df)].index.strftime("%F").to_list()
        if len(dates_nan) > 0:
            msg = f"Temperature data have a null value in dates: {', '.join(dates_nan)}"
            raise ValueError(msg)

    def _dates_slice_is_complete(self, dates: list, start: str, end: str) -> bool:
        if len(dates) > 0:
            return pd.to_datetime(start) == pd.to_datetime(dates[0]) and pd.to_datetime(
                end,
            ) == pd.to_datetime(dates[-1])
        return False

    def _same_year(self, start: str, end: str) -> bool:
        """Check if the start and end date are in the same year."""
        dt = pd.to_datetime([f"2000-{start}", f"2000-{end}"])
        return dt[0] < dt[1]

    def _calculate_DDT(self) -> None:  # noqa: N802
        """Calculate the DDT of every day."""
        list_ddts = []
        list_thaw_complete = []
        offset_year = 0 if self._same_year(self.thaw_start, self.thaw_end) else 1
        for year in self.years:
            date_start = f"{year}-{self.thaw_start}"
            date_end = f"{year + offset_year}-{self.thaw_end}"

            df_thawing = self.data[date_start:date_end].copy()
            df_thawing[df_thawing < 0] = np.nan

            df_ddt_year = np.cumsum(df_thawing)
            # df_ddt_year = df_ddt_year.fillna(method='ffill')

            if self._dates_slice_is_complete(df_thawing.index, date_start, date_end):
                list_thaw_complete.append(True)
            else:  # not complete,set nan to avoid TI is less than true value
                list_thaw_complete.append(False)

            df_ddt_year = pd.DataFrame(df_ddt_year)
            df_ddt_year["year"] = year
            list_ddts.append(df_ddt_year)

        self.df_DDT = pd.concat(list_ddts)
        self._DDT = self.df_DDT["temperature"]
        self._thaw_complete = pd.Series(list_thaw_complete, index=self.years)

    def _calculate_DDF(self) -> None:  # noqa: N802
        """Calculate the DDF of every day."""
        list_ddfs = []
        list_freeze_complete = []
        offset_year = 0 if self._same_year(self.freeze_start, self.freeze_end) else 1
        for year in self.years:
            date_start = f"{year}-{self.freeze_start}"
            date_end = f"{year + offset_year}-{self.freeze_end}"

            df_freezing = -self.data[date_start:date_end].copy()
            df_freezing[df_freezing < 0] = np.nan

            df_ddf_year = np.cumsum(df_freezing)
            # df_ddf_year = df_ddf_year.fillna(method='ffill')

            if self._dates_slice_is_complete(df_freezing.index, date_start, date_end):
                list_freeze_complete.append(True)
            else:  # not complete,set nan to avoid FI is less than true value
                list_freeze_complete.append(False)

            df_ddf_year = pd.DataFrame(df_ddf_year)
            df_ddf_year["year"] = year
            list_ddfs.append(df_ddf_year)

        self.df_DDF = pd.concat(list_ddfs)
        self._DDF = self.df_DDF["temperature"]
        self._freeze_complete = pd.Series(list_freeze_complete, index=self.years)

    def update_FTI(self, strict: bool = False) -> None:  # noqa: N802
        """Update the freezing index(FI) or thawing index(TI) for every year.

        Parameters
        ----------
        strict : bool
            if True, the thaw or freeze period is not complete would get nan.
            if False, the maximum DDT or DDF would be used.

        """
        self._FI = self.DDF.groupby(self.df_DDF.year).max()
        self._TI = self.DDT.groupby(self.df_DDT.year).max()

        if strict:
            self._FI[~self._freeze_complete] = np.nan
            self._TI[~self._thaw_complete] = np.nan

    def get_t1s(
        self,
        day_duration: int = 5,
        years: list[str] | None = None,
    ) -> pd.Series:
        """Get the t1 (onset of thawing) for given years."""
        t1s = []
        for year in self.years:
            thaw_days = self.data[self.df_DDT.index][self.df_DDT["year"] == year] > 0

            if np.all(~thaw_days):
                t1s.append(np.nan)
            else:
                arr_temp = np.zeros(len(thaw_days) + day_duration - 1, np.int16)
                for i in range(day_duration):
                    end = -(day_duration - i - 1) if day_duration - 1 > i else None
                    arr_temp[i:end] += thaw_days
                arr_temp = arr_temp[(day_duration - 1) :]
                df_temp = pd.Series(arr_temp, index=thaw_days.index)
                t1_candidate = df_temp[df_temp == day_duration]
                t1 = t1_candidate.index[0] if len(t1_candidate) > 0 else np.nan
                t1s.append(t1)

        t1s = pd.Series(t1s, index=self.years, dtype="datetime64[ns]")
        if years:
            t1s = t1s[years]
        return t1s

    def get_t2s(
        self,
        day_duration: int = 5,
        years: list[str] | None = None,
    ) -> pd.Series:
        """Get the t2 (onset of freezing) for given years."""
        t2s = []
        for year in self.years:
            freeze_days = (
                self.data[self.df_DDF.index][
                    (self.df_DDF["year"] == year).values.flatten()
                ]
                < 0
            )

            if np.all(~freeze_days):
                t2s.append(np.nan)
            else:
                arr_temp = np.zeros(len(freeze_days) + day_duration - 1, np.int16)
                for i in range(day_duration):
                    end = -(day_duration - i - 1) if day_duration - 1 > i else None
                    arr_temp[i:end] += freeze_days
                arr_temp = arr_temp[day_duration - 1 :]
                df_temp = pd.Series(arr_temp, index=freeze_days.index)
                t2_candidate = df_temp[df_temp == day_duration]
                t2 = t2_candidate.index[0] if len(t2_candidate) > 0 else np.nan
                t2s.append(t2)

        t2s = pd.Series(t2s, index=self.years, dtype="datetime64[ns]")
        if years:
            t2s = t2s[years]
        return t2s

    def get_t3s(
        self,
        ER: float | None = None,
        years: list[str] | None = None,
    ) -> pd.Series:
        """Get date t3 (onset of winter-stable period) for given years.

        Parameters
        ----------
        ER : float | None
            E factor ratio, expressed numerically as A1/A4. Default is None,
            which means using the ER of the class instance.
        years : list[str] | None
            years of t3. Default is None, which means all years will be returned.

        """
        if ER is None:
            ER = self.ER  # noqa: N806
        t3s = []
        for year in self.years:
            FI_year = self.FI[year] if year in self.FI.index else np.nan  # noqa: N806
            TI_year = self.TI[year] if year in self.TI.index else np.nan  # noqa: N806

            date_start = f"{year}-{self.freeze_start}"
            date_end = f"{year + 1}-{self.freeze_end}"
            DDF_year = self.DDF[date_start:date_end]  # noqa: N806

            # FI_year may be nan in last year
            if pd.isna(FI_year):
                FI_year = self.FI.mean()  # noqa: N806
                if pd.isna(FI_year):  # still nan for all year mean
                    FI_year = DDF_year.max()  # noqa: N806

            if ER * TI_year > FI_year or pd.isna(TI_year) or pd.isna(FI_year):
                t3s.append(np.nan)
            else:
                condition = (np.sqrt(TI_year)) <= (ER * np.sqrt(DDF_year))
                DDF_candidate = DDF_year[condition]  # noqa: N806
                if len(DDF_candidate) > 0:
                    t3 = DDF_candidate[DDF_candidate == DDF_candidate.min()].index[0]
                else:
                    t3 = np.nan
                t3s.append(t3)
        t3s = pd.Series(t3s, index=self.years, dtype="datetime64[ns]")
        if years:
            t3s = t3s[years]
        return t3s

    def update_ts(
        self,
        day_duration: int = 5,
        ER: float | None = None,
        years: list[str] | None = None,
    ) -> None:
        """Update the onset of thawing, freezing, and winter-stable period."""
        self._t1s = self.get_t1s(day_duration=day_duration, years=years)
        self._t2s = self.get_t2s(day_duration=day_duration, years=years)
        self._t3s = self.get_t3s(ER, years=years)

    def get_year_start(self, year: int) -> pd.Timestamp | None:
        """Get the thawing onset for given year or None if not available."""
        t1 = self.t1s[year]
        if pd.isna(t1):
            return None
        return t1

    def get_year_end(self, year: int) -> pd.Timestamp:
        """Get thawing onset for next year.

        last date of time series will return if not available.
        """
        end = self.dates[-1]
        if (year + 1) in self.t1s.index:
            end_tmp = self.t1s[year + 1]
            if not pd.isna(end_tmp):
                end = end_tmp
        return end

    def get_years_available(self, img_dates: list[int]) -> list[int]:
        """Get the years that are available in both SAR and temperature data."""
        year_sar = pd.to_datetime(img_dates).year
        year_ftc = self.years
        return sorted(set(year_sar) & set(year_ftc))

    def get_img_dates_available(self, img_dates: list[int]) -> pd.DatetimeIndex:
        """Get the dates that are available in both SAR and temperature data."""
        img_dates_sar = pd.to_datetime(img_dates)
        img_dates_ftc = self.dates
        df = pd.Series(img_dates_sar, index=img_dates_sar)
        df = df[img_dates_ftc[0] : img_dates_ftc[-1]]
        return df.index


def get_ifgs_by_date_interval(
    ifgs: list[str],
    date_start: str,
    date_end: str,
) -> tuple[list[str], np.ndarray]:
    """Filter ifgs by date interval."""
    ifg_list = []
    ifg_mask = []
    for ifg in ifgs:
        primary = pd.to_datetime(ifg[:8])
        secondary = pd.to_datetime(ifg[-8:])
        date_start = pd.to_datetime(date_start)
        date_end = pd.to_datetime(date_end)

        if primary >= date_start and secondary <= date_end:
            ifg_list.append(ifg)
            ifg_mask.append(True)
        else:
            ifg_mask.append(False)
    return np.array(ifg_list), np.array(ifg_mask)


# def _get_FTI(tmp, dates):
#     ftc = FreezeThawCycle(dates, tmp)
#     TI = ftc.TI.mean()
#     FI = ftc.FI.mean()
#     return TI, FI


# def get_patch_labels_from_temperature(
#     tmp, dates, width, height, sar_tf, sar_crs, tmp_tf, tmp_crs, distance_threshold
# ):
#     try:
#         from sklearn.cluster import AgglomerativeClustering
#     except ImportError:
#         msg = "Please install sklearn package"
#         raise ImportError(msg)

#     sar_crs = crs.CRS.from_user_input(sar_crs)
#     tmp_crs = crs.CRS.from_user_input(tmp_crs)
#     n_date, n_row, n_col = tmp.shape
#     n_pt = n_col * n_row
#     tmp = tmp.reshape(n_date, n_pt).transpose()  # (n_pt,n_date)

#     # calculate TI and FI for every pixel
#     args = [(tmp[i, :], dates) for i in range(n_pt)]

#     args = tqdm(args, desc="  Calculate TI and FI", unit=" pixels")
#     with Pool() as pool:
#         _result = pool.starmap(_get_FTI, args)

#     # cluster by TI and FI
#     cluster = AgglomerativeClustering(
#         n_clusters=None, distance_threshold=distance_threshold
#     ).fit(np.asarray(_result))
#     labels_tmp = cluster.labels_.reshape(n_row, n_col).astype(np.int32)

#     # reproject data from temperature to insar
#     labels_sar = np.full((height, width), np.nan, dtype=np.int32)
#     warp.reproject(
#         source=labels_tmp,
#         src_transform=tmp_tf,
#         src_crs=tmp_crs,
#         destination=labels_sar,
#         dst_transform=sar_tf,
#         dst_crs=sar_crs,
#         resampling=warp.Resampling.nearest,
#         dst_nodata=-1,
#     )
#     return labels_sar, labels_tmp


# def get_patch_labels_from_rough_raster(
#     src_arr,
#     src_tf,
#     src_crs,
#     dst_width,
#     dst_height,
#     dst_tf,
#     dst_crs,
# ):
#     """Generate labels depend on number of pixels of rough raster(source
#     raster), labels are range from 0 to number of pixels of source array.

#     Parameters
#     ----------
#     src_arr, src_tf, src_crs:
#         array, transform, crs(coordinate reference system) of source.
#         transform, crs need to be in rasterio format.
#     dst_width, dst_height, dst_tf, dst_crs:
#         width, height, transform, crs of destination array.
#         transform, crs need to be in rasterio format.

#     Returns
#     -------
#     labels_src, labels_dst: labels with the shape of source and destination

#     """
#     # convert coordinate reference system into rasterio.crs
#     dst_crs = crs.CRS.from_user_input(dst_crs)
#     src_crs = crs.CRS.from_user_input(src_crs)

#     # get the number of rows and columns of source array
#     if src_arr.ndim == 2:
#         n_row, n_col = src_arr.shape
#     elif src_arr.ndim == 3:
#         _, n_row, n_col = src_arr.shape
#     else:
#         msg = "dimension of src_arr must be 2 or 3"
#         raise ValueError(msg)

#     n_pt = n_col * n_row

#     # generate the labels depend on the number of pixels of source array
#     labels_src = np.arange(n_pt).reshape(n_row, n_col).astype(np.int32)

#     # reproject data from source to destination
#     labels_dst = np.full((dst_height, dst_width), -1, dtype=np.int32)
#     warp.reproject(
#         source=labels_src,
#         src_transform=src_tf,
#         src_crs=src_crs,
#         destination=labels_dst,
#         dst_transform=dst_tf,
#         dst_crs=dst_crs,
#         resampling=warp.Resampling.nearest,
#         dst_nodata=-1,
#     )
#     return labels_src, labels_dst
