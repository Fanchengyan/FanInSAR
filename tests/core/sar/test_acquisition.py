import numpy as np
import pandas as pd
import pytest

from faninsar.core import Acquisition, DaySpan


class TestAcquisition:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Setup some sample data for testing
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        self.acquisition = Acquisition(dates)

    def test_creation(self):
        # Test if the Acquisition instance is created correctly
        assert isinstance(self.acquisition, Acquisition)
        assert len(self.acquisition) == 10

    def test_to_xarray(self):
        # Test the to_xarray method
        xarray_var = self.acquisition.to_xarray()
        assert xarray_var.dims == ("Acquisition",)
        assert len(xarray_var) == 10

    def test_stats(self):
        # Test the stats property
        stats = self.acquisition.stats
        assert stats["start"] == "2023-01-01"
        assert stats["end"] == "2023-01-10"
        assert stats["unique"] == 10
        assert stats["total"] == 10

    def test_set_operators(self):
        """Set-style operators return Acquisition views."""
        left = Acquisition(pd.DatetimeIndex(["2023-01-01", "2023-01-03"]))
        right = Acquisition(pd.DatetimeIndex(["2023-01-03", "2023-01-05"]))

        union = left + right
        assert isinstance(union, Acquisition)
        expected = pd.DatetimeIndex(["2023-01-01", "2023-01-03", "2023-01-05"])
        assert union.tolist() == list(expected)
        assert (left | right).tolist() == union.tolist()
        assert left.union(right).tolist() == union.tolist()

        diff = left - right
        assert isinstance(diff, Acquisition)
        assert diff.tolist() == list(pd.DatetimeIndex(["2023-01-01"]))
        assert left.difference(right).tolist() == diff.tolist()

        inter = left & right
        assert isinstance(inter, Acquisition)
        assert inter.tolist() == list(pd.DatetimeIndex(["2023-01-03"]))
        assert left.intersection(right).tolist() == inter.tolist()

        sym = left ^ right
        assert isinstance(sym, Acquisition)
        assert sym.tolist() == list(pd.DatetimeIndex(["2023-01-01", "2023-01-05"]))
        assert left.symmetric_difference(right).tolist() == sym.tolist()

    def test_set_operators_with_datetime_index(self):
        """A plain DatetimeIndex operand works the same as Acquisition."""
        acq = Acquisition(pd.DatetimeIndex(["2023-01-01", "2023-01-03"]))
        other = pd.DatetimeIndex(["2023-01-03"])

        assert (acq - other).tolist() == list(pd.DatetimeIndex(["2023-01-01"]))
        assert (acq + other).tolist() == list(
            pd.DatetimeIndex(["2023-01-01", "2023-01-03"])
        )

    def test_scalar_arithmetic_unchanged(self):
        """Timedelta arithmetic keeps pandas element-wise behaviour."""
        acq = Acquisition(pd.DatetimeIndex(["2023-01-01"]))
        shifted = acq + pd.Timedelta(days=1)
        assert isinstance(shifted, Acquisition)
        assert shifted.tolist() == list(pd.DatetimeIndex(["2023-01-02"]))

    def test_set_operators_reject_scalar(self):
        """A non-index operand falls through to pandas' arithmetic error."""
        acq = Acquisition(pd.DatetimeIndex(["2023-01-01"]))
        with pytest.raises(TypeError):
            acq - "2023-01-01"


class TestCreateDaysSpanIndex:
    """Test the creation of DaySpan with different type inputs."""

    def test_creation_int(self):
        """Test the creation of DaySpan with integer inputs."""
        # Test int list
        days = list(range(10))
        days_span = DaySpan(days)
        assert isinstance(days_span, DaySpan)
        assert len(days_span) == 10
        assert np.issubdtype(days_span.dtype, np.integer)

        # Test int numpy
        days = np.arange(10)
        days_span = DaySpan(days)
        assert isinstance(days_span, DaySpan)
        assert len(days_span) == 10
        assert np.issubdtype(days_span.dtype, np.integer)

        # Test int pandas
        days = pd.Series(range(10))
        days_span = DaySpan(days)
        assert isinstance(days_span, DaySpan)
        assert len(days_span) == 10
        assert np.issubdtype(days_span.dtype, np.integer)

    def test_creation_float(self):
        """Test the creation of DaySpan with float inputs."""
        # Test float list
        days = [float(i) for i in range(10)]
        days_span = DaySpan(days)
        assert isinstance(days_span, DaySpan)
        assert len(days_span) == 10
        assert np.issubdtype(days_span.dtype, np.floating)

        # Test float numpy
        days = np.arange(10).astype(float)
        days_span = DaySpan(days)
        assert isinstance(days_span, DaySpan)
        assert len(days_span) == 10
        assert np.issubdtype(days_span.dtype, np.floating)

        # Test float pandas
        days = pd.Series(range(10)).astype(float)
        days_span = DaySpan(days)
        assert isinstance(days_span, DaySpan)
        assert len(days_span) == 10
        assert np.issubdtype(days_span.dtype, np.floating)

    def test_creation_dtype(self):
        """Test the creation of DaySpan with different dtypes."""
        # Test dtype
        days = np.arange(10)
        days_span32 = DaySpan(days, dtype=np.float32)
        days_span64 = DaySpan(days, dtype=np.float64)
        assert days_span32.dtype == np.float32
        assert days_span64.dtype == np.float64


class TestDaysSpanIndex:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Setup some sample data for testing
        days = pd.to_timedelta(range(10), unit="D")
        self.days_span = DaySpan(days)

    def test_creation(self):
        # Test if the DaySpan instance is created correctly
        assert isinstance(self.days_span, DaySpan)
        assert len(self.days_span) == 10

    def test_to_xarray(self):
        # Test the to_xarray method
        xarray_var = self.days_span.to_xarray()
        assert xarray_var.dims == ("days",)
        assert len(xarray_var) == 10

    def test_stats(self):
        # Test the stats property
        stats = self.days_span.stats
        assert stats["min"] == pd.Timedelta(0)
        assert stats["max"] == pd.Timedelta(9, unit="D")
        assert stats["unique"] == 10
        assert stats["total"] == 10

    def test_plot(self):
        """Test the plot method."""
        import matplotlib.pyplot as plt

        # Create DaySpan with duplicate values to test counting
        days = [6, 6, 12, 12, 24]
        day_span = DaySpan(days)

        # Test basic plot creation
        ax = day_span.plot()
        assert ax is not None
        assert isinstance(ax, plt.Axes)

        # Verify the number of bars matches unique day spans
        bars = ax.patches
        assert len(bars) == 3  # Three unique day spans

        # Verify bar heights (counts)
        bar_heights = [bar.get_height() for bar in bars]
        assert bar_heights == [2, 2, 1]  # Counts for each unique day span

        # Verify axis labels
        assert ax.get_xlabel() == "Day Span"
        assert ax.get_ylabel() == "Count"
        assert ax.get_title() == "Day Span Count"

        plt.close()

    def test_plot_with_custom_ax(self):
        """Test the plot method with custom Axes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result_ax = self.days_span.plot(ax=ax)

        # Verify it returns the same axes
        assert result_ax is ax

        plt.close()
