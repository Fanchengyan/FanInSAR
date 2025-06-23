import numpy as np
import pandas as pd
import pytest

from faninsar import Acquisition, DaySpan


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
        assert stats["max"] == pd.Timedelta(9, unit='D')
        assert stats["unique"] == 10
        assert stats["total"] == 10
