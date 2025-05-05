import pytest

from faninsar.constants.sar import Frequency, Sentinel1, Wavelength


@pytest.fixture
def wavelength_m() -> Wavelength:
    """Return a Wavelength object with a data of 0.031 m."""
    return Wavelength(0.031, "m")


@pytest.fixture
def frequency_ghz() -> Frequency:
    """Return a Frequency object with a data of 5.405 GHz."""
    return Frequency(5.405, "GHz")


@pytest.fixture
def sentinel1() -> Sentinel1:
    """Return a Sentinel1 object."""
    return Sentinel1()


class TestWavelength:
    """Test the Wavelength class."""

    def test_to_cm(self, wavelength_m: Wavelength) -> None:
        """Test the to_cm method."""
        result = wavelength_m.to_cm()
        assert result.data == 3.1
        assert result.unit == "cm"

    def test_to_mm(self, wavelength_m: Wavelength) -> None:
        """Test the to_mm method."""
        result = wavelength_m.to_mm()
        assert result.data == 31
        assert result.unit == "mm"

    def test_to_m(self, wavelength_m: Wavelength) -> None:
        """Test the to_m method."""
        result = wavelength_m.to_m()
        assert result.data == 0.031
        assert result.unit == "m"

    def test_to_dm(self, wavelength_m: Wavelength) -> None:
        """Test the to_dm method."""
        result = wavelength_m.to_dm()
        assert result.data == 0.31
        assert result.unit == "dm"

    def test_to_frequency(self, wavelength_m: Wavelength) -> None:
        """Test the to_frequency method."""
        result = wavelength_m.to_frequency("GHz")
        assert result.data == pytest.approx(9.670724451612903, rel=1e-9)
        assert result.unit == "GHz"


class TestFrequency:
    """Test the Frequency class."""

    def test_to_GHz(self, frequency_ghz: Frequency) -> None:
        """Test the to_GHz method."""
        result = frequency_ghz.to_GHz()
        assert result.data == 5.405
        assert result.unit == "GHz"

    def test_to_MHz(self, frequency_ghz: Frequency) -> None:
        """Test the to_MHz method."""
        result = frequency_ghz.to_MHz()
        assert result.data == 5405
        assert result.unit == "MHz"

    def test_to_kHz(self, frequency_ghz: Frequency) -> None:
        """Test the to_kHz method."""
        result = frequency_ghz.to_kHz()
        assert result.data == 5405000
        assert result.unit == "kHz"

    def test_to_Hz(self, frequency_ghz: Frequency) -> None:
        """Test the to_Hz method."""
        result = frequency_ghz.to_Hz()
        assert result.data == 5405000000
        assert result.unit == "Hz"

    def test_to_wavelength(self, frequency_ghz: Frequency) -> None:
        """Test the to_wavelength method."""
        result = frequency_ghz.to_wavelength("mm")
        assert result.data == pytest.approx(55.451, rel=1e-3)
        assert result.unit == "mm"


class TestSentinel1:
    """Test the Sentinel1 class."""

    def test_frequency(self, sentinel1: Sentinel1) -> None:
        """Test the frequency attribute."""
        assert sentinel1.frequency.data == 5.405
        assert sentinel1.frequency.unit == "GHz"

    def test_wavelength(self, sentinel1: Sentinel1) -> None:
        """Test the wavelength attribute."""
        assert sentinel1.wavelength.data == pytest.approx(55.46576466234968, rel=1e-3)
        assert sentinel1.wavelength.unit == "mm"
