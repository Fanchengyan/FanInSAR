import pytest

from faninsar import Frequency, Sentinel1, Wavelength


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

    def test_equality(self) -> None:
        """Test wavelength equality comparison."""
        wl1 = Wavelength(1, "m")
        wl2 = Wavelength(100, "cm")
        wl3 = Wavelength(1000, "mm")
        wl4 = Wavelength(2, "m")

        # Same wavelength, different units
        assert wl1 == wl2
        assert wl1 == wl3
        assert wl2 == wl3

        # Different wavelengths
        assert wl1 != wl4

    def test_hash(self) -> None:
        """Test wavelength hashing."""
        wl1 = Wavelength(1, "m")
        wl2 = Wavelength(100, "cm")
        wl3 = Wavelength(2, "m")

        # Equal wavelengths should have equal hashes
        assert hash(wl1) == hash(wl2)

        # Can be used in sets
        wavelength_set = {wl1, wl2, wl3}
        assert len(wavelength_set) == 2  # wl1 and wl2 are equal

        # Can be used as dictionary keys
        wavelength_dict = {wl1: "L-band", wl2: "L-band", wl3: "S-band"}
        assert len(wavelength_dict) == 2  # wl1 and wl2 are equal

    def test_immutability(self) -> None:
        """Test that Wavelength instances are immutable."""
        wl = Wavelength(5.5, "cm")

        with pytest.raises(AttributeError):
            wl.data = 6.0  # type: ignore

        with pytest.raises(AttributeError):
            wl.unit = "mm"  # type: ignore

    def test_invalid_unit(self) -> None:
        """Test that invalid units are rejected immediately upon instantiation."""
        # Test various invalid units
        with pytest.raises(ValueError, match="Invalid unit"):
            Wavelength(5.5, "km")  # type: ignore

        with pytest.raises(ValueError, match="Invalid unit"):
            Wavelength(5.5, "inch")  # type: ignore

        with pytest.raises(ValueError, match="Invalid unit"):
            Wavelength(5.5, "nm")  # type: ignore

        with pytest.raises(ValueError, match="Invalid unit"):
            Wavelength(5.5, "um")  # type: ignore

    def test_str_representation(self) -> None:
        """Test the __str__ method."""
        wl = Wavelength(5.5, "cm")
        assert str(wl) == "5.5 cm"

        wl2 = Wavelength(0.055, "m")
        assert str(wl2) == "0.055 m"

    def test_repr_representation(self) -> None:
        """Test the __repr__ method."""
        wl = Wavelength(5.5, "cm")
        repr_str = repr(wl)
        assert "Wavelength" in repr_str
        assert "5.5" in repr_str
        assert "cm" in repr_str

    def test_hash_consistency(self) -> None:
        """Test that equal wavelengths have equal hashes."""
        wl1 = Wavelength(1, "m")
        wl2 = Wavelength(100, "cm")
        wl3 = Wavelength(1000, "mm")
        wl4 = Wavelength(10, "dm")

        # All represent the same wavelength
        assert wl1 == wl2 == wl3 == wl4

        # All should have the same hash
        assert hash(wl1) == hash(wl2) == hash(wl3) == hash(wl4)


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

    def test_equality(self) -> None:
        """Test frequency equality comparison."""
        freq1 = Frequency(1, "GHz")
        freq2 = Frequency(1000, "MHz")
        freq3 = Frequency(1000000, "kHz")
        freq4 = Frequency(2, "GHz")

        # Same frequency, different units
        assert freq1 == freq2
        assert freq1 == freq3
        assert freq2 == freq3

        # Different frequencies
        assert freq1 != freq4

    def test_hash(self) -> None:
        """Test frequency hashing."""
        freq1 = Frequency(1, "GHz")
        freq2 = Frequency(1000, "MHz")
        freq3 = Frequency(2, "GHz")

        # Equal frequencies should have equal hashes
        assert hash(freq1) == hash(freq2)

        # Can be used in sets
        frequency_set = {freq1, freq2, freq3}
        assert len(frequency_set) == 2  # freq1 and freq2 are equal

        # Can be used as dictionary keys
        frequency_dict = {freq1: "L-band", freq2: "L-band", freq3: "S-band"}
        assert len(frequency_dict) == 2  # freq1 and freq2 are equal

    def test_immutability(self) -> None:
        """Test that Frequency instances are immutable."""
        freq = Frequency(5.405, "GHz")

        with pytest.raises(AttributeError):
            freq.data = 6.0  # type: ignore

        with pytest.raises(AttributeError):
            freq.unit = "MHz"  # type: ignore

    def test_invalid_unit(self) -> None:
        """Test that invalid units are rejected immediately upon instantiation."""
        # Test various invalid units
        with pytest.raises(ValueError, match="Invalid unit"):
            Frequency(5.405, "THz")  # type: ignore

        with pytest.raises(ValueError, match="Invalid unit"):
            Frequency(5.405, "mHz")  # type: ignore

        with pytest.raises(ValueError, match="Invalid unit"):
            Frequency(5.405, "PHz")  # type: ignore

        with pytest.raises(ValueError, match="Invalid unit"):
            Frequency(5.405, "rad/s")  # type: ignore

    def test_str_representation(self) -> None:
        """Test the __str__ method."""
        freq = Frequency(5.405, "GHz")
        assert str(freq) == "5.405 GHz"

        freq2 = Frequency(5405, "MHz")
        assert str(freq2) == "5405 MHz"

    def test_repr_representation(self) -> None:
        """Test the __repr__ method."""
        freq = Frequency(5.405, "GHz")
        repr_str = repr(freq)
        assert "Frequency" in repr_str
        assert "5.405" in repr_str
        assert "GHz" in repr_str

    def test_hash_consistency(self) -> None:
        """Test that equal frequencies have equal hashes."""
        freq1 = Frequency(1, "GHz")
        freq2 = Frequency(1000, "MHz")
        freq3 = Frequency(1000000, "kHz")
        freq4 = Frequency(1000000000, "Hz")

        # All represent the same frequency
        assert freq1 == freq2 == freq3 == freq4

        # All should have the same hash
        assert hash(freq1) == hash(freq2) == hash(freq3) == hash(freq4)


class TestSentinel1:
    """Test the Sentinel1 class."""

    def test_frequency(self, sentinel1: Sentinel1) -> None:
        """Test the frequency property."""
        assert sentinel1.frequency.data == 5.405
        assert sentinel1.frequency.unit == "GHz"

    def test_wavelength(self, sentinel1: Sentinel1) -> None:
        """Test the wavelength property."""
        assert sentinel1.wavelength.data == pytest.approx(55.46576466234968, rel=1e-3)
        assert sentinel1.wavelength.unit == "mm"

    def test_class_attribute_access(self) -> None:
        """Test accessing frequency as class attribute without instantiation."""
        # Can access _frequency directly from class
        assert Sentinel1._frequency.data == 5.405
        assert Sentinel1._frequency.unit == "GHz"

    def test_immutability(self, sentinel1: Sentinel1) -> None:
        """Test that Sentinel1 instances are immutable."""
        # Test that frequency property is read-only
        with pytest.raises(AttributeError, match="frequency.*read-only"):
            sentinel1.frequency = Frequency(6.0, "GHz")  # type: ignore

        # Test that wavelength property is read-only
        with pytest.raises(AttributeError, match="wavelength.*read-only"):
            sentinel1.wavelength = Wavelength(60, "mm")  # type: ignore

    def test_wavelength_computed_from_frequency(self, sentinel1: Sentinel1) -> None:
        """Test that wavelength is computed from frequency on-the-fly."""
        # Wavelength should be computed from frequency
        expected_wl = sentinel1.frequency.to_wavelength("mm")
        assert sentinel1.wavelength == expected_wl

    def test_inheritance_pattern(self) -> None:
        """Test that SAR class can be easily inherited."""
        from faninsar.constants.sar import SAR

        # Define a new SAR mission by subclassing
        class TestSAR(SAR):
            _frequency = Frequency(9.65, "GHz")

        # Test instantiation
        test_sar = TestSAR()
        assert test_sar.frequency.data == 9.65
        assert test_sar.frequency.unit == "GHz"

        # Wavelength should be computed automatically
        expected_wl = test_sar.frequency.to_wavelength("mm")
        assert test_sar.wavelength == expected_wl

        # Properties should be read-only
        with pytest.raises(AttributeError, match="frequency.*read-only"):
            test_sar.frequency = Frequency(10.0, "GHz")  # type: ignore



class TestWavelengthFrequencyConversion:
    """Test conversions between wavelength and frequency."""

    def test_roundtrip_conversion(self) -> None:
        """Test that wavelength -> frequency -> wavelength preserves value."""
        original_wl = Wavelength(5.5, "cm")
        freq = original_wl.to_frequency("GHz")
        converted_wl = freq.to_wavelength("cm")

        assert converted_wl == original_wl

    def test_roundtrip_conversion_frequency(self) -> None:
        """Test that frequency -> wavelength -> frequency preserves value."""
        original_freq = Frequency(5.405, "GHz")
        wl = original_freq.to_wavelength("mm")
        converted_freq = wl.to_frequency("GHz")

        assert converted_freq == original_freq

    def test_conversion_accuracy(self) -> None:
        """Test the accuracy of wavelength-frequency conversions."""
        # Test with known values
        # c = λ * f
        # 299792458 m/s = 0.055465 m * 5.405 GHz
        freq = Frequency(5.405, "GHz")
        wl = freq.to_wavelength("m")

        # Calculate c from converted values
        c_calculated = wl.data * freq.data * 1e9
        assert c_calculated == pytest.approx(299792458, rel=1e-6)

    def test_multiple_unit_conversions(self) -> None:
        """Test multiple consecutive unit conversions."""
        # Start with a wavelength
        wl1 = Wavelength(5.5, "cm")

        # Convert through multiple units
        wl2 = wl1.to_mm()
        wl3 = wl2.to_m()
        wl4 = wl3.to_dm()
        wl5 = wl4.to_cm()

        # Should end up with the same value (using equality which uses numpy.isclose)
        assert wl1 == wl5
        # Data might have slight floating-point differences
        assert wl5.data == pytest.approx(wl1.data, rel=1e-9)
        assert wl1.unit == wl5.unit


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_wavelength(self) -> None:
        """Test with very small wavelength values."""
        wl = Wavelength(0.001, "mm")  # 1 micrometer
        assert wl.to_m().data == pytest.approx(1e-6, rel=1e-9)

        # Convert to frequency
        freq = wl.to_frequency("GHz")
        assert freq.data > 0

    def test_very_large_wavelength(self) -> None:
        """Test with very large wavelength values."""
        wl = Wavelength(10, "m")  # 10 meters
        assert wl.to_mm().data == 10000

        # Convert to frequency
        freq = wl.to_frequency("MHz")
        assert freq.data == pytest.approx(29.9792458, rel=1e-6)

    def test_very_high_frequency(self) -> None:
        """Test with very high frequency values."""
        freq = Frequency(100, "GHz")  # 100 GHz
        wl = freq.to_wavelength("mm")
        assert wl.data == pytest.approx(2.99792458, rel=1e-6)

    def test_very_low_frequency(self) -> None:
        """Test with very low frequency values."""
        freq = Frequency(100, "MHz")  # 100 MHz
        wl = freq.to_wavelength("m")
        assert wl.data == pytest.approx(2.99792458, rel=1e-6)

    def test_equality_with_floating_point_errors(self) -> None:
        """Test equality comparison handles floating-point errors."""
        # Create wavelengths that should be equal but might have
        # slight floating-point differences
        wl1 = Wavelength(1.0, "m")
        wl2 = wl1.to_cm().to_mm().to_m()  # Multiple conversions

        # Should still be equal due to numpy.isclose()
        assert wl1 == wl2

    def test_hash_stability(self) -> None:
        """Test that hash values are stable across conversions."""
        wl1 = Wavelength(1, "m")
        wl2 = wl1.to_cm().to_m()  # Convert and back

        # Hashes should be equal
        assert hash(wl1) == hash(wl2)

        # Should work in sets
        wl_set = {wl1, wl2}
        assert len(wl_set) == 1
