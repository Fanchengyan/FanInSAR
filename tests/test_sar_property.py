#!/usr/bin/env python3
"""Test script for the refactored SAR classes using pytest."""

import pytest

from faninsar import Frequency, Sentinel1, Wavelength


def test_wavelength_conversions():
    """Test wavelength unit conversions."""
    # Test basic conversions
    w1 = Wavelength(1.0, "m")
    assert w1.to_dm().data == 10.0
    assert w1.to_cm().data == 100.0
    assert w1.to_mm().data == 1000.0

    # Test reverse conversions
    w2 = Wavelength(100.0, "cm")
    assert w2.to_m().data == 1.0

    w3 = Wavelength(1000.0, "mm")
    assert w3.to_m().data == 1.0

    # Test same unit conversion
    w4 = Wavelength(5.0, "cm")
    w4_same = w4.to_cm()
    assert w4.data == w4_same.data
    assert w4 is w4_same


def test_frequency_conversions():
    """Test frequency unit conversions."""
    # Test basic conversions
    f1 = Frequency(1.0, "GHz")
    assert f1.to_MHz().data == 1e3
    assert f1.to_kHz().data == 1e6
    assert f1.to_Hz().data == 1e9

    # Test reverse conversions
    f2 = Frequency(1000.0, "MHz")
    assert f2.to_GHz().data == 1.0

    f3 = Frequency(1e9, "Hz")
    assert f3.to_GHz().data == 1.0

    # Test same unit conversion
    f4 = Frequency(5.405, "GHz")
    f4_same = f4.to_GHz()
    assert f4.data == f4_same.data
    assert f4 is f4_same


def test_frequency_wavelength_conversion():
    """Test conversion between frequency and wavelength."""
    # Test Sentinel-1 frequency
    sentinel1 = Sentinel1()
    freq = sentinel1.frequency
    wavelength = sentinel1.wavelength

    # Convert frequency to wavelength
    converted_wavelength = freq.to_wavelength("cm")
    assert pytest.approx(converted_wavelength.data) == wavelength.to_cm().data

    # Convert wavelength back to frequency
    converted_frequency = converted_wavelength.to_frequency("GHz")
    assert pytest.approx(converted_frequency.data) == freq.data


def test_invalid_wavelength_unit():
    """Test error handling for invalid wavelength unit."""
    w = Wavelength(1.0, "m")
    with pytest.raises(ValueError):
        w.to_unit("invalid")


def test_invalid_frequency_unit():
    """Test error handling for invalid frequency unit."""
    f = Frequency(1.0, "GHz")
    with pytest.raises(ValueError):
        f.to_unit("invalid")
