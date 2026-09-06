"""Tests for `faninsar.plotting.formatters`."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from faninsar.plotting.formatters import PiFormatter, PiLocator, setup_phase_axis


class TestPiFormatter:
    """Tests for the PiFormatter class."""

    def test_common_multiples(self) -> None:
        """PiFormatter should format common multiples of π correctly."""
        formatter = PiFormatter()
        test_cases = {
            0.0: "0",
            np.pi / 2: "π/2",
            -np.pi / 2: "-π/2",
            np.pi: "π",
            -np.pi: "-π",
            3 * np.pi / 4: "3π/4",
        }
        for value, expected in test_cases.items():
            assert formatter(value) == expected

    def test_custom_symbol_and_small_values(self) -> None:
        """PiFormatter should respect custom symbols and handle tiny values."""
        formatter = PiFormatter(symbol="pi")
        assert formatter(np.pi / 2) == "pi/2"
        assert formatter(-np.pi / 4) == "-pi/4"
        tiny_value = 1e-12
        assert formatter(tiny_value) == "0"

    def test_fraction_formatting(self) -> None:
        """PiFormatter should format fractional values correctly."""
        formatter = PiFormatter()
        assert formatter(2 * np.pi / 3) == "2π/3"
        assert formatter(-2 * np.pi / 3) == "-2π/3"

    def test_latex_formatting(self) -> None:
        """PiFormatter should wrap output in mathtext when latex is enabled."""
        formatter = PiFormatter(latex=True)
        assert formatter(0.0) == "$0$"
        assert formatter(np.pi / 2) == "$\\frac{\\pi}{2}$"
        assert formatter(-np.pi) == "$-\\pi$"
        assert formatter(3 * np.pi / 4) == "$\\frac{3\\pi}{4}$"
        decimal_value = np.pi * 1.1
        assert formatter(decimal_value) == "$1.1\\pi$"
        custom_symbol = PiFormatter(latex=True, symbol="\\theta")
        assert custom_symbol(np.pi / 2) == "$\\frac{\\theta}{2}$"


class TestPiLocator:
    """Tests for the PiLocator class."""

    def test_generates_expected_ticks(self) -> None:
        """PiLocator should return multiples of π based on the configured base."""
        locator = PiLocator(base=0.5)
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(locator)
        ax.set_xlim(-np.pi, np.pi)
        ticks = locator()
        expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * np.pi
        assert np.all(np.isin(np.round(expected, 8), np.round(ticks, 8)))
        plt.close(fig)

    def test_tick_values_manual_range(self) -> None:
        """PiLocator.tick_values should honor the provided bounds."""
        locator = PiLocator(base=0.25)
        ticks = locator.tick_values(-np.pi, np.pi / 2)
        assert np.min(ticks) <= -np.pi
        assert np.max(ticks) >= np.pi / 2
        np.testing.assert_allclose(np.diff(ticks), 0.25 * np.pi)


class TestSetupPhaseAxis:
    """Tests for the setup_phase_axis convenience function."""

    def test_applies_formatter_and_locator(self) -> None:
        """setup_phase_axis should configure both locator and formatter."""
        fig, ax = plt.subplots()
        setup_phase_axis(
            ax.xaxis, base=0.25, denominator=8, use_unicode=False, latex=True
        )
        locator = ax.xaxis.get_major_locator()
        formatter = ax.xaxis.get_major_formatter()
        assert isinstance(locator, PiLocator)
        assert locator.base == 0.25
        assert isinstance(formatter, PiFormatter)
        assert formatter.denominator == 8
        assert formatter.symbol == "\\pi"
        assert formatter.latex is True
        plt.close(fig)
