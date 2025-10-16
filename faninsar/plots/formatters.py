"""Matplotlib formatters and locators for phase visualization."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.ticker import Formatter, Locator, MultipleLocator

if TYPE_CHECKING:
    from matplotlib.axis import Axis
    from numpy.typing import NDArray


class PiFormatter(Formatter):
    """Format axis tick labels as multiples of π.

    This formatter displays values in terms of π, making it easier to read
    phase values that are naturally expressed in radians.
    """

    def __init__(
        self,
        denominator: int = 4,
        symbol: str | None = None,
        use_unicode: bool = True,
        latex: bool = False,
    ) -> None:
        r"""Initialize the formatter.

        Parameters
        ----------
        denominator : int, optional
            The maximum denominator for fraction simplification. Default is 4.
        symbol : str, optional
            The symbol to use for pi. Default is 'π'. Can also use 'pi'. When
            ``latex`` is True, the default symbol becomes ``"\\pi"``.
        use_unicode : bool, optional
            Whether to use Unicode π symbol (default) or 'pi' string. Ignored
            when ``latex`` is True and ``symbol`` is not provided.
        latex : bool, optional
            Whether to return labels wrapped in mathtext (LaTeX) for improved
            rendering. Default is False.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from faninsar.plots import PiFormatter, PiLocator
        >>> fig, ax = plt.subplots()
        >>> ax.xaxis.set_major_formatter(PiFormatter())
        >>> ax.xaxis.set_major_locator(PiLocator())

        """
        super().__init__()
        self.denominator = denominator
        self.use_unicode = use_unicode
        self.latex = latex
        if symbol is not None:
            self.symbol = symbol
        elif latex:
            self.symbol = r"\pi"
        else:
            self.symbol = "π" if use_unicode else "pi"

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Format a tick value as a multiple of π.

        Parameters
        ----------
        x : float
            The tick value in radians.
        pos : int, optional
            The tick position (not used).

        Returns
        -------
        str
            Formatted tick label.

        """
        if x == 0:
            return "$0$" if self.latex else "0"

        # Convert to multiples of pi
        x_pi = x / np.pi

        # Handle very small values
        if abs(x_pi) < 1e-10:
            return "$0$" if self.latex else "0"

        # Try to express as a fraction
        try:
            frac = Fraction(x_pi).limit_denominator(self.denominator)
            numerator = frac.numerator
            denominator = frac.denominator

            if abs(float(frac) - x_pi) > 1e-6:
                return self._format_decimal(x_pi)

            if self.latex:
                return self._format_latex(numerator, denominator)
            return self._format_plain(numerator, denominator)
        except (ValueError, ZeroDivisionError):
            # Fallback to decimal representation
            return self._format_decimal(x_pi)

    def _format_plain(self, numerator: int, denominator: int) -> str:
        """Format numerator/denominator as a plain-text multiple of π."""
        if denominator == 1:
            if numerator == 1:
                return f"{self.symbol}"
            if numerator == -1:
                return f"-{self.symbol}"
            return f"{numerator}{self.symbol}"
        if numerator == 1:
            return f"{self.symbol}/{denominator}"
        if numerator == -1:
            return f"-{self.symbol}/{denominator}"
        return f"{numerator}{self.symbol}/{denominator}"

    def _format_latex(self, numerator: int, denominator: int) -> str:
        """Format numerator/denominator as a LaTeX multiple of π."""
        sign = "-" if numerator < 0 else ""
        abs_numerator = abs(numerator)

        if denominator == 1:
            if abs_numerator == 1:
                return rf"${sign}{self.symbol}$"
            return rf"${sign}{abs_numerator}{self.symbol}$"

        if abs_numerator == 1:
            return rf"${sign}\frac{{{self.symbol}}}{{{denominator}}}$"

        return rf"${sign}\frac{{{abs_numerator}{self.symbol}}}{{{denominator}}}$"

    def _format_decimal(self, value: float) -> str:
        """Format a decimal multiple of π."""
        if self.latex:
            sign = "-" if value < 0 else ""
            magnitude = abs(value)
            return rf"${sign}{magnitude:.2g}{self.symbol}$"
        return f"{value:.2g}{self.symbol}"


class PiLocator(Locator):
    """Locate tick positions at multiples of π.

    This locator places ticks at nice intervals based on π, making it easier
    to read phase plots.



    """

    def __init__(self, base: float = 0.5) -> None:
        """Initialize the locator.

        Parameters
        ----------
        base : float, optional
            The base multiple of π for tick spacing. Default is 0.5 (π/2).
            Common values are 1.0 (π), 0.5 (π/2), 0.25 (π/4).

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from faninsar.plots import PiFormatter, PiLocator
        >>> fig, ax = plt.subplots()
        >>> ax.xaxis.set_major_locator(PiLocator(base=0.5))  # Ticks at multiples of π/2
        >>> ax.xaxis.set_major_formatter(PiFormatter())

        """
        super().__init__()
        self.base = base
        self._locator = MultipleLocator(base * np.pi)

    def __call__(self) -> NDArray[np.floating]:
        """Return the locations of the ticks."""
        return self.tick_values(*self.axis.get_view_interval())

    def tick_values(self, vmin: float, vmax: float) -> NDArray[np.floating]:
        """Return tick values within the given range.

        Parameters
        ----------
        vmin : float
            Minimum value of the axis range.
        vmax : float
            Maximum value of the axis range.

        Returns
        -------
        array
            Array of tick locations.

        """
        return self._locator.tick_values(vmin, vmax)

    def view_limits(self, dmin: float, dmax: float) -> tuple[float, float]:
        """Set the view limits to nice values around the data range.

        Parameters
        ----------
        dmin : float
            Minimum data value.
        dmax : float
            Maximum data value.

        Returns
        -------
        tuple
            Nice view limits (vmin, vmax).

        """
        return self._locator.view_limits(dmin, dmax)


def setup_phase_axis(
    axis: Axis,
    base: float = 0.5,
    denominator: int = 4,
    use_unicode: bool = True,
    latex: bool = False,
) -> None:
    """Configure an axis for phase display with π-based ticks.

    This is a convenience function to set up both the formatter and locator
    for a phase axis in one call.

    Parameters
    ----------
    axis : matplotlib.axis.Axis
        The axis to configure (e.g., ax.xaxis or ax.yaxis).
    base : float, optional
        The base multiple of π for tick spacing. Default is 0.5 (π/2).
    denominator : int, optional
        The maximum denominator for fraction simplification. Default is 4.
    use_unicode : bool, optional
        Whether to use Unicode π symbol. Default is True.
    latex : bool, optional
        Whether to format labels using LaTeX. Default is False.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from faninsar.plots import setup_phase_axis
    >>> fig, ax = plt.subplots()
    >>> setup_phase_axis(ax.xaxis, base=0.5)
    >>> ax.plot(
    ...     np.linspace(-np.pi, np.pi, 100), np.sin(np.linspace(-np.pi, np.pi, 100))
    ... )

    """
    axis.set_major_locator(PiLocator(base=base))
    axis.set_major_formatter(
        PiFormatter(
            denominator=denominator,
            use_unicode=use_unicode,
            latex=latex,
        )
    )
