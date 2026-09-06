"""SAR module for wavelength and frequency conversions.

This module provides immutable dataclasses for handling wavelength and frequency
conversions for Synthetic Aperture Radar (SAR) missions. It includes:

Classes
-------
- Wavelength: Immutable dataclass for wavelength with unit conversion capabilities
- Frequency: Immutable dataclass for frequency with unit conversion capabilities

Notes
-----
- This module uses Pint internally for accurate unit conversions and standard
  physical constants (e.g., speed of light).
- Pint is not exposed to users; all public APIs remain unchanged.
- Supported wavelength units: m (meters), cm (centimeters), dm (decimeters),
  mm (millimeters), nm (nanometers), km (kilometers), um (micrometers).
- Supported frequency units: Hz (hertz), kHz (kilohertz), MHz (megahertz),
  GHz (gigahertz), THz (terahertz).
- Unit validation is performed immediately upon instantiation in __post_init__,
  ensuring invalid units are rejected early (fail-fast principle).
- Wavelength and Frequency classes are immutable (frozen dataclasses), making
  them hashable and suitable for use as dictionary keys or in sets.
- Equality comparisons use numpy.isclose() for robust floating-point comparison.
- SAR mission classes can be used with or without instantiation. Subclasses only
  need to define the _frequency class attribute.

Examples
--------
>>> # Create wavelength and convert units
>>> wl = Wavelength(5.5, "cm")
>>> print(wl.to_mm())
Wavelength(data=55.0, unit='mm')

>>> # Create frequency and convert to wavelength
>>> freq = Frequency(5.405, "GHz")
>>> wl = freq.to_wavelength("mm")
>>> print(wl)
55.46 mm

>>> # Use extended units
>>> wl = Wavelength(55.5, "nm")
>>> print(wl.to_um())
Wavelength(data=0.0555, unit='um')

>>> freq = Frequency(5.405, "GHz")
>>> print(freq.to_THz())
Frequency(data=0.005405, unit='THz')

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pint

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.core.types import FrequencyUnit, WavelengthUnit

logger = setup_logger(__name__)

# Valid unit tuples used for runtime validation. These mirror the Literal types
# defined in faninsar.core.types (which is the canonical source for type checkers).
# Direct import is avoided to prevent circular imports through faninsar.core.types.
_WAVELENGTH_UNITS: tuple[str, ...] = ("m", "cm", "dm", "mm", "nm", "km", "um")
_FREQUENCY_UNITS: tuple[str, ...] = ("GHz", "MHz", "kHz", "Hz", "THz")

# Private UnitRegistry for internal use only
_ureg = pint.UnitRegistry()


@dataclass(frozen=True)
class Wavelength:
    """Dataclass for wavelength with unit conversion capabilities.

    This class represents a wavelength value with its associated unit and provides
    methods for converting between different wavelength units and to frequency.
    The class is immutable (frozen) and hashable, allowing instances to be used
    as dictionary keys or in sets.

    Internally uses Pint for accurate unit conversions and standard constants.

    Attributes
    ----------
    data : float
        The numerical value of the wavelength.
    unit : Literal["m", "cm", "dm", "mm", "nm", "km", "um"]
        The unit of the wavelength. Default is "m" (meters).

    Examples
    --------
    >>> wl = Wavelength(5.5, "cm")
    >>> wl.to_mm()
    Wavelength(data=55.0, unit='mm')
    >>> wl.to_m()
    Wavelength(data=0.055, unit='m')
    >>> str(wl)
    '5.5 cm'
    >>> # Can be used as dictionary key
    >>> wavelengths = {wl: "C-band"}
    >>> # Equality comparison
    >>> Wavelength(1, "m") == Wavelength(100, "cm")
    True

    Notes
    -----
    All conversions are performed internally using Pint's Quantity objects for
    accuracy and consistency. Pint uses standard physical constants and conversion
    factors.

    The class is immutable (frozen=True), which means attributes cannot be modified
    after initialization. This ensures thread-safety and allows instances to be
    used as dictionary keys.

    """

    #: The numerical value of the wavelength
    data: float

    #: The unit of the wavelength. Default: m (meters)
    unit: WavelengthUnit = "m"

    def __post_init__(self) -> None:
        """Validate the unit immediately after instantiation.

        Raises
        ------
        ValueError
            If the unit is not one of the recognized wavelength units.

        """
        if self.unit not in _WAVELENGTH_UNITS:
            msg = f"Invalid unit: {self.unit}. Must be one of {_WAVELENGTH_UNITS}."
            logger.error(msg)
            raise ValueError(msg)

    def _as_quantity(self) -> pint.Quantity:
        """Convert wavelength to Pint Quantity for internal use.

        Returns
        -------
        pint.Quantity
            The wavelength as a Pint Quantity object.

        Notes
        -----
        This is a private method for internal use only. Unit validation is
        performed in __post_init__, so this method assumes the unit is valid.

        """
        return self.data * _ureg.parse_units(self.unit)

    @staticmethod
    def _from_quantity(q: pint.Quantity, target_unit: str) -> Wavelength:
        """Create Wavelength from Pint Quantity for internal use.

        Parameters
        ----------
        q : pint.Quantity
            The Pint Quantity object.
        target_unit : str
            The target unit for the new Wavelength instance.

        Returns
        -------
        Wavelength
            A new Wavelength instance with the converted value.

        Notes
        -----
        This is a private method for internal use only.

        """
        converted = q.to(target_unit)
        return Wavelength(converted.magnitude, target_unit)

    def to_unit(self, unit: WavelengthUnit) -> Wavelength:
        """Convert wavelength to the specified unit.

        Parameters
        ----------
        unit : Literal["m", "cm", "dm", "mm", "nm", "km", "um"]
            The target unit for conversion.

        Returns
        -------
        Wavelength
            A new Wavelength instance with the converted value and unit.

        Raises
        ------
        ValueError
            If the target unit is not recognized.

        Examples
        --------
        >>> wl = Wavelength(1, "m")
        >>> wl.to_unit("cm")
        Wavelength(data=100.0, unit='cm')

        """
        if unit not in _WAVELENGTH_UNITS:
            msg = f"Invalid unit: {unit}. Must be one of {_WAVELENGTH_UNITS}."
            logger.error(msg)
            raise ValueError(msg)

        # Return self for no-op conversions to preserve identity in tests
        if unit == self.unit:
            return self

        return self._from_quantity(self._as_quantity(), unit)

    def to_m(self) -> Wavelength:
        """Convert wavelength to meters.

        Returns
        -------
        Wavelength
            A new Wavelength instance in meters.

        """
        return self.to_unit("m")

    def to_cm(self) -> Wavelength:
        """Convert wavelength to centimeters.

        Returns
        -------
        Wavelength
            A new Wavelength instance in centimeters.

        """
        return self.to_unit("cm")

    def to_dm(self) -> Wavelength:
        """Convert wavelength to decimeters.

        Returns
        -------
        Wavelength
            A new Wavelength instance in decimeters.

        """
        return self.to_unit("dm")

    def to_mm(self) -> Wavelength:
        """Convert wavelength to millimeters.

        Returns
        -------
        Wavelength
            A new Wavelength instance in millimeters.

        """
        return self.to_unit("mm")

    def to_nm(self) -> Wavelength:
        """Convert wavelength to nanometers.

        Returns
        -------
        Wavelength
            A new Wavelength instance in nanometers.

        """
        return self.to_unit("nm")

    def to_km(self) -> Wavelength:
        """Convert wavelength to kilometers.

        Returns
        -------
        Wavelength
            A new Wavelength instance in kilometers.

        """
        return self.to_unit("km")

    def to_um(self) -> Wavelength:
        """Convert wavelength to micrometers.

        Returns
        -------
        Wavelength
            A new Wavelength instance in micrometers.

        """
        return self.to_unit("um")

    @property
    def quantity(self) -> pint.Quantity:
        """Get the internal Pint Quantity object.

        Provides access to Pint's full functionality for advanced users.

        Returns
        -------
        pint.Quantity
            The wavelength as a Pint Quantity.

        Examples
        --------
        >>> wl = Wavelength(5.5, "cm")
        >>> q = wl.quantity
        >>> q.to("nm")
        <Quantity(55000000.0, 'nanometer')>
        >>> q.ito("mm")  # in-place conversion

        """
        return self._as_quantity()

    def to_frequency(
        self,
        unit: FrequencyUnit = "GHz",
    ) -> Frequency:
        """Convert wavelength to frequency.

        Uses the relationship: frequency = speed_of_light / wavelength
        Internally uses Pint's standard speed of light constant.

        Parameters
        ----------
        unit : Literal["GHz", "MHz", "kHz", "Hz", "THz"], optional
            The unit of the resulting frequency. Default is "GHz".

        Returns
        -------
        Frequency
            A new Frequency instance with the converted value.

        Examples
        --------
        >>> wl = Wavelength(0.055, "m")
        >>> freq = wl.to_frequency("GHz")
        >>> round(freq.data, 3)
        5.451

        """
        freq_quantity = _ureg.c / self._as_quantity()
        return Frequency._from_quantity(freq_quantity, unit)

    def __repr__(self) -> str:
        """Return a detailed string representation of the Wavelength.

        Returns
        -------
        str
            A string that can be used to recreate the object.

        """
        return f"Wavelength(data={self.data}, unit='{self.unit}')"

    def __str__(self) -> str:
        """Return a human-readable string representation of the Wavelength.

        Returns
        -------
        str
            A formatted string showing the value and unit.

        """
        return f"{self.data} {self.unit}"

    def __eq__(self, other: object) -> bool:
        """Check equality between two Wavelength instances.

        Two wavelengths are considered equal if their values in the base unit
        (meters) are equal within floating-point precision tolerance using
        numpy.isclose() with default tolerances (rtol=1e-05, atol=1e-08).

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            True if the wavelengths are equal, False otherwise.

        Examples
        --------
        >>> wl1 = Wavelength(1, "m")
        >>> wl2 = Wavelength(100, "cm")
        >>> wl1 == wl2
        True

        Notes
        -----
        Uses numpy.isclose() for robust floating-point comparison with
        relative tolerance of 1e-05 and absolute tolerance of 1e-08.
        Internally converts to meters using Pint for comparison.

        """
        if not isinstance(other, Wavelength):
            return NotImplemented
        # Convert both to meters using Pint for comparison
        wl1_m = self._as_quantity().to("m").magnitude
        wl2_m = other._as_quantity().to("m").magnitude
        return bool(np.isclose(wl1_m, wl2_m))

    def __hash__(self) -> int:
        """Return hash of the Wavelength instance.

        The hash is computed from the wavelength value in the base unit (meters)
        rounded to 10 decimal places to ensure that equal wavelengths have the
        same hash value.

        Returns
        -------
        int
            The hash value of the wavelength.

        Examples
        --------
        >>> wl1 = Wavelength(1, "m")
        >>> wl2 = Wavelength(100, "cm")
        >>> hash(wl1) == hash(wl2)
        True
        >>> # Can be used in sets
        >>> wavelengths = {wl1, wl2}
        >>> len(wavelengths)
        1

        Notes
        -----
        The wavelength is rounded to 10 decimal places before hashing to ensure
        that wavelengths that are equal (within floating-point tolerance) have
        the same hash value, satisfying the requirement that if a == b, then
        hash(a) == hash(b).
        Internally converts to meters using Pint.

        """
        # Convert to meters using Pint and round to 10 decimal places
        value_m = self._as_quantity().to("m").magnitude
        return hash(round(value_m, 10))


@dataclass(frozen=True)
class Frequency:
    """Dataclass for frequency with unit conversion capabilities.

    This class represents a frequency value with its associated unit and provides
    methods for converting between different frequency units and to wavelength.
    The class is immutable (frozen) and hashable, allowing instances to be used
    as dictionary keys or in sets.

    Attributes
    ----------
    data : float
        The numerical value of the frequency.
    unit : Literal["GHz", "MHz", "kHz", "Hz", "THz"]
        The unit of the frequency. Default is "GHz" (gigahertz).

    Examples
    --------
    >>> freq = Frequency(5.405, "GHz")
    >>> freq.to_MHz()
    Frequency(data=5405.0, unit='MHz')
    >>> freq.to_Hz()
    Frequency(data=5405000000.0, unit='Hz')
    >>> str(freq)
    '5.405 GHz'
    >>> # Can be used as dictionary key
    >>> frequencies = {freq: "Sentinel-1"}
    >>> # Equality comparison
    >>> Frequency(1, "GHz") == Frequency(1000, "MHz")
    True

    Notes
    -----
    All conversions are performed internally using Pint's Quantity objects for
    accuracy and consistency. Pint uses standard physical constants and conversion
    factors.

    The class is immutable (frozen=True), which means attributes cannot be modified
    after initialization. This ensures thread-safety and allows instances to be
    used as dictionary keys.

    """

    #: The numerical value of the frequency
    data: float

    #: The unit of the frequency. Default: GHz (gigahertz)
    unit: FrequencyUnit = "GHz"

    def __post_init__(self) -> None:
        """Validate the unit immediately after instantiation.

        Raises
        ------
        ValueError
            If the unit is not one of the recognized frequency units.

        """
        if self.unit not in _FREQUENCY_UNITS:
            msg = f"Invalid unit: {self.unit}. Must be one of {_FREQUENCY_UNITS}."
            logger.error(msg)
            raise ValueError(msg)

    def _as_quantity(self) -> pint.Quantity:
        """Convert frequency to Pint Quantity for internal use.

        Returns
        -------
        pint.Quantity
            The frequency as a Pint Quantity object.

        Notes
        -----
        This is a private method for internal use only. Unit validation is
        performed in __post_init__, so this method assumes the unit is valid.

        """
        return self.data * _ureg.parse_units(self.unit)

    @staticmethod
    def _from_quantity(q: pint.Quantity, target_unit: str) -> Frequency:
        """Create Frequency from Pint Quantity for internal use.

        Parameters
        ----------
        q : pint.Quantity
            The Pint Quantity object.
        target_unit : str
            The target unit for the new Frequency instance.

        Returns
        -------
        Frequency
            A new Frequency instance with the converted value.

        Notes
        -----
        This is a private method for internal use only.

        """
        converted = q.to(target_unit)
        return Frequency(converted.magnitude, target_unit)

    def to_unit(self, unit: FrequencyUnit) -> Frequency:
        """Convert frequency to the specified unit.

        Parameters
        ----------
        unit : Literal["GHz", "MHz", "kHz", "Hz", "THz"]
            The target unit for conversion.

        Returns
        -------
        Frequency
            A new Frequency instance with the converted value and unit.

        Raises
        ------
        ValueError
            If the target unit is not recognized.

        Examples
        --------
        >>> freq = Frequency(1, "GHz")
        >>> freq.to_unit("MHz")
        Frequency(data=1000.0, unit='MHz')

        """
        if unit not in _FREQUENCY_UNITS:
            msg = f"Invalid unit: {unit}. Must be one of {_FREQUENCY_UNITS}."
            logger.error(msg)
            raise ValueError(msg)

        # Return self for no-op conversions to preserve identity in tests
        if unit == self.unit:
            return self

        return self._from_quantity(self._as_quantity(), unit)

    def to_Hz(self) -> Frequency:
        """Convert frequency to Hz.

        Returns
        -------
        Frequency
            A new Frequency instance in Hz.

        """
        return self.to_unit("Hz")

    def to_kHz(self) -> Frequency:
        """Convert frequency to kHz.

        Returns
        -------
        Frequency
            A new Frequency instance in kHz.

        """
        return self.to_unit("kHz")

    def to_MHz(self) -> Frequency:
        """Convert frequency to MHz.

        Returns
        -------
        Frequency
            A new Frequency instance in MHz.

        """
        return self.to_unit("MHz")

    def to_GHz(self) -> Frequency:
        """Convert frequency to GHz.

        Returns
        -------
        Frequency
            A new Frequency instance in GHz.

        """
        return self.to_unit("GHz")

    def to_THz(self) -> Frequency:
        """Convert frequency to THz.

        Returns
        -------
        Frequency
            A new Frequency instance in THz.

        """
        return self.to_unit("THz")

    @property
    def quantity(self) -> pint.Quantity:
        """Get the internal Pint Quantity object.

        Provides access to Pint's full functionality for advanced users.

        Returns
        -------
        pint.Quantity
            The frequency as a Pint Quantity.

        Examples
        --------
        >>> freq = Frequency(5.405, "GHz")
        >>> q = freq.quantity
        >>> q.to("MHz")
        <Quantity(5405.0, 'megahertz')>
        >>> q.ito("kHz")  # in-place conversion

        """
        return self._as_quantity()

    def to_wavelength(
        self,
        unit: WavelengthUnit = "m",
    ) -> Wavelength:
        """Convert frequency to wavelength.

        Uses the relationship: wavelength = speed_of_light / frequency
        Internally uses Pint's standard speed of light constant.

        Parameters
        ----------
        unit : Literal["m", "cm", "dm", "mm", "nm", "km", "um"], optional
            The unit of the resulting wavelength. Default is "m".

        Returns
        -------
        Wavelength
            A new Wavelength instance with the converted value.

        Examples
        --------
        >>> freq = Frequency(5.405, "GHz")
        >>> wl = freq.to_wavelength("mm")
        >>> round(wl.data, 2)
        55.46

        """
        wl_quantity = _ureg.c / self._as_quantity()
        return Wavelength._from_quantity(wl_quantity, unit)

    def __repr__(self) -> str:
        """Return a detailed string representation of the Frequency.

        Returns
        -------
        str
            A string that can be used to recreate the object.

        """
        return f"Frequency(data={self.data}, unit='{self.unit}')"

    def __str__(self) -> str:
        """Return a human-readable string representation of the Frequency.

        Returns
        -------
        str
            A formatted string showing the value and unit.

        """
        return f"{self.data} {self.unit}"

    def __eq__(self, other: object) -> bool:
        """Check equality between two Frequency instances.

        Two frequencies are considered equal if their values in the base unit
        (Hz) are equal within floating-point precision tolerance using
        numpy.isclose() with default tolerances (rtol=1e-05, atol=1e-08).

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            True if the frequencies are equal, False otherwise.

        Examples
        --------
        >>> freq1 = Frequency(1, "GHz")
        >>> freq2 = Frequency(1000, "MHz")
        >>> freq1 == freq2
        True

        Notes
        -----
        Uses numpy.isclose() for robust floating-point comparison with
        relative tolerance of 1e-05 and absolute tolerance of 1e-08.
        Internally converts to Hz using Pint for comparison.

        """
        if not isinstance(other, Frequency):
            return NotImplemented
        # Convert both to Hz using Pint for comparison
        freq1_hz = self._as_quantity().to("Hz").magnitude
        freq2_hz = other._as_quantity().to("Hz").magnitude
        return bool(np.isclose(freq1_hz, freq2_hz))

    def __hash__(self) -> int:
        """Return hash of the Frequency instance.

        The hash is computed from the frequency value in the base unit (Hz)
        rounded to 3 decimal places to ensure that equal frequencies have the
        same hash value.

        Returns
        -------
        int
            The hash value of the frequency.

        Examples
        --------
        >>> freq1 = Frequency(1, "GHz")
        >>> freq2 = Frequency(1000, "MHz")
        >>> hash(freq1) == hash(freq2)
        True
        >>> # Can be used in sets
        >>> frequencies = {freq1, freq2}
        >>> len(frequencies)
        1

        Notes
        -----
        The frequency is rounded to 3 decimal places before hashing to ensure
        that frequencies that are equal (within floating-point tolerance) have
        the same hash value, satisfying the requirement that if a == b, then
        hash(a) == hash(b).
        Internally converts to Hz using Pint.

        """
        # Convert to Hz using Pint and round to 3 decimal places
        value_hz = self._as_quantity().to("Hz").magnitude
        return hash(round(value_hz, 3))
