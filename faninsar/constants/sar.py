"""SAR module for wavelength and frequency conversions.

This module provides immutable dataclasses for handling wavelength and frequency
conversions for Synthetic Aperture Radar (SAR) missions. It includes:

Classes
-------
- Wavelength: Immutable dataclass for wavelength with unit conversion capabilities
- Frequency: Immutable dataclass for frequency with unit conversion capabilities
- SAR: Base class for SAR missions with immutable frequency/wavelength properties
- Sentinel1: Predefined Sentinel-1 SAR mission class

Constants
---------
- SENTINEL1_FREQUENCY: Predefined Sentinel-1 frequency (5.405 GHz)
- SENTINEL1_WAVELENGTH: Predefined Sentinel-1 wavelength (55.46 mm)
- SPEED_OF_LIGHT: Speed of light in vacuum (299,792,458 m/s)
- UNIT_WAVELENGTH: Wavelength unit conversion factors
- UNIT_FREQUENCY: Frequency unit conversion factors

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

>>> # Use SAR mission class
>>> s1 = Sentinel1()
>>> print(s1.frequency)
5.405 GHz
>>> print(s1.wavelength)
55.46 mm

Notes
-----
- The speed of light constant is defined as 299,792,458 m/s.
- All unit conversions are performed through a base unit (meters for wavelength,
  Hz for frequency) to ensure consistency and accuracy.
- Unit validation is performed immediately upon instantiation in __post_init__,
  ensuring invalid units are rejected early (fail-fast principle).
- Wavelength and Frequency classes are immutable (frozen dataclasses), making
  them hashable and suitable for use as dictionary keys or in sets.
- Equality comparisons use numpy.isclose() for robust floating-point comparison.
- SAR mission classes can be used with or without instantiation. Subclasses only
  need to define the _frequency class attribute.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import Literal

from faninsar.logging import setup_logger

logger = setup_logger(__name__)


#: Speed of light in vacuum (m/s)
SPEED_OF_LIGHT = 299792458

#: Wavelength unit conversion factors relative to meters
UNIT_WAVELENGTH = {
    "m": 1,
    "cm": 1e-2,
    "dm": 1e-1,
    "mm": 1e-3,
}

#: Frequency unit conversion factors relative to Hz
UNIT_FREQUENCY = {
    "Hz": 1,
    "kHz": 1e3,
    "MHz": 1e6,
    "GHz": 1e9,
}


@dataclass(frozen=True)
class Wavelength:
    """Dataclass for wavelength with unit conversion capabilities.

    This class represents a wavelength value with its associated unit and provides
    methods for converting between different wavelength units and to frequency.
    The class is immutable (frozen) and hashable, allowing instances to be used
    as dictionary keys or in sets.

    Attributes
    ----------
    data : float
        The numerical value of the wavelength.
    unit : Literal["m", "cm", "dm", "mm"]
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
    All conversions are performed by first converting to the base unit (meters)
    and then to the target unit. This ensures consistency and avoids accumulation
    of rounding errors.

    The class is immutable (frozen=True), which means attributes cannot be modified
    after initialization. This ensures thread-safety and allows instances to be
    used as dictionary keys.

    """

    #: The numerical value of the wavelength
    data: float

    #: The unit of the wavelength. Default: m (meters)
    unit: Literal["m", "cm", "dm", "mm"] = "m"

    def __post_init__(self) -> None:
        """Validate the unit immediately after instantiation.

        Raises
        ------
        ValueError
            If the unit is not one of the recognized wavelength units.

        """
        if self.unit not in UNIT_WAVELENGTH:
            msg = (
                f"Invalid unit: {self.unit}. "
                f"Must be one of {list(UNIT_WAVELENGTH.keys())}."
            )
            logger.error(msg)
            raise ValueError(msg)

    def _to_base_unit(self) -> float:
        """Convert wavelength to base unit (meters).

        Returns
        -------
        float
            The wavelength value in meters.

        Notes
        -----
        Unit validation is performed in __post_init__, so this method
        assumes the unit is valid.

        """
        return self.data * UNIT_WAVELENGTH[self.unit]

    def to_unit(self, unit: Literal["m", "cm", "dm", "mm"]) -> Wavelength:
        """Convert wavelength to the specified unit.

        Parameters
        ----------
        unit : Literal["m", "cm", "dm", "mm"]
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
        if unit not in UNIT_WAVELENGTH:
            msg = (
                f"Invalid unit: {unit}. Must be one of {list(UNIT_WAVELENGTH.keys())}."
            )
            logger.error(msg)
            raise ValueError(msg)

        base_value = self._to_base_unit()
        new_value = base_value / UNIT_WAVELENGTH[unit]
        return Wavelength(new_value, unit)

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

    def to_frequency(
        self,
        unit: Literal["GHz", "MHz", "kHz", "Hz"] = "GHz",
    ) -> Frequency:
        """Convert wavelength to frequency.

        Uses the relationship: frequency = speed_of_light / wavelength

        Parameters
        ----------
        unit : Literal["GHz", "MHz", "kHz", "Hz"], optional
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
        return Frequency(SPEED_OF_LIGHT / self.to_m().data, "Hz").to_unit(unit)

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

        """
        if not isinstance(other, Wavelength):
            return NotImplemented
        return bool(np.isclose(self._to_base_unit(), other._to_base_unit()))

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

        """
        # Round to 10 decimal places to ensure equal wavelengths have same hash
        return hash(round(self._to_base_unit(), 10))


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
    unit : Literal["GHz", "MHz", "kHz", "Hz"]
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
    All conversions are performed by first converting to the base unit (Hz)
    and then to the target unit. This ensures consistency and avoids accumulation
    of rounding errors.

    The class is immutable (frozen=True), which means attributes cannot be modified
    after initialization. This ensures thread-safety and allows instances to be
    used as dictionary keys.

    """

    #: The numerical value of the frequency
    data: float

    #: The unit of the frequency. Default: GHz (gigahertz)
    unit: Literal["GHz", "MHz", "kHz", "Hz"] = "GHz"

    def __post_init__(self) -> None:
        """Validate the unit immediately after instantiation.

        Raises
        ------
        ValueError
            If the unit is not one of the recognized frequency units.

        """
        if self.unit not in UNIT_FREQUENCY:
            msg = (
                f"Invalid unit: {self.unit}. "
                f"Must be one of {list(UNIT_FREQUENCY.keys())}."
            )
            logger.error(msg)
            raise ValueError(msg)

    def _to_base_unit(self) -> float:
        """Convert frequency to base unit (Hz).

        Returns
        -------
        float
            The frequency value in Hz.

        Notes
        -----
        Unit validation is performed in __post_init__, so this method
        assumes the unit is valid.

        """
        return self.data * UNIT_FREQUENCY[self.unit]

    def to_unit(self, unit: Literal["GHz", "MHz", "kHz", "Hz"]) -> Frequency:
        """Convert frequency to the specified unit.

        Parameters
        ----------
        unit : Literal["GHz", "MHz", "kHz", "Hz"]
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
        if unit not in UNIT_FREQUENCY:
            msg = f"Invalid unit: {unit}. Must be one of {list(UNIT_FREQUENCY.keys())}."
            logger.error(msg)
            raise ValueError(msg)

        base_value = self._to_base_unit()
        new_value = base_value / UNIT_FREQUENCY[unit]
        return Frequency(new_value, unit)

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

    def to_wavelength(
        self,
        unit: Literal["m", "cm", "dm", "mm"] = "m",
    ) -> Wavelength:
        """Convert frequency to wavelength.

        Uses the relationship: wavelength = speed_of_light / frequency

        Parameters
        ----------
        unit : Literal["m", "cm", "dm", "mm"], optional
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
        return Wavelength(SPEED_OF_LIGHT / self.to_Hz().data).to_unit(unit)

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

        """
        if not isinstance(other, Frequency):
            return NotImplemented
        return bool(np.isclose(self._to_base_unit(), other._to_base_unit()))

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

        """
        # Round to 3 decimal places to ensure equal frequencies have same hash
        return hash(round(self._to_base_unit(), 3))


# ==============================================================================
# SAR Mission Base Class
# ==============================================================================


class SAR:
    """Base class for SAR missions with immutable frequency/wavelength properties.

    This class provides a simple base for defining SAR missions. Subclasses should
    define the `_frequency` class attribute to specify the mission's frequency.
    The wavelength is automatically computed from the frequency.

    Attributes
    ----------
    _frequency : Frequency
        The frequency of the SAR mission (class attribute, defined by subclasses).

    Properties
    ----------
    frequency : Frequency
        The frequency of the SAR mission (read-only).
    wavelength : Wavelength
        The wavelength computed from frequency (read-only).

    Examples
    --------
    >>> # Define a SAR mission by subclassing
    >>> class Sentinel1(SAR):
    ...     _frequency = Frequency(5.405, "GHz")

    >>> # Access frequency and wavelength directly from class
    >>> print(Sentinel1._frequency)
    5.405 GHz

    >>> # Or use with instantiation
    >>> s1 = Sentinel1()
    >>> print(s1.frequency)
    5.405 GHz
    >>> print(s1.wavelength)
    55.46 mm

    Notes
    -----
    - The wavelength is computed from frequency on-the-fly.
    - Properties are read-only - attempting to modify them will raise AttributeError.
    - Subclasses only need to define the `_frequency` class attribute.
    - No __init__ method is needed, making the class simple to inherit.

    """

    _frequency: Frequency

    @property
    def frequency(self) -> Frequency:
        """Get the frequency of the SAR mission.

        Returns
        -------
        Frequency
            The frequency of the SAR mission.

        Examples
        --------
        >>> s1 = Sentinel1()
        >>> print(s1.frequency)
        5.405 GHz

        """
        return self._frequency

    @frequency.setter
    def frequency(self, value: Frequency) -> None:  # noqa: ARG002
        """Prevent modification of frequency.

        Parameters
        ----------
        value : Frequency
            The value to set (ignored).

        Raises
        ------
        AttributeError
            Always raised to prevent modification.

        """
        msg = "frequency for SAR mission is read-only."
        raise AttributeError(msg)

    @property
    def wavelength(self) -> Wavelength:
        """Get the wavelength of the SAR mission.

        The wavelength is computed from frequency.

        Returns
        -------
        Wavelength
            The wavelength of the SAR mission in millimeters.

        Examples
        --------
        >>> s1 = Sentinel1()
        >>> print(s1.wavelength)
        55.46 mm

        """
        return self.frequency.to_wavelength("mm")

    @wavelength.setter
    def wavelength(self, value: Wavelength) -> None:  # noqa: ARG002
        """Prevent modification of wavelength.

        Parameters
        ----------
        value : Wavelength
            The value to set (ignored).

        Raises
        ------
        AttributeError
            Always raised to prevent modification.

        """
        msg = "wavelength for SAR mission is read-only."
        raise AttributeError(msg)


class Sentinel1(SAR):
    """Sentinel-1 SAR mission (C-band, 5.405 GHz).

    Sentinel-1 is a European radar imaging mission with a C-band SAR
    operating at 5.405 GHz frequency (approximately 55.46 mm wavelength).

    Examples
    --------
    >>> # Use without instantiation
    >>> print(Sentinel1._frequency)
    5.405 GHz

    >>> # Or use with instantiation
    >>> s1 = Sentinel1()
    >>> print(s1.frequency)
    5.405 GHz
    >>> print(s1.wavelength)
    55.46 mm

    Notes
    -----
    The frequency is predefined as a class attribute and cannot be changed.

    """

    _frequency = Frequency(5.405, "GHz")


"""
Usage Examples
--------------

1. Basic Unit Conversions
~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> # Wavelength conversions
>>> wl = Wavelength(5.5, "cm")
>>> print(wl.to_mm())
Wavelength(data=55.0, unit='mm')
>>> print(wl.to_m())
Wavelength(data=0.055, unit='m')

>>> # Frequency conversions
>>> freq = Frequency(5.405, "GHz")
>>> print(freq.to_MHz())
Frequency(data=5405.0, unit='MHz')
>>> print(freq.to_Hz())
Frequency(data=5405000000.0, unit='Hz')


2. Wavelength-Frequency Conversions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> # Convert frequency to wavelength
>>> freq = Frequency(5.405, "GHz")
>>> wl = freq.to_wavelength("mm")
>>> print(wl)
55.46 mm

>>> # Convert wavelength to frequency
>>> wl = Wavelength(55.46, "mm")
>>> freq = wl.to_frequency("GHz")
>>> print(freq)
5.405 GHz


3. Using Predefined SAR Mission Constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> # Use Sentinel-1 constants
>>> print(SENTINEL1_FREQUENCY)
5.405 GHz
>>> print(SENTINEL1_WAVELENGTH)
55.46 mm

>>> # Convert to different units
>>> print(SENTINEL1_FREQUENCY.to_MHz())
Frequency(data=5405.0, unit='MHz')
>>> print(SENTINEL1_WAVELENGTH.to_cm())
Wavelength(data=5.546, unit='cm')


4. Equality and Hashing
~~~~~~~~~~~~~~~~~~~~~~~~

>>> # Wavelength equality (different units, same value)
>>> wl1 = Wavelength(1, "m")
>>> wl2 = Wavelength(100, "cm")
>>> print(wl1 == wl2)
True

>>> # Frequency equality
>>> freq1 = Frequency(1, "GHz")
>>> freq2 = Frequency(1000, "MHz")
>>> print(freq1 == freq2)
True

>>> # Use as dictionary keys
>>> wavelengths = {wl1: "L-band", wl2: "L-band"}
>>> print(len(wavelengths))  # Only one entry due to equality
1

>>> # Use in sets
>>> frequencies = {freq1, freq2}
>>> print(len(frequencies))  # Only one entry due to equality
1


5. Unit Validation
~~~~~~~~~~~~~~~~~~

>>> # Invalid units are rejected immediately upon instantiation
>>> try:
...     wl = Wavelength(5.5, "km")  # Invalid unit
... except ValueError as e:
...     print(f"Error: {e}")
Error: Invalid unit: km. Must be one of ['m', 'cm', 'dm', 'mm'].

>>> try:
...     freq = Frequency(5.405, "THz")  # Invalid unit
... except ValueError as e:
...     print(f"Error: {e}")
Error: Invalid unit: THz. Must be one of ['Hz', 'kHz', 'MHz', 'GHz'].


6. Immutability
~~~~~~~~~~~~~~~

>>> # Wavelength and Frequency are immutable (frozen dataclasses)
>>> wl = Wavelength(5.5, "cm")
>>> try:
...     wl.data = 6.0  # This will raise an error
... except AttributeError as e:
...     print(f"Error: {e}")
Error: cannot assign to field 'data'

>>> # SAR mission properties are read-only
>>> s1 = Sentinel1()
>>> try:
...     s1.frequency = Frequency(6.0, "GHz")  # This will raise an error
... except AttributeError as e:
...     print(f"Error: {e}")
Error: frequency for SAR mission is read-only.


Notes
-----
- All unit conversions are performed through base units (meters for wavelength,
  Hz for frequency) to ensure consistency and avoid rounding errors.
- Unit validation is performed immediately upon instantiation in __post_init__,
  ensuring invalid units are rejected early (fail-fast principle).
- Wavelength and Frequency classes are immutable (frozen dataclasses), making
  them thread-safe and suitable for use as dictionary keys or in sets.
- Equality comparisons use numpy.isclose() with default tolerances (rtol=1e-05,
  atol=1e-08) for robust floating-point comparison.
- Hash values are computed from rounded base unit values to ensure that equal
  objects have equal hash values:
  * Wavelength: rounds to 10 decimal places (precision: 0.1 nm)
  * Frequency: rounds to 3 decimal places (precision: 1 Hz)
- SAR mission classes use property setters to prevent modification, raising
  AttributeError when attempting to modify frequency or wavelength.
- To add more SAR missions, simply subclass SAR and define the _frequency
  class attribute, or define new module-level constants following the pattern
  of SENTINEL1_FREQUENCY and SENTINEL1_WAVELENGTH.


See Also
--------
numpy.isclose : Function used for floating-point equality comparison
"""
