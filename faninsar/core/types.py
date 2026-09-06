"""Typing for geospatial data."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, TypeAlias, get_args

from pyproj.crs.crs import CRS as PyprojCRS  # noqa: N811
from rasterio.crs import CRS as RasterioCRS  # noqa: N811
from rasterio.enums import Resampling as RasterioResampling

if TYPE_CHECKING:
    from datetime import datetime

    from numpy.typing import NDArray

    from faninsar.core.loops import TripletLoop, TripletLoops
    from faninsar.core.pair import Pair, Pairs

CrsLike = PyprojCRS | RasterioCRS | tuple[str, str] | dict[str, str] | str | int
ResamplingLike = (
    Literal[
        "nearest",
        "average",
        "bilinear",
        "cubic",
        "cubic_spline",
        "lanczos",
        "mode",
        "gauss",
        "max",
        "min",
        "med",
        "q1",
        "q3",
        "sum",
        "rms",
    ]
    | RasterioResampling
)

PairLike: TypeAlias = "Pair | str | Iterable[datetime] | NDArray"
PairsLike: TypeAlias = "Pairs | Iterable[PairLike] | NDArray"
TripletLoopLike: TypeAlias = "TripletLoop | str | Iterable[datetime]"
TripletLoopsLike: TypeAlias = "TripletLoops | Iterable[TripletLoopLike]"

_PairsOrder = Literal["pairs", "primary", "secondary", "days"]
PairsOrder: TypeAlias = _PairsOrder | Iterable[_PairsOrder]

WavelengthUnit: TypeAlias = Literal["m", "cm", "dm", "mm", "nm", "km", "um"]
FrequencyUnit: TypeAlias = Literal["GHz", "MHz", "kHz", "Hz", "THz"]
WAVELENGTH_UNITS: tuple[str, ...] = get_args(WavelengthUnit)
FREQUENCY_UNITS: tuple[str, ...] = get_args(FrequencyUnit)

__all__ = [
    "FREQUENCY_UNITS",
    "WAVELENGTH_UNITS",
    "CrsLike",
    "FrequencyUnit",
    "PairLike",
    "PairsLike",
    "PairsOrder",
    "ResamplingLike",
    "TripletLoopLike",
    "TripletLoopsLike",
    "WavelengthUnit",
]
