"""Typing for pairs."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Literal

from numpy.typing import NDArray

from faninsar import Pair, Pairs, TripletLoop, TripletLoops

PairLike = Pair | str | Iterable[datetime] | NDArray
PairsLike = Pairs | Iterable[PairLike] | NDArray

TripletLoopLike = TripletLoop | str | Iterable[datetime]
TripletLoopsLike = TripletLoops | Iterable[TripletLoopLike]

_PairsOrder = Literal["pairs", "primary", "secondary", "days"]
PairsOrder = _PairsOrder | Iterable[_PairsOrder]
