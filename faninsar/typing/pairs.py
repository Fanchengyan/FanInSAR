"""Typing for pairs."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Sequence, Union

from faninsar import Pair, Pairs, TripletLoop, TripletLoops

PairLike = Union[Pair, str, Sequence[datetime, datetime]]
PairsLike = Union[Pairs, Sequence[PairLike]]

TripletLoopLike = Union[TripletLoop, str, Sequence[datetime, datetime, datetime]]
TripletLoopsLike = Union[TripletLoops, Sequence[TripletLoopLike]]

_PairsOrder = Literal["pairs", "primary", "secondary", "days"]
PairsOrder = Union[_PairsOrder, Sequence[_PairsOrder]]
