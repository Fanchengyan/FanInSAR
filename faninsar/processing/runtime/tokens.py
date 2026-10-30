"""Array tokens and runtime PhysicalType assertions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from faninsar.core.physical import PhysicalType
from faninsar.processing.errors import ProcessingContractError


@dataclass(frozen=True, slots=True)
class ArrayToken:
    """Lightweight runtime token carrying physical type metadata."""

    physical: PhysicalType
    payload: Any = None
    pair_id: str | None = None
    stage: str | None = None


def assert_token(
    state: Any,
    expected: PhysicalType | set[PhysicalType] | frozenset[PhysicalType],
    *,
    stage: str | None = None,
) -> None:
    """Assert that *state* carries the expected PhysicalType.

    Parameters
    ----------
    state : Any
        Object with a ``physical`` attribute, an :class:`ArrayToken`, or a
        plain :class:`PhysicalType`.
    expected : PhysicalType or set of PhysicalType
        Allowed physical type(s).
    stage : str, optional
        Stage name for error messages.

    Raises
    ------
    ProcessingContractError
        When the physical type is missing or mismatched.

    """
    allowed = {expected} if isinstance(expected, PhysicalType) else set(expected)
    physical = _extract_physical(state)
    stage_name = stage or "assert_token"
    pair_id = getattr(state, "pair_id", None)
    if physical is None:
        hint = (
            "state has no physical type; expected one of "
            f"{sorted(p.value for p in allowed)}"
        )
        raise ProcessingContractError(_format_stage_message(stage_name, pair_id, hint))
    if physical not in allowed:
        hint = (
            f"physical type {physical.value!r} not in "
            f"{sorted(p.value for p in allowed)}"
        )
        raise ProcessingContractError(_format_stage_message(stage_name, pair_id, hint))


def _format_stage_message(
    stage: str,
    pair: str | None,
    hint: str,
) -> str:
    """Render a stage failure message with stage name, pair, and hint."""
    parts = [f"stage={stage}"]
    if pair:
        parts.append(f"pair={pair}")
    parts.append(hint)
    return ": ".join(parts)


def _extract_physical(state: Any) -> PhysicalType | None:
    if isinstance(state, PhysicalType):
        return state
    if isinstance(state, ArrayToken):
        return state.physical
    physical = getattr(state, "physical", None)
    if isinstance(physical, PhysicalType):
        return physical
    if isinstance(physical, str):
        try:
            return PhysicalType(physical)
        except ValueError:
            return None
    return None


__all__ = ["ArrayToken", "assert_token"]
