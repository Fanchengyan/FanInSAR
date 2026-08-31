# ruff: noqa: TRY003, EM102
"""Canonical vertical-datum graph for DEM materialization."""

from __future__ import annotations

from typing import Literal

VerticalDatum = Literal["ellipsoidal", "egm96", "egm2008"]


def validate_datum(value: str) -> VerticalDatum:
    """Validate the closed target datum vocabulary."""
    if value not in {"ellipsoidal", "egm96", "egm2008"}:
        raise ValueError(f"unsupported vertical datum: {value}")
    return value  # type: ignore[return-value]


def conversion_models(source: VerticalDatum, target: VerticalDatum) -> tuple[str, ...]:
    """Return required geoid models in canonical conversion order.

    Same-datum conversion is empty.  A geoid-to-geoid conversion traverses
    ellipsoidal height and therefore requires both independent models.
    """
    validate_datum(source)
    validate_datum(target)
    if source == target:
        return ()
    if source == "ellipsoidal":
        return (target,)
    if target == "ellipsoidal":
        return (source,)
    return (source, target)


def requires_fetch(source: VerticalDatum, target: VerticalDatum) -> bool:
    """Return whether conversion needs at least one geoid model."""
    return bool(conversion_models(source, target))


def fetch_required(
    source: VerticalDatum,
    target: VerticalDatum,
    fetch: callable,
) -> tuple[object, ...]:
    """Fetch required models in graph order and return their loaded handles."""
    return tuple(fetch(model) for model in conversion_models(source, target))


__all__ = [
    "VerticalDatum",
    "conversion_models",
    "fetch_required",
    "requires_fetch",
    "validate_datum",
]
