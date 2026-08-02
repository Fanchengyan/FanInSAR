"""Frame façade — loads geocoded stacks and produces InterferogramStack."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from faninsar.core import Pairs
    from faninsar.processing.contracts.ifg import InterferogramStack


def _import_frame_class() -> type:
    """Late-import datasets.frame.Frame to keep core free of heavy deps at import."""
    from faninsar.datasets.frame.frame import Frame

    return Frame


class Frame:
    """Proxy that resolves to :class:`faninsar.datasets.frame.Frame`.

    Greenfield home is ``core.frame``; implementation still lives under
    ``datasets.frame`` until Phase 5 IO rehome completes. ``to_ifg_stack`` is
    attached there and re-exported for the public surface.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """Construct the concrete Frame implementation."""
        concrete = _import_frame_class()
        return concrete(*args, **kwargs)


def frame_to_ifg_stack(
    frame: Any,
    *,
    pairs: Pairs | None = None,
) -> InterferogramStack:
    """Build an :class:`InterferogramStack` from a Frame-like object.

    Parameters
    ----------
    frame : Frame
        Frame holding unwrapped interferograms.
    pairs : Pairs, optional
        Override pair list; defaults to ``frame.pairs``.

    """
    if hasattr(frame, "to_ifg_stack"):
        return frame.to_ifg_stack(pairs=pairs)
    raise TypeError(f"{type(frame).__name__} does not implement to_ifg_stack()")


__all__ = ["Frame", "frame_to_ifg_stack"]
