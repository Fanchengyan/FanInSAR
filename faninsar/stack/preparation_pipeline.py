"""Bounded preparation/coregistration overlap entry point (PROPOSAL-0088).

The pipeline is deliberately thin: it delegates to the same explicit
``prepare_stack_data`` pass and adds one controlling-thread observer.  Each
qualified date is installed as an incremental durable snapshot while the
remaining acquisition futures keep running, and the caller's ``on_ready``
callback may run bounded local science for that date.  Final network inversion
and final corrected products stay with the caller; this module never starts a
second acquisition loop or a generic scheduler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from faninsar.stack.preparation import prepare_stack_data

if TYPE_CHECKING:
    from collections.abc import Callable

    from faninsar.remote.records import RemoteResourceBudget
    from faninsar.stack.preparation import PreparationReport
    from faninsar.stack.session import Stack

__all__ = ["run_preparation_pipeline"]


def run_preparation_pipeline(
    stack: Stack,
    *,
    on_error: Literal["warn", "raise"] = "warn",
    offline: bool = False,
    budget: RemoteResourceBudget | None = None,
    on_ready: Callable[[Stack, tuple[str, ...]], None] | None = None,
) -> PreparationReport:
    """Prepare inputs and publish each qualified date to local science.

    Parameters
    ----------
    stack : Stack
        Configured Stack whose request is the durable full target.
    on_error : {"warn", "raise"}, default="warn"
        Preparation failure policy for unavailable independent targets.
    offline : bool, default=False
        Require local dependencies without remote acquisition.
    budget : RemoteResourceBudget, optional
        Remote execution policy, independent of catalog item selection.
    on_ready : callable, optional
        Called on the controlling acquisition thread as ``on_ready(stack,
        dates)`` after that event's SLC+EOF+AUX qualification and incremental
        catalog install.  ``dates`` are the newly usable dates, including
        secondaries held until the reference arrived.  The Stack already
        exposes the cumulative catalog, deterministic reference, frozen
        networks, grid and DEM.  The callback must not call ``Stack.open`` or
        publish intent (the shared writer lock is held) and must not mutate
        ``stack.request``.  It may run bounded local coregistration or NESD
        measurement; final inversion remains the caller's decision after the
        planned measurement attempt is terminal.

    Returns
    -------
    PreparationReport
        Original requested, ready and failed target membership.

    """
    return prepare_stack_data(
        stack,
        on_error=on_error,
        offline=offline,
        budget=budget,
        on_ready=on_ready,
    )
