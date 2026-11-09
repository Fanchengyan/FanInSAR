"""Shared automatic-selection and inheritance policies.

``Parameter`` names the two policies callers may pass when an interface
accepts automatic selection or parent/session inheritance. Existing APIs
do not gain these options until they document them.

Notes
-----
``None`` remains an API-specific explicit empty or disabled value and is
never treated as ``AUTO`` or ``INHERIT``. A documented default of
``"inherit"`` means an omitted argument and an explicit inherit request
behave the same; this module does not provide a missing marker.

"""

from __future__ import annotations

from enum import Enum


class Parameter(Enum):
    """Policies for automatic selection and inherited arguments.

    Members are distinct from strings globally. A supporting interface may
    document string aliases such as ``"auto"`` and ``"inherit"`` and treat
    those aliases identically to the corresponding member. Callers are not
    required to import this enum for ordinary calls.

    Attributes
    ----------
    AUTO : Parameter
        Request automatic selection from available information. This does
        not mean inherit.
    INHERIT : Parameter
        Request the parent or session configuration. If the API cannot
        access a parent, it must document or reject unsupported
        inheritance rather than silently guess.

    Notes
    -----
    Compare members with ``is``. Where an interface documents aliases, use
    a guarded string check so array-valued inputs are not compared with
    ``==``:

    ``value is Parameter.AUTO or (isinstance(value, str) and value == "auto")``.

    Do not coerce arbitrary strings with ``Parameter(value)``, dispatch on
    truthiness, or assume every member is accepted by every API. Resolve
    policies to concrete values before calculation or persistence. This
    type is not a universal argument convention; only documented
    interfaces accept it. ``None`` is never automatically ``AUTO`` or
    ``INHERIT``. A default of ``"inherit"`` means omission and explicit
    inherit are the same.

    Examples
    --------
    The following standalone function is illustrative only. No current
    Stack, plot, or device API is claimed to accept these options.

    >>> from typing import Literal
    >>> from faninsar.core import Parameter
    >>> Policy = Literal["auto", "inherit", Parameter.AUTO, Parameter.INHERIT]
    >>> def choose_window(
    ...     window: int | None | Policy = "inherit",
    ...     *,
    ...     parent_window: int = 5,
    ... ) -> int | None:
    ...     if window is Parameter.AUTO or (
    ...         isinstance(window, str) and window == "auto"
    ...     ):
    ...         return 11
    ...     if window is Parameter.INHERIT or (
    ...         isinstance(window, str) and window == "inherit"
    ...     ):
    ...         return parent_window
    ...     return window
    >>> choose_window()
    5
    >>> choose_window("inherit")
    5
    >>> choose_window(Parameter.INHERIT)
    5
    >>> choose_window("auto")
    11
    >>> choose_window(Parameter.AUTO)
    11
    >>> choose_window(7)
    7
    >>> choose_window(None) is None
    True

    """

    AUTO = "auto"
    INHERIT = "inherit"

    def __repr__(self) -> str:
        """Return ``Parameter.AUTO`` or ``Parameter.INHERIT``."""
        return f"{type(self).__name__}.{self.name}"


__all__ = ["Parameter"]
