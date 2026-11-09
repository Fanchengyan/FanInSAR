==========
Parameters
==========

Shared policies for automatic selection and inherited arguments.

Import ``Parameter`` from :mod:`faninsar.core`. It is not part of the
root ``faninsar`` whitelist. Existing APIs do not accept these options
until they document them.

``AUTO`` requests automatic selection from available information. It does
not mean inherit. ``INHERIT`` requests the parent or session
configuration; if an API cannot access a parent, it must document or
reject unsupported inheritance rather than silently guess. ``None`` is an
API-specific explicit empty or disabled value and is never automatically
``AUTO`` or ``INHERIT``. A documented default of ``"inherit"`` means an
omitted argument and an explicit inherit request behave the same.

String aliases such as ``"auto"`` and ``"inherit"`` are a per-interface
documented policy, not a global string equality. Compare members with
``is``. Where aliases are documented, use a guarded string check:

``value is Parameter.AUTO or (isinstance(value, str) and value == "auto")``.

.. autoclass:: faninsar.core.Parameter
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

Example
=======

The following standalone function is illustrative only. No current
Stack, plot, or device API is claimed to accept these options.

.. code-block:: python

   from typing import Literal

   from faninsar.core import Parameter


   def choose_window(
       window: int | None | Literal[
           "auto", "inherit", Parameter.AUTO, Parameter.INHERIT
       ] = "inherit",
       *,
       parent_window: int = 5,
   ) -> int | None:
       if window is Parameter.AUTO or (
           isinstance(window, str) and window == "auto"
       ):
           return 11
       if window is Parameter.INHERIT or (
           isinstance(window, str) and window == "inherit"
       ):
           return parent_window
       return window


   choose_window()  # 5, omitted default inherit
   choose_window("inherit")  # 5, same as omitted
   choose_window(Parameter.INHERIT)  # 5
   choose_window("auto")  # 11
   choose_window(Parameter.AUTO)  # 11
   choose_window(7)  # 7, explicit concrete value
   choose_window(None)  # None, explicit empty/disabled
