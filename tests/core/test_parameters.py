"""Public contract tests for ``faninsar.core.Parameter``."""

from __future__ import annotations

import copy
import doctest
import inspect
import pickle

import faninsar as fis
import faninsar.core as core
import numpy as np
import pytest
from faninsar.core import Parameter, parameters as parameters_module


def test_parameter_has_exactly_auto_and_inherit() -> None:
    """Expose exactly AUTO and INHERIT as the shared policy members."""
    assert {member.name: member.value for member in Parameter} == {
        "AUTO": "auto",
        "INHERIT": "inherit",
    }
    assert list(Parameter) == [Parameter.AUTO, Parameter.INHERIT]
    assert not hasattr(Parameter, "MISSING")


def test_parameter_is_exported_from_core_not_root() -> None:
    """Own Parameter on the core surface and keep the root whitelist unchanged."""
    assert "Parameter" in core.__all__
    assert core.Parameter is Parameter
    assert "Parameter" not in fis.__all__
    assert not hasattr(fis, "Parameter")


def test_parameter_identity_is_distinct_from_strings_and_none() -> None:
    """Keep enum identity distinct from aliases, None, and the other member."""
    auto_alias = "auto"
    inherit_alias = "inherit"
    assert Parameter.AUTO is Parameter.AUTO
    assert Parameter.INHERIT is Parameter.INHERIT
    assert Parameter.AUTO is not Parameter.INHERIT
    assert Parameter.AUTO is not None
    assert Parameter.INHERIT is not None
    assert Parameter.AUTO is not auto_alias
    assert Parameter.INHERIT is not inherit_alias
    assert Parameter.AUTO != auto_alias
    assert Parameter.INHERIT != inherit_alias


def test_parameter_repr_str_and_signature_are_readable() -> None:
    """Show Parameter.AUTO / Parameter.INHERIT in repr, str, and signatures."""

    def demo(value=Parameter.AUTO, other=Parameter.INHERIT):
        return value, other

    assert repr(Parameter.AUTO) == "Parameter.AUTO"
    assert repr(Parameter.INHERIT) == "Parameter.INHERIT"
    assert str(Parameter.AUTO) == "Parameter.AUTO"
    assert str(Parameter.INHERIT) == "Parameter.INHERIT"
    assert str(inspect.signature(demo)) == (
        "(value=Parameter.AUTO, other=Parameter.INHERIT)"
    )


@pytest.mark.parametrize("protocol", list(range(pickle.HIGHEST_PROTOCOL + 1)))
@pytest.mark.parametrize("member", list(Parameter), ids=lambda member: member.name)
def test_parameter_copy_deepcopy_and_pickle_preserve_identity(
    member: Parameter,
    protocol: int,
) -> None:
    """Preserve member identity across copy, deepcopy, and every pickle protocol."""
    assert copy.copy(member) is member
    assert copy.deepcopy(member) is member
    restored = pickle.loads(pickle.dumps(member, protocol=protocol))
    assert restored is member


def test_documented_string_aliases_match_members_and_omit_equals_inherit() -> None:
    """Treat documented aliases like members; omitted inherit matches explicit inherit."""

    def choose_window(
        window: int | None | str | Parameter = "inherit",
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

    assert choose_window() == 5
    assert choose_window("inherit") == 5
    assert choose_window(Parameter.INHERIT) == 5
    assert choose_window("auto") == 11
    assert choose_window(Parameter.AUTO) == 11
    assert choose_window(7) == 7
    assert choose_window(None) is None


def test_guarded_string_alias_check_rejects_array_values() -> None:
    """Avoid array-valued equality by guarding string alias checks."""
    value = np.array(["auto", "inherit"])
    unguarded = value == "auto"
    assert isinstance(unguarded, np.ndarray)
    assert not (
        value is Parameter.AUTO or (isinstance(value, str) and value == "auto")
    )


def test_parameter_docstring_examples() -> None:
    """Execute the standalone Parameter docstring examples."""
    failures, tests = doctest.testmod(
        parameters_module,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )
    assert tests > 0
    assert failures == 0
