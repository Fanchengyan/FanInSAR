"""Stack configuration ownership and retired runner closure tests."""

from __future__ import annotations

import pytest

from faninsar.stack.config import _load_config, _resolve_backend


def test_retired_stack_config_owner_is_absent() -> None:
    """The removed module cannot be imported as a third execution seam."""
    import importlib.util

    assert importlib.util.find_spec("faninsar._stack_config") is None


def test_retired_aggregate_owners_are_absent() -> None:
    """Removed aggregate modules do not remain as importable compatibility paths."""
    import importlib.util

    retired = (
        "faninsar._public",
        "faninsar.provenance",
        "faninsar.processing.provenance",
        "faninsar.processing.release",
        "faninsar.validation",
    )
    assert all(importlib.util.find_spec(name) is None for name in retired)


def test_stack_config_keeps_loader_and_backend_helpers() -> None:
    """Configuration parsing and backend resolution have one canonical owner."""
    assert _load_config({"paths": ["a", "b"]}) == {"paths": ["a", "b"]}
    assert _resolve_backend("numpy").name == "numpy"


def test_stack_config_does_not_accept_legacy_mask_options() -> None:
    """Legacy mask spellings fail at the canonical configuration boundary."""
    with pytest.raises(ValueError, match="legacy mask"):
        _load_config({"water_mask": True})


def test_processing_pipeline_hides_removed_pair_entry_points() -> None:
    """Removed execution callables are absent from the public pipeline."""
    from faninsar.processing import stages

    for name in (
        "run_pair",
        "run_pair_pipeline",
        "run_pair_workflow",
        "PairWorkflowState",
        "run_stack_pipeline",
    ):
        assert not hasattr(stages, name)
        assert name not in getattr(stages, "__all__", ())
