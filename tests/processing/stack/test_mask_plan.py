"""Public behavior tests for PROPOSAL-0040 ``MaskPlan`` normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from affine import Affine

from faninsar.processing.masking import GridSpec, Mask
from faninsar.processing.stack.mask_plan import MaskDefinition, MaskPlan

if TYPE_CHECKING:
    from pathlib import Path


def _mask() -> Mask:
    """Build a deterministic concrete mask for plan tests."""
    return Mask.from_raster(
        np.array([[1]], dtype=np.uint8),
        grid=GridSpec("EPSG:4326", Affine(1, 0, 0, 0, -1, 1), shape=(1, 1)),
    )


def test_python_plan_accepts_concrete_mask_without_mutating_input() -> None:
    """Concrete masks are retained exactly and plan containers are immutable."""
    concrete = _mask()
    config = {
        "masks": {"observed": concrete},
        "stages": {"roi": ["observed"]},
    }

    plan = MaskPlan.from_python(config)

    assert plan.masks["observed"] is concrete
    assert plan.for_stage("roi") == (concrete,)
    assert plan.identity == MaskPlan.from_python(config).identity
    with pytest.raises(TypeError):
        plan.masks["new"] = concrete  # type: ignore[index]


def test_python_plan_keeps_recipe_and_concrete_forms_distinct() -> None:
    """Recipe and concrete registry values both normalize deterministically."""
    recipe = MaskDefinition("observed", "raster", path="mask.tif")
    recipe_plan = MaskPlan({"observed": recipe}, {"roi": ["observed"]})
    concrete_plan = MaskPlan({"observed": _mask()}, {"roi": ["observed"]})

    assert recipe_plan.for_stage("roi") == (recipe,)
    assert recipe_plan.identity != concrete_plan.identity


def test_yaml_mask_registry_is_mapping_only(tmp_path: Path) -> None:
    """YAML does not accept the Python-only list shorthand."""
    path = tmp_path / "mask-plan.yaml"
    path.write_text(
        "masks:\n"
        "  - name: observed\n"
        "    kind: raster\n"
        "    path: mask.tif\n"
        "stages:\n"
        "  roi: [observed]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mapping"):
        MaskPlan.from_yaml(path)
