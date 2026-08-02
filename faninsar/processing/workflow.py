"""Workflow composition with optional PhysicalType lattice check."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from faninsar.core.physical import PhysicalType
from faninsar.processing.contracts.stage import StageNode
from faninsar.processing.errors import StageError

CompositionOp = Literal["seq", "par", "map_over", "reduce"]

# Default pair path: logical stage names mapped to PhysicalType transitions.
# Actual callables are resolved lazily from processing.stages / production.
DEFAULT_STAGE_LATTICE: tuple[tuple[str, PhysicalType | None, PhysicalType], ...] = (
    ("stage_read", None, PhysicalType.SLC_RAW),
    ("stage_deramp", PhysicalType.SLC_RAW, PhysicalType.SLC_DERAMPED),
    ("stage_coreg", PhysicalType.SLC_DERAMPED, PhysicalType.SLC_COREG),
    ("stage_ifg", PhysicalType.SLC_COREG, PhysicalType.IFG_COMPLEX),
    ("stage_flatten", PhysicalType.IFG_COMPLEX, PhysicalType.IFG_FLATTENED),
    ("stage_unwrap", PhysicalType.IFG_FLATTENED, PhysicalType.PHASE_UNWRAPPED),
    ("stage_geocode", PhysicalType.PHASE_UNWRAPPED, PhysicalType.PHASE_UNWRAPPED),
)


def _default_pair_stages() -> tuple[Callable[..., Any], ...]:
    """Import production stage callables for DEFAULT_PAIR_STAGES."""
    from faninsar.processing import stages as st

    return (
        st.stage_read,
        st.stage_deramp,
        st.stage_coreg,
        st.stage_ifg,
        st.stage_flatten,
        st.stage_unwrap,
        st.stage_geocode,
    )


# Populated on first access to avoid circular imports at module load.
DEFAULT_PAIR_STAGES: tuple[Callable[..., Any], ...] | None = None


def get_default_pair_stages() -> tuple[Callable[..., Any], ...]:
    """Return the default linear pair stage tuple."""
    global DEFAULT_PAIR_STAGES  # noqa: PLW0603
    if DEFAULT_PAIR_STAGES is None:
        DEFAULT_PAIR_STAGES = _default_pair_stages()
    return DEFAULT_PAIR_STAGES


@dataclass
class Workflow:
    """Opt-in workflow that can ``build()`` lattice-check stage composition."""

    stages: tuple[Callable[..., Any] | StageNode, ...]
    nodes: tuple[StageNode, ...] = field(default_factory=tuple)
    _built: bool = False

    @classmethod
    def from_stages(
        cls,
        stages: Sequence[Callable[..., Any] | StageNode],
    ) -> Workflow:
        """Create a Workflow from a sequence of callables or StageNodes."""
        return cls(stages=tuple(stages))

    def build(self) -> Workflow:
        """Run lattice ``check()`` and mark the workflow as built."""
        self.check()
        self._built = True
        return self

    def check(self) -> None:
        """Validate sequential PhysicalType compatibility when annotated."""
        nodes = self._resolve_nodes()
        prev_out: PhysicalType | None = None
        for node in nodes:
            if node.op not in {"seq", "par", "map_over", "reduce"}:
                raise StageError(
                    stage=node.name,
                    hint=f"unknown composition op {node.op!r}",
                )
            if node.op == "seq" and prev_out is not None and node.input_type is not None:
                if node.input_type != prev_out:
                    raise StageError(
                        stage=node.name,
                        hint=(
                            f"lattice mismatch: previous output "
                            f"{prev_out.value!r} != input {node.input_type.value!r}"
                        ),
                    )
            if node.output_type is not None:
                prev_out = node.output_type

    def run(self, initial: Any, *, backend: Any | None = None) -> Any:
        """Execute stages eagerly in order."""
        state = initial
        for stage in self.stages:
            fn = stage.fn if isinstance(stage, StageNode) else stage
            if backend is not None:
                try:
                    state = fn(state, backend=backend)
                    continue
                except TypeError:
                    pass
            state = fn(state)
        return state

    def _resolve_nodes(self) -> tuple[StageNode, ...]:
        if self.nodes:
            return self.nodes
        resolved: list[StageNode] = []
        for stage in self.stages:
            if isinstance(stage, StageNode):
                resolved.append(stage)
            else:
                name = getattr(stage, "__name__", "stage")
                in_t = getattr(stage, "input_type", None)
                out_t = getattr(stage, "output_type", None)
                resolved.append(
                    StageNode(
                        fn=stage,
                        name=name,
                        input_type=in_t,
                        output_type=out_t,
                    )
                )
        return tuple(resolved)


__all__ = [
    "DEFAULT_PAIR_STAGES",
    "DEFAULT_STAGE_LATTICE",
    "CompositionOp",
    "Workflow",
    "get_default_pair_stages",
]
