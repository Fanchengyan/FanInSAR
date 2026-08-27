"""Sentinel-1 SAFE adapter for the mission-neutral Stack (PROPOSAL-0037)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from faninsar.logging import setup_logger
from faninsar.processing.stack.provider import StackSceneProvider
from faninsar.processing.stack.session import Stack

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = setup_logger(__name__)


def _produce_s1_pair(
    reference_path: Path | tuple[Path, ...],
    secondary_path: Path | tuple[Path, ...],
    *,
    output_dir: Path,
    options: dict[str, Any],
) -> Any:
    """Dispatch one admitted SAFE pair through the S1 production adapter."""
    from faninsar.processing.pipeline.production import run_pair

    # Preserve the scalar callback shape for ordinary one-frame acquisitions;
    # frame stacks remain tuples and are consumed by the production adapter.
    if isinstance(reference_path, tuple) and len(reference_path) == 1:
        reference_path = reference_path[0]
    if isinstance(secondary_path, tuple) and len(secondary_path) == 1:
        secondary_path = secondary_path[0]

    return run_pair(
        reference_path,
        secondary_path,
        output_dir=output_dir,
        **options,
    )


class S1Stack(Stack):
    """Concrete Stack entry point for Sentinel-1 SAFE source paths.

    The adapter owns raw SAFE initialization and installs the provider used by
    every shared Stack stage.  The base session never imports ``run_pair`` or
    otherwise guesses how a source path should be opened.
    """

    @classmethod
    def from_safes(
        cls,
        paths: Sequence[str | Path],
        *,
        reference: object | None = None,
        master: object | None = None,
        **kwargs: Any,
    ) -> Self:
        """Build an S1 Stack and optionally select its reference acquisition.

        Parameters
        ----------
        paths : sequence of path-like
            Sentinel-1 SAFE directories or archives.
        reference : date-like, optional
            Reference acquisition.  Defaults to the existing Stack default.
        master : date-like, optional
            Deprecated alias for ``reference``.
        **kwargs : Any
            Remaining :meth:`Stack._from_safes` options.

        Returns
        -------
        S1Stack
            A Stack whose scene production is explicitly provider-owned.

        """
        if (
            reference is not None
            and master is not None
            and str(reference)[:8].replace("-", "")
            != str(master)[:8].replace("-", "")
        ):
            message = "reference and master specify different dates"
            logger.error(message)
            raise ValueError(message)
        selected_reference = reference if reference is not None else master
        if selected_reference is not None:
            kwargs["master"] = selected_reference

        stack = Stack._from_safes(paths, **kwargs)
        # ``Stack`` is a regular dataclass without slots; changing the concrete
        # class preserves its validated state and avoids duplicating the large
        # configuration signature in this thin adapter.
        stack.__class__ = cls
        stack.scene_provider = StackSceneProvider(
            name="Sentinel-1 SAFE",
            produce_pair=_produce_s1_pair,
        )
        return stack  # type: ignore[return-value]


__all__ = ["S1Stack"]
