"""Sentinel-1 SAFE adapter for the mission-neutral Stack (PROPOSAL-0037)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_pair_configuration
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
    from faninsar.processing.pipeline.production import produce_interferogram_pair

    # Preserve the scalar callback shape for ordinary one-frame acquisitions;
    # frame stacks remain tuples and are consumed by the production adapter.
    if isinstance(reference_path, tuple) and len(reference_path) == 1:
        reference_path = reference_path[0]
    if isinstance(secondary_path, tuple) and len(secondary_path) == 1:
        secondary_path = secondary_path[0]

    return produce_interferogram_pair(
        reference_path,
        secondary_path,
        output_dir=output_dir,
        **options,
    )


class S1Stack(Stack):
    """Concrete Stack entry point for Sentinel-1 SAFE source paths.

    The adapter owns raw SAFE initialization and installs the provider used by
    every shared Stack stage.  The base session never imports a pair runner or
    otherwise guesses how a source path should be opened.
    """

    @classmethod
    def from_safes(
        cls,
        paths: Sequence[str | Path],
        *,
        reference: object | None = None,
        **kwargs: Any,
    ) -> Self:
        """Build an S1 Stack and optionally select its reference acquisition.

        Parameters
        ----------
        paths : sequence of path-like
            Sentinel-1 SAFE directories or archives.
        reference : date-like, optional
            Reference acquisition.  Defaults to the existing Stack default.
        **kwargs : Any
            Remaining :meth:`Stack._from_safes` options.

        Returns
        -------
        S1Stack
            A Stack whose scene production is explicitly provider-owned.

        """
        if "master" in kwargs:
            reject_pair_configuration(
                "S1Stack.from_safes no longer accepts 'master'; use 'reference'"
            )
        if reference is not None:
            kwargs["reference"] = reference

        # The classmethod remains bound to this concrete adapter through
        # ``super()``, so the raw constructor creates an S1Stack directly
        # without duplicating Stack's large configuration signature.
        stack = super()._from_safes(paths, **kwargs)
        stack.scene_provider = StackSceneProvider(
            name="Sentinel-1 SAFE",
            produce_pair=_produce_s1_pair,
        )
        return stack  # type: ignore[return-value]


__all__ = ["S1Stack"]
