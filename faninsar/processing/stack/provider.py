"""Mission-neutral Stack scene-production providers (PROPOSAL-0035).

The Stack orchestration layer owns lifecycle, persistence, and downstream
products.  Mission adapters own the way a pair of source scenes is converted
into the normalized scene units consumed by that lifecycle.  This module keeps
that seam deliberately small: a provider receives two source paths and the
already-normalized production options, and returns the provider's pair state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

logger = setup_logger(__name__)


class SceneProductionCallback(Protocol):
    """Callable contract for one provider-owned pair scene production."""

    def __call__(
        self,
        reference_path: Path,
        secondary_path: Path,
        *,
        output_dir: Path,
        options: Mapping[str, Any],
    ) -> Any:
        """Produce normalized scene artifacts for one acquisition pair."""


class StackProviderError(RuntimeError):
    """Base class for mission-provider dispatch failures."""


class UnsupportedStackCapabilityError(StackProviderError, NotImplementedError):
    """Raised when a provider has not admitted a requested Stack capability.

    Parameters
    ----------
    mission : str
        Mission/product family that owns the source paths.
    capability : str
        Named Stack capability that was requested.
    reason : str, optional
        Additional provider-specific explanation.

    """

    def __init__(
        self,
        mission: str,
        capability: str,
        reason: str | None = None,
    ) -> None:
        """Build a typed, fail-closed capability error."""
        self.mission = mission
        self.capability = capability
        self.reason = reason or "provider has not admitted this capability"
        message = (
            f"{mission} Stack capability {capability!r} is unsupported: "
            f"{self.reason}"
        )
        logger.error(message)
        super().__init__(message)


@dataclass(frozen=True)
class StackSceneProvider:
    """Small provider descriptor used by Stack scene production.

    Parameters
    ----------
    name : str
        Provider or mission name used in diagnostics.
    produce_pair : SceneProductionCallback, optional
        Normalized pair-scene callback.  If omitted, calls fail closed with
        :class:`UnsupportedStackCapabilityError`.
    unsupported_capability : str, default="scene-production"
        Capability name used by the fail-closed path.
    unsupported_reason : str, optional
        Provider-specific reason included in the typed error.

    """

    name: str
    produce_pair: SceneProductionCallback | None = None
    unsupported_capability: str = "scene-production"
    unsupported_reason: str | None = None

    def __call__(
        self,
        reference_path: Path,
        secondary_path: Path,
        *,
        output_dir: Path,
        options: Mapping[str, Any],
    ) -> Any:
        """Produce a normalized pair or reject the capability explicitly."""
        if self.produce_pair is None:
            raise UnsupportedStackCapabilityError(
                self.name,
                self.unsupported_capability,
                self.unsupported_reason,
            )
        return self.produce_pair(
            reference_path,
            secondary_path,
            output_dir=output_dir,
            options=options,
        )


def unavailable_scene_provider(
    mission: str,
    *,
    capability: str = "scene-production",
    reason: str | None = None,
) -> StackSceneProvider:
    """Return a provider descriptor that fails closed for one capability.

    Parameters
    ----------
    mission : str
        Mission/product family whose provider is incomplete.
    capability : str, optional
        Named capability rejected by the descriptor.
    reason : str, optional
        Explanation included in the raised error.

    Returns
    -------
    StackSceneProvider
        A descriptor with no callback and an explicit fail-closed contract.

    """
    return StackSceneProvider(
        name=mission,
        unsupported_capability=capability,
        unsupported_reason=reason,
    )


def unsupported_stack_capability(
    mission: str,
    capability: str,
    reason: str | None = None,
) -> UnsupportedStackCapabilityError:
    """Build a named capability error for a provider's fail-closed path.

    Parameters
    ----------
    mission : str
        Mission/product family that owns the source paths.
    capability : str
        Named Stack capability that was requested.
    reason : str, optional
        Additional provider-specific explanation.

    Returns
    -------
    UnsupportedStackCapabilityError
        Error ready to raise at the provider boundary.

    """
    return UnsupportedStackCapabilityError(mission, capability, reason)


__all__ = [
    "SceneProductionCallback",
    "StackProviderError",
    "StackSceneProvider",
    "UnsupportedStackCapabilityError",
    "unavailable_scene_provider",
    "unsupported_stack_capability",
]
