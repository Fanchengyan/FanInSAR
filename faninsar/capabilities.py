"""Report availability, versions, and licensing of FanInSAR backends."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
from importlib.util import find_spec
from typing import Final, Protocol, runtime_checkable

SNAPHU_LICENSE_CAVEAT: Final = (
    "snaphu-py bundles SNAPHU code under separate terms; portions prohibit "
    "commercial use. Review the upstream license before use."
)


@runtime_checkable
class _SnaphuVersionProvider(Protocol):
    """Typed surface used to read the bundled SNAPHU version lazily."""

    def get_snaphu_version(self) -> str:
        """Return the bundled SNAPHU version."""
        ...


@dataclass(frozen=True, slots=True)
class BackendCapability:
    """Availability and licensing information for a backend.

    Attributes
    ----------
    display_name
        Human-readable backend distribution name.
    available
        Whether the backend's import package is installed.
    install_requirement
        Package requirement that installs the backend.
    wrapper_version
        Installed snaphu-py wrapper version, if available.
    bundled_snaphu_version
        Version of SNAPHU bundled by the installed wrapper, if available.
    license_caveat
        Separate licensing notice that users must review before use.

    """

    display_name: str
    available: bool
    install_requirement: str
    wrapper_version: str | None
    bundled_snaphu_version: str | None
    license_caveat: str


def snaphu_capability() -> BackendCapability:
    """Return the current snaphu-py backend capability."""
    available = find_spec("snaphu") is not None
    wrapper_version: str | None = None
    bundled_snaphu_version: str | None = None
    if available:
        snaphu_module = import_module("snaphu")
        assert isinstance(snaphu_module, _SnaphuVersionProvider)
        wrapper_version = version("snaphu")
        bundled_snaphu_version = snaphu_module.get_snaphu_version()

    return BackendCapability(
        display_name="snaphu-py",
        available=available,
        install_requirement="faninsar",
        wrapper_version=wrapper_version,
        bundled_snaphu_version=bundled_snaphu_version,
        license_caveat=SNAPHU_LICENSE_CAVEAT,
    )


def format_backend_capabilities() -> str:
    """Format backend availability and licensing for user output."""
    capability = snaphu_capability()
    wrapper_version = capability.wrapper_version or "unavailable"
    bundled_snaphu_version = capability.bundled_snaphu_version or "unavailable"
    return "\n".join(
        (
            f"{capability.display_name} available: {capability.available}",
            f"{capability.display_name} install requirement: "
            f"{capability.install_requirement}",
            f"{capability.display_name} wrapper version: {wrapper_version}",
            f"bundled SNAPHU version: {bundled_snaphu_version}",
            f"{capability.display_name} license caveat: {capability.license_caveat}",
        )
    )
