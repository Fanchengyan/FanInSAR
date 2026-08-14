"""Platform-specific OpenMP compiler and linker flag providers."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PlatformName = Literal["linux", "macos", "windows", "unsupported"]


@dataclass(frozen=True, slots=True)
class OpenMPFlags:
    """Compiler/linker configuration for one OpenMP runtime.

    Parameters
    ----------
    compile_flags : tuple[str, ...]
        Flags passed while compiling each translation unit.
    link_flags : tuple[str, ...]
        Flags passed only at the shared-library link step.
    include_dirs : tuple[pathlib.Path, ...]
        Runtime header search paths, if the platform requires them.
    library_dirs : tuple[pathlib.Path, ...]
        Runtime library search paths, if the platform requires them.
    runtime_name : str
        Expected loaded runtime identity, such as ``libgomp`` or ``libomp``.
    runtime_path : pathlib.Path or None
        Expected runtime path when the provider can identify one.
    supported : bool
        Whether this configuration can produce a qualified OpenMP extension.
    reason : str
        Stable diagnostic for unsupported configurations.

    """

    compile_flags: tuple[str, ...]
    link_flags: tuple[str, ...]
    include_dirs: tuple[Path, ...]
    library_dirs: tuple[Path, ...]
    runtime_name: str
    runtime_path: Path | None
    supported: bool
    reason: str = ""


class OpenMPFlagProvider:
    """Protocol-like base class for platform OpenMP providers."""

    @staticmethod
    def for_platform(
        platform: str | None = None,
        *,
        compiler: str | None = None,
        libomp_root: str | Path | None = None,
    ) -> OpenMPFlags:
        """Return flags for ``platform`` without invoking a compiler.

        Parameters
        ----------
        platform : str, optional
            ``linux``, ``darwin``/``macos``, or ``windows``.  Defaults to the
            current interpreter platform.
        compiler : str, optional
            Compiler executable name.  GCC and Clang receive their normal
            ``-fopenmp`` compile/link flags on Linux.
        libomp_root : path-like, optional
            Explicit LLVM ``libomp`` prefix for macOS.  This is useful for
            hermetic builds and tests.

        Returns
        -------
        OpenMPFlags
            A supported configuration or an explicit unsupported result.

        """
        return _flags_for_platform(
            platform,
            compiler=compiler,
            libomp_root=libomp_root,
        )


@dataclass(frozen=True, slots=True)
class OpenMPProviderResult:
    """Stable provider result that preserves unsupported diagnostics."""

    platform: PlatformName
    flags: OpenMPFlags

    @property
    def supported(self) -> bool:
        """Whether this provider can support a qualified build."""
        return self.flags.supported

    @property
    def reason(self) -> str:
        """Return the provider's unsupported explanation, if any."""
        return self.flags.reason


def openmp_provider_for_platform(
    platform: str | None = None,
    *,
    compiler: str | None = None,
    libomp_root: str | Path | None = None,
) -> OpenMPProviderResult:
    """Select an OpenMP provider and preserve unsupported platforms explicitly.

    Parameters
    ----------
    platform : str, optional
        Platform name.  The host platform is used when omitted.
    compiler : str, optional
        Compiler executable name used to choose the runtime family.
    libomp_root : path-like, optional
        Explicit macOS LLVM ``libomp`` installation prefix.

    Returns
    -------
    OpenMPProviderResult
        Provider flags and a deterministic unsupported status when necessary.

    """
    normalized = _normalize_platform(platform)
    return OpenMPProviderResult(
        platform=normalized,
        flags=_flags_for_platform(
            normalized,
            compiler=compiler,
            libomp_root=libomp_root,
        ),
    )


def _normalize_platform(platform: str | None) -> PlatformName:
    value = (platform or sys.platform).lower()
    if value.startswith("linux"):
        return "linux"
    if value in {"darwin", "macos", "osx"}:
        return "macos"
    if value.startswith("win"):
        return "windows"
    return "unsupported"


def _unsupported(reason: str) -> OpenMPFlags:
    return OpenMPFlags((), (), (), (), "", None, False, reason)


def _flags_for_platform(
    platform: str | None,
    *,
    compiler: str | None,
    libomp_root: str | Path | None,
) -> OpenMPFlags:
    normalized = _normalize_platform(platform)
    compiler_name = Path(compiler or os.environ.get("CXX", "c++")).name.lower()
    if normalized == "linux":
        if "clang" in compiler_name:
            return OpenMPFlags(
                ("-fopenmp",),
                ("-fopenmp", "-lomp"),
                (),
                (),
                "libomp",
                None,
                True,
            )
        return OpenMPFlags(
            ("-fopenmp",),
            ("-fopenmp",),
            (),
            (),
            "libgomp",
            None,
            True,
        )
    if normalized == "macos":
        root_value = libomp_root or os.environ.get("FANINSAR_LIBOMP_ROOT")
        prefix_values = (
            os.environ.get("CONDA_PREFIX"),
            os.environ.get("PIXI_PROJECT_ROOT"),
            sys.prefix,
        )
        prefixes = tuple(Path(value) for value in prefix_values if value)
        if root_value:
            # An explicit root is an authority boundary: do not silently use
            # another runtime when the requested installation is unavailable.
            candidates = (Path(root_value),)
        else:
            candidates = (
                Path("/opt/homebrew/opt/libomp"),
                Path("/usr/local/opt/libomp"),
                *prefixes,
                *(prefix / "opt" / "libomp" for prefix in prefixes),
                *(prefix / "libomp" for prefix in prefixes),
            )
        root = next(
            (
                candidate
                for candidate in candidates
                if candidate is not None
                and (candidate / "include").is_dir()
                and (candidate / "lib").is_dir()
            ),
            None,
        )
        if root is None:
            return _unsupported(
                "macOS LLVM libomp is unavailable; install libomp or pass libomp_root"
            )
        include = root / "include"
        library = root / "lib"
        return OpenMPFlags(
            ("-fopenmp", f"-I{include}"),
            (f"-L{library}", "-lomp", f"-Wl,-rpath,{library}"),
            (include,),
            (library,),
            "libomp",
            library / "libomp.dylib",
            True,
        )
    if normalized == "windows":
        return _unsupported(
            "Windows OpenMP provider is not part of the first native-v2 slice"
        )
    return _unsupported(f"unsupported platform: {platform or sys.platform}")
