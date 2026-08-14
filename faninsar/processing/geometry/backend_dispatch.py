# ruff: noqa: EM101, EM102, TRY003, TRY301, TRY400

"""Prepared native/compile candidate registry and deterministic dispatcher.

The module consumes the foundation candidate/result contracts directly.
Preparation and execution are separate:
``register`` receives an already prepared callable, while :meth:`Dispatcher.dispatch`
only invokes registered callables or the caller-provided eager implementation.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from numbers import Integral
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.geometry.v2 import (
    CandidateKey as FoundationCandidateKey,
)
from faninsar.processing.geometry.v2 import (
    GeometryValidationError,
    TransformResultV2,
)

logger = setup_logger(__name__)

Backend: TypeAlias = Literal["native", "compile", "eager"]
PreparedBackend: TypeAlias = Literal["native", "compile"]
DispatchMode: TypeAlias = Literal["auto", "native", "compile", "eager"]
Executor: TypeAlias = Callable[[], object]
ResultValidator: TypeAlias = Callable[[object], object]

PERFORMANCE_ELIGIBILITY_ORDER: tuple[Backend, ...] = (
    "eager",
    "compile",
    "native",
)


CandidateKeyProtocol: TypeAlias = FoundationCandidateKey
# Short aliases preserve the dispatcher vocabulary while using the foundation
# object as the sole concrete candidate-key contract.
CandidateKey: TypeAlias = FoundationCandidateKey


@runtime_checkable
class EagerExecutorProtocol(Protocol):
    """Callable boundary for same-device eager execution."""

    def __call__(self, key: CandidateKeyProtocol) -> object:
        """Execute eager geometry for the requested key."""


class DispatchError(RuntimeError):
    """Raised when a requested backend cannot execute."""


class RecoverableExecutionError(DispatchError):
    """A candidate execution or validation failure safe for automatic fallback."""


class IncorrectResultError(RecoverableExecutionError):
    """A malformed or scientifically invalid prepared result."""


class FatalExecutionError(DispatchError):
    """A failure that must never fall back to another backend."""


class CudaFailure(StrEnum):
    """Small CUDA failure taxonomy used by automatic dispatch."""

    PRE_LAUNCH_INVALID_CONFIGURATION = "pre_launch_invalid_configuration"
    PRE_LAUNCH_OOM = "pre_launch_oom"
    DEVICE_ASSERT = "device_assert"
    ILLEGAL_ACCESS = "illegal_access"
    LAUNCH_FAILURE = "launch_failure"
    CONTEXT_FAILURE = "context_failure"
    SYNCHRONIZATION_FAILURE = "synchronization_failure"
    UNKNOWN = "unknown"

    @property
    def recoverable(self) -> bool:
        """Return whether the failure can be retried after a healthy probe."""
        return self in {
            CudaFailure.PRE_LAUNCH_INVALID_CONFIGURATION,
            CudaFailure.PRE_LAUNCH_OOM,
        }


CudaFailureKind: TypeAlias = CudaFailure


class CudaExecutionError(DispatchError):
    """CUDA failure carrying phase and context-health evidence.

    Only an explicitly pre-launch invalid configuration or out-of-memory
    failure with a healthy context is recoverable.  Diagnostic text alone is
    never sufficient to authorize fallback.
    """

    def __init__(
        self,
        failure: CudaFailure,
        message: str | None = None,
        *,
        phase: Literal["pre_launch", "submitted", "synchronized"] | None = None,
        context_healthy: bool = False,
    ) -> None:
        """Create a classified CUDA execution error."""
        self.failure = CudaFailure(failure)
        self.phase = phase
        self.context_healthy = bool(context_healthy)
        super().__init__(message or self.failure.value)

    @property
    def recoverable(self) -> bool:
        """Return whether automatic fallback is safe for this failure."""
        return (
            self.failure.recoverable
            and self.phase == "pre_launch"
            and self.context_healthy
        )


def classify_cuda_failure(error: BaseException | str | CudaFailure) -> CudaFailure:
    """Classify a CUDA diagnostic without authorizing automatic fallback.

    Parameters
    ----------
    error : BaseException, str, or CudaFailure
        CUDA exception or diagnostic text.  The returned classification still
        requires explicit pre-launch phase and context-health evidence before
        it can be retried.

    Returns
    -------
    CudaFailure
        The conservative CUDA failure category.

    """
    if isinstance(error, CudaExecutionError):
        return error.failure
    if isinstance(error, CudaFailure):
        return error
    text = str(error).lower()
    patterns = (
        (
            ("invalid configuration", "too many resources"),
            CudaFailure.PRE_LAUNCH_INVALID_CONFIGURATION,
        ),
        (("out of memory",), CudaFailure.PRE_LAUNCH_OOM),
        (("device-side assert", "assert"), CudaFailure.DEVICE_ASSERT),
        (("illegal", "misaligned address"), CudaFailure.ILLEGAL_ACCESS),
        (("launch",), CudaFailure.LAUNCH_FAILURE),
        (("context", "poison"), CudaFailure.CONTEXT_FAILURE),
        (("synchron",), CudaFailure.SYNCHRONIZATION_FAILURE),
    )
    for needles, failure in patterns:
        if any(needle in text for needle in needles):
            return failure
    return CudaFailure.UNKNOWN


@dataclass(frozen=True, slots=True)
class DispatchRecord:
    """Published record for a successful dispatch decision."""

    backend: Backend
    candidate: CandidateKeyProtocol | None
    fallback: str | None = None


@dataclass(slots=True)
class _RegisteredCandidate:
    """Mutable local registry state for one active candidate."""

    key: CandidateKeyProtocol
    executor: Executor
    correctness_qualified: bool
    performance_eligible: bool
    result_validator: ResultValidator | None
    quarantined: bool = False


_SCIENTIFIC_PROFILE_FIELDS = (
    "operation",
    "device",
    "dtype",
    "shape",
    "tolerances",
    "max_iter",
    "extra_iter",
    "orbit_digest",
    "dem_digest",
    "dem_metadata_digest",
    "model_digest",
    "solver",
    "support",
    "profile",
)


def _freeze(value: object) -> object:
    """Convert common key values into a deterministic hashable structure."""
    if isinstance(value, Mapping):
        return tuple(sorted((_freeze(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_freeze(item) for item in value))
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(
            (field.name, _freeze(getattr(value, field.name))) for field in fields(value)
        )
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _scientific_profile_key(key: CandidateKeyProtocol) -> tuple[object, ...]:
    """Return the exact scientific-plus-profile slot identity."""
    if isinstance(key, FoundationCandidateKey):
        return tuple(_freeze(value) for value in key.scientific_identity)
    try:
        return tuple(_freeze(getattr(key, name)) for name in _SCIENTIFIC_PROFILE_FIELDS)
    except AttributeError as error:
        logger.error("candidate key is missing required identity field: %s", error)
        raise DispatchError(
            "candidate key is missing a required identity field"
        ) from error


def _require_prepared_key(key: CandidateKeyProtocol) -> None:
    """Reject eager or malformed keys at the prepared registry boundary."""
    backend = getattr(key, "backend", None)
    if backend not in ("native", "compile"):
        logger.error("prepared registry received unsupported backend %r", backend)
        raise DispatchError("prepared candidates must use native or compile backend")


def _candidate_budget(key: CandidateKeyProtocol) -> int:
    """Return the checked iteration budget from a foundation or legacy key."""
    solver = getattr(key, "solver", None)
    max_iter = getattr(key, "max_iter", getattr(solver, "max_iter", None))
    extra_iter = getattr(key, "extra_iter", getattr(solver, "extra_iter", None))
    if max_iter is None or extra_iter is None:
        raise IncorrectResultError("candidate key has no iteration budget")
    return int(max_iter) + int(extra_iter)


def _call_eager(
    executor: object, key: CandidateKeyProtocol, *args: object, **kwargs: object
) -> object:
    """Invoke a no-argument or key-aware eager callback without masking errors."""
    if not callable(executor):
        logger.error("no eager executor is available for device %r", key.device)
        raise DispatchError("eager executor is not callable")
    try:
        signature = inspect.signature(executor)
    except (TypeError, ValueError):
        return executor(*args, **kwargs)  # type: ignore[call-arg]
    required = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.default is parameter.empty
        and parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    if args or kwargs:
        return executor(*args, **kwargs)  # type: ignore[call-arg]
    if required:
        return executor(key)  # type: ignore[call-arg]
    return executor()  # type: ignore[call-arg]


class Dispatcher:
    """In-memory exact-key registry with native/compile/eager dispatch.

    Parameters
    ----------
    eager : callable or mapping
        Eager implementation.  A callable may accept no arguments or the
        requested key.  A mapping is keyed by the exact ``key.device`` and
        makes same-device eager selection explicit.

    Notes
    -----
    ``register`` never compiles.  Registration is the publication point for a
    previously prepared and qualified candidate.  Re-registering a backend
    atomically replaces the active candidate in its complete scientific and
    execution-profile slot; executable identity fields do not create extra
    active generations.

    """

    def __init__(
        self,
        eager: Callable[..., object] | Mapping[object, Callable[..., object]],
    ) -> None:
        """Create a dispatcher with a caller-owned eager implementation."""
        if not callable(eager) and not isinstance(eager, Mapping):
            logger.error("dispatcher requires a callable or device eager mapping")
            raise DispatchError("eager executor is required")
        self._eager = eager
        self._candidates: dict[tuple[object, ...], _RegisteredCandidate] = {}
        self.records: list[DispatchRecord] = []

    @staticmethod
    def performance_eligibility_order() -> tuple[Backend, ...]:
        """Return benchmark promotion order, independent of runtime fallback."""
        return PERFORMANCE_ELIGIBILITY_ORDER

    def register(
        self,
        key: CandidateKeyProtocol,
        executor: Executor,
        *,
        correctness_qualified: bool = True,
        performance_eligible: bool = True,
        eligible: bool | None = None,
        prepared: bool = True,
        result_validator: ResultValidator | None = None,
        native_module: object | None = None,
    ) -> None:
        """Publish one prepared candidate, atomically superseding its slot.

        Parameters
        ----------
        key : CandidateKeyProtocol
            Complete immutable candidate identity supplied by the foundation.
        executor : callable
            Already prepared native or compiled executor.  It is never invoked
            during registration.
        correctness_qualified, performance_eligible : bool, optional
            Independent gates.  Explicit modes require correctness; ``auto``
            requires both gates.
        eligible : bool or None, optional
            Backward-compatible alias for ``performance_eligible``.
        prepared : bool, optional
            Must remain true; false is an on-demand compilation attempt.
        result_validator : callable or None, optional
            Optional public result validator.  Validation failures are
            recoverable in ``auto`` and quarantine the candidate.
        native_module : object or None, optional
            Strongly retained module handle for native artifact lifetime.

        """
        _require_prepared_key(key)
        if not prepared:
            logger.error("refusing an unprepared candidate for %s", key.backend)
            raise DispatchError("dispatcher accepts prepared candidates only")
        if not callable(executor):
            logger.error("candidate executor is not callable")
            raise DispatchError("candidate executor is required")
        if eligible is not None:
            performance_eligible = bool(eligible)
        slot = (*_scientific_profile_key(key), key.backend)
        self._candidates[slot] = _RegisteredCandidate(
            key=key,
            executor=executor,
            correctness_qualified=bool(correctness_qualified),
            performance_eligible=bool(performance_eligible),
            result_validator=result_validator,
        )
        if native_module is not None:
            pin_native_module(native_module)

    def active_candidates(self) -> tuple[CandidateKeyProtocol, ...]:
        """Return active keys in registration order."""
        return tuple(candidate.key for candidate in self._candidates.values())

    def quarantine(self, key: CandidateKeyProtocol) -> None:
        """Mark an exact active candidate unavailable until re-registration."""
        candidate = self._exact_candidate(key)
        if candidate is None:
            logger.error("cannot quarantine an inactive candidate")
            raise KeyError(key)
        candidate.quarantined = True

    def dispatch(
        self,
        key: CandidateKeyProtocol,
        mode: DispatchMode = "auto",
        *args: object,
        **kwargs: object,
    ) -> object:
        """Execute an exact prepared candidate or same-device eager fallback.

        ``auto`` tries correctness-qualified, performance-eligible native then
        compile candidates, quarantining recoverable failures and continuing
        through every remaining backend before eager.  Explicit native and
        compile modes require the exact candidate and never fall through.
        Dispatch only executes; it never invokes a compiler.
        """
        if mode not in ("auto", "native", "compile", "eager"):
            logger.error("unknown dispatch mode %r", mode)
            raise DispatchError(f"unknown dispatch mode: {mode}")
        if mode == "eager":
            result = self._execute_eager(key, *args, **kwargs)
            result = self._validate_result(result, None, None)
            self.records.append(DispatchRecord("eager", None))
            return result

        if mode in ("native", "compile"):
            candidate = self._exact_candidate(key)
            if candidate is None or candidate.key.backend != mode:
                logger.error("no exact prepared %s candidate", mode)
                raise DispatchError(f"no exact prepared {mode} candidate for key")
            if not candidate.correctness_qualified or candidate.quarantined:
                logger.error("exact prepared %s candidate is unavailable", mode)
                raise DispatchError(f"prepared {mode} candidate is unavailable")
            result = self._execute_candidate(candidate, *args, **kwargs)
            self.records.append(DispatchRecord(mode, candidate.key))
            return result

        identity = _scientific_profile_key(key)
        candidates = [
            candidate
            for backend in ("native", "compile")
            for candidate in self._candidates.values()
            if _scientific_profile_key(candidate.key) == identity
            and candidate.key.backend == backend
            and candidate.correctness_qualified
            and candidate.performance_eligible
            and not candidate.quarantined
        ]
        for candidate in candidates:
            try:
                result = self._execute_candidate(candidate, *args, **kwargs)
            except CudaExecutionError as error:
                if not error.recoverable:
                    raise
                candidate.quarantined = True
                continue
            except RecoverableExecutionError:
                candidate.quarantined = True
                continue
            self.records.append(
                DispatchRecord(
                    candidate.key.backend,
                    candidate.key,
                    "recoverable quarantine"
                    if candidate is not candidates[0]
                    else None,
                )
            )
            return result

        result = self._execute_eager(key, *args, **kwargs)
        result = self._validate_result(result, None, None)
        self.records.append(
            DispatchRecord(
                "eager",
                None,
                "no eligible prepared candidate"
                if not candidates
                else "quarantined prepared candidates",
            )
        )
        return result

    def _exact_candidate(
        self, key: CandidateKeyProtocol
    ) -> _RegisteredCandidate | None:
        """Find a candidate by exact key equality, not only slot identity."""
        slot = (*_scientific_profile_key(key), getattr(key, "backend", None))
        candidate = self._candidates.get(slot)
        if candidate is None or candidate.key != key:
            return None
        return candidate

    def _execute_candidate(
        self, candidate: _RegisteredCandidate, *args: object, **kwargs: object
    ) -> object:
        """Execute and validate one prepared candidate."""
        try:
            result = candidate.executor(*args, **kwargs)
        except (
            CudaExecutionError,
            FatalExecutionError,
            RecoverableExecutionError,
            GeometryValidationError,
        ):
            raise
        except Exception as error:
            logger.exception("prepared %s execution failed", candidate.key.backend)
            if getattr(candidate.key.device, "kind", None) == "cuda":
                failure = classify_cuda_failure(error)
                phase = getattr(error, "cuda_phase", None)
                context_healthy = bool(getattr(error, "context_healthy", False))
                if phase == "pre_launch" and context_healthy and failure.recoverable:
                    raise CudaExecutionError(
                        failure,
                        str(error),
                        phase="pre_launch",
                        context_healthy=True,
                    ) from error
                raise CudaExecutionError(
                    failure,
                    str(error),
                    phase="synchronized",
                    context_healthy=False,
                ) from error
            raise RecoverableExecutionError(str(error)) from error
        return self._validate_result(result, candidate.result_validator, candidate.key)

    def _execute_eager(
        self, key: CandidateKeyProtocol, *args: object, **kwargs: object
    ) -> object:
        """Execute the eager callback bound to the requested device."""
        executor: object = self._eager
        if isinstance(self._eager, Mapping):
            try:
                executor = self._eager[key.device]
            except KeyError as error:
                logger.error("no eager implementation for device %r", key.device)
                raise DispatchError(
                    "no same-device eager executor is available"
                ) from error
        return _call_eager(executor, key, *args, **kwargs)

    @staticmethod
    def _validate_result(
        result: object,
        validator: ResultValidator | None,
        key: CandidateKeyProtocol | None,
    ) -> object:
        """Apply an optional validator and common v2 result checks."""
        if result is None:
            logger.error("geometry executor returned no result")
            raise IncorrectResultError("geometry executor returned no result")
        if isinstance(result, TransformResultV2):
            try:
                result.validate()
            except Exception as error:
                logger.exception("geometry result scientific validation failed")
                raise IncorrectResultError(
                    "geometry executor returned an invalid v2 result"
                ) from error
            if key is not None:
                budget = _candidate_budget(key)
                iterations = result.iterations
                valid = iterations != -1
                if np.any(valid & ((iterations < 1) | (iterations > budget))):
                    raise IncorrectResultError(
                        "executor returned iterations outside {-1, 1..candidate budget}"
                    )
        if validator is not None:
            try:
                validated = validator(result)
                if validated is not None:
                    result = validated
            except RecoverableExecutionError:
                raise
            except Exception as error:
                logger.exception("geometry result validation failed")
                raise IncorrectResultError(
                    "geometry executor returned a malformed result"
                ) from error
        field_names = getattr(result, "field_names", None)
        if callable(field_names):
            try:
                if len(tuple(field_names())) != 14:
                    raise IncorrectResultError(
                        "result ABI is not exactly fourteen fields"
                    )
            except IncorrectResultError:
                raise
            except Exception as error:
                raise IncorrectResultError(
                    "result ABI could not be inspected"
                ) from error
        validate = getattr(result, "validate", None)
        if callable(validate):
            try:
                if key is None:
                    validate()
                else:
                    try:
                        validate(getattr(key, "operation", None))
                    except TypeError:
                        validate()
            except RecoverableExecutionError:
                raise
            except Exception as error:
                logger.exception("geometry result scientific validation failed")
                raise IncorrectResultError(
                    "geometry executor returned an invalid result"
                ) from error
        if key is not None and hasattr(result, "iterations"):
            iterations = result.iterations
            budget = _candidate_budget(key)
            if isinstance(iterations, np.ndarray):
                if not np.issubdtype(iterations.dtype, np.integer):
                    raise IncorrectResultError("result iterations are not integral")
                valid = iterations != -1
                if np.any(valid & ((iterations < 1) | (iterations > budget))):
                    raise IncorrectResultError(
                        "executor returned iterations outside {-1, 1..candidate budget}"
                    )
            else:
                if isinstance(iterations, bool) or not isinstance(iterations, Integral):
                    raise IncorrectResultError("result iterations are not integral")
                value = int(iterations)
                if value != -1 and not 1 <= value <= budget:
                    raise IncorrectResultError(
                        "executor returned iterations outside {-1, 1..candidate budget}"
                    )
        return result


_PINNED_NATIVE_MODULES: list[object] = []


def pin_native_module(module: object) -> object:
    """Retain a loaded native module until process exit."""
    if module is None:
        logger.error("cannot pin a null native module")
        raise DispatchError("native module cannot be None")
    if not any(existing is module for existing in _PINNED_NATIVE_MODULES):
        _PINNED_NATIVE_MODULES.append(module)
    return module


def pinned_native_modules() -> tuple[object, ...]:
    """Return process-lifetime native module handles retained by dispatchers."""
    return tuple(_PINNED_NATIVE_MODULES)


__all__ = [
    "PERFORMANCE_ELIGIBILITY_ORDER",
    "Backend",
    "CandidateKey",
    "CandidateKeyProtocol",
    "CudaExecutionError",
    "CudaFailure",
    "CudaFailureKind",
    "DispatchError",
    "DispatchMode",
    "DispatchRecord",
    "Dispatcher",
    "EagerExecutorProtocol",
    "Executor",
    "FatalExecutionError",
    "IncorrectResultError",
    "PreparedBackend",
    "RecoverableExecutionError",
    "classify_cuda_failure",
    "pin_native_module",
    "pinned_native_modules",
]
