"""NSBAS inversion module."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

import numpy as np
import psutil
import torch
from tqdm import tqdm

from faninsar import Loops, Pairs, parse_device
from faninsar.logging import setup_logger
from faninsar.NSBAS.coherence_weight import (
    WeightFunc,
    coherence_to_weight,
    normalize_weight_func,
)
from faninsar.NSBAS.tsmodels import TimeSeriesModels

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch._tensor import Tensor


logger = setup_logger(__name__)

#: Minimum Cholesky pivot ratio accepted by the fast (SPD) solve path.
#: Floating-point Cholesky can slip through near-zero pivots, so pixels
#: with a smaller ratio fall back to the minimum-norm eigendecomposition.
_PIVOT_RATIO_MIN = 1e-6
#: Null-space column mass below which a parameter counts as constrained.
_NULL_MASS_TOL = 1e-8


@dataclass
class LstsqDiagnostics:
    """Per-pixel reliability diagnostics of a censored least-squares solve.

    Attributes
    ----------
    is_constrained : Tensor
        ``(n_param, n_pt)`` bool. ``True`` where the parameter is
        determined by the observations. ``False`` marks a local gap whose
        value comes from the minimum-norm convention.
    rank : Tensor
        ``(n_pt,)`` int. Numerical rank of each pixel's system.
    n_observations : Tensor
        ``(n_pt,)`` int. Valid (non-NaN) observations per pixel.
    condition_number : Tensor
        ``(n_pt,)`` float. Ratio of the largest to the smallest retained
        eigenvalue on the minimum-norm path; on the Cholesky fast path it
        is the pivot-ratio proxy, a conservative estimate of the same
        quantity.

    """

    is_constrained: Tensor
    rank: Tensor
    n_observations: Tensor
    condition_number: Tensor

    def cpu(self) -> LstsqDiagnostics:
        """Return a copy with every field on the CPU."""
        return LstsqDiagnostics(
            is_constrained=self.is_constrained.cpu(),
            rank=self.rank.cpu(),
            n_observations=self.n_observations.cpu(),
            condition_number=self.condition_number.cpu(),
        )

    def numpy(self) -> LstsqDiagnostics:
        """Return a copy with every field as a NumPy array."""
        return LstsqDiagnostics(
            is_constrained=self.is_constrained.cpu().numpy(),
            rank=self.rank.cpu().numpy(),
            n_observations=self.n_observations.cpu().numpy(),
            condition_number=self.condition_number.cpu().numpy(),
        )


@dataclass
class NSBASResult:
    """Inversion result with per-pixel reliability diagnostics.

    Attributes
    ----------
    increments : ndarray or Tensor
        ``(n_date - 1, n_pt)`` incremental displacement.
    model_parameters : ndarray or Tensor
        ``(n_param, n_pt)`` temporal-model parameters; empty when the
        solver has no model.
    pair_residuals : ndarray or Tensor
        ``(n_pair, n_pt)`` interferogram observation residuals.
    model_residuals : ndarray or Tensor
        ``(n_date, n_pt)`` temporal-model constraint residuals; empty
        when the solver has no model.
    is_constrained : ndarray or Tensor
        ``(n_date - 1, n_pt)`` bool. ``True`` where the increment is
        determined by the interferograms alone. ``False`` marks a local
        gap whose value comes from the minimum-norm convention (pure
        SBAS) or from the temporal model (NSBAS).
    n_observations : ndarray or Tensor
        ``(n_pt,)`` int. Valid interferogram observations per pixel.
    rank : ndarray or Tensor
        ``(n_pt,)`` int. Rank of each pixel's data-only system.
    condition_number : ndarray or Tensor
        ``(n_pt,)`` float. Conditioning estimate of the data-only system
        (see :class:`LstsqDiagnostics` for the fast-path proxy).

    """

    increments: NDArray[np.floating] | Tensor
    model_parameters: NDArray[np.floating] | Tensor
    pair_residuals: NDArray[np.floating] | Tensor
    model_residuals: NDArray[np.floating] | Tensor
    is_constrained: NDArray[np.bool_] | Tensor
    n_observations: NDArray[np.integer] | Tensor
    rank: NDArray[np.integer] | Tensor
    condition_number: NDArray[np.floating] | Tensor


class NSBASSolver:
    """Solver class to build NSBAS matrix and perform inversion.

    The NSBAS inversion is expressed as: ``d = Gm``, where ``d`` is the
    unwrapped interferograms matrix, ``G`` is the NSBAS matrix, and ``m`` is
    the model parameters, which is the combination of the deformation increment
    and the model parameters.
    see following paper for more details: `https://ens.hal.science/hal-02185213`_ ,
    `https://www.sciencedirect.com/science/article/pii/S0924271625000772`_

    Observation weights are selected only through ``weight_func``. ``None`` on
    the constructor or on :meth:`inverse` means "do not change stored state".
    The first fill-in is unweighted (``"no"``). A weighted choice (``"coh"``,
    ``"fim"``, or ``"var"``) without stored coherence raises ``ValueError``.
    Interferogram rows are scaled by ``sqrt(weight)``; NSBAS model-constraint
    rows stay at the original ``gamma`` scale.

    .. note::

        After initialization, the unwrapped interferograms can still be
        replaced with :meth:`set_d`. This is useful when they are divided
        into multiple patches and the NSBAS matrix is calculated for each
        patch separately. The stored coherence mask and weights are reapplied.

    Examples
    --------
    Unweighted NSBAS inversion:

    >>> import numpy as np
    >>> import faninsar as fis
    >>> from faninsar import NSBAS
    >>> names = [
    ...     "20170111_20170204",
    ...     "20170111_20170222",
    ...     "20170111_20170318",
    ...     "20170204_20170222",
    ...     "20170204_20170318",
    ...     "20170204_20170330",
    ...     "20170222_20170318",
    ...     "20170222_20170330",
    ...     "20170222_20170411",
    ...     "20170318_20170330",
    ... ]
    >>> pairs = fis.Pairs.from_names(names)
    >>> unw = np.random.default_rng(0).random((len(pairs), 5))
    >>> model = NSBAS.AnnualSinusoidalModel(pairs.dates)
    >>> solver = NSBAS.NSBASSolver(unw, pairs, model, verbose=False, device="cpu")
    >>> solver
    NSBASSolver(
        pairs: Pairs(10)
        model: AnnualSinusoidalModel(dates: 6, unit: day)
        gamma: 0.0001
        weight_func: no
        rcond: 1e-10
        G shape: (16, 9)
        d shape: (16, 5)
    )
    >>> incs, params, residual_pair, residual_tsm = solver.inverse()

    Replace the unwrapped interferograms for another patch with the same
    pairs:

    >>> solver.set_d(np.random.default_rng(1).random((len(pairs), 10)))
    >>> solver
    NSBASSolver(
        pairs: Pairs(10)
        model: AnnualSinusoidalModel(dates: 6, unit: day)
        gamma: 0.0001
        weight_func: no
        rcond: 1e-10
        G shape: (16, 9)
        d shape: (16, 10)
    )

    Set Fisher-information weights at construction, then switch to DS
    variance weights in a later :meth:`inverse` call. Omitting a kwarg
    keeps the stored value. Weighted inversion needs coherence with the
    same shape as ``unw``:

    >>> coh = np.full(unw.shape, 0.8)
    >>> solver = NSBAS.NSBASSolver(
    ...     unw,
    ...     pairs,
    ...     model,
    ...     coh=coh,
    ...     weight_func="fim",
    ...     n_looks=20,
    ...     verbose=False,
    ...     device="cpu",
    ... )
    >>> solver
    NSBASSolver(
        pairs: Pairs(10)
        model: AnnualSinusoidalModel(dates: 6, unit: day)
        gamma: 0.0001
        weight_func: fim
        rcond: 1e-10
        G shape: (5, 16, 9)
        d shape: (16, 5)
    )
    >>> incs, params, residual_pair, residual_tsm = solver.inverse()
    >>> incs, params, residual_pair, residual_tsm = solver.inverse(weight_func="var")

    """

    _pairs: Pairs
    _model: TimeSeriesModels | None
    _gamma: float
    _G: torch.Tensor
    _d: torch.Tensor
    _unw: np.ndarray
    _coh: np.ndarray | None
    _coh_threshold: float
    _weight_func: WeightFunc
    _n_looks: int
    _device: torch.device
    _dtype: torch.dtype
    _verbose: bool

    __slots__ = [
        "_G",
        "_coh",
        "_coh_threshold",
        "_d",
        "_device",
        "_dtype",
        "_gamma",
        "_model",
        "_n_looks",
        "_pairs",
        "_rcond",
        "_unw",
        "_verbose",
        "_weight_func",
    ]

    def __init__(
        self,
        unw: NDArray[np.floating] | torch.Tensor,
        pairs: Pairs | Sequence[str],
        model: TimeSeriesModels | None = None,
        coh: NDArray[np.floating] | torch.Tensor | None = None,
        coh_threshold: float | None = None,
        weight_func: WeightFunc | str | None = None,
        n_looks: int | None = None,
        rcond: float = 1e-10,
        gamma: float = 0.0001,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        verbose: bool = True,
    ) -> None:
        """Initialize NSBAS solver.

        Parameters
        ----------
        unw : ndarray or torch.Tensor
            Unwrapped interferograms with shape ``(n_pair, n_pixel)``.
        pairs : Pairs or Sequence[str]
            Interferogram pairs corresponding to ``unw``.
        model : TimeSeriesModels, optional
            Parametric NSBAS model. Omit for pure SBAS.
        coh : ndarray or torch.Tensor, optional
            Spatial coherence with the same shape as ``unw``. ``None`` means
            no coherence is stored. Values at or below ``coh_threshold``
            are treated as missing. When a weight function is selected,
            ``coh`` is also converted to observation weights.
        coh_threshold : float, optional
            Coherence mask threshold. ``None`` keeps the internal default
            of ``0.4``. Pixels with ``coh <= coh_threshold`` are set to NaN
            before inversion.
        weight_func : {"no", "coh", "fim", "var"}, optional
            Observation-weight definition. ``None`` keeps the unweighted
            default ``"no"``. ``coh`` uses coherence, ``fim`` uses
            ``2 L γ² / (1 - γ²)``, and ``var`` uses the inverse DS phase
            variance. A weighted choice without ``coh`` raises
            ``ValueError``.
            ``sbas``/``uniform`` map to ``no``, ``lin`` to ``coh``, and
            ``fisher``/``wls`` to ``fim``.
        n_looks : int, optional
            Independent looks ``L`` used by ``fim`` and ``var``. ``None``
            keeps the internal default of ``1``.
        rcond : float, optional
            Relative cutoff separating determined from unconstrained
            directions in the least-squares solve. Rank-deficient pixels
            receive the minimum-norm solution; directions whose
            eigenvalue falls below ``rcond`` times the largest are
            treated as unconstrained. Must be in ``(0, 1)``. Default is
            ``1e-10``.
        gamma : float, optional
            Weight of the NSBAS model-constraint rows.
        device, dtype, verbose
            Torch device, compute dtype, and progress-bar flag.

        Raises
        ------
        TypeError
            If ``pairs`` is not a :class:`~faninsar.Pairs` or sequence of
            pair names.
        ValueError
            If ``weight_func`` is weighted and ``coh`` is omitted, or if
            ``coh`` does not match the shape of ``unw``.

        """
        if isinstance(pairs, Pairs):
            self._pairs = pairs
        elif isinstance(pairs, Sequence):
            self._pairs = Pairs.from_names(pairs)
        else:
            msg = "pairs must be either Pairs or Sequence"
            logger.error(msg, extra={"pairs_type": type(pairs).__name__})
            raise TypeError(msg)

        self._device = parse_device(device)
        self._dtype = dtype
        self._verbose = verbose
        _check_rcond(rcond)

        self._model = None
        self._gamma = 0.0001
        self._unw = _to_numpy(unw)
        self._coh = None if coh is None else _to_numpy(coh)
        self._coh_threshold = 0.4 if coh_threshold is None else coh_threshold
        self._n_looks = 1 if n_looks is None else n_looks
        self._weight_func = normalize_weight_func(weight_func)
        self._rcond = float(rcond)

        if model is not None:
            _check_model(model)
            _check_gamma(gamma)
            self._model = model
            self._gamma = gamma

        self._rebuild_system()

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}(pairs: {self.pairs}, model:"
            f"{self.model}, gamma: {self.gamma})"
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}(\n"
            f"    pairs: {self.pairs}\n"
            f"    model: {self.model!s}\n"
            f"    gamma: {self.gamma}\n"
            f"    weight_func: {self._weight_func}\n"
            f"    rcond: {self._rcond}\n"
            f"    G shape: {tuple(self.G.shape)}\n"
            f"    d shape: {tuple(self.d.shape)}\n"
            ")"
        )

    @property
    def pairs(self) -> Pairs:
        """Return pairs."""
        return self._pairs

    @property
    def model(self) -> TimeSeriesModels | None:
        """Return model."""
        return self._model

    @property
    def device(self) -> torch.device:
        """Return the torch device used to store matrices."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return the torch dtype used to store matrices."""
        return self._dtype

    @property
    def verbose(self) -> bool:
        """Return whether the inversion shows progress."""
        return self._verbose

    @property
    def gamma(self) -> float:
        """Return gamma."""
        return self._gamma

    @property
    def rcond(self) -> float:
        """Return the relative cutoff for unconstrained directions."""
        return self._rcond

    @property
    def d(self) -> torch.Tensor:
        """Return ``d`` matrix for NSBAS ``d = Gm``."""
        return self._d

    def set_d(self, unw: np.ndarray | torch.Tensor) -> None:
        """Set unwrapped interferograms and rebuild ``G`` and ``d``.

        The input is the raw unwrapped phases. Any current coherence mask
        and observation weights are reapplied. If coherence is stored, the
        new array must have the same shape as that coherence.

        Parameters
        ----------
        unw : ndarray or torch.Tensor
            Unwrapped interferograms with shape ``(n_pair, n_pixel)``.

        Raises
        ------
        TypeError
            If ``unw`` is not a NumPy array or torch tensor.
        ValueError
            If ``unw`` is not 2-D, does not match the pair count, or does
            not match the stored coherence shape.

        """
        if not isinstance(unw, (np.ndarray, torch.Tensor)):
            msg = "input unw must be a numpy array or torch tensor"
            logger.error(msg, extra={"unw_type": type(unw).__name__})
            raise TypeError(msg)
        unw_np = _to_numpy(unw)
        if len(unw_np.shape) != 2:
            msg = "input unw must be a 2D array with shape of (n_pair, n_pixel)"
            logger.error(msg, extra={"shape": getattr(unw_np, "shape", None)})
            raise ValueError(msg)
        if unw_np.shape[0] != len(self.pairs):
            msg = "input unw must have the same rows number as pairs number"
            logger.error(
                msg,
                extra={"rows": unw_np.shape[0], "pairs": len(self.pairs)},
            )
            raise ValueError(msg)
        self._unw = unw_np
        self._rebuild_system()

    @property
    def G(self) -> torch.Tensor:  # noqa: N802
        """Return ``G`` matrix for NSBAS ``d = Gm``."""
        return self._G

    # @G.setter
    def set_G(self, G: np.ndarray | torch.Tensor) -> None:  # noqa: N802
        """Set ``G`` matrix for NSBAS ``d = Gm``."""
        if not isinstance(G, (np.ndarray, torch.Tensor)):
            msg = "G must be a numpy array or torch tensor"
            logger.error(msg, extra={"G_type": type(G).__name__})
            raise TypeError(msg)

        self._G = torch.as_tensor(G, dtype=self.dtype, device=self.device)

    def _make_nsbas_matrix(
        self,
        G_br: np.ndarray,
        G_tl: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        """Make NSBAS matrix by input G_br, G_tl, and gamma.

        Parameters
        ----------
        G_br : np.ndarray
            The bottom right part of NSBAS matrix, which is the model matrix for
            time-series model in NSBAS inversion.
        G_tl : np.ndarray
            The top left part of NSBAS matrix, which is the conventional SBAS
            matrix mapping the incremental deformation to interferograms.
        gamma : float
            The weight for the bottom part of NSBAS matrix, which is used to
            balance the data term and the model term in NSBAS inversion.
            A smaller gamma means more weight on the data term. This value is
            typically small enough to avoid affecting the data term.

        Returns
        -------
        G : np.ndarray
            The NSBAS matrix for NSBAS inversion.

        """
        G_br = np.asarray(G_br, dtype=np.float32)  # noqa: N806

        if len(G_br.shape) == 1:
            G_br = G_br.reshape(-1, 1)  # noqa: N806
        n_param = G_br.shape[1]

        n_date = len(self.pairs.dates)
        G_bl = np.tril(np.ones((n_date, n_date - 1), dtype=np.float32), k=-1)  # noqa: N806
        G_b = np.hstack((G_bl, G_br)) * gamma  # noqa: N806
        if G_tl.ndim == 2:
            G_t = np.hstack((G_tl, np.zeros((len(self._pairs), n_param))))  # noqa: N806
            return np.vstack((G_t, G_b))
        if G_tl.ndim == 3:
            n_pt = G_tl.shape[0]
            G_t = np.concat((G_tl, np.zeros((n_pt, len(self._pairs), n_param))), axis=2)  # noqa: N806
            G_b = np.repeat(G_b[None, :, :], n_pt, axis=0)  # noqa: N806
            return np.concat((G_t, G_b), axis=1)
        msg = "G_tl must be either 2D or 3D array"
        logger.error(msg, extra={"G_tlshape": getattr(G_tl, "shape", None)})
        raise ValueError(msg)

    def _make_sbas_matrix(self) -> np.ndarray:
        """Make SBAS matrix."""
        return self.pairs.sbas_matrix()

    def _restructure_unw(
        self,
        unw: np.ndarray,
    ) -> np.ndarray:
        """Restructure unw matrix by appending model matrix part."""
        if self.model is not None:
            unw = np.vstack((unw, np.zeros((len(self.pairs.dates), unw.shape[1]))))
        return unw

    def _rebuild_system(self) -> None:
        """Rebuild ``G`` and ``d`` from stored unwrap, coherence, and weights."""
        g_sbas, observations = self._make_data_system()

        if self.model is not None:
            self.set_G(self._make_nsbas_matrix(self.model.G_br, g_sbas, self._gamma))
        else:
            self.set_G(g_sbas)

        if self.model is None:
            self._d: Tensor = torch.as_tensor(
                observations,
                dtype=self._dtype,
                device=self._device,
            )
        else:
            self._d = torch.as_tensor(
                self._restructure_unw(observations),
                dtype=self._dtype,
                device=self._device,
            )

    def _make_data_system(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the SBAS design matrix and observations before model rows.

        Coherence masking and weighting are applied: observations at or
        below the coherence threshold become NaN, and weighted systems
        scale both the design rows and the observations by
        ``sqrt(weight)``. The SBAS part of the NSBAS matrix and ``d``
        are built from this pair.
        """
        g_sbas = self._make_sbas_matrix().astype(np.float64, copy=False)
        observations = np.array(self._unw, dtype=np.float64, copy=True)
        if observations.shape[0] != len(self.pairs):
            msg = "unwrapped interferograms must have the same rows as pairs"
            logger.error(
                msg,
                extra={"rows": observations.shape[0], "pairs": len(self.pairs)},
            )
            raise ValueError(msg)

        weighted = self._weight_func != "no"
        if weighted and self._coh is None:
            msg = "coherence is required when weight_func is not 'no'"
            logger.error(msg, extra={"weight_func": self._weight_func})
            raise ValueError(msg)

        if self._coh is not None:
            coh_np = np.asarray(self._coh, dtype=np.float64)
            if coh_np.shape != observations.shape:
                msg = "coh must have the same shape as unwrapped interferograms"
                logger.error(
                    msg,
                    extra={"coh_shape": coh_np.shape, "unw_shape": observations.shape},
                )
                raise ValueError(msg)
            observations = np.where(
                coh_np <= self._coh_threshold,
                np.nan,
                observations,
            )
            if weighted:
                weight = coherence_to_weight(
                    coh_np,
                    weight_func=self._weight_func,
                    n_looks=self._n_looks,
                )
                weight_sqrt = np.sqrt(weight)
                observations = observations * weight_sqrt
                g_sbas = (g_sbas[:, :, None] * weight_sqrt[:, None, :]).transpose(
                    2,
                    0,
                    1,
                )
        return g_sbas, observations

    @overload
    def inverse(
        self,
        return_numpy: bool = True,
        *,
        coh: NDArray[np.floating] | torch.Tensor | None = None,
        coh_threshold: float | None = None,
        weight_func: WeightFunc | str | None = None,
        n_looks: int | None = None,
        rcond: float | None = None,
        diagnostics: bool = False,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]: ...
    @overload
    def inverse(
        self,
        return_numpy: bool = False,
        *,
        coh: NDArray[np.floating] | torch.Tensor | None = None,
        coh_threshold: float | None = None,
        weight_func: WeightFunc | str | None = None,
        n_looks: int | None = None,
        rcond: float | None = None,
        diagnostics: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]: ...
    def inverse(
        self,
        return_numpy: bool = True,
        *,
        coh: NDArray[np.floating] | torch.Tensor | None = None,
        coh_threshold: float | None = None,
        weight_func: WeightFunc | str | None = None,
        n_looks: int | None = None,
        rcond: float | None = None,
        diagnostics: bool = False,
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | NSBASResult
    ):
        """Calculate increment displacement difference by NSBAS inversion.

        Parameters
        ----------
        return_numpy : bool, optional
            Return numpy arrays if True, otherwise torch tensors, by default True.
        coh : ndarray or torch.Tensor, optional
            Spatial coherence used for masking and weighting. ``None`` keeps
            the coherence stored at construction or by a previous call.
        coh_threshold : float, optional
            Override the coherence mask threshold. ``None`` keeps the stored
            value.
        weight_func : {"no", "coh", "fim", "var"}, optional
            Override the observation-weight definition. ``None`` keeps the
            stored choice. A weighted choice without stored coherence raises
            ``ValueError``. Aliases match the constructor
            (``sbas``/``uniform`` → ``no``, ``lin`` → ``coh``,
            ``fisher``/``wls`` → ``fim``).
        n_looks : int, optional
            Override the independent-looks value used by ``fim`` and ``var``.
            ``None`` keeps the stored value.
        rcond : float, optional
            Override the relative cutoff separating determined from
            unconstrained directions. ``None`` keeps the stored value
            (default ``1e-10``).
        diagnostics : bool, optional
            If True, return an :class:`NSBASResult` with per-pixel
            reliability diagnostics instead of the plain tuple. The
            diagnostics describe the data-only (SBAS) system, so with a
            temporal model ``is_constrained`` marks increments that the
            interferograms alone do not determine. Default is False.

        Returns
        -------
        incs: np.ndarray | torch.Tensor (n_date - 1, n_pt)
            Incremental displacement
        params: np.ndarray | torch.Tensor (n_param, n_pt)
            parameters of model in NSBAS inversion
        residual_pair: np.ndarray | torch.Tensor (n_pair, n_pt)
            residual between interferograms and model result
        residual_tsm: np.ndarray | torch.Tensor     (n_date, n_pt)
            residual between time-series model and model result
        NSBASResult
            When ``diagnostics=True``, the same four arrays plus
            ``is_constrained``, ``n_observations``, ``rank``, and
            ``condition_number``.

        Raises
        ------
        ValueError
            If ``weight_func`` is weighted and no coherence is stored, or if
            ``coh`` does not match the shape of the unwrapped interferograms.

        """
        rebuild = False
        if coh is not None:
            self._coh = _to_numpy(coh)
            rebuild = True
        if coh_threshold is not None:
            self._coh_threshold = coh_threshold
            rebuild = True
        if n_looks is not None:
            self._n_looks = n_looks
            rebuild = True
        if weight_func is not None:
            self._weight_func = normalize_weight_func(weight_func)
            rebuild = True
        if rcond is not None:
            _check_rcond(rcond)
            self._rcond = float(rcond)
        if rebuild:
            self._rebuild_system()

        n_param = len(self.model.param_names) if self.model is not None else 0
        n_pair = len(self.pairs)
        if diagnostics and self.model is None:
            result, data_diag = batch_lstsq(
                self.G,
                self.d,
                dtype=self.dtype,
                device=self.device,
                verbose=self.verbose,
                tqdm_args={"desc": "  NSBAS inversion"},
                return_numpy=False,
                rcond=self._rcond,
                diagnostics=True,
            )
        else:
            result = batch_lstsq(
                self.G,
                self.d,
                dtype=self.dtype,
                device=self.device,
                verbose=self.verbose,
                tqdm_args={"desc": "  NSBAS inversion"},
                return_numpy=False,
                rcond=self._rcond,
            )
            data_diag = None
        if diagnostics and self.model is not None:
            g_sbas, observations = self._make_data_system()
            _, data_diag = batch_lstsq(
                g_sbas,
                observations,
                dtype=self.dtype,
                device=self.device,
                verbose=self.verbose,
                tqdm_args={"desc": "  SBAS data diagnostics"},
                return_numpy=False,
                rcond=self._rcond,
                diagnostics=True,
            )
        residual = _calculate_residual(
            self.G,
            self.d,
            result,
            dtype=self.dtype,
            device=self.device,
        )
        if return_numpy:
            result = result.cpu().numpy()
            residual = residual.cpu().numpy()

        if n_param == 0:
            incs = result
            params = result[0:0, :]
        else:
            incs = result[:-n_param, :]
            params = result[-n_param:, :]
        residual_pair = residual[:n_pair]
        residual_tsm = residual[n_pair:]

        if diagnostics:
            if return_numpy:
                data_diag = data_diag.numpy()
            return NSBASResult(
                increments=incs,
                model_parameters=params,
                pair_residuals=residual_pair,
                model_residuals=residual_tsm,
                is_constrained=data_diag.is_constrained,
                n_observations=data_diag.n_observations,
                rank=data_diag.rank,
                condition_number=data_diag.condition_number,
            )

        return incs, params, residual_pair, residual_tsm


def device_mem_size(device: str | torch.device | None) -> int:
    """Get memory size (in MB) for GPU or CPU.

    Parameters
    ----------
    device : Optional[str | torch.device]
        device of torch.tensor used for computation.

    Returns
    -------
    mem_size : int
        memory size (in MB) for GPU or CPU.

    """
    device_type = parse_device(device).type
    if device_type == "cuda":
        free_memory, _ = torch.cuda.mem_get_info()
        mem_size = int(free_memory / 1024**2)
    else:
        # macos share memory with CPU
        mem_free = psutil.virtual_memory().available
        mem_size = int(mem_free / 1024**2)

    return mem_size


def _check_model(model: TimeSeriesModels) -> None:
    """Check model is a TimeSeriesModels instance."""
    if not isinstance(model, TimeSeriesModels):
        msg = "model must be a TimeSeriesModels instance"
        logger.error(msg, extra={"model_type": type(model).__name__})
        raise TypeError(msg)


def _check_gamma(gamma: float) -> None:
    """Update gamma and G by input gamma."""
    if not isinstance(gamma, (float, int)):
        msg = "gamma must be either float or int"
        logger.error(msg, extra={"gamma_type": type(gamma).__name__})
        raise TypeError(msg)
    if gamma <= 0:
        msg = "gamma must be positive"
        logger.error(msg, extra={"gamma": gamma})
        raise ValueError(msg)


def _check_rcond(rcond: float) -> None:
    """Validate the relative cutoff for unconstrained directions."""
    if not isinstance(rcond, (float, int)) or isinstance(rcond, bool):
        msg = "rcond must be either float or int"
        logger.error(msg, extra={"rcond_type": type(rcond).__name__})
        raise TypeError(msg)
    if not 0.0 < float(rcond) < 1.0:
        msg = "rcond must be in (0, 1)"
        logger.error(msg, extra={"rcond": rcond})
        raise ValueError(msg)


def _to_numpy(array: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a numpy array or torch tensor to a numpy array."""
    if isinstance(array, np.ndarray):
        return array
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    msg = "array must be a numpy array or torch tensor"
    logger.error(msg, extra={"array_type": type(array).__name__})
    raise TypeError(msg)


def _get_patch_col(
    G: np.ndarray | torch.Tensor,
    d: np.ndarray | torch.Tensor,
    mem_size: int,
    dtype: torch.dtype,
    safe_factor: float = 2,
) -> list[list[int]]:
    """Get patch number of cols for memory size (in MB) for SBAS inversion.

    Parameters
    ----------
    G : np.ndarray | torch.Tensor
        model field matrix with shape of (n_im, n_param) or (n_pt, n_im, n_param).
    d : np.ndarray | torch.Tensor
        data field matrix with shape of (n_im, n_pt).
    mem_size : int
        memory size (in MB) for GPU or CPU.
    dtype: torch.dtype
        dtype of torch tensor
    safe_factor : float, optional
        safe factor for memory size, by default 2

    Returns
    -------
    patch_col : list[list[int]]
        List of the number of rows for each patch.
        eg: [[0, 1234], [1235, 2469],... ]

    """
    if G.ndim not in {2, 3}:
        msg = "Dimension of G must be 2 or 3"
        logger.error(msg, extra={"G_shape": getattr(G, "shape", None)})
        raise ValueError(msg)
    if d.ndim != 2:
        msg = "d must be a 2D matrix"
        logger.error(msg, extra={"d_shape": getattr(d, "shape", None)})
        raise ValueError(msg)
    if mem_size <= 0:
        msg = "mem_size must be positive"
        logger.error(msg, extra={"mem_size": mem_size})
        raise ValueError(msg)
    if safe_factor <= 0:
        msg = "safe_factor must be positive"
        logger.error(msg, extra={"safe_factor": safe_factor})
        raise ValueError(msg)

    m, n = d.shape
    r = G.shape[-1]
    if G.ndim == 3 and G.shape[0] != n:
        msg = "The first dimension of 3D G must match the number of columns in d"
        logger.error(msg, extra={"G_shape": G.shape, "d_shape": d.shape})
        raise ValueError(msg)

    value_size = torch.tensor([], dtype=dtype).element_size()
    bool_size = torch.tensor([], dtype=torch.bool).element_size()
    available_bytes = int(mem_size * 2**20 / safe_factor)

    # Per-pixel tensors created in censored_lstsq include d, d_nan, M, x, rhs,
    # T, and the weighted-G temporary used to build the normal equations.
    per_col_bytes = (
        (m + r + r + r**2) * value_size + (2 * m) * bool_size + (m * r) * value_size
    )
    fixed_bytes = m * r * value_size
    if G.ndim == 3:
        per_col_bytes += m * r * value_size
        fixed_bytes = 0

    row_spacing = max(1, int((available_bytes - fixed_bytes) // per_col_bytes))
    row_spacing = min(n, row_spacing)

    return [[col, min(col + row_spacing, n)] for col in range(0, n, row_spacing)]


def _calculate_residual(
    G: torch.Tensor,
    d: torch.Tensor,
    result: torch.Tensor,
    dtype: torch.dtype,
    device: str | torch.device | None,
) -> torch.Tensor:
    """Calculate residuals without expanding 3D model matrices.

    Parameters
    ----------
    G : torch.Tensor
        Model matrix with shape ``(n_im, n_param)`` or
        ``(n_pt, n_im, n_param)``.
    d : torch.Tensor
        Data matrix with shape ``(n_im, n_pt)``.
    result : torch.Tensor
        Least-squares result matrix with shape ``(n_param, n_pt)``.
    dtype : torch.dtype
        Data type used to estimate patch memory.
    device : str | torch.device | None
        Device used for residual calculation.

    Returns
    -------
    residual : torch.Tensor
        Residual matrix with shape ``(n_im, n_pt)``.

    """
    device = parse_device(device)
    if G.ndim not in {2, 3}:
        msg = "Dimension of G must be 2 or 3"
        logger.error(msg, extra={"G_shape": getattr(G, "shape", None)})
        raise ValueError(msg)
    if d.ndim != 2:
        msg = "d must be a 2D matrix"
        logger.error(msg, extra={"d_shape": getattr(d, "shape", None)})
        raise ValueError(msg)

    residual = torch.empty(d.shape, dtype=dtype, device=device)
    patch_col = _get_patch_col(G, d, device_mem_size(device), dtype)
    if G.ndim == 2:
        G_matrix = G.to(device=device, dtype=dtype)  # noqa: N806
    for c0, c1 in patch_col:
        result_patch = result[:, c0:c1].to(device=device, dtype=dtype)
        d_patch = d[:, c0:c1].to(device=device, dtype=dtype)
        if G.ndim == 2:
            predicted = torch.matmul(G_matrix, result_patch)
        else:
            G_patch = G[c0:c1, :, :].to(device=device, dtype=dtype)  # noqa: N806
            predicted = torch.einsum("pmr,rp->mp", G_patch, result_patch)
        residual[:, c0:c1] = d_patch - predicted
    return residual


@overload
def batch_lstsq(
    G: np.ndarray,
    d: np.ndarray,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
    verbose: bool = True,
    tqdm_args: dict | None = None,
    return_numpy: bool = True,
) -> NDArray[np.floating]: ...
@overload
def batch_lstsq(
    G: torch.Tensor,
    d: torch.Tensor,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
    verbose: bool = True,
    tqdm_args: dict | None = None,
    return_numpy: bool = False,
) -> torch.Tensor: ...
def batch_lstsq(
    G: np.ndarray | torch.Tensor,
    d: np.ndarray | torch.Tensor,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
    verbose: bool = True,
    tqdm_args: dict | None = None,
    return_numpy: bool = True,
    rcond: float = 1e-10,
    diagnostics: bool = False,
) -> np.ndarray | torch.Tensor | tuple:
    """Batch least-squares solver for solving the least squares problem.

    Parameters
    ----------
    G : np.ndarray | torch.Tensor
        model field matrix with shape of (n_im, n_param) or (n_pt, n_im, n_param).
        If G is 3D, the first dimension is the G matrix for every pixel.
    d : np.ndarray | torch.Tensor
        data field matrix with shape of (n_im, n_pt).
    dtype : torch.dtype, optional
        dtype of torch.tensor used for computation
    device : Optional[str | torch.device], optional
        device of torch.tensor used for computation. If None, use GPU if
        available, otherwise use CPU.
    verbose : bool, optional
        If True, show progress bar, by default True
    tqdm_args : dict, optional
        Arguments to be passed to `tqdm.tqdm <https://tqdm.github.io/docs/tqdm#tqdm-objects>`_
        Object for progress bar.
    return_numpy : bool, optional
        Return numpy array if True, otherwise torch tensor, by default True.
    rcond : float, optional
        Relative cutoff separating determined from unconstrained
        directions, forwarded to :func:`censored_lstsq`. Default is 1e-10.
    diagnostics : bool, optional
        If True, also return the :class:`LstsqDiagnostics` collected from
        every patch. Default is False.

    Returns
    -------
    x : np.ndarray | torch.Tensor
        (n_im x n_pt) matrix that minimizes norm(M*(GX - d)). If return_numpy is
        True, return a numpy array, otherwise a torch tensor.
    diagnostics : LstsqDiagnostics, optional
        Per-pixel diagnostics, only when ``diagnostics=True``.

    """
    if tqdm_args is None:
        tqdm_args = {}
    tqdm_args.setdefault("desc", "Batch least-squares")
    tqdm_args.setdefault("unit", "Batch")
    n_pt = d.shape[1]
    n_param = G.shape[1] if G.ndim == 2 else G.shape[2]
    device = parse_device(device)

    result = torch.full((n_param, n_pt), torch.nan, dtype=dtype, device=device)
    result_diag = None
    if diagnostics:
        result_diag = LstsqDiagnostics(
            is_constrained=torch.zeros(
                (n_param, n_pt), dtype=torch.bool, device=device
            ),
            rank=torch.zeros(n_pt, dtype=torch.int64, device=device),
            n_observations=torch.zeros(n_pt, dtype=torch.int64, device=device),
            condition_number=torch.full((n_pt,), torch.nan, dtype=dtype, device=device),
        )
    mem_size = device_mem_size(device)
    patch_col = _get_patch_col(G, d, mem_size, dtype)
    if verbose:
        patch_col = tqdm(patch_col, **tqdm_args)

    for col in patch_col:
        c0, c1 = int(col[0]), int(col[1])
        if G.ndim == 2:
            chunk = censored_lstsq(
                G=G,
                d=d[:, c0:c1],
                dtype=dtype,
                device=device,
                return_numpy=False,
                rcond=rcond,
                diagnostics=diagnostics,
            )
        elif G.ndim == 3:
            chunk = censored_lstsq(
                G=G[c0:c1, :, :],
                d=d[:, c0:c1],
                dtype=dtype,
                device=device,
                return_numpy=False,
                rcond=rcond,
                diagnostics=diagnostics,
            )
        else:
            msg = "Dimension of G must be 2 or 3"
            raise ValueError(msg)
        if diagnostics:
            result[:, c0:c1] = chunk[0]
            chunk_diag = chunk[1]
            result_diag.is_constrained[:, c0:c1] = chunk_diag.is_constrained
            result_diag.rank[c0:c1] = chunk_diag.rank
            result_diag.n_observations[c0:c1] = chunk_diag.n_observations
            result_diag.condition_number[c0:c1] = chunk_diag.condition_number
        else:
            result[:, c0:c1] = chunk
    if return_numpy:
        result = result.cpu().numpy()
        if diagnostics:
            return result, result_diag.numpy()
        return result
    if diagnostics:
        return result, result_diag
    return result


@overload
def censored_lstsq(
    G: np.ndarray | torch.Tensor,
    d: np.ndarray | torch.Tensor,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
    return_numpy: bool = True,
    *,
    rcond: float = 1e-10,
    diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, LstsqDiagnostics]: ...
@overload
def censored_lstsq(
    G: np.ndarray | torch.Tensor,
    d: np.ndarray | torch.Tensor,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
    return_numpy: bool = False,
    *,
    rcond: float = 1e-10,
    diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, LstsqDiagnostics]: ...
def censored_lstsq(
    G: np.ndarray | torch.Tensor,
    d: np.ndarray | torch.Tensor,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
    return_numpy: bool = True,
    *,
    rcond: float = 1e-10,
    diagnostics: bool = False,
) -> (
    np.ndarray
    | torch.Tensor
    | tuple[np.ndarray, LstsqDiagnostics]
    | tuple[torch.Tensor, LstsqDiagnostics]
):
    """Solve least squares subject to missing data, rank-robustly.

    Reference: http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/.

    Every pixel is solved through the normal equations ``T = G' M G``
    with ``M`` the pixel's valid-row mask (NaN rows in ``d`` are missing
    observations):

    1. Pixels with no valid observation stay NaN and are reported with
       ``rank == 0``.
    2. Pixels whose ``T`` passes a Cholesky factorization with a healthy
       pivot ratio are solved exactly through the Cholesky factor.
    3. Remaining pixels are rank deficient (e.g. a date isolated in the
       pair network by missing data). They receive the minimum-norm
       least-squares solution via a batched eigendecomposition with the
       relative cutoff ``rcond``; eigenvalues below ``rcond`` times the
       largest are treated as unconstrained directions.

    Floating-point Cholesky can slip through near-zero pivots, so the
    fast path additionally requires the smallest-to-largest ratio of the
    factor diagonal to exceed ``1e-6``; pixels failing only this screen
    are still solved correctly by the minimum-norm path.

    .. note::
        Missing data is represented by NaN values in ``d``. Systems with
        no NaN and known full rank are solved by the fast path
        automatically; no separate solver is needed.

    Parameters
    ----------
    G : np.ndarray | torch.Tensor
        model field matrix in shape of (n_im, n_param) or (n_pt, n_im, n_param).
        If G is 3D, the first dimension is the G matrix for each pixel.
    d : np.ndarray | torch.Tensor, (n_im, n_pt) matrix
        data field matrix with shape of (n_im, n_pt).
    dtype : torch.dtype
        The compute dtype for torch.Tensor.
    device : Optional[str | torch.device]
        The compute device for torch.Tensor. If None, use GPU if available,
        otherwise use CPU.
    return_numpy : bool, optional
        Return numpy array if True, otherwise torch tensor, by default True.
    rcond : float, optional
        Relative cutoff for unconstrained directions on the minimum-norm
        path. Default is 1e-10.
    diagnostics : bool, optional
        If True, additionally return the per-pixel
        :class:`LstsqDiagnostics`. Default is False.

    Returns
    -------
    X : np.ndarray | torch.Tensor
        (n_param x n_pt) matrix that minimizes norm(M*(GX - d)), with
        minimum norm on rank-deficient pixels and NaN where a pixel has
        no valid observation.
    diagnostics : LstsqDiagnostics, optional
        Per-pixel ``is_constrained``, ``rank``, ``n_observations``, and
        ``condition_number``, only when ``diagnostics=True``.

    """
    device = parse_device(device)

    G = torch.as_tensor(G, dtype=dtype, device=device)  # noqa: N806
    d = torch.as_tensor(d, dtype=dtype, device=device)
    if d.ndim == 1:
        d = d[:, None]

    d_nan = torch.isnan(d)
    valid = ~d_nan
    values = torch.where(d_nan, torch.zeros((), dtype=dtype, device=device), d)

    n_param = G.shape[-1]
    n_pt = d.shape[-1]
    n_obs = valid.sum(dim=0)

    x = torch.full((n_param, n_pt), torch.nan, dtype=dtype, device=device)
    diag = LstsqDiagnostics(
        is_constrained=torch.zeros((n_param, n_pt), dtype=torch.bool, device=device),
        rank=torch.zeros(n_pt, dtype=torch.int64, device=device),
        n_observations=n_obs.to(torch.int64),
        condition_number=torch.full((n_pt,), torch.nan, dtype=dtype, device=device),
    )

    solvable = torch.nonzero(n_obs > 0).flatten()
    gw = None
    t_mat = None
    rhs = None
    if solvable.numel() > 0:
        m_sel = valid[:, solvable]
        d_sel = values[:, solvable]
        if G.ndim == 2:
            gw = m_sel.T.unsqueeze(2) * G.unsqueeze(0)
        else:
            gw = G[solvable] * m_sel.T.unsqueeze(-1)
        t_mat = torch.matmul(gw.mT, gw)
        rhs = torch.matmul(gw.mT, d_sel.T.unsqueeze(2))

        ell, info = torch.linalg.cholesky_ex(t_mat)
        pivots = torch.diagonal(ell, dim1=1, dim2=2)
        pivot_ratio = torch.where(
            info == 0,
            pivots.amin(dim=1) / pivots.amax(dim=1).clamp_min(torch.finfo(dtype).tiny),
            torch.full((solvable.numel(),), torch.inf, dtype=dtype, device=device),
        )
        spd = (info == 0) & (pivot_ratio > _PIVOT_RATIO_MIN)

        n_sel = solvable.numel()
        x_sel = torch.empty((n_param, n_sel), dtype=dtype, device=device)
        rank_sel = torch.empty(n_sel, dtype=torch.int64, device=device)
        cond_sel = torch.empty(n_sel, dtype=dtype, device=device)
        constrained_sel = torch.ones((n_param, n_sel), dtype=torch.bool, device=device)

        if bool(spd.any()):
            x_spd = torch.cholesky_solve(rhs[spd], ell[spd]).squeeze(-1).T
            x_sel[:, spd] = x_spd
            rank_sel[spd] = n_param
            cond_sel[spd] = pivot_ratio[spd]

        deficient = ~spd
        if bool(deficient.any()):
            w_e, v_e = torch.linalg.eigh(t_mat[deficient])
            lam_max = w_e[:, -1:]
            keep = w_e > lam_max * rcond
            rank_sel[deficient] = keep.sum(dim=1).to(torch.int64)
            lam_min = torch.where(keep, w_e, torch.full_like(w_e, torch.inf)).amin(
                dim=1
            )
            cond_sel[deficient] = lam_max[:, 0] / lam_min
            w_inv = torch.where(keep, 1.0 / w_e, torch.zeros_like(w_e))
            x_def = torch.matmul(
                v_e,
                w_inv.unsqueeze(-1) * torch.matmul(v_e.mT, rhs[deficient]),
            )
            x_sel[:, deficient] = x_def.squeeze(-1).T
            # column mass of the null-space eigenvectors: a parameter j is
            # constrained iff e_j has no projection onto the null space
            null_mass = (v_e * v_e * (~keep).unsqueeze(1)).sum(dim=2)
            constrained_sel[:, deficient] = (null_mass < _NULL_MASS_TOL).T

        x[:, solvable] = x_sel
        diag.rank[solvable] = rank_sel
        diag.condition_number[solvable] = cond_sel
        diag.is_constrained[:, solvable] = constrained_sel

    device_type = device.type
    if device_type != "cpu":
        x_np = x.detach().cpu()
        diag_np = diag.cpu()
        # clear cache manually to avoid memory overflow
        del G, d, values, gw, t_mat, rhs, x, diag
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "mps":
            torch.mps.empty_cache()
        if return_numpy:
            x_np = x_np.numpy()
        if diagnostics:
            return x_np, diag_np.numpy() if return_numpy else diag_np
        return x_np
    if return_numpy:
        x = x.numpy()
    if diagnostics:
        return x, diag.numpy() if return_numpy else diag
    return x


def calculate_u(
    loops: Loops,
    unw_phases: np.ndarray | torch.Tensor,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    return_numpy: bool = True,
) -> NDArray[np.floating] | torch.Tensor:
    """Calculate correction matrix u by loop closure phase using least square.

    More details see paper:

    .. tip::
        The pairs in the loops may be fewer than the input pairs. To make sure
        the pairs in the loops are the same as the pairs in the unw_phases, you
        can use the :meth:`.Pairs.where` method to get the index/mask of the
        pairs in the loops from the input pairs.

    Parameters
    ----------
    loops : Loops
        Loops object. Loops are used to calculate the design matrix, which describes
        the relationship between the loops and pairs.
    unw_phases : ndarray | torch.Tensor
        unwrapped interferograms phases with shape of (n_pair, n_pixel).
    device : Optional[str | torch.device], optional
        device of torch.tensor used for computation. If None, use GPU if
        available, otherwise use CPU.
    dtype : torch.dtype, optional
        dtype of torch.tensor used for computation.
    return_numpy : bool, optional
        Return numpy array if True, otherwise torch tensor, by default True.

    Examples
    --------
    Build the loops from a small pair network, then align the unwrapped
    phases with the pairs present in the loops:

    >>> import numpy as np
    >>> import faninsar as fis
    >>> from faninsar import NSBAS
    >>> names = [
    ...     "20170101_20170113",
    ...     "20170101_20170125",
    ...     "20170101_20170206",
    ...     "20170113_20170125",
    ...     "20170113_20170206",
    ...     "20170125_20170206",
    ... ]
    >>> pairs = fis.Pairs.from_names(names)
    >>> unw = np.arange(12, dtype=np.float64).reshape(6, 2)
    >>> loops = pairs.build_loops(edge_days=12)
    >>> idx = pairs.where(loops.pairs)  # index of the loop pairs among the input pairs
    >>> unw_used = unw[idx]

    calculate u by loops and unwrapped interferometric phases:

    >>> u = np.zeros_like(unw)
    >>> u[idx] = np.round(NSBAS.calculate_u(loops, unw_used))
    >>> u.shape
    (6, 2)

    calculate the corrected interferometric phases:

    >>> unw_c = unw - 2 * np.pi * u
    >>> unw_c.shape
    (6, 2)

    """
    unw_phases = torch.as_tensor(unw_phases, dtype=dtype, device=device)
    C = torch.as_tensor(loops.loop_matrix(), dtype=dtype, device=device)  # noqa: N806

    # edge pairs are not contributing to the loop closure phase, remove them
    # from the matrix C to avoid being involved in the calculation of u
    mask = torch.as_tensor(
        loops.pairs.where(loops.diagonal_pairs), dtype=torch.long, device=device
    )
    Cc = C[:, mask]  # noqa: N806

    closure_phase = torch.mm(C, unw_phases)

    contain_nan = False
    if torch.any(torch.isnan(unw_phases)):
        contain_nan = True

    if contain_nan:
        uc = batch_lstsq(
            Cc,
            closure_phase,
            dtype=dtype,
            device=device,
            tqdm_args={"desc": "  Calculate u"},
            return_numpy=False,
        )
    else:
        uc = torch.linalg.lstsq(Cc, closure_phase).solution

    u = torch.zeros_like(unw_phases)
    u[mask] = uc / (2 * np.pi)
    if return_numpy:
        u = u.cpu().numpy()
    return u
