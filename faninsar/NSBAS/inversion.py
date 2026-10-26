"""NSBAS inversion module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import psutil
import torch
from tqdm import tqdm

from faninsar import Loops, Pairs, parse_device
from faninsar.logging import setup_logger
from faninsar.NSBAS.coherence_weight import WeightFunc, coherence_to_weight
from faninsar.NSBAS.tsmodels import TimeSeriesModels

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch._tensor import Tensor


logger = setup_logger(__name__)


class NSBASSolver:
    """Solver class to build NSBAS matrix and perform inversion.

    The NSBAS inversion is expressed as: ``d = Gm``, where ``d`` is the
    unwrapped interferograms matrix, ``G`` is the NSBAS matrix, and ``m`` is
    the model parameters, which is the combination of the deformation increment
    and the model parameters.
    see following paper for more details: `https://ens.hal.science/hal-02185213`_ ,
    `https://www.sciencedirect.com/science/article/pii/S0924271625000772`_


    .. note::

        After initialization, the ``d`` can still be updated by assigning a new
        unwrapped interferograms matrix to ``d``. This is useful when the
        unwrapped interferograms is divided into multiple patches and the NSBAS
        matrix is calculated for each patch separately.

    Examples
    --------
    >>> import numpy as np
    >>> import faninsar as fis
    >>> from faninsar import NSBAS

    >>> names = ['20170111_20170204',
                '20170111_20170222',
                '20170111_20170318',
                '20170204_20170222',
                '20170204_20170318',
                '20170204_20170330',
                '20170222_20170318',
                '20170222_20170330',
                '20170222_20170411',
                '20170318_20170330']

    >>> pairs = fis.Pairs.from_names(names)
    >>> unw = np.random.randint(0, 255, (len(pairs), 5))
    >>> model = NSBAS.AnnualSinusoidalModel(pairs.dates)
    >>> solver = NSBAS.NSBASSolver(unw, pairs, model)
    >>> solver
    NSBASSolver(
        pairs: Pairs(10)
        model: AnnualSinusoidalModel(dates: 6, unit: day)
        gamma: 0.0001
        G shape: (16, 9)
        d shape: (16, 5)
    )


    Now we can perform NSBAS inversion to get the incremental deformation,
    model parameters, and residuals:

    >>> incs, params, residual_pair, residual_tsm = solver.inverse(return_numpy=True)

    reset ``d`` by assigning a new unwrapped interferograms matrix with same pairs

    >>> solver.set_d(np.random.randint(0, 255, (len(pairs), 10)))
    >>> solver
    NSBASSolver(
        pairs: Pairs(10)
        model: AnnualSinusoidalModel(dates: 6, unit: day)
        gamma: 0.0001
        G shape: (16, 9)
        d shape: (16, 10)
    )


    """

    _pairs: Pairs
    _model: TimeSeriesModels | None
    _gamma: float
    _G: torch.Tensor
    _d: torch.Tensor
    _unw: np.ndarray
    _coh: np.ndarray | None
    _coh_threshold: float
    _weight_func: str
    _n_looks: int
    _inv_alg: Literal["ls", "wls"]
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
        "_inv_alg",
        "_model",
        "_n_looks",
        "_pairs",
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
        inv_alg: Literal["ls", "wls"] | None = None,
        weight_func: WeightFunc | str | None = None,
        n_looks: int | None = None,
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
        inv_alg : {"ls", "wls"}, optional
            ``None`` keeps ordinary least squares. ``wls`` selects weighted
            least squares. If ``weight_func`` is still unset, ``wls`` uses
            Fisher-information weights.
        weight_func : {"no", "coh", "fim", "var"}, optional
            Observation-weight definition. ``None`` keeps the unweighted
            default. ``coh`` uses coherence, ``fim`` uses
            ``2 L γ² / (1 - γ²)``, and ``var`` uses the inverse DS phase
            variance.
        n_looks : int, optional
            Independent looks ``L`` used by ``fim`` and ``var``. ``None``
            keeps the internal default of ``1``.
        gamma : float, optional
            Weight of the NSBAS model-constraint rows.
        device, dtype, verbose
            Torch device, compute dtype, and progress-bar flag.

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

        self._model = None
        self._gamma = 0.0001
        self._unw = _to_numpy(unw)
        self._coh = None if coh is None else _to_numpy(coh)
        self._coh_threshold = 0.4 if coh_threshold is None else coh_threshold
        self._inv_alg = "ls" if inv_alg is None else inv_alg
        self._n_looks = 1 if n_looks is None else n_looks
        self._weight_func = "no" if weight_func is None else str(weight_func)
        if self._inv_alg == "wls" and self._weight_func in {"no", "sbas", "uniform"}:
            self._weight_func = "fim"

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
            f"    G shape: {self.G.shape}\n"
            f"    d shape: {self.d.shape}\n"
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
    def d(self) -> torch.Tensor:
        """Return ``d`` matrix for NSBAS ``d = Gm``."""
        return self._d

    def set_d(self, unw: np.ndarray | torch.Tensor) -> None:
        """Set unwrapped interferograms and rebuild ``G`` and ``d``.

        The input is the raw unwrapped phases. Any current coherence mask
        and observation weights are reapplied.
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
        g_sbas = self._make_sbas_matrix().astype(np.float64, copy=False)
        observations = np.array(self._unw, dtype=np.float64, copy=True)
        if observations.shape[0] != len(self.pairs):
            msg = "unwrapped interferograms must have the same rows as pairs"
            logger.error(
                msg,
                extra={"rows": observations.shape[0], "pairs": len(self.pairs)},
            )
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
            if self._weight_func not in {"no", "sbas", "uniform"}:
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

    @overload
    def inverse(
        self,
        return_numpy: bool = True,
        *,
        coh: NDArray[np.floating] | torch.Tensor | None = None,
        coh_threshold: float | None = None,
        inv_alg: Literal["ls", "wls"] | None = None,
        weight_func: WeightFunc | str | None = None,
        n_looks: int | None = None,
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
        inv_alg: Literal["ls", "wls"] | None = None,
        weight_func: WeightFunc | str | None = None,
        n_looks: int | None = None,
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
        inv_alg: Literal["ls", "wls"] | None = None,
        weight_func: WeightFunc | str | None = None,
        n_looks: int | None = None,
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Calculate increment displacement difference by NSBAS inversion.

        Parameters
        ----------
        return_numpy : bool, optional
            Return numpy arrays if True, otherwise torch tensors, by default True.
        coh : ndarray or torch.Tensor, optional
            Spatial coherence used for masking and weighting. When provided,
            it replaces the coherence stored at construction time.
        coh_threshold : float, optional
            Override the coherence mask threshold.
        inv_alg : {"ls", "wls"}, optional
            Override the inversion algorithm. ``wls`` with the default
            ``weight_func="no"`` selects Fisher-information weights.
        weight_func : {"no", "coh", "fim", "var"}, optional
            Override the observation-weight definition.
        n_looks : int, optional
            Override the independent-looks value used by ``fim`` and ``var``.

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
        if inv_alg is not None:
            self._inv_alg = inv_alg
            rebuild = True
        if weight_func is not None:
            self._weight_func = str(weight_func)
            rebuild = True
        if self._inv_alg == "wls" and self._weight_func in {"no", "sbas", "uniform"}:
            self._weight_func = "fim"
            rebuild = True
        if rebuild:
            self._rebuild_system()

        n_param = len(self.model.param_names) if self.model is not None else 0
        n_pair = len(self.pairs)
        result = batch_lstsq(
            self.G,
            self.d,
            dtype=self.dtype,
            device=self.device,
            verbose=self.verbose,
            tqdm_args={"desc": "  NSBAS inversion"},
            return_numpy=False,
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
) -> np.ndarray | torch.Tensor:
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

    Returns
    -------
    x : np.ndarray | torch.Tensor
        (n_im x n_pt) matrix that minimizes norm(M*(GX - d)). If return_numpy is
        True, return a numpy array, otherwise return a torch tensor.

    """
    if tqdm_args is None:
        tqdm_args = {}
    tqdm_args.setdefault("desc", "Batch least-squares")
    tqdm_args.setdefault("unit", "Batch")
    n_pt = d.shape[1]
    n_param = G.shape[1] if G.ndim == 2 else G.shape[2]
    device = parse_device(device)

    result = torch.full((n_param, n_pt), torch.nan, dtype=dtype)
    mem_size = device_mem_size(device)
    patch_col = _get_patch_col(G, d, mem_size, dtype)
    if verbose:
        patch_col = tqdm(patch_col, **tqdm_args)

    for col in patch_col:
        c0, c1 = int(col[0]), int(col[1])
        if G.ndim == 2:
            result[:, c0:c1] = censored_lstsq(
                G=G,
                d=d[:, c0:c1],
                dtype=dtype,
                device=device,
                return_numpy=False,
            )
        elif G.ndim == 3:
            result[:, c0:c1] = censored_lstsq(
                G=G[c0:c1, :, :],
                d=d[:, c0:c1],
                dtype=dtype,
                device=device,
                return_numpy=False,
            )
        else:
            msg = "Dimension of G must be 2 or 3"
            raise ValueError(msg)
    if return_numpy:
        result = result.cpu().numpy()
    return result


@overload
def censored_lstsq(
    G: np.ndarray | torch.Tensor,
    d: np.ndarray | torch.Tensor,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
    return_numpy: bool = True,
) -> np.ndarray: ...
@overload
def censored_lstsq(
    G: np.ndarray | torch.Tensor,
    d: np.ndarray | torch.Tensor,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
    return_numpy: bool = False,
) -> torch.Tensor: ...
def censored_lstsq(
    G: np.ndarray | torch.Tensor,
    d: np.ndarray | torch.Tensor,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
    return_numpy: bool = True,
) -> np.ndarray | torch.Tensor:
    """Solves least squares problem subject to missing data.

    Reference: http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/.

    .. note::
        This function is used for solving the least squares problem with **missing
        data**. The missing data is represented by nan values in the data matrix
        ``d``. If there are no nan values in d, you are recommended to use
        :func:`torch.linalg.lstsq` instead.

    Parameters
    ----------
    G : np.ndarray | torch.Tensor,
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

    Returns
    -------
    X : np.ndarray | torch.Tensor
        (n_im x n_pt) matrix that minimizes norm(M*(GX - d)).

    """
    device = parse_device(device)

    G = torch.as_tensor(G, dtype=dtype, device=device)  # noqa: N806
    d = torch.as_tensor(d, dtype=dtype, device=device)

    # set nan values to zero for calculation, and get the mask for valid pixels
    d_nan = torch.isnan(d)
    d[d_nan] = 0
    M = ~d_nan  # noqa: N806

    # simplest way to detect rank deficiency caused by missing data
    m = torch.sum(M, dim=0) > G.shape[-1]

    x = torch.full((G.shape[-1], d.shape[-1]), torch.nan, dtype=dtype, device=device)
    if G.ndim == 2:
        rhs = torch.matmul(G.T, M[:, m] * d[:, m]).T[:, :, None]  # n x r x 1 tensor
        T = torch.matmul(  # noqa: N806
            G.T[None, :, :],
            M[:, m].T[:, :, None] * G[None, :, :],
        )  # n x r x r tensor
    else:
        rhs = torch.matmul(
            G[m].permute(0, 2, 1),
            (M[:, m] * d[:, m]).T[:, :, None],
        )  # n x r x 1 tensor
        # n x r x r tensor
        T = torch.matmul(G[m].permute(0, 2, 1), M[:, m].T[:, :, None] * G[m])  # noqa: N806

    # transpose to get r x n
    x[:, m] = torch.squeeze(torch.linalg.solve(T, rhs), dim=2).T

    device_type = device.type
    if device_type != "cpu":
        x_np = x.detach().cpu()
        # clear cache manually to avoid memory overflow
        del G, d, M, m, d_nan, T, rhs, x
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "mps":
            torch.mps.empty_cache()
        if return_numpy:
            x_np = x_np.numpy()
        return x_np
    if return_numpy:
        return x.numpy()
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
    get the loops from the pairs:

    >>> loops = pairs.build_loops()
    >>> idx = pairs.where(
    ...     loops.pairs
    ... )  # get the index of the pairs in the loops from the input pairs
    >>> unw_used = unw[idx]

    calculate u by loops and unwrapped interferometric phases:

    >>> u = np.zeros_like(unw, dtype=np.float32)
    >>> u[idx] = np.round(NSBAS.calculate_u(loops, unw_used))

    calculate the corrected interferometric phases

    >>> unw_c = unw - 2 * np.pi * u

    """
    C = loops.loop_matrix()  # noqa: N806

    # edge pairs are not contributing to the loop closure phase, remove them
    # from the matrix C to avoid being involved in the calculation of u
    mask = loops.pairs.where(loops.diagonal_pairs)
    Cc = C[:, mask]  # noqa: N806

    C = torch.as_tensor(C, dtype=dtype, device=device)  # noqa: N806
    unw_phases = torch.as_tensor(unw_phases, dtype=dtype, device=device)
    Cc = torch.as_tensor(Cc, dtype=dtype, device=device)  # noqa: N806

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
