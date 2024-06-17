from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import psutil
import torch
from numpy.typing import NDArray
from tqdm.auto import tqdm

from faninsar._core.device import parse_device
from faninsar._core.pair_tools import Loops, Pairs
from faninsar.NSBAS.tsmodels import TimeSeriesModels


class NSBASMatrixFactory:
    """Factory class to generate/format NSBAS matrix  The NSBAS matrix is usually
    expressed as: ``d = Gm``, where ``d`` is the unwrapped interferograms matrix,
    ``G`` is the NSBAS matrix, and ``m`` is the model parameters, which is the
    combination of the deformation increment and the model parameters.
    see paper: TODO for more details.

    .. note::

        After initialization, the ``d`` can still be updated by assigning a new
        unwrapped interferograms matrix to ``d``. This is useful when the
        unwrapped interferograms is divided into multiple patches and the NSBAS
        matrix is calculated for each patch separately.

    Examples
    --------
    >>> import faninsar as fis
    >>> import numpy as np

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
    >>> unw = np.random.randint(0, 255, (len(pairs),5))
    >>> model = fis.AnnualSinusoidalModel(pairs.dates)
    >>> nsbas_matrix = fis.NSBASMatrixFactory(unw, pairs, model)
    >>> nsbas_matrix
    NSBASMatrixFactory(
        pairs: Pairs(10)
        model: AnnualSinusoidalModel(dates: 6, unit: day)
        gamma: 0.0001
        G shape: (16, 9)
        d shape: (16, 5)
    )

    reset ``d`` by assigning a new unwrapped interferograms matrix with same pairs

    >>> nsbas_matrix.d = np.random.randint(0, 255, (len(pairs), 10))
    NSBASMatrixFactory(
        pairs: Pairs(10)
        model: AnnualSinusoidalModel(dates: 6, unit: day)
        gamma: 0.0001
        G shape: (16, 9)
        d shape: (16, 10)
    )
    """

    _pairs: Pairs
    _model: Optional[TimeSeriesModels]
    _gamma: float
    _G: NDArray[np.float32]
    _d: NDArray[np.float32 | np.float64]

    slots = ["_pairs", "_model", "_gamma", "_G", "_d"]

    def __init__(
        self,
        unw: NDArray[np.floating],
        pairs: Pairs | Sequence[str],
        model: Optional[TimeSeriesModels] = None,
        gamma: float = 0.0001,
    ):
        """Initialize NSBASMatrixFactory

        Parameters
        ----------
        unw : NDArray (n_pairs, n_pixels)
            Unwrapped interferograms matrix
        pairs : Pairs | Sequence[str]
            Pairs or Sequence of pair names
        model : Optional[TimeSeriesModels], optional
            Time series model. If None, generate SBAS matrix rather than NSBAS
            matrix, by default None.
        gamma : float, optional
            weight for the model component, by default 0.0001. This parameter
            will be ignored if model is None.
        """
        if isinstance(pairs, Pairs):
            self._pairs = pairs
        elif isinstance(pairs, Sequence):
            self._pairs = Pairs.from_names(pairs)
        else:
            raise TypeError("pairs must be either Pairs or Sequence")

        if isinstance(unw, np.ma.MaskedArray):
            unw = unw.filled(np.nan)

        self._model = None
        self._gamma = None

        if model is not None:
            self._check_model(model)
            self._check_gamma(gamma)
            self._model = model
            self._gamma = gamma
            self.G = self._make_nsbas_matrix(model.G_br, gamma)
        else:
            self.G = self._make_sbas_matrix()
        self.d = unw

    def __str__(self):
        return f"{self.__class__.__name__}(pairs: {self.pairs}, model: {self.model}, gamma: {self.gamma})"

    def __repr__(self):
        _str = (
            f"{self.__class__.__name__}(\n"
            f"    pairs: {self.pairs}\n"
            f"    model: {str(self.model)}\n"
            f"    gamma: {self.gamma}\n"
            f"    G shape: {self.G.shape}\n"
            f"    d shape: {self.d.shape}\n"
            ")"
        )
        return _str

    @property
    def pairs(self) -> Pairs:
        """Return pairs"""
        return self._pairs

    @property
    def model(self) -> Optional[TimeSeriesModels]:
        """Return model"""
        return self._model

    def _check_model(self, model) -> None:
        """Check model"""
        if not isinstance(model, TimeSeriesModels):
            raise TypeError("model must be a TimeSeriesModels instance")

    @property
    def gamma(self) -> float:
        """Return gamma"""
        return self._gamma

    def _check_gamma(self, gamma) -> None:
        """Update gamma and G by input gamma"""
        if not isinstance(gamma, (float, int)):
            raise TypeError("gamma must be either float or int")
        if gamma <= 0:
            raise ValueError("gamma must be positive")

    @property
    def d(self) -> NDArray[np.float32 | np.float64]:
        """Return ``d`` matrix for NSBAS ``d = Gm``"""
        return self._d

    @d.setter
    def d(self, unw):
        """Update d: restructure unw by appending model matrix part"""
        if not isinstance(unw, np.ndarray):
            raise TypeError("d must be a numpy array")
        if len(unw.shape) != 2:
            raise ValueError("d must be a 2D array")
        if unw.shape[0] != len(self.pairs):
            raise ValueError("input unw must have the same rows number as pairs number")

        if self.model is None:
            self._d = unw
        else:
            self._d = self._restructure_unw(unw)

    @property
    def G(self) -> NDArray[np.float32]:
        """Return ``G`` matrix for NSBAS ``d = Gm``"""
        return self._G

    @G.setter
    def G(self, G):
        """Update G by input G"""
        if not isinstance(G, np.ndarray):
            raise TypeError("G must be a numpy array")
        if (self.model is not None) & (
            G.shape[0] != (len(self.pairs) + len(self.pairs.dates))
        ):
            raise ValueError(
                "G must have the same number of rows as (n_pairs + n_dates)"
                " if model is not None."
            )

        self._G = G

    def _make_nsbas_matrix(self, G_br, gamma) -> NDArray[np.float32]:
        G_br = np.asarray(G_br, dtype=np.float32)
        G_tl = self.pairs.to_matrix()

        if len(G_br.shape) == 1:
            G_br = G_br.reshape(-1, 1)
        n_param = G_br.shape[1]

        n_date = len(self.pairs.dates)
        G_bl = np.tril(np.ones((n_date, n_date - 1), dtype=np.float32), k=-1)
        G_b = np.hstack((G_bl, G_br)) * gamma
        G_t = np.hstack((G_tl, np.zeros((len(self._pairs), n_param))))
        G = np.vstack((G_t, G_b))

        return G

    def _make_sbas_matrix(self):
        return self.pairs.to_matrix()

    def _restructure_unw(self, unw) -> NDArray[np.float32 | np.float64]:
        if self.model is not None:
            unw = np.vstack((unw, np.zeros((len(self.pairs.dates), unw.shape[1]))))
        return unw


class NSBASInversion:
    """a class used to operate NSBAS inversion. The NSBAS inversion is usually
    expressed as: ``d = Gm``, where ``d`` is the unwrapped interferograms matrix,
    ``G`` is the NSBAS matrix, and ``m`` is the model parameters, which is the
    combination of the deformation increment and the model parameters.
    see paper: TODO for more details.

    Examples
    --------

    """

    def __init__(
        self,
        matrix_factory: NSBASMatrixFactory,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.float64,
        verbose=True,
    ):
        """Initialize NSBASInversion

        Parameters
        ----------
        matrix_factory : NSBASMatrixFactory
            NSBASMatrixFactory instance
        device : Optional[str | torch.device], optional
            device of torch.tensor used for computation. If None, use GPU if
            available, otherwise use CPU.
        dtype : torch.dtype
            dtype of torch.tensor used for computation.
        """
        self.matrix_factory = matrix_factory
        self.device = parse_device(device)
        self.dtype = dtype
        self.verbose = verbose

        self.G = matrix_factory.G
        self.d = matrix_factory.d
        self.n_param = len(matrix_factory.model.param_names)
        self.n_pair = len(matrix_factory.pairs)

    def __str__(self):
        return f"{self.__class__.__name__}()"

    def inverse(
        self,
    ) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """Calculate increment displacement difference by NSBAS inversion.

        Returns
        -------
        incs: np.ndarray (n_date - 1, n_pt)
            Incremental displacement
        params: np.ndarray (n_param, n_pt)
            parameters of model in NSBAS inversion
        residual_pair: np.ndarray (n_pair, n_pt)
            residual between interferograms and model result
        residual_tsm: np.ndarray (n_date, n_pt)
            residual between time-series model and model result
        """
        result = batch_lstsq(
            self.G,
            self.d,
            dtype=self.dtype,
            device=self.device,
            verbose=self.verbose,
            tqdm_args={"desc": "  NSBAS inversion"},
        )
        residual = self.d - np.dot(self.G, result)

        incs = result[: -self.n_param, :]
        params = result[-self.n_param :, :]

        residual_pair = residual[: self.n_pair]
        residual_tsm = residual[self.n_pair :]

        return incs, params, residual_pair, residual_tsm


class PhaseDeformationConverter:
    """A class to convert between phase and deformation (mm) for SAR interferometry."""

    def __init__(self, frequency: float = None, wavelength: float = None) -> None:
        """Initialize the converter. Either wavelength or frequency should be provided.
        If both are provided, wavelength will be recalculated by frequency.

        Parameters
        ----------
        frequency : float
            The frequency of the radar signal. Unit: GHz.
        wavelength : float
            The wavelength of the radar signal. Unit: meter.
            this parameter will be ignored if frequency is provided.
        """
        speed_of_light = 299792458

        if frequency is not None:
            frequency = frequency * 1e9  # GHz to Hz
            self.wavelength = speed_of_light / frequency  # meter
            self.frequency = frequency
        elif wavelength is not None:
            self.wavelength = wavelength
            self.frequency = speed_of_light / wavelength
        else:
            raise ValueError("Either wavelength or frequency should be provided.")

        # convert radian to mm
        self.coef_rd2mm = -self.wavelength / 4 / np.pi * 1000

    def __str__(self) -> str:
        return f"PhaseDeformationConverter(wavelength={self.wavelength})"

    def __repr__(self) -> str:
        return str(self)

    def phase2deformation(self, phase: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert phase to deformation (mm)"""
        return phase * self.coef_rd2mm

    def deformation2phase(
        self, deformation: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Convert deformation (mm) to phase (radian)"""
        return deformation / self.coef_rd2mm

    def wrap_phase(self, phase: NDArray[np.floating]) -> NDArray[np.floating]:
        """Wrap phase to [0, 2π]"""
        return np.mod(phase, 2 * np.pi)


def device_mem_size(device: Optional[str | torch.device]) -> int:
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


def _get_patch_col(G, d, mem_size, dtype, safe_factor=2):
    """
    Get patch number of cols for memory size (in MB) for SBAS inversion.

    Parameters:
    -----------
    dtype: numpy.dtype or torch.dtype
        dtype of ndarray

    Returns:
        patch_col : List of the number of rows for each patch.
                ex) [[0, 1234], [1235, 2469],... ]
    """
    m, n = d.shape
    r = G.shape[-1]

    # rough value of n_patch
    n_patch = int(
        np.ceil(
            m
            * n
            * r**2
            * torch.tensor([], dtype=dtype).element_size()
            * safe_factor
            / 2**20
            / mem_size
        )
    )

    # accurate value of n_patch
    row_spacing = int(np.ceil(n / n_patch))
    n_patch = int(np.ceil(n / row_spacing))

    patch_col = []
    for i in range(n_patch):
        patch_col.append([i * row_spacing, (i + 1) * row_spacing])
        if i == n_patch - 1:
            patch_col[-1][-1] = n

    return patch_col


def batch_lstsq(
    G: np.ndarray | torch.Tensor,
    d: np.ndarray | torch.Tensor,
    dtype: torch.dtype = torch.float64,
    device: Optional[str | torch.device] = None,
    verbose: bool = True,
    tqdm_args: dict = {},
    return_numpy: bool = True,
) -> NDArray[np.floating] | torch.Tensor:
    """This function calculates the least-squares solution for a batch of linear
    equations using the given G matrix and the data in d.

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
        If True, return a numpy array, otherwise return a torch tensor.

    Returns
    -------
    X : np.ndarray | torch.Tensor
        (n_im x n_pt) matrix that minimizes norm(M*(GX - d)). If return_numpy is
        True, return a numpy array, otherwise return a torch tensor.
    """
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
        if G.ndim == 2:
            result[:, col[0] : col[1]] = censored_lstsq(
                G, d[:, col[0] : col[1]], dtype, device, return_numpy=False
            )
        elif G.ndim == 3:
            result[:, col[0] : col[1]] = censored_lstsq(
                G[col[0] : col[1], :, :],
                d[:, col[0] : col[1]],
                dtype,
                device,
                return_numpy=False,
            )
        else:
            raise ValueError("Dimension of G must be 2 or 3")
    if return_numpy:
        result = result.cpu().numpy()
    return result


def censored_lstsq(
    G: np.ndarray | torch.Tensor,
    d: np.ndarray | torch.Tensor,
    dtype: torch.dtype = torch.float64,
    device: Optional[str | torch.device] = None,
    return_numpy: bool = True,
) -> NDArray[np.floating] | torch.Tensor:
    """Solves least squares problem subject to missing data.
    Reference: http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/

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
        data field matrix.
    dtype : torch.dtype
        dtype of torch.tensor used for computation.
    device : Optional[str | torch.device]
        device of torch.tensor used for computation. If None, use GPU if
        available, otherwise use CPU.

    Returns
    -------
    X : np.ndarray | torch.Tensor
        (n_im x n_pt) matrix that minimizes norm(M*(GX - d)). If return_numpy is
        True, return a numpy array, otherwise return a torch tensor.
    """
    device = parse_device(device)

    G = torch.tensor(G, dtype=dtype, device=device)
    d = torch.tensor(d, dtype=dtype, device=device)

    # set nan values to zero
    d_nan = torch.isnan(d)
    d[d_nan] = 0
    M = ~d_nan

    # get the filter for pixels that could be solved
    m = torch.sum(M, axis=0) > G.shape[-1]

    X = torch.full((G.shape[-1], d.shape[-1]), torch.nan, dtype=dtype, device=device)

    if G.ndim == 2:
        rhs = torch.matmul(G.T, M[:, m] * d[:, m]).T[:, :, None]  # n x r x 1 tensor
        T = torch.matmul(
            G.T[None, :, :], M[:, m].T[:, :, None] * G[None, :, :]
        )  # n x r x r tensor
    else:
        rhs = torch.matmul(
            G[m].transpose(0, 2, 1), (M[:, m] * d[:, m]).T[:, :, None]
        )  # n x r x 1 tensor
        # n x r x r tensor
        T = torch.matmul(G[m].transpose(0, 2, 1), M[:, m].T[:, :, None] * G[m])

    X[:, m] = torch.squeeze(
        torch.linalg.solve(T, rhs), dim=2
    ).T  # transpose to get r x n

    device_type = device.type
    if device_type != "cpu":
        X_np = X.detach().cpu()
        G, d, M, d_nan = None, None, None, None
        rhs, T, X = None, None, None
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "mps":
            torch.mps.empty_cache()
        if return_numpy:
            X_np = X_np.numpy()
        return X_np
    else:
        if return_numpy:
            return X.numpy()
        return X


def calculate_u(
    loops: Loops,
    unw_phases: np.ndarray,
    device: Optional[str | torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> NDArray[np.floating]:
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
    unw_phases : ndarray
        unwrapped interferograms phases with shape of (n_pair, n_pixel).
    device : Optional[str | torch.device], optional
        device of torch.tensor used for computation. If None, use GPU if
        available, otherwise use CPU.
    dtype : torch.dtype, optional
        dtype of torch.tensor used for computation.

    Examples
    --------
    get the loops from the pairs:

    >>> loops = pairs.to_loops()
    >>> idx = pairs.where(loops.pairs) # get the index of the pairs in the loops from the input pairs
    >>> unw_used = unw[idx]

    calculate u by loops and unwrapped interferometric phases:

    >>> u = np.zeros_like(unw, dtype=np.float32)
    >>> u[idx] = np.round(NSBAS.calculate_u(loops, unw_used))
    """
    contain_nan = False
    if np.any(np.isnan(unw_phases)):
        contain_nan = True
    C = loops.to_matrix()

    # edge pairs are not contributing to the loop closure phase, remove them
    # from the matrix C to avoid being involved in the calculation of u
    mask = loops.pairs.where(loops.diagonal_pairs)
    Cc = C[:, mask]

    u = np.zeros_like(unw_phases)

    C = torch.tensor(C, dtype=dtype, device=device)
    unw_phases = torch.tensor(unw_phases, dtype=dtype, device=device)
    Cc = torch.tensor(Cc, dtype=dtype, device=device)

    closure_phase = torch.mm(C, unw_phases)

    if contain_nan:
        _Uc = batch_lstsq(
            Cc,
            closure_phase,
            dtype=dtype,
            device=device,
            tqdm_args={"desc": "  Calculate u"},
        ).numpy()
    else:
        _Uc = torch.linalg.lstsq(Cc, closure_phase).solution.numpy()

    u[mask] = _Uc / (2 * np.pi)
    return u
