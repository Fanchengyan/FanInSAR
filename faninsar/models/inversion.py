
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import psutil
from tqdm import tqdm

from faninsar.models.ts_models import TimeSeriesModels
from faninsar.utils.pair_tools import Pairs


class NSBASMatrixFactory:
    '''Factory class for NSBAS matrix. The NSBAS matrix is usually expressed as:
    ``d = Gm``, where ``d`` is the unwrapped interferograms matrix, ``G`` is the
    NSBAS matrix, and ``m`` is the model parameters, which is the combination of
    the deformation increment and the model parameters. see paper: TODO for more
    details.

    .. note::

        After initialization, the ``model`` and ``gamma`` can still be updated 
        by setting the corresponding attributes. The ``G`` and ``d`` will be
        updated automatically.

    Examples
    --------

    '''
    _pairs: Pairs = []
    _model: TimeSeriesModels = None
    _gamma: float = 0.0001
    _G: np.ndarray = None
    _d: np.ndarray = None

    slots = ['_pairs', '_model', '_gamma', '_G', '_d']

    def __init__(
        self,
        unw: np.ndarray,
        pairs: Union[Pairs, Iterable[str]],
        model: TimeSeriesModels,
        gamma: float = 0.0001
    ):
        '''Initialize NSBASMatrixFactory

        Parameters
        ----------
        unw : np.ndarray (n_pairs, n_pixels)
            Unwrapped interferograms matrix
        pairs : Union[Pairs, Iterable[str]]
            Pairs or iterable of pair names
        model : TimeSeriesModels
            Time series model
        gamma : float, optional
            weight for the model component, by default 0.0001
        '''
        if isinstance(pairs, Pairs):
            self._pairs = pairs
        elif isinstance(pairs, Iterable):
            self._pairs = Pairs.from_names(pairs)
        else:
            raise TypeError('pairs must be either Pairs or Iterable')

        self.d = unw
        self.model = model
        self.gamma = gamma

    @property
    def pairs(self):
        '''Return pairs'''
        return self._pairs

    @property
    def model(self):
        '''Return model'''
        return self._model

    @model.setter
    def model(self, model):
        '''Update model and G by input model'''
        if not isinstance(model, TimeSeriesModels):
            raise TypeError('model must be a TimeSeriesModels instance')
        if self._model == model:
            return

        self._model = model
        if hasattr(self, 'gamma'):
            self.G = self._make_nsbas_matrix(model.G_br, self.gamma)

    @property
    def gamma(self):
        '''Return gamma'''
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        '''Update gamma and G by input gamma'''
        if not isinstance(gamma, (float, int)):
            raise TypeError('gamma must be either float or int')
        if gamma == self._gamma:
            return
        if gamma <= 0:
            raise ValueError('gamma must be positive')

        self._gamma = gamma
        if hasattr(self, 'model'):
            self.G = self._make_nsbas_matrix(self.model.G_br, gamma)

    @property
    def d(self):
        '''Return d for NSBAS: d = Gm'''
        return self._d

    @d.setter
    def d(self, unw):
        '''Update d: restructure unw by appending model matrix part'''
        if not isinstance(unw, np.ndarray):
            raise TypeError('d must be a numpy array')
        if len(unw.shape) != 2:
            raise ValueError('d must be a 2D array')
        if unw.shape[0] != len(self.pairs):
            raise ValueError(
                'input unw must have the same rows number as pairs number')

        self._d = self._restructure_unw(unw)

    @property
    def G(self):
        '''Return G for NSBAS: d = Gm'''
        return self._G

    @G.setter
    def G(self, G):
        '''Update G by input G'''
        if not isinstance(G, np.ndarray):
            raise TypeError('G must be a numpy array')
        if G.shape[0] != (len(self.pairs) + len(self.pairs.dates)):
            raise ValueError(
                'G must have the same number of rows as (pairs number + dates number)')

        self._G = G

    def _make_nsbas_matrix(self, G_br, gamma):
        G_br = np.asarray(G_br)
        G_tl = self.pairs.to_matrix()

        if len(G_br.shape) == 1:
            G_br = G_br.reshape(-1, 1)
        n_param = G_br.shape[1]

        n_date = len(self.pairs.dates)
        G_bl = np.tril(np.ones((n_date, n_date-1),
                               dtype=np.float32), k=-1)
        G_b = np.hstack((G_bl, G_br))*gamma
        G_t = np.hstack((G_tl, np.zeros((len(self._pairs), n_param))))
        G = np.vstack((G_t, G_b))

        return G

    def _restructure_unw(self, unw):
        unw = np.vstack((unw, np.zeros((len(self.pairs.dates), unw.shape[1]))))
        return unw


class NSBASInversion:
    '''NSBAS inversion class. The NSBAS inversion is usually expressed as: 
    ``d = Gm``, where ``d`` is the unwrapped interferograms matrix, ``G`` is the
    NSBAS matrix, and ``m`` is the model parameters, which is the combination of
    the deformation increment and the model parameters. see paper: TODO for more
    details.

    Examples
    --------

    '''

    def __init__(
            self,
            matrix_factory: NSBASMatrixFactory,
            gpu_id: Optional[int] = None
    ):
        '''Initialize NSBASInversion

        Parameters
        ----------
        matrix_factory : NSBASMatrixFactory
            NSBASMatrixFactory instance
        gpu_id : int or None, optional
            GPU ID, if None, use CPU, by default None
        '''
        self.matrix_factory = matrix_factory
        self.gpu_id = gpu_id

        self.G = matrix_factory.G
        self.d = matrix_factory.d
        self.n_param = len(matrix_factory.model.param_names)
        self.n_pair = len(matrix_factory.pairs)

    def inverse(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''Calculate increment displacement difference by NSBAS inversion.

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
        '''
        result = batch_lstsq(
            self.G,
            self.d,
            self.gpu_id,
            desc='  NSBAS inversion'
        )

        residual = np.dot(self.G, result) - self.d

        incs = result[:-self.n_param, :]
        params = result[-self.n_param:, :]

        residual_pair = residual[:self.n_pair]
        residual_tsm = residual[self.n_pair:]

        return incs, params, residual_pair, residual_tsm


def device_mem_size(gpu_id):
    if gpu_id is not None:
        import cupy as cp
        mem_gpu_free = cp.cuda.Device(gpu_id).mem_info[0]
        mem_size = int(mem_gpu_free/1024**2)
    else:
        mem_free = psutil.virtual_memory().available
        mem_size = int(mem_free/1024**2)

    return mem_size


def get_patchcol_matric(G, d, mem_size, dtype, safe_factor=2):
    """
    Get patch number of cols for memory size (in MB) for GPU.

    Parameters:
    -----------
    dtype: numpy.dtype
        dtype of ndarray

    Returns:
        patchcol : List of the number of rows for each patch.
                ex) [[0, 1234], [1235, 2469],... ]
    """
    m, n = d.shape
    r = G.shape[-1]

    # rough value of n_patch
    n_patch = int(np.ceil(
        m * n * r**2 * dtype.itemsize * safe_factor
        / 2 ** 20 / mem_size)
    )

    # accurate value of n_patch
    rowspacing = int(n / n_patch)
    n_patch = int(np.ceil(n / rowspacing))

    patchcol = []
    for i in range(n_patch):
        patchcol.append([i * rowspacing, (i + 1) * rowspacing])
        if i == n_patch - 1:
            patchcol[-1][-1] = n

    return patchcol


def batch_lstsq(G, d, gpu_id=None, desc='', unit='patch'):
    '''This function calculates the least-squares solution for a batch of linear equations 
    using the given G matrix and the data in  d , using (optionally) the GPU with 
     gpu_id . 

    Parameters 
    ---------- 
    G : array-like (n_im, n_param) or (n_pt, n_im, n_param)
        A 2D/3D array containing the linear equations 
    d : 2D array (n_im, n_pt)
        2D array containing the data associated with the equations in G 
    gpu_id : int, optional 
        An integer specifying the ID of the GPU to use for accelerated computation 
    desc : string, optional 
        A brief description of the task for the progress bar. If None, no progress bar

    Returns 
    ------- 
    result : 2D array (n_param, n_pt)
        A 2D array containing the least-squares solutions to the equations 
        specified in G 
    '''
    n_pt = d.shape[1]
    n_param = G.shape[1] if G.ndim == 2 else G.shape[2]

    result = np.full((n_param, n_pt), np.nan, dtype=np.float32)

    mem_size = device_mem_size(gpu_id)

    patchcol = get_patchcol_matric(
        G, d, mem_size, np.dtype(np.float64))

    if desc:
        patchcol = tqdm(patchcol, desc=desc, unit=unit)
    for col in patchcol:
        if G.ndim == 2:
            result[:, col[0]:col[1]] = censored_lstsq(
                G, d[:, col[0]:col[1]], gpu_id)
        elif G.ndim == 3:
            result[:, col[0]:col[1]] = censored_lstsq(
                G[col[0]:col[1], :, :], d[:, col[0]:col[1]], gpu_id)
        else:
            raise ValueError('Dimension of G must be 2 or 3')
    return result


def censored_lstsq(G, d, gpu_id=None):
    """Solves least squares problem subject to missing data.
    Reference: http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/

    Note: uses a broadcasted solve for speed.

    Args
    ----
    G (ndarray) : m x r or n x m x r matrix
        model field
    d (ndarray) : m x n matrix
        data field

    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(GX - d))
    """
    if gpu_id is not None:
        import cupy as xp
        xp.cuda.Device(gpu_id).use()
        G = xp.asarray(G)
        d = xp.asarray(d)
    else:
        xp = np

    G = G.astype('f8')
    d = d.astype('f8')

    # set nan values to zero
    d_nan = xp.isnan(d)
    d[d_nan] = 0
    M = ~d_nan

    # get the filter for pixels that could be solved
    m = xp.sum(M, axis=0) > G.shape[-1]

    X = xp.full((G.shape[-1], d.shape[-1]), np.nan, dtype=np.float32)

    if G.ndim == 2:
        rhs = xp.dot(G.T, M[:, m] * d[:, m]).T[:, :, None]  # n x r x 1 tensor
        T = xp.matmul(G.T[None, :, :], M[:, m].T[:, :, None] *
                      G[None, :, :])  # n x r x r tensor
    else:
        rhs = xp.matmul(G[m].transpose(0, 2, 1),
                        (M[:, m] * d[:, m]).T[:, :, None])  # n x r x 1 tensor
        # n x r x r tensor
        T = xp.matmul(G[m].transpose(0, 2, 1), M[:, m].T[:, :, None] * G[m])
    X[:, m] = xp.squeeze(xp.linalg.solve(T, rhs)).T  # transpose to get r x n

    if gpu_id is not None:
        X_np = xp.asnumpy(X)
        G, d, M, d_nan = None, None, None, None
        rhs, T, X = None, None, None
        return X_np
    else:
        return X
