from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import psutil
import xarray as xr
from netCDF4 import Dataset
from tqdm import tqdm


class Uncertainty:
    """
    A class for representing uncertainty in a model.

    Attributes:
    -----------
        variance (numpy.ndarray): A 2D array of variances for each pixel and parameter.

    Examples:
    ---------

    >>> import numpy as np
    >>> from uncertainty_lib import Uncertainty

    >>> # Create an instance of the Uncertainty class and Set the variance attribute
    >>> uncertainty1 = Uncertainty()
    >>> uncertainty1.variance = np.array([[1, 2, 3, 4]])

    >>> # Create another instance of the Uncertainty class Set the variance attribute
    >>> uncertainty2 = Uncertainty()
    >>> uncertainty2.variance = np.array([[5], [6], [7], [8]])

    >>> # Add the two instances of the Uncertainty class together
    >>> new_uncertainty = uncertainty1 + uncertainty2

    >>> # Print the result
    >>> print(new_uncertainty)
    >>> Uncertainty(variance=
    >>>  [[ 6  7  8  9]
    >>>  [ 7  8  9 10]
    >>>  [ 8  9 10 11]
    >>>  [ 9 10 11 12]])
    """

    def __init__(
        self,
        variance: np.ndarray,
    ) -> None:
        """Initializes the Uncertainty class.

        Parameters
        ----------
        variance : numpy.ndarray (n_param, n_pixel)
            A 2D array of variances for each pixel and parameter.
        """
        if not isinstance(variance, np.ndarray):
            raise TypeError("variance must be a numpy array.")
        if variance.ndim != 2:
            raise ValueError("variance must be 2D array (n_param, n_pixel)")

        self._variance = variance

    @property
    def variance(self):
        """
        Gets the variance attribute.

        Returns:
            numpy.ndarray: The variance attribute.
        """
        return self._variance

    def __repr__(self) -> str:
        return f"Uncertainty(variance=\n{self.variance.shape})"

    def __str__(self) -> str:
        return self.__repr__()

    def __add__(self, other: "Uncertainty") -> "Uncertainty":
        """
        Adds two Uncertainty objects together.

        Parameters
        ----------
        other :  Uncertainty
            The other Uncertainty object to add.

        Returns
        -------
        new_var : Uncertainty
            A new Uncertainty object with the sum of the variances.
        """
        if not isinstance(other, Uncertainty):
            raise TypeError("other must be a Uncertainty object.")
        if self.variance is None or other.variance is None:
            raise ValueError("Warning: variance is None. The result will be None.")

        new_var = self.variance + other.variance

        return Uncertainty(new_var)

    @property
    def shape(self) -> tuple[int, int]:
        """
        Gets the shape of the variance attribute.

        Returns:
            tuple: The shape of the variance attribute.
        """
        return self.variance.shape


class ReferencePointsUncertainty(Uncertainty):
    """This class is used to compute the uncertainty (variance) of reference points."""

    def __init__(self, ref_dfm: np.ndarray) -> None:
        """Initializes the ReferencePointsUncertainty class.

        Parameters
        ----------
        ref_dfm : np.ndarray (n_img, n_ref)
            The deformation time series of reference points.
        """

        self.dfm = ref_dfm
        variance, covariance = self._deformation2variance()
        self._covariance = covariance

        super().__init__(variance)

    def _deformation2variance(self):
        """derives the variance from the deformation of the reference points."""
        covariance = np.cov(self.dfm)
        variance = np.diag(covariance)

        covariance = covariance
        variance = variance[np.newaxis, :]
        return variance, covariance

    @property
    def covariance(self):
        """
        Gets the covariance attribute.

        Returns:
            numpy.ndarray: The covariance attribute.
        """
        return self._covariance


class UncertaintyPropagation:
    """This class is used to propagate the uncertainty from the data to the model parameters."""

    @staticmethod
    def weight_from_variance(variance):
        """This function calculates the weight from the variance of data (var_d).

        Parameters
        ----------
        variance: array
            variance of data (sum of the squares of the residuals)

        Returns:
        --------
        W: array
            weight from the variance of data
        """
        W = 1 / np.sqrt(variance)
        return W

    @staticmethod
    def data2param_simplified(G, var_data):
        """This function computes the variance of parameters (var_model)
        from the variance of data (var_d) which is the sum of the
        squares of the residuals and is a constant for each pixel.

        The var_param is calculated by the following equation:
            $Cov(m) = \sigma^2 * (G^T * G)^{-1}$

        Parameters:
        -----------
        G: 2D array (n_im, n_param)
            design matrix
        var_data: 1D array (n_pixel) or 2D array (n_pixel, 1)
            variance of data (sum of the squares of the residuals)

        Returns:
        --------
        var_param: 2D array (n_pixel, n_param)
            variance of model parameters
        """
        if var_data.ndim == 1:
            var_data = var_data[:, None]

        A = np.linalg.inv(G.T @ G)
        var_param = var_data * np.diag(A)[None, :]

        return var_param

    @staticmethod
    def data2param_weighted_simplified(G, W):
        """This function calculates the variances of the model parameters
        given a design matrix G and a vector of weights W. This only works
        when the weight is expressed as 1/σ, where σ^2 is variance. The
        var_param is calculated by the following equation:
            Cov(m) =(G_w^T * G_w)^{-1}

        Parameters:
        -----------
        G: 2D array (n_im, n_param)
            design matrix
        W: 1D array (n_pixel, n_im)
            weight matrix

        Returns:
        --------
        var_param: np.ndarray (1, n_param)
            variance of model parameters
        """
        G_w = W[:, :, None] * G[None, :, :]
        A = np.linalg.inv(G_w.transpose(0, 2, 1) @ G_w)
        var_param = np.diagonal(A, axis1=1, axis2=2)

        return var_param

    @staticmethod
    def data2param(
        G, var_data, weighted=False, W=None, desc="  Data to model variance"
    ):
        """This function is used to calculate the variance of model parameter
        (var_param) from a given variance of data (var_d). if the var_d is derived
        from weighted least squares method, then weight matrix W is required.

        Parameters:
        ------------
        G: 2D array (n_im, n_param)
            design matrix
        var_data: 2D array (n_pixel, n_im)
            deformation variance matrix
        weighted: bool
            if True, means that weighted least squares method was used to calculate parameters variance matrix. Default is False.
        W: 1D array (n_im)
            weight matrix. If weighted is True and W is None, then W is set by variance. Default is None.
        desc: str
            description of progress bar

        Returns:
        --------
        var_param: np.ndarray (n_pixel, n_param)
            variance of model parameters
        """
        n_im, n_param = G.shape
        n_pixel = var_data.shape[0]

        # remove all nan pixels in var_data
        m = (~np.isnan(var_data)).sum(axis=1) > 0
        var_data = var_data[m]

        # set nan to zero to avoid all nan in var_param
        var_data[np.isnan(var_data)] = 0

        C = np.eye(n_im, dtype=np.float32)
        if not weighted:
            M = np.linalg.inv(G.T @ G) @ (G.T)

        # write empty variance into file to give a structure
        var_param = np.full((n_pixel, n_param), np.nan, dtype=np.float32)
        var_param_m = var_param[m]

        patch = get_var_patch(n_pixel, n_im, n_param, np.dtype(np.float64))
        for col in tqdm(patch, desc=desc, unit="pixels"):
            start, end = col[0], col[1]
            var_data_i = var_data[start:end, :, None]
            cov_data = C[None, :, :].repeat((var_data_i.shape[0]), axis=0) * var_data_i

            if weighted:
                if W is None:
                    C = np.linalg.inv(cov_data)
                else:
                    C = np.linalg.inv(np.diag((1 / W) ** 2))
                M = np.linalg.inv(G.T @ C @ G) @ (G.T) @ C

            cov_model = M[None, :, :] @ cov_data @ M.T[None, :, :]

            var_param_m[start:end, :] = np.diagonal(cov_model, axis1=1, axis2=2)

        var_param[m] = var_param_m
        return var_param

    @classmethod
    def data2param_sequence(
        Gs: list,
        var_data: np.ndarray,
    ):
        """This function is used to calculate the variance of model parameter for a sequence of design matrices.

        Parameters
        ----------
        Gs : list
            A list of design matrices.
        var_data : np.ndarray
            The deformation variance matrix.

        Returns
        -------
        var_param : np.ndarray
            The variance of model parameters for the last design matrix in the sequence.
        """
        pass

    @staticmethod
    def data2param_file(
        G,
        var_data,
        var_param_file,
        weighted=False,
        W=None,
        desc="  Computing model variance",
    ):
        """This function is used to calculate the variance of  model parameter
        (var_param) from a given variance of data (var_d). if the var_d is derived
        from weighted least squares method, then weight matrix W is required. The
        var_param is written to a netCDF file.

        Parameters:
        ------------
        G: 2D array (n_im, n_param)
            design matrix
        var_data: 2D array (n_pixel, n_im)
            deformation variance matrix
        var_param_file: pathlib.Path object
            path to output var_param file
        weighted: bool
            if True, means that weighted least squares method was used to calculate parameters variance matrix. Default is False.
        W: 1D array (n_im)
            weight matrix. If weighted is True and W is None, then W is set by variance. Default is None.
        desc: str
            description of progress bar

        Returns:
        --------
        var_param: np.ndarray (n_pixel, n_param)
            variance of model parameters
        """
        n_im, n_param = G.shape
        n_pixel = var_data.shape[0]

        # remove all nan pixels in var_data
        m = (~np.isnan(var_data)).sum(axis=1) == 0

        # set nan to zero to avoid all nan in var_param
        var_data[np.isnan(var_data)] = 0

        C = np.eye(n_im, dtype=np.float32)
        if not weighted:
            M = np.linalg.inv(G.T @ G) @ (G.T)

        # write empty variance into file to give a structure
        var_param = np.full((n_pixel, n_param), np.nan, dtype=np.float32)

        safe_remove(var_param_file)
        (
            xr.Dataset({"variance": (["pixels", "n_param"], var_param)}).to_netcdf(
                var_param_file
            )
        )

        # write actual variance data into nc file
        ds_var_param = Dataset(var_param_file, "r+")
        var_param = ds_var_param["variance"]

        patch = get_var_patch(n_pixel, n_im, n_param, np.dtype(np.float64))
        for col in tqdm(patch, desc=desc, unit="pixels"):
            start, end = col[0], col[1]
            var_data_i = var_data[start:end, :, None]
            cov_data = C[None, :, :].repeat((var_data_i.shape[0]), axis=0) * var_data_i

            if weighted:
                if W is None:
                    C = np.linalg.inv(cov_data)
                else:
                    C = np.linalg.inv(np.diag((1 / W) ** 2))
                M = np.linalg.inv(G.T @ C @ G) @ (G.T) @ C

            cov_model = M[None, :, :] @ cov_data @ M.T[None, :, :]

            var_param[start:end, :] = np.diagonal(cov_model, axis1=1, axis2=2)
        var_param[m, :] = np.nan
        ds_var_param.close()
        print(f"Saved model variance to {var_param_file}.")

    @staticmethod
    def data_cov2param(G, cov_data, weighted=False, W=None):
        """This function is used to calculate the variance of model parameter
        (var_param) from a given covariance of data (cov_d) for one pixel.
        if the var_d is derived from weighted least squares method, then
        the weight matrix W is required.

        Parameters:
        ------------
        G: 2D array (n_im, n_param)
            design matrix
        cov_data: 2D array (n_im, n_im)
            covariance matrix of data
        weighted: bool
            if True, the var_d is derived from weighted least squares method.
            Default is False.
        W: 1D array (n_im)
            weight matrix. If weighted is True and W is None, then W is set
            by variance. Default is None.

        Returns:
        --------
        var_param: np.ndarray (n_pixel, n_param)
            variance of model parameters
        """
        if not weighted:
            M = np.linalg.inv(G.T @ G) @ (G.T)
        else:
            if W is None:
                C = np.linalg.inv(cov_data)
            else:
                C = np.linalg.inv(np.diag((1 / W) ** 2))
            M = np.linalg.inv(G.T @ C @ G) @ (G.T) @ C

        cov_model = M[:, :] @ cov_data @ M.T[:, :]
        var_param = np.diag(cov_model)

        return var_param


def safe_remove(file):
    """remove file if exists"""
    file = Path(file)
    if file.exists():
        file.unlink()


def get_var_patch(n_pixel, n_im, n_param, dtype):
    """This function divides all pixels into the different patches (n_patch)
    by takeing account of memory will be used and the memory free in the
    system. The pixels of the patch are spaced by the result of dividing n_pixel
    by the number of patches. Lastly, the patch columns are appended to a list
    and returned.

    Parameters:
    -----------
    n_pixel: int
        number of pixels
    n_im: int
        number of images
    n_param: int
        number of parameters of model
    mem_size: int
        memory size (in MB)
    dtype: numpy.dtype
        dtype of ndarray

    Returns:
        patch : List of the number of rows for each patch.
                ex) [[0, 1234], [1235, 2469],... ]
    """
    mem_free = psutil.virtual_memory().available
    mem_size = int(mem_free / 1024**2)

    # a rough number of patch
    n_patch = np.ceil(
        (n_pixel * n_im**2 * n_param * dtype.itemsize) / 1024**2 / mem_size
    )

    # number of pixels for each patch
    pixel_space = int(np.ceil(n_pixel / n_patch))

    # accurate number of patch
    n_patch = int(np.ceil(n_pixel / pixel_space))

    # Divide pixels into different patches
    patch = []
    for i in range(n_patch):
        patch.append([i * pixel_space, (i + 1) * pixel_space])
        # to correct the end value for last patch
        if i == n_patch - 1:
            patch[-1][-1] = n_pixel

    return patch


def data2param(
    G,
    uc_data,
    in_type: Literal["variance", "covariance"] = "variance",
    out_type: Literal["variance", "covariance"] = "variance",
    desc="  Uncertainty[Data -> Model]",
):
    """This function is used to derive the uncertainty of model domain from
    the data domain for the least squares inversion solution.

    Parameters:
    ------------
    G: 2D array (n_img, n_param)
        design matrix
    uc_data: 2D array (n_pixel, n_img) or 3D array (n_pixel, n_img, n_img)
        Uncertainty of data. If in_type is 'variance', then uc_data is 2D array.
        If in_type is 'covariance', then uc_data is 3D array.
    in_type: Literal["variance", "covariance"]
        Type of uncertainty of data. Default is 'variance'.
    out_type: Literal["variance", "covariance"]
        Type of uncertainty of model. Default is 'variance'.
    desc: str
        description of progress bar

    Returns:
    --------
    uc_param: np.ndarray (n_pixel, n_param) or (n_pixel, n_param, n_param)
        Uncertainty of model parameters
    """
    n_im, n_param = G.shape
    n_pixel = uc_data.shape[0]

    # remove pixels contains nan in uc_data
    if in_type == "variance":
        m = (np.isnan(uc_data)).sum(axis=1) == 0
    elif in_type == "covariance":
        m = (np.isnan(uc_data)).sum(axis=(1, 2)) == 0
    else:
        raise ValueError("in_type must be 'variance' or 'covariance'.")
    uc_data = uc_data[m]

    C = np.eye(n_im, dtype=np.float32)
    M = np.linalg.inv(G.T @ G) @ (G.T)

    #
    if out_type == "variance":
        uc_param = np.full((n_pixel, n_param), np.nan, dtype=np.float32)
    elif out_type == "covariance":
        uc_param = np.full((n_pixel, n_param, n_param), np.nan, dtype=np.float32)
    uc_param_m = uc_param[m]

    patch = get_var_patch(n_pixel, n_im, n_param, np.dtype(np.float64))
    for col in tqdm(patch, desc=desc, unit="pixels"):
        start, end = col[0], col[1]
        uc_data_i = uc_data[start:end, :, None]
        if in_type == "variance":
            cov_data = C[None, :, :].repeat((uc_data_i.shape[0]), axis=0) * uc_data_i
        elif in_type == "covariance":
            cov_data = uc_data_i

        cov_model = M[None, :, :] @ cov_data @ M.T[None, :, :]
        if out_type == "variance":
            uc_param_m[start:end, :] = np.diagonal(cov_model, axis1=1, axis2=2)
        elif out_type == "covariance":
            uc_param_m[start:end, :, :] = cov_model

    uc_param[m] = uc_param_m
    return uc_param


def data2param_cov(G, cov_data, desc="  Data to model covariance"):
    """This function is used to calculate the variance of model parameter
    (var_param) from a given covariance of data (cov_d) for one pixel.

    Parameters:
    ------------
    G: 2D array (n_im, n_param)
        design matrix
    cov_data: 2D array (n_pixel, n_im, n_im)
        covariance matrix of data
    desc: str
        description of progress bar

    Returns:
    --------
    var_param: np.ndarray (n_pixel, n_param)
        variance of model parameters
    """
    n_im, n_param = G.shape
    n_pixel = cov_data.shape[0]

    M = np.linalg.inv(G.T @ G) @ (G.T)

    # write empty variance into file to give a structure
    var_param = np.full((n_pixel, n_param), np.nan, dtype=np.float32)

    patch = get_var_patch(n_pixel, n_im, n_param, np.dtype(np.float64))
    for col in tqdm(patch, desc=desc, unit="pixels"):
        start, end = col[0], col[1]
        cov_data_i = cov_data[start:end, :, :]
        cov_model = M[None, :, :] @ cov_data_i @ M.T[None, :, :]
        var_param[start:end, :] = np.diagonal(cov_model, axis1=1, axis2=2)

    return var_param


def data2param_sequence(
    Gs: list,
    var_data: np.ndarray,
    verbose: bool = True,
    desc: str = "  Data to model covariance",
):
    """This function is used to calculate the variance of model parameter for a sequence of design matrices.

    Parameters
    ----------
    Gs : list
        A list of design matrices.
    var_data : np.ndarray
        The deformation variance matrix.
    verbose : bool
        If True, display the progress bar.
    desc : str
        Description of progress bar. Only used if verbose is True.

    Returns
    -------
    var_param : np.ndarray
        The variance of model parameters for the last design matrix in the sequence.
    """
    n_im = Gs[0].shape[0]
    n_param = Gs[-1].shape[1]
    n_pixel = var_data.shape[0]

    if isinstance(var_data, np.ma.MaskedArray):
        var_data = var_data.filled(np.nan)

    # remove all nan pixels in var_data
    m = (~np.isnan(var_data)).sum(axis=1) > 0
    var_data = var_data[m]

    # set nan to zero to avoid all nan in var_param
    var_data[np.isnan(var_data)] = 0

    C = np.eye(n_im, dtype=np.float32)
    M = G2M(Gs[0])

    # write empty variance into file to give a structure
    var_param = np.full((n_pixel, n_param), np.nan, dtype=np.float32)
    var_param_m = var_param[m]

    patch = get_var_patch(n_pixel, n_im, n_param, np.dtype(np.float64))
    if verbose:
        patch = tqdm(patch, desc=desc, unit="patch")
    for col in patch:
        start, end = col[0], col[1]
        var_data_i = var_data[start:end, :, None]
        cov_data = C[None, :, :].repeat((var_data_i.shape[0]), axis=0) * var_data_i

        cov_model = M[None, :, :] @ cov_data @ M.T[None, :, :]
        for G in Gs[1:]:
            M_i = G2M(G)
            cov_model = M_i[None, :, :] @ cov_model @ M_i.T[None, :, :]
        var_param_m[start:end, :] = np.diagonal(cov_model, axis1=1, axis2=2)

        var_param[m] = var_param_m
    return var_param


def G2M(G: np.ndarray):
    M = np.linalg.inv(G.T @ G) @ (G.T)
    return M
