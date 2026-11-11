"""Horner evaluation and normalized, rank-aware polynomial surface fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Literal

from faninsar.core.algorithms._validation import (
    _as_tensor,
    _checked_size,
    _execution_device,
    _inspect_array,
    _require_torch,
)
from faninsar.core.algorithms.linalg import least_squares
from faninsar.core.algorithms.linalg._least_squares import (
    _admitted,
    _preflight,
    _validate_options,
)
from faninsar.logging import reject, setup_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

logger = setup_logger(__name__)
_INPUT_LIMIT = 1024**3
_OUTPUT_LIMIT = 256 * 1024**2
_WORKSPACE_LIMIT = 64 * 1024**2
_QUERY_BATCH = 65536


def _inspect_bounded(
    value: object, name: str, *, allow_bool: bool = False
) -> tuple[int, ...]:
    info = _inspect_array(value, name=name, allow_bool=allow_bool)
    _checked_size(info.shape, itemsize=9, name=name, limit_bytes=_INPUT_LIMIT)
    return info.shape


def _flat_mask(
    mask: Tensor | bool | None, shape: tuple[int, ...]
) -> Tensor | bool | None:
    if mask is None or isinstance(mask, bool):
        return mask
    return mask.expand(shape).reshape(-1)


def _chunk_mask(
    mask: Tensor | bool | None, start: int, stop: int
) -> Tensor | bool | None:
    return mask if mask is None or isinstance(mask, bool) else mask[start:stop]


def polyval(coefficients: object, x: object) -> Tensor:
    """Evaluate nonempty ascending 1-D coefficients by float64 Horner arithmetic.

    Parameters
    ----------
    coefficients : numpy.ndarray or torch.Tensor
        Real float32/float64 coefficients ``[c0,c1,...]``. Any nonfinite or
        native-masked coefficient makes the entire polynomial unavailable.
    x : numpy.ndarray or torch.Tensor
        Real evaluation coordinates of arbitrary shape. Nonfinite/native-masked
        queries produce NaN only at those query positions.

    Returns
    -------
    torch.Tensor
        Float64 values shaped like x, on the common CPU/CUDA input device.

    Raises
    ------
    ValueError
        For empty/non-1-D coefficients, unsupported devices, gradients, or
        impractical allocations.
    TypeError
        For unsupported containers or dtypes.

    Notes
    -----
    Source arrays/masks are not mutated. Queries are promoted in bounded chunks;
    no explicit-power evaluation or NumPy numerical executor is used.

    """
    device = _execution_device(coefficients, x)
    c_shape = _inspect_bounded(coefficients, "coefficients")
    x_shape = _inspect_bounded(x, "x")
    if len(c_shape) != 1 or not c_shape[0]:
        reject(
            "coefficients must be a nonempty one-dimensional array.",
            ValueError,
            logger=logger,
        )
    _checked_size(
        c_shape, itemsize=8, name="coefficient workspace", limit_bytes=_WORKSPACE_LIMIT
    )
    _checked_size(
        x_shape, itemsize=8, name="polynomial output", limit_bytes=_OUTPUT_LIMIT
    )
    torch = _require_torch()
    c, c_mask = _as_tensor(coefficients, device=device, name="coefficients")
    query, query_mask = _as_tensor(x, device=device, name="x")
    output = torch.full(x_shape, torch.nan, dtype=torch.float64, device=device)
    if not bool(_admitted(c, c_mask).all()):
        return output
    c = c.to(dtype=torch.float64)
    query = query.reshape(-1)
    query_mask = _flat_mask(query_mask, x_shape)
    flat_output = output.reshape(-1)
    for start in range(0, query.numel(), _QUERY_BATCH):
        stop = start + _QUERY_BATCH
        values = query[start:stop].to(dtype=torch.float64)
        available = _admitted(values, _chunk_mask(query_mask, start, stop))
        safe = torch.where(available, values, 0)
        result = torch.zeros_like(values) + c[-1]
        for coefficient in c[:-1].flip(0):
            result = result * safe + coefficient
        flat_output[start:stop] = torch.where(available, result, torch.nan)
    return output


@dataclass(frozen=True)
class PolynomialSurface:
    """Polynomial value with its fitted index transform and SVD diagnostics.

    Attributes
    ----------
    coefficients : torch.Tensor
        Float64 minimum-norm coefficients, shaped ``(*images,n_terms)``.
    exponents : tuple of tuple of int
        Ordered ``(column_power,row_power)`` terms.
    row_offset, column_offset : float
        Original index midpoint for each fit axis.
    row_scale, column_scale : float
        Multiplier after subtracting the offset. Non-singleton coordinates map
        to [-1,1]; a singleton has offset/scale zero and maps to zero.
    rank, n_observations : torch.Tensor
        Same-device diagnostics shaped like the leading image dimensions.
    is_constrained : torch.Tensor
        Identification flags shaped like coefficients; see LeastSquaresResult.
    condition_number : torch.Tensor
        Actual retained singular-value ratio per image; see LeastSquaresResult.

    """

    coefficients: Tensor
    exponents: tuple[tuple[int, int], ...]
    row_offset: float
    row_scale: float
    column_offset: float
    column_scale: float
    rank: Tensor
    n_observations: Tensor
    is_constrained: Tensor
    condition_number: Tensor

    def evaluate(self, rows: object, cols: object) -> Tensor:
        """Evaluate broadcast index queries using the stored fit transform.

        Parameters
        ----------
        rows, cols : numpy.ndarray or torch.Tensor
            Real float32/float64 row and column indices with broadcastable
            shapes. Coordinates may extend outside the original fitting grid.
            Native masks/nonfinite values exclude the corresponding query.

        Returns
        -------
        torch.Tensor
            Float64 values shaped ``(*images,*broadcast_query_shape)`` on the
            fitted device. Unavailable coefficients invalidate that image;
            unavailable coordinates invalidate only their query positions.

        Raises
        ------
        ValueError
            For incompatible shapes/devices or allocations exceeding limits.
        TypeError
            For unsupported coordinate containers/dtypes.

        """
        device = _execution_device(self.coefficients, rows, cols)
        r_shape = _inspect_bounded(rows, "rows")
        c_shape = _inspect_bounded(cols, "cols")
        torch = _require_torch()
        try:
            query_shape = tuple(torch.broadcast_shapes(r_shape, c_shape))
        except RuntimeError as exc:
            reject(
                "rows and cols must broadcast.", ValueError, logger=logger, cause=exc
            )
        leading = tuple(self.coefficients.shape[:-1])
        _checked_size(
            (*leading, *query_shape),
            itemsize=8,
            name="surface evaluation output",
            limit_bytes=_OUTPUT_LIMIT,
        )
        # Expanded query copies and masks must also fit before transfer/reshape.
        _checked_size(
            query_shape,
            itemsize=18,
            name="surface query workspace",
            limit_bytes=_OUTPUT_LIMIT,
        )
        r, r_mask = _as_tensor(rows, device=device, name="rows")
        c, c_mask = _as_tensor(cols, device=device, name="cols")
        r = r.expand(query_shape).reshape(-1)
        c = c.expand(query_shape).reshape(-1)
        r_mask = _flat_mask(r_mask, query_shape)
        c_mask = _flat_mask(c_mask, query_shape)
        coefficients = self.coefficients.reshape(-1, len(self.exponents))
        output = torch.full(
            (*leading, *query_shape), torch.nan, dtype=torch.float64, device=device
        )
        flat_output = output.reshape(coefficients.shape[0], r.numel())
        for image, terms in enumerate(coefficients):
            if not bool(torch.isfinite(terms).all()):
                continue
            for start in range(0, r.numel(), _QUERY_BATCH):
                stop = start + _QUERY_BATCH
                rr = r[start:stop].to(dtype=torch.float64)
                cc = c[start:stop].to(dtype=torch.float64)
                available = _admitted(rr, _chunk_mask(r_mask, start, stop))
                available &= _admitted(cc, _chunk_mask(c_mask, start, stop))
                yy = torch.where(available, (rr - self.row_offset) * self.row_scale, 0)
                xx = torch.where(
                    available, (cc - self.column_offset) * self.column_scale, 0
                )
                result = torch.zeros_like(xx)
                for coefficient, (i, j) in zip(terms, self.exponents, strict=True):
                    result = result + coefficient * xx**i * yy**j
                flat_output[image, start:stop] = torch.where(
                    available, result, torch.nan
                )
        return output


def _terms(
    degree: int, exponents: Sequence[tuple[int, int]] | None
) -> tuple[tuple[int, int], ...]:
    if isinstance(degree, bool) or not isinstance(degree, Integral) or degree < 0:
        reject("degree must be a nonnegative integer.", ValueError, logger=logger)
    if exponents is None:
        count = (int(degree) + 1) * (int(degree) + 2) // 2
        _checked_size(
            (count,),
            itemsize=32,
            name="polynomial term metadata",
            limit_bytes=_WORKSPACE_LIMIT,
        )
        return tuple(
            (total - row, row)
            for total in range(degree + 1)
            for row in range(total + 1)
        )
    if degree != 1:
        reject(
            "degree must remain its default when exponents are supplied.",
            ValueError,
            logger=logger,
        )
    if not isinstance(exponents, (tuple, list)) or not exponents:
        reject(
            "exponents must be a nonempty sequence of unique pairs.",
            ValueError,
            logger=logger,
        )
    _checked_size(
        (len(exponents),),
        itemsize=32,
        name="polynomial term metadata",
        limit_bytes=_WORKSPACE_LIMIT,
    )
    result = []
    for pair in exponents:
        if (
            not isinstance(pair, (tuple, list))
            or len(pair) != 2
            or any(
                isinstance(power, bool) or not isinstance(power, Integral) or power < 0
                for power in pair
            )
        ):
            reject(
                "exponents require nonnegative integer (column,row) pairs.",
                ValueError,
                logger=logger,
            )
        result.append((int(pair[0]), int(pair[1])))
    if len(set(result)) != len(result):
        reject("exponents must be unique.", ValueError, logger=logger)
    return tuple(result)


def _image(data: Tensor, shape: tuple[int, ...], index: int) -> Tensor:
    """Select leading dimensions as views before flattening a single image."""
    expanded = data.expand(shape)
    indices = []
    for size in reversed(shape[:-2]):
        indices.append(index % size)
        index //= size
    return expanded[tuple(reversed(indices))].reshape(-1)


def _image_mask(
    mask: Tensor | bool | None, shape: tuple[int, ...], index: int
) -> Tensor | bool | None:
    return (
        mask if mask is None or isinstance(mask, bool) else _image(mask, shape, index)
    )


def fit_surface2d(
    source: object,
    *,
    degree: int = 1,
    method: Literal["ols", "wls"] = "ols",
    exponents: Sequence[tuple[int, int]] | None = None,
    weights: object = None,
    valid: object = None,
) -> PolynomialSurface:
    """Fit each image with a normalized total-degree or custom polynomial basis.

    Parameters
    ----------
    source : numpy.ndarray or torch.Tensor
        Real image ``(...,height,width)``; masked/nonfinite samples are omitted.
        Spatial dimensions must be nonempty; empty leading batches are allowed.
    degree : int, default=1
        Nonnegative total degree. Terms are ordered by increasing total degree,
        then increasing row exponent. Ignored combinations are rejected.
    method : {"ols", "wls"}, default="ols"
        Least-squares method; WLS requires weights, OLS forbids them.
    exponents : sequence of (int, int), optional
        Unique nonnegative ``(column_power,row_power)`` terms replacing the
        generated basis. degree must remain its default when supplied.
    weights : real array, optional
        Shape ``(height,width)`` or exactly source.shape. WLS follows
        :func:`faninsar.core.algorithms.linalg.least_squares`.
    valid : Boolean array, optional
        Additional observation exclusion with the same shape rules as weights.

    Returns
    -------
    PolynomialSurface
        Stored normalized-index transform, ordered terms, minimum-norm
        coefficients and least-squares diagnostics per leading image slice.

    Raises
    ------
    ValueError
        For invalid basis/options/shapes/devices, weights, or impractical
        design/SVD allocations.
    TypeError
        For unsupported containers/dtypes.

    Notes
    -----
    Every admitted observation participates: no hidden sampling. The shared
    design is capped at 64 MiB and reused; one image is solved at a time with
    the new SVD solver. No source or native mask is mutated. Legacy fitting and
    time-series routines are not changed or implicitly redirected here.

    """
    _validate_options(method, weights, 1e-10)
    device = _execution_device(source, weights, valid)
    shape = _inspect_bounded(source, "source")
    if len(shape) < 2 or not all(shape[-2:]):
        reject(
            "source requires nonempty height and width dimensions.",
            ValueError,
            logger=logger,
        )
    height, width = shape[-2:]
    batch = math.prod(shape[:-2])
    m = height * width
    # Preflight term count and the design before constructing even its metadata.
    if (
        exponents is None
        and isinstance(degree, Integral)
        and not isinstance(degree, bool)
        and degree >= 0
    ):
        term_count = (int(degree) + 1) * (int(degree) + 2) // 2
    elif isinstance(exponents, (tuple, list)):
        term_count = len(exponents)
    else:
        term_count = 0
    _checked_size(
        (m, term_count),
        itemsize=8,
        name="surface design workspace",
        limit_bytes=_WORKSPACE_LIMIT,
    )
    terms = _terms(degree, exponents)
    n = len(terms)
    _preflight(m, n, batch)
    for name, value in (("valid", valid), ("weights", weights)):
        if value is None:
            continue
        info = _inspect_array(value, name=name, allow_bool=name == "valid")
        if info.shape not in {shape[-2:], shape}:
            reject(
                f"{name} must be spatial or exactly source-shaped.",
                ValueError,
                logger=logger,
            )
        if name == "valid" and str(info.dtype) not in {"bool", "torch.bool"}:
            reject("valid must have Boolean dtype.", TypeError, logger=logger)
        _checked_size(info.shape, itemsize=9, name=name, limit_bytes=_INPUT_LIMIT)
    torch = _require_torch()
    data, native = _as_tensor(source, device=device, name="source")
    v, v_mask = (
        (None, None)
        if valid is None
        else _as_tensor(valid, device=device, name="valid", allow_bool=True)
    )
    w, w_mask = (
        (None, None)
        if weights is None
        else _as_tensor(weights, device=device, name="weights")
    )
    if w is not None:
        weight_shape = tuple(w.shape)
        for image in range(math.prod(weight_shape[:-2])):
            weight = _image(w, weight_shape, image)
            mask = _image_mask(w_mask, weight_shape, image)
            bad = (weight < 0) | ~torch.isfinite(weight)
            if mask is True:
                bad.fill_(False)
            elif mask is not None and mask is not False:
                bad &= ~mask
            if bool(bad.any()):
                reject(
                    "Unmasked weights must be finite and nonnegative.",
                    ValueError,
                    logger=logger,
                )
    row_offset = (height - 1) / 2
    col_offset = (width - 1) / 2
    row_scale = 2 / (height - 1) if height > 1 else 0.0
    col_scale = 2 / (width - 1) if width > 1 else 0.0
    yy = (
        torch.arange(height, dtype=torch.float64, device=device) - row_offset
    ) * row_scale
    xx = (
        torch.arange(width, dtype=torch.float64, device=device) - col_offset
    ) * col_scale
    design = torch.empty((m, n), dtype=torch.float64, device=device)
    for term, (i, j) in enumerate(terms):
        design[:, term] = (yy[:, None] ** j * xx[None, :] ** i).reshape(-1)
    coefficients = torch.empty((batch, n), dtype=torch.float64, device=device)
    ranks = torch.empty(batch, dtype=torch.int64, device=device)
    counts = torch.empty_like(ranks)
    constrained = torch.empty((batch, n), dtype=torch.bool, device=device)
    conditions = torch.empty(batch, dtype=torch.float64, device=device)
    for image in range(batch):
        allowed = torch.ones(m, dtype=torch.bool, device=device)
        source_mask = _image_mask(native, shape, image)
        if source_mask is True:
            allowed.fill_(False)
        elif source_mask is not None and source_mask is not False:
            allowed &= ~source_mask
        if v is not None:
            mask = _image_mask(v_mask, shape, image)
            permitted = _image(v, shape, image)
            allowed &= permitted & _admitted(permitted, mask)
        weight = None
        if w is not None:
            mask = _image_mask(w_mask, shape, image)
            weight = _image(w, shape, image)
            if mask is True:
                weight = torch.zeros_like(weight)
            elif mask is not None and mask is not False:
                weight = torch.where(mask, 0, weight)
        result = least_squares(
            design,
            _image(data, shape, image),
            method=method,
            valid=allowed,
            weights=weight,
        )
        coefficients[image] = result.solution
        ranks[image] = result.rank
        counts[image] = result.n_observations
        constrained[image] = result.is_constrained
        conditions[image] = result.condition_number
    leading = shape[:-2]
    return PolynomialSurface(
        coefficients.reshape(*leading, n),
        terms,
        row_offset,
        row_scale,
        col_offset,
        col_scale,
        ranks.reshape(leading),
        counts.reshape(leading),
        constrained.reshape(*leading, n),
        conditions.reshape(leading),
    )
