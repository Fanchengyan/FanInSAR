===========================
Core Values and Algorithms
===========================

``faninsar.core`` contains foundational scientific values and contracts.
The developer toolbox :mod:`faninsar.core.algorithms` additionally provides
domain-neutral numerical operations. It does not own SAR interpretation,
CRS transformations, DEM source policy, storage, or execution scheduling.
Import numerical operations from this child package, not from ``faninsar``.

.. toctree::
   :maxdepth: 2

   pairs
   parameters
   baselines
   phase_deformation_converter
   sar
   resampling


Numerical toolbox
=================

The six operations and two mathematical result types have stable imports::

    from faninsar.core.algorithms import (
        LeastSquaresResult,
        PolynomialSurface,
        block_reduce2d,
        filter2d,
        fit_surface2d,
        interpolate2d,
        least_squares,
        polyval,
    )

Operation families are loaded lazily by the toolbox. This is not a claim that
importing the existing root ``faninsar`` package is Torch-free: its existing
imports already load Torch. The toolbox does not introduce a backend registry,
a root-level numerical facade, or imports of processing/runtime orchestration.

Inputs, precision, placement, and validity
-----------------------------------------

* Numerical inputs are real float32/float64 Torch tensors or NumPy arrays,
  including NumPy masked arrays. Boolean arrays are masks, not numerical data.
  Integer/categorical, complex, sparse, and gradient-tracking inputs are not
  supported by this interface. Scalar options do not imply support for nested
  Python lists as image inputs.
* Computation and floating numerical outputs are float64. Outputs are **Torch
  tensors even for NumPy inputs**; integer diagnostics and Boolean validity or
  constraint flags retain their appropriate nonfloating dtypes. Existing domain
  wrappers retain their own return-container and dtype contracts.
* NumPy-only calls execute on CPU. Any Tensor argument fixes the execution
  device, and all Tensor arguments must share the same concrete CPU or CUDA
  device. NumPy companions are adapted to that device. Select CUDA explicitly
  by constructing or moving a Tensor; there is no ``backend`` parameter,
  automatic GPU selection, MPS support, or silent CPU fallback in this toolbox.
* Inputs and native masks are not modified. A masked value, NaN, or either
  infinity is unavailable automatically. Optional ``valid`` is additional
  exclusion: ``valid=True`` means allowed but cannot revive unavailable data.
  NumPy ``mask=True`` has the opposite meaning: unavailable. These conditions
  intersect, rather than replacing one another.
* Nonnegative weights are distinct from validity. Unmasked negative or
  nonfinite weights are rejected; native-masked or zero weights exclude their
  observations. A finite placeholder used internally is not a recovered value.
  Each operation documents how admitted observations determine output support.

CPU and CUDA are the admitted execution devices, not interchangeable evidence.
CPU tests do not qualify CUDA. Actual device-specific correctness, complete-call
performance, memory, and downstream scientific-identity gates are required
before promoting an existing consumer; no blanket CUDA qualification or speedup
is implied here. The new APIs do not remove unrelated legacy MPS behavior.

Coordinate interpolation
------------------------

.. code-block:: python

    interpolate2d(source, rows, cols, *, method="bilinear", valid=None,
                  boundary="nan", return_valid=False)

``source`` has shape ``(H, W)`` or ``(..., H, W)``. Broadcast ``rows`` and
``cols`` define query shape ``Q``, shared across leading image slices; the
output has shape ``(..., *Q)``. Source validity is spatial ``(H, W)`` or the
exact source shape. ``return_valid=True`` returns ``(values, valid_output)``
with matching shape and device. Empty queries are supported.

Coordinates are array sample indices: integer ``i`` denotes sample ``i``.
No physical pixel-origin or half-pixel shift is inferred. Nonfinite queries
are invalid. The method determines the required source support:

* ``nearest`` uses ``floor(coordinate + 0.5)``, including negative ties.
* ``bilinear`` requires every nonzero-weight contributing tap to be available;
  exact zero-weight taps do not invalidate a value. Missing neighbors are not
  renormalized.
* ``biquintic`` is the established six-sample **natural spline**, applied along
  columns and then rows. It requires the entire finite, unmasked 6x6 patch at
  offsets ``floor(coordinate) - 1`` through ``floor(coordinate) + 4``, even at
  integer queries. It is not a generic degree-five polynomial or B-spline.

With ``boundary="nan"``, missing support returns NaN/false. For nearest and
bilinear, ``boundary="clamp"`` clamps continuous coordinates to the source
center domain; it does not repair invalid coordinates or source values.
Biquintic rejects clamp and requires a source large enough for its full stencil.
Signed spline weights never justify renormalizing incomplete support.

DEM adapters retain their existing inverse-affine convention, fixed natural
spline, cross-tile window assembly, asymmetric halo, and scientific identity.
This low-level method selector is **not** a new DEM interpolation selector.

.. code-block:: python

    import numpy as np
    import torch
    from faninsar.core.algorithms import interpolate2d

    source = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float64)
    values, supported = interpolate2d(
        source, np.array([0.5]), np.array([0.5]), return_valid=True
    )
    assert values.dtype == torch.float64 and values.device.type == "cpu"
    torch.testing.assert_close(values, torch.tensor([4.0], dtype=torch.float64))
    assert supported.tolist() == [True]

Block reduction
---------------

.. code-block:: python

    block_reduce2d(source, block_shape, *, method="mean", valid=None,
                   invalid="propagate", remainder="drop", return_valid=False)

Reduce independent, origin-aligned blocks over the final two axes using
``mean``, ``sum``, ``min``, or ``max``. ``remainder="drop"`` uses floor output
sizes; ``partial`` uses ceil sizes and actual available members; ``error``
rejects nondivisible input. An empty dropped output is well defined.

``invalid="propagate"`` invalidates any block containing an unavailable member.
``omit`` reduces only finite, unmasked, allowed members. Every all-invalid
block returns NaN/false, **including sum**. ``return_valid=True`` returns the
values and their Boolean admission mask. SAR look origins, cross-products,
power/coherence estimators, and block-center interpretation remain domain-owned.
A legacy finite-mean adapter must explicitly select omit and drop semantics.

.. code-block:: python

    from faninsar.core.algorithms import block_reduce2d

    means = block_reduce2d(
        np.array([[1.0, np.nan], [3.0, 5.0]]),
        (2, 2), invalid="omit", remainder="drop",
    )
    torch.testing.assert_close(means, torch.tensor([[3.0]], dtype=torch.float64))

Nonnegative local smoothing
---------------------------

.. code-block:: python

    filter2d(source, *, method, window=None, sigma=None, min_radius=None,
             weights=None, valid=None, boundary="constant", return_valid=False)

``method="boxcar"`` requires a positive odd ``(height, width)`` window and no
sigma or explicit minimum radius. ``method="gaussian"`` requires finite,
strictly positive ``(sigma_y, sigma_x)`` and no window. Unsupported parameter
combinations raise rather than silently ignoring arguments.

Gaussian support is computed independently per axis::

    resolved_min_radius = 1 if min_radius is None else min_radius
    radius = max(resolved_min_radius, floor(3 * sigma + 0.5))
    kernel_length = 2 * radius + 1

``min_radius`` is Gaussian-only and accepts a nonnegative integer, not a
Boolean, fraction, or per-axis tuple. **None resolves to one**, preserving the
legacy positive-sigma minimum support. A domain adapter that selects support
through this public API must explicitly pass ``min_radius=1``. The existing
inverse-variance adapter instead constructs its legacy minimum-one or identity
kernels and delegates their normalized convolution to the shared private leaf.
Standalone users can pass zero to *permit*, not force, a single-tap axis at small sigma;
ordinary larger sigma still determines the natural radius. Zero minimum does
not permit zero or negative sigma. An explicit minimum with boxcar is rejected.
This is a truncation policy, not an assertion that minimum radius one is
universally superior or required by ISCE3.

Filtering computes normalized nonnegative convolution ``K*(w*x) / (K*w)``.
Invalid data has zero effective weight; default weights are one on available
samples. Weight shape is spatial ``(H, W)`` or the exact source shape.
Constant boundary adds neither values nor support. Reflect means **no repeated
edge**, corresponding to SciPy ``mirror``, not SciPy ``reflect``, including
repeated extension of singleton axes. No support gives NaN/false.

Unlike a center-mask-preserving domain filter, this operation may fill an
invalid center when neighbors contribute positive support. If both radii are
zero, finite allowed unit-weight values remain unchanged and an invalid center
stays invalid. At extremely small positive sigma, off-center weights may
underflow to zero even with radius one; validity follows actual positive
numerical support. Anisotropic sigma can yield a single-tap axis in only one
direction. Independent leading image slices do not share support.

.. code-block:: python

    from faninsar.core.algorithms import filter2d

    image = np.ones((3, 3), dtype=np.float64)
    image[1, 1] = np.nan
    single, single_valid = filter2d(
        image, method="gaussian", sigma=(0.16, 0.16), min_radius=0,
        return_valid=True,
    )
    legacy, legacy_valid = filter2d(
        image, method="gaussian", sigma=(0.16, 0.16), min_radius=1,
        return_valid=True,
    )
    assert not bool(single_valid[1, 1])
    assert bool(legacy_valid[1, 1])
    torch.testing.assert_close(legacy[1, 1], torch.tensor(1.0, dtype=torch.float64))

Existing smoothing wrappers still own sigma-nonpositive identity axes,
negative-weight clamping, output dtype, original-center mask policy, and any
complex behavior. Chunked consumers must carry the actual halo and trim it;
a chunk edge is not an image boundary. Complex interferogram filters and
Goldstein are not replaced by this real-only operation.

Polynomial evaluation and surface fitting
-----------------------------------------

.. code-block:: python

    polyval(coefficients, x)
    fit_surface2d(source, *, degree=1, method="ols", exponents=None,
                  weights=None, valid=None)

``polyval`` evaluates a nonempty one-dimensional ascending coefficient array
``c0 + c1*x + ...`` with Horner's rule, returning a Tensor shaped like ``x``.
A masked or nonfinite coefficient invalidates the whole polynomial; an
unavailable query invalidates only that query. Coefficients are never omitted.
Domain tuple coefficients need explicit array conversion. There is no inferred
polynomial batch dimension.

``fit_surface2d`` fits each leading image slice independently and returns a
:class:`~faninsar.core.algorithms.PolynomialSurface`. The default basis contains
all terms of total degree ``column_power + row_power <= degree``, ordered by
increasing total degree and then increasing row power. Custom ``exponents``
are unique nonnegative ``(column_power, row_power)`` pairs; when provided,
``degree`` must remain its default. The existing detrending selected-power
basis is not equivalent to every same-degree total-degree basis.

The fit normalizes row/column indices to [-1, 1], with singleton axes at zero.
``PolynomialSurface`` stores ``coefficients``, ordered ``exponents``, the
``row_offset``/``row_scale`` and ``column_offset``/``column_scale`` transform,
``rank``, ``n_observations``, ``is_constrained``, and ``condition_number``.
Its ``evaluate(rows, cols)`` method applies that stored transform, including
outside the fit mask. Source missing observations are omitted; no hidden
subsampling occurs. A rank-deficient fit
reports diagnostics instead of implying every coefficient is identified.
Subtracting an evaluated surface must not overwrite original missing data.

``method="ols"`` is unweighted and rejects weights; ``wls`` requires
nonnegative weights and uses square-root row scaling as in least squares.
The existing domain wrappers retain their selected bases, explicit-power
arithmetic order, dtype, masks, and sample policy while delegating Tensor math
to private compatibility leaves in ``fitting._compat``. These adapters do not
silently adopt Horner evaluation or the public fit's different rank semantics.
The specific NumPy explicit-power evaluator remains an independent reference;
preservation does not require entire mixed NumPy/Tensor modules to be unchanged.

.. code-block:: python

    from faninsar.core.algorithms import fit_surface2d, polyval

    x = np.array([0.0, 1.0, 2.0])
    evaluated = polyval(np.array([1.0, 2.0, 3.0]), x)
    torch.testing.assert_close(
        evaluated, torch.tensor([1.0, 6.0, 17.0], dtype=torch.float64)
    )
    rows, cols = np.indices((3, 4), dtype=np.float64)
    plane = 1.0 + 2.0 * cols - rows
    surface = fit_surface2d(plane, degree=1)
    torch.testing.assert_close(surface.evaluate(rows, cols), torch.from_numpy(plane))

Missing-observation least squares
---------------------------------

.. code-block:: python

    least_squares(A, B, *, method="ols", valid=None, weights=None, rcond=1e-10)

This general matrix solver returns a
:class:`~faninsar.core.algorithms.LeastSquaresResult`. It is not an SBAS/NSBAS
model, outlier-robust regression, or Tobit censored-likelihood estimator.
Missing observations are independent of method selection. OLS rejects weights;
WLS requires them and minimizes ``sum(w * residual**2)`` using ``sqrt(w)`` row
scaling. Finite large residuals are not automatically excluded.

The shared design shape is ``(m, n)`` or per-RHS ``(k, m, n)``. Observations
have shape ``(m, k)`` or a single RHS ``(m,)``. Validity and weights accept
``(m,)`` or the exact observation shape. For each RHS, retain an equation only
when its complete design row and observation are finite/unmasked, its optional
valid flag allows it, and its WLS weight is available and positive. A bad shared
design row excludes that row for all RHS; a bad observation excludes only its
own RHS. There is **no intersection of all observation-column masks** and no
zero imputation of missing design coefficients.

The result exposes solution, rank, observation count, parameter constraint
flags, and condition diagnostics on the same device. A single RHS restores
solution shape ``(n,)`` and scalar system diagnostics; multiple RHS give
``(n, k)`` solutions/constraint flags and ``(k,)`` system diagnostics.
Rank retains singular values strictly greater than ``rcond * largest``.
``is_constrained`` flags describe identifiable coordinate directions, not
merely a finite minimum-norm solution: squared projection onto the structural
or discarded null space must be at most ``1e-10``. ``condition_number`` is
the actual largest/smallest retained singular-value ratio, not the legacy
Cholesky pivot proxy. In deficient systems it describes only the retained
subspace.

No-observation systems return NaN solutions, zero rank/count, false constraint
flags, and NaN condition. Observed rank-zero systems instead have zero
minimum-norm solutions and infinite condition. Rank-deficient systems retain
their minimum-norm solution with explicit identification diagnostics. The
legacy time-series solver's Gram-eigenvalue cutoff, pivot screen, containers,
and condition proxies remain a distinct contract implemented in a private
compatibility leaf, not an alias of this public SVD solver. A numerical ``rcond``
cannot be copied blindly between eigenvalue and singular-value thresholds.
No new ``Network.invert`` selector is added. Reusing shared-design/mask groups
can accelerate public least squares, but does not guarantee faster execution
for arbitrary distinct masks or numerical equivalence to the legacy Gram path.

.. code-block:: python

    from faninsar.core.algorithms import least_squares

    design = np.ones((2, 1), dtype=np.float64)
    observed = np.array([[2.0, np.nan], [np.nan, 4.0]])
    result = least_squares(design, observed)
    torch.testing.assert_close(
        result.solution, torch.tensor([[2.0, 4.0]], dtype=torch.float64)
    )
    assert result.n_observations.tolist() == [1, 1]

Shared arithmetic and domain adapters
------------------------------------

The public toolbox contracts above are not replacements for every legacy
wrapper contract. Existing domain import paths and result types remain
meaningful; compatible Tensor arithmetic is owned by private core leaves:

* ``smooth_inverse_variance`` delegates to
  ``filtering._local._normalized_convolution`` after preserving its value/weight
  adaptation and legacy Gaussian/identity kernels. ``multilook_real_torch``
  delegates to ``reduction._block._complete_block_mean`` after its existing
  runtime device, precision, and crop handling. The corresponding public core
  operations also use these leaves on matching profiles; their stricter input
  and invalidity rules still apply at the public boundary.
* Geometry's Torch weights, weighted natural spline, and DEM sampling adapters
  call ``sampling._spline_six``. Physical-coordinate conversion remains in the
  geometry adapter. The public biquintic implementation retains its recurrence
  path; native C++/CUDA solvers and DEM selection are not replaced.
* Polynomial domain adapters call ``fitting._compat`` for their selected-basis
  fit/evaluation, weighted total-degree surface, and explicit-power evaluation.
  They preserve the domain policy instead of pretending those operations are
  identical to the public default basis and Horner arithmetic.
* ``timeseries.solver.censored_lstsq`` delegates historical Tensor arithmetic to
  ``linalg._censored_compat._censored_lstsq_kernel``. The domain wrapper still
  owns device/dtype adaptation, ``LstsqResult``, optional NumPy conversion, and
  device-cache handling. Its B-NaN exclusion, Gram cutoff, pivot and condition
  diagnostics are preserved rather than changed to the new public SVD policy.

Only specific historical NumPy definitions are preserved as independent
references. Tensor math may move out of a module that also contains NumPy code;
there is no whole-old-module byte-identity requirement. These private leaves
are not extra public exports. Passing CPU or compile-capture checks does not
qualify CUDA, native/public-dispatch, or every downstream scientific product.

Operation and result reference
------------------------------

.. automodule:: faninsar.core.algorithms
   :members: interpolate2d, block_reduce2d, filter2d, polyval, fit_surface2d, least_squares, PolynomialSurface, LeastSquaresResult
   :undoc-members:
   :show-inheritance:
