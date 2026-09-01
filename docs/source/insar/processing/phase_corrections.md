# Phase Corrections

## Intuition

Deformation is usually much smaller than the phase caused by the two satellite
positions and terrain height. "Flattening" subtracts the deterministic phase
predicted by orbit and reference geometry. Differential InSAR also removes the
phase predicted from a DEM so the remainder is easier to filter and unwrap.

## Exact range-based model

For ground point $\mathbf{x}$ derived from the reference radar sample and DEM,
compute modeled ranges to both acquisitions:

$$
R_1=\|\mathbf{x}-\mathbf{s}_1(t_1)\|,\qquad
R_2=\|\mathbf{x}-\mathbf{s}_2(t_2)\|.
$$

For $z_{12}=s_1s_2^*$, the modeled geometric phase is

$$
\phi_{geom}=\frac{4\pi}{\lambda}(R_2-R_1).
$$

Remove it on the unit circle:

$$
z_{corr}=z_{12}\exp(-j\phi_{geom}).
$$

Implementations often compute $R_2-R_1$ indirectly from a geo2rdr range-offset
raster. The sign depends on whether the stored offset is secondary-minus-
reference and on conjugation order; a synthetic geometry test is mandatory.

## Flat Earth versus topography

**Flat-Earth phase** is the smooth phase caused by reference-ellipsoid geometry
and baseline. **Topographic phase** is the extra phase caused by surface height
relative to that reference. Software may model them in one exact range
calculation or as two corrections.

For small height perturbations, the topographic sensitivity is approximately

$$
\frac{\partial\phi}{\partial h}
\approx -\frac{4\pi}{\lambda}
\frac{B_\perp}{R\sin\theta},
$$

where $B_\perp$ is perpendicular baseline and $\theta$ is incidence angle. The
height of ambiguity, the elevation change producing $2\pi$, is

$$
h_{2\pi}=\frac{\lambda R\sin\theta}{2|B_\perp|}.
$$

A large $h_{2\pi}$ means weak topographic sensitivity; a small value means DEM
error creates strong residual fringes.

## DEM error after correction

If the DEM height error is $\delta h$, the residual phase is approximately

$$
\delta\phi_{DEM}\approx
-\frac{4\pi}{\lambda}
\frac{B_\perp}{R\sin\theta}\delta h.
$$

Across a stack this term correlates with perpendicular baseline. That
correlation is a powerful diagnostic and can be estimated in time-series
processing, but it does not excuse inconsistent DEM datums in pair processing.

## Other corrections that are not "topographic phase"

- orbit ramps from ephemeris error;
- tropospheric stratification and turbulent delay;
- ionospheric dispersive phase, especially at longer wavelengths;
- solid-Earth tides and ocean loading;
- azimuth carrier/coregistration residuals;
- reference-point and long-wavelength detrending.

Keep these terms separately named and separately recorded. A generic polynomial
"detrend" may remove deformation together with orbit or atmosphere.

## Quality-control gates

- A zero-height/ellipsoid simulation removes the broad flat-Earth ramp.
- DEM-correlated fringes decrease after topographic correction.
- The correction reverses sign when acquisition order is swapped.
- A synthetic $\delta h$ produces phase consistent with the local height
  sensitivity and height of ambiguity.
- No correction is evaluated outside converged geometry/DEM support.
- The model phase is stored or reproducible from provenance.

## Common mistakes

- Subtracting wrapped phase as ordinary real numbers instead of complex phasors.
- Using orthometric DEM height as ellipsoidal height.
- Applying the correction twice because both resampling and crossmul offer a
  `flatten` option.
- Treating a visually flat result as proof: an incorrect sign can sometimes be
  hidden by later detrending or filtering.
