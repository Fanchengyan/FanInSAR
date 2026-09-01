# Geometry, Orbits, and DEMs

## Intuition

Coregistration is first a geometry problem. Given a radar sample's sensing time
and slant range, find the ground point observed by the satellite. Given that
ground point and the secondary orbit, find where it appears in the secondary
SLC. A DEM closes the problem by supplying surface height.

## The range-Doppler equations

Let $\mathbf{s}(t)$ and $\mathbf{v}(t)$ be satellite position and velocity, and
let $\mathbf{x}$ be a target position in Earth-centered coordinates. A radar
observation satisfies the range equation

$$
\|\mathbf{x}-\mathbf{s}(t)\|=R
$$

and the Doppler equation

$$
f_D(t,R)=-\frac{2}{\lambda}
\frac{\mathbf{v}(t)\cdot[\mathbf{x}-\mathbf{s}(t)]}
{\|\mathbf{x}-\mathbf{s}(t)\|}.
$$

The target must also lie on the DEM/ellipsoid surface,

$$
h(\mathbf{x})=h_{\mathrm{DEM}}(\varphi,\lambda_g),
$$

where $\varphi$ and $\lambda_g$ are geodetic latitude and longitude. Solving
these equations from $(t,R)$ to $\mathbf{x}$ is **rdr2geo/topo**. Solving from
$\mathbf{x}$ to $(t,R)$ is **geo2rdr**.

## Why orbit quality matters

An orbit error changes the predicted range and azimuth position and leaves a
long-wavelength phase ramp. Use the best orbit class supported by the mission
and record the exact source, coverage, and processing time; "precise orbit" is
not enough provenance. Sentinel-1, for example, normally prefers POEORB when
available and uses RESORB for rapid processing.

Orbit state vectors are interpolated to arbitrary sensing times. Polynomial or
Hermite interpolation is common, but interpolation order is less important
than adequate temporal support, consistent reference frames, and avoiding
extrapolation.

## Why the DEM datum matters

Radar geometry requires ellipsoidal height. Many DEMs store orthometric height
$H$ relative to a geoid. Convert using

$$
h_{\mathrm{ellipsoid}} = H_{\mathrm{orthometric}} + N,
$$

where $N$ is geoid undulation in a declared vertical datum. A silent EGM96,
EGM2008, or ellipsoid mismatch creates range errors and DEM-correlated phase.

The DEM also controls layover/shadow geometry and simulated topographic phase.
Its pixel spacing, interpolation kernel, void handling, and acquisition date
therefore belong in provenance.

## Geometry products worth materializing

A reusable geometry layer commonly contains:

- latitude, longitude, and ellipsoidal height;
- incidence angle and local look vector;
- reference and secondary slant range;
- secondary range and azimuth offsets on the reference grid;
- layover/shadow/water/validity masks;
- perpendicular and parallel baseline estimates.

## Quality checks

- The solver converges for nearly all valid reference pixels.
- The mapped secondary coordinates lie inside valid acquisition support.
- Offset fields vary smoothly except at known data boundaries.
- DEM heights and geolocation-grid heights use the same vertical datum.
- Forward and inverse transforms approximately close:
  $\operatorname{geo2rdr}(\operatorname{rdr2geo}(t,R))\approx(t,R)$.
- Baseline magnitude and sign are consistent with independent metadata or a
  second implementation.

## Software notes

ISCE2 and ISCE3 expose explicit `topo/rdr2geo` and `geo2rdr` stages. GMTSAR
uses orbit/DEM tools such as `SAT_llt2rat` and radar-coordinate topography.
InSAR.dev's Sentinel-1 preprocessing computes inverse transforms and can fuse
alignment with the radar-to-geographic remap. Detailed call chains appear in
the [software implementation guides](../software/index.md).
