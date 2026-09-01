# Sentinel-1 SLC

## Intuition

An SLC pixel is a complex phasor, not a photograph pixel:

$$
s(a,r)=A(a,r)e^{j\phi(a,r)}=I(a,r)+jQ(a,r).
$$

The magnitude $A$ is related to radar backscatter. The phase $\phi$ contains a
large propagation term, focusing/carrier terms, scatterer phase, and noise. An
individual absolute phase is not geophysically useful; the stable phase
difference between two acquisitions is.

## What a Sentinel-1 IW SLC contains

A SAFE product separates information that must be reunited by the reader:

- measurement TIFFs containing complex focused samples;
- annotation XML with burst timing, valid sample masks, Doppler centroid,
  azimuth FM rate, range sampling, orbit state vectors, and geolocation grid;
- calibration and noise XML;
- manifest metadata and polarization/subswath identifiers;
- usually one measurement/annotation set per IW subswath and polarization.

IW TOPS data are divided into bursts. Within a burst the antenna beam sweeps in
azimuth, causing the Doppler centroid and azimuth carrier to vary strongly. The
overlap between adjacent bursts is deliberately observed with different squint
angles and later provides a sensitive azimuth-registration measurement.

## A useful signal model

For acquisition $k$, a simplified focused sample is

$$
s_k(\eta,\tau) = A_k(\eta,\tau)
\exp\!\left[-j\frac{4\pi}{\lambda}R_k(\eta,\tau)
+j\phi_{\mathrm{TOPS},k}(\eta,\tau)
+j\phi_{\mathrm{scat},k}\right]+n_k,
$$

where $\eta$ is azimuth time, $\tau$ is fast/range time, $R_k$ is slant
range, $\phi_{\mathrm{TOPS}}$ is the steering/focusing carrier, and $n_k$ is
noise. Coregistration makes $s_1$ and $s_2$ refer to the same ground scattering
cell; deramping temporarily removes $\phi_{\mathrm{TOPS}}$ so interpolation is
safe.

## Inputs to validate before processing

1. **Acquisition compatibility:** same relative orbit, mode, overlapping bursts,
   compatible polarization, and usable temporal/perpendicular baselines.
2. **Orbit coverage:** state vectors cover the sensing interval with margin.
3. **Timing:** burst start/stop times and valid-sample arrays agree with the TIFF
   dimensions.
4. **Complex convention:** determine I/Q order, dtype scaling, and no-data rules.
5. **Wavelength:** use product metadata; do not silently hard-code C-band.
6. **Radiometry:** decide whether raw DN amplitude, beta-nought, sigma-nought, or
   gamma-nought is required. Phase processing and radiometric calibration are
   related but not interchangeable.

:::{warning}
Never interpolate `np.angle(slc)` directly. The discontinuity between $+\pi$
and $-\pi$ is representational, not physical. Resample the complex I/Q samples
with a phase-preserving kernel.
:::

## Developer invariants

- Store burst identity, acquisition time, polarization, subswath, orbit source,
  radar wavelength, PRF/azimuth interval, starting range, and range spacing.
- Preserve the valid-data mask. Zero-filled burst edges are not observations.
- Keep storage dtype separate from decoded dtype and physical scaling.
- Treat calibration/noise corrections as explicit provenance-bearing steps.
- Retain enough Doppler/FM-rate metadata to reproduce deramp and reramp.

## Handoff

The mission adapter passes the complex samples and metadata above to the shared
[geometry](../../processing/geometry.md) and
[coregistration](../../processing/coregistration.md) stages. Any interpolation
of IW data must also follow the [TOPS processing](tops.md) contract.
