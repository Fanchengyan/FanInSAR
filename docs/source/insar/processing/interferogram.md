# Interferogram Formation

## Intuition

Once two SLCs are coregistered, multiply one by the complex conjugate of the
other. The scatterer's largely stable unknown phase cancels, leaving the change
in propagation geometry plus atmosphere, noise, and processing residuals.

## Cross-multiplication

With reference $s_1=A_1e^{j\phi_1}$ and coregistered secondary
$\tilde{s}_2=A_2e^{j\phi_2}$,

$$
z_{12}=s_1\tilde{s}_2^*=A_1A_2e^{j(\phi_1-\phi_2)}.
$$

The wrapped phase is

$$
\phi_w=\arg(z_{12})=\operatorname{atan2}(\Im z_{12},\Re z_{12}).
$$

Do not subtract `np.angle(s1) - np.angle(s2)` and then average. Complex
cross-multiplication preserves circular phase statistics and amplitude weights.

## Range change and phase

The two-way propagation phase is approximately $-4\pi R/\lambda$. Therefore

$$
\phi_{12}\approx -\frac{4\pi}{\lambda}(R_1-R_2)
=\frac{4\pi}{\lambda}(R_2-R_1),
$$

for $s_1s_2^*$. One full $2\pi$ cycle corresponds to a line-of-sight range
change of $\lambda/2$. For Sentinel-1 C-band this is only a few centimetres,
which explains both InSAR sensitivity and its ambiguity.

## Common-band filtering

Different viewing geometries can shift the range or azimuth spectra. Only the
overlapping spectral band is mutually coherent. A common-band filter retains
that intersection before cross-multiplication:

$$
S'_k(f)=S_k(f)H_{common}(f).
$$

This can improve coherence for larger baselines or Doppler differences at the
cost of resolution. It must be distinguished from Goldstein phase filtering,
which acts on the interferogram after formation.

## When flattening happens

Many processors compute the simulated geometric phase before this step and
apply it during resampling or cross-multiplication:

$$
z_{flat}=s_1\left(\tilde{s}_2e^{-j\phi_{model}}\right)^*
\quad\text{or}\quad
z_{flat}=z_{12}e^{-j\phi_{model}}.
$$

These are algebraically equivalent when sign conventions match. Therefore a
workflow may say "flatten during crossmul" even though a conceptual diagram
places phase correction after interferogram formation.

## Output products

Keep at least:

- complex interferogram, not only wrapped phase;
- reference and secondary power or amplitude support;
- coherence estimate and its window/look definition;
- valid-data and processing masks;
- conjugation order, wavelength, reference/secondary identifiers;
- model phase or range offsets used for flattening.

The complex interferogram is the source of truth. Wrapped phase, magnitude,
coherence, filtered phase, and multilooked variants are derived products.

## Checks

- A self-interferogram has near-zero phase and coherence near one over valid
  support.
- Swapping acquisitions conjugates the interferogram and reverses phase.
- Multiplying either SLC by a constant phase rotates the interferogram by the
  predicted sign.
- Invalid/zero-padded samples do not enter looks or coherence windows.
- The complex dtype and accumulator precision prevent overflow or avoidable
  cancellation.
