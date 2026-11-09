Scientific lineage
==================

Sentinel-1 processing can optionally record a lightweight operation log.
The record names each scientific stage, stores the parameters that
selected it, and tracks phase-state conventions (carrier, residual
application, geometric flattening). It is **not** a content checksum of
SLC, interferogram, offset, LUT, or unwrapped-phase arrays.

Enable recording
----------------

The flag defaults to off. Turn it on only when you want the operation
log::

   from faninsar.stack import StackConfig

   config = StackConfig(
       work_dir="work/",
       record_scientific_lineage=True,
   )

When enabled, each pair keeps an ordered list of operations such as
``deramp``, ``measure_residual_solution``,
``apply_carrier_residual_range_phase``, ``form_interferogram``,
``apply_geometric_phase_flatten``, and
``spatial_unwrap_and_residual_screen``. Duplicate or out-of-order
operations are rejected. Applying a residual solution more than once is
rejected by :class:`~faninsar.processing.geometry.prepared.PhaseState`.

What is recorded
----------------

New records store:

- the operation name and a metadata hash of the operation chain
  (pair id, operation, previous state, parameter JSON);
- scalar parameters such as multilook, Goldstein exponent, residual
  policy, and carrier model;
- phase-state fields: carrier, registration model, geometric phase,
  and residual application identity.

New records **omit** ``input_payload_digest``, ``output_payload_digest``,
and other array-checksum fields. They do not copy SLCs solely to hash
them later, and they do not call ``ndarray.tobytes`` for provenance.

What is not recorded
--------------------

Lineage is not a substitute for scientific validation. Matching metadata
does not prove that pixels are unchanged, and the library does not scan
full arrays to detect same-size local mutation. Explicit reruns use the
existing ``overwrite=True`` / ``replace_existing=True`` controls.

Native geometry qualification identities (for example
:func:`~faninsar.processing.geometry.prepared.geo_grid_identity`) remain
small metadata hashes of grid descriptors. They are not scientific
payload checksums and are not removed by this option.

Legacy records
--------------

Older products may still contain payload-digest fields. Those fields are
accepted when a record is reopened and are otherwise inert: they are not
rehashed, not compared against current arrays, and not treated as proof
of byte identity. New writes do not emit empty strings or dummy
hex-digest placeholders under those names.
