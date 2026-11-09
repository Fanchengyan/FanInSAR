Workspace recovery after a FanInSAR upgrade
============================================

A FanInSAR upgrade that renames a ``StackConfig`` field, changes a
``faninsar:`` payload encoding, or otherwise drifts the ``stack.json``
layout used to make a work directory unreadable. PROPOSAL-0069 splits
recovery into two layers:

- the **identity** layer — ``stack.json`` plus the ``item.json`` metadata —
  is rewritable. :meth:`Stack.rescan <faninsar.stack.session.Stack.rescan>`
  diagnoses what survives and
  :meth:`Stack.repair <faninsar.stack.session.Stack.repair>` rewrites a
  fresh ``faninsar.stack.v1`` identity;
- the **product** layer — the zarr payloads and their ``write_id`` lineage —
  is never rewritten. A product whose upstream ``write_id`` no longer
  matches is reported *stale* by :meth:`Stack.status` and must be
  reprocessed, not repaired in place.

Tolerant ``Stack.open``
-----------------------

:meth:`Stack.open <faninsar.stack.session.Stack.open>` is tolerant. A
``stack.json`` that is readable but drifted still opens a degraded-but-usable
Stack: decodable science fields are recovered, unrecoverable ones are left
at their ``StackConfig`` defaults, and the planned Pair network falls back
to whatever the committed ``ifg/ml_*`` products and their nested
``unwrap/`` siblings imply. Legacy top-level ``unwrap/`` and ``ion/``
trees are not treated as the current products.
Only a *missing* ``stack.json`` still raises — there is nothing to recover
from. ``open`` never rewrites the file; persistence is only through
``repair``::

   stack = S1Stack.open("work/")
   if stack.repair_report is not None:
       print(stack.repair_report.as_dict()["science_dropped"])

``stack.repair_report`` is ``None`` for a clean open and carries the
:class:`~faninsar.stack.repair.RepairReport` for a recovered one; the same
report is exposed under ``stack.status()["identity"]["repair"]``.

Rescan (read-only diagnosis)
----------------------------

``Stack.rescan(work_dir)`` inspects a work directory without writing
anything and returns a :class:`~faninsar.stack.repair.RepairReport`
describing what a repair would recover versus leave defaulted::

   report = S1Stack.rescan("work/")
   print(report.schema_found)         # e.g. "faninsar.stack.v99"
   print(report.science_dropped)      # fields that failed to decode
   print(report.inputs_changed)       # missing or size-changed inputs
   print(report.inferred_pairs)       # pairs rebuilt from committed products

Repair (explicit rewrite)
-------------------------

``Stack.repair(work_dir)`` runs the rescan, rebuilds a ``StackConfig`` from
recovered science plus conservative defaults, reconstructs the catalog and
Pair network from the committed products, writes ``stack.json`` via
``save_stack_identity`` (still ``faninsar.stack.v1`` — no schema bump), then
runs the normal open tail. ``inputs_changed`` is reported but never blocks::

   stack = S1Stack.repair("work/")
   print(stack.status()["identity"]["repair"])

Command line
------------

The ``faninsar stack`` subcommands mirror the Python entry points::

   faninsar stack status work/      # status() report, or a rescan report
   faninsar stack repair work/      # rewrite stack.json, print RepairReport
   faninsar stack repair --dry-run work/   # rescan only, no write
   faninsar stack verify work/      # structural check of committed products

``verify`` inspects required assets, paths, and declared array metadata. It
does not compare historical payload checksums. Local inputs are treated as
stable during a run, and one writer owns each work directory. Concurrent
overwrite of the same product is unsupported.

Explicit reruns
---------------

A full scientific rerun is an explicit caller sequence, not an implicit
overwrite. Keep the default reuse behavior, then replace products in
dependency order with the existing ``overwrite=True`` /
``replace_existing=True`` flags. ``prepare_data`` has no overwrite
argument; it reuses a complete local ready set or continues qualification
with ``on_error``, ``offline``, and ``budget``. Scientific stages that
publish products accept ``overwrite=True``::

   stack.prepare_data()
   stack.coregister_scenes(overwrite=True)
   stack.form_interferograms(overwrite=True)
   stack.unwrap(overwrite=True)

Each requested overwrite publishes a fresh ``write_id``. Downstream products
bound to the previous id become stale until they are also replaced.

What repair cannot fix
----------------------

Repair is deliberately narrow. It does **not** migrate ``item.json``
payloads, reinterpret committed product semantics, read pre-``item.json``
layouts, or edit ``write_id`` lineage. When the recorded science meaning
itself changed across versions, the honest outcome is a stale product —
``status()`` flags it and the ``overwrite=True`` stage recomputes it.
