Durable Stack workspaces and additive extension
===============================================

Every supported constructor with a ``work_dir`` publishes the complete Stack
intent before it returns. Construction performs no catalog search, download,
DEM fetch, or scientific stage. The returned Stack is immediately reopenable
with :meth:`Stack.open <faninsar.stack.session.Stack.open>`::

   stack = S1Stack.from_search(search, work_dir="study")
   # stack.json now holds the selected search request and science contract.

   reopened = Stack.open("study", verification="fast")
   assert reopened.intent_revision == stack.intent_revision

``stack.json`` is the single authoritative intent document. It records a
``schema_version``, ``intent_revision``, the normalized source request,
mission targets (ROI, swaths, bursts), the requested science contract and the
durable resolved pins (Reference, inputs, planned Pair networks). Readiness
and progress remain derived from committed products; there is no second
persisted configured/prepared/done state machine.

Constructing twice with the same normalized request is idempotent. A different
intent in an existing directory is rejected instead of overwritten::

   stack = S1Stack.from_search(search, work_dir="study")
   stack.extend(search=additional_search, roi=additional_roi)

Additive extension
------------------

:meth:`Stack.extend <faninsar.stack.session.Stack.extend>` unions additional
search results or local paths plus ROI, burst, and sub-swath targets. It never
replaces the existing ROI and never changes the Reference, grid/CRS,
resolution, alignment, or scientific method. Source modes are exclusive:
``search`` and ``paths`` cannot be combined in one call.

Sentinel-1 search items are deduplicated by stable catalog identity and sorted,
so provider result order does not change the intent. Polygons are compared as a
set, so reordering or repeating a polygon does not grow the target. A repeated
extension is a no-op and leaves ``intent_revision`` unchanged.

Reuse remains conditional on the actual scientific support. Unchanged burst
additions can reuse existing units, but Radar ROI changes conservatively
recompute affected date units using the complete canonical ROI identity.
Projected Geo grids use the same conservative policy unless an exact affine
geographic support window can be established. Sparse sampled footprints are
not used to certify unchanged support. This may do more work than a pixel-wise
incremental update, but prevents stale masks or payloads from being reused.

Supported legacy adoption
-------------------------

Opening a supported version-1 ``stack.json`` or a request-only workspace
(``inputs/request.json``) adopts it into the version-2 intent. The previous
metadata is preserved as ``stack.json.bak`` or ``inputs/request.json.bak``
before the atomic replacement. Adoption never redownloads, rehashes, or
reruns a scientific stage. An ambiguous legacy identity conflict is reported
and does not authorize reuse; pass ``on_error="raise"`` to stop instead. A
newer schema is rejected rather than downgraded.

Integrity policy
----------------

``verification="fast"`` is the default and scans no bulk SLC payloads.
``verification="strict"`` is an execution policy, not persisted science; it
rereads a committed product's declared payload against a trusted baseline when
one exists. A product without a trusted baseline is reported as unverifiable
rather than hashed against itself, and strict coregistration recomputes that
scene instead of reusing it. Strict runs baseline the scenes they publish, so
the next strict run can verify them. The policy is supplied to
:meth:`Stack.open <faninsar.stack.session.Stack.open>` or to the combined
execution entry point.

Strict source qualification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Strict sessions also qualify each date's original source before cached science
is reused. A local source needs an explicit trusted baseline, recorded once
through :func:`faninsar.stack.source_integrity.establish_source_integrity`
under ``<work_dir>/integrity/sources`` after preparation::

   root = stack.config.work_dir / "integrity" / "sources"
   for paths in stack.catalog.paths.values():
       for source in paths:
           establish_source_integrity(root, source)

There is no automatic self-baseline: strict verification never hashes current
bytes into a baseline, and a source without one is
``integrity_unverifiable``. A freshly admitted single-file provider download
may instead be baselined from its qualified provider receipt; receipts without
a digest, and multi-file directory sources, never become a self-baseline.
Provider evidence is accepted only for the exact receipted single file.

With ``on_error="warn"`` preparation continues with the dates whose sources
verify and excludes the rest from ready inputs and dependent science; the
report records each failure in the ``source_integrity`` phase. With
``on_error="raise"`` it raises
:class:`faninsar.stack.preparation.DataPreparationError` whose report still
lists the dates that verified, so strict mode never silently broadens trust.

NISAR tile checkpoints
~~~~~~~~~~~~~~~~~~~~~~

NISAR strict execution applies the same policy to its committed per-tile
checkpoints. Tiles live under ``<work_dir>/unit-checkpoints/nisar`` and their
baselines under ``<work_dir>/integrity``. A reused tile is reread against its
baseline, an unverifiable tile is recomputed rather than trusted, and a newly
committed tile is baselined immediately. Fast runs keep the checkpoint
metadata path and hash no tile payloads.

Baselines record bytes, not scientific truth. Sources and historical products
created before strict verification have no baseline, so strict mode reports
them unverifiable and can only recompute them; it cannot retroactively prove
their provenance. Fast mode retains the same assurance limits.

Streaming preparation
---------------------

:meth:`Stack.prepare_and_coregister
<faninsar.stack.session.Stack.prepare_and_coregister>` runs preparation with a
bounded controlling-thread callback. Each date is handed to local science as
soon as its sources and required orbit/AUX inputs are qualified, while the
remaining acquisition futures keep running. In ``geometry`` mode a newly ready
secondary is coregistered immediately. In ``nesd`` mode only measurement edges
whose two endpoints are ready are measured; the network is solved once after
the measurement attempt is terminal, then corrected final products are
applied. Preparation holds the same OS writer lock as the scientific stages,
and nested stage calls re-enter it on the owning thread instead of opening a
second lock.

Automatic NESD recovery
-----------------------

Each measured pair arc is committed independently under
``unit-checkpoints/misreg/<primary>_<secondary>`` with its endpoint source,
orbit/AUX, quality and actual overlap-support identity. A repeated call reuses
edges whose inputs are unchanged and retries only missing or failed edges, so a
campaign interrupted after some arcs resumes instead of remeasuring them.

Before final apply the current graph is qualified with the reference-connected,
identifiable network solve. Disconnected dates and components are reported as
blocked and are not published as zero corrections. Missing redundant edges warn
and continue while the remaining graph is qualified; pass ``strict=True`` to
the combined entry to require every planned edge and target. A changed global
solution does not by itself invalidate a corrected scene: reuse is decided by
the date's effective correction signature, while the scene keeps the exact
solution provenance it consumed. Measurement never creates a provisional full
scene, and final apply reads the original SLC samples once.

Pair records distinguish an axis omitted by the method, an explicit absence of
overlap, and a measurement rejected by quality control. A rejected azimuth
observation is not a zero correction: the target remains blocked when no
qualifying azimuth solution connects it to the Reference, even if its range
observation is usable. Reports retain ``quality_rejected`` rather than describing
that failure as an unmeasured axis. Measurement recipe revisions invalidate
older ambiguous pair records so they are remeasured; quality thresholds are not
relaxed automatically to make a run complete.

Combined execution and reports
------------------------------

:meth:`Stack.prepare_data <faninsar.stack.session.Stack.prepare_data>` remains
preparation-only and
:meth:`Stack.coregister_scenes <faninsar.stack.session.Stack.coregister_scenes>`
remains processing-only; coregistration never acquires missing inputs.
:meth:`Stack.prepare_and_coregister
<faninsar.stack.session.Stack.prepare_and_coregister>` authorizes both stages
and returns one diagnostic report. The latest persisted attempt is available
through ``stack.coregistration_report``::

   report = stack.prepare_and_coregister(on_error="warn")
   print(report.to_markdown())

   latest = stack.coregistration_report
   print(latest.counts)

Reports are written atomically under
``reports/coregistration/<attempt-id>/report.json`` with a derived
``report.md``. They retain failed and reused unit dispositions, are
redacted, and never authorize product completion or replace intent. An
attempt left unfinished by process death is recognized and recorded as
interrupted only while the next writer owns the workspace lock; the next
attempt keeps that reconciliation evidence in its scope. The default selection
commits the Reference scene even when no secondary is ready, and an explicit
empty ``dates=[]`` selection performs no scene work.
