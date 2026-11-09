Transfer completion without payload self-hashing
================================================

FanInSAR reuses completed downloads, measurement spans, and ROI burst canvases
from durable **completion records**, not from a second SHA of local bytes.
Provider-supplied checksums are still compared while bytes are streamed. After
a record is published, same-size edits of a trusted local file are not
detected; use ``overwrite=True`` for a deliberate refetch.

Completed downloads
-------------------

``remote.download`` reuses a destination when:

* the asset is qualified (provider checksum or immutable version);
* a private identity-qualified manifest exists;
* the recorded identity matches the asset;
* the recorded size matches the file size;
* the record is complete (not a failed provider-verification attempt).

Reuse does not open the destination for hashing. A legacy manifest that only
stores ``identity``, ``sha256``, and ``size`` is accepted on identity and size;
the stored digest is not compared. New manifests add a small schema
discriminator and an explicit ``complete`` flag.

Provider verification still happens on the transfer itself. If the provider
advertises SHA-256 (or another digest), incoming bytes — including a resumed
``.part`` prefix — are compared with that digest. Unverifiable inputs stay
``unverifiable``. SAFE member MD5 checks from ``manifest.safe`` are unchanged.
A failed verification record is never treated as completion, even when a
previous same-sized destination is still on disk.

Interrupted complete-file downloads still resume with
``Range: bytes=<received>-``. ``resume=False`` fetches from byte 0.

Spans and ROI canvases
----------------------

Span cache keys remain ``sha1(url|start|end)`` of the scrubbed URL. Completion
is ``state="complete"`` plus ``part_size == actual .part size == expected
length``. Partial files resume from the retained prefix; ETag / If-Range
behaviour and orphan cleanup are unchanged. Completed span bodies are not
SHA-1 scanned.

ROI burst progress records completed indices after the raster and mask have
been synchronized. The caller publishes progress only after ``write_burst``
returns; a failed sync raises and does not discard the only raw span bytes.
Plan fingerprints still cover planner facts (annotation, layout, windows).
Legacy ``bursts: {index: "<digest>"}`` maps are treated as completed indices
once the current plan fingerprint and canvas structure (shape, dtype, XML,
burst slots) match. Torn JSON, unknown schemas, and incompatible fingerprints
replay that unit.

What this does not do
---------------------

* No audit mode, stat watcher, or extra version framework.
* No claim that a matching completion record proves scientific correctness.
* No automatic recovery of same-size local corruption after completion.
