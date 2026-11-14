Serving a Stack as a read-only STAC API
=======================================

A Stack can expose its committed products as a local, read-only STAC API and
HTTP asset service. The service is a disposable view over the canonical
``item.json`` records and payloads already in the work directory. It does not
prepare inputs, geocode, tile, reprocess, republish, or change scientific
products. ``stack.json`` remains the single workspace identity and canonical
``item.json`` remains product truth.

Start the service
-----------------

Serve in Python with :meth:`Stack.stac_serve
<faninsar.stack.session.Stack.stac_serve>`. The handle is a context manager so
the server always stops::

   from faninsar.stack.session import Stack

   stack = Stack.open("study")
   with stack.stac_serve(host="127.0.0.1", port=8000) as service:
       print(service.catalog_url)
       print(service.api_url)
       service.wait()

``stack.stac_serve`` prints the actual Catalog and API addresses once, after
the socket is bound, the application is ready, and the optional catalog write
has been attempted. Copy either printed address into a STAC client or browser.
A failed startup prints no address.

Pass ``port=0`` to let the operating system assign a free port. The returned
handle and the printed URLs always report the assigned port. Call
``service.refresh()`` after new products are committed and ``service.stop()``
to stop the server; ``stop()`` is idempotent and ``wait()`` blocks until the
server stops.

The command line runs the same service in the foreground::

   faninsar stack stac-serve study --port 8000

``Ctrl-C`` stops it cleanly. Use ``--no-persist-catalog`` on a read-only work
directory or when no derived files should be written.

Reuse a running service in Notebooks
------------------------------------

In one Python process, a second ``Stack.stac_serve`` call for the same
canonical Stack, bind host, and fixed port returns the identical live handle
and logs an info-level reuse message. It does not refresh the index, rewrite
the portable catalog, print the URLs again, or start another server. The
original ``persist_catalog`` choice remains effective even when a later call
requests a different value. Stop the service explicitly and start it again to
change the host, port, persistence, or other configuration.

``port=0`` always starts a new service on an assigned port; a later call using
that host and assigned port returns the same handle. A different Stack, an
external process already bound to the port, or any bind failure is reported as
an error and never reuses another handle. ``stop()`` clears the registration so
a later call can start again; reuse is process-local. Kernels started before
this behavior was installed are not retro-registered: keep the original handle
or, if it is lost, restart the kernel once; that migration is one-time. Use
``stack.stac_serve(port=0)`` for a temporary service without stopping an
existing server. Reacquire a fixed-port service as follows::

   s1 = stack.stac_serve(host="127.0.0.1", port=8000)
   s2 = stack.stac_serve(host="127.0.0.1", port=8000)
   assert s1 is s2  # one live service; no duplicate server or URL print
   s2.stop()        # stop only when intended; no automatic shutdown here

Stopping ``s2`` stops the shared service, so do not attach an automatic
shutdown to a reused handle unless stopping that service is intended. If the
first call raises during bind, execution never reaches a later ``stop()`` line,
so no service was started to stop.

What is served
--------------

Every committed product that passes normalization becomes one STAC Item. Public
ids are unique across the workspace even when product basenames collide,
because they include the product kind, a sanitized source locator, and the
current ``faninsar:write_id``. The canonical raw Item id is preserved as a
property; product families are grouped by family Catalogs rather than merged
into one Item.

Radar-coordinate products keep their radar grid. They are published with
``geometry: null`` and no ``bbox``, and they remain discoverable through the
family Catalog tree and through global Item search with no spatial filter.
They never match ``bbox`` or ``intersects`` queries. A client that enumerates
only Collections cannot list them; use ``/search`` or the family Catalog views
instead.

Committed ``geocoded`` children appear as their own Items with true geographic
geometry and the scientific family and observation coverage of their recorded
source input. The service never geocodes on request. Only geocoded products
already committed to the workspace are visible, and a newly committed child
appears after ``refresh()``. In-memory or uncommitted results are not served.

Search supports ``ids``, ``collections``, ``datetime``, ``bbox``, ``intersects``,
``limit``, and the opaque pagination token. The service advertises STAC API Core,
Item Search, and Collections. Assets are served through the same origin with
HTTP ``Range`` support, including ``206`` partial responses.

Persistence and refresh
-----------------------

With ``persist_catalog=True`` (the default), startup and explicit ``refresh()``
write a portable derived tree under the work directory::

   catalog.json
   stac/items/<public_id>.json
   stac/catalogs/<family>.json
   stac/collections/<collection_id>.json

The tree uses relative hrefs, including valid ``../`` pointers to original
assets. Payloads are not copied, canonical Items are not edited, and the tree
is written only after the service binds successfully. It is not written on
every product commit.

``catalog.json`` and ``stac/`` are disposable derived views, never scientific
commit truth. Refresh swaps the in-memory index atomically and retains the
previous good index if a rebuild fails. If a product is overwritten, its old
public id disappears after ``refresh()``. Before refresh, old asset URLs fail
rather than returning replacement bytes because every request revalidates the
source write id and the opened file identity. The service makes no
cross-request snapshot guarantee without retained product generations.

``persist_catalog=False`` performs no derived filesystem writes, creates no
``catalog.json`` or ``stac/``, and works on a read-only workspace.

Formats and clients
-------------------

The service promises read-only discovery and byte-range access, not conversion.
COG and TIFF payloads are served as bytes, and a client can request windows with
HTTP ``Range``. A Zarr directory asset is published with
``type: application/vnd+zarr`` when the canonical writer did not declare a media
type, and its asset href addresses the store directory root with a trailing
slash (``.../assets/<public_id>/<asset>/``). The href is not replaced with a
pointer to ``zarr.json``; Zarr clients append the metadata document or chunk
paths beneath it and use the same HTTP ``Range`` support. A direct ``GET`` or
``HEAD`` of a verified Zarr root returns a temporary ``307`` redirect to its
actual ``zarr.json``, ``.zarray``, or ``.zgroup`` metadata. This is not a download
of the entire store, and plain directories still have no listing or root
response. A client must support Zarr storage to read the arrays.
Multi-dimensional time-series Zarr clients are not qualified in this version,
and the service does not create per-epoch Items, tiles, COGs, or reprojected
copies.

No graphical-client display is claimed here. In particular, QGIS, GeoLibre, or
other GIS display of radar-coordinate Items and Zarr assets is not verified by
this feature; client behavior depends on the client and its format support.
Use the printed API and Catalog URLs with a suitable STAC client, and verify
rendering in that client before relying on it.

Georeferenced Zarr view
~~~~~~~~~~~~~~~~~~~~~~~

For supported two-dimensional geographic Zarr v3 arrays, the HTTP Item keeps
the original scientific asset key. There is no separate ``_georef`` asset and
no format or view choice to make: the asset the client already uses for the
science product becomes the georeferenced view. Its href points at
``/zarr/<public_id>/<source_asset_key>.zarr/``, its type is
``application/vnd+zarr``, and its title is the existing human title if there is
one, otherwise the scientific asset key.

The view asset adds ``proj:code``, ``proj:shape``, and ``proj:transform`` from
the view's actual CRS and grid, plus ``faninsar:source_asset``,
``faninsar:gdal_uri``, and ``faninsar:native_href``. The native href is the
original ``/assets/...`` URL, which continues to serve the raw store for
advanced clients; the ``/zarr/`` view route serves the georeferenced group.
Neither the canonical Item nor its asset keys, hrefs, checksums, payloads, or
scientific samples are edited. Native payload facts such as file checksums and
sizes are not copied onto the HTTP-only view asset because they describe the
raw store.

The virtual group publishes a fixed four-entry discovery manifest
(``zarr.json``, ``values/``, ``x/``, and ``y/``), the generated group and array
metadata, bounded float64 pixel-centre coordinates, and the original
``values`` chunks through the existing pinned, write-id-checked, range-capable
asset path. There is no directory enumeration, parent link, or undeclared
child. Requests for unknown children, traversal, symbolic links, out-of-bounds
chunks, and stale views return ``404``.

Unsupported assets keep their original raw asset document and key. This
includes radar-coordinate Items, rotated or sheared grids, rank-1 arrays,
unsupported data types, and assets the metadata worker rejects. They receive
no synthesized CRS or projection metadata, their ``/zarr/`` route returns
``404``, and the service makes no universal format promise for the raw asset:
read it with a client that supports its actual format.

Open the values with GDAL's Zarr driver rather than relying on generic plain
HTTP autodetection::

   ZARR:"/vsicurl/http://127.0.0.1:8000/zarr/<public_id>/coherence.zarr/":/values

The connection string above is the supported contract; opening the bare view
root also works when the driver reads the fixed HTML manifest. A generic GIS
"add raster from URL" click is not guaranteed to select the Zarr driver, so use
the GDAL connection string or a client with explicit Zarr support. The service
does not resample, convert dtype, write COG files, recompute arrays, or geocode
on request.

The portable catalog remains the raw, frozen disk rendering. It contains
native asset hrefs and no virtual view hrefs or ``faninsar:native_href``. View
augmentation is an HTTP presentation layer rebuilt from current backend
descriptors, so no offline or static view matching is promised.

Lazy radar geolocation preview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Radar-coordinate Items stay radar-coordinate Items: ``geometry`` remains
``null`` and no synthetic Item extent is published. When a committed Reference
scene already declares full-grid ``latitude`` and ``longitude`` arrays, the
service can serve a lazily resampled EPSG:4326 preview of a bound scalar asset
under that asset's existing scientific key. Binding follows recorded
provenance only: an IFG's ``faninsar:inputs`` primary and secondary scene write
ids lead to the scenes' ``faninsar:reference`` identifier and declared
Reference write id, and an unwrap product follows its recorded IFG input. Shape
or grid identity alone never selects a Reference, and an ambiguous or cyclic
chain is not guessed.

A supported asset gets a ``/radar/<public_id>/<asset_key>.zarr/`` href with
``type: application/vnd+zarr``, its existing title and roles,
``faninsar:native_href`` pointing at the original raw asset URL,
``faninsar:gdal_uri`` with the exact GDAL Zarr connection string,
``faninsar:resampling: bilinear``, ``faninsar:preview: true``, and
``faninsar:view_status: lazy-geolocation``. No projected shape or transform is
invented on the asset or Item; GDAL derives the actual grid from the served
group. Native checksums and file sizes are not copied because they describe
the raw store, not the preview.

The virtual group has the same fixed four-entry discovery manifest
(``zarr.json``, ``values/``, ``x/``, and ``y/``) and the same range-capable
HTTP behavior as the geographic view. Its ``values`` array declares the
backend's actual warped shape, preserved ``float32``/``float64`` type, an
uncompressed little-endian byte codec, NaN fill, and EPSG:4326 ``_CRS``.
``x`` and ``y`` carry bounded generated pixel-centre longitude and latitude
chunks. A target image chunk is read through the lazy backend only for the
requested block; metadata or Item rendering never initializes a warp or reads
the scientific image array. The first metadata or chunk request for one asset
initializes its geolocation maps once, under a bounded pixel cap. Open warps
are cached up to eight entries with a byte-bounded target-chunk LRU and are
closed on eviction and service shutdown.

Supported inputs are scalar real ``coherence`` and continuous
``unwrapped_phase`` assets (and scalar scene unit assets with declared
``faninsar:rows``, ``faninsar:cols``, ``faninsar:row_origin``, and
``faninsar:col_origin``). Complex SLC, wrapped phase, integer labels, and other
non-continuous arrays remain raw assets. Missing or undeclared Reference
geolocation, an ambiguous or cyclic chain, a mismatched parent grid, or an
unsupported scalar type leaves the native asset unchanged and adds a stable
``faninsar:view_unavailable`` diagnostic such as ``missing-geolocation``; the
``/radar/`` route returns ``404`` rather than fabricating a preview. A backend
geolocation cap is reported as ``503``. Geometry assets added while a scene
write id is preserved require ``refresh()`` and are detected by a current
receipt/asset fingerprint, not by the index generation or a permanent negative
cache. Stale source or Reference-map receipts make existing preview URLs fail
with ``404``.

Open the preview with the advertised GDAL connection string::

   ZARR:"/vsicurl/http://127.0.0.1:8000/radar/<public_id>/coherence.zarr/":/values

This is a read-only derived preview. It does not change canonical products,
convert dtype, write COGs, run an external GDAL command, or geocode at large.

Network scope and CORS
----------------------

The default bind is loopback (``127.0.0.1``). ``localhost``, ``127.0.0.1``,
and ``::1`` are treated as loopback. Passing any other host is honored
explicitly and warns that the service is reachable beyond the local machine.
The service has no authentication, TLS, accounts, or multi-host deployment
contract; do not bind it to an untrusted network as a publication endpoint.

Read-only CORS is enabled with ``Access-Control-Allow-Origin: *``. That is a
deliberate access policy, not a confidentiality boundary: any origin that can
reach the bound port can read the served metadata and payloads, and the
wildcard additionally lets a browser origin read responses the network already
permits. Treat starting the service as a data-sharing decision.

Browser mixed-content and private-network policies may block or restrict an
HTTPS page from fetching a local ``http://`` service. Those rules vary by
browser, version, and user policy, and CORS cannot override them. If a browser
refuses the request, use a local tool or client instead of assuming the service
is broken.

Limitations
-----------

- No writes to canonical Items or payloads, and no geocoding, tiling,
  reprojection, COG conversion, repair, or recomputation on request.
- No database, external catalog service, tile server, authentication, TLS, or
  public deployment contract.
- No merging of a product family into one Item and no virtual assets for
  unwrap, ion, or related sibling products.
- No per-epoch Items or COGs synthesized from a persisted TimeSeries; the
  persisted TimeSeries is one aggregate Item.
- Collection-only clients cannot enumerate radar-coordinate Items.
- The derived index is disposable; in-place mutation outside the commit and
  write-id contract is outside the service guarantee.
