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
HTTP ``Range``. Zarr directory assets expose their metadata documents and
chunks over the same route. Multi-dimensional time-series Zarr clients are not
qualified in this version, and the service does not create per-epoch Items,
tiles, COGs, or reprojected copies.

No graphical-client display is claimed here. In particular, QGIS, GeoLibre, or
other GIS display of radar-coordinate Items and Zarr assets is not verified by
this feature; client behavior depends on the client and its format support.
Use the printed API and Catalog URLs with a suitable STAC client, and verify
rendering in that client before relying on it.

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
