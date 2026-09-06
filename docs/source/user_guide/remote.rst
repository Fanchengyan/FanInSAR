Remote discovery and downloads
==============================

FanInSAR's remote boundary has two operations: ``search`` returns normalized
catalog items and ``download`` publishes one complete, verified asset.  Both
operations use the same registered catalog name, so a provider adapter does
not add a second public engine or scientific object model.

General CMR discovery
---------------------

NASA's Common Metadata Repository (CMR) can be registered for any collection.
The adapter handles CMR JSON or UMM-JSON pagination and normalizes footprints,
acquisition metadata, and explicit ``GET DATA`` links::

   from faninsar import remote
   from faninsar.data.query import BoundingBox

   remote.register_cmr_catalog(
       "asf-s1",
       provider="ASF",
       collection="S1-SLC",
       data_origins=("https://datapool.asf.alaska.edu",),
   )
   items = remote.search(
       BoundingBox(-123, 45, -122, 46, crs=4326),
       catalog="asf-s1",
       limit=10,
   )
   remote.download(items[0].assets["data"], "cache/scene.zip")

The catalog registration is explicit and performs no network request.  CMR
discovery starts when ``search`` is called.  A catalog identity, collection,
provider, and approved data origins remain attached to every returned item;
metadata or browse links are never promoted to download assets.

Provider profiles
-----------------

The optional ``asf`` extra provides the certified ``asf-search`` 13.x profile::

   pip install 'FanInSAR[asf]'

Register it with the same catalog registry and use the same public calls::

   remote.register_asf_search_catalog(
       "asf-s1-search", collection="S1-SLC"
   )
   items = remote.search(roi, catalog="asf-s1-search")

Planetary Computer Copernicus DEM discovery is available through the ``pc``
extra (``planetary-computer`` and ``pystac-client``).  The DEM facade keeps
planning offline and starts STAC search, anonymous asset signing, and bytes at
materialization::

   from faninsar.processing.dem import DEM

   dem = DEM.from_source("glo30:pc", cache_dir=".cache/dem")
   raster = dem.to_raster(grid)

``DEM.from_source`` and ``dem_catalog`` are safe for offline inspection.
Provider selection uses the explicit ``<product>:<provider>`` grammar; an
unknown or unwired pair fails before any request.  ``auto`` only performs the
documented same-family GLO-30 to GLO-90 rescue for withheld cells.

FanInSAR STAC Profile
---------------------

STAC records accepted at the remote boundary use STAC Core 1.1.0 and the
version-pinned SAR, Satellite, Projection, File, and optional InSAR extension
profiles.  ``remote.validate_stac_item`` validates an item and
``remote.normalize_stac_item`` maps it to the provider-neutral catalog shape.
Unknown extensions, malformed geometry, missing assets, and invalid numeric
metadata are rejected before catalog records enter the scientific API.

Real-service requirements and failure behavior
-----------------------------------------------

Remote examples require access to the real HTTPS service and, where noted,
the matching optional extra.  Tests inject clients or response fixtures and do
not imply service availability.  Registrations define endpoint and redirect
allowlists, and each search/download operation has finite request, byte,
retry, redirect, and elapsed-time budgets.

Provider failures are terminal.  FanInSAR never silently switches from ASF to
CMR, between providers, or between DEM products.  Missing credentials,
unsupported provider versions, unavailable services, malformed records, and
budget exhaustion are surfaced as typed errors with stable ``reason`` values.
Signed URLs and credentials are removed from persisted metadata and diagnostic
records.
