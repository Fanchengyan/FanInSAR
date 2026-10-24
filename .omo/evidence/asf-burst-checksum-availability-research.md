# ASF Sentinel-1 BURST checksum availability: primary-source evidence

**Scope.** Does ASF provide an official checksum (MD5/SHA-256/digest) for the
per-burst `.SAFE.zip` delivered by the burst extractor
(`sentinel1-burst.asf.alaska.edu`), or for the burst TIFF/XML? Investigated
2026-10-23 against live CMR, the ASF search API, `asf_search` source, the
official burst extractor docs, and authenticated downloads from the burst
extractor itself. All claims cited inline.

## TL;DR / Conclusion

**No official checksum is published for the delivered burst product container
(`.SAFE.zip`, `.tiff`, or `.xml`) as a whole.** Integrity verification against a
provider checksum is therefore **not feasible** for the burst extractor products;
FanInSAR's `unverifiable` status is the correct verdict.

What *is* available, precisely:

- **CMR/UMM-G (collection `C2709161906-ASF`, `SENTINEL-1_BURSTS`): no
  checksum field at all.** The granule UMM record has **no `DataGranule`
  object** — hence no `DataGranule.ArchiveAndDistributionInformation[i].Checksum`
  — and no `RelatedUrls` entry pointing at a `.md5`/`.sha256`/`.cksum` sidecar.
  (Confirmed live; see §1.)
- **The burst extractor's job API does not expose a checksum** in its `202` or
  `307` responses, and the final product is served from a private S3 bucket via
  a presigned URL whose headers carry only a **multipart-upload ETag**
  (`"<hex>-21"`), not a content MD5/SHA. Appending `.md5`/`.sha256`/`.cksum` to
  the product URL yields `500 Internal Server Error` — there is no sidecar
  endpoint. (Confirmed live; see §2, §4.)
- **`asf_search` does not expose a burst checksum.** Its `md5sum` property is
  populated only from UMM `AdditionalAttributes[Name="MD5SUM"]` or
  `DataGranule.ArchiveAndDistributionInformation[].Checksum.Value`; burst
  granules carry neither, so `properties["md5sum"]` is absent/`None`. The ASF
  search API burst record contains no checksum-like key at all. (Confirmed in
  source + live API; see §3.)
- **The only official checksums are SAFE-internal MD5s** in `manifest.safe`
  (`<checksum checksumName="MD5">`) covering the *files inside* the SAFE
  (measurement TIFF, annotation XML, preview, support XSDs). These cover
  content files, **not** the downloaded `.zip` container, so they cannot verify
  the zip as delivered. Verified live: every manifest MD5 matches the
  corresponding extracted member byte-for-byte. (Confirmed; see §5.)
- **Contrast:** the full SLC collection (`SENTINEL-1A_SLC`) *does* publish an
  official MD5 in CMR (`AdditionalAttributes[Name="MD5SUM"]`) for the complete
  SLC product — but this is for the full `.zip`, not any burst product. (See §1.)

**Recommendation:** keep the current `unverifiable` outcome for burst products.
There is no official digest to wire in. If stronger integrity guarantees are
ever wanted, the only defensible source is the SAFE-internal `manifest.safe`
MD5s (verify each extracted member after unzip), which is a *content* check on
the SAFE's internal files, not a check on the delivered zip container.

---

## 1. CMR / UMM-G metadata — NO checksum field (CONFIRMED)

Query (live, 2026-10-23):
`https://cmr.earthdata.nasa.gov/search/granules.umm_json?collection_concept_id=C2709161906-ASF&page_size=1`

Granule `G3128067030-ASF` (`S1_301495_IW3_20141003T054235_VV_D5C8-BURST`),
`MetadataSpecification: UMM-G v1.6.7`. The full `umm` object contains **exactly
these top-level keys**:

```
AdditionalAttributes, CollectionReference, GranuleUR, InputGranules,
MetadataSpecification, OrbitCalculatedSpatialDomains, PGEVersionClass,
Platforms, ProviderDates, RelatedUrls, SpatialExtent, TemporalExtent
```

**`DataGranule` is ABSENT** — so there is no
`DataGranule.ArchiveAndDistributionInformation` and no `Checksum` sub-object.
The `RelatedUrls` list contains only the three burst extractor service links
(`0.tiff`, `0.xml`, `0.zip`, each `Type: "USE SERVICE API"`) plus documentation
links — **no `.md5`/`.sha256`/`.cksum`/checksum URL**. `AdditionalAttributes`
are burst physics/orbit fields (`BURST_ID_*`, `BYTE_LENGTH`, `BYTE_OFFSET`,
`LINES_PER_BURST`, `SAMPLES_PER_BURST`, `SV_*`, …) — **no `MD5SUM`, `CHECKSUM`,
or digest attribute**.

Sampled a further 20 *most recent* burst granules and 100 more (page 2) — all
have the same key set, **zero** contain `DataGranule` and **zero** contain any
checksum field or string. (Live scan; matches reported `hits` 167,141,964.)

### UMM-G schema confirms where a checksum *would* live (CONFIRMED)

Official UMM-G v1.6.7 schema:
`https://cdn.earthdata.nasa.gov/umm/granule/v1.6.7/umm-g-json-schema.json`

`definitions.ChecksumType` = `{"Value": str, "Algorithm": enum[MD5, SHA-1,
SHA-2, SHA-256, …]}`, and it is referenced from
`DataGranule.ArchiveAndDistributionInformation[].Checksum` — described as
"Allows the provider to provide the checksum value for the file." ASF burst
granules simply do **not populate** `DataGranule`, so this field is never
present.

### Contrast: full SLC collection DOES publish an MD5 (CONFIRMED)

Query (live):
`https://cmr.earthdata.nasa.gov/search/granules.umm_json?short_name=SENTINEL-1A_SLC&provider=ASF&page_size=1&sort_key=-start_date`

Granule `S1A_S2_SLC__1SSV_…-SLC` contains both:
- `DataGranule.ArchiveAndDistributionInformation: [{"Name":"Not provided","Size":807.73,"SizeUnit":"MB"}]` — **no `Checksum` key here**,
- `AdditionalAttributes`: `{"Name":"BYTES","Values":["846967860"]}` and
  `{"Name":"MD5SUM","Values":["e307e1620d0543cdf6b69c3430ef34a9"]}`.

So ASF's *full-product* MD5 is delivered via the `MD5SUM` attribute, and is
provided for the whole SLC `.zip` only. This is the field `asf_search`'s
`md5sum` property maps to (§3). **Burst granules have no equivalent.**

## 2. ASF data pool / download endpoints — no sidecar for bursts (CONFIRMED)

### Burst extractor sidecar probes (authenticated, 2026-10-23)

Base product:
`https://sentinel1-burst.asf.alaska.edu/S1A_IW_SLC__1SDV_20141003T054235_20141003T054304_002661_002F66_D5C8/IW3/VV/0.zip`

Authenticated requests (Earthdata Login + ASF `asf-urs` cookie) to sidecar
suffixed URLs all return **`500 {"message":"Internal Server Error"}`**:

| URL suffix | result |
|---|---|
| `.zip.md5` | `500 Internal Server Error` |
| `.zip.sha256` | `500 Internal Server Error` |
| `.zip.cksum` | `500 Internal Server Error` |
| `.zip.checksum` | `500 Internal Server Error` |
| `.tiff.md5` | `500 Internal Server Error` |
| `.xml.md5` | `500 Internal Server Error` |
| `0.md5` (no `zip`) | `404 {"message":"Not Found","errors":["'0.md5' is an invalid value for path parameter 'resource'"]}` |

The `404` on `0.md5` proves the extractor validates `file_type ∈ {xml, tiff,
zip}`; the `500` on `*.md5` shows the extension is accepted as a path but no
checksum product exists to serve. **There is no `.md5` sidecar for burst
products.**

(Unauthenticated, the same URLs just `307` → `auth.asf.alaska.edu` →
`urs.earthdata.nasa.gov` login wall, so no public checksum file either.)

### ASF `.md5` sidecar convention applies to datapool products, not bursts (PARTIAL)

ASF's traditional `datapool.asf.alaska.edu/...zip.md5` sidecar is a different
host/service from the burst extractor. Probing a full-SLC datapool `.md5`
(`…/SLC/SA/S1A_IW_SLC__1SSV_20260629T194151_…_A40C.zip.md5`) returned `404 File
Not Found` and the product itself `403 MissingAuthenticationTokenException`
from behind an API gateway — i.e. this path is not publicly fetchable in this
environment and, in any case, **is not the burst path**. Marked
**UNCERTAIN/NOT-FOUND** for datapool sidecar behavior; irrelevant to bursts.

## 3. ASF search API / `asf_search` — no burst checksum (CONFIRMED)

### ASF search API (live)

`https://api.daac.asf.alaska.edu/services/search/param?granule_list=S1_301495_IW3_20141003T054235_VV_D5C8-BURST&output=jsonlite`

Burst result fields: `beamMode, browse, burst, canInSAR, collectionName,
conceptID, dataset, downloadUrl, faradayRotation, fileName, flightDirection,
flightLine, frame, granuleName, groupID, instrument, missionName,
offNadirAngle, orbit, path, pgeVersion, pointingAngle, polarization, productID,
productType, productTypeDisplay, sizeMB, stackSize, startTime, stopTime, thumb,
wkt, wkt_unwrapped`. A regex scan for `md5|checksum|sha256|cksum|digest` over
the whole JSON returns **no matches**.

### `asf_search` source (asfadmin/Discovery-asf_search, `master`)

`asf_search/ASFProduct.py`:
- `md5sum` property path:
  `["AdditionalAttributes", ("Name","MD5SUM"), "Values", 0]` → absent for bursts.
- `_get_file_sizes_and_sums()` reads
  `DataGranule.ArchiveAndDistributionInformation[].Checksum.Value` → returns
  `None` when `DataGranule`/`ArchiveAndDistributionInformation` is missing
  (exactly the burst case).
- `_get_additional_urls()` explicitly **filters out** `url.endswith(".md5")`,
  i.e. `.md5` sidecars are treated as an ASF full-product convention and are
  never surfaced as data.

`asf_search/download/download.py` contains **no** md5/checksum verification —
it writes chunks to disk only; the `202` burst-processing retry
(`_is_burst_processing`) polls, and `strip_auth_if_aws` clears auth headers on
the S3 redirect. Nothing computes or compares a digest.

Official docs
([ASFProduct — ASF SAR Data Search Manual](https://docs.asf.alaska.edu/asf_search/ASFProduct/))
document no checksum attribute; the burst download note only covers `.tiff` +
`.xml` (`FileDownloadType`), with no `.zip`/checksum discussion.

## 4. Burst extractor service docs & job API (CONFIRMED)

Official docs: [sentinel1-burst-documentation.asf.alaska.edu](https://sentinel1-burst-documentation.asf.alaska.edu/)

- Job model: `GET /{granule}/{subswath}/{polarization}/{burst_index}.{file_type}`
  → `202 {'message':'Accepted'}` (job triggered) or `307` (Location → product
  when ready). `file_type ∈ {xml, tiff, zip}`. Response codes 202/307/403/404/500
  are documented; **none carry a checksum**, and the docs never mention a
  digest/checksum/MD5/SHA anywhere.
- Confirmed live: authenticated `0.zip` request → `202` then `307` to a
  presigned S3 URL
  `extractor-main-extractorstagingbucket-…s3.amazonaws.com/…/S1A_IW_SLC__…_4480.zip`
  whose response headers are only:
  `Content-Type: binary/octet-stream`, `Content-Length: 172088850`,
  `Content-Disposition: attachment; filename=…zip`, `ETag:
  "834fcd9efd28000c0a39486495d4c38c-21"`, `x-amz-server-side-encryption: AES256`.
  The `"-21"` suffix identifies an S3 **multipart** ETag, which is **not** the
  MD5 of the object bytes — it is unusable as a content checksum. There is no
  `Content-MD5`, `Digest`, `x-amz-checksum-sha256`, or `x-amz-meta-md5` header.
- The docs describe no `jobs/<id>` REST resource; probing
  `sentinel1-burst.asf.alaska.edu/jobs` and `/jobs/<id>` returns `404
  {"message":"Not Found"}`. (FanInSAR's URL shape `/jobs/<id>/…zip` is thus not
  the documented/public API surface; the documented surface is the
  `/{granule}/{subswath}/{pol}/{idx}.{ext}` form used above.)

## 5. Sentinel-1 SAFE `manifest.safe` — internal MD5s only (CONFIRMED)

The delivered burst `.SAFE.zip` (product
`S1A_IW_SLC__1SSV_20141003T054235_20141003T054235_002661_002F66_4480.SAFE`)
contains `manifest.safe` with per-file `<checksum checksumName="MD5">…</checksum>`
entries, e.g.:

```xml
<byteStream mimeType="application/octet-stream" size="171248944">
  <fileLocation locatorType="URL" href="./measurement/s1a-iw3-slc-vv-…-001.tiff"/>
  <checksum checksumName="MD5">297e5a30ce191c30c26f25319646a3dc</checksum>
</byteStream>
```

Verified live: recomputed MD5 of every zip member **matches** its manifest entry
(e.g. measurement TIFF → `297e5a30ce191c30c26f25319646a3dc` ✓, annotation XML →
`bdd7771fe3ecabd36ac36953f4e63a5b` ✓, calibration XML →
`d8fadcfbf5e219f38c1f7d40e1721846` ✓, noise XML →
`cc3833d6b20e038f104e5164b06eb470` ✓).

**Distinction (important):** these MD5s are for the SAFE's **internal files**,
which are the *ESA product* contents. They are **not** a checksum of the
downloaded `.SAFE.zip` container as delivered by ASF's extractor, and they are
**not** published by ASF in any catalog/API field — they only exist *inside* the
payload you are trying to verify. The `.zip`/`.tiff`/`.xml` products themselves
have no container-level digest (the `.xml` burst product similarly embeds only
SAFE-internal `<checksum>` elements).

## 6. FanInSAR's current handling is consistent with these findings

`faninsar/remote/transfer.py` records `{"status": "unverifiable", "size": size}`
when `asset.checksum` is absent, and never fetches a sidecar. Given §1–§5, no
provider checksum is obtainable for burst products, so `unverifiable` is the
accurate state and no `.md5`/`.sha256` fetch should be added for bursts.

---

## Sources

- CMR granule UMM-JSON (burst): `https://cmr.earthdata.nasa.gov/search/granules.umm_json?collection_concept_id=C2709161906-ASF&page_size=1` (and 20+100 sampled pages)
- CMR granule UMM-JSON (full SLC, MD5SUM contrast): `https://cmr.earthdata.nasa.gov/search/granules.umm_json?short_name=SENTINEL-1A_SLC&provider=ASF&page_size=1&sort_key=-start_date`
- UMM-G v1.6.7 schema (`ChecksumType`, `ArchiveAndDistributionInformationType`): `https://cdn.earthdata.nasa.gov/umm/granule/v1.6.7/umm-g-json-schema.json`
- ASF search API burst record: `https://api.daac.asf.alaska.edu/services/search/param?granule_list=S1_301495_IW3_20141003T054235_VV_D5C8-BURST&output=jsonlite`
- `asf_search` source: `https://github.com/asfadmin/Discovery-asf_search/blob/master/asf_search/ASFProduct.py`, `…/asf_search/download/download.py`
- asf_search docs: `https://docs.asf.alaska.edu/asf_search/ASFProduct/`
- Burst extractor docs: `https://sentinel1-burst-documentation.asf.alaska.edu/`
- Live authenticated burst extractor probes: `…/0.zip`, `…/0.zip.md5`, `…/0.zip.sha256`, `…/0.tiff`, `…/0.tiff.md5`, `…/0.xml`, `…/0.xml.md5`, `…/0.md5` (results in §2, §4)
- Delivered burst `.SAFE.zip` and `manifest.safe` internal MD5 verification (§5)

**Not found / uncertain:** public `datapool.asf.alaska.edu/....md5` sidecar
behavior (403/404 in this environment; burst products are not served from
datapool at all). No ASF/ESA/CMR source publishes a checksum for the burst
`.SAFE.zip` container.
