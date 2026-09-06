"""Remote burst extraction from ASF / ESA SAFE products via ``requests``.

Uses only the standard library and ``requests`` (already a project dependency)
to authenticate against Earthdata Login via URS form login + OAuth, perform
byte-range reads on ZIP members, and reconstruct TOPS burst pixel data.

Usage example::

    from faninsar.missions.s1.io.remote import RemoteSafe

    safe = RemoteSafe("https://datapool.asf.alaska.edu/SLC/SA/S1A_...1884.zip")
    burst = safe.extract_burst("IW1", "VV", burst_index=2)

    # Or one-shot convenience:
    from faninsar.missions.s1.io.remote import extract_remote_burst

    burst = extract_remote_burst(url, "IW1", "VV", burst_index=2)
"""

from __future__ import annotations

import netrc
import re
import struct
import time
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import requests

from faninsar.logging import setup_logger
from faninsar.missions.s1.errors import reject_product
from faninsar.missions.s1.io.read import (
    BurstArray,
    _valid_column_bounds,
)

logger = setup_logger(__name__)

_BYTES_PER_PIXEL = 4
_EDL_CLIENT_ID = "BO_n7nTIlMljdvU6kRRB3g"
_RANGE_TIMEOUT = 300


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _authenticate(session: requests.Session) -> None:
    """URS form login + OAuth via ``requests`` (pure Python).

    Retries aggressively with exponential backoff and jitter to work around
    intermittent TLS timeouts between Python's ``ssl`` module and Earthdata
    Login servers.  Once the ``asf-urs`` cookie is obtained the session can
    be used for all subsequent byte-range requests.
    """
    import random as _random
    import time as _time

    creds = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
    user, _, pwd = creds
    last_err: Exception | None = None

    for attempt in range(8):
        try:
            r = session.get(
                "https://urs.earthdata.nasa.gov/login",
                timeout=60,
            )
            m = re.search(r'name="authenticity_token"\s+value="([^"]+)"', r.text)
            if not m:
                msg = "authenticity_token not found in URS login page"
                raise RuntimeError(msg)  # noqa: TRY301

            session.post(
                "https://urs.earthdata.nasa.gov/login",
                data={
                    "authenticity_token": m.group(1),
                    "username": user,
                    "password": pwd,
                },
                timeout=60,
            )

            auth_url = (
                "https://urs.earthdata.nasa.gov/oauth/authorize"
                f"?splash=false&client_id={_EDL_CLIENT_ID}"
                "&response_type=code"
                "&redirect_uri=https://cumulus.asf.alaska.edu/login"
            )
            session.get(auth_url, timeout=60)

            if "asf-urs" in session.cookies:
                logger.info("ASF auth OK after %d attempt(s)", attempt + 1)
                return  # success

            msg = (
                "asf-urs cookie not set after OAuth; "
                f"cookies={list(session.cookies.keys())}"
            )
            raise RuntimeError(msg)  # noqa: TRY301

        except Exception as exc:
            last_err = exc
            logger.warning(
                "Auth attempt %d/8 failed: %s",
                attempt + 1,
                exc,
            )
            if attempt < 7:
                _time.sleep(2**attempt + _random.uniform(0, 2))

    reject_product(f"ASF authentication failed after 8 attempts: {last_err}")


def _range_get(
    session: requests.Session,
    url: str,
    start: int,
    end: int,
    timeout: int = _RANGE_TIMEOUT,
) -> bytes:
    """Perform a byte-range GET returning the raw response body."""
    r = session.get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=timeout,
    )
    if r.status_code != 206:
        reject_product(
            f"range request {start}-{end} returned HTTP {r.status_code} "
            f"(expected 206); body={r.content[:120]!r}"
        )
    return r.content


def _parse_zip_central_directory(
    session: requests.Session, url: str
) -> tuple[dict[str, tuple[int, int, int, int]], int]:
    """Read ZIP End-of-Central-Directory + central directory.

    Returns ``(members, total_size)`` where ``members[name] = (lhdr_offset,
    compressed_size, method, uncompressed_size)``.
    """
    # Get total file size via Range 0-0 -> Content-Range header
    resp = session.get(url, headers={"Range": "bytes=0-0"}, timeout=30)
    cr = resp.headers.get("Content-Range", "")
    if "/" not in cr:
        reject_product(
            f"Content-Range header not found in 206 response; "
            f"headers={dict(resp.headers)}"
        )
    total = int(cr.split("/")[-1])
    if total <= 0:
        reject_product(f"invalid file size from Content-Range: {cr}")

    # 2. read tail to find EOCD
    tail_size = min(total, 65536)
    tail = _range_get(session, url, total - tail_size, total - 1)
    eocd_off = tail.rfind(b"PK\x05\x06")
    if eocd_off < 0:
        reject_product("End-of-Central-Directory signature not found in ZIP tail")
    eocd = tail[eocd_off : eocd_off + 22]
    cd_off = struct.unpack("<I", eocd[16:20])[0]
    cd_size = struct.unpack("<I", eocd[12:16])[0]
    cd_count = struct.unpack("<H", eocd[10:12])[0]

    # 3. read central directory
    cd = _range_get(session, url, cd_off, cd_off + cd_size - 1)

    members: dict[str, tuple[int, int, int, int]] = {}
    p = 0
    for _ in range(cd_count):
        if p + 46 > len(cd):
            break
        sig = cd[p : p + 4]
        if sig != b"PK\x01\x02":
            break
        method = struct.unpack("<H", cd[p + 10 : p + 12])[0]
        comp_size = struct.unpack("<I", cd[p + 20 : p + 24])[0]
        file_size = struct.unpack("<I", cd[p + 24 : p + 28])[0]
        name_len = struct.unpack("<H", cd[p + 28 : p + 30])[0]
        extra_len = struct.unpack("<H", cd[p + 30 : p + 32])[0]
        comment_len = struct.unpack("<H", cd[p + 32 : p + 34])[0]
        lhdr_off = struct.unpack("<I", cd[p + 42 : p + 46])[0]
        name = cd[p + 46 : p + 46 + name_len].decode("utf-8", errors="replace")
        members[name] = (lhdr_off, comp_size, method, file_size)
        p += 46 + name_len + extra_len + comment_len
    return members, total


def _fetch_inflate(
    session: requests.Session,
    url: str,
    lhdr_off: int,
    comp_size: int,
) -> bytes:
    """Download a deflated ZIP member via byte-range and inflate in memory."""
    # local file header: 30 B + filename_len + extra_len
    lh = _range_get(session, url, lhdr_off, lhdr_off + 30 + 256, timeout=30)
    name_len = struct.unpack("<H", lh[26:28])[0]
    extra_len = struct.unpack("<H", lh[28:30])[0]
    data_start = lhdr_off + 30 + name_len + extra_len
    raw = _range_get(session, url, data_start, data_start + comp_size - 1)

    if comp_size == 0:
        return b""
    return zlib.decompress(raw, -15)


def _zip_dir_members(members: dict, prefix: str, suffix: str) -> list[str]:
    """Return member names matching *prefix* (path) and *suffix*."""
    return [n for n in members if prefix in n and n.endswith(suffix)]


class RemoteSafe:
    """Authenticated view of a remote SAFE ZIP product.

    Handles authentication (URS Login + OAuth), ZIP central directory
    parsing, and lazy burst extraction via byte-range downloads.

    Parameters
    ----------
    url : str
        HTTP(S) URL to the SAFE ZIP, e.g. an ASF ``datapool.asf.alaska.edu``
        download link.
    session : requests.Session, optional
        Pre-authenticated session.  If omitted a new session is created and
        authenticated via the netrc entry for ``urs.earthdata.nasa.gov``.

    """

    def __init__(self, url: str, session: requests.Session | None = None) -> None:
        """Initialise from a remote SAFE ZIP URL, authenticating if needed.

        Parameters
        ----------
        url : str
            HTTP(S) URL to the SAFE ZIP, e.g. an ASF ``datapool.asf.alaska.edu``
            download link.
        session : requests.Session, optional
            Pre-authenticated session.  If omitted a new session is created and
            authenticated via the netrc entry for ``urs.earthdata.nasa.gov``.

        """
        self._url = url
        if session is None:
            session = requests.Session()
            session.trust_env = False
            _authenticate(session)
        self._session = session
        # Lazily parsed
        self._members: dict[str, tuple[int, int, int, int]] | None = None
        self._total_bytes: int = 0
        self._annotations: dict[str, bytes] = {}

    @property
    def url(self) -> str:
        """The remote SAFE ZIP URL."""
        return self._url

    @property
    def total_bytes(self) -> int:
        """Total size of the remote ZIP in bytes."""
        if self._members is None:
            self._lazy_load_directory()
        return self._total_bytes

    def _lazy_load_directory(self) -> None:
        """Parse the ZIP central directory if not already loaded."""
        if self._members is not None:
            return
        logger.info("Parsing remote ZIP central directory from %s", self._url)
        t0 = time.time()
        self._members, self._total_bytes = _parse_zip_central_directory(
            self._session, self._url
        )
        logger.info(
            "Parsed %d members from %.2f GiB ZIP in %.1fs",
            len(self._members),
            self._total_bytes / (1024**3),
            time.time() - t0,
        )

    def _parse_annotation(self, swath_name: str, polarization: str) -> bytes:
        """Return the (cached) annotation XML for a sub-swath."""
        key = f"{swath_name}/{polarization}"
        if key in self._annotations:
            return self._annotations[key]
        self._lazy_load_directory()
        assert self._members is not None  # loaded above

        # Find the annotation member
        candidates = _zip_dir_members(self._members, "/annotation/", ".xml")
        # Filter: not calibration, not rfi, matching swath/pol
        pol = polarization.lower()
        sw = swath_name.lower()
        target = next(
            (
                n
                for n in candidates
                if sw in n and pol in n and "calibration" not in n and "rfi-" not in n
            ),
            None,
        )
        if target is None:
            reject_product(
                f"annotation not found for {swath_name}/{polarization} "
                f"in remote ZIP; candidates={candidates[:6]}"
            )
        lhdr, comp, *_ = self._members[target]
        xml_bytes = _fetch_inflate(self._session, self._url, lhdr, comp)
        self._annotations[key] = xml_bytes
        return xml_bytes

    def extract_burst(
        self,
        swath_name: str,
        polarization: str,
        burst_index: int = 0,
    ) -> BurstArray:
        """Extract a single TOPS burst from the remote product.

        Parameters
        ----------
        swath_name : str
            Sub-swath name, e.g. ``"IW1"``, ``"IW2"``, ``"IW3"``.
        polarization : str
            Polarisation channel, e.g. ``"VV"``, ``"VH"``.
        burst_index : int, optional
            Burst index within the sub-swath.

        Returns
        -------
        BurstArray
            Complex64 samples with valid-sample mask, matching the
            convention of :func:`~faninsar.missions.s1.io.read.read_full_burst`.

        """
        self._lazy_load_directory()
        assert self._members is not None

        # Parse annotation for burst geometry
        xml_text = self._parse_annotation(swath_name, polarization).decode("utf-8")
        root = ET.fromstring(xml_text)
        st = next(c for c in root if _local(c.tag) == "swathTiming")
        lpb = int(next(c for c in st if _local(c.tag) == "linesPerBurst").text)
        spb = int(next(c for c in st if _local(c.tag) == "samplesPerBurst").text)
        bl = next(c for c in st if _local(c.tag) == "burstList")
        bursts = [c for c in bl if _local(c.tag) == "burst"]

        if burst_index < 0 or burst_index >= len(bursts):
            reject_product(
                f"burst_index {burst_index} out of range (0-{len(bursts) - 1})"
            )

        burst_el = bursts[burst_index]
        byte_off = int(next(c for c in burst_el if _local(c.tag) == "byteOffset").text)
        first_vals = tuple(
            int(v)
            for v in next(
                c for c in burst_el if _local(c.tag) == "firstValidSample"
            ).text.split()
        )
        last_vals = tuple(
            int(v)
            for v in next(
                c for c in burst_el if _local(c.tag) == "lastValidSample"
            ).text.split()
        )

        if len(first_vals) != lpb or len(last_vals) != lpb:
            reject_product("burst valid-sample length mismatch")

        # Find the TIFF member
        tiff_name = next(
            (
                n
                for n in self._members
                if "/measurement/" in n
                and n.endswith(".tiff")
                and swath_name.lower() in n
                and polarization.lower() in n
            ),
            None,
        )
        if tiff_name is None:
            reject_product(
                f"measurement TIFF not found for {swath_name}/{polarization}"
            )
        lhdr, comp_size, *_ = self._members[tiff_name]

        # Download + inflate the TIFF member
        tiff_raw = _fetch_inflate(self._session, self._url, lhdr, comp_size)
        length = lpb * spb * _BYTES_PER_PIXEL
        burst_raw = tiff_raw[byte_off : byte_off + length]
        if len(burst_raw) < length:
            reject_product(
                f"burst byte slice is {len(burst_raw)} bytes, expected {length}"
            )

        # Reinterpret int16 -> complex64
        pairs = np.frombuffer(burst_raw, dtype="<i2").reshape(lpb, spb, 2)
        full = (
            pairs[..., 0].astype(np.float32) + 1j * pairs[..., 1].astype(np.float32)
        ).astype(np.complex64)

        # Apply valid-sample envelope (same logic as read_full_burst)
        col0, col1 = _valid_column_bounds(
            type(
                "_Burst",
                (),
                {
                    "index": burst_index,
                    "first_valid_sample": first_vals,
                    "last_valid_sample": last_vals,
                    "samples": spb,
                },
            )()
        )
        first = np.asarray(first_vals, dtype=np.int32)
        last = np.asarray(last_vals, dtype=np.int32)
        cols = np.arange(col0, col1, dtype=np.int32)[None, :]
        valid_mask = (
            (first[:, None] >= 0) & (cols >= first[:, None]) & (cols <= last[:, None])
        )
        samples = np.where(valid_mask, full[:, col0:col1], 0)

        row0 = burst_index * lpb

        return BurstArray(
            samples=samples,
            row0=row0,
            col0=col0,
            burst_index=burst_index,
            valid_mask=valid_mask,
        )

    def batch_extract(
        self,
        selections: list[tuple[str, str, int]],
        max_workers: int = 3,
    ) -> list[BurstArray]:
        """Extract multiple bursts in parallel across sub-swaths.

        Each sub-swath TIFF member is downloaded concurrently in a thread
        pool, then bursts are extracted from the inflated data.  This can
        significantly reduce wall-clock time when bursts span multiple
        sub-swaths (IW1 / IW2 / IW3).

        Parameters
        ----------
        selections : list of (swath_name, polarization, burst_index)
            Each entry selects one burst to extract.
        max_workers : int, optional
            Number of parallel connections (default 3, one per sub-swath).

        Returns
        -------
        list[BurstArray]
            Results in the same order as *selections*.

        Examples
        --------
        >>> safe = RemoteSafe(url)
        >>> results = safe.batch_extract(
        ...     [
        ...         ("IW1", "VV", 0),
        ...         ("IW2", "VV", 1),
        ...         ("IW3", "VV", 2),
        ...     ],
        ...     max_workers=3,
        ... )

        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Pre-load the central directory before spawning threads
        self._lazy_load_directory()

        # Deduplicate TIFF downloads: group selections by (swath, polarization)
        # since one TIFF covers one (swath, pol) and contains all bursts.
        groups: dict[tuple[str, str], set[int]] = {}
        order: list[tuple[str, str, int]] = []
        for sw, pol, bidx in selections:
            groups.setdefault((sw, pol), set()).add(bidx)
            order.append((sw, pol, bidx))

        # Submit one task per unique (swath, pol) pair
        def _download(args: tuple[str, str, set[int]]) -> dict[int, BurstArray]:
            sw, pol, indices = args
            return {i: self.extract_burst(sw, pol, burst_index=i) for i in indices}

        results: dict[tuple[str, str, int], BurstArray] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {
                pool.submit(_download, (sw, pol, idxs)): (sw, pol)
                for (sw, pol), idxs in groups.items()
            }
            for fut in as_completed(futs):
                sw, pol = futs[fut]
                bursts = fut.result()
                for bidx, b in bursts.items():
                    results[(sw, pol, bidx)] = b

        # Return in selection order
        return [results[(sw, pol, bidx)] for sw, pol, bidx in order]


def extract_remote_burst(
    url: str,
    swath_name: str,
    polarization: str,
    *,
    burst_index: int = 0,
    session: requests.Session | None = None,
) -> BurstArray:
    """Authenticate, open a remote SAFE, and extract one burst.

    One-shot convenience that wraps :class:`RemoteSafe`.  For repeated
    extractions against the same product prefer ``RemoteSafe`` directly.

    Parameters
    ----------
    url : str
        Remote SAFE ZIP URL (ASF datapool or compatible endpoint).
    swath_name : str
        Sub-swath name (``"IW1"``, ``"IW2"``, etc.).
    polarization : str
        Polarisation (``"VV"``, ``"VH"``).
    burst_index : int, optional
        Burst index within the sub-swath.
    session : requests.Session, optional
        Pre-authenticated session.

    Returns
    -------
    BurstArray
        The extracted burst.

    Examples
    --------
    >>> burst = extract_remote_burst(
    ...     "https://datapool.asf.alaska.edu/SLC/SA/S1A_...1884.zip",
    ...     "IW1",
    ...     "VV",
    ...     burst_index=2,
    ... )

    """
    safe = RemoteSafe(url, session=session)
    return safe.extract_burst(swath_name, polarization, burst_index=burst_index)
