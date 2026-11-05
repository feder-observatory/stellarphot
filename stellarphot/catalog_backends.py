"""
Sources that can answer a catalog cone search for `CatalogData.from_vizier`.

`CatalogData.from_vizier` only needs one thing from the network: a table with
the catalog's native VizieR column names for the stars in a cone. Everything
after that (``prepare_catalog``, tidying, ``colname_map``, frame clipping) runs
on that table. The backends in this module each produce such a table from a
different source, so a notebook that calls `stellarphot.catalogs.vsx_vizier`,
`stellarphot.catalogs.apass_dr9` or `stellarphot.catalogs.refcat2` keeps
working when VizieR does not.

Which backends are tried, and in what order, is controlled by environment
variables (see `default_backends`) or by passing ``backends=`` explicitly.
"""

import io
import json
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord, angular_separation
from astropy.io.votable import parse
from astropy.table import MaskedColumn, Table
from astropy.utils.exceptions import AstropyUserWarning
from astroquery.vizier import Vizier

__all__ = [
    "VIZIER_SERVERS",
    "VIZIER_TIMEOUT",
    "CatalogUnavailableError",
    "ConeQuery",
    "CatalogBackend",
    "VizierBackend",
    "LocalStoreBackend",
    "AavsoVsxBackend",
    "TapApassBackend",
    "EsaApassDr9Backend",
    "default_backends",
    "query_backends",
    "write_local_store",
    "describe_backends",
]

# VizieR servers to try, in order, when fetching a catalog. A server that is up
# but whose database is broken answers with a VOTable containing only error
# messages, which astroquery silently turns into an empty result -- so the
# Vizier backend falls through this list whenever a query comes back empty.
# The CDS server is last: during the November 2026 outage it does not even
# accept connections, so every call to it costs the full connect timeout.
# Change this tuple (or set STELLARPHOT_VIZIER_SERVERS) to steer around a
# dead mirror.
VIZIER_SERVERS = (
    "vizier.cfa.harvard.edu",
    "vizier.idia.ac.za",
    "vizier.cds.unistra.fr",  # astroquery's default, at CDS
)

# Seconds to wait for a response from a VizieR server before moving on to the
# next one. The largest healthy query we have timed (APASS DR9 over an M31
# field, cold cache) took ~23 s; a broken server can hang far longer than any
# healthy one takes, and each of these blocks the notebook kernel.
VIZIER_TIMEOUT = 30

VSX_VIZIER_ID = "B/vsx/vsx"
APASS_VIZIER_ID = "II/336/apass9"
REFCAT2_VIZIER_ID = "J/ApJ/867/105/refcat2"

# Environment variables that select backends without touching notebooks.
ENV_CATALOG_DIR = "STELLARPHOT_CATALOG_DIR"
ENV_CATALOG_SOURCES = "STELLARPHOT_CATALOG_SOURCES"
ENV_VIZIER_SERVERS = "STELLARPHOT_VIZIER_SERVERS"

DEFAULT_SOURCE_ORDER = ("local", "vizier", "vsx_api", "apass_esa")

_USER_AGENT = "stellarphot-catalog-backends"


class CatalogUnavailableError(RuntimeError):
    """
    A backend could not answer a query it nominally provides, e.g. because the
    service is down or the local store does not cover the region. The next
    backend in line is tried.
    """


@dataclass(frozen=True)
class ConeQuery:
    """
    A cone search request, in the terms VizieR uses.

    Parameters
    ----------
    catalog_id : str
        VizieR catalog identifier, e.g. ``"II/336/apass9"``. This is the key
        every backend uses, whatever its own source calls the catalog.
    center : `astropy.coordinates.SkyCoord`
        Center of the cone.
    radius : `astropy.units.Quantity`
        Angular radius of the cone.
    column_filters : dict, optional
        Filters in VizieR's ``column_filters`` syntax, keyed by native column
        name, e.g. ``{"rmag": "<=14"}``.
    """

    catalog_id: str
    center: SkyCoord
    radius: u.Quantity
    column_filters: dict = field(default_factory=dict)

    @property
    def radius_deg(self):
        return u.Quantity(self.radius, u.deg).to_value(u.deg)


class CatalogBackend:
    """
    Base class for catalog sources.

    Subclasses set ``name`` (recorded in `CatalogData.catalog_source`) and
    implement `provides` and `cone_search`.
    """

    name = "backend"

    def provides(self, catalog_id):
        """Return ``True`` if this backend can answer queries for ``catalog_id``."""
        raise NotImplementedError

    def cone_search(self, query):
        """
        Return the rows in the cone as a table with VizieR column names.

        Missing values must be masked, as astroquery masks them, and
        coordinates must be float columns in degrees. An empty table means
        "no sources here"; raise `CatalogUnavailableError` if the backend
        could not answer at all.
        """
        raise NotImplementedError

    def is_complete(self, query):  # noqa: ARG002
        """
        Return ``True`` if an empty answer to ``query`` from this backend
        means the region really is empty.

        `query_backends` stops at an empty answer from a complete backend
        and moves on from an empty answer from an incomplete one (a local
        copy shallower than the query, or a VizieR mirror that may not carry
        the catalog).
        """
        return True

    def __repr__(self):
        return f"<{type(self).__name__} {self.name}>"


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_FILTER_RE = re.compile(
    rf"^\s*(<=|>=|!=|<|>|=)?\s*({_NUMBER})\s*(?:\.\.\s*({_NUMBER}))?\s*$"
)


def _parse_filter(spec):
    """
    Parse the subset of VizieR ``column_filters`` syntax stellarphot uses:
    ``<=x``, ``<x``, ``>=x``, ``>x``, ``=x``, ``!=x`` and ``a..b``.
    Returns ``(op, a, b)`` with ``b`` only set for ranges.
    """
    match = _FILTER_RE.match(str(spec))
    if not match:
        raise ValueError(f"Unsupported column filter {spec!r}")
    op, a, b = match.groups()
    if b is not None:
        return "..", float(a), float(b)
    return op or "=", float(a), None


def _apply_column_filters(table, column_filters):
    """
    Apply VizieR-style ``column_filters`` to an astropy table, client-side.
    Masked values never pass a filter, which is what VizieR does with blanks.
    """
    if not column_filters:
        return table
    keep = np.ones(len(table), dtype=bool)
    for col, spec in column_filters.items():
        if col not in table.colnames:
            raise ValueError(f"Cannot filter on {col!r}: column not in catalog")
        op, a, b = _parse_filter(spec)
        values = np.ma.asarray(table[col])
        if values.dtype.kind == "f":
            # Compare in the column's own precision so a star sitting exactly
            # at a limit is kept (a float32 15.123 is below a float64 15.123).
            a = values.dtype.type(a)
            b = values.dtype.type(b) if b is not None else None
        else:
            values = np.ma.asarray(values, dtype=float)
        if op == "..":
            ok = (values >= a) & (values <= b)
        else:
            ok = {
                "<=": values <= a,
                "<": values < a,
                ">=": values >= a,
                ">": values > a,
                "=": values == a,
                "!=": values != a,
            }[op]
        keep &= np.ma.filled(ok, False)
    return table[keep]


def _angular_separation_deg(ra1, dec1, ra2, dec2):
    """Great-circle separation in degrees, inputs in degrees."""
    ra1, dec1, ra2, dec2 = (
        np.radians(np.asarray(x, dtype=float)) for x in (ra1, dec1, ra2, dec2)
    )
    return np.degrees(angular_separation(ra1, dec1, ra2, dec2))


def _add_distance_column(table, query, ra_col, dec_col):
    """Prepend the ``_r`` column VizieR adds to cone searches, in degrees."""
    sep = _angular_separation_deg(
        table[ra_col], table[dec_col], query.center.ra.deg, query.center.dec.deg
    )
    table.add_column(MaskedColumn(sep, name="_r", unit=u.deg), index=0)
    return table


# Row cap requested from TAP services. Their defaults are small (GAVO: 20,000)
# and a service that hits its cap truncates the table and only says so in a
# VOTable INFO element, so the cap is raised and the INFO checked.
TAP_MAXREC = 2_000_000


def _votable_to_table(content):
    """Parse a TAP VOTable response, raising on error or truncation."""
    with warnings.catch_warnings():
        # VOTable parsers are chatty about minor spec deviations.
        warnings.simplefilter("ignore")
        votable = parse(io.BytesIO(content))
    infos = list(votable.infos)
    for resource in votable.resources:
        infos.extend(resource.infos)
        for table in resource.tables:
            infos.extend(table.infos)
    for info in infos:
        if info.name != "QUERY_STATUS":
            continue
        if info.value == "ERROR":
            raise ValueError(f"TAP query failed: {info.content}")
        if info.value == "OVERFLOW":
            raise ValueError("TAP result was truncated at the service's row limit")
    return votable.get_first_table().to_table()


def _tap_payload(adql, maxrec):
    return dict(
        REQUEST="doQuery", LANG="ADQL", FORMAT="votable", MAXREC=maxrec, QUERY=adql
    )


def _tap_sync(url, adql, timeout, maxrec=TAP_MAXREC):
    """Run a synchronous TAP query and return the result as an astropy Table."""
    response = requests.post(
        url.rstrip("/") + "/sync",
        data=_tap_payload(adql, maxrec),
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    return _votable_to_table(response.content)


def _tap_async(url, adql, timeout, maxrec=TAP_MAXREC, poll_interval=3.0):
    """
    Run a TAP query as a queued (UWS) job and wait up to ``timeout`` seconds
    for it. Some services, ESA's Gaia archive among them, answer queued jobs
    when their synchronous endpoint is overloaded.

    ``timeout`` bounds the whole exchange: submitting the job, polling it
    and fetching the result. No single request waits longer than the time
    left, and a job still running at the deadline is aborted.
    """
    headers = {"User-Agent": _USER_AGENT}
    deadline = time.monotonic() + timeout

    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise requests.exceptions.Timeout(
                f"TAP job did not finish within {timeout} s"
            )
        return left

    response = requests.post(
        url.rstrip("/") + "/async",
        data=dict(_tap_payload(adql, maxrec), PHASE="RUN"),
        headers=headers,
        timeout=remaining(),
        allow_redirects=False,
    )
    response.raise_for_status()
    location = response.headers.get("Location")
    if location:
        # UWS says Location is absolute, but some services send a path.
        job_url = urljoin(response.url, location)
    else:
        match = re.search(r"<(?:uws:)?jobId>([^<]+)</", response.text)
        if match is None:
            raise ValueError("TAP service did not return a job identifier")
        job_url = url.rstrip("/") + "/async/" + match.group(1)
    job_url = job_url.rstrip("/")

    try:
        while True:
            phase_response = requests.get(
                job_url + "/phase", headers=headers, timeout=remaining()
            )
            phase_response.raise_for_status()
            phase = phase_response.text.strip()
            if phase == "COMPLETED":
                break
            if phase in ("ERROR", "ABORTED"):
                raise ValueError(f"TAP job ended in phase {phase}")
            time.sleep(min(poll_interval, remaining()))
        result = requests.get(
            job_url + "/results/result", headers=headers, timeout=remaining()
        )
    except requests.exceptions.Timeout:
        # Do not leave the job running on the service.
        try:
            requests.post(
                job_url + "/phase",
                data={"PHASE": "ABORT"},
                headers=headers,
                timeout=10,
            )
        except requests.exceptions.RequestException:
            pass
        raise
    result.raise_for_status()
    return _votable_to_table(result.content)


# --------------------------------------------------------------------------
# VizieR
# --------------------------------------------------------------------------


class VizierBackend(CatalogBackend):
    """
    Query VizieR, falling through a list of mirrors.

    A mirror that answers with an error page, a VOTable astroquery cannot
    parse, a network error or no rows at all is skipped and the next one
    tried. Good answers stay in astroquery's cache; failed ones are evicted
    from it so that a retry asks the server again instead of replaying the
    failure for a week.

    Parameters
    ----------
    servers : sequence of str, optional
        Bare hostnames to try in order. Defaults to `VIZIER_SERVERS`, looked
        up at query time so it can be changed after import.
    timeout : float, optional
        Seconds to wait on each server. Defaults to `VIZIER_TIMEOUT`.
    """

    name = "Vizier"

    def __init__(self, servers=None, timeout=None):
        self.servers = tuple(servers) if servers is not None else None
        self.timeout = timeout

    def provides(self, catalog_id):  # noqa: ARG002
        return True

    def is_complete(self, query):  # noqa: ARG002
        # Every mirror answering nothing may mean an empty region, but it may
        # also mean that none of them carries the catalog, as happened for
        # APASS during the 2026 CDS outage.
        return False

    def cone_search(self, query):
        servers = self.servers if self.servers is not None else VIZIER_SERVERS
        timeout = self.timeout if self.timeout is not None else VIZIER_TIMEOUT

        last_error = None
        for server in servers:
            vizier = Vizier(
                columns=["all"],
                row_limit=-1,
                column_filters=dict(query.column_filters),
                timeout=timeout,
                vizier_server=server,
            )
            try:
                result = vizier.query_region(
                    query.center, radius=query.radius, catalog=query.catalog_id
                )
            except Exception as err:
                # Network errors (nothing cached) and parse errors (the bad
                # answer is cached) alike mean "try the next mirror".
                last_error = err
                _evict_last_answer(vizier)
                continue
            # astroquery returns None for an HTML error page and an empty
            # TableList for a VOTable that holds only error messages.
            if result is not None and len(result) > 0:
                return result[0]
            _evict_last_answer(vizier)
        if last_error is not None:
            # Re-raise the real error so callers see the cause.
            raise last_error
        return Table()


def _evict_last_answer(vizier):
    """
    Remove the answer to ``vizier``'s last query from astroquery's cache.

    astroquery caches every response, error pages included, so without this
    a retry would replay the failure until the cache entry expires.
    """
    last_query = getattr(vizier, "_last_query", None)
    if last_query is None:
        return
    try:
        last_query.remove_cache_file(vizier.cache_location)
    except (FileNotFoundError, PermissionError):
        # Nothing was cached, or the cache is not ours to change.
        pass


# --------------------------------------------------------------------------
# Local Parquet store
# --------------------------------------------------------------------------

STORE_FORMAT = "stellarphot-catalog-store"
STORE_FORMAT_VERSION = 1
_INDEX_COLS = ("zone",)


class LocalStoreBackend(CatalogBackend):
    """
    Serve catalogs from a directory of Parquet files on local (or shared) disk.

    The directory holds one sub-directory per catalog, each with a
    ``catalog.json`` manifest and the data partitioned into one-degree
    declination zones; see `write_local_store` for the layout. Reading
    requires ``pyarrow``.

    Parameters
    ----------
    root : str or `pathlib.Path`
        Directory containing the catalog sub-directories.
    """

    name = "local"

    def __init__(self, root):
        self.root = Path(root)
        self.stores = {}
        self.name = f"local:{self.root}"
        if not self.root.is_dir():
            warnings.warn(
                f"Local catalog store {self.root} does not exist; "
                "no catalogs will be read from it.",
                AstropyUserWarning,
                stacklevel=2,
            )
            return
        for manifest in sorted(self.root.glob("*/catalog.json")):
            info = json.loads(manifest.read_text())
            info["path"] = manifest.parent / info.get("data_dir", "data")
            self.stores[info["vizier_id"]] = info
        if not self.stores:
            warnings.warn(
                f"Local catalog store {self.root} contains no catalogs "
                "(no */catalog.json found).",
                AstropyUserWarning,
                stacklevel=2,
            )

    def provides(self, catalog_id):
        return catalog_id in self.stores

    def is_complete(self, query):
        return self._covers(self.stores[query.catalog_id], query)

    def __repr__(self):
        return f"<LocalStoreBackend {self.root} catalogs={sorted(self.stores)}>"

    def cone_search(self, query):
        try:
            import pyarrow.compute as pc
            import pyarrow.dataset as ds
        except ImportError as err:
            raise CatalogUnavailableError(
                "Reading a local catalog store requires pyarrow"
            ) from err

        info = self.stores[query.catalog_id]
        ra_col, dec_col = info["ra_col"], info["dec_col"]
        zone_deg = info.get("zone_deg", 1.0)
        ra0, dec0 = query.center.icrs.ra.deg, query.center.icrs.dec.deg
        radius = query.radius_deg

        dec_min, dec_max = max(dec0 - radius, -90.0), min(dec0 + radius, 90.0)
        expr = (
            (pc.field("zone") >= int(np.floor((dec_min + 90) / zone_deg)))
            & (pc.field("zone") <= int(np.floor((dec_max + 90) / zone_deg)))
            & (pc.field(dec_col) >= dec_min)
            & (pc.field(dec_col) <= dec_max)
        )
        # An RA bounding box only makes sense away from the poles.
        if dec_max < 90 and dec_min > -90:
            half_width = radius / np.cos(np.radians(max(abs(dec_min), abs(dec_max))))
            if half_width < 180:
                lo, hi = ra0 - half_width, ra0 + half_width

                def ra_between(a, b):
                    return (pc.field(ra_col) >= a) & (pc.field(ra_col) <= b)

                if lo < 0:
                    expr &= ra_between(lo + 360, 360) | ra_between(0, hi)
                elif hi > 360:
                    expr &= ra_between(lo, 360) | ra_between(0, hi - 360)
                else:
                    expr &= ra_between(lo, hi)

        try:
            dataset = ds.dataset(info["path"], format="parquet", partitioning="hive")
            arrow_table = dataset.to_table(filter=expr)
        except Exception as err:
            # Missing directory (disk unmounted), corrupt file, schema trouble.
            raise CatalogUnavailableError(
                f"Could not read local copy of {query.catalog_id} at "
                f"{info['path']}: {err}"
            ) from err
        arrow_table = arrow_table.drop_columns(
            [c for c in arrow_table.column_names if c in _INDEX_COLS]
        )
        table = _arrow_to_masked_table(arrow_table, info.get("units", {}))
        table.meta["local_store"] = str(info["path"])
        table.meta["local_store_version"] = info.get("version")

        # Exact cut on the sphere, then the filters VizieR would have applied.
        sep = _angular_separation_deg(table[ra_col], table[dec_col], ra0, dec0)
        table = table[sep <= radius]
        table = _apply_column_filters(table, query.column_filters)

        if not self._covers(info, query):
            described = ", ".join(
                f"{col} <= {lim}" for col, lim in info["mag_limit"].items()
            )
            warnings.warn(
                f"Local copy of {info['vizier_id']} at {info['path']} only "
                f"contains sources with {described}; fainter sources are "
                "missing.",
                AstropyUserWarning,
                stacklevel=4,
            )
        return table

    @staticmethod
    def _covers(info, query):
        """
        Return ``True`` if the store holds every source the query could
        return: it has no recorded magnitude limit, or the query limits
        itself to at most that depth.
        """
        limits = info.get("mag_limit") or {}
        if not limits:
            return True
        for col, spec in query.column_filters.items():
            if col in limits:
                op, a, _ = _parse_filter(spec)
                if op in ("<=", "<") and a <= limits[col]:
                    return True
        return False


def _arrow_to_masked_table(arrow_table, units):
    """Convert a pyarrow table to a masked astropy table, nulls -> masks."""
    import pyarrow as pa

    table = Table(masked=True)
    for name in arrow_table.column_names:
        column = arrow_table[name]
        mask = column.is_null().to_numpy(zero_copy_only=False)
        if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
            values = np.array(column.fill_null("").to_pylist(), dtype=str)
        else:
            # Fill nulls before converting: pyarrow turns an integer or
            # boolean column with nulls into float64 (NaN for null), which
            # changes the dtype and rounds large integers.
            fill = False if pa.types.is_boolean(column.type) else 0
            values = column.fill_null(fill).to_numpy(zero_copy_only=False)
        table[name] = MaskedColumn(values, mask=mask, unit=units.get(name))
    return table


def write_local_store(
    table,
    root,
    name,
    vizier_id,
    ra_col,
    dec_col,
    version=None,
    mag_limit=None,
    zone_deg=1.0,
):
    """
    Write a table with VizieR column names as a local catalog store that
    `LocalStoreBackend` can serve.

    Parameters
    ----------
    table : `astropy.table.Table` or iterable of tables
        Catalog rows, with the catalog's native VizieR column names and masked
        values where VizieR leaves blanks. An iterable of tables with the same
        columns is written chunk by chunk, so a catalog too large to hold in
        memory can be streamed in.
    root : str or `pathlib.Path`
        Store directory; the catalog goes in ``root / name``.
    name : str
        Short directory name, e.g. ``"refcat2"``.
    vizier_id : str
        VizieR identifier the store answers for, e.g. ``"J/ApJ/867/105/refcat2"``.
    ra_col, dec_col : str
        Names of the coordinate columns, in degrees.
    version : str, optional
        Free-form provenance string recorded in the manifest.
    mag_limit : dict, optional
        ``{column: limit}`` describing how deep the store is complete, e.g.
        ``{"rmag": 16.0}``. Queries that go fainter get a warning.
    zone_deg : float, optional
        Height of the declination zones the data is partitioned into.

    Returns
    -------
    `pathlib.Path`
        Path to the catalog directory.

    Notes
    -----
    Layout, with ``zone = floor((dec + 90) / zone_deg)``::

        root/name/catalog.json
        root/name/data/zone=NNN/part-0.parquet

    Rows are sorted by right ascension within each zone so Parquet row-group
    statistics prune most of a zone for a small cone.
    """
    import shutil

    import pyarrow as pa
    import pyarrow.dataset as ds

    cat_dir = Path(root) / name
    data_dir = cat_dir / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    chunks = [table] if isinstance(table, Table) else table
    units = {}
    rows = 0
    for chunk_number, chunk in enumerate(chunks):
        columns = {}
        for colname in chunk.colnames:
            column = chunk[colname]
            if column.unit is not None:
                units[colname] = column.unit.to_string()
            values = np.ma.asarray(column)
            mask = np.ma.getmaskarray(values)
            data = np.ma.getdata(values)
            if data.dtype.kind in ("U", "S", "O"):
                columns[colname] = pa.array(
                    [None if m else str(v) for v, m in zip(data, mask, strict=True)],
                    type=pa.string(),
                )
            else:
                columns[colname] = pa.array(data, mask=mask)
        dec = np.asarray(chunk[dec_col], dtype=float)
        columns["zone"] = pa.array(
            np.floor((dec + 90) / zone_deg).astype(np.int16), type=pa.int16()
        )
        arrow_table = pa.table(columns).sort_by(
            [("zone", "ascending"), (ra_col, "ascending")]
        )
        ds.write_dataset(
            arrow_table,
            data_dir,
            format="parquet",
            partitioning=ds.partitioning(
                pa.schema([("zone", pa.int16())]), flavor="hive"
            ),
            basename_template=f"part-{chunk_number}-{{i}}.parquet",
            file_options=ds.ParquetFileFormat().make_write_options(compression="zstd"),
            max_rows_per_group=20_000,
            existing_data_behavior="overwrite_or_ignore",
        )
        rows += len(chunk)
    manifest = dict(
        format=STORE_FORMAT,
        format_version=STORE_FORMAT_VERSION,
        vizier_id=vizier_id,
        name=name,
        layout="dec_zone_parquet",
        zone_deg=zone_deg,
        ra_col=ra_col,
        dec_col=dec_col,
        data_dir="data",
        rows=rows,
        version=version,
        mag_limit=mag_limit,
        units=units,
    )
    (cat_dir / "catalog.json").write_text(json.dumps(manifest, indent=1))
    return cat_dir


# --------------------------------------------------------------------------
# AAVSO VSX web API
# --------------------------------------------------------------------------

_VSX_COLUMNS = [
    "OID", "n_OID", "Name", "V", "Type", "l_max", "max", "u_max", "n_max",
    "f_min", "l_min", "min", "u_min", "n_min", "l_Period", "Period",
    "u_Period", "Sp", "RAJ2000", "DEJ2000",
]  # fmt: skip

# VizieR "V" flag: 0 = variable, 1 = suspected, 2 = constant/non-existent,
# 3 = possible duplicate.
_VSX_CATEGORY_TO_V = {
    "variable": 0, "suspected": 1, "non-variable": 2, "constant": 2, "duplicate": 3,
}  # fmt: skip

# MaxMag / MinMag strings look like "14.27 V", "(0.32) CV", "<14.6 V",
# "17.0: p", "(0.453)Ic", "?" (unknown).
_VSX_MAG_RE = re.compile(
    r"^\s*(?P<lim>[<>])?\s*(?P<amp>\()?\s*(?P<val>[-+]?\d*\.?\d+)\s*\)?"
    r"\s*(?P<unc>:+)?\s*(?P<band>.*?)\s*$"
)
_VSX_PERIOD_RE = re.compile(
    r"^\s*(?P<lim>[<>])?\s*(?P<val>\d*\.?\d+)\s*(?P<unc>[:/A-Za-z]+)?\s*$"
)


def _vsx_parse_mag(text):
    """Return ``(limit_flag, amplitude_flag, value, uncertainty_flag, passband)``."""
    if not text:
        return None, None, None, None, None
    match = _VSX_MAG_RE.match(text)
    if not match:  # e.g. "?" or "? V"
        band = text.replace("?", "").strip() or None
        return None, None, None, None, band
    return (
        match["lim"],
        "Y" if match["amp"] else None,
        float(match["val"]),
        match["unc"],
        match["band"] or None,
    )


def _vsx_parse_period(text):
    if not text:
        return None, None, None
    match = _VSX_PERIOD_RE.match(text)
    if not match:
        return None, None, None
    return match["lim"], float(match["val"]), match["unc"]


def vsx_api_objects_to_table(objects):
    """
    Convert the ``VSXObject`` records of an AAVSO VSX API response into a
    table with the column names and types of VizieR's ``B/vsx/vsx``.
    """
    columns = {name: [] for name in _VSX_COLUMNS}
    for obj in objects:
        l_max, _, v_max, u_max, n_max = _vsx_parse_mag(obj.get("MaxMag"))
        l_min, f_min, v_min, u_min, n_min = _vsx_parse_mag(obj.get("MinMag"))
        l_period, period, u_period = _vsx_parse_period(obj.get("Period"))
        row = dict(
            OID=obj.get("OID"),
            n_OID=None,
            Name=obj["Name"],
            V=_VSX_CATEGORY_TO_V.get(str(obj.get("Category", "")).lower(), 0),
            Type=obj.get("VariabilityType"),
            l_max=l_max, max=v_max, u_max=u_max, n_max=n_max,
            f_min=f_min, l_min=l_min, min=v_min, u_min=u_min, n_min=n_min,
            l_Period=l_period, Period=period, u_Period=u_period,
            Sp=obj.get("SpectralType"),
            RAJ2000=float(obj["RA2000"]),
            DEJ2000=float(obj["Declination2000"]),
        )  # fmt: skip
        for key, value in row.items():
            columns[key].append(value)

    def string_column(name, masked=True):
        values = ["" if v is None else str(v) for v in columns[name]]
        if masked:
            return MaskedColumn(values, mask=[v == "" for v in values], dtype=str)
        return np.array(values, dtype=str)

    def float_column(name, dtype, unit):
        values = np.array(
            [np.nan if v is None else float(v) for v in columns[name]], dtype=dtype
        )
        mask = np.isnan(values)
        return MaskedColumn(np.where(mask, 0, values), mask=mask, unit=unit)

    table = Table(masked=True)
    oid = columns["OID"]
    table["OID"] = MaskedColumn(
        [-1 if o is None else int(o) for o in oid],
        mask=[o is None for o in oid],
        dtype=np.int32,
    )
    table["n_OID"] = string_column("n_OID")
    table["Name"] = string_column("Name", masked=False)
    table["V"] = np.array(columns["V"], dtype=np.uint8)
    table["Type"] = string_column("Type", masked=False)
    table["l_max"] = string_column("l_max")
    table["max"] = float_column("max", np.float32, u.mag)
    table["u_max"] = string_column("u_max")
    table["n_max"] = string_column("n_max", masked=False)
    table["f_min"] = string_column("f_min")
    table["l_min"] = string_column("l_min")
    table["min"] = float_column("min", np.float32, u.mag)
    table["u_min"] = string_column("u_min")
    table["n_min"] = string_column("n_min", masked=False)
    table["l_Period"] = string_column("l_Period")
    table["Period"] = float_column("Period", np.float64, u.day)
    table["u_Period"] = string_column("u_Period")
    table["Sp"] = string_column("Sp")
    table["RAJ2000"] = MaskedColumn(
        np.array(columns["RAJ2000"], dtype=float), unit=u.deg
    )
    table["DEJ2000"] = MaskedColumn(
        np.array(columns["DEJ2000"], dtype=float), unit=u.deg
    )
    return table


class AavsoVsxBackend(CatalogBackend):
    """
    Query the AAVSO Variable Star Index directly through its web API.

    Returns the same objects as VizieR's ``B/vsx/vsx`` with VizieR's column
    names, from AAVSO's live database.

    Parameters
    ----------
    url : str, optional
        API endpoint. The ``www.aavso.org`` host refuses scripted clients;
        ``vsx.aavso.org`` does not.
    timeout : float, optional
        Seconds to wait for a response.
    """

    name = "AAVSO VSX API"
    url = "https://vsx.aavso.org/index.php"

    def __init__(self, url=None, timeout=90):
        if url is not None:
            self.url = url
        self.timeout = timeout

    def provides(self, catalog_id):
        return catalog_id == VSX_VIZIER_ID

    def cone_search(self, query):
        center = query.center.icrs
        params = dict(
            view="api.list",
            ra=f"{center.ra.deg:.6f}",
            dec=f"{center.dec.deg:+.6f}",
            radius=f"{query.radius_deg:.6f}",  # in degrees
            format="json",
        )
        try:
            response = requests.get(
                self.url,
                params=params,
                headers={"User-Agent": _USER_AGENT},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json() if response.content else {}
        except (requests.exceptions.RequestException, ValueError) as err:
            raise CatalogUnavailableError(f"AAVSO VSX API failed: {err}") from err
        objects = (payload.get("VSXObjects") or {}).get("VSXObject", [])
        if isinstance(objects, dict):  # a single match comes back as an object
            objects = [objects]
        table = vsx_api_objects_to_table(objects)
        return _apply_column_filters(table, query.column_filters)


# --------------------------------------------------------------------------
# APASS over TAP (ESA DR9)
# --------------------------------------------------------------------------

# ESA ``external.apassdr9`` column -> VizieR II/336/apass9 column. The ESA
# table is a straight copy of the VizieR one with lower-cased names.
_ESA_APASS_COLUMNS = {
    "raj2000": "RAJ2000", "dej2000": "DEJ2000",
    "e_ra": "e_RAJ2000", "e_dec": "e_DEJ2000",
    "field": "Field", "nobs": "nobs", "mobs": "mobs",
    "b_v": "B-V", "e_b_v": "e_B-V",
    "vmag": "Vmag", "e_vmag": "e_Vmag", "u_e_vmag": "u_e_Vmag",
    "bmag": "Bmag", "e_bmag": "e_Bmag", "u_e_bmag": "u_e_Bmag",
    "g_mag": "g'mag", "e_g_mag": "e_g'mag", "u_e_g_mag": "u_e_g'mag",
    "r_mag": "r'mag", "e_r_mag": "e_r'mag", "u_e_r_mag": "u_e_r'mag",
    "i_mag": "i'mag", "e_i_mag": "e_i'mag", "u_e_i_mag": "u_e_i'mag",
}  # fmt: skip


class TapApassBackend(CatalogBackend):
    """
    Base class for APASS served by a TAP service. Subclasses supply the
    service URL and the conversion of the service's table to VizieR's
    ``II/336/apass9`` column names.
    """

    name = "APASS TAP"
    url = None
    table_name = None
    # "sync" posts to the service's /sync endpoint; "async" queues a job and
    # polls it, which is slower to start but survives an overloaded service.
    mode = "sync"
    timeout = 120

    def __init__(self, url=None, timeout=None, mode=None):
        if url is not None:
            self.url = url
        if timeout is not None:
            self.timeout = timeout
        if mode is not None:
            self.mode = mode

    def provides(self, catalog_id):
        return catalog_id == APASS_VIZIER_ID

    def adql(self, query):
        raise NotImplementedError

    def to_vizier(self, table):
        raise NotImplementedError

    def cone_search(self, query):
        run = _tap_async if self.mode == "async" else _tap_sync
        try:
            raw = run(self.url, self.adql(query), self.timeout)
        except (requests.exceptions.RequestException, ValueError) as err:
            raise CatalogUnavailableError(f"{self.name} failed: {err}") from err
        table = self.to_vizier(raw)
        # A row without a position cannot be matched to anything.
        has_position = ~(
            np.ma.getmaskarray(table["RAJ2000"]) | np.ma.getmaskarray(table["DEJ2000"])
        )
        table = table[has_position]
        table = _add_distance_column(table, query, "RAJ2000", "DEJ2000")
        return _apply_column_filters(table, query.column_filters)


class EsaApassDr9Backend(TapApassBackend):
    """
    APASS DR9 from the ESA Gaia archive's copy (``external.apassdr9``), which
    is the VizieR table with lower-cased column names. Identical content to
    VizieR. The service's synchronous endpoint is often overloaded, so the
    query is run as a queued job, which typically takes 10-20 s.
    """

    name = "ESA APASS DR9"
    url = "https://gea.esac.esa.int/tap-server/tap"
    table_name = "external.apassdr9"
    mode = "async"
    timeout = 180

    def adql(self, query):
        center = query.center.icrs
        columns = ", ".join(_ESA_APASS_COLUMNS)
        return (
            f"SELECT {columns} FROM {self.table_name} "
            f"WHERE 1=CONTAINS(POINT('ICRS', raj2000, dej2000), "
            f"CIRCLE('ICRS', {center.ra.deg:.8f}, {center.dec.deg:.8f}, "
            f"{query.radius_deg:.8f}))"
        )

    def to_vizier(self, raw):
        out = Table(masked=True)
        for src, dst in _ESA_APASS_COLUMNS.items():
            column = raw[src]
            if dst in ("RAJ2000", "DEJ2000"):
                out[dst] = MaskedColumn(np.ma.asarray(column, dtype=float), unit=u.deg)
            else:
                out[dst] = MaskedColumn(column, unit=column.unit)
        return out


# --------------------------------------------------------------------------
# Selection and dispatch
# --------------------------------------------------------------------------

_SOURCE_FACTORIES = {
    "vizier": lambda: [VizierBackend(servers=_vizier_servers_from_env())],
    "vsx_api": lambda: [AavsoVsxBackend()],
    "apass_esa": lambda: [EsaApassDr9Backend()],
    "local": lambda: [
        LocalStoreBackend(root)
        for root in os.environ.get(ENV_CATALOG_DIR, "").split(os.pathsep)
        if root
    ],
}


def _vizier_servers_from_env():
    value = os.environ.get(ENV_VIZIER_SERVERS, "")
    servers = tuple(s.strip() for s in value.split(",") if s.strip())
    return servers or None


# The backend list built from the environment, keyed by the variables it was
# built from, so that a notebook querying once per image does not re-scan
# the local store directories and re-read their manifests every time.
_default_backends_cache = (None, None)


def default_backends():
    """
    Build the backend list from the environment.

    The list is cached and rebuilt only when one of the variables changes.

    Environment variables
    ---------------------
    ``STELLARPHOT_CATALOG_SOURCES``
        Comma-separated, ordered list of sources to try, from ``local``,
        ``vizier``, ``vsx_api`` and ``apass_esa``. Default:
        ``local,vizier,vsx_api,apass_esa``.
    ``STELLARPHOT_CATALOG_DIR``
        One or more local store directories (separated by ``os.pathsep``)
        for the ``local`` source; without it ``local`` is skipped.
    ``STELLARPHOT_VIZIER_SERVERS``
        Comma-separated VizieR hostnames to try, overriding `VIZIER_SERVERS`.
    """
    global _default_backends_cache

    key = tuple(
        os.environ.get(name, "")
        for name in (ENV_CATALOG_SOURCES, ENV_CATALOG_DIR, ENV_VIZIER_SERVERS)
    )
    if _default_backends_cache[0] == key:
        return _default_backends_cache[1]

    order = key[0]
    names = [n.strip() for n in order.split(",") if n.strip()] or list(
        DEFAULT_SOURCE_ORDER
    )
    backends = []
    for name in names:
        try:
            factory = _SOURCE_FACTORIES[name]
        except KeyError:
            raise ValueError(
                f"Unknown catalog source {name!r} in {ENV_CATALOG_SOURCES}; "
                f"choose from {', '.join(_SOURCE_FACTORIES)}"
            ) from None
        backends.extend(factory())
    _default_backends_cache = (key, backends)
    return backends


def query_backends(query, backends=None):
    """
    Run ``query`` against each backend that provides its catalog until one
    answers.

    A backend that raises is skipped and the error recorded. An empty answer
    from a backend whose `~CatalogBackend.is_complete` is true for the query
    ends the search, so that an empty field does not cost a round of every
    remaining source; an empty answer from an incomplete backend (a local
    copy shallower than the query, a VizieR mirror) is passed over.

    Parameters
    ----------
    query : `ConeQuery`
        The cone search.
    backends : list of `CatalogBackend`, optional
        Backends to try, in order. Defaults to `default_backends`.

    Returns
    -------
    table, source : `astropy.table.Table`, str
        The rows and the ``name`` of the backend that produced them.

    Raises
    ------
    RuntimeError
        If no backend returned any rows.
    """
    backends = default_backends() if backends is None else list(backends)
    attempts = []
    tried = []
    last_error = None
    for backend in backends:
        if not backend.provides(query.catalog_id):
            continue
        tried.append(backend.name)
        try:
            table = backend.cone_search(query)
        except Exception as err:
            # Whatever went wrong in this backend (service down, schema
            # change, unreadable file), the others still get their turn.
            attempts.append(f"{backend.name}: {type(err).__name__}: {err}")
            last_error = err
            continue
        if len(table) > 0:
            return table, backend.name
        attempts.append(f"{backend.name}: no sources")
        if backend.is_complete(query):
            break
    detail = "; ".join(attempts) if attempts else "no backend provides this catalog"
    raise RuntimeError(
        f"No results returned from {', '.join(tried) or 'any source'} for "
        f"catalog {query.catalog_id} around {query.center}. The region may "
        f"contain no matching sources, or the servers may be unavailable. "
        f"Tried: {detail}"
    ) from last_error


def describe_backends(backends=None):
    """
    Return a human-readable description of which backend would answer each
    of the catalogs stellarphot uses, for debugging a configuration.
    """
    backends = default_backends() if backends is None else list(backends)
    lines = []
    for catalog_id in (VSX_VIZIER_ID, APASS_VIZIER_ID, REFCAT2_VIZIER_ID):
        providers = [b.name for b in backends if b.provides(catalog_id)]
        lines.append(f"{catalog_id}: {' -> '.join(providers) or 'nothing'}")
    return "\n".join(lines)
