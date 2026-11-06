import json
import shutil
import warnings

import numpy as np
import pyarrow as pa
import pytest
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.votable import from_table, writeto
from astropy.nddata import CCDData
from astropy.table import Column, MaskedColumn, Table
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS
from astropy.wcs.wcs import FITSFixedWarning
from astroquery.exceptions import TableParseError

from stellarphot import catalog_backends
from stellarphot.catalog_backends import (
    AavsoVsxBackend,
    CatalogBackend,
    CatalogUnavailableError,
    ConeQuery,
    EsaApassDr9Backend,
    LocalStoreBackend,
    VizierBackend,
    _apply_column_filters,
    _arrow_to_masked_table,
    _filter_expression,
    _parse_filter,
    _tap_async,
    default_backends,
    describe_backends,
    query_backends,
    vsx_api_objects_to_table,
    write_local_store,
)
from stellarphot.catalogs import apass_dr9, refcat2, vsx_vizier

_QUERY_REGION = "astroquery.vizier.core.VizierClass.query_region"


def _ey_uma_header():
    # Same WCS/header construction used by test_catalogs.py.
    CCD_SHAPE = [2048, 3073]
    wcs_file = get_pkg_data_filename("data/sample_wcs_ey_uma.fits")
    with fits.open(wcs_file) as hdulist:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The WCS transformation has more",
                category=FITSFixedWarning,
            )
            wcs = WCS(hdulist[0].header)
    wcs.pixel_shape = list(reversed(CCD_SHAPE))
    ccd = CCDData(data=np.zeros(CCD_SHAPE), wcs=wcs, unit="adu")
    return ccd.to_hdu()[0].header


def _ey_uma_center():
    header = _ey_uma_header()
    wcs = WCS(header)
    return SkyCoord(*wcs.wcs.crval, unit="deg")


def _raw_refcat2():
    # The fixture holds refcat2 rows that already survived _process_refcat2,
    # so nothing in it is masked. Mask a few values the way VizieR would for
    # galaxies (no proper-motion error) and non-Gaia stars (no G error) so
    # the mask round trip, which refcat2() keys on, is exercised.
    raw = Table(
        Table.read(get_pkg_data_filename("data/all_refcat2_ey_uma.ecsv")),
        masked=True,
    )
    raw.remove_columns(["_r", "id"])  # added by VizieR / by an earlier run
    raw["e_pmRA"].mask[:2] = True
    raw["e_pmDE"].mask[:2] = True
    raw["e_Gmag"].mask[2:4] = True
    raw["pmRA"].mask[5] = True
    return raw


def _raw_apass():
    return Table.read(get_pkg_data_filename("data/all_apass_ey_uma.ecsv"))


def _query(center, radius_deg, catalog_id="J/ApJ/867/105/refcat2", **filters):
    return ConeQuery(
        catalog_id=catalog_id,
        center=center,
        radius=radius_deg * u.deg,
        column_filters=filters,
    )


def _xfail_if_server_down(err, service):
    """
    Turn a failure caused by an unreachable service into an xfail.

    Parameters
    ----------
    err : Exception
        The exception raised by the live query.
    service : str
        Name of the service, used in the xfail reason.

    Raises
    ------
    Exception
        ``err`` itself, if no exception in its ``__cause__`` chain is a
        network error or a `CatalogUnavailableError`.
    """
    cause = err
    while cause is not None:
        if isinstance(
            cause, (requests.exceptions.RequestException, CatalogUnavailableError)
        ):
            pytest.xfail(f"{service} unreachable: {err}")
        cause = cause.__cause__
    raise err


def test_xfail_if_server_down_xfails_on_chained_network_error():
    try:
        try:
            raise requests.exceptions.ConnectionError("down")
        except requests.exceptions.ConnectionError as inner:
            raise RuntimeError("no results from any source") from inner
    except RuntimeError as err:
        with pytest.raises(pytest.xfail.Exception, match="Some service unreachable"):
            _xfail_if_server_down(err, "Some service")


def test_xfail_if_server_down_reraises_unrelated_errors():
    with pytest.raises(ValueError, match="a real bug"):
        _xfail_if_server_down(ValueError("a real bug"), "Some service")


def test_cone_query_normalizes_center_to_icrs():
    galactic = SkyCoord(l=150 * u.deg, b=40 * u.deg, frame="galactic")
    query = _query(galactic, 0.5)
    assert query.center.frame.name == "icrs"
    icrs = galactic.icrs
    assert query.center.ra.deg == pytest.approx(icrs.ra.deg)
    assert query.center.dec.deg == pytest.approx(icrs.dec.deg)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("<=14", ("<=", 14.0, None)),
        ("<14", ("<", 14.0, None)),
        (">=3", (">=", 3.0, None)),
        (" > 3.5 ", (">", 3.5, None)),
        ("=2", ("=", 2.0, None)),
        ("2", ("=", 2.0, None)),
        ("!=2", ("!=", 2.0, None)),
        ("10..12", ("..", 10.0, 12.0)),
    ],
)
def test_parse_filter(spec, expected):
    assert _parse_filter(spec) == expected


def test_parse_filter_rejects_garbage():
    with pytest.raises(ValueError, match="Unsupported column filter"):
        _parse_filter("<= 14 mag")


def test_apply_column_filters_masks_never_pass():
    table = Table(
        {
            "rmag": MaskedColumn(
                [10.0, 15.0, 20.0, 12.0], mask=[False, False, False, True]
            )
        }
    )
    kept = _apply_column_filters(table, {"rmag": "<=15"})
    assert list(kept["rmag"]) == [10.0, 15.0]
    kept = _apply_column_filters(table, {"rmag": "11..16"})
    assert list(kept["rmag"]) == [15.0]
    assert len(_apply_column_filters(table, {})) == 4
    with pytest.raises(ValueError, match="not in catalog"):
        _apply_column_filters(table, {"nope": "<1"})


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class _Fake(CatalogBackend):
    def __init__(
        self,
        name,
        result,
        catalog_ids=("J/ApJ/867/105/refcat2",),
        complete=False,
    ):
        self.name = name
        self.result = result
        self.catalog_ids = catalog_ids
        self.complete = complete
        self.calls = 0

    def provides(self, catalog_id):
        return catalog_id in self.catalog_ids

    def is_complete(self, query):  # noqa: ARG002
        return self.complete

    def cone_search(self, query):  # noqa: ARG002
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_query_backends_falls_through_to_first_with_rows():
    rows = Table({"RA_ICRS": [1.0]})
    down = _Fake("down", CatalogUnavailableError("nope"))
    empty = _Fake("empty", Table())
    other = _Fake("other-catalog", rows, catalog_ids=("B/vsx/vsx",))
    good = _Fake("good", rows)
    never = _Fake("never", rows)

    table, source = query_backends(
        _query(SkyCoord(0, 0, unit="deg"), 0.1), [down, empty, other, good, never]
    )
    assert source == "good"
    assert len(table) == 1
    assert (down.calls, empty.calls, other.calls, good.calls, never.calls) == (
        1,
        1,
        0,
        1,
        0,
    )


def test_query_backends_reports_every_attempt():
    down = _Fake("down", CatalogUnavailableError("server on fire"))
    empty = _Fake("empty", Table())
    with pytest.raises(
        RuntimeError, match="No results returned from down, empty"
    ) as exc:
        query_backends(_query(SkyCoord(0, 0, unit="deg"), 0.1), [down, empty])
    assert "server on fire" in str(exc.value)
    assert "empty: no sources" in str(exc.value)


def test_query_backends_no_provider():
    only_vsx = _Fake("vsx-only", Table({"a": [1]}), catalog_ids=("B/vsx/vsx",))
    with pytest.raises(RuntimeError, match="no backend provides"):
        query_backends(_query(SkyCoord(0, 0, unit="deg"), 0.1), [only_vsx])


def test_query_backends_falls_through_on_unexpected_error():
    # A backend that blows up on a schema change (KeyError), a corrupt file
    # (any pyarrow error) or a bug must not take the whole lookup down with
    # it: the next backend is tried and the error is reported.
    rows = Table({"RA_ICRS": [1.0]})
    broken = _Fake("broken", KeyError("RA2000"))
    good = _Fake("good", rows)
    table, source = query_backends(
        _query(SkyCoord(0, 0, unit="deg"), 0.1), [broken, good]
    )
    assert source == "good"
    assert len(table) == 1

    last = ValueError("Cannot filter on 'rmag'")
    also_broken = _Fake("also-broken", last)
    with pytest.raises(RuntimeError, match="No results returned") as exc:
        query_backends(_query(SkyCoord(0, 0, unit="deg"), 0.1), [broken, also_broken])
    assert "broken: KeyError: 'RA2000'" in str(exc.value)
    assert "also-broken: ValueError: Cannot filter" in str(exc.value)
    assert exc.value.__cause__ is last


def test_query_backends_stops_at_authoritative_empty():
    # An empty answer from a backend that holds the whole catalog to the
    # requested depth is the answer; later backends are not consulted, so a
    # field with no VSX variables does not wait on three dead mirrors.
    empty = _Fake("complete-empty", Table(), complete=True)
    never = _Fake("never", Table({"RA_ICRS": [1.0]}))
    with pytest.raises(RuntimeError, match="complete-empty: no sources") as exc:
        query_backends(_query(SkyCoord(0, 0, unit="deg"), 0.1), [empty, never])
    assert never.calls == 0
    assert "never" not in str(exc.value)

    # The default for a backend is "complete".
    assert CatalogBackend().is_complete(_query(SkyCoord(0, 0, unit="deg"), 0.1))


def test_default_backends_is_cached_per_environment(monkeypatch, tmp_path):
    first = default_backends()
    assert default_backends() is first
    monkeypatch.setenv("STELLARPHOT_CATALOG_SOURCES", "vizier,vsx_api")
    second = default_backends()
    assert second is not first
    assert [type(b) for b in second] == [VizierBackend, AavsoVsxBackend]
    assert default_backends() is second
    # Any of the three variables invalidates the cache.
    monkeypatch.setenv("STELLARPHOT_VIZIER_SERVERS", "a.example.com")
    third = default_backends()
    assert third is not second
    assert third[0].servers == ("a.example.com",)
    monkeypatch.setenv("STELLARPHOT_CATALOG_SOURCES", "local,vizier")
    monkeypatch.setenv("STELLARPHOT_CATALOG_DIR", str(tmp_path))
    with pytest.warns(AstropyUserWarning, match="no catalogs"):
        fourth = default_backends()
    assert fourth is not third


def test_default_backends_is_an_immutable_tuple_rebuilt_after_cache_clear():
    first = default_backends()
    assert isinstance(first, tuple)
    with pytest.raises(TypeError):
        first[0] = AavsoVsxBackend()
    catalog_backends._backends_for.cache_clear()
    second = default_backends()
    assert second is not first
    assert [type(b) for b in second] == [type(b) for b in first]


def test_default_backends_from_environment(monkeypatch, tmp_path):
    # The autouse conftest fixture pins the sources to "vizier".
    assert [type(b) for b in default_backends()] == [VizierBackend]

    monkeypatch.setenv("STELLARPHOT_CATALOG_SOURCES", "vsx_api, apass_esa,vizier")
    monkeypatch.setenv("STELLARPHOT_VIZIER_SERVERS", "a.example.com, b.example.com")
    backends = default_backends()
    assert [type(b) for b in backends] == [
        AavsoVsxBackend,
        EsaApassDr9Backend,
        VizierBackend,
    ]
    assert backends[-1].servers == ("a.example.com", "b.example.com")

    # "local" expands to one backend per directory, and to nothing when the
    # directory variable is unset.
    monkeypatch.setenv("STELLARPHOT_CATALOG_SOURCES", "local,vizier")
    assert [type(b) for b in default_backends()] == [VizierBackend]
    monkeypatch.setenv("STELLARPHOT_CATALOG_DIR", str(tmp_path))
    with pytest.warns(AstropyUserWarning, match="no catalogs"):
        backends = default_backends()
    assert [type(b) for b in backends] == [LocalStoreBackend, VizierBackend]
    assert backends[0].root == tmp_path

    monkeypatch.setenv("STELLARPHOT_CATALOG_SOURCES", "vizier,bogus")
    with pytest.raises(ValueError, match="Unknown catalog source 'bogus'"):
        default_backends()


def test_describe_backends_lists_providers_in_order():
    text = describe_backends([AavsoVsxBackend(), VizierBackend(), EsaApassDr9Backend()])
    assert "B/vsx/vsx: AAVSO VSX API -> Vizier" in text
    assert "II/336/apass9: Vizier -> ESA APASS DR9" in text
    assert "J/ApJ/867/105/refcat2: Vizier" in text


def test_from_vizier_records_backend_as_source(mocker):
    raw = _raw_apass()
    query = mocker.patch(_QUERY_REGION, return_value=[raw])
    cat = apass_dr9(_ey_uma_header(), radius=10 * u.arcmin)
    assert query.called
    assert cat.catalog_source == "Vizier"

    cat = apass_dr9(
        _ey_uma_header(),
        radius=10 * u.arcmin,
        backends=[_Fake("canned", raw, ("II/336/apass9",))],
    )
    assert cat.catalog_source == "canned"
    assert query.call_count == 1


# ---------------------------------------------------------------------------
# VizieR
# ---------------------------------------------------------------------------


class _FakeLastQuery:
    def __init__(self):
        self.removed = 0

    def remove_cache_file(self, cache_location):  # noqa: ARG002
        self.removed += 1


class _FakeVizier:
    """
    Stand-in for astroquery's Vizier: one canned answer per server, which may
    be a list of tables, ``None`` (what astroquery returns for an HTML error
    page) or an exception to raise. Records the servers queried, the
    ``cache`` keyword and cache evictions.
    """

    results_by_server = {}
    servers_queried = []
    cache_kwargs = []
    last_queries = {}

    cache_location = "/nonexistent/astroquery/cache"

    def __init__(self, *_args, vizier_server=None, **_kwargs):
        self.vizier_server = vizier_server
        _FakeVizier.last_queries[vizier_server] = _FakeLastQuery()

    def query_region(self, *_args, **kwargs):
        _FakeVizier.servers_queried.append(self.vizier_server)
        _FakeVizier.cache_kwargs.append(kwargs.get("cache"))
        result = _FakeVizier.results_by_server[self.vizier_server]
        if isinstance(result, requests.exceptions.RequestException):
            raise result
        # Like astroquery, the query is recorded (and its answer cached) once
        # the HTTP exchange succeeded, even if parsing the answer then fails.
        self._last_query = _FakeVizier.last_queries[self.vizier_server]
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def fake_vizier(monkeypatch):
    monkeypatch.setattr(catalog_backends, "Vizier", _FakeVizier)
    _FakeVizier.results_by_server = {}
    _FakeVizier.servers_queried = []
    _FakeVizier.cache_kwargs = []
    _FakeVizier.last_queries = {}
    return _FakeVizier


def test_vizier_backend_skips_mirrors_that_answer_garbage(fake_vizier):
    # An HTML error page (astroquery returns None), a VOTable it cannot parse
    # (TableParseError) and an empty answer all mean "try the next mirror".
    good = Table({"RAJ2000": [1.0]})
    fake_vizier.results_by_server = {
        "html.example.com": None,
        "broken.example.com": TableParseError("Failed to parse VIZIER result!"),
        "empty.example.com": [],
        "good.example.com": [good],
    }
    backend = VizierBackend(servers=tuple(fake_vizier.results_by_server))
    table = backend.cone_search(_query(SkyCoord(0, 0, unit="deg"), 0.1))
    assert len(table) == 1
    assert fake_vizier.servers_queried == list(fake_vizier.results_by_server)


def test_vizier_backend_keeps_cache_but_evicts_failed_answers(fake_vizier):
    fake_vizier.results_by_server = {
        "html.example.com": None,
        "empty.example.com": [],
        "broken.example.com": TableParseError("nope"),
        "timeout.example.com": requests.exceptions.ReadTimeout("slow"),
        "good.example.com": [Table({"RAJ2000": [1.0]})],
    }
    backend = VizierBackend(servers=tuple(fake_vizier.results_by_server))
    backend.cone_search(_query(SkyCoord(0, 0, unit="deg"), 0.1))
    # Successful answers stay cached, so re-running a notebook on the same
    # field does not download the catalog again.
    assert all(c is not False for c in fake_vizier.cache_kwargs)
    removed = {s: q.removed for s, q in fake_vizier.last_queries.items()}
    assert removed == {
        "html.example.com": 1,
        "empty.example.com": 1,
        "broken.example.com": 1,
        "timeout.example.com": 0,  # nothing was cached
        "good.example.com": 0,
    }


def test_vizier_backend_all_mirrors_fail(fake_vizier):
    timeout = requests.exceptions.ReadTimeout("slow")
    fake_vizier.results_by_server = {
        "empty.example.com": [],
        "timeout.example.com": timeout,
        "html.example.com": None,
    }
    backend = VizierBackend(servers=tuple(fake_vizier.results_by_server))
    query = _query(SkyCoord(0, 0, unit="deg"), 0.1)
    # The last network error is what callers see.
    with pytest.raises(requests.exceptions.ReadTimeout):
        backend.cone_search(query)
    # Without a network error, every mirror answering nothing is an empty
    # table, which is not authoritative: a mirror may simply lack the catalog.
    fake_vizier.results_by_server = {"empty.example.com": [], "html.example.com": None}
    backend = VizierBackend(servers=tuple(fake_vizier.results_by_server))
    assert len(backend.cone_search(query)) == 0
    assert not backend.is_complete(query)


def test_vizier_backend_reads_module_defaults(fake_vizier, monkeypatch):
    # Users steer around a dead mirror by changing the list in this module.
    monkeypatch.setattr(catalog_backends, "VIZIER_SERVERS", ("only.example.com",))
    fake_vizier.results_by_server = {"only.example.com": [Table({"RAJ2000": [1.0]})]}
    VizierBackend().cone_search(_query(SkyCoord(0, 0, unit="deg"), 0.1))
    assert fake_vizier.servers_queried == ["only.example.com"]


# ---------------------------------------------------------------------------
# Local store
# ---------------------------------------------------------------------------


@pytest.fixture
def refcat2_store(tmp_path):
    write_local_store(
        _raw_refcat2(),
        tmp_path,
        "refcat2",
        "J/ApJ/867/105/refcat2",
        "RA_ICRS",
        "DE_ICRS",
        version="test fixture",
    )
    return tmp_path


def _manifest(root, name):
    return json.loads((root / name / "catalog.json").read_text())


def _data_dir(root, name):
    return root / name / _manifest(root, name)["data_dir"]


def _stars(names, ras, decs):
    return Table({"Name": names, "RAJ2000": ras, "DEJ2000": decs})


def _write_stars(stars, root):
    return write_local_store(stars, root, "stars", "test/stars", "RAJ2000", "DEJ2000")


def _star_names(backend, ra, dec, radius):
    table = backend.cone_search(
        _query(SkyCoord(ra, dec, unit="deg"), radius, catalog_id="test/stars")
    )
    return set(table["Name"])


def _assert_tables_equal(expected_table, table):
    """Assert same columns, rows in the same order, masks, values and units."""
    assert table.colnames == expected_table.colnames
    assert len(table) == len(expected_table)
    for colname in expected_table.colnames:
        expected = np.ma.asarray(expected_table[colname])
        got = np.ma.asarray(table[colname])
        np.testing.assert_array_equal(
            np.ma.getmaskarray(expected), np.ma.getmaskarray(got), err_msg=colname
        )
        if expected.dtype.kind in "fiu":
            np.testing.assert_array_equal(
                expected.filled(0), got.filled(0), err_msg=colname
            )
        else:
            assert list(expected.filled("")) == list(got.filled("")), colname
        assert table[colname].unit == expected_table[colname].unit, colname


def test_write_local_store_layout_and_manifest(refcat2_store):
    manifest = _manifest(refcat2_store, "refcat2")
    assert manifest["vizier_id"] == "J/ApJ/867/105/refcat2"
    assert manifest["rows"] == len(_raw_refcat2())
    assert manifest["units"]["pmRA"] == "mas / yr"
    assert manifest["units"]["RA_ICRS"] == "deg"
    parts = sorted(
        p.parent.name
        for p in _data_dir(refcat2_store, "refcat2").glob("zone=*/*.parquet")
    )
    # The EY UMa field straddles Dec = +50, so the rows land in two zones.
    assert parts == ["zone=139", "zone=140"]
    assert manifest["zones"] == [139, 140]


# Three stars in zones 100, 101 and 102 (zone = floor(dec + 90)).
_THREE_ZONES = _stars(["z100", "z101", "z102"], [30.0, 30.0, 30.0], [10.5, 11.5, 12.5])


@pytest.mark.parametrize("zones_in_manifest", [True, False])
def test_local_store_missing_zone_is_unavailable(tmp_path, zones_in_manifest):
    _write_stars(_THREE_ZONES, tmp_path)
    if not zones_in_manifest:
        # Manifests written before zones were recorded; zones come from disk.
        manifest = _manifest(tmp_path, "stars")
        del manifest["zones"]
        (tmp_path / "stars" / "catalog.json").write_text(json.dumps(manifest))
    shutil.rmtree(_data_dir(tmp_path, "stars") / "zone=101")
    backend = LocalStoreBackend(tmp_path)
    expected_zones = {100, 101, 102} if zones_in_manifest else {100, 102}
    assert backend.stores["test/stars"]["zones"] == expected_zones

    with pytest.raises(CatalogUnavailableError, match=r"zone.*101"):
        _star_names(backend, 30.0, 11.5, 0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Zones that are all present still answer...
        assert _star_names(backend, 30.0, 12.5, 0.2) == {"z102"}
        # ...and so does a cone outside the zones the store was written for.
        assert _star_names(backend, 30.0, -30.0, 0.2) == set()
        assert _star_names(backend, 30.0, 60.0, 0.2) == set()


def test_local_store_zone_deleted_after_opening_is_unavailable(tmp_path):
    _write_stars(_THREE_ZONES, tmp_path)
    backend = LocalStoreBackend(tmp_path)
    assert _star_names(backend, 30.0, 11.5, 0.2) == {"z101"}
    shutil.rmtree(_data_dir(tmp_path, "stars") / "zone=101")
    with pytest.raises(CatalogUnavailableError, match=r"zone.*101"):
        _star_names(backend, 30.0, 11.5, 0.2)


def test_write_local_store_rebuild_replaces_data(tmp_path):
    _write_stars(_stars(["old"], [30.0], [10.5]), tmp_path)
    old_backend = LocalStoreBackend(tmp_path)
    old_path = old_backend.stores["test/stars"]["path"]
    assert old_path.is_dir()

    _write_stars(_stars(["new"], [30.0], [10.5]), tmp_path)
    assert _manifest(tmp_path, "stars")["data_dir"] != old_path.name
    assert not old_path.exists()
    assert [p.name for p in (tmp_path / "stars").glob("data*")] == [
        _manifest(tmp_path, "stars")["data_dir"]
    ]
    assert _star_names(LocalStoreBackend(tmp_path), 30.0, 10.5, 0.2) == {"new"}


def test_write_local_store_failed_rebuild_keeps_old_store(tmp_path):
    _write_stars(_stars(["old"], [30.0], [10.5]), tmp_path)
    manifest_before = (tmp_path / "stars" / "catalog.json").read_text()
    backend = LocalStoreBackend(tmp_path)

    def chunks():
        yield _stars(["new"], [30.0], [10.5])
        raise RuntimeError("source catalog went away")

    with pytest.raises(RuntimeError, match="went away"):
        _write_stars(chunks(), tmp_path)
    assert (tmp_path / "stars" / "catalog.json").read_text() == manifest_before
    # The partial new data directory is cleaned up.
    assert [p.name for p in (tmp_path / "stars").glob("data*")] == [
        backend.stores["test/stars"]["path"].name
    ]
    assert _star_names(backend, 30.0, 10.5, 0.2) == {"old"}


def test_write_local_store_big_endian_and_bytes_columns(tmp_path):
    # Tables read from FITS have big-endian numbers and byte-string text.
    stars = Table(
        [
            Column(np.array([b"EY UMa", b"star 2"], dtype="S"), name="Name"),
            Column(np.array([30.0, 30.1], dtype=">f8"), name="RAJ2000"),
            Column(np.array([10.5, 10.6], dtype=">f8"), name="DEJ2000"),
            MaskedColumn(
                np.array([12.5, 13.5], dtype=">f4"), mask=[False, True], name="Vmag"
            ),
            Column(np.array([7, 2**40], dtype=">i8"), name="id"),
        ]
    )
    _write_stars(stars, tmp_path)
    table = LocalStoreBackend(tmp_path).cone_search(
        _query(SkyCoord(30.0, 10.5, unit="deg"), 1.0, catalog_id="test/stars")
    )
    table.sort("RAJ2000")
    assert list(table["Name"]) == ["EY UMa", "star 2"]
    for colname in ["RAJ2000", "DEJ2000", "Vmag", "id"]:
        np.testing.assert_array_equal(
            np.ma.getmaskarray(table[colname]), np.ma.getmaskarray(stars[colname])
        )
        np.testing.assert_array_equal(
            np.ma.filled(table[colname], 0), np.ma.filled(stars[colname], 0)
        )


class _UnitReadOnceColumn(Column):
    """Column whose unit reads as `None` after the first read once armed."""

    real_unit_reads = None  # None means not armed; the unit always reads.

    @property
    def unit(self):
        if self.real_unit_reads is None:
            return Column.unit.fget(self)
        self.real_unit_reads -= 1
        return Column.unit.fget(self) if self.real_unit_reads >= 0 else None

    @unit.setter
    def unit(self, unit):
        Column.unit.fset(self, unit)


def test_write_local_store_reads_each_unit_once(tmp_path):
    """
    Guard against a unit seen as set and then as `None` within one chunk.

    One chunk in 720 of a 785-million-row build raised ``AttributeError``
    because ``column.unit`` was not `None` when checked and `None` when used.
    """
    stars = Table(
        [
            _UnitReadOnceColumn([10.0, 20.0], name="RAJ2000", unit="deg"),
            Column([1.0, 2.0], name="DEJ2000", unit="deg"),
        ]
    )
    stars["RAJ2000"].real_unit_reads = 1
    _write_stars(stars, tmp_path)
    assert _manifest(tmp_path, "stars")["units"] == {"RAJ2000": "deg", "DEJ2000": "deg"}


def test_local_store_roundtrip_matches_fixture(refcat2_store):
    raw = _raw_refcat2()
    backend = LocalStoreBackend(refcat2_store)
    assert backend.provides("J/ApJ/867/105/refcat2")
    assert not backend.provides("B/vsx/vsx")

    table = backend.cone_search(_query(_ey_uma_center(), 1.0))
    table.sort("RA_ICRS")
    raw.sort("RA_ICRS")
    _assert_tables_equal(raw, table)
    # The masks are what refcat2()'s galaxy/non-Gaia filters key on.
    assert np.ma.getmaskarray(table["e_pmRA"]).sum() == 2
    assert np.ma.getmaskarray(table["e_Gmag"]).sum() == 2


def test_local_store_honors_radius_and_filters(refcat2_store):
    raw = _raw_refcat2()
    backend = LocalStoreBackend(refcat2_store)
    center = _ey_uma_center()

    small = backend.cone_search(_query(center, 0.1))
    sep = center.separation(SkyCoord(small["RA_ICRS"], small["DE_ICRS"], unit="deg"))
    assert 0 < len(small) < len(raw)
    assert (sep <= 0.1 * u.deg).all()

    limited = backend.cone_search(_query(center, 1.0, rmag="<=14"))
    assert len(limited) == np.sum(np.ma.filled(raw["rmag"] <= 14, False))


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("<=15.1", [14.0, 15.1]),
        ("<15.1", [14.0]),
        (">=15.1", [15.1, 16.0]),
        (">15.1", [16.0]),
        ("=15.1", [15.1]),
        ("!=15.1", [14.0, 16.0]),
        ("15.1..16", [15.1, 16.0]),
        ("14..15.1", [14.0, 15.1]),
    ],
)
def test_filter_expression_float32_limits_and_nulls(spec, expected):
    # 15.1 as float32 is not 15.1 as float64; a star exactly at the limit
    # must still pass, and a null (masked) value must never pass.
    pa = pytest.importorskip("pyarrow")
    pc = pytest.importorskip("pyarrow.compute")
    mags = pa.array([14.0, 15.1, 16.0, None], type=pa.float32())
    arrow_table = pa.table({"rmag": mags})
    expr = _filter_expression(pc.field("rmag"), spec, mags.type)
    kept = arrow_table.filter(expr)
    np.testing.assert_array_equal(
        kept["rmag"].to_numpy(), np.array(expected, dtype=np.float32)
    )
    # Same answer as the client-side filter that remains the authority.
    reference = _apply_column_filters(
        _arrow_to_masked_table(arrow_table, {}), {"rmag": spec}
    )
    np.testing.assert_array_equal(kept["rmag"].to_numpy(), reference["rmag"])


def test_filter_expression_integer_and_string_columns():
    pa = pytest.importorskip("pyarrow")
    pc = pytest.importorskip("pyarrow.compute")
    flags = pa.array([14, 15, 16, None], type=pa.int16())
    expr = _filter_expression(pc.field("flag"), "<=15.5", flags.type)
    kept = pa.table({"flag": flags}).filter(expr)
    assert kept["flag"].to_pylist() == [14, 15]
    # A string column has no numeric comparison; leave it to the post-filter.
    assert _filter_expression(pc.field("name"), "<=1", pa.string()) is None


@pytest.mark.parametrize(
    "filters",
    [{"rmag": "<=15"}, {"rmag": "13..15"}, {"gmag": ">14"}, {"pmRA": "!=0"}],
)
def test_local_store_filters_match_client_side_filtering(refcat2_store, filters):
    backend = LocalStoreBackend(refcat2_store)
    center = _ey_uma_center()
    unfiltered = backend.cone_search(_query(center, 0.5))
    expected = _apply_column_filters(unfiltered, filters)
    got = backend.cone_search(_query(center, 0.5, **filters))
    assert 0 < len(got) < len(unfiltered)
    _assert_tables_equal(expected, got)


def test_local_store_masked_value_never_passes_filter(refcat2_store):
    # Some fixture rows have pmRA masked; a filter every real value passes
    # must still drop them.
    backend = LocalStoreBackend(refcat2_store)
    center = _ey_uma_center()
    everything = backend.cone_search(_query(center, 1.0))
    n_masked = np.ma.getmaskarray(everything["pmRA"]).sum()
    assert n_masked > 0
    kept = backend.cone_search(_query(center, 1.0, pmRA=">-1e9"))
    assert len(kept) == len(everything) - n_masked
    assert not np.ma.getmaskarray(kept["pmRA"]).any()


def test_local_store_unknown_filter_column_is_a_value_error(refcat2_store):
    backend = LocalStoreBackend(refcat2_store)
    query = _query(_ey_uma_center(), 0.5, nope="<1")
    # Not CatalogUnavailableError: the query is wrong, not the store.
    with pytest.raises(ValueError, match="Cannot filter on 'nope'") as exc:
        backend.cone_search(query)
    assert not isinstance(exc.value, CatalogUnavailableError)
    with pytest.raises(RuntimeError, match="ValueError: Cannot filter") as exc:
        query_backends(query, [backend])
    assert isinstance(exc.value.__cause__, ValueError)


def _write_shallow_refcat2_store(root):
    """Write the refcat2 fixture to a store cut at rmag 16 and open it."""
    write_local_store(
        _raw_refcat2(), root, "refcat2", "J/ApJ/867/105/refcat2", "RA_ICRS",
        "DE_ICRS", mag_limit={"rmag": 16.0},
    )  # fmt: skip
    return LocalStoreBackend(root)


def _faint_network_backend():
    """A stand-in for a network source with one star fainter than the store."""
    return _Fake(
        "network", Table({"RA_ICRS": [10.0], "DE_ICRS": [-10.0], "rmag": [17.0]})
    )


def test_local_store_warns_when_shallower_than_query(tmp_path):
    backend = _write_shallow_refcat2_store(tmp_path)
    center = _ey_uma_center()
    with pytest.warns(
        AstropyUserWarning, match="rmag <= 16.0; fainter sources are missing"
    ):
        backend.cone_search(_query(center, 1.0))
    with pytest.warns(
        AstropyUserWarning,
        match="Put 'vizier' before 'local' in STELLARPHOT_CATALOG_SOURCES",
    ):
        backend.cone_search(_query(center, 1.0, rmag="<=17"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend.cone_search(_query(center, 1.0, rmag="<=14"))


def test_local_store_is_complete_only_within_its_depth(tmp_path):
    backend = _write_shallow_refcat2_store(tmp_path)
    center = _ey_uma_center()
    assert backend.is_complete(_query(center, 1.0, rmag="<=14"))
    assert backend.is_complete(_query(center, 1.0, rmag="<16"))
    assert not backend.is_complete(_query(center, 1.0, rmag="<=17"))
    assert not backend.is_complete(_query(center, 1.0))
    # A store without a recorded limit is the whole catalog.
    assert LocalStoreBackend(tmp_path) is not None
    write_local_store(
        _raw_refcat2(), tmp_path, "refcat2", "J/ApJ/867/105/refcat2", "RA_ICRS",
        "DE_ICRS",
    )  # fmt: skip
    assert LocalStoreBackend(tmp_path).is_complete(_query(center, 1.0))


def test_local_store_empty_answer_falls_through_only_when_shallow(tmp_path):
    store = _write_shallow_refcat2_store(tmp_path)
    nowhere = SkyCoord(10, -10, unit="deg")  # no fixture stars here
    network = _faint_network_backend()

    with pytest.warns(AstropyUserWarning, match="fainter sources are missing"):
        table, source = query_backends(
            _query(nowhere, 0.1, rmag="<=17"), [store, network]
        )
    assert source == "network"
    assert len(table) == 1

    with pytest.raises(RuntimeError, match="no sources"):
        query_backends(_query(nowhere, 0.1, rmag="<=15"), [store, network])
    assert network.calls == 1


def test_shallow_local_store_rows_win_over_later_backends(tmp_path):
    # A deliberate design choice: a store that has rows for the cone answers
    # the query even when the query is deeper than the store, with a warning
    # that fainter sources are missing. Only an *empty* answer from a shallow
    # store falls through to the next backend (see the test above).
    store = _write_shallow_refcat2_store(tmp_path)
    network = _faint_network_backend()

    with pytest.warns(AstropyUserWarning, match="fainter sources are missing"):
        table, source = query_backends(_query(_ey_uma_center(), 1.0), [store, network])
    assert source.startswith("local:")
    assert len(table) > 0
    assert network.calls == 0


def test_local_store_unreadable_data_is_unavailable(refcat2_store):
    backend = LocalStoreBackend(refcat2_store)
    query = _query(_ey_uma_center(), 0.5)
    assert len(backend.cone_search(query)) > 0
    # The store's data goes away (external disk unmounted) or a file is bad.
    data_dir = _data_dir(refcat2_store, "refcat2")
    for parquet in data_dir.rglob("*.parquet"):
        parquet.write_bytes(b"not a parquet file")
    with pytest.raises(CatalogUnavailableError, match="Could not read"):
        backend.cone_search(query)
    shutil.rmtree(data_dir)
    with pytest.raises(CatalogUnavailableError, match="Could not read"):
        backend.cone_search(query)


def test_local_store_warns_about_missing_or_empty_root(tmp_path):
    with pytest.warns(AstropyUserWarning, match="does not exist"):
        backend = LocalStoreBackend(tmp_path / "typo")
    assert backend.stores == {}
    with pytest.warns(AstropyUserWarning, match="no catalogs"):
        LocalStoreBackend(tmp_path)


@pytest.mark.parametrize(
    "manifest_text",
    [
        '{"format": "stellarphot-catalog-store", "vizier_id": "te',  # truncated
        json.dumps({"format": "something-else", "vizier_id": "test/other"}),
        json.dumps({"vizier_id": "test/other"}),  # no format
        json.dumps({"format": "stellarphot-catalog-store"}),  # no vizier_id
        json.dumps(["not", "a", "dict"]),
    ],
)
def test_local_store_skips_bad_manifest(refcat2_store, manifest_text):
    bad = refcat2_store / "other" / "catalog.json"
    bad.parent.mkdir()
    bad.write_text(manifest_text)
    with pytest.warns(AstropyUserWarning, match="other.catalog.json") as record:
        backend = LocalStoreBackend(refcat2_store)
    assert len(record) == 1
    assert set(backend.stores) == {"J/ApJ/867/105/refcat2"}
    assert len(backend.cone_search(_query(_ey_uma_center(), 0.5))) > 0


def test_arrow_to_masked_table_keeps_integer_dtypes():
    big = 2**60 + 1  # not representable as float64
    arrow_table = pa.table(
        {
            "id": pa.array([big, None, 3], type=pa.int64()),
            "flag": pa.array([1, None, 0], type=pa.uint8()),
            "ok": pa.array([True, None, False], type=pa.bool_()),
            "mag": pa.array([1.5, None, 2.5], type=pa.float32()),
            "name": pa.array(["a", None, "c"], type=pa.string()),
        }
    )
    table = _arrow_to_masked_table(arrow_table, {"mag": "mag"})
    assert table["id"].dtype == np.int64
    assert table["id"][0] == big
    assert table["flag"].dtype == np.uint8
    assert table["ok"].dtype == np.bool_
    assert table["mag"].dtype == np.float32
    assert table["mag"].unit == u.mag
    for colname in table.colnames:
        assert list(np.ma.getmaskarray(table[colname])) == [False, True, False]


@pytest.mark.parametrize(
    "ra, dec, radius, expected_names",
    [
        (0.0, 0.0, 0.5, {"wrap-west", "wrap-east"}),  # across RA = 0/360
        (180.0, 0.0, 0.5, {"antimeridian"}),
        (10.0, 89.9, 0.5, {"pole-a", "pole-b"}),  # all RAs near the pole
        (10.0, -89.9, 0.5, {"south-pole"}),
        (45.0, 30.0, 0.05, set()),  # empty cone
    ],
)
def test_local_store_cone_geometry(tmp_path, ra, dec, radius, expected_names):
    stars = _stars(
        [
            "wrap-west", "wrap-east", "antimeridian", "pole-a", "pole-b",
            "south-pole", "far",
        ],
        [359.8, 0.2, 180.1, 100.0, 300.0, 200.0, 45.0],
        [0.1, -0.1, 0.2, 89.95, 89.8, -89.95, 31.0],
    )  # fmt: skip
    _write_stars(stars, tmp_path)
    backend = LocalStoreBackend(tmp_path)
    assert _star_names(backend, ra, dec, radius) == expected_names


def test_local_store_end_to_end_refcat2(refcat2_store, mocker):
    # The local store must give refcat2() exactly what the mocked VizieR
    # response gives it.
    raw = _raw_refcat2()
    # VizieR applies the magnitude limit server-side; the mock has to do it.
    query = mocker.patch(_QUERY_REGION, return_value=[raw[raw["rmag"] <= 15]])
    header = _ey_uma_header()

    from_vizier = refcat2(
        header,
        radius=0.5 * u.deg,
        clip_by_frame=True,
        magnitude_limit=15,
        magnitude_limit_passband="SR",
    )
    assert query.called
    from_store = refcat2(
        header,
        radius=0.5 * u.deg,
        clip_by_frame=True,
        magnitude_limit=15,
        magnitude_limit_passband="SR",
        backends=[LocalStoreBackend(refcat2_store)],
    )
    assert query.call_count == 1
    assert from_store.catalog_source.startswith("local:")

    assert len(from_store) == len(from_vizier)
    for cat in (from_vizier, from_store):
        cat.sort(["id", "passband"])
    assert list(from_store["id"]) == list(from_vizier["id"])
    assert list(from_store["passband"]) == list(from_vizier["passband"])
    np.testing.assert_allclose(from_store["mag"], from_vizier["mag"])
    np.testing.assert_allclose(from_store["mag_error"], from_vizier["mag_error"])


# ---------------------------------------------------------------------------
# AAVSO VSX API
# ---------------------------------------------------------------------------

_VSX_API_OBJECTS = [
    {
        "OID": "25934", "Name": "EY UMa", "AUID": "000-BBD-397",
        "RA2000": "135.6755", "Declination2000": "49.8215",
        "VariabilityType": "UG", "MaxMag": "13.0 V", "MinMag": "18.6 V",
        "Period": "0.0659", "Category": "Variable", "SpectralType": "",
    },
    {
        "OID": "123", "Name": "Fake Amp", "RA2000": "135.70",
        "Declination2000": "49.80", "VariabilityType": "EW",
        "MaxMag": "12.5 CV", "MinMag": "(0.32) CV", "Period": "<0.5:",
        "Category": "Suspected",
    },
    {
        "OID": "456", "Name": "Fake Limit", "RA2000": "135.60",
        "Declination2000": "49.85", "VariabilityType": "M",
        "MaxMag": "<14.6 p", "MinMag": "?", "Category": "Variable",
    },
]  # fmt: skip


def test_vsx_api_objects_to_table_schema_matches_vizier():
    fixture = Table.read(get_pkg_data_filename("data/test_vsx_table.ecsv"))
    table = vsx_api_objects_to_table(_VSX_API_OBJECTS)
    assert table.colnames == fixture.colnames
    for colname in fixture.colnames:
        assert table[colname].dtype.kind == fixture[colname].dtype.kind, colname
        assert table[colname].unit == fixture[colname].unit, colname

    ey_uma, amp, limit = table
    assert ey_uma["Name"] == "EY UMa"
    assert ey_uma["OID"] == 25934
    assert ey_uma["V"] == 0
    assert ey_uma["max"] == np.float32(13.0)
    assert ey_uma["n_max"] == "V"
    assert ey_uma["min"] == np.float32(18.6)
    assert ey_uma["Period"] == 0.0659
    assert np.ma.is_masked(ey_uma["Sp"])

    assert amp["V"] == 1
    assert amp["f_min"] == "Y"
    assert amp["min"] == np.float32(0.32)
    assert amp["n_min"] == "CV"
    assert amp["l_Period"] == "<"
    assert amp["u_Period"] == ":"
    assert amp["Period"] == 0.5

    assert limit["l_max"] == "<"
    assert limit["max"] == np.float32(14.6)
    assert limit["n_max"] == "p"
    assert np.ma.is_masked(limit["min"])

    empty = vsx_api_objects_to_table([])
    assert empty.colnames == fixture.colnames
    assert len(empty) == 0


class _FakeResponse:
    def __init__(self, payload, content=None):
        self._payload = payload
        self.content = json.dumps(payload).encode() if content is None else content

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_aavso_vsx_backend_end_to_end_offline(mocker):
    get = mocker.patch(
        "stellarphot.catalog_backends.requests.get",
        return_value=_FakeResponse({"VSXObjects": {"VSXObject": _VSX_API_OBJECTS}}),
    )
    header = _ey_uma_header()
    cat = vsx_vizier(header, radius=0.5 * u.deg, backends=[AavsoVsxBackend()])
    assert get.called
    params = get.call_args.kwargs["params"]
    assert params["radius"] == "0.500000"  # degrees, not arcmin
    assert get.call_args.args[0].startswith("https://vsx.aavso.org/")
    assert cat.catalog_source == "AAVSO VSX API"
    assert set(cat["id"]) == {"EY UMa", "Fake Amp", "Fake Limit"}
    assert cat["ra"].unit == u.deg

    # A magnitude limit becomes a client-side filter on ``max``.
    cat = vsx_vizier(
        header,
        radius=0.5 * u.deg,
        magnitude_limit=12.8,
        backends=[AavsoVsxBackend()],
    )
    assert set(cat["id"]) == {"Fake Amp"}

    # A single match comes back as a bare object rather than a list.
    get.return_value = _FakeResponse({"VSXObjects": {"VSXObject": _VSX_API_OBJECTS[0]}})
    cat = vsx_vizier(header, radius=0.5 * u.deg, backends=[AavsoVsxBackend()])
    assert list(cat["id"]) == ["EY UMa"]


def test_aavso_vsx_backend_network_failure_is_unavailable(mocker):
    import requests

    mocker.patch(
        "stellarphot.catalog_backends.requests.get",
        side_effect=requests.exceptions.ConnectionError("down"),
    )
    with pytest.raises(CatalogUnavailableError, match="AAVSO VSX API failed"):
        AavsoVsxBackend().cone_search(
            _query(_ey_uma_center(), 0.5, catalog_id="B/vsx/vsx")
        )


@pytest.mark.parametrize(
    "payload, content, problem",
    [
        (None, b"", "empty response"),
        ({}, None, "VSXObjects"),
        ([], None, "VSXObjects"),
        ({"VSXObjects": "nope"}, None, "VSXObjects"),
    ],
)
def test_aavso_vsx_backend_malformed_answer_is_unavailable(
    mocker, payload, content, problem
):
    mocker.patch(
        "stellarphot.catalog_backends.requests.get",
        return_value=_FakeResponse(payload, content=content),
    )
    with pytest.raises(CatalogUnavailableError, match=problem) as excinfo:
        AavsoVsxBackend().cone_search(
            _query(_ey_uma_center(), 0.5, catalog_id="B/vsx/vsx")
        )
    assert AavsoVsxBackend.url in str(excinfo.value)


def test_aavso_vsx_backend_no_variables_is_an_empty_table(mocker):
    # The live API answers a field with no variables with a list, not a dict.
    mocker.patch(
        "stellarphot.catalog_backends.requests.get",
        return_value=_FakeResponse({"VSXObjects": []}),
    )
    table = AavsoVsxBackend().cone_search(
        _query(_ey_uma_center(), 0.5, catalog_id="B/vsx/vsx")
    )
    assert len(table) == 0
    assert table.colnames == vsx_api_objects_to_table([]).colnames


def test_aavso_vsx_backend_single_object_is_one_row(mocker):
    mocker.patch(
        "stellarphot.catalog_backends.requests.get",
        return_value=_FakeResponse({"VSXObjects": {"VSXObject": _VSX_API_OBJECTS[0]}}),
    )
    table = AavsoVsxBackend().cone_search(
        _query(_ey_uma_center(), 0.5, catalog_id="B/vsx/vsx")
    )
    assert list(table["Name"]) == ["EY UMa"]


# ---------------------------------------------------------------------------
# APASS over TAP
# ---------------------------------------------------------------------------


def test_esa_apass_adql_and_mapping():
    backend = EsaApassDr9Backend()
    adql = backend.adql(_query(_ey_uma_center(), 0.25, catalog_id="II/336/apass9"))
    assert "FROM external.apassdr9" in adql
    assert "raj2000, dej2000" in adql

    fixture = _raw_apass()
    raw = Table()
    for colname in fixture.colnames:
        if colname == "_r":
            continue
        raw[
            colname.lower()
            .replace("'", "_")
            .replace("-", "_")
            .replace("raj2000", "raj2000")
        ] = fixture[colname]
    raw.rename_columns(["e_raj2000", "e_dej2000"], ["e_ra", "e_dec"])
    table = backend.to_vizier(raw)
    assert table.colnames == [c for c in fixture.colnames if c != "_r"]
    np.testing.assert_array_equal(table["Vmag"], fixture["Vmag"])


def _votable_bytes(table):
    buffer = __import__("io").BytesIO()
    writeto(from_table(table), buffer)
    return buffer.getvalue()


def test_esa_apass_drops_rows_without_coordinates(mocker):
    raw = Table(masked=True)
    for src in catalog_backends._ESA_APASS_COLUMNS:
        raw[src] = MaskedColumn([1.0, 2.0, 3.0])
    raw["raj2000"] = MaskedColumn([135.5, 135.6, 135.7], mask=[False, True, False])
    raw["dej2000"] = MaskedColumn([49.8, 49.9, 50.0], mask=[False, False, True])
    backend = EsaApassDr9Backend()
    converted = backend.to_vizier(raw)
    assert list(np.ma.getmaskarray(converted["RAJ2000"])) == [False, True, False]
    assert list(np.ma.getmaskarray(converted["DEJ2000"])) == [False, False, True]

    mocker.patch("stellarphot.catalog_backends._tap_async", return_value=raw)
    table = backend.cone_search(
        _query(_ey_uma_center(), 1.0, catalog_id="II/336/apass9")
    )
    assert len(table) == 1
    assert table["RAJ2000"][0] == 135.5


class _ChunkedBody:
    """Raw body of a streamed response, read back a few bytes at a time."""

    def __init__(self, content, chunk_bytes, clock=None, seconds_per_chunk=0):
        self.content = content
        self.chunk_bytes = chunk_bytes
        self.clock = clock
        self.seconds_per_chunk = seconds_per_chunk
        self.position = 0
        self.reads = 0

    def read(self, size=-1, **_kwargs):
        if self.position >= len(self.content):
            return b""
        size = self.chunk_bytes if size is None or size < 0 else size
        size = min(size, self.chunk_bytes)
        chunk = self.content[self.position : self.position + size]
        self.position += len(chunk)
        self.reads += 1
        if self.clock is not None:
            self.clock.advance(self.seconds_per_chunk)
        return chunk

    def close(self):
        pass


class _FakeHttp:
    """
    requests.post / requests.get stand-in for a UWS job.

    The result is served as a streamed body of ``chunk_bytes`` per read, each
    read advancing ``clock`` by ``seconds_per_chunk``.
    """

    def __init__(
        self,
        phases,
        location="/tap/async/42",
        clock=None,
        result=None,
        chunk_bytes=1 << 20,
        seconds_per_chunk=0,
    ):
        self.phases = list(phases)
        self.location = location
        self.calls = []
        self.clock = clock
        self.result = Table({"a": [1, 2]}) if result is None else result
        self.chunk_bytes = chunk_bytes
        self.seconds_per_chunk = seconds_per_chunk
        self.body = None
        self.result_kwargs = None

    def _response(self, url, content=b"", status=200):
        response = requests.Response()
        response.status_code = status
        response._content = content
        response.url = url
        return response

    def post(self, url, data=None, timeout=None, **_kwargs):
        self.calls.append(("POST", url, timeout, data))
        if self.clock is not None:
            self.clock.advance(10)
        response = self._response(url, status=303)
        response.headers["Location"] = self.location
        return response

    def get(self, url, timeout=None, **kwargs):
        self.calls.append(("GET", url, timeout, None))
        if self.clock is not None:
            self.clock.advance(10)
        if url.endswith("/phase"):
            phase = self.phases.pop(0) if len(self.phases) > 1 else self.phases[0]
            return self._response(url, phase.encode())
        self.result_kwargs = kwargs
        response = self._response(url, content=False)
        self.body = _ChunkedBody(
            _votable_bytes(self.result),
            self.chunk_bytes,
            clock=self.clock,
            seconds_per_chunk=self.seconds_per_chunk,
        )
        response.raw = self.body
        return response

    def install(self, mocker):
        mocker.patch("stellarphot.catalog_backends.requests.post", self.post)
        mocker.patch("stellarphot.catalog_backends.requests.get", self.get)


def test_tap_async_joins_relative_location(mocker):
    http = _FakeHttp(["EXECUTING", "COMPLETED"])
    http.install(mocker)
    mocker.patch("stellarphot.catalog_backends.time.sleep")
    table = _tap_async("https://host.example/tap", "SELECT 1", timeout=60)
    assert len(table) == 2
    urls = [c[1] for c in http.calls]
    assert urls == [
        "https://host.example/tap/async",
        "https://host.example/tap/async/42/phase",
        "https://host.example/tap/async/42/phase",
        "https://host.example/tap/async/42/results/result",
    ]


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def advance(self, seconds):
        self.now += seconds

    def monotonic(self):
        return self.now

    def install(self, mocker):
        """Drive ``time`` in catalog_backends from this clock; return the sleeps."""
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            self.advance(seconds)

        mocker.patch("stellarphot.catalog_backends.time.monotonic", self.monotonic)
        mocker.patch("stellarphot.catalog_backends.time.sleep", sleep)
        return sleeps


def test_tap_async_timeout_is_an_upper_bound(mocker):
    clock = _Clock()
    http = _FakeHttp(["EXECUTING"], clock=clock)
    http.install(mocker)
    sleeps = clock.install(mocker)
    start = clock.now
    with pytest.raises(requests.exceptions.Timeout, match="did not finish"):
        _tap_async("https://host.example/tap", "SELECT 1", timeout=45)
    # Every request waits at most for the time left, and the job is aborted.
    assert all(0 < c[2] <= 45 for c in http.calls)
    assert http.calls[-1][0] == "POST"
    assert http.calls[-1][3] == {"PHASE": "ABORT"}
    # The request in flight at the deadline may run to its own (bounded)
    # timeout, then the abort is posted: at most two requests over.
    assert clock.now - start <= 45 + 2 * 10
    assert sleeps and all(0 < seconds <= 3 for seconds in sleeps)


def test_tap_async_streams_result_in_chunks(mocker):
    expected = Table({"a": np.arange(500), "b": np.arange(500) * 0.5})
    http = _FakeHttp(["COMPLETED"], result=expected, chunk_bytes=1000)
    http.install(mocker)
    table = _tap_async("https://host.example/tap", "SELECT 1", timeout=60)
    assert http.result_kwargs.get("stream") is True
    assert http.body.reads > 2
    np.testing.assert_array_equal(table["a"], expected["a"])
    np.testing.assert_array_equal(table["b"], expected["b"])


def test_tap_async_deadline_applies_while_downloading(mocker):
    # Each chunk arrives quickly enough to satisfy requests' inactivity
    # timeout, but together they take longer than the overall timeout.
    clock = _Clock()
    http = _FakeHttp(
        ["COMPLETED"],
        clock=clock,
        result=Table({"a": np.arange(500)}),
        chunk_bytes=1000,
        seconds_per_chunk=5,
    )
    http.install(mocker)
    clock.install(mocker)
    start = clock.now
    with pytest.raises(requests.exceptions.Timeout, match="did not finish"):
        _tap_async("https://host.example/tap", "SELECT 1", timeout=60)
    # The download stopped at the deadline and the job was aborted.
    assert clock.now - start <= 60 + 5 + 10
    assert http.calls[-1][0] == "POST"
    assert http.calls[-1][3] == {"PHASE": "ABORT"}


@pytest.mark.parametrize(
    "waiting", ["PENDING", "QUEUED", "EXECUTING", "HELD", "SUSPENDED", "UNKNOWN"]
)
def test_tap_async_keeps_polling_while_job_is_waiting(mocker, waiting):
    http = _FakeHttp([waiting, waiting, "COMPLETED"])
    http.install(mocker)
    mocker.patch("stellarphot.catalog_backends.time.sleep")
    table = _tap_async("https://host.example/tap", "SELECT 1", timeout=60)
    assert len(table) == 2
    assert sum(c[1].endswith("/phase") for c in http.calls) == 3


@pytest.mark.parametrize(
    "phase, message",
    [
        ("ERROR", "ended in phase ERROR"),
        ("ABORTED", "ended in phase ABORTED"),
        ("ARCHIVED", "ended in phase ARCHIVED"),
        ("<html>Gateway error</html>", "unexpected phase"),
    ],
)
def test_tap_async_failed_or_unknown_phase_raises(mocker, phase, message):
    http = _FakeHttp([phase])
    http.install(mocker)
    mocker.patch("stellarphot.catalog_backends.time.sleep")
    with pytest.raises(ValueError, match=message):
        _tap_async("https://host.example/tap", "SELECT 1", timeout=60)
    assert not any(c[1].endswith("/results/result") for c in http.calls)


def test_esa_apass_backend_with_galactic_center(mocker):
    center = _ey_uma_center()
    raw = Table()
    for src in catalog_backends._ESA_APASS_COLUMNS:
        raw[src] = [12.0, 13.0]
    raw["raj2000"] = [center.ra.deg, center.ra.deg]
    raw["dej2000"] = [center.dec.deg, center.dec.deg + 0.1]
    http = _FakeHttp(["COMPLETED"], result=raw)
    http.install(mocker)
    query = _query(center.galactic, 0.25, catalog_id="II/336/apass9")

    adql = EsaApassDr9Backend().adql(query)
    assert f"CIRCLE('ICRS', {center.ra.deg:.8f}, {center.dec.deg:.8f}," in adql

    table = EsaApassDr9Backend().cone_search(query)
    assert table.colnames[0] == "_r"
    assert table["_r"].unit == u.deg
    np.testing.assert_allclose(table["_r"], [0, 0.1], atol=1e-6)


# ---------------------------------------------------------------------------
# Live services
# ---------------------------------------------------------------------------


@pytest.mark.remote_data
def test_aavso_vsx_backend_live():
    expected = Table.read(get_pkg_data_filename("data/unclipped_ey_uma_vsx.fits"))
    try:
        cat = vsx_vizier(
            _ey_uma_header(), radius=0.5 * u.deg, backends=[AavsoVsxBackend()]
        )
    except Exception as err:
        _xfail_if_server_down(err, "AAVSO VSX API")
    # New variables may be added over time, so only require the known ones.
    assert set(expected["id"]) <= set(cat["id"])


@pytest.mark.remote_data
def test_esa_apass_backend_live():
    try:
        cat = apass_dr9(
            _ey_uma_header(), radius=10 * u.arcmin, backends=[EsaApassDr9Backend()]
        )
    except Exception as err:
        _xfail_if_server_down(err, "ESA TAP")
    assert len(cat) > 0
    assert cat.catalog_source == "ESA APASS DR9"
    assert "V" in set(cat["passband"])
