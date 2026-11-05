import json
import warnings

import numpy as np
import pytest
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.votable import from_table, writeto
from astropy.nddata import CCDData
from astropy.table import MaskedColumn, Table
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
    _parse_filter,
    _tap_async,
    default_backends,
    describe_backends,
    query_backends,
    vsx_api_objects_to_table,
    write_local_store,
)
from stellarphot.catalogs import apass_dr9, refcat2, vsx_vizier
from stellarphot.conftest import SERVER_DOWN_ERRORS

pytest.importorskip("pyarrow")

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


def test_write_local_store_layout_and_manifest(refcat2_store):
    manifest = json.loads((refcat2_store / "refcat2" / "catalog.json").read_text())
    assert manifest["vizier_id"] == "J/ApJ/867/105/refcat2"
    assert manifest["rows"] == len(_raw_refcat2())
    assert manifest["units"]["pmRA"] == "mas / yr"
    assert manifest["units"]["RA_ICRS"] == "deg"
    parts = sorted(
        p.parent.name
        for p in (refcat2_store / "refcat2" / "data").glob("zone=*/*.parquet")
    )
    # The EY UMa field straddles Dec = +50, so the rows land in two zones.
    assert parts == ["zone=139", "zone=140"]


def test_local_store_roundtrip_matches_fixture(refcat2_store):
    raw = _raw_refcat2()
    backend = LocalStoreBackend(refcat2_store)
    assert backend.provides("J/ApJ/867/105/refcat2")
    assert not backend.provides("B/vsx/vsx")

    table = backend.cone_search(_query(_ey_uma_center(), 1.0))
    assert len(table) == len(raw)
    table.sort("RA_ICRS")
    raw.sort("RA_ICRS")
    for colname in raw.colnames:
        expected = np.ma.asarray(raw[colname])
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
        assert table[colname].unit == raw[colname].unit, colname
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


def test_local_store_warns_when_shallower_than_query(tmp_path):
    write_local_store(
        _raw_refcat2(), tmp_path, "refcat2", "J/ApJ/867/105/refcat2", "RA_ICRS",
        "DE_ICRS", mag_limit={"rmag": 16.0},
    )  # fmt: skip
    backend = LocalStoreBackend(tmp_path)
    center = _ey_uma_center()
    with pytest.warns(
        AstropyUserWarning, match="rmag <= 16.0; fainter sources are missing"
    ):
        backend.cone_search(_query(center, 1.0))
    with pytest.warns(AstropyUserWarning, match="fainter sources are missing"):
        backend.cone_search(_query(center, 1.0, rmag="<=17"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend.cone_search(_query(center, 1.0, rmag="<=14"))


def test_local_store_is_complete_only_within_its_depth(tmp_path):
    write_local_store(
        _raw_refcat2(), tmp_path, "refcat2", "J/ApJ/867/105/refcat2", "RA_ICRS",
        "DE_ICRS", mag_limit={"rmag": 16.0},
    )  # fmt: skip
    backend = LocalStoreBackend(tmp_path)
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
    write_local_store(
        _raw_refcat2(), tmp_path, "refcat2", "J/ApJ/867/105/refcat2", "RA_ICRS",
        "DE_ICRS", mag_limit={"rmag": 16.0},
    )  # fmt: skip
    store = LocalStoreBackend(tmp_path)
    nowhere = SkyCoord(10, -10, unit="deg")  # no fixture stars here
    rows = Table({"RA_ICRS": [10.0], "DE_ICRS": [-10.0], "rmag": [17.0]})
    network = _Fake("network", rows)

    with pytest.warns(AstropyUserWarning, match="fainter sources are missing"):
        table, source = query_backends(
            _query(nowhere, 0.1, rmag="<=17"), [store, network]
        )
    assert source == "network"
    assert len(table) == 1

    with pytest.raises(RuntimeError, match="no sources"):
        query_backends(_query(nowhere, 0.1, rmag="<=15"), [store, network])
    assert network.calls == 1


def test_local_store_unreadable_data_is_unavailable(refcat2_store):
    backend = LocalStoreBackend(refcat2_store)
    query = _query(_ey_uma_center(), 0.5)
    assert len(backend.cone_search(query)) > 0
    # The store's data goes away (external disk unmounted) or a file is bad.
    for parquet in (refcat2_store / "refcat2" / "data").rglob("*.parquet"):
        parquet.write_bytes(b"not a parquet file")
    with pytest.raises(CatalogUnavailableError, match="Could not read"):
        backend.cone_search(query)
    import shutil

    shutil.rmtree(refcat2_store / "refcat2" / "data")
    with pytest.raises(CatalogUnavailableError, match="Could not read"):
        backend.cone_search(query)


def test_local_store_warns_about_missing_or_empty_root(tmp_path):
    with pytest.warns(AstropyUserWarning, match="does not exist"):
        backend = LocalStoreBackend(tmp_path / "typo")
    assert backend.stores == {}
    with pytest.warns(AstropyUserWarning, match="no catalogs"):
        LocalStoreBackend(tmp_path)


def test_arrow_to_masked_table_keeps_integer_dtypes():
    pa = pytest.importorskip("pyarrow")
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
    stars = Table(
        {
            "Name": [
                "wrap-west", "wrap-east", "antimeridian", "pole-a", "pole-b",
                "south-pole", "far",
            ],
            "RAJ2000": [359.8, 0.2, 180.1, 100.0, 300.0, 200.0, 45.0],
            "DEJ2000": [0.1, -0.1, 0.2, 89.95, 89.8, -89.95, 31.0],
        }
    )  # fmt: skip
    write_local_store(stars, tmp_path, "stars", "test/stars", "RAJ2000", "DEJ2000")
    backend = LocalStoreBackend(tmp_path)
    table = backend.cone_search(
        _query(SkyCoord(ra, dec, unit="deg"), radius, catalog_id="test/stars")
    )
    assert set(table["Name"]) == expected_names


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
    def __init__(self, payload):
        self._payload = payload
        self.content = b"{}"

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


class _FakeHttp:
    """requests.post / requests.get stand-in for a UWS job."""

    def __init__(self, phases, location="/tap/async/42", clock=None):
        self.phases = list(phases)
        self.location = location
        self.calls = []
        self.clock = clock

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

    def get(self, url, timeout=None, **_kwargs):
        self.calls.append(("GET", url, timeout, None))
        if self.clock is not None:
            self.clock.advance(10)
        if url.endswith("/phase"):
            phase = self.phases.pop(0) if len(self.phases) > 1 else self.phases[0]
            return self._response(url, phase.encode())
        return self._response(url, _votable_bytes(Table({"a": [1, 2]})))


def test_tap_async_joins_relative_location(mocker):
    http = _FakeHttp(["EXECUTING", "COMPLETED"])
    mocker.patch("stellarphot.catalog_backends.requests.post", http.post)
    mocker.patch("stellarphot.catalog_backends.requests.get", http.get)
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


def test_tap_async_timeout_is_an_upper_bound(mocker):
    clock = _Clock()
    http = _FakeHttp(["EXECUTING"], clock=clock)
    mocker.patch("stellarphot.catalog_backends.requests.post", http.post)
    mocker.patch("stellarphot.catalog_backends.requests.get", http.get)
    mocker.patch("stellarphot.catalog_backends.time.monotonic", clock.monotonic)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.advance(seconds)

    mocker.patch("stellarphot.catalog_backends.time.sleep", sleep)
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
    except SERVER_DOWN_ERRORS as e:
        pytest.xfail(f"AAVSO VSX API unavailable: {e}")
    # New variables may be added over time, so only require the known ones.
    assert set(expected["id"]) <= set(cat["id"])


@pytest.mark.remote_data
def test_esa_apass_backend_live():
    try:
        cat = apass_dr9(
            _ey_uma_header(), radius=10 * u.arcmin, backends=[EsaApassDr9Backend()]
        )
    except SERVER_DOWN_ERRORS as e:
        pytest.xfail(f"ESA TAP unavailable: {e}")
    assert len(cat) > 0
    assert cat.catalog_source == "ESA APASS DR9"
    assert "V" in set(cat["passband"])
