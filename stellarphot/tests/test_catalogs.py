# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Structural tests for the catalog-fetcher move (#194).

The catalog-fetcher factory functions ``apass_dr9``, ``vsx_vizier`` and
``refcat2`` live in :mod:`stellarphot.catalogs`. They remain importable from the
top-level ``stellarphot`` namespace (public, no warning) and from
``stellarphot.core`` via a back-compat shim that emits an
``AstropyDeprecationWarning``.
"""

import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.wcs import WCS
from astropy.wcs.wcs import FITSFixedWarning

from stellarphot.catalogs import (
    _attach_gaia_ids,
    _process_refcat2,
    apass_dr9,
    refcat2,
    vsx_vizier,
)

# Gaia DR2 source_id of EY UMa, the center of the field used in the
# remote-data refcat2 tests. Value from SIMBAD.
EY_UMA_GAIA_DR2_ID = 1015789765851950336


def test_fetchers_importable_from_catalogs():
    """The new home exposes all three fetchers and they are callable."""
    from stellarphot.catalogs import apass_dr9, refcat2, vsx_vizier

    assert callable(apass_dr9)
    assert callable(vsx_vizier)
    assert callable(refcat2)


def test_top_level_namespace_is_public_no_warning():
    """``from stellarphot import <fetcher>`` stays public with no deprecation."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import stellarphot
        from stellarphot import apass_dr9, refcat2, vsx_vizier  # noqa: F401

    deprecations = [
        w for w in caught if issubclass(w.category, AstropyDeprecationWarning)
    ]
    assert not deprecations, (
        "Top-level catalog imports should not warn: "
        f"{[str(w.message) for w in deprecations]}"
    )

    import stellarphot.catalogs as catalogs

    assert stellarphot.apass_dr9 is catalogs.apass_dr9
    assert stellarphot.vsx_vizier is catalogs.vsx_vizier
    assert stellarphot.refcat2 is catalogs.refcat2


@pytest.mark.parametrize("name", ["apass_dr9", "vsx_vizier", "refcat2"])
def test_core_shim_is_deprecated(name):
    """``stellarphot.core.<fetcher>`` warns and returns the catalogs object."""
    import stellarphot.catalogs as catalogs
    import stellarphot.core as core

    with pytest.warns(AstropyDeprecationWarning, match="catalogs"):
        obj = getattr(core, name)

    assert obj is getattr(catalogs, name)


# ---------------------------------------------------------------------------
# Behavioral tests for the catalog fetchers, moved here from test_core.py as
# part of #194 (the fetchers themselves moved to stellarphot.catalogs). Most
# require network access (``remote_data``).
# ---------------------------------------------------------------------------


@pytest.mark.remote_data
def test_catalog_from_vizier_search_apass():
    # Nothing special about this point...
    sc = SkyCoord(ra=0, dec=0, unit="deg")

    # Small enough radius to get only one star
    radius = 0.03 * u.deg

    # Use the APASS class factory to get the catalog
    apass = apass_dr9(sc, radius=radius)
    assert len(apass) == 6

    # Calculated the value below by constructing a coordinate-based
    # designation following IAU guidelines.
    assert apass["id"][0] == "APASSSP J+359.9896+00.0122"

    just_V = apass[apass["passband"] == "V"]
    assert np.abs(just_V["mag"][0] - 15.559) < 1e-6


@pytest.mark.remote_data
@pytest.mark.parametrize(
    "clip, data_file, mag_limit",
    [
        (True, "data/clipped_ey_uma_vsx.fits", None),
        (False, "data/unclipped_ey_uma_vsx.fits", 13),
    ],
)
def test_vsx_results(clip, data_file, mag_limit):
    # Check that a catalog search of VSX gives us what we expect.
    # I suppose this really isn't future-proof, since more variables
    # could be discovered in the future....
    data = get_pkg_data_filename(data_file)
    expected = Table.read(data)
    wcs_file = get_pkg_data_filename("data/sample_wcs_ey_uma.fits")
    with fits.open(wcs_file) as hdulist:
        with warnings.catch_warnings():
            # Ignore the warning about the WCS having a different number of
            # axes than the (non-existent) image.
            warnings.filterwarnings(
                "ignore",
                message="The WCS transformation has more",
                category=FITSFixedWarning,
            )
            wcs = WCS(hdulist[0].header)
    CCD_SHAPE = [2048, 3073]
    wcs.pixel_shape = list(reversed(CCD_SHAPE))
    ccd = CCDData(data=np.zeros(CCD_SHAPE), wcs=wcs, unit="adu")

    # Turn this into an HDU to get the standard FITS image keywords
    ccd_im = ccd.to_hdu()

    actual = vsx_vizier(
        ccd_im[0].header,
        radius=0.5 * u.degree,
        clip_by_frame=clip,
        magnitude_limit=mag_limit,
    )

    if mag_limit:
        # If we have a magnitude limit, we need to filter the expected data
        # to match.
        expected = expected[expected["mag"] <= mag_limit]

    assert set(actual["OID"]) == set(expected["OID"])


@pytest.mark.remote_data
@pytest.mark.parametrize(
    "mag_limit,mag_limit_band",
    [
        (None, None),
        (
            13,
            "V",
        ),  # Limit chosen so that some of the expected data will be filtered out
        (13, None),  # Default passband for apass_dr9 is V
    ],
)
def test_find_apass(mag_limit, mag_limit_band):
    CCD_SHAPE = [2048, 3073]
    # This is really checking from APASS DR9 on Vizier, or at least that
    # is where the "expected" data is drawn from.
    expected_all = Table.read(get_pkg_data_filename("data/all_apass_ey_uma.ecsv"))

    wcs_file = get_pkg_data_filename("data/sample_wcs_ey_uma.fits")
    with fits.open(wcs_file) as hdulist:
        with warnings.catch_warnings():
            # Ignore the warning about the WCS having a different number of
            # axes than the (non-existent) image.
            warnings.filterwarnings(
                "ignore",
                message="The WCS transformation has more",
                category=FITSFixedWarning,
            )
            wcs = WCS(hdulist[0].header)
    wcs.pixel_shape = list(reversed(CCD_SHAPE))
    ccd = CCDData(data=np.zeros(CCD_SHAPE), wcs=wcs, unit="adu")

    # Turn this into an HDU to get the standard FITS image keywords
    ccd_im = ccd.to_hdu()

    all_apass = apass_dr9(
        ccd_im[0].header,
        radius=10 * u.arcmin,
        magnitude_limit=mag_limit,
        magnitude_limit_passband=mag_limit_band,
    )

    # Impose the magnitude limit on the expected result, if any
    if mag_limit is not None:
        expected_all = expected_all[expected_all["Vmag"] <= mag_limit]
        # Apparently there are also some masked entries 🙄
        expected_all = expected_all[~expected_all["Vmag"].mask]

    # It is hard to imagine the RAs matching and other entries not matching,
    # so just check the RAs.
    assert set(ra.value for ra in all_apass["ra"]) == set(expected_all["RAJ2000"])

    # The passbands ought to have been translated to the AAVSO standard names.
    # This is a regression test for #439.
    for band in ["B", "V", "SG", "SR", "SI"]:
        assert band in all_apass["passband"]


@pytest.mark.remote_data
@pytest.mark.parametrize(
    "mag_limit,mag_limit_band",
    [
        (None, None),
        (
            13,
            "SR",
        ),  # Limit chosen so that some of the expected data will be filtered out
        (13, None),  # Default passband for refcat2 is SR
    ],
)
def test_find_refcat2(mag_limit, mag_limit_band):
    CCD_SHAPE = [2048, 3073]
    # The "expected" data used for comparison is derived from refcat2 on Vizier.
    expected_all = Table.read(get_pkg_data_filename("data/all_refcat2_ey_uma.ecsv"))

    wcs_file = get_pkg_data_filename("data/sample_wcs_ey_uma.fits")
    with fits.open(wcs_file) as hdulist:
        with warnings.catch_warnings():
            # Ignore the warning about the WCS having a different number of
            # axes than the (non-existent) image.
            warnings.filterwarnings(
                "ignore",
                message="The WCS transformation has more",
                category=FITSFixedWarning,
            )
            wcs = WCS(hdulist[0].header)
    wcs.pixel_shape = list(reversed(CCD_SHAPE))
    ccd = CCDData(data=np.zeros(CCD_SHAPE), wcs=wcs, unit="adu")

    # Turn this into an HDU to get the standard FITS image keywords
    ccd_im = ccd.to_hdu()
    all_refcat2 = refcat2(
        ccd_im[0].header,
        radius=10 * u.arcmin,
        magnitude_limit=mag_limit,
        magnitude_limit_passband=mag_limit_band,
    )

    # Impose the magnitude limit on the expected result, if any
    if mag_limit is not None:
        expected_all = expected_all[expected_all["rmag"] <= mag_limit]

    # # It is hard to imagine the RAs matching and other entries not matching,
    # # so just check the RAs.
    assert set(ra.value for ra in all_refcat2["ra"]) == set(expected_all["RA_ICRS"])

    # # The passbands ought to have been translated to the AAVSO standard names.
    for band in ["GBP", "GRP", "GG", "SG", "SR", "SI", "SZ", "J", "H", "K"]:
        assert band in all_refcat2["passband"]

    # The Gaia IDs must be unique per star. The catalog is tidy (one row per
    # star per passband), so compare the number of distinct IDs to the number
    # of distinct positions.
    n_stars = len(set(ra.value for ra in all_refcat2["ra"]))
    assert len(set(all_refcat2["id"])) == n_stars

    # Spot check: the star nearest the field center (EY UMa) must carry EY UMa's
    # known Gaia DR2 source_id, i.e. the crossmatch attached the right ID to the
    # right row. EY UMa's refcat2 rmag is 15.24, so it is only present in the
    # runs without a magnitude limit.
    if mag_limit is None:
        ey_uma = SkyCoord.from_name("EY UMa")
        cat_coords = SkyCoord(ra=all_refcat2["ra"], dec=all_refcat2["dec"])
        nearest = cat_coords.separation(ey_uma).argmin()
        assert all_refcat2["id"][nearest] == EY_UMA_GAIA_DR2_ID


def _synthetic_refcat_stars(n=5):
    """A stand-in for the filtered refcat2 table handed to the Gaia ID join."""
    return Table(
        {
            "RA_ICRS": np.linspace(135.0, 135.5, n),
            "DE_ICRS": np.linspace(49.0, 49.5, n),
            "rmag": np.linspace(10.0, 14.0, n),
        }
    )


def _fake_xmatch_result(indices, ang_dist=None, source_ids=None):
    """
    Build a table shaped like an XMatch result: the echoed ``_sp_index``
    column plus ``angDist`` and Gaia's ``source_id``. By default each input
    row ``i`` matches Gaia source ``1000 + i``.
    """
    indices = np.asarray(indices)
    if ang_dist is None:
        ang_dist = np.full(len(indices), 0.001)
    if source_ids is None:
        source_ids = 1000 + indices
    return Table(
        {
            "_sp_index": indices,
            "angDist": ang_dist,
            "source_id": source_ids,
        }
    )


def test_attach_gaia_ids_shuffled_result():
    # XMatch does not preserve input row order; every star must still get its
    # own source_id.
    catalog = _synthetic_refcat_stars(5)
    result = _fake_xmatch_result([3, 0, 4, 1, 2])

    matched = _attach_gaia_ids(catalog, result)

    assert len(matched) == 5
    assert list(matched["id"]) == [1000, 1001, 1002, 1003, 1004]
    # The rest of the catalog must be untouched.
    np.testing.assert_array_equal(matched["RA_ICRS"], catalog["RA_ICRS"])


def test_attach_gaia_ids_unmatched_row_dropped_with_warning():
    # A star with no Gaia match is dropped, with a warning saying how many.
    catalog = _synthetic_refcat_stars(5)
    result = _fake_xmatch_result([0, 1, 3, 4])  # star 2 has no match

    with pytest.warns(UserWarning, match="1 of 5"):
        matched = _attach_gaia_ids(catalog, result)

    assert len(matched) == 4
    assert list(matched["id"]) == [1000, 1001, 1003, 1004]
    assert catalog["RA_ICRS"][2] not in matched["RA_ICRS"]


def test_attach_gaia_ids_duplicate_match_keeps_nearest():
    # One star matching two Gaia sources keeps only the nearest one.
    catalog = _synthetic_refcat_stars(5)
    # Star 1 appears twice; the farther match comes first in the result and
    # carries a bogus source_id that must not survive.
    result = _fake_xmatch_result(
        [0, 1, 1, 2, 3, 4],
        ang_dist=[0.001, 0.008, 0.002, 0.001, 0.001, 0.001],
        source_ids=[1000, 9999, 1001, 1002, 1003, 1004],
    )

    matched = _attach_gaia_ids(catalog, result)

    assert len(matched) == 5
    assert list(matched["id"]) == [1000, 1001, 1002, 1003, 1004]


def test_process_refcat2_uploads_slim_table(monkeypatch):
    # The XMatch upload must contain only the index and coordinate columns,
    # not the full 40+ column refcat2 table, and the echoed index must be
    # used (not row order) to assign IDs.
    n = 6
    catalog = Table(
        {
            "RA_ICRS": np.linspace(135.0, 135.5, n),
            "DE_ICRS": np.linspace(49.0, 49.5, n),
            "e_pmRA": np.ones(n),
            "e_pmDE": np.ones(n),
            "e_Gmag": np.full(n, 0.01),
            "rmag": np.linspace(10.0, 14.0, n),
        },
        masked=True,
    )
    # Row 4 is a galaxy (masked proper-motion errors), row 5 is a non-Gaia
    # star (masked e_Gmag); both must be filtered out before the crossmatch.
    catalog["e_pmRA"].mask = [False, False, False, False, True, False]
    catalog["e_pmDE"].mask = [False, False, False, False, True, False]
    catalog["e_Gmag"].mask = [False, False, False, False, False, True]

    captured = {}

    def fake_query(cat1=None, **kwargs):  # noqa: ARG001
        captured["cat1"] = cat1
        # Echo the index back in reversed order, like XMatch reordering rows.
        indices = np.asarray(cat1["_sp_index"])[::-1]
        return _fake_xmatch_result(indices)

    import stellarphot.catalogs as catalogs_module

    monkeypatch.setattr(catalogs_module.XMatch, "query", fake_query)

    processed = _process_refcat2(catalog)

    assert set(captured["cat1"].colnames) == {"_sp_index", "RA_ICRS", "DE_ICRS"}
    # Only the four genuine Gaia stars survive the filters and are uploaded.
    np.testing.assert_array_equal(captured["cat1"]["_sp_index"], np.arange(4))
    assert len(processed) == 4
    assert list(processed["id"]) == [1000, 1001, 1002, 1003]


@pytest.mark.parametrize("catalog", [apass_dr9, refcat2, vsx_vizier])
def test_catalog_errors(catalog):
    # Check some of the errors expected in catalog classes

    # Giving a bad band should raise an error
    error_msg = (
        "magnitude_limit_passband must be one of"
        if catalog is not vsx_vizier
        else "no straightforward way to limit the VSX catalog by passband"
    )
    with pytest.raises(ValueError, match=error_msg):
        catalog(
            {},  # Dummy header
            radius=0.1 * u.arcmin,
            magnitude_limit=13,
            magnitude_limit_passband="not a band, more like an ensemble",
        )

    # Giving a band but not a magnitude limit should raise an error except
    # for vsx_vizier, which has weird settings because the VSX catalog mag
    # column has a bunch of different passbands in it.
    if catalog is not vsx_vizier:
        # Need to provide a valid passband for each catalog
        passband = "SR"
        with pytest.raises(ValueError, match="you provide a .* you must also provide"):
            catalog(
                {},  # Dummy header
                radius=0.1 * u.arcmin,
                magnitude_limit_passband=passband,
            )

    # Giving a magnitude limit but not a band should not raise an error because each
    # catalog has a default passband.
    #
    # Since magnitude limit functionality is test elsewhere, we don't test it here.
