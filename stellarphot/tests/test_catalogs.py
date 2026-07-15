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
    _iau_designation_ids,
    _process_refcat2,
    apass_dr9,
    refcat2,
    vsx_vizier,
)

# ICRS coordinates of EY UMa, the center of the field used in the remote-data
# refcat2 tests. Values from SIMBAD; a fixed coordinate is used instead of
# SkyCoord.from_name so the tests do not depend on the Sesame name resolver
# being up.
EY_UMA_COORD = SkyCoord(ra=135.58650087 * u.deg, dec=49.81921088 * u.deg)

# The coordinate-based designation refcat2() generates for EY UMa, built by
# truncating the star's RA_ICRS/DE_ICRS in data/all_refcat2_ey_uma.ecsv
# (135.58651998, 49.81918018) to five decimal places.
EY_UMA_REFCAT2_ID = "REFCAT2SP J135.58651+49.81918"


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
    # designation following IAU guidelines from the star's APASS coordinates
    # (RA 359.989602, Dec 0.012185; coordinates truncated, not rounded, so
    # the Dec gives .0121).
    assert apass["id"][0] == "APASSSP J359.9896+00.0121"

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

    # The IDs must be unique per star. The catalog is tidy (one row per
    # star per passband), so compare the number of distinct IDs to the number
    # of distinct positions.
    n_stars = len(
        set(zip(all_refcat2["ra"].value, all_refcat2["dec"].value, strict=True))
    )
    assert len(set(all_refcat2["id"])) == n_stars

    # Spot check: the star nearest the field center (EY UMa) must carry the
    # coordinate-based designation built from its own RA/Dec, i.e. the ID
    # generation attached the right ID to the right row. EY UMa's refcat2
    # rmag is 15.24, so it is only present in the runs without a magnitude
    # limit.
    if mag_limit is None:
        cat_coords = SkyCoord(ra=all_refcat2["ra"], dec=all_refcat2["dec"])
        nearest = cat_coords.separation(EY_UMA_COORD).argmin()
        assert all_refcat2["id"][nearest] == EY_UMA_REFCAT2_ID


def test_iau_designation_ids_format():
    # The designation format: acronym, a space, then J + RA and Dec in
    # degrees, RA unsigned and Dec signed per the IAU spec, zero-padded (RA
    # to three digits before the decimal, Dec to two) and, at the default
    # precision, four digits after the decimal. Per the IAU spec the
    # coordinates are truncated, not rounded, so 359.98957 gives .9895 and
    # 0.01223 gives .0122.
    ids = _iau_designation_ids(
        "REFCAT2SP",
        [135.0, 45.5, 359.98957],
        [49.0, -0.5, 0.01223],
    )
    assert ids == [
        "REFCAT2SP J135.0000+49.0000",
        "REFCAT2SP J045.5000-00.5000",
        "REFCAT2SP J359.9895+00.0122",
    ]


def test_iau_designation_ids_matches_apass_star():
    # The helper must reproduce the designation apass_dr9 generates for the
    # star in test_catalog_from_vizier_search_apass (RA 359.989602,
    # Dec 0.012185, truncated to four decimals).
    assert _iau_designation_ids("APASSSP", [359.989602], [0.012185]) == [
        "APASSSP J359.9896+00.0121"
    ]


def test_iau_designation_ids_precision():
    # precision sets the number of decimal digits; truncation applies at
    # whatever precision is requested.
    assert _iau_designation_ids(
        "REFCAT2SP", [135.58651998], [49.81918018], precision=5
    ) == ["REFCAT2SP J135.58651+49.81918"]


def test_iau_designation_ids_truncates_at_bin_edges():
    # Truncation must be exact at bin edges: naive trunc(x * 10**p) / 10**p
    # turns 135.5865 into .5864 (135.5865 * 10_000 is 1355864.999...), and
    # format-then-drop-a-digit turns 135.58651998 into .58652 via round-up
    # carry. Both inputs must come out unchanged.
    assert _iau_designation_ids("REFCAT2SP", [135.5865], [49.8191]) == [
        "REFCAT2SP J135.5865+49.8191"
    ]
    assert _iau_designation_ids(
        "REFCAT2SP", [135.58651998], [49.81918018], precision=5
    ) == ["REFCAT2SP J135.58651+49.81918"]


def test_iau_designation_ids_boundaries():
    # RA just below 360 must truncate to 359.9999, not round out of range to
    # 360.0000, and Dec just below the pole must stay below 90. RA at or
    # above 360 wraps into [0, 360).
    assert _iau_designation_ids("REFCAT2SP", [359.99996], [89.99996]) == [
        "REFCAT2SP J359.9999+89.9999"
    ]
    assert _iau_designation_ids("REFCAT2SP", [360.0], [-89.99996]) == [
        "REFCAT2SP J000.0000-89.9999"
    ]


def test_iau_designation_ids_no_collision_at_refcat2_precision():
    # Two stars 0.11 arcsec apart collide in the same bin at the default
    # four decimals but get distinct ids at refcat2's five decimals.
    ra = [135.00001, 135.00004]
    dec = [49.00001, 49.00004]
    default_ids = _iau_designation_ids("REFCAT2SP", ra, dec)
    assert default_ids[0] == default_ids[1]
    refcat2_ids = _iau_designation_ids("REFCAT2SP", ra, dec, precision=5)
    assert refcat2_ids == [
        "REFCAT2SP J135.00001+49.00001",
        "REFCAT2SP J135.00004+49.00004",
    ]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_iau_designation_ids_rejects_non_finite(bad):
    # A masked or corrupt coordinate must raise instead of silently
    # producing a garbage designation.
    with pytest.raises(ValueError, match="finite"):
        _iau_designation_ids("REFCAT2SP", [bad], [49.0])
    with pytest.raises(ValueError, match="finite"):
        _iau_designation_ids("REFCAT2SP", [135.0], [bad])


def test_process_refcat2_filters_and_ids():
    # _process_refcat2 must drop galaxies and non-Gaia objects, then give
    # every surviving star a coordinate-based designation — all offline, with
    # no crossmatch service involved.
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
    # star (masked e_Gmag); both must be filtered out.
    catalog["e_pmRA"].mask = [False, False, False, False, True, False]
    catalog["e_pmDE"].mask = [False, False, False, False, True, False]
    catalog["e_Gmag"].mask = [False, False, False, False, False, True]

    processed = _process_refcat2(catalog)

    assert len(processed) == 4
    assert list(processed["id"]) == [
        "REFCAT2SP J135.00000+49.00000",
        "REFCAT2SP J135.10000+49.10000",
        "REFCAT2SP J135.20000+49.20000",
        "REFCAT2SP J135.30000+49.30000",
    ]


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
