import logging
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.exceptions import AstropyUserWarning

from stellarphot.conftest import SERVER_DOWN_ERRORS

from ...catalogs import apass_dr9, refcat2
from ...core import PhotometryData
from .. import magnitude_transforms
from ..magnitude_system_transforms import (
    transform_apass_bands,
    transform_refcat2_bands,
)
from ..magnitude_transforms import (
    _MIN_FIT_SIGMA,
    _excess_scatter,
    _to_float_array,
    calibrated_from_instrumental,
    filter_transform,
    transform_to_catalog,
)

# Logger name for magnitude_transforms module
_MAGNITUDE_TRANSFORMS_LOGGER = "stellarphot.utils.magnitude_transforms"


def _generate_star_coordinates(
    n_stars, ra_start=180 * u.degree, dec_start=45 * u.degree, separation=10 * u.arcsec
):
    """
    Generate RA/Dec coordinates for a set of stars on a square grid.

    Parameters
    ----------

    n_stars : int
        Number of coordinates to generate.

    ra_start : `astropy.units.Quantity`, optional
        Right ascension of the first star.

    dec_start : `astropy.units.Quantity`, optional
        Declination of the first star.

    separation : `astropy.units.Quantity`, optional
        Spacing between adjacent grid positions in right ascension and in
        declination.

    Returns
    -------
    ra : `astropy.units.Quantity`
        Right ascension of each star.

    dec : `astropy.units.Quantity`
        Declination of each star.
    """
    # The plus one guarantees we'll have enough positions for
    # all of our stars.
    max_index = int(np.sqrt(n_stars)) + 1

    grids = np.mgrid[:max_index, :max_index]
    dec_grid, ra_grid = grids
    dec_offsets = separation * dec_grid
    ra_offsets = separation * ra_grid
    ra = (ra_start + ra_offsets).flatten()
    dec = (dec_start + dec_offsets).flatten()

    # Slice to return the correct number of positions.
    return ra[:n_stars], dec[:n_stars]


@pytest.mark.parametrize("bad_system", [None, "monkeys"])
def test_filter_transform_bad_system(bad_system):
    fake_data = Table()
    with pytest.raises(ValueError) as e:
        filter_transform(fake_data, "B", transform=bad_system)
    assert "Must be one of" in str(e.value)
    assert str(bad_system) in str(e.value)


@pytest.mark.parametrize("system", ["ivezic", "jester"])
def test_filter_transform(system):
    data_file = get_pkg_data_filename("data/mag_transform.csv")
    data = Table.read(data_file)
    in_system = data["system"] == system
    data = data[in_system]
    for output_filter in ["B", "V", "R", "I"]:
        f = filter_transform(data, output_filter, g="g", r="r", i="i", transform=system)
        np.testing.assert_allclose(f, data[output_filter])


def test_filter_transform_bad_filter():
    with pytest.raises(ValueError) as e:
        filter_transform([], "not a filter", transform="jester")
    assert "the desired filter must be a string R B V or I" in str(e)


class _FakeCatalogTable(Table):
    """
    Table that quacks like a stellarphot catalog just enough for
    transform_to_catalog, which calls passband_columns on the catalog.

    The arguments of the `passband_columns` call are recorded in ``meta`` so
    that tests can check which transformer the catalog was asked for.
    """

    def passband_columns(self, passbands=None, transformer=None):
        self.meta["passband_columns_call"] = {
            "passbands": passbands,
            "transformer": transformer,
        }
        return self


def test_calibrated_from_instrumental_matches_hand_computed_values():
    # Everything below generates its synthetic catalogs by calling this
    # function and then fits them with the same function, which is what keeps
    # the tests from encoding a second, drifting copy of the model -- but it
    # also means a mistake in the model itself would appear identically on
    # both sides of every comparison and cancel out. These numbers are worked
    # out by hand from
    #
    #     mag_cal = a * mag_inst + b * mag_inst**2 + c * color + d * color**2 + z
    #
    # so a sign flip on any term, a swap of the two arguments, or a wrong
    # exponent shows up here.
    mag_inst = np.array([-10.0, -5.0])
    color = np.array([0.5, 2.0])
    a, b, c, d, z = 0.1, 0.01, 0.2, 0.03, 20.0

    hand_computed = np.array(
        [
            # 0.1 * -10 + 0.01 * 100 + 0.2 * 0.5 + 0.03 * 0.25 + 20
            -1.0 + 1.0 + 0.1 + 0.0075 + 20.0,
            # 0.1 * -5 + 0.01 * 25 + 0.2 * 2 + 0.03 * 4 + 20
            -0.5 + 0.25 + 0.4 + 0.12 + 20.0,
        ]
    )

    np.testing.assert_allclose(
        calibrated_from_instrumental((mag_inst, color), a, b, c, d, z),
        hand_computed,
        rtol=0,
        atol=1e-12,
    )


# Default zero point of the synthetic catalog below. The catalog magnitudes
# follow the fit model, so a fit to them recovers whichever coefficients were
# used to build the catalog.
_FAKE_CATALOG_ZERO_POINT = 20.0

# Scatter added to the synthetic catalog magnitudes. Without it they follow the
# model to the last bit, the residual of a fit to them reaches exactly zero,
# and lmfit divides zero by zero working out the correlations between the
# parameters -- a RuntimeWarning, which this suite turns into a failure. Real
# photometry always has some scatter, so the fix belongs here rather than in
# the production code. This is far below every tolerance asserted anywhere in
# this file: a fit to a catalog built this way still recovers the coefficients
# it was built from to about 5e-12, six orders of magnitude inside the 1e-6
# the recovery tests allow.
_FAKE_CATALOG_SCATTER = 1e-12

# Error the synthetic catalog reports for each of its magnitudes. A real
# catalog reports one for every band it measured itself, and the fit weights
# by it, so a catalog with no errors at all is the exception rather than the
# rule -- it is what a *transformed* band looks like, which the tests that
# care about say explicitly by passing ``cat_error=None``. Small enough
# against the observed errors used here that it changes no number any other
# test in this file asserts on: the fit weights are 1/sqrt(obs^2 + cat^2), so
# a catalog error this far below the observed ones moves the weights, and the
# uncertainties believed from them, by well under a percent. The tests that
# predict an uncertainty to better than that fold it into their prediction.
_FAKE_CATALOG_ERROR = 0.001


def _generate_fake_catalog(
    n_stars,
    a=0.0,
    b=0.0,
    c=0.0,
    d=0.0,
    z=_FAKE_CATALOG_ZERO_POINT,
    coordinates=None,
    instrumental=None,
    color=None,
    cat_error=_FAKE_CATALOG_ERROR,
):
    """
    Generate a catalog whose magnitudes follow the transform model.

    The catalog magnitudes are built by calling the production model,
    `~stellarphot.utils.magnitude_transforms.calibrated_from_instrumental`,
    rather than re-deriving the arithmetic, so the tests cannot drift away
    from the model actually being fit. A fit to the result recovers the
    coefficients passed in here to within `_FAKE_CATALOG_SCATTER`, the
    negligible amount of scatter added so that the magnitudes do not follow
    the model to the last bit. The model itself is pinned independently,
    against hand-computed values, by
    `test_calibrated_from_instrumental_matches_hand_computed_values`.

    Parameters
    ----------

    n_stars : int
        Number of stars to generate.

    a, b, c, d, z : float, optional
        Coefficients of the transform model used to build the catalog
        magnitudes. The defaults make the catalog magnitude the instrumental
        magnitude plus `_FAKE_CATALOG_ZERO_POINT`.

    coordinates : tuple of `astropy.units.Quantity`, optional
        Right ascension and declination of the stars. Generated on a grid if
        not given.

    instrumental : `numpy.ndarray`, optional
        Instrumental magnitudes to build the catalog from. Evenly spaced over
        the range the fit accepts if not given.

    color : `numpy.ndarray`, optional
        Color of each star. Random, but seeded, if not given.

    cat_error : float or array-like, optional
        Error to report for each catalog magnitude, in the
        ``mag_error_R`` and ``mag_error_I`` columns a real catalog carries for
        the bands it measured itself. Pass `None` for a catalog with no error
        columns at all, which is what a band transformed from other bands
        looks like -- and is what makes `transform_to_catalog` fall back to
        weighting by the observed errors alone, with a warning.

    Returns
    -------
    catalog : `_FakeCatalogTable`
        Catalog with columns ``ra``, ``dec``, ``mag_R`` and ``mag_I``, and,
        unless ``cat_error`` is `None`, ``mag_error_R`` and ``mag_error_I``.

    ra : `astropy.units.Quantity`
        Right ascension of each star.

    dec : `astropy.units.Quantity`
        Declination of each star.

    instrumental : `numpy.ndarray`
        Instrumental magnitudes from which the catalog magnitudes were
        generated.
    """
    if coordinates is None:
        ra, dec = _generate_star_coordinates(n_stars)
    else:
        ra, dec = coordinates

    if instrumental is None:
        instrumental = np.linspace(-10.0, -5.0, num=n_stars)

    if color is None:
        # The color must not be a linear function of the instrumental
        # magnitude. If it is, the a, c and z terms of the model are exactly
        # degenerate -- any combination that adds up to the same thing fits
        # equally well -- and no fitter can recover the individual
        # coefficients. The generator is seeded, so the colors are the same
        # from one run to the next.
        color = np.random.default_rng(432).uniform(0.0, 1.0, size=n_stars)

    # The trailing ``+ instrumental`` is the fit_diff=True offset: the model is
    # fit to the difference between the catalog and instrumental magnitudes.
    cat_r = (
        calibrated_from_instrumental((instrumental, color), a, b, c, d, z)
        + instrumental
    )

    # Seeded, so the scatter is the same from one run to the next. Added only
    # to mag_R, which leaves the color of each star exactly what was asked for.
    cat_r = cat_r + np.random.default_rng(9021).normal(
        0.0, _FAKE_CATALOG_SCATTER, size=np.shape(cat_r)
    )

    catalog = _FakeCatalogTable(
        {
            "ra": ra,
            "dec": dec,
            "mag_R": cat_r,
            "mag_I": cat_r - color,
        }
    )

    if cat_error is not None:
        errors = np.broadcast_to(
            np.asarray(cat_error, dtype=float), np.shape(cat_r)
        ).copy()
        catalog["mag_error_R"] = errors
        catalog["mag_error_I"] = errors.copy()

    return catalog, ra, dec, instrumental


def _catalog_posing_as_bands(catalog, **new_from_old):
    """
    Copy a catalog's magnitudes under other passband names.

    ``_catalog_posing_as_bands(catalog, SR="R", SI="I")`` gives the catalog a
    ``mag_SR`` column holding exactly what ``mag_R`` holds, along with the
    matching ``mag_error_SR``, so that a catalog built in R and I can stand in
    for one in any other pair of bands. Only the names change, so a fit
    against the renamed band recovers exactly what it recovers against the
    original.

    Parameters
    ----------

    catalog : `_FakeCatalogTable`
        Catalog to add the columns to. Modified in place.

    **new_from_old
        The new passband names, each with the name of the passband whose
        columns it should copy.

    Returns
    -------
    `_FakeCatalogTable`
        The catalog, for convenience.
    """
    for new, old in new_from_old.items():
        catalog[f"mag_{new}"] = catalog[f"mag_{old}"]
        old_error = f"mag_error_{old}"
        if old_error in catalog.colnames:
            catalog[f"mag_error_{new}"] = catalog[old_error]

    return catalog


def _generate_observed_table(
    ra,
    dec,
    instrumental,
    file_name="image_1.fit",
    passband="R",
    mag_error=None,
    noise_sigma=0.0,
    seed=None,
):
    """
    Generate observations in the form ``transform_to_catalog`` expects.

    Parameters
    ----------

    ra : `astropy.units.Quantity`
        Right ascension of each observed star.

    dec : `astropy.units.Quantity`
        Declination of each observed star.

    instrumental : `numpy.ndarray`
        Instrumental magnitude of each observed star.

    file_name : str, optional
        Name of the image these observations came from, i.e. the value of the
        ``file`` column the observations are grouped by.

    passband : str, optional
        Passband to label every observation with.

    mag_error : float or array-like, optional
        Uncertainty to report for each instrumental magnitude. Defaults to
        ``noise_sigma`` when noise is added -- the truthful value -- and to
        0.01 otherwise.

    noise_sigma : float, optional
        Standard deviation of Gaussian noise added to the instrumental
        magnitudes. Tests of coefficient recovery leave this at zero; tests
        of the fit diagnostics need data that misses the model by a known,
        realistic amount.

    seed : int, optional
        Seed for the noise, so that a test gets the same numbers every run.

    Returns
    -------
    `astropy.table.Table`
        Observations of a single image, grouped by file name, with the
        ``file``, ``passband``, ``ra``, ``dec``, ``mag_inst`` and
        ``mag_error`` columns `transform_to_catalog` requires.
    """
    if mag_error is None:
        mag_error = noise_sigma if noise_sigma else 0.01

    instrumental = np.asarray(instrumental, dtype=float)
    if noise_sigma:
        instrumental = instrumental + np.random.default_rng(seed).normal(
            0.0, noise_sigma, size=instrumental.shape
        )

    n_stars = len(instrumental)

    observed = Table(
        {
            "file": [file_name] * n_stars,
            "passband": [passband] * n_stars,
            "ra": ra.to_value("degree"),
            "dec": dec.to_value("degree"),
            "mag_inst": instrumental,
            "mag_error": np.broadcast_to(
                np.asarray(mag_error, dtype=float), (n_stars,)
            ).copy(),
        }
    )

    return observed.group_by("file")


def _combine_observed_tables(*tables):
    """
    Stack several per-image observation tables and regroup them by file.

    Parameters
    ----------

    *tables : `astropy.table.Table`
        Tables from `_generate_observed_table`.

    Returns
    -------
    `astropy.table.Table`
        The stacked observations, grouped by file name.
    """
    return vstack([Table(table) for table in tables]).group_by("file")


def _patch_catalog_fetch(mocker, catalog, cat_name="apass_dr9"):
    """
    Make the catalog fetch return a synthetic catalog instead of querying.

    Patching the fetch keeps the tests offline, so they do not need the
    ``remote_data`` marker.

    Parameters
    ----------

    mocker : `pytest_mock.MockerFixture`
        Fixture used to patch the catalog fetch.

    catalog : `_FakeCatalogTable`
        Catalog to use in place of the fetched one.

    cat_name : str, optional
        Name of the catalog to patch. The function of that name in
        `~stellarphot.utils.magnitude_transforms` is the one replaced, so
        ``"refcat2"`` is mocked exactly the way ``"apass_dr9"`` is.
    """
    mocker.patch.object(magnitude_transforms, cat_name, return_value=catalog)


def _run_transform_to_catalog(
    mocker, catalog, observed, obs_filter="R", cat_name="apass_dr9", **kwargs
):
    """
    Run ``transform_to_catalog`` against a synthetic catalog.

    Parameters
    ----------

    mocker : `pytest_mock.MockerFixture`
        Fixture used to patch the catalog fetch.

    catalog : `_FakeCatalogTable`
        Catalog to use in place of the fetched one.

    observed : `astropy.table.Table`
        Observations to transform, grouped by file name.

    obs_filter : str, optional
        Passband of the observations to transform.

    cat_name : str, optional
        Name of the catalog to use, passed to `_patch_catalog_fetch` as well
        as to the function under test.

    **kwargs
        Passed on to `~stellarphot.utils.magnitude_transforms.transform_to_catalog`,
        overriding the defaults this helper supplies.

    Returns
    -------
    `astropy.table.Table`
        The observations with the calibrated magnitude, fit coefficient and
        matched-catalog columns added.
    """
    _patch_catalog_fetch(mocker, catalog, cat_name=cat_name)

    # Only the error column is supplied here. The catalog band and color are
    # left to default from ``obs_filter``, because naming a catalog band that
    # is not the observed one is now an error, and a helper that quietly
    # passed ``cat_filter="R"`` would make every call for another band one.
    call_kwargs = {"obs_error_column": "mag_error"}
    call_kwargs.update(kwargs)

    return transform_to_catalog(observed, obs_filter, cat_name=cat_name, **call_kwargs)


# Every output column whose value comes from the catalog entry a star was
# matched to. All of them stand or fall together: a star matched closely
# enough to be calibrated keeps the catalog magnitude and color it was
# calibrated against, and a star that is not gets none of them.
_CATALOG_DERIVED_COLUMNS = ("mag_cal", "mag_cal_error", "mag_cat", "color_cat")


def test_transform_to_catalog_excludes_distant_matches(mocker):
    # A star whose nearest catalog match is far away (much more than
    # 1 arcsec) should be excluded from the fit for the transform
    # coefficients. See issue #588 -- an operator precedence error
    # disabled the distance cut, so badly-matched stars polluted
    # the fit.
    n_good = 20

    # Good stars sit exactly on top of their catalog counterparts.
    catalog, ra, dec, good_mags = _generate_fake_catalog(n_good)

    # One more observed star, a degree away from every catalog star, so
    # its nearest catalog match is bogus. Its instrumental magnitude is
    # chosen so that the bogus match is 0.5 magnitude off the true zero
    # point -- close enough to survive the median-based outlier cut, so
    # only the distance cut can remove it from the fit.
    bad_ra = ra[0]
    bad_dec = dec[0] - 1 * u.degree
    bad_mag = good_mags[0] - 0.5

    observed = _generate_observed_table(
        u.Quantity([*ra, bad_ra]),
        u.Quantity([*dec, bad_dec]),
        np.append(good_mags, bad_mag),
    )

    result = _run_transform_to_catalog(mocker, catalog, observed)

    # The distant star has no real match, so nothing derived from that match
    # should come back with a value -- the catalog magnitude and color are of
    # an unrelated star, and an error without a magnitude to go with it is
    # worse than no error at all. See issue #678.
    for column in _CATALOG_DERIVED_COLUMNS:
        assert np.isnan(result[column][-1]), column

    # With the distant star excluded from the fit, the good stars are
    # an exact fit, so their calibrated magnitudes should match the
    # catalog. If the bad match leaks into the fit it drags the
    # coefficients away from the true values.
    np.testing.assert_allclose(
        result["mag_cal"][:n_good],
        good_mags + _FAKE_CATALOG_ZERO_POINT,
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result["z"][:n_good], _FAKE_CATALOG_ZERO_POINT, rtol=0, atol=1e-6
    )


@pytest.mark.parametrize(
    "offset, expect_nan",
    [
        # The VSX and APASS DR9 positions of V2480 Cyg differ by about this much.
        (1.3 * u.arcsec, False),
        # Far enough away that the match is no longer plausible.
        (1.6 * u.arcsec, True),
    ],
)
def test_transform_to_catalog_match_tolerance(mocker, offset, expect_nan):
    # A star can be a little more than an arcsec from its catalog position --
    # the VSX position of a variable need not agree that closely with its
    # APASS position -- and should still end up with a calibrated magnitude.
    # It should not, however, be used in the fit for the transform
    # coefficients, which requires a match within 1 arcsec. See issue #668.
    n_good = 20

    catalog, ra, dec, good_mags = _generate_fake_catalog(n_good)

    # An extra observation of the first catalog star, offset in declination so
    # that the separation from its catalog position is exactly the offset. Its
    # instrumental magnitude is 0.5 magnitude off the true zero point -- close
    # enough to survive the median-based outlier cut, so if it makes it into
    # the fit it drags the zero point along with it.
    offset_mag = good_mags[0] - 0.5

    observed = _generate_observed_table(
        u.Quantity([*ra, ra[0]]),
        u.Quantity([*dec, dec[0] + offset]),
        np.append(good_mags, offset_mag),
    )

    result = _run_transform_to_catalog(mocker, catalog, observed)

    # The offset star gets a calibrated magnitude only if its match is close
    # enough -- and everything else derived from that match goes with it,
    # rather than mag_cat and color_cat being kept for a match too distant to
    # calibrate against. This is where that decision is pinned. See issue #678.
    for column in _CATALOG_DERIVED_COLUMNS:
        assert np.isnan(result[column][-1]) == expect_nan, column

    if not expect_nan:
        # The fit is exact, so the calibrated magnitude is just the
        # instrumental magnitude plus the zero point, and the catalog values
        # are those of the star it was matched to.
        assert result["mag_cal"][-1] == pytest.approx(
            offset_mag + _FAKE_CATALOG_ZERO_POINT, abs=1e-6
        )
        assert result["mag_cat"][-1] == pytest.approx(catalog["mag_R"][0], abs=1e-12)
        assert result["color_cat"][-1] == pytest.approx(
            catalog["mag_R"][0] - catalog["mag_I"][0], abs=1e-12
        )

    # Either way the offset star is more than an arcsec from its catalog
    # position, so it should be left out of the fit, leaving the good stars
    # as an exact fit.
    np.testing.assert_allclose(
        result["mag_cal"][:n_good],
        good_mags + _FAKE_CATALOG_ZERO_POINT,
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result["z"][:n_good], _FAKE_CATALOG_ZERO_POINT, rtol=0, atol=1e-6
    )


# Everything that describes the fit for one image rather than one star: the
# coefficients, how well each of them was pinned down, and how well the model
# came out fitting the data. All of them are repeated down every row of the
# image, and all of them are NaN for an image that could not be fit, so the
# tests for an unfittable image loop over the lot.
_FIT_COLUMNS = (
    "a",
    "b",
    "c",
    "d",
    "z",
    "a_error",
    "b_error",
    "c_error",
    "d_error",
    "z_error",
    "fit_redchi",
    "fit_cat_error_missing_frac",
    "fit_max_weight_share",
    "fit_excess_scatter",
)

# The columns transform_to_catalog adds to the table it is given.
_TRANSFORM_OUTPUT_COLUMNS = {
    "mag_cal",
    "mag_cal_error",
    *_FIT_COLUMNS,
    "mag_cat",
    "color_cat",
}

# The same, less the one column that exists to tell apart a catalog error the
# fit cannot use from one it can. A star whose catalog error is missing, zero
# or negative is weighted exactly as if the catalog had reported a negligible
# error, so every column below must come out identical between those two runs
# -- but `fit_cat_error_missing_frac` reports which of the two it was, and so
# must not. See issue #694.
_COLUMNS_BLIND_TO_UNUSABLE_CATALOG_ERRORS = _TRANSFORM_OUTPUT_COLUMNS - {
    "fit_cat_error_missing_frac"
}


def _fit_a_catalog(
    mocker,
    n_stars=20,
    vary=None,
    sigma=0.0,
    seed=None,
    mag_error=None,
    min_fit_sigma=None,
    **catalog_coefficients,
):
    """
    Build a synthetic catalog and observations of it, then transform them.

    Left at its defaults (``sigma=0.0``) this generates noiseless
    observations, for tests that check *where* the fit lands. Passing a
    nonzero ``sigma``, and usually a ``seed`` so the noise is reproducible,
    generates noisy observations instead, for tests that ask what the fit's
    own uncertainties *mean* -- a noiseless fit is exact and has nothing to
    report an uncertainty about. Noiseless observations taken straight from
    the magnitudes a synthetic catalog was built from fit it to within
    `_FAKE_CATALOG_SCATTER`, which leaves every uncertainty derived from a
    noiseless fit at the 1e-13 level; see the ``noise_sigma`` parameter of
    `_generate_observed_table` for the noise added on top of that.

    Parameters
    ----------

    mocker : `pytest_mock.MockerFixture`
        Fixture used to patch the catalog fetch.

    n_stars : int, optional
        Number of stars to generate.

    vary : sequence of str, optional
        Terms to fit. The default of `~stellarphot.utils.transform_to_catalog`
        is used if this is not given.

    sigma : float, optional
        Standard deviation of Gaussian noise added to the observed
        magnitudes. Zero, the default, adds none.

    seed : int, optional
        Seed for that noise, so a test gets the same numbers every run.
        Meaningless when ``sigma`` is zero.

    mag_error : float or array-like, optional
        Uncertainty to report for each instrumental magnitude, passed on to
        `_generate_observed_table`. Left at that function's own default --
        ``sigma`` when noise was added, ``0.01`` otherwise -- if not given
        here.

    min_fit_sigma : float, optional
        Passed on to `~stellarphot.utils.transform_to_catalog`, whose
        default is used if this is not given.

    **catalog_coefficients
        Coefficients ``a``, ``b``, ``c``, ``d`` and ``z`` of the transform
        model used to build the catalog, passed to `_generate_fake_catalog`.

    Returns
    -------
    result : `astropy.table.Table`
        The transformed observations.

    catalog : `_FakeCatalogTable`
        The catalog the observations were transformed against.

    instrumental : `numpy.ndarray`
        Instrumental magnitude of each star, i.e. the magnitudes the catalog
        was generated from. When ``sigma`` is nonzero this is the noiseless
        value -- the observed ``mag_inst`` column of ``result`` is these plus
        the noise.
    """
    catalog, ra, dec, instrumental = _generate_fake_catalog(
        n_stars, **catalog_coefficients
    )
    observed = _generate_observed_table(
        ra, dec, instrumental, mag_error=mag_error, noise_sigma=sigma, seed=seed
    )
    fit_kwargs = {} if vary is None else {"vary": vary}
    if min_fit_sigma is not None:
        fit_kwargs["min_fit_sigma"] = min_fit_sigma
    result = _run_transform_to_catalog(mocker, catalog, observed, **fit_kwargs)

    return result, catalog, instrumental


@pytest.fixture(scope="module")
def _noisy_fit_result(module_mocker):
    """
    The one noisy fit shared by tests that ask what its uncertainties mean.

    `test_transform_to_catalog_reports_coefficient_uncertainties`,
    `test_transform_to_catalog_error_includes_the_transform_uncertainty` and
    `test_transform_to_catalog_error_uses_the_whole_covariance` all read --
    and none of them mutate -- the result of fitting the same catalog: 50
    stars, noise sigma 0.02, seed 20260811, a=0.02 and c=0.15. Computing it
    once here rather than three times keeps the tests from drifting apart on
    inputs they mean to share.
    """
    result, _, _ = _fit_a_catalog(
        module_mocker, n_stars=50, sigma=0.02, seed=20260811, a=0.02, c=0.15
    )
    # The patch has done its job once the fit is computed. Left in place it
    # would outlive every later test in the module -- module_mocker unwinds at
    # module teardown -- and hand the remote-data tests this fake catalog in
    # place of the real fetch.
    module_mocker.stopall()
    return result


@pytest.fixture(scope="module")
def _unweighted_fit_result(module_mocker):
    """
    The one unweighted fit shared by tests that ask what it reports.

    `test_transform_to_catalog_reports_unweighted_fit_statistic` and
    `test_transform_to_catalog_diagnostics_for_an_unweighted_fit` both read
    -- and neither mutates -- the result of the same fit with no error
    column: 100 stars, noise sigma 0.02, seed 13579. Computing it once here
    rather than twice keeps the tests from drifting apart on inputs they
    mean to share.
    """
    n_stars = 100

    catalog, ra, dec, instrumental = _generate_fake_catalog(n_stars)
    observed = _generate_observed_table(
        ra, dec, instrumental, noise_sigma=0.02, seed=13579
    )

    with pytest.warns(AstropyUserWarning, match="rror weighting"):
        result = _run_transform_to_catalog(
            module_mocker, catalog, observed, obs_error_column=None
        )
    # Same reasoning as `_noisy_fit_result` just above: the patch must not
    # outlive the fit it was made for.
    module_mocker.stopall()
    return result


@pytest.mark.parametrize("c", [0.0, 0.15, -0.2])
def test_transform_to_catalog_recovers_color_coefficient(mocker, c):
    # The color term is the reason a transform is needed at all, so a fit that
    # ignored color entirely should not be able to pass.
    result, catalog, _ = _fit_a_catalog(mocker, c=c)

    np.testing.assert_allclose(result["c"], c, rtol=0, atol=1e-6)
    np.testing.assert_allclose(result["mag_cal"], catalog["mag_R"], rtol=0, atol=1e-6)


@pytest.mark.parametrize("a", [0.0, 0.02, -0.03])
def test_transform_to_catalog_recovers_scale_coefficient(mocker, a):
    # The companion to the color test above: how the calibrated magnitude
    # depends on the instrumental one. Both signs are tried because a is small,
    # and a fit that got its sign backwards would still land close to the truth
    # when a is zero.
    result, catalog, _ = _fit_a_catalog(mocker, a=a)

    np.testing.assert_allclose(result["a"], a, rtol=0, atol=1e-6)
    np.testing.assert_allclose(result["mag_cal"], catalog["mag_R"], rtol=0, atol=1e-6)


def test_transform_to_catalog_recovers_a_and_c_together(mocker):
    # The scale and color terms can trade off against each other, so recovering
    # each one on its own is not enough.
    a = 0.02
    c = 0.15

    result, catalog, _ = _fit_a_catalog(mocker, a=a, c=c)

    np.testing.assert_allclose(result["a"], a, rtol=0, atol=1e-6)
    np.testing.assert_allclose(result["c"], c, rtol=0, atol=1e-6)
    np.testing.assert_allclose(result["mag_cal"], catalog["mag_R"], rtol=0, atol=1e-6)


def test_transform_to_catalog_recovers_quadratic_terms(mocker):
    # The quadratic terms are held at zero by default, so nothing else in this
    # file ever fits them -- a wrong exponent or sign confined to b or d would
    # be invisible. Here the data really does contain them and they are asked
    # for by name.
    a, b, c, d = 0.02, 0.01, 0.15, 0.05

    result, catalog, _ = _fit_a_catalog(
        mocker, a=a, b=b, c=c, d=d, vary=("a", "b", "c", "d", "z")
    )

    for term, value in (("a", a), ("b", b), ("c", c), ("d", d)):
        np.testing.assert_allclose(result[term], value, rtol=0, atol=1e-6)
    np.testing.assert_allclose(result["z"], _FAKE_CATALOG_ZERO_POINT, rtol=0, atol=1e-6)
    np.testing.assert_allclose(result["mag_cal"], catalog["mag_R"], rtol=0, atol=1e-6)


def test_transform_to_catalog_omitted_quadratic_term_degrades_gracefully(mocker):
    # The other half of the story: data with a real quadratic term, fit with
    # the default vary that holds b at zero. The fit does not fail, it absorbs
    # what it can into the linear terms, and what is left over is a bias. Worth
    # pinning that the failure is a modest one -- a few hundredths of a
    # magnitude, not a wild answer -- and that it lands mostly in the zero
    # point rather than in the calibrated magnitudes.
    b = 0.01

    result, catalog, _ = _fit_a_catalog(mocker, b=b)

    # The zero point soaks up most of the missing term...
    assert np.abs(result["z"][0] - _FAKE_CATALOG_ZERO_POINT) > 0.1
    # ...and a, which is supposed to be zero here, takes up the rest.
    assert np.abs(result["a"][0]) > 0.1
    # The calibrated magnitudes themselves stay close, because the fit is free
    # to trade the terms off against each other over this range of magnitudes.
    assert (
        np.abs(np.asarray(result["mag_cal"]) - np.asarray(catalog["mag_R"])).max() < 0.1
    )


def _one_star_with_a_large_error(mocker, n_stars=20, cat_error=_FAKE_CATALOG_ERROR):
    """
    Fit an image in which one star is wrong and says so with a large error.

    The star's observed magnitude is 0.8 mag from where the catalog puts it,
    which is close enough to survive the median-based outlier cut, so nothing
    but the weighting can keep it from dragging the fit. Every other star is
    measured well and agrees with the catalog.

    Parameters
    ----------

    mocker : `pytest_mock.MockerFixture`
        Fixture used to patch the catalog fetch.

    n_stars : int, optional
        Number of stars to generate.

    cat_error : float or array-like or None, optional
        Error the catalog reports for each of its magnitudes, passed to
        `_generate_fake_catalog`.

    Returns
    -------
    `astropy.table.Table`
        The transformed observations.
    """
    sigma = 1e-4

    catalog, ra, dec, instrumental = _generate_fake_catalog(
        n_stars, cat_error=cat_error
    )

    observed_mags = instrumental.copy()
    observed_mags[0] -= 0.8

    errors = np.full(n_stars, 0.001)
    errors[0] = 10.0

    # The noise is an order of magnitude below the tolerances asserted by the
    # callers, so it changes nothing about what they ask; it is added only
    # because a residual of exactly zero is a state real data never reaches.
    observed = _generate_observed_table(
        ra, dec, observed_mags, noise_sigma=sigma, seed=8811, mag_error=errors
    )

    return _run_transform_to_catalog(mocker, catalog, observed)


def _assert_the_bad_star_was_ignored(result):
    """
    Assert a fit recovered the truth despite the badly measured star in it.

    See `_one_star_with_a_large_error`, which builds the image this is asked
    of. Weighted backwards the bad star takes over and ``z`` comes out near
    18.5 rather than 20.
    """
    np.testing.assert_allclose(result["z"], _FAKE_CATALOG_ZERO_POINT, rtol=0, atol=1e-3)
    np.testing.assert_allclose(result["a"], 0.0, rtol=0, atol=1e-3)
    np.testing.assert_allclose(result["c"], 0.0, rtol=0, atol=1e-3)


def test_transform_to_catalog_weights_by_inverse_error(mocker):
    # Every other fit here is of noiseless data with a single error value, and
    # for a residual that reaches exactly zero the weights make no difference
    # to where the fit lands -- so weighting by the error rather than by one
    # over the error would pass the rest of this file.
    _assert_the_bad_star_was_ignored(_one_star_with_a_large_error(mocker))


def test_transform_to_catalog_weights_by_inverse_error_without_catalog_errors(mocker):
    # The same thing where the catalog supplies no errors of its own, which is
    # the path the Johnson-Cousins bands take (see the log-message test below).
    # The observed errors have to be used on their own there, and used the same
    # way round -- the fallback is not a path that gets to behave differently.
    result = _one_star_with_a_large_error(mocker, cat_error=None)

    _assert_the_bad_star_was_ignored(result)


def test_transform_to_catalog_weights_by_the_catalog_error_too(mocker):
    # What is being fit is the difference between an observed magnitude and a
    # catalog one, so a star the *catalog* barely knows constrains the fit as
    # poorly as one the observation barely knows, and the catalog says which
    # stars those are. This is what replaced the faintest_mag_for_transform
    # cut #676 removed: the poorly known stars are down-weighted rather than a
    # magnitude being drawn across the field. See issue #680.
    n_stars = 20
    sigma = 1e-4

    catalog, ra, dec, instrumental = _generate_fake_catalog(n_stars)

    # One star's observed magnitude is 0.8 mag from where the catalog puts it
    # -- close enough to survive the median-based outlier cut -- and it is the
    # catalog that is unsure about it. The observation is as good as any other.
    observed_mags = instrumental.copy()
    observed_mags[0] -= 0.8

    catalog["mag_error_R"][0] = 10.0

    observed = _generate_observed_table(
        ra, dec, observed_mags, noise_sigma=sigma, seed=8811, mag_error=0.001
    )

    result = _run_transform_to_catalog(mocker, catalog, observed)

    np.testing.assert_allclose(result["z"], _FAKE_CATALOG_ZERO_POINT, rtol=0, atol=1e-3)

    # The other half: with no catalog errors to weight by, nothing marks that
    # star out and it drags the fit. Without this the test above would pass on
    # a version that ignored the catalog errors entirely.
    del catalog["mag_error_R"]

    unweighted = _run_transform_to_catalog(mocker, catalog, observed)

    assert abs(unweighted["z"][0] - _FAKE_CATALOG_ZERO_POINT) > 0.01


def test_transform_to_catalog_combines_the_errors_in_quadrature(mocker):
    # Which combination it is matters, and the test above cannot tell: adding
    # the two errors, or taking the larger of them, would down-weight that
    # star just as well. So run the same data twice, once against a catalog
    # that supplies the errors and once with the test having folded them into
    # the observed errors itself. The two fits must be identical bit for bit,
    # because they are literally the same weights.
    n_stars = 20
    sigma = 0.02

    catalog, ra, dec, instrumental = _generate_fake_catalog(n_stars, a=0.02, c=0.15)

    # Both errors vary from star to star, and in opposite directions, so no
    # single-error rule reproduces the combination by accident.
    cat_error = np.linspace(0.005, 0.05, n_stars)
    obs_error = np.linspace(0.05, 0.005, n_stars)
    catalog["mag_error_R"] = cat_error

    observed = _generate_observed_table(
        ra, dec, instrumental, noise_sigma=sigma, seed=97531, mag_error=obs_error
    )
    combined_by_the_code = _run_transform_to_catalog(mocker, catalog, observed)

    del catalog["mag_error_R"]
    pre_combined = _generate_observed_table(
        ra,
        dec,
        instrumental,
        noise_sigma=sigma,
        seed=97531,
        mag_error=np.hypot(obs_error, cat_error),
    )

    combined_by_the_test = _run_transform_to_catalog(mocker, catalog, pre_combined)

    # Exact equality, not approximate: the fits saw the same magnitudes and
    # the same weights, so anything but the same answer means the weights were
    # built some other way. Only what the weights decide is compared --
    # mag_cal_error is the star's own measurement error propagated, and the
    # observed errors really are different between the two runs.
    for column in ("a", "c", "z", "mag_cal"):
        np.testing.assert_array_equal(
            np.asarray(combined_by_the_code[column]),
            np.asarray(combined_by_the_test[column]),
            err_msg=column,
        )


def test_transform_to_catalog_warns_when_the_catalog_has_no_error_for_the_band(
    mocker, caplog
):
    # The bands the shipped notebook calibrates in, R and I, are exactly the
    # ones neither APASS nor refcat2 supplies errors for: they are transformed
    # from the catalog's native bands by a transform that does not propagate
    # errors yet. The fit still runs, weighted by the observed errors alone,
    # and says which band it could not do better for. Issue #685 is the fix.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20, cat_error=None)
    observed = _combine_observed_tables(
        _generate_observed_table(ra, dec, instrumental, file_name="image_1.fit"),
        _generate_observed_table(ra, dec, instrumental, file_name="image_2.fit"),
    )

    with caplog.at_level(logging.WARNING, logger=_MAGNITUDE_TRANSFORMS_LOGGER):
        result = _run_transform_to_catalog(mocker, catalog, observed)

    about_the_band = [
        r for r in caplog.records if "mag_error_R" in r.message and "#685" in r.message
    ]
    # One message per call, not one per image: the column is missing from the
    # catalog, which is a fact about the call rather than about an image.
    assert len(about_the_band) == 1

    # The result has one row per star per image, so repeat the catalog
    # magnitudes for each image.
    expected_mag = np.tile(catalog["mag_R"], 2)
    np.testing.assert_allclose(result["mag_cal"], expected_mag, rtol=0, atol=1e-6)


def test_transform_to_catalog_negligible_catalog_error_changes_nothing(mocker):
    # The other side of the fallback: adding catalog errors to the weighting
    # must not have moved the answer for anyone whose catalog errors are too
    # small to matter. An extremely small catalog error -- one that is
    # deliberately far below the observed errors -- allows
    # ``np.hypot(err, 1e-30) == err`` to the last bit in float64, which is what
    # makes exact bit-for-bit equality between the quadrature-weighted path and
    # the observed-errors-only path a legitimate expectation. A plausible small
    # catalog error would force tolerance-based comparison, which could not
    # distinguish the two weighting code paths at all.
    n_stars = 20
    sigma = 0.02

    catalog, ra, dec, instrumental = _generate_fake_catalog(
        n_stars, a=0.02, c=0.15, cat_error=1e-30
    )
    observed = _generate_observed_table(
        ra, dec, instrumental, noise_sigma=sigma, seed=13579
    )

    negligible = _run_transform_to_catalog(mocker, catalog, observed, in_place=False)

    del catalog["mag_error_R"]
    absent = _run_transform_to_catalog(mocker, catalog, observed, in_place=False)

    assert negligible is not absent
    for column in _COLUMNS_BLIND_TO_UNUSABLE_CATALOG_ERRORS:
        np.testing.assert_array_equal(
            np.asarray(negligible[column]),
            np.asarray(absent[column]),
            err_msg=column,
        )

    # The single exception, and the reason the loop above is not over every
    # output column: a catalog that reports a negligible error still reported
    # one, and a catalog with no error column for the band reported nothing.
    # The weights cannot tell those apart, which is the point -- this column
    # can, which is also the point. See issue #694.
    assert negligible["fit_cat_error_missing_frac"][0] == 0.0
    assert absent["fit_cat_error_missing_frac"][0] == 1.0


@pytest.mark.parametrize("case", ["masked", "zero", "negative", "nan"])
def test_transform_to_catalog_treats_unusable_catalog_errors_as_observed_only(
    mocker, case
):
    # A catalog error the fit cannot use -- masked, NaN, zero or negative --
    # no longer drops the star from the fit. The catalog simply does not know
    # its own uncertainty for that star, so the star is weighted by its
    # observed error alone, exactly as an APASS star with a single, zero-error
    # observation was before catalog-error weighting existed. See issue #680.
    n_stars = 20
    sigma = 1e-4

    # Every star's catalog magnitude is poorly known, so that a star weighted
    # by its catalog error would be pulled off the true zero point if the
    # substitution below were not happening.
    catalog, ra, dec, instrumental = _generate_fake_catalog(n_stars, cat_error=0.5)

    observed_mags = instrumental.copy()
    observed_mags[0] -= 0.8

    observed = _generate_observed_table(
        ra, dec, observed_mags, noise_sigma=sigma, seed=8811, mag_error=0.01
    )

    unusable_catalog = catalog.copy()
    match case:
        case "masked":
            unusable_catalog["mag_error_R"] = np.ma.masked_array(
                unusable_catalog["mag_error_R"], mask=np.arange(n_stars) == 0
            )
        case "zero":
            unusable_catalog["mag_error_R"][0] = 0.0
        case "negative":
            unusable_catalog["mag_error_R"][0] = -0.02
        case "nan":
            unusable_catalog["mag_error_R"][0] = np.nan
        case _:  # pragma: no cover
            raise ValueError(f"Unknown case {case!r}")

    treated_as_unusable = _run_transform_to_catalog(
        mocker, unusable_catalog, observed, in_place=False
    )

    # The reference run substitutes a catalog error so small that
    # ``np.hypot(obs_err, 1e-30) == obs_err`` to the last bit in float64 --
    # see `test_transform_to_catalog_negligible_catalog_error_changes_nothing`
    # for why that makes exact equality a legitimate expectation here rather
    # than an approximate one.
    reference_catalog = catalog.copy()
    reference_catalog["mag_error_R"][0] = 1e-30
    reference = _run_transform_to_catalog(
        mocker, reference_catalog, observed, in_place=False
    )

    for column in _COLUMNS_BLIND_TO_UNUSABLE_CATALOG_ERRORS:
        np.testing.assert_array_equal(
            np.asarray(treated_as_unusable[column]),
            np.asarray(reference[column]),
            err_msg=column,
        )

    # Everything above is identical because the weights are identical. What
    # differs is the count of stars the catalog knew no uncertainty for -- one
    # here, none in the reference run -- which is the diagnostic issue #694
    # added precisely because the weights cannot show it.
    assert treated_as_unusable["fit_cat_error_missing_frac"][0] == pytest.approx(
        1 / n_stars
    )
    assert reference["fit_cat_error_missing_frac"][0] == 0.0


def test_transform_to_catalog_reports_coefficient_uncertainties(_noisy_fit_result):
    # How well the fit pinned each term down is what separates a transform
    # worth applying from one fit to stars that could barely tell its terms
    # apart, and until now the fit worked it out and threw it away. See issue
    # #677.
    result = _noisy_fit_result

    for term in ("a", "c", "z"):
        reported = np.asarray(result[f"{term}_error"])
        assert np.isfinite(reported).all(), term
        assert (reported > 0).all(), term
        # One fit per image, so every row of the image carries the same value.
        assert (reported == reported[0]).all(), term

    # A term that is not fit is held at exactly zero and is therefore known
    # exactly. Its uncertainty is exactly zero -- not NaN, and not a small
    # number, which is the same exact comparison
    # test_transform_to_catalog_fixes_unvaried_terms makes of the value.
    assert (np.asarray(result["b_error"]) == 0.0).all()
    assert (np.asarray(result["d_error"]) == 0.0).all()

    # The numbers are the right size, not merely positive: each term really is
    # within a few of its reported uncertainties of the value the catalog was
    # built from.
    for term, truth in (("a", 0.02), ("c", 0.15), ("z", _FAKE_CATALOG_ZERO_POINT)):
        assert abs(result[term][0] - truth) < 3 * result[f"{term}_error"][0], term


def test_transform_to_catalog_uncertainty_falls_as_stars_are_added(mocker):
    # The test that makes the numbers above uncertainties rather than arbitrary
    # ones: ten times as many stars, each measured just as well, pin the zero
    # point down about sqrt(10) times better. Nothing that was not a real
    # uncertainty would scale that way. See issue #677.
    sigma = 0.02

    # One seed is enough: the reported uncertainty comes from the quoted
    # errors and the design of the fit (see the issue #690 tests below), so
    # the noise realization barely moves it -- it only jitters the design
    # matrix's magnitude column.
    seed = 54321
    z_error = {}

    for n_stars in (25, 250):
        catalog, ra, dec, instrumental = _generate_fake_catalog(n_stars)
        z_error[n_stars] = _run_transform_to_catalog(
            mocker,
            catalog,
            _generate_observed_table(
                ra, dec, instrumental, noise_sigma=sigma, seed=seed
            ),
        )["z_error"][0]

    # Roughly sqrt(10), not exactly: the 25-star fit is not an exactly scaled
    # copy of the 250-star one, since its stars sample the same magnitude and
    # color ranges more coarsely. The measured ratio is 3.58 against a
    # sqrt(10) of 3.16, so the tolerance is set to leave real headroom on
    # both sides -- what matters is that the ratio is nowhere near 1, which
    # is what a number that only looked like an uncertainty would give.
    assert z_error[25] / z_error[250] == pytest.approx(np.sqrt(10), rel=0.25)


# What the coefficient uncertainties are measured against, all issue #690.
# The covariance is reported as the fit was weighted -- lmfit's default
# rescaling of it by the reduced chi-square, which retells the observed
# scatter as grown coefficient errors, is turned off -- so the ``*_error``
# columns believe the quoted errors and ``fit_redchi`` is an independent
# alarm, the same convention issue #694 chose for the diagnostics.

# Inputs of the fit `_underquoted_fit_result` shares between the tests that
# read it: errors quoted five times smaller than the true scatter.
_UNDERQUOTED_CLAIMED_ERROR = 0.02
_UNDERQUOTED_NOISE_SIGMA = 0.1


@pytest.fixture(scope="module")
def _underquoted_fit_result(module_mocker):
    """
    The one under-quoted-errors fit shared by tests of what its errors mean.

    `test_transform_to_catalog_uncertainties_believe_the_quoted_errors` and
    `test_transform_to_catalog_error_uses_the_unscaled_covariance` both read
    -- and neither mutates -- the result of fitting the same catalog: 50
    stars whose true scatter, `_UNDERQUOTED_NOISE_SIGMA`, is five times the
    `_UNDERQUOTED_CLAIMED_ERROR` they quote, seed 20260813, a=0.02 and
    c=0.15. Computing it once here rather than twice keeps the tests from
    drifting apart on inputs they mean to share.
    """
    result, _, _ = _fit_a_catalog(
        module_mocker,
        n_stars=50,
        sigma=_UNDERQUOTED_NOISE_SIGMA,
        seed=20260813,
        mag_error=_UNDERQUOTED_CLAIMED_ERROR,
        a=0.02,
        c=0.15,
    )
    # Same reasoning as `_noisy_fit_result` above: the patch must not
    # outlive the fit it was made for.
    module_mocker.stopall()
    return result


def _design_matrix(result):
    """
    Design matrix of the default terms, rebuilt from a transform's output.

    One row per star and one column per varied term, in the order ``a``,
    ``c``, ``z`` -- the order every prediction and assertion below indexes
    by, which is why this is written down exactly once.

    Parameters
    ----------

    result : `astropy.table.Table`
        A transform of a single image in which every star was fit.

    Returns
    -------
    `numpy.ndarray`
        The design matrix.
    """
    mag_inst = np.asarray(result["mag_inst"])
    color = np.asarray(result["color_cat"])

    return np.column_stack([mag_inst, color, np.ones_like(mag_inst)])


def _predicted_coefficient_covariance(result, sigma_quoted):
    """
    The covariance a fit of the default terms should report, by hand.

    The transform model is linear in its coefficients, so the covariance of
    a weighted least-squares fit of it is exactly ``(J^T W J)^-1`` with
    ``J`` the design matrix and ``W`` the diagonal of the squared weights,
    floored exactly as the fit floors them. Nothing about where the stars
    actually landed enters, which is the behavior under test: the fit's own
    residuals must not rescale it.

    Parameters
    ----------

    result : `astropy.table.Table`
        A transform of a single image in which every star was fit, from
        which the design matrix is rebuilt.

    sigma_quoted : float
        The one uncertainty every star quoted, observed and catalog errors
        already combined.

    Returns
    -------
    `numpy.ndarray`
        Predicted covariance of the default varied terms, in
        `_design_matrix` order.
    """
    jacobian = _design_matrix(result)
    weight = 1.0 / max(sigma_quoted, _MIN_FIT_SIGMA)

    return np.linalg.inv(weight**2 * (jacobian.T @ jacobian))


def _predicted_mag_cal_error(result, covariance, mag_error):
    """
    The ``mag_cal_error`` every star should report, by hand.

    The measurement half is the star's own quoted error through the
    ``fit_diff`` sensitivity of ``1 + a``; the transform half is the given
    coefficient covariance pushed through the model's gradient --
    which, the model being linear, is the star's design-matrix row again --
    correlations included.

    Parameters
    ----------

    result : `astropy.table.Table`
        A transform of a single image in which every star was fit.

    covariance : `numpy.ndarray`
        Coefficient covariance to propagate, in `_design_matrix` order.

    mag_error : float
        The one observed error every star quoted, *not* combined with the
        catalog's: the catalog error weights the fit but is not part of any
        star's own measurement.

    Returns
    -------
    `numpy.ndarray`
        Predicted ``mag_cal_error`` of each star.
    """
    gradients = _design_matrix(result)
    transform_var = np.einsum("si,ij,sj->s", gradients, covariance, gradients)
    measurement_var = ((1 + result["a"][0]) * mag_error) ** 2

    return np.sqrt(measurement_var + transform_var)


def test_transform_to_catalog_uncertainties_believe_the_quoted_errors(
    _underquoted_fit_result,
):
    # The choice issue #690 is about. These stars scatter five times as far
    # from the model as their errors claim, and the reported coefficient
    # uncertainties must stay at what the quoted errors predict rather than
    # growing ~sqrt(fit_redchi) to match the observed scatter, which is what
    # lmfit's scale_covar default silently did. The scatter is still
    # reported, once, in fit_redchi -- under the old default the grown
    # errors and that alarm were one observation dressed as two.
    result = _underquoted_fit_result

    sigma_quoted = np.hypot(_UNDERQUOTED_CLAIMED_ERROR, _FAKE_CATALOG_ERROR)
    covariance = _predicted_coefficient_covariance(result, sigma_quoted)

    for index, term in enumerate(("a", "c", "z")):
        assert result[f"{term}_error"][0] == pytest.approx(
            np.sqrt(covariance[index, index]), rel=1e-6
        ), term

    # The alarm survives, and is now the only place the observed scatter
    # shows up.
    assert result["fit_redchi"][0] == pytest.approx(
        (_UNDERQUOTED_NOISE_SIGMA / sigma_quoted) ** 2, rel=0.3
    )


def test_transform_to_catalog_error_uses_the_unscaled_covariance(
    _underquoted_fit_result,
):
    # mag_cal_error's transform half must come from the same unscaled
    # covariance the coefficient columns report, so that everything in one
    # row shares one convention: the quoted errors, believed. The prediction
    # is the star's own quoted error through the fit_diff sensitivity of
    # 1 + a, plus the hand-computed covariance pushed through the model's
    # gradient, correlations included -- and nothing from the observed
    # scatter, which is the part scale_covar used to fold in. See issue
    # #690.
    result = _underquoted_fit_result

    sigma_quoted = np.hypot(_UNDERQUOTED_CLAIMED_ERROR, _FAKE_CATALOG_ERROR)
    covariance = _predicted_coefficient_covariance(result, sigma_quoted)

    np.testing.assert_allclose(
        np.asarray(result["mag_cal_error"]),
        _predicted_mag_cal_error(result, covariance, _UNDERQUOTED_CLAIMED_ERROR),
        rtol=1e-6,
    )


def test_transform_to_catalog_uncertainties_believe_the_floor_when_it_binds(mocker):
    # The other side of believing the errors as the fit was *weighted*: when
    # every star quotes an error under the min_fit_sigma floor, the weights
    # -- and so the covariance the reported uncertainties come from -- are
    # the floor's, not the quoted errors'. Truthful sub-floor errors
    # therefore come out overstated, by about floor/quoted, which is the
    # price of bounding any one star's leverage; ``min_fit_sigma=0`` is the
    # documented escape. Pinned here so the trade-off stays a choice rather
    # than becoming an accident. Under the old scale_covar default the floor
    # cancelled out of the rescaled covariance and could not reach the
    # reported errors at all. See issue #690.
    claimed = 0.002

    result, _, _ = _fit_a_catalog(
        mocker,
        n_stars=40,
        sigma=claimed,
        seed=20260814,
        mag_error=claimed,
        cat_error=None,
        a=0.02,
        c=0.15,
    )

    # The helper floors its weights exactly as the fit does, so with a
    # sub-floor sigma this prediction is the floor's covariance -- about
    # (floor / claimed) = 5 times the quoted errors' prediction.
    covariance = _predicted_coefficient_covariance(result, claimed)

    for index, term in enumerate(("a", "c", "z")):
        assert result[f"{term}_error"][0] == pytest.approx(
            np.sqrt(covariance[index, index]), rel=1e-6
        ), term

    # mag_cal_error's two halves split at the floor: the transform half
    # comes from the floored covariance just pinned, while the measurement
    # half is the star's own error exactly as quoted -- never raised. This
    # is the only place that split is pinned in absolute terms below the
    # floor; the twin-fit tests above cancel the transform half by
    # construction and could not see it drift.
    np.testing.assert_allclose(
        np.asarray(result["mag_cal_error"]),
        _predicted_mag_cal_error(result, covariance, claimed),
        rtol=1e-6,
    )


def test_transform_to_catalog_uncertainties_scale_with_the_quoted_errors(mocker):
    # Under the old scale_covar default this was structurally impossible:
    # rescaling every weight by k scales the raw covariance by k**2 and the
    # internal reduced chi-square by 1/k**2, so the reported uncertainties
    # could not move whatever the quoted errors said -- which would make
    # error plumbing like issues #680 and #692 invisible in every reported
    # error. Believing the quoted errors means tripling them must triple
    # each one. See issue #690.
    base_error = 0.02
    scale = 3.0

    # cat_error=None so the quoted observed error is the whole sigma and
    # scaling it scales the weights exactly; same seed, so the two fits see
    # the very same stars.
    quoted, tripled = (
        _fit_a_catalog(
            mocker,
            n_stars=30,
            sigma=0.02,
            seed=24680,
            mag_error=k * base_error,
            a=0.02,
            c=0.15,
            cat_error=None,
        )[0]
        for k in (1.0, scale)
    )

    # mag_cal_error is in the list because both of its halves quote: the
    # measurement half is the tripled error itself and the transform half
    # comes from the tripled-error covariance.
    for column in ("a_error", "c_error", "z_error", "mag_cal_error"):
        np.testing.assert_allclose(
            np.asarray(tripled[column]),
            scale * np.asarray(quoted[column]),
            rtol=1e-7,
            err_msg=column,
        )


def test_transform_to_catalog_unweighted_uncertainties_use_the_scatter(
    _unweighted_fit_result,
):
    # An unweighted fit quotes no errors to believe, so the observed scatter
    # is the only scale its covariance can take: scale_covar stays on for
    # exactly this case, and the reported errors are the unweighted
    # (J^T J)^-1 times the residual variance per degree of freedom -- which
    # is what fit_redchi holds for an unweighted fit. See issue #690.
    result = _unweighted_fit_result

    jacobian = _design_matrix(result)
    covariance = np.linalg.inv(jacobian.T @ jacobian) * result["fit_redchi"][0]

    for index, term in enumerate(("a", "c", "z")):
        assert result[f"{term}_error"][0] == pytest.approx(
            np.sqrt(covariance[index, index]), rel=1e-6
        ), term


@pytest.mark.parametrize(
    "error_scale, expected_redchi",
    [
        # Errors that describe the data: the model is as far from the stars as
        # they claim to be uncertain, which is what a reduced chi-square of one
        # means.
        (1.0, 1.0),
        # Errors ten times too small. Chi-square goes as the square of that, so
        # a hundred -- and that factor of a hundred is the whole reason to
        # report the number.
        (0.1, 100.0),
    ],
)
def test_transform_to_catalog_reports_reduced_chi_square(
    mocker, error_scale, expected_redchi
):
    # See issue #677.
    n_stars = 100
    sigma = 0.2

    catalog, ra, dec, instrumental = _generate_fake_catalog(n_stars)
    observed = _generate_observed_table(
        ra,
        dec,
        instrumental,
        noise_sigma=sigma,
        seed=13579,
        mag_error=error_scale * sigma,
    )

    # The floor is pinned off so that what is measured here is the reduced
    # chi-square itself, with no help from the weighting the floor changes.
    result = _run_transform_to_catalog(mocker, catalog, observed, min_fit_sigma=0)

    reported = np.asarray(result["fit_redchi"])
    # One fit per image, so this too is repeated down every row.
    assert (reported == reported[0]).all()
    assert reported[0] == pytest.approx(expected_redchi, rel=0.3)


def test_transform_to_catalog_reports_unweighted_fit_statistic(
    _unweighted_fit_result,
):
    # Without an error column nothing divides the residuals, so the same
    # column holds the summed squared residuals per degree of freedom in mag
    # squared -- for Gaussian noise, the noise variance. The weighted values
    # above sit near one; this sits near 4e-4, which is the documented "same
    # column, completely different scale" behavior.
    sigma = 0.02  # must match the noise sigma _unweighted_fit_result fit with

    result = _unweighted_fit_result

    reported = np.asarray(result["fit_redchi"])
    assert (reported == reported[0]).all()
    assert reported[0] == pytest.approx(sigma**2, rel=0.3)


# The sigma floor and the three fit diagnostics, all of them issue #694:
# one star claiming a tiny uncertainty quietly held most of a fit's weight.
# Flooring the sigma bounds how much any one star can be worth; the
# diagnostics say when it happened.


@pytest.mark.parametrize("cat_error", [None, 0.0], ids=["observed", "combined"])
def test_transform_to_catalog_floors_the_fit_sigma(mocker, cat_error):
    # The floor is what makes every sigma below it weigh the same, so the
    # direct test is that replacing the sub-floor sigmas with the floor itself
    # changes nothing at all. Both weighted paths are checked: a catalog with
    # no error column at all (``None``), which is the path the
    # Johnson-Cousins bands take, and one whose errors are all the unusable
    # zero APASS reports (``0.0``), which is the case issue #694 is about.
    # ``hypot(x, 0.0)`` is exactly ``x``, so the two paths see the same
    # sigmas and the same expectation holds for both.
    n_stars = 20

    # Half the stars claim an error below the floor, spanning two orders of
    # magnitude so that no single substituted value could reproduce them, and
    # half claim one well above it so the weights are not uniform -- a
    # uniformly weighted fit lands in the same place whatever the weights are
    # scaled by, and would pass this test with no floor at all.
    errors = np.full(n_stars, 0.05)
    errors[:10] = np.geomspace(0.0001, 0.005, 10)

    at_the_floor = errors.copy()
    at_the_floor[:10] = _MIN_FIT_SIGMA

    fit_kwargs = dict(
        mocker=mocker, n_stars=n_stars, sigma=0.02, seed=24680, cat_error=cat_error
    )
    below, _, _ = _fit_a_catalog(mag_error=errors, **fit_kwargs)
    floored, _, _ = _fit_a_catalog(mag_error=at_the_floor, **fit_kwargs)

    # Exact equality, not approximate: the two fits saw the same magnitudes
    # and, once floored, the same weights, so anything but the same answer
    # means a sub-floor sigma reached the fit. ``mag_cal_error`` and
    # ``fit_redchi`` are left out because both are reported against the
    # star's own quoted errors, which really do differ between the two runs.
    for column in ("a", "c", "z", "mag_cal", "fit_max_weight_share"):
        np.testing.assert_array_equal(
            np.asarray(below[column]),
            np.asarray(floored[column]),
            err_msg=column,
        )


def test_transform_to_catalog_floor_preserves_the_redchi_alarm(mocker):
    # The floor bounds a star's leverage inside the fit, but the reported
    # fit_redchi is measured against the errors as quoted, so an image whose
    # quoted errors are a hundred times too small still reports a redchi near
    # 1e4 -- not one capped at the scatter in units of the floor.
    sigma = 0.02
    claimed = 0.0002

    result, _, _ = _fit_a_catalog(
        mocker, n_stars=100, sigma=sigma, seed=13579, mag_error=claimed, cat_error=None
    )

    assert result["fit_redchi"][0] == pytest.approx((sigma / claimed) ** 2, rel=0.3)


def test_transform_to_catalog_floor_preserves_the_excess_scatter_alarm(mocker):
    # Quoted errors and true scatter both below the floor. Measured against
    # the floored sigmas the redchi would come out below one and the excess
    # would read exactly zero -- an affirmative all-clear for precisely the
    # under-quoted-errors case this diagnostic exists to catch. Measured
    # against the errors as quoted, the alarm survives.
    sigma = 0.008
    claimed = 0.002

    result, _, _ = _fit_a_catalog(
        mocker, n_stars=300, sigma=sigma, seed=97531, mag_error=claimed, cat_error=None
    )

    assert result["fit_excess_scatter"][0] == pytest.approx(
        np.sqrt(sigma**2 - claimed**2), rel=0.15
    )


def test_transform_to_catalog_respects_a_custom_min_fit_sigma(mocker):
    # The same replacing-sub-floor-sigmas-changes-nothing expectation as the
    # default-floor test above, at a floor the caller chose. fit_redchi is
    # left out of the columns compared: it is measured against the errors as
    # quoted, which really do differ between the two runs.
    n_stars = 20
    floor = 0.05

    errors = np.full(n_stars, 0.2)
    errors[:10] = np.geomspace(0.001, 0.04, 10)

    at_the_floor = errors.copy()
    at_the_floor[:10] = floor

    fit_kwargs = dict(
        mocker=mocker,
        n_stars=n_stars,
        sigma=0.02,
        seed=24680,
        cat_error=None,
        min_fit_sigma=floor,
    )
    below, _, _ = _fit_a_catalog(mag_error=errors, **fit_kwargs)
    floored, _, _ = _fit_a_catalog(mag_error=at_the_floor, **fit_kwargs)

    for column in ("a", "c", "z", "mag_cal", "fit_max_weight_share"):
        np.testing.assert_array_equal(
            np.asarray(below[column]),
            np.asarray(floored[column]),
            err_msg=column,
        )


def test_transform_to_catalog_min_fit_sigma_zero_disables_the_floor(mocker):
    # Zero means no floor: a star whose quoted error is far below the default
    # floor keeps the full weight its error claims, which the weight-share
    # column reports directly.
    n_stars = 20

    errors = np.full(n_stars, 0.05)
    errors[0] = 0.002

    result, _, _ = _fit_a_catalog(
        mocker,
        n_stars=n_stars,
        sigma=0.02,
        seed=2468,
        mag_error=errors,
        cat_error=None,
        min_fit_sigma=0,
    )

    statistical = 1.0 / errors**2
    expected = statistical.max() / statistical.sum()
    assert result["fit_max_weight_share"][0] == pytest.approx(expected)


def test_transform_to_catalog_rejects_a_negative_min_fit_sigma(mocker):
    with pytest.raises(ValueError, match="min_fit_sigma"):
        _fit_a_catalog(mocker, min_fit_sigma=-0.01)


def test_excess_scatter_returns_zero_when_redchi_is_barely_above_one():
    # lmfit's redchi and the bracket function recompute the same sum in
    # different orders, so around 1.0 they can disagree by a few ULPs. A gate
    # on lmfit's value could pass while both bracket endpoints evaluate
    # negative, which is a ValueError from brentq for a healthy image. The
    # gate is therefore the bracket function itself, evaluated at zero.
    weighted_residual = np.full(4, np.nextafter(1.0, 0.0))
    sigma = np.full(4, 2.0)

    fit_result = SimpleNamespace(
        residual=weighted_residual,
        nfree=4,
        redchi=np.nextafter(1.0, 2.0),
    )

    assert _excess_scatter(fit_result, sigma, 1.0 / sigma) == 0.0


@pytest.mark.parametrize("n_zero", [0, 5, 20])
def test_transform_to_catalog_reports_fraction_with_no_catalog_error(mocker, n_zero):
    # The diagnostic that says when two bands' fit_redchi values are not
    # comparable because one band's catalog barely knew its own errors; see
    # issue #694.
    n_stars = 20

    cat_error = np.full(n_stars, 0.03)
    cat_error[:n_zero] = 0.0

    result, _, _ = _fit_a_catalog(
        mocker,
        n_stars=n_stars,
        sigma=0.02,
        seed=1234,
        mag_error=0.02,
        cat_error=cat_error,
    )

    reported = np.asarray(result["fit_cat_error_missing_frac"])
    # One fit per image, so this too is repeated down every row.
    assert (reported == reported[0]).all()
    assert reported[0] == pytest.approx(n_zero / n_stars)


def test_transform_to_catalog_reports_every_catalog_error_missing_without_the_column(
    mocker,
):
    # A catalog with no error column for the band knows nothing about any of
    # its stars' uncertainties, which is the same statement as a column of
    # zeros and is reported the same way rather than as "not applicable".
    result, _, _ = _fit_a_catalog(mocker, n_stars=20, cat_error=None)

    assert result["fit_cat_error_missing_frac"][0] == 1.0


def test_transform_to_catalog_reports_max_single_star_weight_share(mocker):
    # The number that catches a fit one star is quietly running: one star
    # held ~88% of a fit's weight and nothing in the output said so; see
    # issue #694.
    n_stars = 20

    errors = np.full(n_stars, 0.05)
    # Exactly at the floor, so the floor leaves it alone and the expectation
    # below is a statement about the weight share rather than about the floor.
    errors[0] = _MIN_FIT_SIGMA

    result, _, _ = _fit_a_catalog(
        mocker,
        n_stars=n_stars,
        sigma=0.02,
        seed=2468,
        mag_error=errors,
        cat_error=None,
    )

    # Statistical weight, i.e. one over the variance, which is what a
    # least-squares fit actually apportions among the stars -- roughly 0.57
    # here, so a share computed as ``1 / sigma`` rather than its square
    # cannot pass by accident.
    statistical = 1.0 / errors**2
    expected = statistical.max() / statistical.sum()

    reported = np.asarray(result["fit_max_weight_share"])
    assert (reported == reported[0]).all()
    assert reported[0] == pytest.approx(expected)


def test_transform_to_catalog_reports_excess_scatter(mocker):
    # The third diagnostic: how much scatter, added in quadrature to every
    # star's sigma, the fit would need before its residuals matched the errors
    # it was given. Reported rather than folded into the weights, because a
    # fit that absorbs it has a redchi of one by construction and stops being
    # able to report that anything is wrong.
    n_stars = 200
    sigma = 0.05
    claimed = 0.02

    # Both values are above the floor, so what is measured is the excess
    # rather than the flooring.
    result, _, _ = _fit_a_catalog(
        mocker,
        n_stars=n_stars,
        sigma=sigma,
        seed=31415,
        mag_error=claimed,
        cat_error=None,
    )

    reported = np.asarray(result["fit_excess_scatter"])
    assert (reported == reported[0]).all()
    assert reported[0] == pytest.approx(np.sqrt(sigma**2 - claimed**2), rel=0.15)


def test_transform_to_catalog_reports_no_excess_scatter_when_errors_describe_the_data(
    mocker,
):
    # Errors that are, if anything, generous: there is no excess to infer and
    # the column says exactly zero rather than a small positive number that
    # would read as a real finding.
    result, _, _ = _fit_a_catalog(
        mocker, n_stars=200, sigma=0.02, seed=31415, mag_error=0.05, cat_error=None
    )

    assert result["fit_excess_scatter"][0] == 0.0


def test_transform_to_catalog_diagnostics_for_an_unweighted_fit(
    _unweighted_fit_result,
):
    # Without an error column there are no sigmas, so there is no scatter to
    # call excessive and the column says so with NaN rather than zero, which
    # would claim the errors were checked and found adequate. The weight share
    # is still meaningful: every star counts the same, so each holds 1/N.
    n_stars = 100  # must match the count _unweighted_fit_result fit

    result = _unweighted_fit_result

    assert np.isnan(result["fit_excess_scatter"][0])
    assert result["fit_max_weight_share"][0] == pytest.approx(1.0 / n_stars)
    assert result["fit_cat_error_missing_frac"][0] == 1.0


def test_transform_to_catalog_fits_each_image_separately(mocker):
    # Coefficients are fit per group, so two images of the same stars with
    # different zero points should each get their own.
    n_stars = 20
    zero_point_offset = 1.0

    catalog, ra, dec, instrumental = _generate_fake_catalog(n_stars)

    # The second image is fainter by a constant, so its zero point is larger by
    # that constant while the other coefficients stay at zero.
    observed = _combine_observed_tables(
        _generate_observed_table(ra, dec, instrumental, file_name="image_1.fit"),
        _generate_observed_table(
            ra,
            dec,
            instrumental - zero_point_offset,
            file_name="image_2.fit",
        ),
    )

    result = _run_transform_to_catalog(mocker, catalog, observed)

    first = result["file"] == "image_1.fit"
    second = result["file"] == "image_2.fit"

    np.testing.assert_allclose(
        result["z"][first], _FAKE_CATALOG_ZERO_POINT, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        result["z"][second],
        _FAKE_CATALOG_ZERO_POINT + zero_point_offset,
        rtol=0,
        atol=1e-6,
    )
    # Both images are an exact fit, so both recover the catalog magnitudes.
    np.testing.assert_allclose(
        result["mag_cal"][first], catalog["mag_R"], rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        result["mag_cal"][second], catalog["mag_R"], rtol=0, atol=1e-6
    )


def test_transform_to_catalog_handles_regrouped_row_order(mocker):
    # Results are written back by row index, and grouping a table sorts it, so
    # the row a result belongs to is not the row it came in on. Every other
    # multi-image test here happens to add its images in an order that is
    # already sorted, which makes the sort a no-op and hides any assumption
    # that insertion order survives. Here the image added first sorts last,
    # and the two images have different numbers of stars.
    n_stars = 20
    n_in_short_image = 12
    zero_point_offset = 1.0

    catalog, ra, dec, instrumental = _generate_fake_catalog(n_stars)

    observed = _combine_observed_tables(
        # Added first, sorts second.
        _generate_observed_table(ra, dec, instrumental, file_name="z_image.fit"),
        # Added second, sorts first, and has fewer stars.
        _generate_observed_table(
            ra[:n_in_short_image],
            dec[:n_in_short_image],
            instrumental[:n_in_short_image] - zero_point_offset,
            file_name="a_image.fit",
        ),
    )

    result = _run_transform_to_catalog(mocker, catalog, observed)

    from_short_image = result["file"] == "a_image.fit"

    # Each image gets its own zero point, and neither picks up the other's.
    np.testing.assert_allclose(
        result["z"][from_short_image],
        _FAKE_CATALOG_ZERO_POINT + zero_point_offset,
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result["z"][~from_short_image], _FAKE_CATALOG_ZERO_POINT, rtol=0, atol=1e-6
    )

    # Star by star, not just image by image: every row must have been matched
    # to, and calibrated against, its own catalog entry.
    catalog_mags = np.asarray(catalog["mag_R"])
    for selection, truth in (
        (from_short_image, catalog_mags[:n_in_short_image]),
        (~from_short_image, catalog_mags),
    ):
        np.testing.assert_allclose(
            result["mag_cal"][selection], truth, rtol=0, atol=1e-6
        )
        np.testing.assert_allclose(
            result["mag_cat"][selection], truth, rtol=0, atol=1e-6
        )


def test_transform_to_catalog_output_columns(mocker):
    # The set of columns this adds is its interface with everything downstream
    # -- the AAVSO writer, the shipped notebook's plots -- so adding or
    # renaming one should be a deliberate change to this list, not a surprise.
    result, _, _ = _fit_a_catalog(mocker)

    added = set(result.colnames) - {
        "file",
        "passband",
        "ra",
        "dec",
        "mag_inst",
        "mag_error",
    }

    assert added == _TRANSFORM_OUTPUT_COLUMNS


def test_transform_to_catalog_in_place_mutates_input(mocker):
    # in_place=True is how the shipped notebook accumulates one passband at a
    # time, so it has to hand back the very table it was given rather than a
    # copy that happens to have the same contents.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)

    result = _run_transform_to_catalog(mocker, catalog, observed, in_place=True)

    assert "mag_cal" in observed.colnames
    assert result is observed


def test_transform_to_catalog_not_in_place_leaves_input_alone(mocker):
    # The other half of the contract. Checking the column names is not enough:
    # a shallow copy that shared its underlying arrays would add its columns
    # only to the copy while writing through to the original's values, so the
    # values are snapshotted and compared too.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)

    original_columns = set(observed.colnames)
    original_values = {
        name: np.array(observed[name]) for name in ("mag_inst", "mag_error")
    }

    result = _run_transform_to_catalog(mocker, catalog, observed, in_place=False)

    assert set(observed.colnames) == original_columns
    assert "mag_cal" in result.colnames

    for name, values in original_values.items():
        np.testing.assert_array_equal(observed[name], values)


def _measurement_half_of_the_error(mocker, **fit_kwargs):
    """
    The measurement half of ``mag_cal_error``, isolated by a twin fit.

    `_fit_a_catalog` is run twice on the same noiseless catalog: once with
    every star quoting `_MIN_FIT_SIGMA` and once with every star quoting a
    negligible 1e-12. The transform half of the error cannot simply be
    assumed away -- the covariance it comes from believes the quoted errors
    rather than the residuals (issue #690), so it does not vanish even for a
    perfect fit -- but it can be subtracted out: both runs' quoted sigmas
    sit at or under the ``min_fit_sigma`` floor, and ``cat_error=None``
    keeps the catalog from lifting either off it, so the two fits share
    their weights and with them their transform term exactly. What survives
    the difference in quadrature is the measurement half of the first run's
    error, for stars that quoted `_MIN_FIT_SIGMA`.

    Parameters
    ----------

    mocker : `pytest_mock.MockerFixture`
        Fixture used to patch the catalog fetch.

    **fit_kwargs
        Passed on to `_fit_a_catalog`, e.g. the coefficients the catalog is
        built from and the terms to vary.

    Returns
    -------
    measurement : `numpy.ndarray`
        The measurement half of each star's reported ``mag_cal_error``.

    instrumental : `numpy.ndarray`
        Instrumental magnitude of each star, from `_fit_a_catalog`.
    """
    result, _, instrumental = _fit_a_catalog(
        mocker=mocker, mag_error=_MIN_FIT_SIGMA, cat_error=None, **fit_kwargs
    )
    negligible, _, _ = _fit_a_catalog(
        mocker=mocker, mag_error=1e-12, cat_error=None, **fit_kwargs
    )

    measurement = np.sqrt(
        np.asarray(result["mag_cal_error"]) ** 2
        - np.asarray(negligible["mag_cal_error"]) ** 2
    )

    return measurement, instrumental


def test_transform_to_catalog_error_scales_the_input_error_by_the_fit(mocker):
    # The calibrated error has two parts: the star's own measurement error,
    # scaled by how much the calibrated magnitude moves when the instrumental
    # one does -- 1 + a under the default terms, the instrumental magnitude
    # appearing both inside the model and in the fit_diff add-back -- and the
    # uncertainty of the transform the star was calibrated through, which the
    # twin-fit helper above subtracts away. Pinning the measurement half
    # exactly is what makes the noisy tests below interpretable: anything
    # they see above (1 + a) * mag_error there is the transform term and not
    # an artifact of the fixtures. See issue #674.
    a = 0.02

    measurement, _ = _measurement_half_of_the_error(mocker, a=a)

    np.testing.assert_allclose(measurement, (1 + a) * _MIN_FIT_SIGMA, rtol=0, atol=1e-8)


def test_transform_to_catalog_error_includes_the_transform_uncertainty(
    _noisy_fit_result,
):
    # The failure issue #674 exists to fix: the calibrated error used to be
    # the star's own measurement error and nothing else, as though the
    # coefficients it was calibrated through were known exactly. They are not,
    # and on data with realistic scatter the difference is not small.
    sigma = 0.02  # must match the sigma _noisy_fit_result fit with

    result = _noisy_fit_result
    n_stars = len(result)

    reported = np.asarray(result["mag_cal_error"])

    # What used to be reported, and what is now a floor rather than an answer.
    measurement_only = (1 + result["a"][0]) * sigma
    assert (reported > measurement_only).all()

    # The half that proves the transform term is worked out per star rather
    # than added on as a constant: a fit predicts best at the centroid of the
    # stars it was fit to and worst at the ends of their range. The rows are
    # in order of instrumental magnitude, because _generate_fake_catalog
    # builds them from a linspace, so the ends of the range are the ends of
    # the table.
    excess = np.sqrt(reported**2 - measurement_only**2)
    fifth = n_stars // 5
    ends = np.concatenate([excess[:fifth], excess[-fifth:]]).mean()
    middle = excess[2 * fifth : 3 * fifth].mean()

    # Measured at about 1.35, and set by the geometry of the fit rather than
    # by the noise -- it comes out the same to three figures for every seed
    # tried, because the shape of the covariance comes from the design matrix
    # and the weights, which the noise never enters.
    assert ends > 1.2 * middle


def test_transform_to_catalog_error_uses_the_whole_covariance(_noisy_fit_result):
    # The terms of the transform are strongly correlated with each other --
    # over a range of instrumental magnitudes that never crosses zero, the
    # zero point and the scale term trade off almost exactly -- so a
    # propagation that used only the individual uncertainties, the diagonal of
    # the covariance matrix, would be badly wrong rather than slightly wrong.
    # This is also the test that would notice the propagation silently
    # degrading to that uncorrelated answer, which is one of the ways it can
    # fail without raising. See issue #674.
    sigma = 0.02  # must match the sigma _noisy_fit_result fit with

    result = _noisy_fit_result

    reported = np.asarray(result["mag_cal_error"])

    # The same propagation with every correlation thrown away, built from the
    # columns the table already reports.
    mag_inst = np.asarray(result["mag_inst"])
    color = np.asarray(result["color_cat"])
    diagonal_only = np.sqrt(
        ((1 + result["a"][0]) * sigma) ** 2
        + (mag_inst * result["a_error"][0]) ** 2
        + (color * result["c_error"][0]) ** 2
        + result["z_error"][0] ** 2
    )

    # The correlations are large and they reduce the answer: measured, the
    # diagonal-only version is 1.4 to 1.8 times too big. The lower bound is
    # what stops "ignores the covariance entirely" passing this from below.
    assert ((1 + result["a"][0]) * sigma < reported).all()
    assert (reported < 0.8 * diagonal_only).all()


def test_transform_to_catalog_error_matches_a_monte_carlo(mocker):
    # The test that makes the reported number an uncertainty rather than an
    # arbitrary one: fix a catalog, observe it several hundred times with
    # fresh noise, and the spread of one star's calibrated magnitudes should
    # be the uncertainty reported for it. See issue #674.
    # Few enough stars that the transform is not pinned down all that well,
    # so its share of the calibrated error is a fifth of the total rather than
    # a few percent of it. With a large field the two answers -- with and
    # without the transform term -- are too close for 300 trials to tell
    # apart, and this test would pass whether or not it had been fixed.
    n_stars = 12
    sigma = 0.02
    n_trials = 300

    catalog, ra, dec, instrumental = _generate_fake_catalog(n_stars, a=0.02, c=0.15)

    # The star this is measured on is an extra observation of the first
    # catalog star, 1.3 arcsec away from it. That is close enough to be
    # calibrated and too far to be used in the fit, which is the point: a star
    # that was in the fit moved the coefficients with its own noise, and both
    # this propagation and the Monte Carlo treat the two as independent, so
    # the comparison there is about 10% conservative. Out of the fit there is
    # nothing to correct for.
    ra = u.Quantity([*ra, ra[0]])
    dec = u.Quantity([*dec, dec[0] + 1.3 * u.arcsec])
    instrumental = np.append(instrumental, instrumental[0])

    _patch_catalog_fetch(mocker, catalog)

    calibrated = []
    reported = []
    for trial in range(n_trials):
        observed = _generate_observed_table(
            ra, dec, instrumental, noise_sigma=sigma, seed=10000 + trial
        )
        result = transform_to_catalog(
            observed,
            "R",
            obs_error_column="mag_error",
            cat_filter="R",
            cat_color=("R", "I"),
        )
        calibrated.append(result["mag_cal"][-1])
        reported.append(result["mag_cal_error"][-1])

    # Measured ratio 1.03. The 15% is room for the sampling error of a
    # 300-trial standard deviation, about 4%, on top of the ~10%
    # conservatism explained above.
    assert np.std(calibrated) == pytest.approx(np.mean(reported), rel=0.15)


def test_transform_to_catalog_error_includes_the_quadratic_sensitivity(mocker):
    # How much the calibrated magnitude moves when the instrumental one does
    # is 1 + a + 2*b*mag_inst, not 1 + a, whenever the quadratic term is being
    # fit -- and it is different for every star. Nothing writes that
    # expression down: it falls out of the instrumental magnitude appearing
    # twice in the model, once inside it and once in the fit_diff add-back,
    # which is exactly why it needs a test of its own. The twin-fit helper
    # subtracts away the transform term, leaving the sensitivity times the
    # quoted error alone. See issue #674.
    a, b = 0.02, 0.01

    measurement, instrumental = _measurement_half_of_the_error(
        mocker, a=a, b=b, c=0.15, d=0.05, vary=("a", "b", "c", "d", "z")
    )

    sensitivity = 1 + a + 2 * b * instrumental

    # The quadratic term really does change the answer here -- the
    # sensitivity runs from 0.82 to 0.92 across the field -- so a propagation
    # that ignored b could not pass.
    assert np.ptp(sensitivity) > 0.05

    np.testing.assert_allclose(
        measurement, np.abs(sensitivity) * _MIN_FIT_SIGMA, rtol=0, atol=1e-8
    )


def test_transform_to_catalog_uses_refcat2(mocker):
    # Only the apass_dr9 branch of the catalog fetch has ever been exercised,
    # so a typo in the refcat2 branch would have gone unnoticed.
    #
    # Because the catalog is mocked, what this checks is the *wiring*: that
    # asking for refcat2 really queries refcat2 rather than APASS, that the
    # refcat2 band transform is the one handed to passband_columns, and that
    # the rest of the pipeline works on what comes back. It says nothing about
    # the real refcat2 query or the real band transform -- in particular
    # whether masked catalog entries survive that transform. Testing those
    # needs a remote_data test per catalog; see issue #680.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)

    result = _run_transform_to_catalog(mocker, catalog, observed, cat_name="refcat2")

    # The patch is still in place, so the module attribute is the mock.
    assert magnitude_transforms.refcat2.call_count == 1
    assert (
        catalog.meta["passband_columns_call"]["transformer"]
        is magnitude_transforms.transform_refcat2_bands
    )
    np.testing.assert_allclose(result["mag_cal"], catalog["mag_R"], rtol=0, atol=1e-6)


def test_transform_to_catalog_unknown_catalog_raises():
    # Only two catalogs are supported. A misspelled name should say so rather
    # than falling through to whatever the last branch happens to do.
    _, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)

    with pytest.raises(ValueError, match="Unknown catalog name"):
        transform_to_catalog(
            observed,
            "R",
            obs_error_column="mag_error",
            cat_name="not-a-catalog",
            cat_filter="R",
            cat_color=("R", "I"),
        )


def test_transform_to_catalog_fixes_unvaried_terms(mocker):
    # Terms that are not varied are held at exactly zero. Faking a fixed
    # parameter with a narrow box, as the old bounds did, leaves them near
    # zero but not at it, so the comparison here is exact on purpose.
    result, _, _ = _fit_a_catalog(mocker, a=0.02, c=0.15)

    assert (result["b"] == 0.0).all()
    assert (result["d"] == 0.0).all()


def test_transform_to_catalog_vary_selects_terms(mocker):
    # Asking for only the zero point should fit only the zero point.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)

    result = _run_transform_to_catalog(mocker, catalog, observed, vary=("z",))

    for term in ("a", "b", "c", "d"):
        assert (result[term] == 0.0).all(), f"{term} should be fixed at zero"

    np.testing.assert_allclose(result["z"], _FAKE_CATALOG_ZERO_POINT, rtol=0, atol=1e-6)


@pytest.mark.parametrize("bad_vary", ["z", ("a", "q"), ()])
def test_transform_to_catalog_bad_vary_raises(mocker, bad_vary):
    # Each of these would otherwise fail silently or confusingly: a bare string
    # iterates into single characters, an unknown term would simply never be
    # fit, and varying nothing at all leaves every term pinned at zero.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)

    with pytest.raises((TypeError, ValueError), match="vary"):
        _run_transform_to_catalog(mocker, catalog, observed, vary=bad_vary)


@pytest.mark.parametrize(
    "term, true_value, expected",
    [
        # The default expected ranges, which cover only the zero point.
        ("z", 25.0, None),
        # Terms other than z, to catch an implementation that checked the
        # hardcoded default key rather than whatever it was handed.
        ("a", 0.05, {"a": (0.0, 0.01)}),
        ("c", 0.5, {"c": (-0.1, 0.1)}),
    ],
)
def test_transform_to_catalog_warns_when_term_outside_expected(
    mocker, term, true_value, expected, caplog
):
    # A zero point outside the expected range used to rail at the bound and
    # report a confident wrong answer. Every term should now be fit freely and
    # merely logged about, by name and by the value it reached. See issue #601.
    coefficients = {term: true_value}
    if term != "z":
        coefficients["z"] = _FAKE_CATALOG_ZERO_POINT

    catalog, ra, dec, instrumental = _generate_fake_catalog(20, **coefficients)
    observed = _generate_observed_table(ra, dec, instrumental)

    message_pattern = f"{term}={true_value:.4f} is outside"
    with caplog.at_level(logging.WARNING, logger=_MAGNITUDE_TRANSFORMS_LOGGER):
        result = _run_transform_to_catalog(mocker, catalog, observed, expected=expected)

    assert any(message_pattern in r.message for r in caplog.records)

    # The value is reported, not clamped to the edge of the expected range.
    np.testing.assert_allclose(result[term], true_value, rtol=0, atol=1e-6)
    np.testing.assert_allclose(result["mag_cal"], catalog["mag_R"], rtol=0, atol=1e-6)


def test_transform_to_catalog_unknown_expected_term_raises(mocker):
    # A typo in an expected range would otherwise be silently ignored, leaving
    # the caller believing a check is happening that is not.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)

    with pytest.raises(ValueError, match=r"Unknown term\(s\) in expected"):
        _run_transform_to_catalog(
            mocker, catalog, observed, expected={"zero_point": (18, 22)}
        )


def test_transform_to_catalog_empty_expected_disables_check(mocker):
    # Warnings are errors in this test suite, so a warning here fails the test.
    true_zero_point = 25.0

    catalog, ra, dec, instrumental = _generate_fake_catalog(20, z=true_zero_point)
    observed = _generate_observed_table(ra, dec, instrumental)

    result = _run_transform_to_catalog(mocker, catalog, observed, expected={})

    np.testing.assert_allclose(result["z"], true_zero_point, rtol=0, atol=1e-6)


def test_transform_to_catalog_warns_when_fit_fails(mocker, caplog):
    # A fit that does not converge is not reliably reproducible, so mock one.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)

    failed_fit = mocker.MagicMock()
    failed_fit.success = False
    failed_fit.message = "the fitter gave up"
    mocker.patch.object(magnitude_transforms.lmfit, "minimize", return_value=failed_fit)

    with caplog.at_level(logging.WARNING, logger=_MAGNITUDE_TRANSFORMS_LOGGER):
        result = _run_transform_to_catalog(mocker, catalog, observed)

    assert any("did not succeed" in r.message for r in caplog.records)

    assert np.isnan(result["mag_cal"]).all()


def test_transform_to_catalog_fit_diff_agrees(mocker):
    # fit_diff changes what is fit, not what comes out: fitting the difference
    # between the catalog and instrumental magnitudes should give the same
    # calibrated magnitudes *and the same errors* as fitting the catalog
    # magnitude directly. The coefficients themselves legitimately differ --
    # with fit_diff=False the true value of a is 1 rather than 0 -- which is
    # exactly why the error column has to be scaled by a different factor in
    # the two cases.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20, a=0.02, c=0.15)
    observed = _generate_observed_table(ra, dec, instrumental)

    with_diff = _run_transform_to_catalog(
        mocker, catalog, observed.copy(), fit_diff=True
    )
    without_diff = _run_transform_to_catalog(
        mocker, catalog, observed.copy(), fit_diff=False
    )

    np.testing.assert_allclose(
        with_diff["mag_cal"], without_diff["mag_cal"], rtol=0, atol=1e-6
    )
    # Looser than the magnitudes above, and deliberately so: the two modes fit
    # different numbers, so their covariances differ in the last few bits --
    # measured at 1.5e-11 -- and the propagation carries that into the errors
    # at about 4e-10. That clears 1e-8 by only a factor of 25, close enough to
    # the edge to be worth not waiting for.
    np.testing.assert_allclose(
        with_diff["mag_cal_error"], without_diff["mag_cal_error"], rtol=0, atol=1e-7
    )


def _make_unusable_observations(case, catalog, ra, dec, instrumental, **kwargs):
    """
    Make observations from which no transform can be fit.

    Parameters
    ----------

    case : str
        Which way the data is unusable: ``"out_of_range"`` for instrumental
        magnitudes outside the range the fit accepts, ``"nan"`` for
        instrumental magnitudes that are all NaN, or ``"masked_catalog"`` for
        a catalog whose magnitudes are entirely masked.

    catalog : `_FakeCatalogTable`
        Catalog the observations will be matched against. Modified in place
        for the ``"masked_catalog"`` case.

    ra, dec : `astropy.units.Quantity`
        Position of each star.

    instrumental : `numpy.ndarray`
        Instrumental magnitudes to start from.

    **kwargs
        Passed on to `_generate_observed_table`.

    Returns
    -------
    `astropy.table.Table`
        Observations, grouped by file name.
    """
    match case:
        case "out_of_range":
            # The fit only accepts instrumental magnitudes between -20 and -3.
            instrumental = np.zeros_like(instrumental)
        case "nan":
            instrumental = np.full_like(instrumental, np.nan)
        case "masked_catalog":
            catalog["mag_R"] = np.ma.masked_array(
                catalog["mag_R"], mask=np.ones(len(catalog), dtype=bool)
            )
        case _:  # pragma: no cover
            raise ValueError(f"Unknown case {case!r}")

    return _generate_observed_table(ra, dec, instrumental, **kwargs)


@pytest.mark.parametrize("case", ["out_of_range", "nan", "masked_catalog"])
def test_transform_to_catalog_no_good_data_warns_and_nans(mocker, case, caplog):
    # An image can be unusable from either side -- bad measurements or no
    # catalog to compare them to -- and the answer is the same either way: say
    # which image it was, and leave its outputs NaN rather than reporting
    # coefficients fit to nothing.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _make_unusable_observations(case, catalog, ra, dec, instrumental)

    with caplog.at_level(logging.WARNING, logger=_MAGNITUDE_TRANSFORMS_LOGGER):
        result = _run_transform_to_catalog(mocker, catalog, observed)

    assert any("image_1.fit" in r.message for r in caplog.records)

    assert np.isnan(result["mag_cal"]).all()
    for column in _FIT_COLUMNS:
        assert np.isnan(result[column]).all(), column


def _one_good_one_bad_image(catalog, ra, dec, instrumental):
    """
    Observations of two images, the first usable and the second not.

    The unusable image is unusable in a way that makes `transform_to_catalog`
    give up on it partway through the loop over images, warning and naming
    ``image_2.fit`` as it does. That is the setup both for "one bad image does
    not cost the others their results" and for "those results are in the table
    even when the warning is escalated to an error".

    Parameters
    ----------

    catalog : `_FakeCatalogTable`
        Catalog the observations will be matched against.

    ra, dec : `astropy.units.Quantity`
        Position of each star. Both images observe the same stars.

    instrumental : `numpy.ndarray`
        Instrumental magnitudes of the usable image.

    Returns
    -------
    `astropy.table.Table`
        Observations of both images, grouped by file name.
    """
    return _combine_observed_tables(
        _generate_observed_table(ra, dec, instrumental, file_name="image_1.fit"),
        _make_unusable_observations(
            "out_of_range",
            catalog,
            ra,
            dec,
            instrumental,
            file_name="image_2.fit",
        ),
    )


def test_transform_to_catalog_one_bad_image_does_not_poison_others(mocker, caplog):
    # A night's data can easily contain one unusable image. That image should
    # cost its own results and nothing else -- the fits for the other images
    # still have to come back.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)

    observed = _one_good_one_bad_image(catalog, ra, dec, instrumental)

    with caplog.at_level(logging.WARNING, logger=_MAGNITUDE_TRANSFORMS_LOGGER):
        result = _run_transform_to_catalog(mocker, catalog, observed)

    assert any("image_2.fit" in r.message for r in caplog.records)

    good = result["file"] == "image_1.fit"

    np.testing.assert_allclose(
        result["mag_cal"][good], catalog["mag_R"], rtol=0, atol=1e-6
    )
    assert np.isnan(result["mag_cal"][~good]).all()


def test_transform_to_catalog_bad_image_logs_instead_of_raising(mocker, caplog):
    # Problems with one image are log messages, not warnings, so the call
    # completes and the table is written even with warnings escalated to
    # errors -- which this suite's own configuration does. See issue #679.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)

    observed = _one_good_one_bad_image(catalog, ra, dec, instrumental)

    with caplog.at_level(logging.WARNING, logger=_MAGNITUDE_TRANSFORMS_LOGGER):
        result = _run_transform_to_catalog(mocker, catalog, observed, in_place=True)

    assert any("image_2.fit" in r.message for r in caplog.records)

    # The call completed and the table is complete.
    assert _TRANSFORM_OUTPUT_COLUMNS <= set(result.colnames)

    good = result["file"] == "image_1.fit"

    np.testing.assert_allclose(
        result["mag_cal"][good], catalog["mag_R"], rtol=0, atol=1e-6
    )
    assert np.isnan(result["mag_cal"][~good]).all()


def test_transform_to_catalog_out_of_range_term_logs_instead_of_raising(mocker, caplog):
    # A term outside its expected range is a report on a fit that succeeded,
    # so it is a log message rather than a warning: every star keeps its
    # calibrated magnitude even with warnings escalated to errors. See
    # issue #679.
    true_zero_point = 25.0

    catalog, ra, dec, instrumental = _generate_fake_catalog(20, z=true_zero_point)
    observed = _generate_observed_table(ra, dec, instrumental)

    with caplog.at_level(logging.WARNING, logger=_MAGNITUDE_TRANSFORMS_LOGGER):
        result = _run_transform_to_catalog(mocker, catalog, observed, in_place=True)

    assert any("z=25.0000 is outside" in r.message for r in caplog.records)
    assert _TRANSFORM_OUTPUT_COLUMNS <= set(result.colnames)

    np.testing.assert_allclose(result["mag_cal"], catalog["mag_R"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(result["z"], true_zero_point, rtol=0, atol=1e-6)


def test_transform_to_catalog_excludes_masked_catalog_entries(mocker):
    # Real APASS data is masked, so this is the path users actually take.
    n_stars = 20

    catalog, ra, dec, instrumental = _generate_fake_catalog(n_stars)
    observed = _generate_observed_table(ra, dec, instrumental)

    masked = np.zeros(n_stars, dtype=bool)
    masked[::2] = True
    catalog["mag_R"] = np.ma.masked_array(catalog["mag_R"], mask=masked)

    result = _run_transform_to_catalog(mocker, catalog, observed)

    assert np.isnan(result["mag_cal"][masked]).all()
    # The fit still recovers the truth from the unmasked half.
    np.testing.assert_allclose(
        result["mag_cal"][~masked],
        np.asarray(catalog["mag_R"])[~masked],
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result["z"][~masked], _FAKE_CATALOG_ZERO_POINT, rtol=0, atol=1e-6
    )


def _catalog_without_color(n_stars, no_color, **kwargs):
    """
    Build a synthetic catalog in which some stars have no color.

    Masking ``mag_I`` is how a real catalog says it has no measurement in that
    band. The color `transform_to_catalog` works out is masked in turn, and
    reaches the fit as NaN, so nothing in `_generate_fake_catalog` has to
    change to produce a star with no color.

    Parameters
    ----------

    n_stars : int
        Number of stars to generate.

    no_color : `numpy.ndarray`
        Boolean mask, `True` for the stars that should have no color.

    **kwargs
        Passed on to `_generate_fake_catalog`.

    Returns
    -------
    Whatever `_generate_fake_catalog` returns, with ``mag_I`` masked.
    """
    catalog, ra, dec, instrumental = _generate_fake_catalog(n_stars, **kwargs)
    catalog["mag_I"] = np.ma.masked_array(catalog["mag_I"], mask=no_color)

    return catalog, ra, dec, instrumental


def test_transform_to_catalog_fits_colorless_stars_when_no_color_term(mocker, caplog):
    # A star with no catalog color has nothing missing as far as a fit with no
    # color term in it is concerned, so demanding a color of it throws away
    # data for no reason. See issue #681.
    n_stars = 20

    catalog, ra, dec, instrumental = _catalog_without_color(
        n_stars, np.ones(n_stars, dtype=bool)
    )
    observed = _generate_observed_table(ra, dec, instrumental)

    with caplog.at_level(logging.WARNING, logger=_MAGNITUDE_TRANSFORMS_LOGGER):
        result = _run_transform_to_catalog(mocker, catalog, observed, vary=("a", "z"))

    assert not any("No good data" in r.message for r in caplog.records)

    np.testing.assert_allclose(result["mag_cal"], catalog["mag_R"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(result["z"], _FAKE_CATALOG_ZERO_POINT, rtol=0, atol=1e-6)
    # The color is genuinely unknown, and the output column still says so.
    assert np.isnan(result["color_cat"]).all()


def test_transform_to_catalog_still_needs_color_when_fitting_a_color_term(
    mocker, caplog
):
    # The other half of the relaxation above, which is what keeps it from being
    # unconditional: a color term fit to stars with no color is not a fit.
    n_stars = 20

    catalog, ra, dec, instrumental = _catalog_without_color(
        n_stars, np.ones(n_stars, dtype=bool)
    )
    observed = _generate_observed_table(ra, dec, instrumental)

    with caplog.at_level(logging.WARNING, logger=_MAGNITUDE_TRANSFORMS_LOGGER):
        result = _run_transform_to_catalog(mocker, catalog, observed)

    assert any("No good data" in r.message for r in caplog.records)
    assert np.isnan(result["mag_cal"]).all()


def test_transform_to_catalog_mixes_stars_with_and_without_color(mocker):
    # The realistic case: a catalog with a color for some stars and not others.
    # With no color term being fit, all of them belong in the fit, and the
    # color column is what says which ones had a color to begin with.
    n_stars = 20

    no_color = np.zeros(n_stars, dtype=bool)
    no_color[::2] = True

    catalog, ra, dec, instrumental = _catalog_without_color(n_stars, no_color)
    observed = _generate_observed_table(ra, dec, instrumental)

    result = _run_transform_to_catalog(mocker, catalog, observed, vary=("a", "z"))

    # Every star is calibrated, color or no color...
    np.testing.assert_allclose(result["mag_cal"], catalog["mag_R"], rtol=0, atol=1e-6)
    # ...and the color column reports exactly the ones that had none.
    assert np.isnan(result["color_cat"][no_color]).all()
    assert np.isfinite(result["color_cat"][~no_color]).all()


def _two_passband_observations(ra, dec, instrumental, i_offset=0.5):
    """
    Observations of one image containing two passbands.

    Grouping such a table by file alone puts rows of both passbands in each
    group, which is the case the per-passband filtering has to handle.
    """
    return _combine_observed_tables(
        _generate_observed_table(ra, dec, instrumental, passband="R"),
        _generate_observed_table(ra, dec, instrumental - i_offset, passband="I"),
    )


def test_transform_to_catalog_handles_multiple_passbands(mocker):
    # Groups containing rows in passbands other than the one being fit used to
    # raise, because the output columns were built from the filtered rows but
    # assigned to the whole table.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _two_passband_observations(ra, dec, instrumental)

    result = _run_transform_to_catalog(mocker, catalog, observed, obs_filter="R")

    assert len(result) == len(observed)

    is_r = result["passband"] == "R"
    np.testing.assert_allclose(
        result["mag_cal"][is_r], catalog["mag_R"], rtol=0, atol=1e-6
    )
    assert np.isnan(result["mag_cal"][~is_r]).all()


def test_transform_to_catalog_successive_passband_calls_accumulate(mocker):
    # Calling once per passband with in_place=True should build up one table
    # rather than each call clobbering the last.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _two_passband_observations(ra, dec, instrumental)

    result = _run_transform_to_catalog(mocker, catalog, observed, obs_filter="R")
    result = _run_transform_to_catalog(
        mocker, catalog, result, obs_filter="I", cat_filter="I"
    )

    is_r = result["passband"] == "R"

    # The R values survived the I call...
    np.testing.assert_allclose(
        result["mag_cal"][is_r], catalog["mag_R"], rtol=0, atol=1e-6
    )
    # ...and the I rows are now filled in too.
    np.testing.assert_allclose(
        result["mag_cal"][~is_r], np.asarray(catalog["mag_I"]), rtol=0, atol=1e-6
    )


def test_transform_to_catalog_without_error_column(mocker):
    # The error column is optional but nearly always wanted, so leaving it out
    # has to say so. It also used to crash outright rather than doing the
    # unweighted fit the docstring promised, so this checks the fit still runs
    # and that no calibrated error is invented for it.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)

    with pytest.warns(AstropyUserWarning, match="rror weighting"):
        result = _run_transform_to_catalog(
            mocker, catalog, observed, obs_error_column=None
        )

    assert "mag_cal_error" not in result.colnames
    np.testing.assert_allclose(result["mag_cal"], catalog["mag_R"], rtol=0, atol=1e-6)


def test_transform_to_catalog_passband_not_in_table_raises(mocker):
    # Asking for a passband the table does not contain is a mistake, not an
    # empty result, and the message should say which passbands are there --
    # the usual cause is a name that differs from the one in the file.
    #
    # The catalog band is deliberately a different one, which makes this the
    # test that pins the order of the two checks: the table is checked before
    # the arguments are, so a caller who got the passband name wrong hears
    # about the name rather than about a mismatch that is a consequence of it.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental, passband="R")

    with pytest.raises(ValueError, match="No rows with passband 'B'"):
        _run_transform_to_catalog(
            mocker, catalog, observed, obs_filter="B", cat_filter="R"
        )


def test_transform_to_catalog_mismatched_passbands_warns_and_proceeds(mocker):
    # Calibrating V observations against the catalog's B folds the B - V of
    # every star into the fit, which comes back as a zero point and a color
    # term and is then applied to every star in the image. That is exactly
    # what unfiltered and DSLR observations -- AAVSO's CV and TG -- need,
    # because there is no observed V to match against instead, so naming a
    # different catalog band explicitly is taken as deliberate rather than
    # rejected. The contamination it can cause is the warning's job to flag,
    # not the function's job to prevent. See issue #680.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    _catalog_posing_as_bands(catalog, V="R", B="I")
    observed = _generate_observed_table(ra, dec, instrumental, passband="V")

    with pytest.warns(AstropyUserWarning, match="passband 'V'.*passband 'B'"):
        result = _run_transform_to_catalog(
            mocker, catalog, observed, obs_filter="V", cat_filter="B"
        )

    # The fit proceeds and calibrates against the named catalog band.
    np.testing.assert_allclose(result["mag_cal"], catalog["mag_B"], rtol=0, atol=1e-6)


@pytest.mark.parametrize("cat_filter", ["R", "RC", "Rc", "rc"])
def test_transform_to_catalog_accepts_equivalent_passband_names(mocker, cat_filter):
    # The half that keeps the check above from being over-eager. Cousins R is
    # written several ways, and refcat2's band transform really does add
    # mag_R and mag_RC as copies of one column, so R against RC is one band
    # spelled two ways rather than two bands. Canonicalization is
    # case-insensitive, which is what "rc" exercises here. Because this suite
    # turns warnings into errors, a passing call already proves the mismatch
    # warning above was not raised for any of these spellings.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    _catalog_posing_as_bands(catalog, RC="R", Rc="R")
    observed = _generate_observed_table(ra, dec, instrumental)

    result = _run_transform_to_catalog(mocker, catalog, observed, cat_filter=cat_filter)

    np.testing.assert_allclose(result["mag_cal"], catalog["mag_R"], rtol=0, atol=1e-6)


def test_transform_to_catalog_unknown_default_color_raises(mocker):
    # The color that goes with a band is a convention rather than something
    # derivable, so a band with no convention recorded has to be asked about
    # instead of guessed at -- a wrong guess would name a column that is not
    # there, and the KeyError that follows says nothing about what to do. This
    # only bites when a color term is actually being fit, which is what the
    # default ``vary`` does.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    _catalog_posing_as_bands(catalog, TG="R")
    observed = _generate_observed_table(ra, dec, instrumental, passband="TG")

    with pytest.raises(ValueError, match="cat_color must be given"):
        _run_transform_to_catalog(mocker, catalog, observed, obs_filter="TG")

    # Naming the color is all it takes.
    result = _run_transform_to_catalog(
        mocker, catalog, observed, obs_filter="TG", cat_color=("R", "I")
    )
    np.testing.assert_allclose(result["mag_cal"], catalog["mag_TG"], rtol=0, atol=1e-6)


def test_transform_to_catalog_unknown_default_color_runs_without_color_term(mocker):
    # The other side of the test above: with no color term in ``vary``, the
    # model never needs a color, so a band with no conventional color does
    # not have to raise -- it can only leave ``color_cat`` unknown, which is
    # what it already is for a star the catalog has no color for.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    _catalog_posing_as_bands(catalog, TG="R")
    observed = _generate_observed_table(ra, dec, instrumental, passband="TG")

    result = _run_transform_to_catalog(
        mocker, catalog, observed, obs_filter="TG", vary=("a", "z")
    )

    np.testing.assert_allclose(result["mag_cal"], catalog["mag_TG"], rtol=0, atol=1e-6)
    assert np.isnan(result["color_cat"]).all()


def test_transform_to_catalog_skips_image_without_the_passband(mocker):
    # An image with nothing in the passband being fit is not an image with bad
    # data -- it should be passed over quietly rather than warned about.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)

    observed = _combine_observed_tables(
        _generate_observed_table(
            ra, dec, instrumental, file_name="image_1.fit", passband="R"
        ),
        _generate_observed_table(
            ra, dec, instrumental, file_name="image_2.fit", passband="I"
        ),
    )

    result = _run_transform_to_catalog(mocker, catalog, observed, obs_filter="R")

    has_the_passband = result["file"] == "image_1.fit"
    np.testing.assert_allclose(
        result["mag_cal"][has_the_passband], catalog["mag_R"], rtol=0, atol=1e-6
    )
    assert np.isnan(result["mag_cal"][~has_the_passband]).all()


def test_transform_to_catalog_warns_when_outlier_cut_removes_everything(mocker, caplog):
    # Stars more than a magnitude from the median offset between the catalog
    # and instrumental magnitudes are dropped. Here every star is, which
    # leaves nothing to fit even though the data and the catalog matches are
    # individually fine.
    catalog, ra, dec, instrumental = _generate_fake_catalog(2)

    scattered = instrumental + np.array([2.5, -2.5])
    observed = _generate_observed_table(ra, dec, scattered)

    with caplog.at_level(logging.WARNING, logger=_MAGNITUDE_TRANSFORMS_LOGGER):
        result = _run_transform_to_catalog(mocker, catalog, observed)

    assert any("No good data" in r.message for r in caplog.records)

    assert np.isnan(result["mag_cal"]).all()


def test_transform_to_catalog_disjoint_good_data_and_catalog(mocker, caplog):
    # The stars with usable instrumental magnitudes and the stars with usable
    # catalog entries can be two entirely different sets. Checking that each is
    # non-empty on its own is not enough: the median offset used for the
    # outlier cut is then taken over an empty selection, which numpy reports as
    # a RuntimeWarning -- a hard failure for anyone running with warnings as
    # errors, as this suite itself does.
    n_stars = 20
    half = n_stars // 2

    catalog, ra, dec, instrumental = _generate_fake_catalog(n_stars)

    # Catalog usable only for the second half of the stars...
    masked = np.zeros(n_stars, dtype=bool)
    masked[:half] = True
    catalog["mag_R"] = np.ma.masked_array(catalog["mag_R"], mask=masked)

    # ...and instrumental magnitudes usable only for the first half.
    instrumental = instrumental.copy()
    instrumental[half:] = np.nan

    observed = _generate_observed_table(ra, dec, instrumental)

    with caplog.at_level(logging.WARNING, logger=_MAGNITUDE_TRANSFORMS_LOGGER):
        result = _run_transform_to_catalog(mocker, catalog, observed)

    assert any("No good data" in r.message for r in caplog.records)
    assert np.isnan(result["mag_cal"]).all()


def test_transform_to_catalog_warns_when_too_few_stars_to_fit(mocker, caplog):
    # Three stars cannot pin down three coefficients: the fit has no degrees of
    # freedom left, so it reproduces those three stars exactly and says nothing
    # at all about any other star in the image. lmfit reports success anyway,
    # so counting has to happen here -- and it has to happen before the fit is
    # run, because with fewer stars than terms lmfit takes the square root of a
    # negative number working out the uncertainties.
    catalog, ra, dec, instrumental = _generate_fake_catalog(3)
    observed = _generate_observed_table(ra, dec, instrumental)

    with caplog.at_level(logging.WARNING, logger=_MAGNITUDE_TRANSFORMS_LOGGER):
        result = _run_transform_to_catalog(mocker, catalog, observed)

    assert any("underdetermined" in r.message for r in caplog.records)

    assert np.isnan(result["mag_cal"]).all()
    for column in _FIT_COLUMNS:
        assert np.isnan(result[column]).all(), column


def test_transform_to_catalog_warns_when_terms_are_degenerate(mocker, caplog):
    # Plenty of stars is not enough if they carry no independent information. A
    # color that is a linear function of instrumental magnitude makes a, c and
    # z exactly degenerate -- infinitely many combinations fit equally well --
    # and a star-count guard alone would let this through.
    n_stars = 20

    instrumental = np.linspace(-10.0, -5.0, num=n_stars)
    catalog, ra, dec, _ = _generate_fake_catalog(
        n_stars, instrumental=instrumental, color=0.1 * (instrumental + 10.0)
    )
    observed = _generate_observed_table(ra, dec, instrumental)

    with caplog.at_level(logging.WARNING, logger=_MAGNITUDE_TRANSFORMS_LOGGER):
        result = _run_transform_to_catalog(mocker, catalog, observed)

    assert any("underdetermined" in r.message for r in caplog.records)

    assert np.isnan(result["mag_cal"]).all()


def test_transform_to_catalog_warns_when_terms_are_nearly_degenerate(mocker, caplog):
    # Exactly degenerate data is a synthetic construct: it takes a color that is
    # a linear function of instrumental magnitude to the last bit, which real
    # stars never are. What real data does produce is *near* degeneracy -- a
    # field whose stars run red-and-faint to blue-and-bright, so color is very
    # nearly a linear function of instrumental magnitude. The fit then reports
    # success and lands on coefficients that are wrong by whole magnitudes, so
    # a check that only rejects data of exactly deficient rank is no use here.
    n_stars = 200
    color_scatter = 3e-4

    instrumental = np.linspace(-12.0, -6.0, num=n_stars)
    color = 0.1 * (instrumental + 12.0) + np.random.default_rng(2317).normal(
        0.0, color_scatter, size=n_stars
    )
    catalog, ra, dec, _ = _generate_fake_catalog(
        n_stars, a=0.02, c=0.15, instrumental=instrumental, color=color
    )
    observed = _generate_observed_table(ra, dec, instrumental)

    with caplog.at_level(logging.WARNING, logger=_MAGNITUDE_TRANSFORMS_LOGGER):
        result = _run_transform_to_catalog(mocker, catalog, observed)

    assert any("underdetermined" in r.message for r in caplog.records)

    assert np.isnan(result["mag_cal"]).all()


def test_transform_to_catalog_warns_when_a_term_has_no_leverage(mocker, caplog):
    # The extreme end of the same problem: if the two catalog bands the color
    # is built from are identical, every star's color is zero and the color
    # term does nothing whatever -- any value of c fits exactly as well as any
    # other. Nothing needs to handle that specially, which is the point of
    # measuring the fit's Jacobian rather than its covariance: the column for
    # a term the data cannot see is zero, so the condition number comes out
    # infinite by itself.
    n_stars = 20

    catalog, ra, dec, instrumental = _generate_fake_catalog(
        n_stars, a=0.02, color=np.zeros(n_stars)
    )
    observed = _generate_observed_table(ra, dec, instrumental)

    with caplog.at_level(logging.WARNING, logger=_MAGNITUDE_TRANSFORMS_LOGGER):
        result = _run_transform_to_catalog(mocker, catalog, observed)

    assert any("cannot be told apart" in r.message for r in caplog.records)

    assert np.isnan(result["mag_cal"]).all()


def test_transform_to_catalog_fits_correlated_but_usable_data(mocker):
    # The other side of the test above. Color and instrumental magnitude are
    # correlated in almost any real field, and a check on how well the terms
    # can be told apart must not reject data merely for being correlated. Ten
    # times the scatter of the test above is still a strong correlation, and it
    # still recovers the coefficients, so it has to go through without warning.
    n_stars = 200
    color_scatter = 3e-2

    instrumental = np.linspace(-12.0, -6.0, num=n_stars)
    color = 0.1 * (instrumental + 12.0) + np.random.default_rng(2317).normal(
        0.0, color_scatter, size=n_stars
    )
    catalog, ra, dec, _ = _generate_fake_catalog(
        n_stars, a=0.02, c=0.15, instrumental=instrumental, color=color
    )
    observed = _generate_observed_table(ra, dec, instrumental)

    result = _run_transform_to_catalog(mocker, catalog, observed)

    np.testing.assert_allclose(result["a"], 0.02, rtol=0, atol=1e-4)
    np.testing.assert_allclose(result["c"], 0.15, rtol=0, atol=1e-4)


def test_transform_to_catalog_warns_when_fixed_term_outside_expected(mocker, caplog):
    # Leaving z out of vary pins it at zero, which is nowhere near the expected
    # zero point of a real image. A term held at a wrong value is exactly what
    # the caller needs to be told about, so the expected ranges are checked for
    # fixed terms as well as fitted ones.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20, a=0.02, c=0.15)
    observed = _generate_observed_table(ra, dec, instrumental)

    with caplog.at_level(logging.WARNING, logger=_MAGNITUDE_TRANSFORMS_LOGGER):
        result = _run_transform_to_catalog(mocker, catalog, observed, vary=("a", "c"))

    assert any("z=0.0000 is outside" in r.message for r in caplog.records)

    assert (result["z"] == 0.0).all()


# Number of leading rows `_observations_with_unusable_rows` spoils.
_N_UNUSABLE_ROWS = 2


def _observations_with_unusable_rows(case, ra, dec, instrumental):
    """
    Observations whose first `_N_UNUSABLE_ROWS` rows cannot carry an error.

    The rest of the rows are untouched, so the fit still has plenty to work
    with and the spoiled rows are the only thing under test.

    Parameters
    ----------

    case : str
        What is wrong with those rows: ``"bad_error"`` for magnitude errors
        that are not positive finite numbers, or ``"no_magnitude"`` for rows
        with no instrumental magnitude at all.

    ra, dec : `astropy.units.Quantity`
        Position of each star.

    instrumental : `numpy.ndarray`
        Instrumental magnitude of each star.

    Returns
    -------
    `astropy.table.Table`
        Observations, grouped by file name.
    """
    observed = _generate_observed_table(ra, dec, instrumental)

    match case:
        case "bad_error":
            # Zero and negative are the two ways an error can be meaningless.
            observed["mag_error"][0] = 0.0
            observed["mag_error"][1] = -0.02
        case "no_magnitude":
            observed["mag_inst"][:_N_UNUSABLE_ROWS] = np.nan
        case _:  # pragma: no cover
            raise ValueError(f"Unknown case {case!r}")

    return observed


def _assert_rows_blind_to_the_spoiled_ones(mocker, catalog, observed, result):
    """
    Assert the surviving rows never learn the spoiled rows existed.

    The reference is the same catalog transformed against the same
    observations with the first `_N_UNUSABLE_ROWS` rows removed rather than
    spoiled. The fit sees exactly the same usable stars either way, so
    every output column of the surviving rows -- per-star and per-image
    diagnostics alike -- must come out identical, bit for bit, not
    approximately: any difference at all means a spoiled row reached the
    fit, the diagnostics, or the propagation.
    """
    reference = _run_transform_to_catalog(
        mocker, catalog, Table(observed[_N_UNUSABLE_ROWS:]).group_by("file")
    )

    for column in sorted(_TRANSFORM_OUTPUT_COLUMNS):
        np.testing.assert_array_equal(
            np.asarray(result[column][_N_UNUSABLE_ROWS:]),
            np.asarray(reference[column]),
            err_msg=column,
        )


def test_transform_to_catalog_nans_error_for_unusable_input_error(mocker):
    # An error that is zero or negative is meaningless, and the fit already
    # excludes those stars. The calibrated error must not be finite for them
    # either: the AAVSO writer only turns non-finite errors into "na", so a
    # zero would be written into a submission as a real uncertainty -- and an
    # error of zero is infinite weight in any downstream average.
    #
    # One of the two spoiled rows carries a *negative* error, which is also
    # what pins the order the propagation is done in: `uncertainties` refuses
    # outright to build a value with a negative standard deviation, so the
    # sanitizing below has to happen before the propagation rather than after
    # it. NaN it accepts, and carries through to exactly the NaN wanted here.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _observations_with_unusable_rows("bad_error", ra, dec, instrumental)

    result = _run_transform_to_catalog(mocker, catalog, observed)

    assert np.isnan(result["mag_cal_error"][:_N_UNUSABLE_ROWS]).all()
    _assert_rows_blind_to_the_spoiled_ones(mocker, catalog, observed, result)

    # The stars are only left out of the *fit*. They still have perfectly good
    # instrumental magnitudes, so they still get calibrated magnitudes.
    np.testing.assert_allclose(result["mag_cal"], catalog["mag_R"], rtol=0, atol=1e-6)


def test_transform_to_catalog_nans_error_where_there_is_no_magnitude(mocker):
    # A calibrated error with no calibrated magnitude beside it is a trap:
    # selecting rows on a finite mag_cal_error keeps them, and whatever reads
    # mag_cal next gets NaN. A finite error must always mean there is a
    # magnitude for it to be the error of, whatever the reason the magnitude
    # is missing -- here an unmeasured star rather than a distant match. See
    # issue #678.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _observations_with_unusable_rows("no_magnitude", ra, dec, instrumental)

    result = _run_transform_to_catalog(mocker, catalog, observed)

    assert np.isnan(result["mag_cal"][:_N_UNUSABLE_ROWS]).all()
    assert np.isnan(result["mag_cal_error"][:_N_UNUSABLE_ROWS]).all()

    # The stars that were measured are unaffected.
    np.testing.assert_allclose(
        result["mag_cal"][_N_UNUSABLE_ROWS:],
        np.asarray(catalog["mag_R"])[_N_UNUSABLE_ROWS:],
        rtol=0,
        atol=1e-6,
    )
    _assert_rows_blind_to_the_spoiled_ones(mocker, catalog, observed, result)


@pytest.mark.parametrize(
    "band, color",
    [
        # The band this was written for, and its conventional color.
        ("R", ("R", "I")),
        # A band whose conventional color is a different pair entirely, which
        # is what stops the defaults being a pair of literals. Before this the
        # default was ``cat_filter="R"``, so calibrating V without naming the
        # catalog band silently fit V observations against catalog R.
        ("V", ("B", "V")),
    ],
)
def test_transform_to_catalog_default_catalog_columns(mocker, band, color):
    # cat_filter and cat_color name columns the catalog is indexed by as
    # mag_<name>, which means they have to be passband names. Left out, they
    # follow the observed passband: the catalog band is the one that was
    # observed, since anything else is an error, and the color is the pair
    # conventionally used with it.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)

    # The catalog's own R stands in for the band being calibrated, and its I
    # for the other half of the color pair. Which half that is depends on the
    # band: V is the second of B - V and R is the first of R - I.
    other = color[0] if color[1] == band else color[1]
    _catalog_posing_as_bands(catalog, **{band: "R", other: "I"})

    observed = _generate_observed_table(ra, dec, instrumental, passband=band)

    _patch_catalog_fetch(mocker, catalog)

    result = transform_to_catalog(observed, band, obs_error_column="mag_error")

    np.testing.assert_allclose(
        result["mag_cal"], catalog[f"mag_{band}"], rtol=0, atol=1e-6
    )


def test_transform_to_catalog_accepts_photometry_data(mocker):
    # The shipped notebook hands this function a PhotometryData, not a plain
    # Table, and every other test here uses a plain Table. PhotometryData
    # carries units on its columns and has real photometry's gaps in it -- rows
    # with no magnitude at all -- so it exercises input this file otherwise
    # builds too tidily.
    n_unmeasured = 2

    photometry = PhotometryData.read(
        get_pkg_data_filename(
            "tests/data/test_photometry_data.ecsv", package="stellarphot"
        )
    )
    # One image's worth of rows, so that each star appears exactly once and
    # every observation has an unambiguous nearest catalog entry. Selecting on
    # the file name only does that if no two nights reuse a name, which is the
    # property actually being relied on, so check it rather than the data file.
    photometry = photometry[photometry["file"] == sorted(set(photometry["file"]))[0]]
    assert len(set(photometry["star_id"])) == len(
        photometry
    ), "the rows selected are not one image's worth"

    catalog, ra, dec, instrumental = _generate_fake_catalog(len(photometry))

    # The photometry is in Sloan r', which is a different passband from
    # Cousins R however close the two are, so the catalog has to be asked for
    # r' as well -- until this was fixed the test calibrated SR observations
    # against catalog R, exactly the mistake #680 makes an error. Renaming the
    # synthetic catalog's bands is enough: nothing here depends on the
    # magnitudes being real ones.
    _catalog_posing_as_bands(catalog, SR="R", SI="I")

    # Real photometry has gaps in it -- stars that fell off the chip, or were
    # too faint to measure -- so keep some here.
    instrumental[:n_unmeasured] = np.nan

    photometry["ra"] = ra
    photometry["dec"] = dec
    photometry["mag_inst"] = instrumental
    photometry["mag_error"] = np.full(len(photometry), 0.01)

    result = _run_transform_to_catalog(
        mocker, catalog, photometry.group_by("file"), obs_filter="SR"
    )

    assert isinstance(result, PhotometryData)

    # The measured stars are calibrated back to the catalog; the unmeasured
    # ones stay NaN rather than turning into a number.
    np.testing.assert_allclose(
        result["mag_cal"][n_unmeasured:],
        np.asarray(catalog["mag_SR"])[n_unmeasured:],
        rtol=0,
        atol=1e-6,
    )
    assert np.isnan(result["mag_cal"][:n_unmeasured]).all()


def test_transform_to_catalog_non_numeric_existing_column_raises(mocker):
    # Values for rows in other passbands are kept by reading the column that
    # is already there, which cannot work if that column holds something other
    # than numbers.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)
    observed["mag_cal"] = ["not a magnitude"] * len(observed)

    with pytest.raises(ValueError, match="'mag_cal' is already in the table"):
        _run_transform_to_catalog(mocker, catalog, observed)


def test_transform_to_catalog_records_weighting_mode_in_meta(mocker):
    # Which errors were available to weight the fit by is a fact about the
    # call, not about any one star, so it belongs in the table's meta rather
    # than in a per-row column -- a caller reading mag_cal_error still needs
    # to know whether it is backed by the catalog's own uncertainty or by the
    # observed error alone.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)

    result = _run_transform_to_catalog(mocker, catalog, observed, in_place=False)
    assert result.meta["transform_weighting"]["R"] == "combined"

    del catalog["mag_error_R"]
    result = _run_transform_to_catalog(mocker, catalog, observed, in_place=False)
    assert result.meta["transform_weighting"]["R"] == "observed"

    with pytest.warns(AstropyUserWarning, match="rror weighting"):
        result = _run_transform_to_catalog(
            mocker, catalog, observed, obs_error_column=None, in_place=False
        )
    assert result.meta["transform_weighting"]["R"] == "unweighted"


def test_transform_to_catalog_weighting_meta_keeps_earlier_passbands(mocker):
    # transform_to_catalog is called once per passband on the same table, so a
    # flat key in meta would be overwritten by the second call. Keying by
    # obs_filter, exactly as the caller passed it, is what keeps both calls'
    # entries alive.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _two_passband_observations(ra, dec, instrumental)

    result = _run_transform_to_catalog(mocker, catalog, observed, obs_filter="R")
    result = _run_transform_to_catalog(
        mocker, catalog, result, obs_filter="I", cat_filter="I"
    )

    assert result.meta["transform_weighting"] == {"R": "combined", "I": "combined"}


def test_transform_to_catalog_missing_catalog_band_raises_clean_error(mocker):
    # Asking for a band the catalog cannot supply used to pass every argument
    # check, spend the Vizier round trip, and then die in a bare KeyError.
    # The message should say plainly which band is missing and which ones the
    # catalog actually has.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental, passband="V")

    with pytest.warns(AstropyUserWarning, match="passband 'V'.*passband 'U'"):
        with pytest.raises(ValueError, match=r"'U'.*'I', 'R'") as exc_info:
            _run_transform_to_catalog(
                mocker,
                catalog,
                observed,
                obs_filter="V",
                cat_filter="U",
                cat_color=("R", "I"),
            )

    assert not isinstance(exc_info.value, KeyError)


# Everything below runs only with --remote-data, in the single tox coverage
# job that passes it. The rest of this file mocks the catalog fetch, which
# leaves two things untested: the real Vizier query, and the real band
# transform that turns a catalog's native passbands into Johnson-Cousins ones
# -- in particular whether the masked entries it produces survive as masked
# rather than as a fill value. See issue #680.

# Center of the field the remote tests calibrate against: the north galactic
# pole. The field matters more than it looks. `transform_to_catalog` searches
# a hardcoded one degree around the first observation and
# `CatalogData.from_vizier` asks for every row in that cone, so the test pulls
# a full degree-radius catalog however small a field it builds observations
# from -- see issue #686. At the galactic pole that cone holds as few stars as
# any patch of sky does, which is what keeps the query to something Vizier
# will answer in reasonable time.
_REMOTE_FIELD_CENTER = SkyCoord(ra=192.85948 * u.degree, dec=27.12825 * u.degree)

# Radius of the separate, small query that supplies the stars the fake
# observations are built from. Deliberately far smaller than the one degree
# above: the point is to observe a handful of real stars near the center, not
# to reproduce the query under test.
_REMOTE_OBSERVED_RADIUS = 10 * u.arcmin

# Zero point the fake observations are built with, and the scatter added to
# them. The scatter keeps the fit from being exactly degenerate and gives
# `fit_redchi` and the covariance something to describe; the zero point is
# inside the default expected range for z, so a healthy fit warns about
# nothing.
_REMOTE_ZERO_POINT = 20.0
_REMOTE_SCATTER = 0.005

# Range of catalog magnitude the observations are built from. The lower end
# keeps every instrumental magnitude inside the -20 to -3 window the fit
# accepts, and the span is wide enough that the terms of the transform can be
# told apart from each other -- real stars' colors correlate with their
# magnitudes, so a narrow range is how a fit to a real field goes degenerate.
_REMOTE_MAG_RANGE = (10.5, 16.5)

# Most stars the query returns are thrown away, evenly, to keep the fit quick
# without narrowing the magnitude range sampling every Nth row preserves.
_REMOTE_MAX_STARS = 150

# Fewest stars, and fewest calibrated magnitudes, worth calling a result.
_REMOTE_MIN_STARS = 20

# The passbands and band transform each catalog is fetched with, mirroring
# what `transform_to_catalog` asks for. Both native and transformed bands have
# to be requested: `passband_columns` needs the native ones to transform from.
_REMOTE_CATALOGS = {
    "apass_dr9": (
        apass_dr9,
        ["B", "V", "R", "I", "SR", "SG", "SI"],
        transform_apass_bands,
    ),
    "refcat2": (refcat2, ["B", "V", "R", "I"], transform_refcat2_bands),
}

# `ValueError` is in SERVER_DOWN_ERRORS, because the GAIA aperture service
# reports failure that way, but `transform_to_catalog` also raises it for its
# own reasons -- so xfailing on it around the call under test would turn a
# real failure into a pass. The query inside that call is the only part that
# can find the server down, and it cannot fail with a ValueError.
_QUERY_DOWN_ERRORS = tuple(
    error for error in SERVER_DOWN_ERRORS if error is not ValueError
)


@pytest.mark.remote_data
@pytest.mark.parametrize("cat_name", sorted(_REMOTE_CATALOGS))
def test_transform_to_catalog_against_a_real_catalog(cat_name):
    fetch, passbands, transformer = _REMOTE_CATALOGS[cat_name]

    try:
        nearby = fetch(
            _REMOTE_FIELD_CENTER,
            radius=_REMOTE_OBSERVED_RADIUS,
            clip_by_frame=False,
            padding=0,
        )
    except _QUERY_DOWN_ERRORS as e:
        pytest.xfail(f"Vizier is down or misbehaving: {e}")

    stars = nearby.passband_columns(passbands=passbands, transformer=transformer)

    catalog_mag = _to_float_array(stars["mag_R"])
    no_catalog_mag = ~np.isfinite(catalog_mag)

    # Stars the catalog has no R magnitude for are kept on purpose -- they are
    # what this test exists to check -- and so are stars over a range of
    # magnitude wide enough to fit.
    in_range = (catalog_mag >= _REMOTE_MAG_RANGE[0]) & (
        catalog_mag <= _REMOTE_MAG_RANGE[1]
    )
    stars = stars[in_range | no_catalog_mag]
    if len(stars) > _REMOTE_MAX_STARS:
        stars = stars[:: int(np.ceil(len(stars) / _REMOTE_MAX_STARS))]

    catalog_mag = _to_float_array(stars["mag_R"])
    no_catalog_mag = ~np.isfinite(catalog_mag)

    # Only apass_dr9 is required to have stars with no R magnitude: its native
    # bands really are missing for some stars, and those stars are what the
    # masked-row checks at the bottom exist for. refcat2's R is transformed
    # from Sloan bands the catalog is essentially complete in, so this field
    # -- and perhaps any field -- has no refcat2 star without one; the checks
    # below then check nothing rather than failing a premise the catalog
    # cannot meet.
    if cat_name == "apass_dr9":
        assert no_catalog_mag.any(), (
            "No stars with missing catalog magnitudes in the selected field. "
            "Either the band transform stopped producing masked catalog "
            "entries or the test field needs re-choosing."
        )

    assert np.isfinite(catalog_mag).sum() >= _REMOTE_MIN_STARS, (
        f"only {np.isfinite(catalog_mag).sum()} usable {cat_name} stars near "
        f"{_REMOTE_FIELD_CENTER.to_string('hmsdms')}"
    )

    # A star with no catalog magnitude was still observed, so it needs an
    # instrumental magnitude like any other; the middle of the field's range
    # will do, since nothing here depends on its value.
    instrumental = catalog_mag - _REMOTE_ZERO_POINT
    instrumental[no_catalog_mag] = np.nanmedian(instrumental)

    observed = _generate_observed_table(
        u.Quantity(np.asarray(stars["ra"]), u.degree),
        u.Quantity(np.asarray(stars["dec"]), u.degree),
        instrumental,
        noise_sigma=_REMOTE_SCATTER,
        seed=680,
    )

    try:
        # Neither catalog supplies an error for the Johnson-Cousins R it
        # transforms into, which is the case issue #685 is about, so this
        # logs a message about the band every time until that is fixed.
        result = transform_to_catalog(
            observed, "R", obs_error_column="mag_error", cat_name=cat_name
        )
    except _QUERY_DOWN_ERRORS as e:
        pytest.xfail(f"Vizier is down or misbehaving: {e}")

    calibrated = np.ma.getdata(result["mag_cal"])
    finite = np.isfinite(calibrated)
    assert finite.sum() >= _REMOTE_MIN_STARS

    # The observations were built from the catalog's own magnitudes, so the
    # fit should recover the zero point they were built with and give every
    # star its catalog magnitude back.
    assert result["z"][0] == pytest.approx(_REMOTE_ZERO_POINT, abs=0.5)

    has_a_catalog_magnitude = finite & ~no_catalog_mag
    recovered = np.abs(
        calibrated[has_a_catalog_magnitude] - catalog_mag[has_a_catalog_magnitude]
    )
    # A percentile rather than a maximum: a real field contains variables and
    # blends, and one star matched to the wrong catalog entry a degree-wide
    # query turned up should not fail a test about the pipeline.
    assert np.percentile(recovered, 90) < 0.05

    # The fit's account of itself, which is only meaningful on data that
    # scatters -- hence the noise added above.
    for column in ("a_error", "c_error", "z_error", "fit_redchi"):
        assert np.isfinite(result[column][0]), column
        assert result[column][0] > 0, column

    # The point of the whole test. A star the catalog has no magnitude for
    # must come back with no magnitude, not with whatever the catalog's
    # missing-value convention happens to be -- and it is only the real
    # catalog and the real band transform that can say whether that survives.
    for column in ("mag_cal", "mag_cat"):
        # np.ma.getdata rather than the column itself: `np.isnan` of a
        # masked array is masked in turn, and `.all()` of that is True
        # whatever the numbers underneath are -- exactly the check that
        # passes when it should not.
        missing = np.ma.getdata(result[column])[no_catalog_mag]
        assert np.isnan(missing).all(), column
        for fill_value in (-999, 1e20, 0.0):
            assert not (missing == fill_value).any(), (column, fill_value)
