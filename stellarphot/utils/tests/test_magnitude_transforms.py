import logging

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table, vstack
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.exceptions import AstropyUserWarning

from ...core import PhotometryData
from .. import magnitude_transforms
from ..magnitude_transforms import (
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

    Returns
    -------
    catalog : `_FakeCatalogTable`
        Catalog with columns ``ra``, ``dec``, ``mag_R`` and ``mag_I``.

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

    return catalog, ra, dec, instrumental


def _generate_observed_table(
    ra, dec, instrumental, file_name="image_1.fit", passband="R", mag_error=0.01
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
        Uncertainty of each instrumental magnitude. A single value is used for
        every star.

    Returns
    -------
    `astropy.table.Table`
        Observations of a single image, grouped by file name, with the
        ``file``, ``passband``, ``ra``, ``dec``, ``mag_inst`` and
        ``mag_error`` columns `transform_to_catalog` requires.
    """
    n_stars = len(instrumental)

    observed = Table(
        {
            "file": [file_name] * n_stars,
            "passband": [passband] * n_stars,
            "ra": ra.to_value("degree"),
            "dec": dec.to_value("degree"),
            "mag_inst": np.asarray(instrumental, dtype=float),
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

    call_kwargs = {
        "obs_error_column": "mag_error",
        "cat_filter": "R",
        "cat_color": ("R", "I"),
    }
    call_kwargs.update(kwargs)

    return transform_to_catalog(observed, obs_filter, cat_name=cat_name, **call_kwargs)


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

    # The distant star has no real match, so its calibrated
    # magnitude should be NaN.
    assert np.isnan(result["mag_cal"][-1])

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
    # enough. The fit is exact, so that magnitude is just the instrumental
    # magnitude plus the zero point.
    if expect_nan:
        assert np.isnan(result["mag_cal"][-1])
    else:
        assert result["mag_cal"][-1] == pytest.approx(
            offset_mag + _FAKE_CATALOG_ZERO_POINT, abs=1e-6
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


# The columns transform_to_catalog adds to the table it is given.
_TRANSFORM_OUTPUT_COLUMNS = {
    "mag_cal",
    "mag_cal_error",
    "a",
    "b",
    "c",
    "d",
    "z",
    "mag_cat",
    "color_cat",
}


def _fit_a_catalog(mocker, n_stars=20, vary=None, **catalog_coefficients):
    """
    Build a synthetic catalog and observations of it, then transform them.

    Parameters
    ----------

    mocker : `pytest_mock.MockerFixture`
        Fixture used to patch the catalog fetch.

    n_stars : int, optional
        Number of stars to generate.

    vary : sequence of str, optional
        Terms to fit. The default of `~stellarphot.utils.transform_to_catalog`
        is used if this is not given.

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
        Instrumental magnitude of each observation.
    """
    catalog, ra, dec, instrumental = _generate_fake_catalog(
        n_stars, **catalog_coefficients
    )
    observed = _generate_observed_table(ra, dec, instrumental)
    fit_kwargs = {} if vary is None else {"vary": vary}
    result = _run_transform_to_catalog(mocker, catalog, observed, **fit_kwargs)

    return result, catalog, instrumental


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


def test_transform_to_catalog_weights_by_inverse_error(mocker):
    # Every other fit here is of noiseless data with a single error value, and
    # for a residual that reaches exactly zero the weights make no difference
    # to where the fit lands -- so weighting by the error rather than by one
    # over the error would pass the rest of this file. Here one star is wrong
    # and says so with a large error: weighted correctly it is ignored,
    # weighted backwards it takes over (z comes out near 18.5, not 20).
    n_stars = 20

    catalog, ra, dec, instrumental = _generate_fake_catalog(n_stars)

    # Close enough to the other stars to survive the outlier cut, so only the
    # weighting can keep it from dragging the fit.
    observed_mags = instrumental.copy()
    observed_mags[0] -= 0.8

    errors = np.full(n_stars, 0.001)
    errors[0] = 10.0

    observed = _generate_observed_table(ra, dec, observed_mags, mag_error=errors)

    result = _run_transform_to_catalog(mocker, catalog, observed)

    np.testing.assert_allclose(result["z"], _FAKE_CATALOG_ZERO_POINT, rtol=0, atol=1e-3)
    np.testing.assert_allclose(result["a"], 0.0, rtol=0, atol=1e-3)
    np.testing.assert_allclose(result["c"], 0.0, rtol=0, atol=1e-3)


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


def test_transform_to_catalog_error_is_scaled_instrumental_error(mocker):
    # The calibrated error is currently just the instrumental error scaled by
    # the fitted linear term -- not a real propagation. See issue #674; this
    # test pins the current behavior so that changing it is a deliberate edit.
    a = 0.02
    mag_error = 0.01

    result, _, _ = _fit_a_catalog(mocker, a=a)

    np.testing.assert_allclose(
        result["mag_cal_error"], (1 + a) * mag_error, rtol=0, atol=1e-8
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
    np.testing.assert_allclose(
        with_diff["mag_cal_error"], without_diff["mag_cal_error"], rtol=0, atol=1e-8
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
    for term in ("a", "b", "c", "d", "z"):
        assert np.isnan(result[term]).all()


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
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental, passband="R")

    with pytest.raises(ValueError, match="No rows with passband 'B'"):
        _run_transform_to_catalog(mocker, catalog, observed, obs_filter="B")


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
    for term in ("a", "b", "c", "d", "z"):
        assert np.isnan(result[term]).all()


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


def test_transform_to_catalog_nans_error_for_unusable_input_error(mocker):
    # An error that is zero or negative is meaningless, and the fit already
    # excludes those stars. The calibrated error must not be finite for them
    # either: the AAVSO writer only turns non-finite errors into "na", so a
    # zero would be written into a submission as a real uncertainty -- and an
    # error of zero is infinite weight in any downstream average.
    n_bad = 2

    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)
    observed["mag_error"][0] = 0.0
    observed["mag_error"][1] = -0.02

    result = _run_transform_to_catalog(mocker, catalog, observed)

    assert np.isnan(result["mag_cal_error"][:n_bad]).all()
    np.testing.assert_allclose(result["mag_cal_error"][n_bad:], 0.01, rtol=0, atol=1e-8)

    # The stars are only left out of the *fit*. They still have perfectly good
    # instrumental magnitudes, so they still get calibrated magnitudes.
    np.testing.assert_allclose(result["mag_cal"], catalog["mag_R"], rtol=0, atol=1e-6)


def test_transform_to_catalog_default_catalog_columns(mocker):
    # Every call site passes cat_filter and cat_color explicitly, so nothing
    # ever executed the defaults. They name columns the catalog is indexed by
    # as mag_<name>, which means they have to be passband names.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)

    _patch_catalog_fetch(mocker, catalog)

    result = transform_to_catalog(observed, "R", obs_error_column="mag_error")

    np.testing.assert_allclose(result["mag_cal"], catalog["mag_R"], rtol=0, atol=1e-6)


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
        np.asarray(catalog["mag_R"])[n_unmeasured:],
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
