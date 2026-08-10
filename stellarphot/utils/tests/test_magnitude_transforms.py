import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.modeling import models
from astropy.table import Column, Table
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.exceptions import AstropyUserWarning

from .. import magnitude_transforms
from ..magnitude_transforms import (
    calculate_transform_coefficients,
    filter_transform,
    transform_magnitudes,
    transform_to_catalog,
)


def _generate_input_mags(n_stars):
    """
    Generate instrumental magnitudes in the range 10 to 15.

    The random number generator is seeded, so the magnitudes are the same
    from one run to the next.

    Parameters
    ----------

    n_stars : int
        Number of magnitudes to generate.

    Returns
    -------
    `astropy.table.Column`
        Column of instrumental magnitudes named ``instrumental``.
    """
    rg = np.random.default_rng(1024)
    input_mags = rg.integers(0, high=50, size=n_stars) / 10 + 10
    instr_mags = Column(name="instrumental", data=input_mags)
    return instr_mags


def _generate_catalog_mags(instrument_mags, color, model):
    """
    Generate catalog magnitudes from instrumental magnitudes.

    Parameters
    ----------

    instrument_mags : `astropy.table.Column`
        Instrumental magnitudes.

    color : `astropy.table.Column`
        Color of each star.

    model : `astropy.modeling.Model`
        Model relating the color to the difference between the catalog and
        instrumental magnitudes.

    Returns
    -------
    `astropy.table.Column`
        Catalog magnitudes.
    """
    return instrument_mags + model(color)


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


def _generate_tables(n_stars, mag_model):
    """
    Generate both tables needed for transforming magnitudes.

    Parameters
    ----------

    n_stars : int
        Number of stars to generate.

    mag_model : `astropy.modeling.Model`
        Model relating the color to the difference between the catalog and
        instrumental magnitudes.

    Returns
    -------
    instrumental : `astropy.table.Table`
        Instrumental magnitudes, with columns ``mag_inst_r``, ``ra`` and
        ``dec``.

    catalog_table : `astropy.table.Table`
        Catalog magnitudes, with columns ``r_mag``, ``RAJ2000``, ``DEJ2000``
        and ``B-V``.
    """
    instr_mags = _generate_input_mags(n_stars)

    # Set name to match default value in function.
    instr_mags.name = "mag_inst_r"

    # Set name to be default name for color.
    color = Column(name="B-V", data=np.linspace(0.0, 1.0, num=len(instr_mags)))
    catalog = _generate_catalog_mags(instr_mags, color, mag_model)

    # Again, set default name.
    catalog.name = "r_mag"

    # We'll use the same RA/Dec for the catalog and and the instrumental
    # magnitudes.
    ra, dec = _generate_star_coordinates(n_stars)

    # Instrumental magnitudes
    ra_col = Column(name="ra", data=ra)
    dec_col = Column(name="dec", data=dec)

    instrumental = Table([instr_mags, ra_col, dec_col])

    # Yes, these really do need to be renamed for the catalog table
    ra_col.name = "RAJ2000"
    dec_col.name = "DEJ2000"

    catalog_table = Table([catalog, ra_col, dec_col, color])
    return instrumental, catalog_table


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


@pytest.mark.parametrize("order", [1, 2, 5])
def test_catalog_same_as_input(order):
    # Check that we get the correct transform when catalog magnitudes
    # are identical to instrument magnitudes.
    instr_mags = Column(name="instrumental", data=[10, 12.5, 11])
    zero = models.Const1D(0.0)
    color = Column(name="color", data=[1.0] * len(instr_mags))
    catalog = _generate_catalog_mags(instr_mags, color, zero)

    # We expect these fits to be poorly conditioned because the two
    # sets of magnitudes are identical.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The fit may be poorly conditioned",
            category=AstropyUserWarning,
        )
        _, fit_model = calculate_transform_coefficients(
            instr_mags, catalog, color, order=order
        )

    assert len(fit_model.parameters) == order + 1
    assert all(fit_model.parameters == 0)


@pytest.mark.parametrize("order", [1, 2, 5])
def test_catalog_linear_to_input(order):
    # Check that we recover the correct relationship between
    # the catalog and instrumental magnitudes when the relationship
    # between the two is linear.
    n_stars = 100
    instr_mags = _generate_input_mags(n_stars)
    true_relationship = models.Polynomial1D(1, c0=0.5, c1=0.75)
    color = Column(name="color", data=np.linspace(0.0, 1.0, num=len(instr_mags)))
    catalog = _generate_catalog_mags(instr_mags, color, true_relationship)

    _, fit_model = calculate_transform_coefficients(
        instr_mags, catalog, color, order=order
    )
    assert len(fit_model.parameters) == order + 1
    assert np.abs(fit_model.c0 - true_relationship.c0) < 1e-7
    assert np.abs(fit_model.c1 - true_relationship.c1) < 1e-7

    if order >= 2:
        # Spot check some higher order terms -- they should be zero
        assert all(np.abs(fit_model.parameters[2:]) < 1e-7)


@pytest.mark.parametrize("order", [1, 2, 5])
def test_catalog_quadratic_to_input(order):
    # Check that we recover the correct relationship between
    # the catalog and instrumental magnitudes when the relationship
    # between the two is linear.
    n_stars = 100
    instr_mags = _generate_input_mags(n_stars)
    true_relationship = models.Polynomial1D(2, c0=0.5, c1=0.75, c2=0.25)
    color = Column(name="color", data=np.linspace(0.0, 1.0, num=len(instr_mags)))
    catalog = _generate_catalog_mags(instr_mags, color, true_relationship)
    _, fit_model = calculate_transform_coefficients(
        instr_mags, catalog, color, order=order
    )
    assert len(fit_model.parameters) == order + 1
    if order >= 2:
        # We expect a good fit in this case
        assert np.abs(fit_model.c0 - true_relationship.c0) < 1e-7
        assert np.abs(fit_model.c1 - true_relationship.c1) < 1e-7
        assert np.abs(fit_model.c2 - true_relationship.c2) < 1e-7
    else:
        # But a line just can't fit a quadratic that well
        assert np.abs(fit_model.c0 - true_relationship.c0) > 1e-7
        assert np.abs(fit_model.c1 - true_relationship.c1) > 1e-7
    if order >= 2:
        # Spot check some higher order terms -- they should be zero
        assert all(np.abs(fit_model.parameters[3:]) < 1e-7)


@pytest.mark.parametrize("faintest_magnitude", [None, 14])
def test_faintest_magnitude_has_effect(faintest_magnitude):
    # Check that the limit on magnitude when doing fits is respected.
    # We'll do this by setting up a linear relationship then
    # setting the catalog data fainter than the limit to nonsense.
    # Two outcomes we expect:
    #
    # 1. Without a limiting magnitude the fit should not be very good.
    # 2. With a limit the fit should be as good as it was before.
    n_stars = 100

    instr_mags = _generate_input_mags(n_stars)
    true_relationship = models.Polynomial1D(1, c0=0.5, c1=0.75)
    color = Column(name="color", data=np.linspace(0.0, 1.0, num=len(instr_mags)))
    catalog = _generate_catalog_mags(instr_mags, color, true_relationship)

    faint_ones = catalog >= 14
    assert faint_ones.sum() > 0
    assert faint_ones.sum() < n_stars / 2

    # Scramble the faint ones
    rg = np.random.default_rng(40482)
    catalog[faint_ones] = catalog[faint_ones] + 5 * rg.random(faint_ones.sum())

    _, fit_model = calculate_transform_coefficients(
        instr_mags,
        catalog,
        color,
        order=1,
        faintest_mag=faintest_magnitude,
        sigma=5000,  # So that nothing is clipped
    )

    if faintest_magnitude:
        assert np.abs(fit_model.c0 - true_relationship.c0) < 1e-7
        assert np.abs(fit_model.c1 - true_relationship.c1) < 1e-7
    else:
        assert np.abs(fit_model.c0 - true_relationship.c0) > 1e-2
        assert np.abs(fit_model.c1 - true_relationship.c1) > 1e-2


@pytest.mark.parametrize("order", [1, 2, 5])
def test_transform_magnitudes_identical_input(order):
    # Analogous to the test case for calculate_transform_coefficients
    # above where the input magnitudes are identical, except the input
    # objects have coordinates.
    n_stars = 100

    zero = models.Const1D(0.0)

    instrumental, catalog_table = _generate_tables(n_stars, zero)

    calib_mags, stars_with_match, transform = transform_magnitudes(
        instrumental, catalog_table, catalog_table, order=order
    )

    print(calib_mags)
    assert all(calib_mags == catalog_table["r_mag"])
    assert all(stars_with_match)
    assert len(transform.parameters) == order + 1
    assert all(transform.parameters == 0)


@pytest.mark.parametrize("order", [1, 2, 5])
def test_transform_magnitudes_identical_coord_quad_mags(order):
    # Analogous to the test case for calculate_transform_coefficients
    # above where the input magnitudes are identical, except the input
    # objects have coordinates.
    n_stars = 100

    true_relationship = models.Polynomial1D(2, c0=0.5, c1=0.75, c2=0.25)

    instrumental, catalog_table = _generate_tables(n_stars, true_relationship)

    calib_mags, stars_with_match, transform = transform_magnitudes(
        instrumental, catalog_table, catalog_table, order=order
    )

    assert all(stars_with_match)
    assert len(transform.parameters) == order + 1
    if order >= 2:
        # We expect a good fit in this case
        np.testing.assert_allclose(
            calib_mags, catalog_table["r_mag"], rtol=1e-7, atol=1e-7
        )
        assert np.abs(transform.c0 - true_relationship.c0) < 1e-7
        assert np.abs(transform.c1 - true_relationship.c1) < 1e-7
        assert np.abs(transform.c2 - true_relationship.c2) < 1e-7
    else:
        # But a line just can't fit a quadratic that well
        assert (np.abs(calib_mags - catalog_table["r_mag"]) > 1e-5).all()
        assert np.abs(transform.c0 - true_relationship.c0) > 1e-7
        assert np.abs(transform.c1 - true_relationship.c1) > 1e-7
    if order >= 2:
        # Spot check some higher order terms -- they should be zero
        assert all(np.abs(transform.parameters[3:]) < 1e-7)


def test_coordinate_mismatches():
    # Test that stars without close coordinate matches end up
    # marked appropriately.
    n_stars = 100

    true_relationship = models.Polynomial1D(1, c0=0.5, c1=0.75)

    instrumental, catalog_table = _generate_tables(n_stars, true_relationship)

    # Mess up the coordinates of half of the stars so that they don't match.
    catalog_table["RAJ2000"][50:] = catalog_table["RAJ2000"][50:] + 0.5 * u.degree

    calib_mags, stars_with_match, transform = transform_magnitudes(
        instrumental, catalog_table, catalog_table[:50], order=2
    )

    assert all(stars_with_match[:50])
    assert all(~stars_with_match[50:])


def test_coordinate_all_mismatches():
    # Test that when no stars match stuff goes badly.
    n_stars = 100

    true_relationship = models.Polynomial1D(1, c0=0.5, c1=0.75)

    instrumental, catalog_table = _generate_tables(n_stars, true_relationship)

    # Mess up the coordinates of half of the stars so that they don't match.
    catalog_table["RAJ2000"] = catalog_table["RAJ2000"] + 0.5 * u.degree

    # Since no stars match we expect a divide by zero in the fitting,
    # so we'll ignore that.
    #
    # We also expect the fit to be poorly conditioned in this case.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="The fit may be poorly conditioned",
            category=AstropyUserWarning,
        )
        calib_mags, stars_with_match, transform = transform_magnitudes(
            instrumental, catalog_table, catalog_table[:50], order=2
        )

    assert not any(stars_with_match)


class _FakeCatalogTable(Table):
    """
    Table that quacks like a stellarphot catalog just enough for
    transform_to_catalog, which calls passband_columns on the catalog.
    """

    def passband_columns(self, passbands=None, transformer=None):  # noqa: ARG002
        return self


# Zero point of the synthetic catalog below. The catalog magnitudes follow the
# fit model exactly with a=b=c=d=0, so this is the only non-zero coefficient.
_FAKE_CATALOG_ZERO_POINT = 20.0


def _generate_fake_catalog(n_stars):
    """
    Generate a catalog whose magnitudes are an exact fit to the transform model.

    Catalog magnitudes are generated with a=b=c=d=0 and
    z=_FAKE_CATALOG_ZERO_POINT, so a fit to them recovers those values exactly.

    Parameters
    ----------

    n_stars : int
        Number of stars to generate.

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
    ra, dec = _generate_star_coordinates(n_stars)

    instrumental = np.linspace(-10.0, -5.0, num=n_stars)
    color = np.linspace(0.0, 1.0, num=n_stars)
    cat_r = instrumental + _FAKE_CATALOG_ZERO_POINT

    catalog = _FakeCatalogTable(
        {
            "ra": ra,
            "dec": dec,
            "mag_R": cat_r,
            "mag_I": cat_r - color,
        }
    )

    return catalog, ra, dec, instrumental


def _generate_observed_table(ra, dec, instrumental):
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
            "file": ["image_1.fit"] * n_stars,
            "passband": ["R"] * n_stars,
            "ra": ra.to_value("degree"),
            "dec": dec.to_value("degree"),
            "mag_inst": instrumental,
            "mag_error": [0.01] * n_stars,
        }
    )

    return observed.group_by("file")


def _run_transform_to_catalog(mocker, catalog, observed):
    """
    Run ``transform_to_catalog`` against a synthetic catalog.

    Patching the catalog fetch keeps the test offline, so it does not need
    the ``remote_data`` marker.

    Parameters
    ----------

    mocker : `pytest_mock.MockerFixture`
        Fixture used to patch the `apass_dr9` catalog fetch.

    catalog : `_FakeCatalogTable`
        Catalog to use in place of the fetched one.

    observed : `astropy.table.Table`
        Observations to transform, grouped by file name.

    Returns
    -------
    `astropy.table.Table`
        The observations with the calibrated magnitude, fit coefficient and
        matched-catalog columns added.
    """
    mocker.patch.object(magnitude_transforms, "apass_dr9", return_value=catalog)

    return transform_to_catalog(
        observed,
        "R",
        obs_error_column="mag_error",
        cat_name="apass_dr9",
        cat_filter="R",
        cat_color=("R", "I"),
    )


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
    assert np.isnan(result["mag_inst_cal"][-1])

    # With the distant star excluded from the fit, the good stars are
    # an exact fit, so their calibrated magnitudes should match the
    # catalog. If the bad match leaks into the fit it drags the
    # coefficients away from the true values.
    np.testing.assert_allclose(
        result["mag_inst_cal"][:n_good],
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
        assert np.isnan(result["mag_inst_cal"][-1])
    else:
        assert result["mag_inst_cal"][-1] == pytest.approx(
            offset_mag + _FAKE_CATALOG_ZERO_POINT, abs=1e-6
        )

    # Either way the offset star is more than an arcsec from its catalog
    # position, so it should be left out of the fit, leaving the good stars
    # as an exact fit.
    np.testing.assert_allclose(
        result["mag_inst_cal"][:n_good],
        good_mags + _FAKE_CATALOG_ZERO_POINT,
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result["z"][:n_good], _FAKE_CATALOG_ZERO_POINT, rtol=0, atol=1e-6
    )
