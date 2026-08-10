import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table, vstack
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.exceptions import AstropyUserWarning

from .. import magnitude_transforms
from ..magnitude_transforms import (
    calibrated_from_instrumental,
    filter_transform,
    transform_to_catalog,
)


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


# Default zero point of the synthetic catalog below. The catalog magnitudes
# follow the fit model exactly, so a fit to them recovers whichever
# coefficients were used to build the catalog.
_FAKE_CATALOG_ZERO_POINT = 20.0


def _generate_fake_catalog(
    n_stars, a=0.0, b=0.0, c=0.0, d=0.0, z=_FAKE_CATALOG_ZERO_POINT
):
    """
    Generate a catalog whose magnitudes are an exact fit to the transform model.

    The catalog magnitudes are built by calling the production model,
    `~stellarphot.utils.magnitude_transforms.calibrated_from_instrumental`,
    rather than re-deriving the arithmetic, so the tests cannot drift away
    from the model actually being fit. A fit to the result recovers the
    coefficients passed in here exactly.

    Parameters
    ----------

    n_stars : int
        Number of stars to generate.

    a, b, c, d, z : float, optional
        Coefficients of the transform model used to build the catalog
        magnitudes. The defaults make the catalog magnitude the instrumental
        magnitude plus `_FAKE_CATALOG_ZERO_POINT`.

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

    # The color must not be a linear function of the instrumental magnitude.
    # If it is, the a, c and z terms of the model are exactly degenerate --
    # any combination that adds up to the same thing fits equally well -- and
    # no fitter can recover the individual coefficients. The generator is
    # seeded, so the colors are the same from one run to the next.
    color = np.random.default_rng(432).uniform(0.0, 1.0, size=n_stars)

    # The trailing ``+ instrumental`` is the fit_diff=True offset: the model is
    # fit to the difference between the catalog and instrumental magnitudes.
    cat_r = (
        calibrated_from_instrumental((instrumental, color), a, b, c, d, z)
        + instrumental
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

    mag_error : float, optional
        Uncertainty to give every instrumental magnitude.

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
            "mag_error": [mag_error] * n_stars,
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


def _run_transform_to_catalog(
    mocker, catalog, observed, obs_filter="R", cat_name="apass_dr9", **kwargs
):
    """
    Run ``transform_to_catalog`` against a synthetic catalog.

    Patching the catalog fetch keeps the test offline, so it does not need
    the ``remote_data`` marker.

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
        Name of the catalog to use. The function of the same name in
        `~stellarphot.utils.magnitude_transforms` is the one patched, so
        ``"refcat2"`` is mocked exactly the way ``"apass_dr9"`` is.

    **kwargs
        Passed on to `~stellarphot.utils.magnitude_transforms.transform_to_catalog`,
        overriding the defaults this helper supplies.

    Returns
    -------
    `astropy.table.Table`
        The observations with the calibrated magnitude, fit coefficient and
        matched-catalog columns added.
    """
    mocker.patch.object(magnitude_transforms, cat_name, return_value=catalog)

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


def _fit_a_catalog(mocker, n_stars=20, **catalog_coefficients):
    """
    Build a synthetic catalog and observations of it, then transform them.

    Parameters
    ----------

    mocker : `pytest_mock.MockerFixture`
        Fixture used to patch the catalog fetch.

    n_stars : int, optional
        Number of stars to generate.

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
    result = _run_transform_to_catalog(mocker, catalog, observed)

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


def test_transform_to_catalog_output_columns(mocker):
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
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)

    result = _run_transform_to_catalog(mocker, catalog, observed, in_place=True)

    assert "mag_cal" in observed.colnames
    assert result is observed


def test_transform_to_catalog_not_in_place_leaves_input_alone(mocker):
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)
    original_columns = set(observed.colnames)

    result = _run_transform_to_catalog(mocker, catalog, observed, in_place=False)

    assert set(observed.colnames) == original_columns
    assert "mag_cal" in result.colnames


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
    # Only the apass_dr9 branch of the catalog fetch has ever been exercised.
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
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)

    with pytest.raises((TypeError, ValueError), match="vary"):
        _run_transform_to_catalog(mocker, catalog, observed, vary=bad_vary)


def test_transform_to_catalog_warns_when_term_outside_expected(mocker):
    # A zero point outside the expected range used to rail at the bound and
    # report a confident wrong answer. It should now be fit freely and merely
    # warned about. See issue #601.
    true_zero_point = 25.0

    catalog, ra, dec, instrumental = _generate_fake_catalog(20, z=true_zero_point)
    observed = _generate_observed_table(ra, dec, instrumental)

    with pytest.warns(AstropyUserWarning, match="z"):
        result = _run_transform_to_catalog(mocker, catalog, observed)

    # The value is reported, not clamped to the top of the expected range.
    np.testing.assert_allclose(result["z"], true_zero_point, rtol=0, atol=1e-6)
    np.testing.assert_allclose(result["mag_cal"], catalog["mag_R"], rtol=0, atol=1e-6)


def test_transform_to_catalog_empty_expected_disables_check(mocker):
    # Warnings are errors in this test suite, so a warning here fails the test.
    true_zero_point = 25.0

    catalog, ra, dec, instrumental = _generate_fake_catalog(20, z=true_zero_point)
    observed = _generate_observed_table(ra, dec, instrumental)

    result = _run_transform_to_catalog(mocker, catalog, observed, expected={})

    np.testing.assert_allclose(result["z"], true_zero_point, rtol=0, atol=1e-6)


def test_transform_to_catalog_warns_when_fit_fails(mocker):
    # A fit that does not converge is not reliably reproducible, so mock one.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)

    failed_fit = mocker.MagicMock()
    failed_fit.success = False
    failed_fit.message = "the fitter gave up"
    mocker.patch.object(magnitude_transforms.lmfit, "minimize", return_value=failed_fit)

    with pytest.warns(AstropyUserWarning, match="did not succeed"):
        result = _run_transform_to_catalog(mocker, catalog, observed)

    assert np.isnan(result["mag_cal"]).all()


def test_transform_to_catalog_fit_diff_agrees(mocker):
    # Fitting the difference between the catalog and instrumental magnitudes
    # should give the same calibrated magnitudes as fitting the catalog
    # magnitude directly. The coefficients legitimately differ -- with
    # fit_diff=False the true value of a is 1 rather than 0 -- so only the
    # calibrated magnitudes are compared.
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
def test_transform_to_catalog_no_good_data_warns_and_nans(mocker, case):
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _make_unusable_observations(case, catalog, ra, dec, instrumental)

    with pytest.warns(AstropyUserWarning, match="image_1.fit"):
        result = _run_transform_to_catalog(mocker, catalog, observed)

    assert np.isnan(result["mag_cal"]).all()
    for term in ("a", "b", "c", "d", "z"):
        assert np.isnan(result[term]).all()


def test_transform_to_catalog_one_bad_image_does_not_poison_others(mocker):
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)

    observed = _combine_observed_tables(
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

    with pytest.warns(AstropyUserWarning, match="image_2.fit"):
        result = _run_transform_to_catalog(mocker, catalog, observed)

    good = result["file"] == "image_1.fit"

    np.testing.assert_allclose(
        result["mag_cal"][good], catalog["mag_R"], rtol=0, atol=1e-6
    )
    assert np.isnan(result["mag_cal"][~good]).all()


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
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)

    with pytest.warns(AstropyUserWarning, match="rror weighting"):
        result = _run_transform_to_catalog(
            mocker, catalog, observed, obs_error_column=None
        )

    assert "mag_cal_error" not in result.colnames
    np.testing.assert_allclose(result["mag_cal"], catalog["mag_R"], rtol=0, atol=1e-6)


def test_transform_to_catalog_passband_not_in_table_raises(mocker):
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


def test_transform_to_catalog_warns_when_outlier_cut_removes_everything(mocker):
    # Stars more than a magnitude from the median offset between the catalog
    # and instrumental magnitudes are dropped. Here every star is, which
    # leaves nothing to fit even though the data and the catalog matches are
    # individually fine.
    catalog, ra, dec, instrumental = _generate_fake_catalog(2)

    scattered = instrumental + np.array([2.5, -2.5])
    observed = _generate_observed_table(ra, dec, scattered)

    with pytest.warns(AstropyUserWarning, match="No good data"):
        result = _run_transform_to_catalog(mocker, catalog, observed)

    assert np.isnan(result["mag_cal"]).all()


def test_transform_to_catalog_non_numeric_existing_column_raises(mocker):
    # Values for rows in other passbands are kept by reading the column that
    # is already there, which cannot work if that column holds something other
    # than numbers.
    catalog, ra, dec, instrumental = _generate_fake_catalog(20)
    observed = _generate_observed_table(ra, dec, instrumental)
    observed["mag_cal"] = ["not a magnitude"] * len(observed)

    with pytest.raises(ValueError, match="'mag_cal' is already in the table"):
        _run_transform_to_catalog(mocker, catalog, observed)
