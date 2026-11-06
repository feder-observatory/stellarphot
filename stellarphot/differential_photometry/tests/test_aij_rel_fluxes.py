from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.utils.exceptions import AstropyDeprecationWarning, AstropyUserWarning

from stellarphot import PhotometryData, SourceListData
from stellarphot.differential_photometry.aij_rel_fluxes import (
    add_relative_flux_column,
    calc_aij_relative_flux,
)


def _repeat(array, count):
    """
    Concatenate ``count`` copies of an array.

    Parameters
    ----------

    array : array-like
        The array to repeat.

    count : int
        Number of copies.

    Returns
    -------

    `numpy.ndarray`
        The copies, one after another.
    """
    return np.concatenate([array for _ in range(count)])


def _expected_at_all_times(expected, input_table):
    """
    Repeat per-star expected values once for each time in the input table.

    Parameters
    ----------

    expected : array-like
        One value per star, in star id order.

    input_table : `astropy.table.Table`
        Photometry table whose ``date-obs`` column sets the number of times.

    Returns
    -------

    `numpy.ndarray`
        ``expected`` tiled once per time, matching the row order of a table
        sorted by time then star id.
    """
    n_times = len(np.unique(input_table["date-obs"]))
    return _repeat(expected, n_times)


def _add_input_positions(input_table):
    """
    Add ``ra_input``/``dec_input`` columns that are copies of ``ra``/``dec``.

    Parameters
    ----------

    input_table : `astropy.table.Table`
        Photometry table to modify in place.

    Notes
    -----

    This is what the photometry table will have once ``ra``/``dec`` become
    the measured centroid positions (see #710).
    """
    input_table["ra_input"] = input_table["ra"].copy()
    input_table["dec_input"] = input_table["dec"].copy()


def _comp_stars_without_ids(comp_star):
    """
    Copy a comparison star table without its ``star_id`` column.

    Parameters
    ----------

    comp_star : `astropy.table.Table`
        Comparison star table with a ``star_id`` column.

    Returns
    -------

    `astropy.table.Table`
        Copy of ``comp_star`` without ``star_id``, which forces the
        deprecated matching by position.
    """
    comps = comp_star.copy()
    comps.remove_column("star_id")
    return comps


def _calc_with_positional_fallback(input_table, comp_star, **kwargs):
    """
    Call ``calc_aij_relative_flux`` through the deprecated position matching.

    Parameters
    ----------

    input_table : `stellarphot.PhotometryData`
        Photometry data.

    comp_star : `astropy.table.Table`
        Comparison star table; its ``star_id`` column is removed before the
        call.

    **kwargs
        Passed on to ``calc_aij_relative_flux``.

    Returns
    -------

    `stellarphot.PhotometryData`
        Whatever ``calc_aij_relative_flux`` returns.

    Notes
    -----

    The project runs with warnings as errors, so the deprecation warning
    must be caught here, and this also checks that it is issued.
    """
    with pytest.warns(AstropyDeprecationWarning, match="deprecated"):
        return calc_aij_relative_flux(
            input_table, _comp_stars_without_ids(comp_star), **kwargs
        )


def _spoil_last_comp_row(input_table, bad_thing):
    """
    Make the row for comparison star 4 at the last time bad in some way.

    Parameters
    ----------

    input_table : `stellarphot.PhotometryData`
        Photometry table, which is sorted in place by time then star id so
        that its last row is comparison star 4 at the last time.

    bad_thing : str
        One of ``"RA"`` (move the position by 3 arcsec), ``"NaN"`` (set the
        net counts to NaN) or ``"missing"`` (remove the row).

    Returns
    -------

    `astropy.table.Table`
        Copy of the row before it was modified.
    """
    input_table.sort(["date-obs", "star_id"])

    # Force a copy of this row so we have access to the original values
    last_one = Table(input_table[-1])

    if bad_thing == "RA":
        # "Jiggle" one of the stars by moving it by a few arcsec in one image.
        coord_inp = SkyCoord(
            ra=last_one["ra"][0], dec=last_one["dec"][0], unit=u.degree
        )
        coord_bad_ra = coord_inp.ra + 3 * u.arcsecond
        input_table["ra"][-1] = coord_bad_ra
    elif bad_thing == "NaN":
        input_table["aperture_net_cnts"][-1] = np.nan
    elif bad_thing == "missing":
        input_table.remove_row(-1)

    return last_one


def _check_last_comp_excluded(
    output_table, expected_flux, comp_star, last_one, bad_thing
):
    """
    Check the relative fluxes at the last time with comparison star 4 excluded.

    Parameters
    ----------

    output_table : `stellarphot.PhotometryData`
        Output of ``calc_aij_relative_flux``.

    expected_flux : array-like
        Expected relative flux of each star with all comparison stars in
        the comparison set.

    comp_star : `astropy.table.Table`
        Table of the three comparison stars.

    last_one : `astropy.table.Table`
        The row of comparison star 4 at the last time, as returned by
        `_spoil_last_comp_row`.

    bad_thing : str
        The ``bad_thing`` passed to `_spoil_last_comp_row`.
    """
    old_total_flux = comp_star["aperture_net_cnts"].sum()
    new_flux = old_total_flux - last_one["aperture_net_cnts"]
    # This works for target stars, i.e. those never in comparison set
    new_expected_flux = old_total_flux / new_flux * expected_flux

    # Oh wow, this is terrible....
    # Need to manually calculate for the only two that are still in comparison
    new_expected_flux[1] = (
        comp_star["aperture_net_cnts"][0] / comp_star["aperture_net_cnts"][1]
    )
    new_expected_flux[2] = (
        comp_star["aperture_net_cnts"][1] / comp_star["aperture_net_cnts"][0]
    )

    new_expected_flux[3] = expected_flux[3]
    if bad_thing == "NaN":
        new_expected_flux[3] = np.nan

    comparison_start = -4 if bad_thing != "missing" else -3
    np.testing.assert_allclose(
        new_expected_flux[:-comparison_start],
        output_table["relative_flux"][comparison_start:],
    )


def _raw_photometry_table():
    """
    Generate a photometry table and the relative fluxes expected from it.

    Returns
    -------

    expected_flux_ratios : `astropy.units.Quantity`
        Expected relative flux of each of the four stars.

    expected_flux_error : `astropy.units.Quantity`
        Expected relative flux error of each of the four stars.

    photom : `stellarphot.PhotometryData`
        Photometry of four stars at ten times, with star ids 1 to 4. Stars 2,
        3 and 4 are the comparison stars.

    comp_star : `stellarphot.PhotometryData`
        Rows for the three comparison stars at the first time, which serve
        as the comparison star table. This is a slice of ``photom`` that
        shares its data, so assign new columns to it rather than modifying
        its columns in place.
    """

    n_times = 10
    n_stars = 4
    # How about ten times...
    times = Time("2018-06-25T01:00:00", format="isot", scale="utc")
    times = times + np.arange(n_times) * 30 * u.second
    times = times.value

    # and four stars
    star_ra = 250.0 * u.degree + np.arange(n_stars) * 10 * u.arcmin
    star_dec = np.array([45.0] * n_stars) * u.degree
    fluxes = np.array([10000.0, 20000, 30000, 40000]) * u.adu
    errors = (np.sqrt(fluxes.value) + 50) * u.electron
    star_ids = np.arange(1, 5, dtype="int")

    # Stars 2, 3 and 4 will be the comparison stars
    comp_stars = np.array([0, 1, 1, 1])
    expected_comp_fluxes = np.sum(fluxes[1:])

    # A comparison star is excluded from its own comparison ensemble, both
    # in the comparison total and in the comparison error, so the expected
    # values are different for each comparison star (see #605).
    comp_flux_offset = -comp_stars * fluxes
    expected_comp_flux_by_star = expected_comp_fluxes + comp_flux_offset
    expected_flux_ratios = fluxes / expected_comp_flux_by_star

    comp_error_total = np.sqrt((errors[1:] ** 2).sum())
    comp_error_by_star = np.sqrt(comp_error_total**2 - comp_stars * errors**2)

    expected_flux_error = (
        fluxes
        / expected_comp_flux_by_star
        * np.sqrt(
            errors**2 / fluxes**2
            + comp_error_by_star**2 / expected_comp_flux_by_star**2
        )
    )

    raw_table = Table(
        data=[
            np.sort(_repeat(times, n_stars)),
            _repeat(star_ra, n_times),
            _repeat(star_dec, n_times),
            _repeat(fluxes, n_times),
            _repeat(errors, n_times),
            _repeat(star_ids, n_times),
        ],
        names=[
            "date-obs",
            "ra",
            "dec",
            "aperture_net_cnts",
            "noise_electrons",
            "star_id",
        ],
        units=[
            None,
            u.degree,
            u.degree,
            u.adu,
            u.electron,
            None,
        ],
    )

    photom = PhotometryData(raw_table)
    # MAKE SURE to return photom, not raw_table, below to trigger the bug
    # https://github.com/feder-observatory/stellarphot/issues/421
    # in which, it turns out, QTable columns with units cannot be aggregated.
    return expected_flux_ratios, expected_flux_error, photom, photom[1:4]


def _expected_with_two_comps(input_table):
    """
    Relative fluxes expected when only stars 3 and 4 are comparison stars.

    Parameters
    ----------

    input_table : `stellarphot.PhotometryData`
        Photometry table from `_raw_photometry_table`.

    Returns
    -------

    `numpy.ndarray`
        Expected relative flux of every row of ``input_table``, in the order
        of a table sorted by time then star id.
    """
    fluxes = np.array([10000.0, 20000, 30000, 40000])
    comp_total = np.array([70000.0, 70000, 40000, 30000])
    return _expected_at_all_times(fluxes / comp_total, input_table)


@pytest.mark.parametrize("comp_ra_dec_have_units", [True, False])
@pytest.mark.parametrize("star_ra_dec_have_units", [True, False])
@pytest.mark.parametrize("in_place", [True, False])
def test_relative_flux_calculation(
    in_place, star_ra_dec_have_units, comp_ra_dec_have_units
):
    # In addition to checking the flux calculation values, this is also a regression
    # test for #421.
    expected_flux, expected_error, input_table, comp_star = _raw_photometry_table()

    # Try doing it all at once
    all_expected_flux = _expected_at_all_times(expected_flux, input_table)
    all_expected_error = _expected_at_all_times(expected_error, input_table)

    if not star_ra_dec_have_units:
        input_table["ra"] = input_table["ra"].data
        input_table["dec"] = input_table["dec"].data

    if not comp_ra_dec_have_units:
        comp_star["ra"] = comp_star["ra"].data
        comp_star["dec"] = comp_star["dec"].data

    output_table = calc_aij_relative_flux(input_table, comp_star, in_place=in_place)
    output_flux = output_table["relative_flux"]
    output_error = output_table["relative_flux_error"]

    np.testing.assert_allclose(output_flux, all_expected_flux)
    np.testing.assert_allclose(output_error, all_expected_error)
    if in_place:
        assert "relative_flux" in input_table.colnames
    else:
        assert "relative_flux" not in input_table.colnames


@pytest.mark.parametrize("bad_thing", ["NaN", "missing"])
def test_bad_comp_star(bad_thing):
    # A comparison star with NaN counts at, or missing from, one time is
    # excluded from the comparison set at every time.
    expected_flux, _, input_table, comp_star = _raw_photometry_table()
    last_one = _spoil_last_comp_row(input_table, bad_thing)

    output_table = calc_aij_relative_flux(input_table, comp_star, in_place=False)

    _check_last_comp_excluded(
        output_table, expected_flux, comp_star, last_one, bad_thing
    )


def test_one_drifted_frame_does_not_exclude_comp_star():
    # Comparison stars are matched to the photometry by star_id, and the
    # position is checked once per star, not once per frame, since the input
    # position is the same at every time. So a measured position that is off
    # at one time (there is no ra_input column here, so ra/dec is checked)
    # neither excludes the star nor aborts the calculation.
    expected_flux, _, input_table, comp_star = _raw_photometry_table()
    _spoil_last_comp_row(input_table, "RA")

    output_table = calc_aij_relative_flux(input_table, comp_star, in_place=False)

    np.testing.assert_allclose(
        output_table["relative_flux"],
        _expected_at_all_times(expected_flux, input_table),
    )


def test_positional_fallback_excludes_comp_star_on_position_mismatch():
    # The deprecated matching by position keeps its old behavior: a
    # comparison star whose position is off by more than 1.2 arcsec at even
    # one time is excluded from the comparison set at every time.
    expected_flux, _, input_table, comp_star = _raw_photometry_table()
    last_one = _spoil_last_comp_row(input_table, "RA")

    output_table = _calc_with_positional_fallback(
        input_table, comp_star, in_place=False
    )

    _check_last_comp_excluded(output_table, expected_flux, comp_star, last_one, "RA")


def test_positional_fallback_is_deprecated_and_matches_id_path():
    # Without a star_id column in the comparison star table the stars are
    # matched by position, with a deprecation warning, and on clean data the
    # result is the same as matching by star_id.
    _, _, input_table, comp_star = _raw_photometry_table()

    by_id = calc_aij_relative_flux(input_table, comp_star, in_place=False)
    by_position = _calc_with_positional_fallback(input_table, comp_star, in_place=False)

    for column in ["relative_flux", "relative_flux_error", "comparison counts"]:
        np.testing.assert_allclose(by_position[column], by_id[column])


def test_positional_fallback_reports_times_with_no_comp_stars():
    # In the deprecated matching by position, a comparison star whose position
    # is off at one time is excluded at every time. When that leaves some time
    # with no comparison star the error must say so, rather than claim that no
    # comparison star matched at all; here every comparison star matches at
    # all but one time.
    _, _, input_table, comp_star = _raw_photometry_table()
    input_table.sort(["date-obs", "star_id"])
    times = np.unique(input_table["date-obs"])

    for star_id, time in zip([2, 3, 4], times[:3], strict=True):
        row = (input_table["star_id"] == star_id) & (input_table["date-obs"] == time)
        input_table["ra"][row] = input_table["ra"][row] + 3 * u.arcsec

    with pytest.raises(RuntimeError, match="one or more times"):
        _calc_with_positional_fallback(input_table, comp_star, in_place=False)


def test_id_match_survives_measured_position_offsets():
    # Once ra/dec in the photometry table are the measured centroid positions
    # (#710) they may differ from the source list positions by more than the
    # old 1.2 arcsec match limit. The match is by star_id, so that must not
    # exclude any comparison star. The wrong-file check uses ra_input/dec_input,
    # which are the source list positions.
    expected_flux, expected_error, input_table, comp_star = _raw_photometry_table()
    _add_input_positions(input_table)
    input_table["ra"] = input_table["ra"] + 3 * u.arcsec
    input_table["dec"] = input_table["dec"] + 3 * u.arcsec

    output_table = calc_aij_relative_flux(input_table, comp_star, in_place=False)

    np.testing.assert_allclose(
        output_table["relative_flux"],
        _expected_at_all_times(expected_flux, input_table),
    )
    np.testing.assert_allclose(
        output_table["relative_flux_error"],
        _expected_at_all_times(expected_error, input_table),
    )


def test_nan_measured_position_does_not_exclude_comp_star():
    # A measured position that is NaN at one time (e.g. a rejected centroid)
    # must not exclude the comparison star; only the counts matter, and the
    # NaN-counts exclusion is separate from the position check.
    expected_flux, _, input_table, comp_star = _raw_photometry_table()
    _add_input_positions(input_table)
    input_table.sort(["date-obs", "star_id"])
    input_table["ra"][-1] = np.nan
    input_table["dec"][-1] = np.nan

    output_table = calc_aij_relative_flux(input_table, comp_star, in_place=False)

    np.testing.assert_allclose(
        output_table["relative_flux"],
        _expected_at_all_times(expected_flux, input_table),
    )


def test_only_one_input_position_column_falls_back_to_ra_dec():
    # The input positions are used only when both ra_input and dec_input are
    # present. With just one of them the check uses ra/dec, rather than
    # failing on the missing column.
    expected_flux, _, input_table, comp_star = _raw_photometry_table()
    # Garbage that would fail the check if it were used.
    input_table["ra_input"] = input_table["ra"] + 1 * u.degree

    output_table = calc_aij_relative_flux(input_table, comp_star, in_place=False)

    np.testing.assert_allclose(
        output_table["relative_flux"],
        _expected_at_all_times(expected_flux, input_table),
    )


def test_wrong_source_list_raises_error():
    # The star_ids match but one comparison star's position is 5 arcsec from
    # the input position in the photometry, so the comparison star table is
    # not from the source list the photometry was made from.
    _, _, input_table, comp_star = _raw_photometry_table()
    _add_input_positions(input_table)
    # Assign a new column rather than modifying in place because comp_star
    # is a slice of input_table and shares its data.
    offsets = np.zeros(len(comp_star)) * u.arcsec
    offsets[1] = 5 * u.arcsec
    comp_star["dec"] = comp_star["dec"] + offsets

    with pytest.raises(ValueError, match="does not appear to be") as excinfo:
        calc_aij_relative_flux(input_table, comp_star, in_place=False)

    # Star 3 is the second comparison star, the one that was moved. The
    # separation and the limit are formatted the same way.
    assert "star_id 3 (5 arcsec)" in str(excinfo.value)
    assert "more than 2 arcsec" in str(excinfo.value)


def test_coord_column_is_used_for_position_check():
    # When coord_column is given, those coordinates, not ra/dec, are the
    # ones checked against the input positions in the photometry.
    _, _, input_table, comp_star = _raw_photometry_table()
    coords = SkyCoord(ra=comp_star["ra"], dec=comp_star["dec"])
    offsets = np.zeros(len(comp_star)) * u.arcsec
    offsets[2] = 5 * u.arcsec
    comp_star["coord"] = SkyCoord(ra=coords.ra, dec=coords.dec + offsets)

    with pytest.raises(ValueError, match=r"star_id 4 \(5 arcsec\)"):
        calc_aij_relative_flux(
            input_table, comp_star, in_place=False, coord_column="coord"
        )


def test_position_check_limit_can_be_widened_or_skipped():
    # A 5 arcsec offset fails the default 2 arcsec check (see above) but
    # passes a wider limit, and None skips the check altogether.
    expected_flux, _, input_table, comp_star = _raw_photometry_table()
    comp_star["dec"] = comp_star["dec"] + 5 * u.arcsec
    expected = _expected_at_all_times(expected_flux, input_table)

    for limit in [10 * u.arcsec, None]:
        output_table = calc_aij_relative_flux(
            input_table, comp_star, in_place=False, position_check_limit=limit
        )
        np.testing.assert_allclose(output_table["relative_flux"], expected)


@pytest.mark.parametrize(
    "bad_limit,message",
    [
        (5, "must be an angle"),
        ("two", "must be an angle"),
        (0 * u.arcsec, "must be a positive, finite angle"),
        (-1 * u.arcsec, "must be a positive, finite angle"),
        ([1, 2] * u.arcsec, "must be a single angle"),
    ],
)
def test_invalid_position_check_limit_raises_error(bad_limit, message):
    _, _, input_table, comp_star = _raw_photometry_table()

    with pytest.raises(ValueError, match=f"position_check_limit {message}"):
        calc_aij_relative_flux(
            input_table, comp_star, in_place=False, position_check_limit=bad_limit
        )


def test_comp_star_with_nan_position_is_not_checked_but_warns():
    # A comparison star whose position is NaN in comp_stars (as it is for
    # every star in an x/y-only source list) cannot be checked. It stays in
    # the comparison set, and a warning says that it was not checked rather
    # than passing silently.
    expected_flux, _, input_table, comp_star = _raw_photometry_table()
    dec = comp_star["dec"].copy()
    dec[1] = np.nan
    comp_star["dec"] = dec

    with pytest.warns(AstropyUserWarning, match=r"star_id \[3\] could not be checked"):
        output_table = calc_aij_relative_flux(input_table, comp_star, in_place=False)

    np.testing.assert_allclose(
        output_table["relative_flux"],
        _expected_at_all_times(expected_flux, input_table),
    )


def test_comp_stars_without_coordinates_are_not_checked_but_warn():
    # A comparison star table with ids but no coordinates at all is enough to
    # match by, so it works, with a warning that the positions were not
    # checked. With the check turned off there is nothing to warn about.
    expected_flux, _, input_table, comp_star = _raw_photometry_table()
    comps = Table({"star_id": comp_star["star_id"]})
    expected = _expected_at_all_times(expected_flux, input_table)

    with pytest.warns(AstropyUserWarning, match="no coordinates"):
        output_table = calc_aij_relative_flux(input_table, comps, in_place=False)
    np.testing.assert_allclose(output_table["relative_flux"], expected)

    output_table = calc_aij_relative_flux(
        input_table, comps, in_place=False, position_check_limit=None
    )
    np.testing.assert_allclose(output_table["relative_flux"], expected)


def test_comp_star_with_no_photometry_warns():
    # A comparison star whose id is in comp_stars but not in the photometry
    # is not part of the comparison set. That must not be silent, because a
    # regenerated source list renumbers its stars, so this is how a wrong
    # comparison star table shows up when a few ids happen to line up.
    _, _, input_table, comp_star = _raw_photometry_table()
    comp_star["star_id"] = [100, 3, 4]

    with pytest.warns(AstropyUserWarning, match=r"no photometry: \[100\]"):
        output_table = calc_aij_relative_flux(input_table, comp_star, in_place=False)

    np.testing.assert_allclose(
        output_table["relative_flux"], _expected_with_two_comps(input_table)
    )


def test_duplicate_comp_star_ids_raise_error():
    # A comparison star that appears twice in comp_stars has no sensible
    # meaning, so it is an error rather than a matter of which row wins.
    _, _, input_table, comp_star = _raw_photometry_table()
    comps = vstack([Table(comp_star), Table(comp_star[:1])])

    with pytest.raises(ValueError, match=r"more than one row for star_id \[2\]"):
        calc_aij_relative_flux(input_table, comps, in_place=False)


def test_comp_star_error_uses_self_excluded_ensemble():
    # Regression test for #605 -- the error of a comparison star must be
    # computed against the same ensemble as its relative flux, that is, with
    # the star itself excluded from both the comparison total counts and the
    # comparison error added in quadrature. Previously the error used the
    # full ensemble even though the flux excluded the star itself.
    _, _, input_table, comp_star = _raw_photometry_table()

    output = calc_aij_relative_flux(input_table, comp_star, in_place=False)

    # Hand-computed expectation for star 2, a comparison star whose
    # comparison ensemble is stars 3 and 4. These values match the ones
    # in _raw_photometry_table.
    flux_2 = 20000.0
    error_2 = np.sqrt(flux_2) + 50
    comp_total = 30000.0 + 40000.0
    comp_error = np.sqrt((np.sqrt(30000.0) + 50) ** 2 + (np.sqrt(40000.0) + 50) ** 2)
    expected_error_2 = (
        flux_2
        / comp_total
        * np.sqrt((error_2 / flux_2) ** 2 + (comp_error / comp_total) ** 2)
    )

    star_2 = output["star_id"] == 2
    np.testing.assert_allclose(
        output["relative_flux_error"][star_2].value, expected_error_2
    )
    # The SNR should be consistent with the flux and error columns.
    np.testing.assert_allclose(
        output["relative_flux_snr"][star_2],
        (output["relative_flux"][star_2] / output["relative_flux_error"][star_2]),
    )


def test_comp_star_with_zero_flux_is_counted():
    # Regression test for #605 -- exactly zero net counts is a legitimate
    # measured value (net counts can even be negative after sky subtraction).
    # The per-image consistency check used np.count_nonzero, so a comparison
    # star with exactly zero flux in one image was miscounted as a missing
    # star, raising a misleading "Different number of stars in comparison
    # sets" error.
    _, _, input_table, comp_star = _raw_photometry_table()
    input_table.sort(["date-obs", "star_id"])

    # Star 4, a comparison star, has exactly zero net counts in the
    # last image.
    input_table["aperture_net_cnts"][-1] = 0.0

    output = calc_aij_relative_flux(input_table, comp_star, in_place=False)

    # In the last image the comparison total is 20000 + 30000 + 0.
    last_image = output["date-obs"] == np.unique(output["date-obs"])[-1]
    target_star = last_image & (output["star_id"] == 1)
    zero_flux_comp = last_image & (output["star_id"] == 4)

    np.testing.assert_allclose(output["relative_flux"][target_star], 10000.0 / 50000.0)
    np.testing.assert_allclose(output["relative_flux"][zero_flux_comp], 0.0)
    # The error should be finite everywhere, including the zero-flux row.
    assert np.all(np.isfinite(output["relative_flux_error"]))


def test_no_matching_comp_stars_raises_error():
    # Regression test for #590 -- if none of the comparison stars match
    # the positions in the photometry data the result used to be a silent
    # "success" in which the comparison counts were set to 1, i.e. the
    # relative flux was just the net counts. It should raise an error instead.
    # This is the deprecated matching by position; with star_ids present a
    # position mismatch is a wrong-file error instead.
    _, _, input_table, comp_star = _raw_photometry_table()

    # Shift the comparison star positions by a degree so that none of them
    # match the positions in the photometry table.
    comp_star["ra"] = comp_star["ra"] + 1 * u.degree

    with pytest.raises(RuntimeError, match="No comparison stars"):
        _calc_with_positional_fallback(input_table, comp_star, in_place=False)


def test_no_matching_comp_star_ids_raises_error():
    # Same as above, but matching by star_id: none of the comparison star
    # ids are in the photometry data.
    _, _, input_table, comp_star = _raw_photometry_table()

    comp_star["star_id"] = comp_star["star_id"] + 100

    with pytest.raises(RuntimeError, match="No comparison stars"):
        calc_aij_relative_flux(input_table, comp_star, in_place=False)


def test_comp_stars_missing_at_one_time_raises_error():
    # Related to #590 -- if the comparison stars have no valid data at one
    # (or more) of the times, they are excluded as comparison stars at every
    # time. The error in that case should explain that, rather than say that
    # no comparison star positions matched, since the positions match just
    # fine at the other times.
    _, _, input_table, comp_star = _raw_photometry_table()
    input_table.sort(["date-obs", "star_id"])

    # Remove the comparison stars (stars 2, 3 and 4) from the last time,
    # keeping the target star at that time and all of the comparison stars
    # at the other times.
    last_time = np.unique(input_table["date-obs"])[-1]
    comps_at_last_time = (input_table["date-obs"] == last_time) & (
        input_table["star_id"] != 1
    )
    input_table = input_table[~comps_at_last_time]

    with pytest.raises(RuntimeError, match="one or more times"):
        calc_aij_relative_flux(input_table, comp_star, in_place=False)


# Run in a temporary directory because add_relative_flux_column writes its
# output file to the current working directory.
@pytest.mark.usefixtures("change_to_tmp_dir")
@pytest.mark.parametrize("bjd_already_present", [True, False])
def test_add_relative_flux_column(simple_photometry_data, bjd_already_present):
    # With bjd_already_present this is a regression test for #597 --
    # add_relative_flux_column used to raise a NameError when the input
    # photometry data already had a bjd column. Without it, the bjd column
    # should be computed and added to the output.
    phot_data = simple_photometry_data

    # The test data already has a bjd column, which triggered #597.
    assert "bjd" in phot_data.colnames
    if not bjd_already_present:
        del phot_data["bjd"]

    phot_file = Path("photometry.ecsv")
    phot_data.write(phot_file)

    # Make a source list from the photometry data itself so that the
    # comparison star coordinates are guaranteed to match. Use all but the
    # first star as comparison stars.
    comp_ids = sorted(set(phot_data["star_id"]))[1:]
    marker_name = [
        "APASS comparison" if star_id in comp_ids else "TESS Target"
        for star_id in phot_data["star_id"]
    ]
    source_table = Table(
        {
            "star_id": phot_data["star_id"],
            "ra": phot_data["ra"],
            "dec": phot_data["dec"],
            "xcenter": phot_data["xcenter"],
            "ycenter": phot_data["ycenter"],
            "marker name": marker_name,
        }
    )
    source_list = SourceListData(input_data=source_table)
    source_list_file = Path("source_list.ecsv")
    source_list.write(source_list_file)

    # With a pre-existing bjd column this used to raise a NameError because
    # the grouped table was only created when the bjd column was missing.
    add_relative_flux_column(phot_file, source_list_file, verbose=True)

    output_file = Path("photometry-relative-flux.ecsv")
    assert output_file.exists()

    output_data = PhotometryData.read(output_file)
    assert "bjd" in output_data.colnames
    assert np.all(np.isfinite(output_data["bjd"].jd))
    assert "relative_flux" in output_data.colnames
    assert np.all(np.isfinite(output_data["relative_flux"]))
    assert np.all(output_data["relative_flux"] > 0)
