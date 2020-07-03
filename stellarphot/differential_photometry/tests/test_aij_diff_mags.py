import numpy as np

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

from stellarphot.differential_photometry.aij_diff_mags import calc_aij_mags


def _repeat(array, count):
    return np.concatenate([array for _ in range(count)])


def _raw_photometry_table():
    """
    Generate an input raw photometry table and expected flux ratios
    for use in tests.
    """

    n_times = 10
    n_stars = 4
    # How about ten times...
    times = Time('2018-06-25T01:00:00', format='isot', scale='utc')
    times = times + np.arange(n_times) * 30 * u.second
    times = times.value

    # and four stars
    star_ra = 250.0 * u.degree + np.arange(n_stars) * 10 * u.arcmin
    star_dec = np.array([45.0] * n_stars) * u.degree
    fluxes = np.array([10000., 20000, 30000, 40000])

    # Stars 2, 3 and 4 will be the comparison stars
    comp_stars = np.array([0, 1, 1, 1])
    expected_comp_fluxes = np.sum(fluxes[1:])

    comp_flux_offset = - comp_stars * fluxes
    expected_flux_ratios = fluxes / (expected_comp_fluxes + comp_flux_offset)

    raw_table = Table(data=[np.sort(_repeat(times, n_stars)), _repeat(star_ra,  n_times),
                            _repeat(star_dec, n_times), _repeat(fluxes, n_times)],
                      names=['date-obs', 'RA', 'Dec', 'aperture_net_flux'])

    return expected_flux_ratios, raw_table, raw_table[1:4]


def test_relative_flux_calculation():
    expected_flux, input_table, comp_star = _raw_photometry_table()
    # print(input_table)
    grouped_input = input_table.group_by('date-obs')

    # Try running the fluxes one exposure at a time
    # for one_time in grouped_input.groups:
    #     output = calc_aij_mags(one_time, comp_star)
    #     #print(one_time, comp_star, output)
    #     np.testing.assert_allclose(output, expected_flux)

    # Try doing it all at once
    n_times = len(np.unique(input_table['date-obs']))
    all_expected_flux = _repeat(expected_flux, n_times)

    output = calc_aij_mags(input_table, comp_star)
    print(all_expected_flux - output)
    np.testing.assert_allclose(output, all_expected_flux)
