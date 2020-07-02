import numpy as np

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

from stellarphot.differential_photometry.aij_diff_mags import calc_aij_mags


def _raw_photometry_table():
    """
    Generate an input raw photometry table and expected flux ratios
    for use in tests.
    """

    # How about ten times...
    times = Time('2018-06-25T01:00:00', format='isot', scale='utc')
    times = times + np.arange(10) * 30 * u.second

    # and four stars
    star_ra = (250.0 * u.degree + np.arange(4) * 10 * u.arcmin).value
    star_dec = [45.0] * 4

    # Stars 2, 3 and 4 will be the comparison stars


    return times, star_ra, star_dec

