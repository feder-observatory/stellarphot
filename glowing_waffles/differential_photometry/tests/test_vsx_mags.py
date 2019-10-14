from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.utils.data import get_pkg_data_filename
import numpy as np
from ..one_vsx_mag import calc_vmag

def test_one_vmag():
    find_var_data = get_pkg_data_filename('data/variables.fits')
    var_stars = Table.read(find_var_data)
    find_star_data = get_pkg_data_filename('data/2014-12-29-ey-uma-9.fits')
    star_data = Table.read(find_star_data)
    find_comp_data = get_pkg_data_filename('data/comp_stars.fits')
    comp_stars = Table.read(find_comp_data)
    vmag, error = calc_vmag(var_stars, star_data, comp_stars)

    np.testing.assert_almost_equal(vmag, 11.07127, decimal = 5)
    np.testing.assert_almost_equal(error, 1.47238, decimal = 5)
