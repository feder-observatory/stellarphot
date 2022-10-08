from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.utils.data import get_pkg_data_filename
import numpy as np
from .. import calc_multi_vmag, calc_vmag


def test_one_vmag():
    find_var_data = get_pkg_data_filename('data/variables.fits')
    var_stars = Table.read(find_var_data)
    find_star_data = get_pkg_data_filename('data/2014-12-29-ey-uma-9.fits')
    star_data = Table.read(find_star_data)
    find_comp_data = get_pkg_data_filename('data/comp_stars.fits')
    comp_stars = Table.read(find_comp_data)
    comp_stars['coords'] = SkyCoord(ra=comp_stars['ra'], dec=comp_stars['dec'], unit='degree')
    vmag, error = calc_vmag(var_stars, star_data, comp_stars, band='Rc', star_data_mag_column='mag_inst_R')
    np.testing.assert_almost_equal(vmag, 14.54827, decimal=5)
    np.testing.assert_almost_equal(error, 0.028685, decimal=5)


def test_multi_vmag():
    find_var_data = get_pkg_data_filename('data/variables.fits')
    var_stars = Table.read(find_var_data)
    find_star_data = get_pkg_data_filename('data/2014-12-29-ey-uma-9.fits')
    star_data = Table.read(find_star_data)
    find_comp_data = get_pkg_data_filename('data/comp_stars.fits')
    comp_stars = Table.read(find_comp_data)
    comp_stars['coords'] = SkyCoord(ra=comp_stars['ra'], dec=comp_stars['dec'], unit='degree')

    vmag, error = calc_vmag(var_stars, star_data, comp_stars, band='Rc', star_data_mag_column='mag_inst_R')
    del var_stars['coords']
    v_data = vstack([var_stars, var_stars])
    v_data['coords'] = SkyCoord(ra=v_data['RAJ2000'],
                                dec=v_data['DEJ2000'], unit='degree')
    v_table = calc_multi_vmag(v_data, star_data, comp_stars, band='Rc', star_data_mag_column='mag_inst_R')

    assert v_table['Mag'][0] == v_table['Mag'][1]
    assert v_table['StDev'][0] == v_table['StDev'][1]
