from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from ..one_vsx_mag import calc_vmag

def test_one_vmag():
    var_stars = Table.read('variables.fits')
    star_data = Table.read('2014-12-29-ey-uma-9.fits')
    comp_stars = Table.read('comp_stars.fits')
    vmag, error = calc_vmag(var_stars, star_data, comp_stars)

    assert vamg == 11.07127
    assert error == 1.47238
