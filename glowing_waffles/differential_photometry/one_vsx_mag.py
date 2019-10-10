from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
def calc_vmag(var_stars, star_data, comp_stars):
    var_coords = var_stars['coords']
    star_data_coords = SkyCoord(ra=star_data['RA'], dec=star_data['Dec'])
    v_index, v_d2d, _ = var_coords.match_to_catalog_sky(star_data_coords)
    rcomps = comp_stars[comp_stars['band'] == 'Rc']
    comp_coords = SkyCoord(ra=rcomps['ra'], dec=rcomps['dec'])
    index, d2d, _ = comp_coords.match_to_catalog_sky(star_data_coords)
    good = d2d < 1 * u.arcsec
    good_d2d = d2d[good]
    good_comp = comp_coords[good]
    good_index = index[good]
    vmag_image = star_data[v_index]['mag_inst_R']
    comp_star_mag = star_data[good_index]['mag_inst_R']
    a_index, a_d2d, _ = comp_coords.match_to_catalog_sky(comp_coords)
    good_a_index = a_index[good]
    accepted_comp = comp_stars[good_a_index]['mag']
    vmag = ([])
    new_mag = vmag_image - comp_star_mag + accepted_comp
    avg = new_mag.mean()

    return avg
