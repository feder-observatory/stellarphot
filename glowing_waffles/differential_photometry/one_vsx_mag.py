from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
def calcVmag(var_stars, star_data, comp_stars):
    var_coords = var_stars['coords']
    star_data_coords = SkyCoord(ra=star_data['RA'], dec=star_data['Dec'])
    v_index, v_d2d, _ = var_coords.match_to_catalog_sky(star_data_coords)
    rcomps = comp_stars[comp_stars['band']=='Rc']
    comp_coords = SkyCoord(ra=rcomps['ra'], dec=rcomps['dec'])
    index, d2d, _ = comp_coords.match_to_catalog_sky(star_data_coords)
    good = d2d < 1 * u.arcsec
    good_d2d = d2d[good]
    good_comp = comp_coords[good]
    good_index = index[good]
    vmag_image = star_data[v_index]['mag_inst_R']
    comp_star_mag = ([])
    for star in good_index:
        comp_star_mag.append(star_data[star]['mag_inst_R'])
    a_index, a_d2d, _ = comp_coords.match_to_catalog_sky(comp_coords)
    good_a_index = a_index[good]
    accepted_comp = ([])
    for star in good_a_index:
        accepted_comp.append(comp_stars[star]['mag'])
    vmag = ([])
    item = 0
    for comp in comp_star_mag:
        new_mag = vmag_image - comp + accepted_comp[item]
        vmag.append(new_mag)
        item+=1
    tot = 0
    for item in vmag:
        tot+= item
    avg = tot/ len(vmag)

    return avg
