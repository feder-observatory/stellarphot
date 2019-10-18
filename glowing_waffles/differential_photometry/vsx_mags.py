from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

__all__ = ['calc_multi_vmag', 'calc_vmag']


def calc_multi_vmag(var_stars, star_data, comp_stars):
    """
    Calculate the average magnitude and standard deviation of multiple variable
    stars in a field.

    Parameters
    ----------
    var_stars : '~astropy.table.Table'
        Table of variable stars known in field

    star_data : '~astropy.table.Table'
        Table of star data from observation image

    comp_stars : '~astropy.table.Table'
        Table of known comparison stars in the field, given by AAVSO

    Returns
    -------
    vmag_table : '~astropy.table.Table'
        Table including the name, avgerage magnitude, and standard deviation of
        multiple variable stars in a field.
    """
    name = []
    vmag = []
    stdev = []
    for vsx in var_stars:
        name.append(vsx['Name'])
        avg_vmag, error = calc_vmag(vsx, star_data, comp_stars)
        vmag.append(avg_vmag)
        stdev.append(error)
    vmag_table = Table([name, vmag, stdev], names=('Name', 'Mag', 'StDev'))
    return vmag_table


def calc_vmag(var_stars, star_data, comp_stars):
    """
    Calculate the average magnitude and standard deviation of a variable star in field.

    Parameters
    ----------
    var_stars : '~astropy.table.Table'
        Table of variable stars known in field

    star_data : '~astropy.table.Table'
        Table of star data from observation image

    comp_stars : '~astropy.table.Table'
        Table of known comparison stars in the field, given by AAVSO

    Returns
    -------
    avg : float
        Average magnitude for the variable star

    stdev : float
        Standard deviation for variable star values
    """
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
    accepted_comp = rcomps[good_a_index]['mag']
    new_mag = vmag_image - comp_star_mag + accepted_comp
    avg = new_mag.mean()
    stdev = new_mag.std()

    return avg, stdev
