import numpy as np

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

__all__ = ['calc_multi_vmag', 'calc_vmag']


def calc_multi_vmag(var_stars, star_data, comp_stars, **kwd):
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

    kwd : Keyword arguments passed through to calc_vmag

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
        avg_vmag, error = calc_vmag(vsx, star_data, comp_stars, **kwd)
        vmag.append(avg_vmag)
        stdev.append(error)
    vmag_table = Table([name, vmag, stdev], names=('Name', 'Mag', 'StDev'))
    return vmag_table


def calc_vmag(var_stars, star_data, comp_stars, band=None,
              star_data_mag_column='mag_inst'):
    """
    Calculate the average magnitude and standard deviation of a variable star in field.

    Parameters
    ----------

    var_stars : '~astropy.table.Table'
        Table of variable stars known in field. It should contain a column
        called ``coords`` with the coordinates for each variable star as
        astropy ``SkyCoord`` objects.

    star_data : '~astropy.table.Table'
        Table of star data from observed image(s). One column should be named
        ``band`` and contain the passband in which observations were done.
        The column containing instrumental magnitudes is passed in with the
        argument ``star_data_mag_column``.

    comp_stars : '~astropy.table.Table'
        Table of known comparison stars in the field, given by AAVSO. The
        column containing the reference magnitudes for the filter specified
        by filter is passed in with the argument ``comp_star_mag_column``.

    band : str
        Filter/passband in which magnitude is being calculated.

    star_data_mag_column : str, optional
        Name of the column containing the instrumental magnitudes
        in ``star_data``.

    Returns
    -------
    avg : float
        Average magnitude for the variable star

    stdev : float
        Standard deviation for variable star values
    """

    if not band:
        raise ValueError("You must provide a band for this function.")

    # Match variable stars (essentially a list of targets) to instrumental
    # magnitude information.
    var_coords = var_stars['coords']
    star_data_coords = SkyCoord(ra=star_data['RA'], dec=star_data['Dec'])
    v_index, v_d2d, _ = var_coords.match_to_catalog_sky(star_data_coords)

    rcomps = comp_stars[comp_stars['band'] == band]

    # Match comparison star list to instrumental magnitude information
    try:
        comp_coords = rcomps['coords']
    except KeyError:
        comp_coords = SkyCoord(ra=rcomps['RAJ2000'], dec=rcomps['DEJ2000'])
    index, d2d, _ = comp_coords.match_to_catalog_sky(star_data_coords)
    good = d2d < 1 * u.arcsec
    good_index = index[good]

    vmag_image = star_data[v_index][star_data_mag_column]
    comp_star_mag = star_data[good_index][star_data_mag_column]
    a_index, a_d2d, _ = comp_coords.match_to_catalog_sky(comp_coords)
    good_a_index = a_index[good]
    accepted_comp = rcomps[good_a_index]['mag']
    new_mag = vmag_image - comp_star_mag + accepted_comp
    avg = np.nanmean(new_mag)
    stdev = np.nanstd(new_mag)

    return avg, stdev
