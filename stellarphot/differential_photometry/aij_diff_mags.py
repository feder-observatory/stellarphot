import numpy as np

from astropy.coordinates import SkyCoord


def calc_aij_mags(star_data, comp_stars, in_place=True):
    """
    Calculate AstroImageJ-style flux ratios.

    Parameters
    ----------

    star_data : '~astropy.table.Table'
        Table of star data from observation image

    comp_stars : '~astropy.table.Table'
        Table of known comparison stars in the field, given by AAVSO

    in_place : ``bool``, optional
        If ``True``, add new columns to input table. Otherwise, return
        new table with those columns added.

    Returns
    -------

    `astropy.table.Table` or None
        The return type depends on the value of ``in_place``. If it is
        ``False``, then the new columns are returned as a separate table,
        otherwise the columns are simply added to the input table.
    """

    # Match comparison star list to instrumental magnitude information
    star_data_coords = SkyCoord(ra=star_data['RA'], dec=star_data['Dec'])

    comp_coords = SkyCoord(ra=comp_stars['ra'], dec=comp_stars['dec'])

    index, d2d, _ = comp_coords.match_to_catalog_sky(star_data_coords)

    # Not sure this is really close enough for a good match...
    good = d2d < 1 * u.arcsec
    good_index = index[good]


    flux_column_name = 'aperture_net_flux'
    # Calculate comp star counts for each time

    # Make a small table with just counts and time for all of the comparison
    # stars.

    comp_fluxes = star_data['date-obs', flux_column_name][good_index]
    comp_fluxes = comp_fluxes.group_by('date-obs')
    comp_totals = comp_fluxes.groups.aggregate(np.add)[flux_column_name]

    # Calculate relative flux for every star
    relative_flux = star_data[flux_column_name] / comp_totals

    # Calculate relative flux error and SNR for each target

    return relative_flux
