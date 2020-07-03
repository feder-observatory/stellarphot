import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u


def calc_aij_mags(star_data, comp_stars, in_place=True, index_column=None):
    """
    Calculate AstroImageJ-style flux ratios.

    Parameters
    ----------

    star_data : '~astropy.table.Table'
        Table of star data from a SINGLE observation image BECAUSE
        MATCHIGN BY RA/DEC WON't work otherwise

    comp_stars : '~astropy.table.Table'
        Table of known comparison stars in the field, given by AAVSO

    in_place : ``bool``, optional
        If ``True``, add new columns to input table. Otherwise, return
        new table with those columns added.

    index_column : ``str``, optional
        If provided, use this column as the unique identifier for the star.
        If not provided, a temporary one is generated.

    Returns
    -------

    `astropy.table.Table` or None
        The return type depends on the value of ``in_place``. If it is
        ``False``, then the new columns are returned as a separate table,
        otherwise the columns are simply added to the input table.
    """

    # Match comparison star list to instrumental magnitude information
    star_data_coords = SkyCoord(ra=star_data['RA'], dec=star_data['Dec'])

    comp_coords = SkyCoord(ra=comp_stars['RA'], dec=comp_stars['Dec'])

    index, d2d, _ = star_data_coords.match_to_catalog_sky(comp_coords)

    # Not sure this is really close enough for a good match...
    good = d2d < 1 * u.arcsec

    flux_column_name = 'aperture_net_flux'
    # Calculate comp star counts for each time

    # Make a small table with just counts and time for all of the comparison
    # stars.

    comp_fluxes = star_data['date-obs', flux_column_name][good]
    # print(comp_fluxes)

    comp_fluxes = comp_fluxes.group_by('date-obs')
    comp_totals = comp_fluxes.groups.aggregate(np.add)[flux_column_name]

    comp_total_vector = np.ones_like(star_data[flux_column_name])
    # print(comp_totals)

    # Calculate relative flux for every star

    # Have to remove the flux of the star if the star is a comparison
    # star.
    is_comp = np.zeros_like(star_data[flux_column_name])
    is_comp[good] = 1
    flux_offset = -star_data[flux_column_name] * is_comp

    # This seems a little hacky; there must be a better way
    for date_obs, comp_total in zip(comp_fluxes.groups.keys, comp_totals):
        this_time = star_data['date-obs'] == date_obs[0]
        comp_total_vector[this_time] *= comp_total

    relative_flux = star_data[flux_column_name] / (comp_total_vector + flux_offset)
    relative_flux = relative_flux.flatten()
    # Calculate relative flux error and SNR for each target

    return relative_flux
