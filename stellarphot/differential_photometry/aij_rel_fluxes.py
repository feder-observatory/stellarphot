import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u


def _add_in_quadrature(array):
    """
    Add an array of numbers in quadrature.
    """
    return np.sqrt((array**2).sum())


def calc_aij_relative_flux(star_data, comp_stars,
                           in_place=True, index_column=None):
    """
    Calculate AstroImageJ-style flux ratios.

    Parameters
    ----------

    star_data : '~astropy.table.Table'
        Table of star data from one or more images.

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
    error_column_name = 'noise-aij'
    # Calculate comp star counts for each time

    # Make a small table with just counts, errors and time for all of the comparison
    # stars.

    comp_fluxes = star_data['date-obs', flux_column_name, error_column_name][good]
    # print(comp_fluxes)

    comp_fluxes = comp_fluxes.group_by('date-obs')
    comp_totals = comp_fluxes.groups.aggregate(np.add)[flux_column_name]
    comp_errors = comp_fluxes.groups.aggregate(_add_in_quadrature)[error_column_name]

    comp_total_vector = np.ones_like(star_data[flux_column_name])
    comp_error_vector = np.ones_like(star_data[flux_column_name])
    # print(comp_totals)

    # Calculate relative flux for every star

    # Have to remove the flux of the star if the star is a comparison
    # star.
    is_comp = np.zeros_like(star_data[flux_column_name])
    is_comp[good] = 1
    flux_offset = -star_data[flux_column_name] * is_comp

    # This seems a little hacky; there must be a better way
    for date_obs, comp_total, comp_error in zip(comp_fluxes.groups.keys,
                                                comp_totals, comp_errors):
        this_time = star_data['date-obs'] == date_obs[0]
        comp_total_vector[this_time] *= comp_total
        comp_error_vector[this_time] = comp_error

    relative_flux = star_data[flux_column_name] / (comp_total_vector + flux_offset)
    relative_flux = relative_flux.flatten()

    rel_flux_error = (star_data[flux_column_name] / comp_total_vector *
                      np.sqrt((star_data[error_column_name] / star_data[flux_column_name])**2 +
                              (comp_error_vector / comp_total_vector)**2
                              )
                     )

    # Add these columns to table
    if not in_place:
        star_data = star_data.copy()

    star_data['relative_flux'] = relative_flux
    star_data['relative_flux_error'] = rel_flux_error
    star_data['relative_flux_snr'] = relative_flux / rel_flux_error

    # AIJ records the total comparison counts even though that total is used
    # only for the targets, not the comparison.
    star_data['comparison counts'] = comp_total_vector # + flux_offset
    star_data['comparison error'] = comp_error_vector

    return star_data
