import re

import numpy as np

from astroquery.vizier import Vizier

from astropy.coordinates import SkyCoord
import astropy.units as units
from astropy.nddata.utils import block_reduce

__all__ = [
    'in_frame',
    'catalog_search',
    'catalog_clean',
    'find_apass_stars',
    'find_known_variables',
    'filter_catalog'
]


def in_frame(frame_wcs, coordinates, padding=0):
    """
    Given a WCS and list of coordinates check whether the coordinates
    are within the frame.

    Parameters
    ----------

    frame_wcs : astropy WCS object
        WCS for the image.
    coordinates : astropy.coordinates.SkyCoord
        Coordinate(s) whose position will be checked to see if they are in the
        field of view.
    padding : int, optional
        Coordinates need to be at least this many pixels in from the edge
        of the frame to be considered in the field of view. Default value
        is 0.

    Returns
    -------

    numpy.ndarray of bool
        One value for each input coordinate; values are ``True`` if the
        coordinate was in the field of view, ``False`` otherwise.

    """
    x, y = frame_wcs.all_world2pix(coordinates.ra, coordinates.dec, 0)
    in_x = (x >= padding) & (x <= frame_wcs.pixel_shape[0] - padding)
    in_y = (y >= padding) & (y <= frame_wcs.pixel_shape[1] - padding)
    return in_x & in_y


def catalog_search(frame_wcs, shape, desired_catalog,
                   ra_column='RAJ2000',
                   dec_column='DEJ2000',
                   radius=0.5,
                   clip_by_frame=True,
                   padding=100):
    """
    Return the items from catalog that are within the search radius and
    (optionally) within the field of view of a frame.

    Parameters
    ----------

    frame_wcs : astropy.wcs.WCS
        WCS of the image of interest.

    shape : tuple of int

    Description: This function takes coordinate data from an image and a
    catalog name and returns the positions of those stars.
    Preconditions:frame_wcs is a WCS object, shape is tuple, list or array of
    numerical values, desired_catalog is a string and radius is a numerical
    value.
    Postconditions:
    """
    rad = radius * units.deg
    # Find the center of the frame
    center_coord = frame_wcs.all_pix2world([[shape[1] / 2, shape[0] / 2]], 0)
    center = SkyCoord(center_coord, frame='icrs', unit='deg')

    # Get catalog via cone search
    Vizier.ROW_LIMIT = -1  # Set row_limit to have no limit
    cat = Vizier.query_region(center, radius=rad, catalog=desired_catalog)
    # Vizier always returns list even if there is only one element. Grab that
    # element.
    cat = cat[0]
    cat_coords = SkyCoord(ra=cat[ra_column], dec=cat[dec_column])
    if clip_by_frame:
        in_fov = in_frame(frame_wcs, cat_coords, padding=padding)
    else:
        in_fov = np.ones([len(cat_coords)], dtype=np.bool)
    x, y = frame_wcs.all_world2pix(cat_coords.ra, cat_coords.dec, 0)
    return (cat[in_fov], x[in_fov], y[in_fov])


def catalog_clean(catalog, remove_rows_with_mask=True,
                  **other_restrictions):
    """
    Return a catalog with only the rows that meet the criteria specified.

    Parameters
    ----------

    catalog : astropy.Table
        Table of catalog information. There are no restrictions on the columns.
    remove_rows_with_mask : bool, optional
        If ``True``, remove rows in which one or more of the values is masked.
    other_restrictions: dict, optional
        Key/value pairs in which the key is the name of a column in the
        catalog and the value is the criteria that values in that column
        must satisfy to be kept in the cleaned catalog. The criteria must be
        simple, beginning with a comparison operator and including a value.
        See Examples below.
    """

    comparisons = {
        '<': np.less,
        '=': np.equal,
        '>': np.greater,
        '<=': np.less_equal,
        '>=': np.greater_equal,
        '!=': np.not_equal
    }

    recognized_comparison_ops = '|'.join(comparisons.keys())
    keepers = np.ones([len(catalog)], dtype=np.bool)

    if remove_rows_with_mask and catalog.masked:
        for c in catalog.columns:
            keepers &= ~catalog[c].mask

    for column, restriction in other_restrictions.items():
        criteria_re = re.compile(r'({})([-+a-zA-Z0-9]+)'.format(recognized_comparison_ops))
        results = criteria_re.match(restriction)
        if not results:
            raise ValueError("Criteria {}{} not "
                             "understood.".format(column, restriction))
        comparison_func = comparisons[results.group(1)]
        comparison_value = results.group(2)
        new_keepers = comparison_func(catalog[column],
                                      np.float(comparison_value))
        keepers = keepers & new_keepers

    return catalog[keepers]


def find_apass_stars(image,
                     max_mag_error=0.05,
                     max_color_error=0.1):
    # use the catalog_search function to find the apass stars in the frame of the image read above
    apass, apass_x, apass_y = catalog_search(
        image.wcs, image.shape, 'II/336/apass9', 'RAJ2000', 'DEJ2000', 1, False)

    # Creates a boolean array of the apass stars that have well defined magnitudes and color
    apass_bright = (apass['e_r_mag'] < max_mag_error) & (
        apass['e_B-V'] < max_color_error)  # & (apass['u_e_r_mag'] == 0)

    # create new lists of apass stars and x y pixel coordinates using boolean array
    apass_in_bright, in_apass_x, in_apass_y = apass[
        apass_bright], apass_x[apass_bright], apass_y[apass_bright]

    return apass, apass_x, apass_y, apass_in_bright, in_apass_x, in_apass_y


def find_known_variables(image):
    # Get any known variable stars from a new catalog search of VSX
    vsx, vsx_x, vsx_y = catalog_search(
        image.wcs, image.shape, 'B/vsx/vsx', 'RAJ2000', 'DEJ2000')
    vsx_names = vsx['Name']  # Get the names of the variables
    return vsx, vsx_x, vsx_y, vsx_names


def filter_catalog(catalog, **kwd):
    """
    Filter catalog by key/value pairs in arguments and return bool
    area ``True`` where values in catalog meet the criteria.

    Parameters
    ----------

    catalog : astropy Table
        Table whose values are to be filtered.

    kwd : key/value pairs, e.g. ``e_r_mag=0.1``
        The key must be the name of a column and the value the
        *upper limit* on the acceptable values.
    """
    good_ones = np.ones(len(catalog), dtype='bool')

    for key, value in kwd.items():
        good_ones |= (catalog[key] <= value)

    return good_ones
