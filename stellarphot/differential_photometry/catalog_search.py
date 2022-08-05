import re

import numpy as np

from astroquery.vizier import Vizier

from astropy.coordinates import SkyCoord
import astropy.units as units
from astropy.nddata import block_reduce

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


def catalog_search(frame_wcs_or_center, shape, desired_catalog,
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
    if isinstance(frame_wcs_or_center, SkyCoord):
        # Center was passed in, just use it.
        center = frame_wcs_or_center
        if clip_by_frame:
            raise ValueError('To clip entries by frame you must use '
                             'a WCS as the first argument.')
    else:
        # Find the center of the frame
        center = frame_wcs_or_center.pixel_to_world(shape[1] / 2,
                                                    shape[0] / 2)

    # Get catalog via cone search
    Vizier.ROW_LIMIT = -1  # Set row_limit to have no limit
    cat = Vizier.query_region(center, radius=rad, catalog=desired_catalog)
    # Vizier always returns list even if there is only one element. Grab that
    # element.
    cat = cat[0]
    cat_coords = SkyCoord(ra=cat[ra_column], dec=cat[dec_column])
    if clip_by_frame:
        in_fov = in_frame(frame_wcs_or_center, cat_coords, padding=padding)
    else:
        in_fov = np.ones([len(cat_coords)], dtype=np.bool)
    return cat[in_fov]


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


def find_apass_stars(image_or_center,
                     radius=1,
                     max_mag_error=0.05,
                     max_color_error=0.1):
    """
    Get APASS data from Vizer.

    Parameters
    ---------

    image_or_center : `astropy.nddata.CCDData` or `astropy.coordinates.SkyCoord`
        Either an image with a WCS (from which the RA/Dec will be extracted) or coordinate of
        the center.

    radius : float, optional
        Radius, in degrees, around which to search. Not needed if the first argument is an image.
    """
    if isinstance(image_or_center, SkyCoord):
        # Center was passed in, just use it.
        cen_wcs = image_or_center
        shape = None
    else:
        cen_wcs = image_or_center.wcs
        shape = image_or_center.shape
    # use the catalog_search function to find the apass stars in the frame of the image read above
    all_apass = catalog_search(cen_wcs, shape, 'II/336/apass9',
                               ra_column='RAJ2000', dec_column='DEJ2000', radius=radius,
                               clip_by_frame=False)

    # Creates a boolean array of the apass stars that have well defined
    # magnitudes and color.
    apass_lower_error = (all_apass['e_r_mag'] < max_mag_error) & (
        all_apass['e_B-V'] < max_color_error)

    # create new table  of apass stars that meet error restrictions
    apass_lower_error = all_apass[apass_lower_error]

    return all_apass, apass_lower_error


def find_known_variables(image):
    # Get any known variable stars from a new catalog search of VSX
    try:
        vsx = catalog_search(image.wcs, image.shape, 'B/vsx/vsx',
                             ra_column='RAJ2000', dec_column='DEJ2000')
    except IndexError:
        raise RuntimeError('No variables found in this field of view '
                           f'centered on {image.wcs}')
    return vsx


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
        print(key, value, catalog[key] <= value)
        good_ones &= (catalog[key] <= value)

    return good_ones
