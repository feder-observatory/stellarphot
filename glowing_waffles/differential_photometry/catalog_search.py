from __future__ import print_function, division

import re

import numpy as np

from astroquery.vizier import Vizier

from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as units

from itertools import izip

from ccdproc import CCDData
__all__ = [

    'in_frame',
    'catalog_search',
    'catalog_clean',
]


def scale_and_downsample(data, downsample=4, min_percent=20, max_percent=99.5):
    if downsample > 1:
        scaled_data = block_reduce(scaled_data, block_size=(downsample, downsample))
    return scaled_data


def in_frame(frame_wcs, coordinates, padding=0):
    """
    Description:Check which of a set of coordinates are in the footprint of
    the WCS of an image
    Preconditions:
    Postconditions:
    """
    x, y = frame_wcs.all_world2pix(coordinates.ra, coordinates.dec, 0)
    in_x = (x >= padding) & (x <= frame_wcs._naxis1 - padding)
    in_y = (y >= padding) & (y <= frame_wcs._naxis2 - padding)
    return in_x & in_y


def catalog_search(frame_wcs, shape, desired_catalog,
                   ra_column='RAJ2000',
                   dec_column='DEJ2000',
                   radius=0.5,
                   clip_by_frame=True,
                   padding=100):
    """
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
        in_fov = in_frame(frame_wcs, cat_coords)
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

    for column, restriction in other_restrictions.iteritems():
        criteria_re = re.compile(r'({})(.+)'.format(recognized_comparison_ops))
        results = criteria_re.match(restriction)
        if not results:
            raise ValueError("Criteria {}{} not "
                             "understood.".format(column, restriction))
        comparison_func = comparisons[results.group(1)]
        comparison_value = results.group(2)
        new_keepers = comparison_func(catalog[column], np.float(comparison_value))
        keepers = keepers & new_keepers

    return catalog[keepers]
