from __future__ import print_function, division
import numpy as np

from ccdproc import CCDData
from astropy.wcs import WCS

def convert_pixel_wcs(ccd_image, lon_or_ra, lat_or_dec, isPix=True):
    """
    Takes either pixel or World Coordinate System (RA/Dec)
    coordinates and converts to the other type.

    Parameters
    ----------
    ccd_image: CCDData object
        Image which has the coordinates to convert.

    lon_or_ra: numpy.ndarray (or float)
        An array of coordinates (or a single coordinate).
        Represents the lon/ra axis, depending on what is
        passed in.

    lat_or_dec: numpy.ndarray (or float)
        An array of coordinates (or a single coordinate).
        Represents the lat/dec axis, depending on what is
        passed in.

    isPix: Bool
        Used to determine what coordinates are being inputted.
        Assumes pixel coordinates by default.

    Returns
    -------
    Returns two numpy.ndarray objects, the first being lon/RA,
    and the second lat/Dec.

    """
    if isPix:
        return ccd_image.wcs.all_pix2world(lon_or_ra, lat_or_dec, 0)
    else:
        return ccd_image.wcs.all_world2pix(lon_or_ra, lat_or_dec, 0)