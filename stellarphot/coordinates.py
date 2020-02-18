def convert_pixel_wcs(ccd_image, lon_or_ra,
                      lat_or_dec, is_pix=True, origin=0):
    """
    Takes either pixel or World Coordinate System (RA/Dec)
    coordinates and converts to the other type.

    Parameters
    ----------
    ccd_image: CCDData object
        Image which has the coordinates to convert.

    lon_or_ra: `numpy.ndarray' (or float)
        An array of coordinates (or a single coordinate).
        Represents the lon/ra axis, depending on what is
        passed in.

    lat_or_dec: `numpy.ndarray' (or float)
        An array of coordinates (or a single coordinate).
        Represents the lat/dec axis, depending on what is
        passed in.

    is_pix: bool
        Used to determine what coordinates are being inputted.
        Assumes pixel coordinates by default.

    origin: int
        The coordinate in the upper left corner of the image.
        Generally, this is 1 in FITS and Fortran standards,
        or 0 in Numpy and C standards. Defaults to 0.

    Returns
    -------
    Returns two `numpy.ndarray' objects, the first being lon/RA,
    and the second lat/Dec.

    """
    if is_pix:
        return ccd_image.wcs.all_pix2world(lon_or_ra, lat_or_dec, origin)
    else:
        return ccd_image.wcs.all_world2pix(lon_or_ra, lat_or_dec, origin)
