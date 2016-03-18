from __future__ import print_function, division, absolute_import
import numpy as np
from astropy import units as u
from astropy.stats import mad_std
from astropy.io import fits
from astropy.table import Table, join
from photutils import aperture_photometry, CircularAperture, CircularAnnulus
from photutils.background import Background


def photutils_stellar_photometry(ccd_image, sources, aperture_radius,
                                 annulus_radius, #coord_type = pixel,
                                 bkgd_est_fcn = mad_std):
    """
    Perform aperture photometry on an image, with a few options for estimating
    the local background from an annulus around the aperture.

    Parameters
    ----------
    ccd_image : CCDData object
        Image on which to perform aperture photometry.

    sources : astropy Table object
        Table of extracted sources. Assumed to be the output of
        photutils.daofind() source extraction function.

    aperture_radius : float
        Radius of aperture(s) in pixels. This is also used as
        inner radius for annulus object.

    annulus_radius : float (will possibly be made optional)
        Outer radius of the annulus in pixels.

    coord_type : ???
        Type of coordinates. (because sources is assumed to be the
        output of photutils.daofind() not sure if this is necessary?)

    bkgd_est_fcn : function, optional
        Function used to estimate the "typical" background pixel.
        Defaults to `astropy.stats.mad_std()`

    Returns
    -------
    phot_table : astropy Table
        Output of photutils.aperture_photometry() function with
        additional columns for RA/dec coordinates of center,
        sky background per pixel, flux, and flux error.
    """

    # NOTE: THIS MAY NEED TO BE BROKEN UP INTO SMALLER FUNCTIONS.

    #if coord_type is not pixel:
        # /*****  Call Stefan's WCS function to convert to pixels *****/

    #check that the outer radius is greater or equal the inner radius for annulus
    if aperture_radius > annulus_radius:
        raise ValueError("annulus_radius must be greater or equal apertue_radius")

    #Estimate the backroud noise, and subtract this from the image data
    bkgd_noise = bkgd_est_fcn(ccd_image)
    ccd_image -= bkgd_noise

    #Extract x,y coordinates from sources table, contstruct aperture and
    #annulus objects from coordinates, and peform aperture photometry
    coords = (sources['xcentroid'], sources['ycentroid'])
    apertures = CircularAperture(coords, aperture_radius)
    annulus = CircularAnnulus(coords, aperture_radius, annulus_radius)
    phot_table = aperture_photometry(image, apertures)

    # Subtract the background from the aperture flux.

    # Return a table that includes at least:
    #   + ra, dec of center
    #   + x, y of center
    #   + flux
    #   + sky background per pixel
    #   + aperture radius used
    #   + annulus radius used
    #   + flux error
    return phot_table
