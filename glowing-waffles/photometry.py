from __future__ import print_function, division, absolute_import
import numpy as np
from astropy import units as u
from astropy.stats import mad_std
from astropy.io import fits
from astropy.table import Table, join
from photutils import aperture_photometry, CircularAperture, CircularAnnulus
from coordinates.py import convert_pixel_wcs

__all__ = ['photutils_stellar_photometry']

def photutils_stellar_photometry(ccd_image, sources, aperture_radius, inner_annulus
                                 outer_annulus):
    """
    Perform aperture photometry on an image, with a few options for estimating
    the local background from an annulus around the aperture.

    Parameters
    ----------
    ccd_image : `~ccdproc.CCDData`
        Image on which to perform aperture photometry.

    sources : `~astropy.table.Table`
        Table of extracted sources. Assumed to be the output of
        `~photutils.daofind()` source extraction function.

    aperture_radius : float
        Radius of aperture(s) in pixels.

    inner_annulus : float
        Inner radius of the annulus in pixels.

    outer_annulus : float
        Outer radius of the annulus in pixels.

    Returns
    -------
    phot_table : `~astropy.table.Table`
        Output of `~photutils.aperture_photometry()` function with
        additional columns for RA/dec coordinates of center,
        sky background per pixel, flux, and flux error.
    """

    #if coord_type is not pixel:
        # /*****  Call Stefan's WCS function to convert to pixels *****/

    #check that the outer radius is greater or equal the inner radius for annulus
    if inner_annulus > outer_annulus:
        raise ValueError("annulus_radius must be greater or equal apertue_radius")

    #check that the annulus inner radius is greater or equal the aperture radius
    if aperture_radius > inner_annulus:
        raise ValueError("inner_radius must be greater or equal aperture_radius")

    #Extract x,y coordinates from sources table, contstruct aperture and
    #annulus objects from coordinates, and peform aperture photometry
    coords = (sources['xcentroid'], sources['ycentroid'])
    apertures = CircularAperture(coords, aperture_radius)
    annulus = CircularAnnulus(coords, inner_radius, outer_radius)
    phot_table = aperture_photometry(ccd_image, apertures)
    phot_table_1 = aperture_photometry(ccd_image, annulus)

    #Obtain the local background/pixel and net flux between the aperture and annulus objects
    n_pix_ap = 3.14 * (aperture_radius**2)
    n_pix_ann = 3.14 * ((outer_annulus**2) - (inner_annulus**2))
    bkgd_pp = phot_table_1['aperture_sum'] / n_pix_ann
    net_flux = phot_table['aperture_sum'] -  (n_pix_ap * bkgd_pp)
    phot_table['background_per_pixel'] = bkgd_pp
    phot_table['net_flux'] = net_flux

    #Return a columns with the aperture radius and the inner/outer annulus radii
    phot_table['aperture_radius'] = np.ones(len(phot_table['aperture_sum'])) *  aperture_radius
    phot_table['inner_radius'] = np.ones(len(phot_table['aperture_sum'])) *  inner_radius
    phot_table['outer_radius'] = np.ones(len(phot_table['aperture_sum'])) *  outer_radius

    #Obtain RA/Dec coordinates and add them to table
    ra, dec = convert_pixel_wcs(ccd_image, coords[0], coords[1], 1)
    phot_table['RA_center'] = ra
    phot_table['Dec_center'] = dec


    # Return a table that includes at least:
    #   + ra, dec of center
    #   + x, y of center
    #   + flux
    #   + sky background per pixel
    #   + aperture radius used
    #   + annulus radii used
    #   + net flux
    #   + flux error
    return phot_table
