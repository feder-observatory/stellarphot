from __future__ import print_function, division, absolute_import
import numpy as np
from astropy.table import Table, join
from photutils import aperture_photometry, CircularAperture, CircularAnnulus
from .coordinates import convert_pixel_wcs

__all__ = ['photutils_stellar_photometry']

def photutils_stellar_photometry(ccd_image, sources, aperture_radius, inner_annulus
                                 outer_annulus, gain):
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

    gain : float
        Gain of the CCD. In units of electrons per DN.

    N_R : float
        Read noise of the CCD in electrons per pixel.

    N_dark_pp : float
        Number of dark counts per pixel.

    Returns
    -------
    phot_table : `~astropy.table.Table`
        Astropy table with columns for flux, x/y coordinates of center,
        RA/dec coordinates of center, sky background per pixel,
        net flux, aperture and annulus radii used, and flux error.
    """

    # check that the outer radius is greater or equal the inner radius for annulus
    if inner_annulus >= outer_annulus:
        raise ValueError("outer_annulus must be greater than inner_annulus")

    # check that the annulus inner radius is greater or equal the aperture radius
    if aperture_radius >= inner_annulus:
        raise ValueError("inner_radius must be greater than aperture_radius")

    # Extract x,y coordinates from sources table, contstruct aperture and
    # annulus objects from coordinates, and peform aperture photometry
    coords = (sources['xcentroid'], sources['ycentroid'])
    apertures = CircularAperture(coords, aperture_radius)
    annulus = CircularAnnulus(coords, inner_annulus, outer_annulus)
    phot_table = aperture_photometry(ccd_image, apertures)
    phot_table_1 = aperture_photometry(ccd_image, annulus)

    # Obtain the local background/pixel and net flux between the aperture and annulus objects
    n_pix_ap = np.pi * (aperture_radius**2)
    n_pix_ann = np.pi * ((outer_annulus**2) - (inner_annulus**2))
    bkgd_pp = phot_table_1['aperture_sum'] / n_pix_ann
    net_flux = phot_table['aperture_sum'] -  (n_pix_ap * bkgd_pp)
    phot_table['background_per_pixel'] = bkgd_pp
    phot_table['net_flux'] = net_flux

    # Return a columns with the aperture radius and the inner/outer annulus radii
    phot_table['aperture_radius'] = np.ones(len(phot_table['aperture_sum'])) *  aperture_radius
    phot_table['inner_radius'] = np.ones(len(phot_table['aperture_sum'])) *  inner_annulus
    phot_table['outer_radius'] = np.ones(len(phot_table['aperture_sum'])) *  outer_annulus

    # Obtain RA/Dec coordinates and add them to table
    ra, dec = convert_pixel_wcs(ccd_image, coords[0], coords[1], 1)
    phot_table['RA_center'] = ra
    phot_table['Dec_center'] = dec

    # Obtain flux error and add column to return table
    noise = np.sqrt(gain*net_flux + n_pix_ap*(1 + (n_pix_ap/n_pix_ann))
			 *(gain*(bkgd_pp + N_dark_pp) + (N_R**2)))
    phot_table['aperture_sum_err'] = noise

    return phot_table
