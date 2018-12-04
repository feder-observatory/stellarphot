from __future__ import print_function, division, absolute_import
import numpy as np
from photutils import aperture_photometry, CircularAperture, CircularAnnulus
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NoOverlapError
from .coordinates import convert_pixel_wcs

__all__ = ['photutils_stellar_photometry',
           'detect_sources', 'faster_sigma_clip_stats',
           'find_too_close', 'clipped_sky_per_pix_stats',
           'add_to_photometry_table', 'photometry_on_directory']


def photutils_stellar_photometry(ccd_image, sources,
                                 aperture_radius, inner_annulus,
                                 outer_annulus, gain=1.0, N_R=0, N_dark_pp=0,
                                 reject_background_outliers=True):
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

    reject_background_outliers : bool, optional
        If ``True``, sigma clip the pixels in the annulus to reject outlying
        pixels (e.g. like stars in the annulus)

    Returns
    -------
    phot_table : `~astropy.table.Table`
        Astropy table with columns for flux, x/y coordinates of center,
        RA/dec coordinates of center, sky background per pixel,
        net flux, aperture and annulus radii used, and flux error.
    """

    # check that the outer radius is greater or equal the inner radius
    # for annulus
    if inner_annulus >= outer_annulus:
        raise ValueError("outer_annulus must be greater than inner_annulus")

    # check that the annulus inner radius is greater or equal
    # the aperture radius
    if aperture_radius >= inner_annulus:
        raise ValueError("inner_radius must be greater than aperture_radius")

    # Extract x,y coordinates from sources table, construct aperture and
    # annulus objects from coordinates, and perform aperture photometry
    coords = (sources['xcentroid'], sources['ycentroid'])
    apertures = CircularAperture(coords, aperture_radius)
    annulus = CircularAnnulus(coords, inner_annulus, outer_annulus)
    phot_table = aperture_photometry(ccd_image, apertures)
    phot_table_1 = aperture_photometry(ccd_image, annulus)

    # Obtain the local background/pixel and net flux between the aperture and
    # annulus objects
    n_pix_ap = apertures.area()
    n_pix_ann = annulus.area()

    if reject_background_outliers:
        annulus_masks = annulus.to_mask()
        bkgd_pp = []
        for mask in annulus_masks:
            annulus_data = mask.cutout(ccd_image)
            bool_mask = mask.array < 1
            # Only include whole pixels in the estimate of the background
            bkgd, _, _ = sigma_clipped_stats(annulus_data * mask,
                                             mask=bool_mask)
            bkgd_pp.append(bkgd)
        bkgd_pp = np.array(bkgd_pp)
    else:
        bkgd_pp = phot_table_1['aperture_sum'] / n_pix_ann

    net_flux = phot_table['aperture_sum'] - (n_pix_ap * bkgd_pp)
    phot_table['background_per_pixel'] = bkgd_pp
    phot_table['net_flux'] = net_flux

    # Return a columns with the aperture radius and
    # the inner/outer annulus radii
    phot_table['aperture_radius'] = \
        np.ones(len(phot_table['aperture_sum'])) * aperture_radius
    phot_table['inner_radius'] = \
        np.ones(len(phot_table['aperture_sum'])) * inner_annulus
    phot_table['outer_radius'] = \
        np.ones(len(phot_table['aperture_sum'])) * outer_annulus

    # Obtain RA/Dec coordinates and add them to table
    try:
        ra, dec = convert_pixel_wcs(ccd_image, coords[0], coords[1], 1)
        phot_table['RA_center'] = ra
        phot_table['Dec_center'] = dec
    except AttributeError:
        pass

    # Obtain flux error and add column to return table
    noise = np.sqrt(gain * net_flux + n_pix_ap * (1 + (n_pix_ap / n_pix_ann)) *
                    (gain * (bkgd_pp + N_dark_pp) + (N_R**2)))
    phot_table['aperture_sum_err'] = noise

    return phot_table



# coding: utf-8

# In[1]:


import numpy as np
import bottleneck as bn

from astropy.coordinates import SkyCoord
from astropy.table import vstack
from astropy import units as u
from astropy.time import Time

from ccdproc import ImageFileCollection

from photutils import (DAOStarFinder, CircularAperture, CircularAnnulus,
                       aperture_photometry, centroid_sources)


np.seterr(all='ignore')

# In[11]:


def faster_sigma_clip_stats(data, sigma=5, iters=5, axis=None):

    data = data.copy()
    for _ in range(iters):
        central = bn.nanmedian(data, axis=axis)
        try:
            central = central[:, np.newaxis]
        except (ValueError, IndexError, TypeError):
            pass

        std_dif = 1.4826 * bn.nanmedian(np.abs(data - central))

        clips = np.abs(data - central) / std_dif > sigma

        if np.nansum(clips) == 0:
            break
        data[clips] = np.nan
    return (bn.nanmean(data, axis=axis), bn.nanmedian(data, axis=axis),
            bn.nanstd(data, axis=axis))


def detect_sources(ccd, fwhm, thresh):

    men, med, std = faster_sigma_clip_stats(ccd.data, sigma=5)


    # This sets up the source detection. The FWHM turns out to be key...making it too small results in a single star being detected as two separate sources.
    #
    # Stars must be brighter than the threshold to count as sources. Making the number higher gives you fewer detected sources, lower gives you more. There is no "magic" number.

    # In[13]:


    dao = DAOStarFinder(threshold=10 * std, fwhm=8, exclude_border=True)


    # Actually detect the stars...

    # In[14]:


    stars = dao(ccd - med)
    return stars


def find_too_close(star_locs, aperture_rad):
    star_coords = SkyCoord(ra=star_locs[0], dec=star_locs[1],
                           frame='icrs', unit='degree')
    idxc, d2d, d3d = star_coords.match_to_catalog_sky(star_coords,
                                                      nthneighbor=2)
    return d2d < (aperture_rad * 2 * 0.563 * u.arcsec)


def clipped_sky_per_pix_stats(data, annulus, sigma=5, iters=5):
    # Get a list of masks from the annuli
    # Use the 'center' method because then pixels are either in or out. To use
    # 'partial' or 'exact' we would need to do a weighted sigma clip and
    # I'm not sure how to do that.
    masks = annulus.to_mask(method='center')

    anul_list = []
    for mask in masks:
        # Multiply the mask times the data
        to_clip = mask.multiply(data.data, fill_value=np.nan)
        anul_list.append(to_clip.flatten())
    # Convert the list to an array for doing the sigma clip
    anul_array = np.array(anul_list)
    # Turn all zeros into np.nan...
    anul_array[anul_array == 0] = np.nan
    avg_sky_per_pix, med_sky_per_pix, std_sky_per_pix = \
        faster_sigma_clip_stats(anul_array,
                                sigma=sigma,
                                iters=iters,
                                axis=1
                               )

    return (avg_sky_per_pix * data.unit, med_sky_per_pix * data.unit,
            std_sky_per_pix * data.unit)


# ### Add more columns to the data table

# In[27]:


def add_to_photometry_table(phot, ccd, annulus, apertures, fname='',
                            star_ids=None, gain=None):
    phot.rename_column('aperture_sum_0', 'aperture_sum')
    phot['aperture_sum'].unit = u.adu
    phot.rename_column('aperture_sum_1', 'annulus_sum')
    star_locs = ccd.wcs.all_pix2world(phot['xcenter'], phot['ycenter'], 0)
    star_coords = SkyCoord(ra=star_locs[0], dec=star_locs[1],
                           frame='icrs', unit='degree')
    phot['RA'] = star_coords.ra
    phot['Dec'] = star_coords.dec
    print('        ...calculating clipped sky stats')
    avg_sky_per_pix, med_sky_per_pix, std_sky_per_pix = \
        clipped_sky_per_pix_stats(ccd, annulus)
    print('        ...DONE calculating clipp sky stats')
    phot['sky_per_pix_avg'] = avg_sky_per_pix
    phot['sky_per_pix_med'] = med_sky_per_pix
    phot['sky_per_pix_std'] = std_sky_per_pix
    phot['aperture'] = apertures.r * u.pixel
    phot['aperture_area'] = apertures.area()  # * u.pixel * u.pixel
    phot['annulus_inner'] = annulus.r_in * u.pixel
    phot['annulus_outer'] = annulus.r_out * u.pixel
    phot['annulus_area'] = annulus.area()  # * u.pixel * u.pixel
    phot['exposure'] = [ccd.header['exposure']] * len(phot) * u.second
    phot['date-obs'] = [ccd.header['DATE-OBS']] * len(phot)
    night = Time(ccd.header['DATE-OBS'], scale='utc')
    night.format = 'mjd'
    phot['night'] = np.int(np.floor(night.value - 0.5))
    phot['aperture_net_flux'] = (phot['aperture_sum'] -
                                 (phot['aperture_area'] *
                                  phot['sky_per_pix_avg']))

    if gain is not None:
        phot['mag_inst_{}'.format(ccd.header['filter'])] = \
            (-2.5 * np.log10(gain * phot['aperture_net_flux'].value /
                             phot['exposure'].value))

    metadata_to_add = ['AIRMASS', 'FILTER']
    for meta in metadata_to_add:
        phot[meta.lower()] = [ccd.header[meta]] * len(phot)
    if fname:
        phot['file'] = fname
    if star_ids is not None:
        phot['star_id'] = star_ids


def photometry_on_directory(directory_with_images, object_of_interest,
                            star_locs, aperture_rad,
                            inner_annulus, outer_annulus,
                            max_adu, star_ids,
                            gain, read_noise, dark_current):
    ifc = ImageFileCollection(directory_with_images)
    phots = []
    missing_stars = []
    for a_ccd, fname in ifc.ccds(object=object_of_interest, return_fname=True):
        print('on image ', fname)
        try:
            # Convert RA/Dec to pixel coordinates for this image
            pix_coords = a_ccd.wcs.all_world2pix(star_locs[0], star_locs[1], 0)
        except AttributeError:
            print('    ....SKIPPING THIS IMAGE, NO WCS')
            continue

        xs, ys = pix_coords

        # Remove anything that is too close to the edges/out of frame
        padding = 3 * aperture_rad
        out_of_bounds = ((xs < padding) | (xs > (a_ccd.shape[0] - padding)) |
                         (ys < padding) | (ys > (a_ccd.shape[1] - padding)))
        in_bounds = ~out_of_bounds

        # Find centroids of each region around star that is in_bounds
        xs_in = xs[in_bounds]
        ys_in = ys[in_bounds]
        print('    ...finding centroids')
        try:
            xcen, ycen = centroid_sources(a_ccd.data, xs_in, ys_in,
                                          box_size=2 * aperture_rad + 1)
        except NoOverlapError:
            print('    ....SKIPPING THIS IMAGE, CENTROID FAILED')
            continue

        # Calculate offset between centroid in this image and the positions
        # based on input RA/Dec. Later we will set the magnitude of those with
        # large differences to an invalid value (maybe).
        center_diff = np.sqrt((xs_in - xcen)**2 + (ys_in - ycen)**2)

        # Set up apertures and annuli based on the centroids in this image.
        aps = CircularAperture((xcen, ycen), r=aperture_rad)
        anuls = CircularAnnulus((xcen, ycen), inner_annulus, outer_annulus)

        # Set any clearly bad values to NaN
        a_ccd.data[a_ccd.data > max_adu] = np.nan
        print('    ...doing photometry')
        # Do the photometry...
        pho = aperture_photometry(a_ccd.data, (aps, anuls),
                                  mask=a_ccd.mask, method='center')

        # We may have some stars we did not do photometry for because
        # those stars were out of bounds.
        # Add the ones we missed to the list of missing
        missed = star_ids[out_of_bounds]
        missing_stars.append(missed)

        # Add all the extra goodies to the table
        print('    ...adding extra columns')
        add_to_photometry_table(pho, a_ccd, anuls, aps,
                                fname=fname, star_ids=star_ids[in_bounds],
                                gain=1.47)
        # And add the final table to the list of tables
        phots.append(pho)

    # ### Combine all of the individual photometry tables into one

    all_phot = vstack(phots)

    # ### Eliminate any stars that are missing from one or more images
    #
    # This makes life a little easier later...

    uniques = set()
    for miss in missing_stars:
        uniques.update(set(miss))

    actually_bad = sorted([u for u in uniques if u in all_phot['star_id']])
    len(uniques), len(actually_bad)

    all_phot.add_index('star_id')
    if actually_bad:
        bad_rows = all_phot.loc_indices[actually_bad]
        try:
            bad_rows = list(bad_rows)
        except TypeError:
            bad_rows = [bad_rows]
        all_phot.remove_indices('star_id')
        all_phot.remove_rows(sorted(bad_rows))

    all_phot.remove_indices('star_id')

    snr = (gain * all_phot['aperture_net_flux'] /
           np.sqrt(gain * all_phot['aperture_net_flux'].value +
                   all_phot['aperture_area'] *
                   (1 + all_phot['aperture_area'] / all_phot['annulus_area']) *
                   (gain * all_phot['sky_per_pix_avg'].value +
                    gain * dark_current * all_phot['exposure'].value +
                    read_noise**2
                   )
                  ))

    all_phot['mag_error'] = 1.085736205 / snr

    return all_phot
