import bottleneck as bn
import numpy as np
from photutils import (DAOStarFinder, aperture_photometry, CircularAperture,
                       CircularAnnulus, extract_stars)
from photutils.centroids import centroid_sources

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.table import vstack
from astropy import units as u
from astropy.time import Time

from ccdproc import ImageFileCollection

from astropy.stats import sigma_clipped_stats
from astropy.nddata import NoOverlapError

from .coordinates import convert_pixel_wcs
from .source_detection import compute_fwhm

__all__ = ['photutils_stellar_photometry',
           'faster_sigma_clip_stats',
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
    coords = np.array([sources['xcentroid'], sources['ycentroid']]).T
    apertures = CircularAperture(coords, aperture_radius)
    annulus = CircularAnnulus(coords, inner_annulus, outer_annulus)
    phot_table = aperture_photometry(ccd_image, apertures)
    phot_table_1 = aperture_photometry(ccd_image, annulus)

    # Obtain the local background/pixel and net flux between the aperture and
    # annulus objects
    n_pix_ap = apertures.area
    n_pix_ann = annulus.area

    if reject_background_outliers:
        annulus_masks = annulus.to_mask()
        bkgd_pp = []
        for mask in annulus_masks:
            annulus_data = mask.cutout(ccd_image)
            bool_mask = mask.data < 1
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


def faster_sigma_clip_stats(data, sigma=5, iters=5, axis=None):
    """
    Calculate sigma clipped stats quickly using NaNs instead of masking
    and using bottleneck where possible.

    Parameters
    ----------

    data : numpy array
        The data to be clipped. *The data should have had masked values
        replaced with NaN prior to calling this function.*

    sigma : float, optional
        Number of standard deviations (estimated with the MAD) a point must
        be from the central value (median) to be rejected.

    iters : int, optional
        Number of sigma clipping iterations to perform. Fewer iterations than
        this may be performed because iterations stop when no new data is
        being clipped.

    axis : int, optional
        axis along which to perform the median.

    Returns
    -------
    mean, median, std : float or numpy array
        Clipped statistics; shape depends on the shape of the input.
    """
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


def find_too_close(star_locs, aperture_rad, pixel_scale=0.563):
    """
    Identify sources that are closer together than twice the aperture radius.

    Parameters
    ----------

    star_locs : tuple of numpy array
        The first entry in the tuple should be the right ascension of the
        sources, in degrees. The second should be the declination of
        the sources, in degrees.

    aperture_rad : int
        Radius of the aperture, in pixels.

    pixel_scale : float, optional
        Pixel scale of the image in arcsec/pixel

    Returns
    -------

    numpy array of bool
        Array the same length as the RA/Dec that is ``True`` where the sources
        are closer than two aperture radii, ``False`` otherwise.
    """
    star_coords = SkyCoord(ra=star_locs[0], dec=star_locs[1],
                           frame='icrs', unit='degree')
    idxc, d2d, d3d = star_coords.match_to_catalog_sky(star_coords,
                                                      nthneighbor=2)
    return d2d < (aperture_rad * 2 * 0.563 * u.arcsec)


def clipped_sky_per_pix_stats(data, annulus, sigma=5, iters=5):
    """
    Calculate sigma-clipped statistics on an annulus.

    Parameters
    ----------

    data : `astropy.nddata.CCDData`
        CCD image on which the annuli are defined.

    annulus : photutils annulus
        One or more annulus (of any shape) from photutils.

    sigma : float, optional
        Number of standard deviations from the central value a
        pixel must be to reject it.

    iters : int, optional
        Maximum number of sigma clip iterations to perform. Iterations stop
        automatically if no pixels are rejected.
    """
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


def add_to_photometry_table(phot, ccd, annulus, apertures, fname='',
                            star_ids=None, camera=None,
                            bjd_coords=None, observatory_location=None,
                            fwhm_by_fit=True):
    """
    Calculate several columns for photometry table.

    Parameters
    ----------

    phot : astropy.table.Table
        An astropy Table with raw photometry data in it (generated by
        `photutils.aperture_photometry`).

    ccd : astropy.nddata.CCDData
        Image on which photometry is being done.

    annulus : photutils annulus
        One or more annulus (of any shape) from photutils.

    apertures : photutils apertures
        One or more apertures (of any shape) from photutils.

    fname : str, optional
        Name of the image file on which photometry is being performed.

    star_ids : str or int, optional
        ID for each of the sources.

    gain : float, optional
        Gain, in electrons/ADU, of the camera that took the image. The gain
        is used in calculating the instrumental magnitude.
    """
    phot.rename_column('aperture_sum_0', 'aperture_sum')
    phot['aperture_sum'].unit = u.adu
    phot.rename_column('aperture_sum_1', 'annulus_sum')
    star_locs = ccd.wcs.all_pix2world(phot['xcenter'], phot['ycenter'], 0)
    star_coords = SkyCoord(ra=star_locs[0], dec=star_locs[1],
                           frame='icrs', unit='degree')
    phot['RA'] = star_coords.ra
    phot['Dec'] = star_coords.dec
    print('        ...calculating clipped sky stats')
    try:
        avg_sky_per_pix, med_sky_per_pix, std_sky_per_pix = \
            clipped_sky_per_pix_stats(ccd, annulus)
    except AttributeError:
        print('BAD ANNULUS!')
        avg_sky_per_pix, med_sky_per_pix, std_sky_per_pix = \
            np.nan, np.nan, np.nan
    print('        ...DONE calculating clipped sky stats')

    phot['sky_per_pix_avg'] = avg_sky_per_pix
    phot['sky_per_pix_med'] = med_sky_per_pix
    phot['sky_per_pix_std'] = std_sky_per_pix
    # Add width x/y:
    # Make small table with renamed columns
    fwhm_x, fwhm_y = compute_fwhm(ccd, phot, fit=fwhm_by_fit)

    # Set bad values to NaN now
    bad_fwhm = (fwhm_x < 0) | (fwhm_y < 0)
    fwhm_x[bad_fwhm] = np.nan
    fwhm_y[bad_fwhm] = np.nan

    phot['fwhm_x'] = fwhm_x
    phot['fwhm_y'] = fwhm_y
    phot['width'] = (fwhm_x + fwhm_y) / 2

    phot['aperture'] = apertures.r * u.pixel
    phot['aperture_area'] = apertures.area  # * u.pixel * u.pixel
    phot['annulus_inner'] = annulus.r_in * u.pixel
    phot['annulus_outer'] = annulus.r_out * u.pixel
    phot['annulus_area'] = annulus.area  # * u.pixel * u.pixel
    phot['exposure'] = [ccd.header['exposure']] * len(phot) * u.second
    phot['date-obs'] = [ccd.header['DATE-OBS']] * len(phot)
    night = Time(ccd.header['DATE-OBS'], scale='utc')
    night.format = 'mjd'
    phot['night'] = np.int(np.floor(night.value - 0.5))
    phot['aperture_net_flux'] = (phot['aperture_sum'] -
                                 (phot['aperture_area'] *
                                  phot['sky_per_pix_avg']))

    # This can happen, for example, when the object is faint
    # and centroiding is bad.
    bad_flux = phot['aperture_net_flux'] < 0
    all_bads = bad_flux | bad_fwhm

    phot['aperture_net_flux'][all_bads] = np.nan

    if observatory_location.lower() == 'feder':
        phot['BJD'] = find_times(phot['date-obs'][0], phot['exposure'][0],
                                 ra=bjd_coords.ra, dec=bjd_coords.dec)

    if camera is not None:
        phot['mag_inst'] = \
            (-2.5 * np.log10(camera.gain * phot['aperture_net_flux'].value /
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
                            camera,
                            bjd_coords=None,
                            observatory_location=None,
                            fwhm_by_fit=True):
    """
    Perform aperture photometry on a directory of images.

    Parameters
    ----------

    directory_with_images : str
        Folder containing the images on which to do photometry. Photometry
        will only be done on images that contain the ``object_of_interest``.

    object_of_interest : str
        Name of the object of interest. The only files on which photometry
        will be done are those whose header contains the keyword ``OBJECT``
        whose value is ``object_of_interest``.

    star_locs : tuple of numpy array
        The first entry in the tuple should be the right ascension of the
        sources, in degrees. The second should be the declination of
        the sources, in degrees.

    aperture_rad : int
        Radius of the aperture to use when performing photometry.

    inner_annulus : int
        Inner radius of annulus to use in for performing local sky
        subtraction.

    outer_annulus : int
        Outer radius of annulus to use in for performing local sky
        subtraction.

    max_adu : int
        Maximum allowed pixel value before a source is considered
        saturated.

    star_ids : array-like
        Unique identifier for each source in ``star_locs``.

    camera : `stellarphot.Camera` object
        Camera object which has gain, read noise and dark current set.

    gain : float
        Gain, in electrons/ADU, of the camera that took the image. The gain
        is used in calculating the instrumental magnitude.

    read_noise : float
        Read noise of the camera in electrons. Used in the CCD equation
        to calculate error.

    dark_current : float
        Dark current, in electron/sec. Used in the CCD equation
        to calculate error.
    """
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
        out_of_bounds = ((xs < padding) | (xs > (a_ccd.shape[1] - padding)) |
                         (ys < padding) | (ys > (a_ccd.shape[0] - padding)))
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

        # FWHM is typically 5-6 pixels. The center really shouldn't move
        # by more than that.
        too_much_shift = center_diff > 6

        xcen[too_much_shift] = xs_in[too_much_shift]
        ycen[too_much_shift] = ys_in[too_much_shift]

        # Set up apertures and annuli based on the centroids in this image.
        ap_locs = np.array([xcen, ycen]).T
        aps = CircularAperture(ap_locs, r=aperture_rad)

        anuls = CircularAnnulus(ap_locs, inner_annulus, outer_annulus)

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
                                camera=camera, bjd_coords=bjd_coords,
                                observatory_location=observatory_location,
                                fwhm_by_fit=fwhm_by_fit)

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

    gain = camera.gain

    noise = calculate_noise(gain=camera.gain, read_noise=camera.read_noise,
                            dark_current_per_sec=camera.dark_current,
                            flux=all_phot['aperture_net_flux'],
                            sky_per_pix=all_phot['sky_per_pix_avg'].value,
                            aperture_area=all_phot['aperture_area'],
                            annulus_area=all_phot['annulus_area'],
                            exposure=all_phot['exposure'].value,
                            include_digitization=False)

    snr = gain * all_phot['aperture_net_flux'] / noise

    all_phot['mag_error'] = 1.085736205 / snr
    all_phot['noise'] = noise
    # AstroImageJ includes a factor of gain in the noise. IMHO it is part of the
    # flux but, for convenience, here it is
    all_phot['noise-aij'] = noise / gain
    all_phot['snr'] = snr

    return all_phot


def calculate_noise(gain=1.0, read_noise=0.0, dark_current_per_sec=0.0,
                    flux=0.0, sky_per_pix=0.0,
                    aperture_area=0, annulus_area=0,
                    exposure=0,
                    include_digitization=False):

    try:
        no_annulus = (annulus_area == 0).all()
    except AttributeError:
        no_annulus = annulus_area == 0

    if no_annulus:
        area_ratio = aperture_area
    else:
        area_ratio = aperture_area * (1 + aperture_area / annulus_area)

    # Units in here are an absolute mess
    poisson_source = gain * flux

    try:
        poisson_source = poisson_source.value
    except AttributeError:
        pass

    sky = area_ratio * gain * sky_per_pix
    dark = area_ratio * dark_current_per_sec * exposure
    rn_error = area_ratio * read_noise ** 2

    digitization = 0.0

    if include_digitization:
        digitization = area_ratio * (gain * 0.289) ** 2

    return np.sqrt(poisson_source + sky + dark + rn_error + digitization)


def find_times(phot_column, exposure,
               ra=331.1170417, dec=81.5659444,
               latitude=46.86678, longitude=263.54672):
    """
    Returns a numpy array of barycentric julien date times

    Parameters
    ----------

    phot_column : astropy Table column or numpy array
        numpy array or column of observation dates from the photometry table

    exposure : float; optional
        exposure time in seconds

    RA : float; optional
        Right ascension in degree format, default is for TIC-470127886

    Dec : float; optional
        Declination  in degree format, default is for TIC-470127886

    latitude : float; optional
        latitude of the observatory, default is for Paul P. Feder Observatory

    longitude : float; optional
        longitude of the observatory, default is for Paul P. Feder Observatory


    Returns
    -------

    new_time : numpy array
        array of barycentric times by julien date

    """
    location = EarthLocation(lat=latitude, lon=longitude)

    times = Time(phot_column, scale='utc', format='isot', location=location)
    ip_peg = SkyCoord(ra=ra, dec=dec, unit='degree')
    ltt_bary = times.light_travel_time(ip_peg)
    times_tdb = times.tdb
    time_barycenter = times_tdb + ltt_bary

    #adjust to midpoint of exposure
    bary_time = time_barycenter + exposure / 2

    new_time = bary_time.jd

    return new_time


# ra=331.1170417, dec=81.5659444, latitude=46.86678, longitude=263.54672
def find_bjd_mid_exposure(utc_times, exposure=0,
                          coords=None, location=None):
    """
    Returns a numpy array of barycentric Julian date times

    Parameters
    ----------

    phot_column : `astropy.table.Column` of `astropy.times.Time`  or numpy array
        numpy array or column of observation dates/times.

    exposure : float; optional
        exposure time in seconds

    coords : `astropy.coordinates.SkyCoord`
        Ra/Dec of the object to be used for the light travel time
        calculation.

    location: `astropy.coordinates.EarthLocation`
        Location of the observatory.

    Returns
    -------

    new_time : numpy array
        array of barycentric times by Julian date

    """
    location = EarthLocation(lat=latitude, lon=longitude)
    ip_peg = SkyCoord(ra=[ra], dec=[dec], unit='degree')

    if coords is None or observatory_location is None:
        raise ValueError
    times = Time(phot_column, scale='utc', format='isot', location=location)
    ltt_bary = times.light_travel_time(coords)
    times_tdb = times.tdb
    time_barycenter = times_tdb + ltt_bary

    # adjust to midpoint of exposure
    bary_time = time_barycenter + exposure * u.second / 2

    return bary_time.jd
