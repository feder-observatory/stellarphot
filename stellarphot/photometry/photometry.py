import bottleneck as bn
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.nddata import CCDData, NoOverlapError
from astropy.stats import sigma_clipped_stats
from astropy.table import vstack, Column
from astropy.time import Time

from photutils.aperture import CircularAnnulus, aperture_photometry, CircularAperture
from photutils.centroids import centroid_sources

from scipy.spatial.distance import cdist

from ccdproc import ImageFileCollection

from stellarphot import Camera, PhotometryData, SourceListData
from .source_detection import compute_fwhm

__all__ = ['single_image_photometry',
           'photutils_stellar_photometry',
           'faster_sigma_clip_stats',
           'find_too_close', 'clipped_sky_per_pix_stats',
           'add_to_photometry_table', 'photometry_on_directory',
           'calculate_noise', 'find_bjd']


def single_image_photometry(ccd_image, sourcelist, camera, observatory_location,
                            aperture_radius, inner_annulus, outer_annulus,
                            shift_tolerance, max_adu,
                            include_dig_noise=True,
                            reject_too_close=True,
                            passband_map=None,
                            reject_background_outliers=True,
                            fwhm_by_fit=True, fname=None):
    """
    Perform aperture photometry on a single image, with an options for estimating
    the local background from sigma clipped stats of the counts in an annulus around
    the aperture.  If the sky positions of the sources are in the sourcelist and not
    the image positions (x,y), then the function will compute the image positions from
    the sky positions using the WCS of the image.  It assumes the input sky positions
    are in decimal degree units and in the ICRS frame.

    Parameters
    ----------
    ccd_image : `astropy.nddata.CCDData`
        Image on which to perform aperture photometry.  It should have AIRMASS and
        FILTER keywords in the header.  It must also have a WCS header associated with
        it if you want to use sky positions as inputs.

    sourcelist : `stellarphot.SourceList`
        Table of extracted sources with positions in terms of pixel coordinates OR
        RA/Dec coordinates. If both positions provided, pixel coordinates will be used.
        For RA/Dec coordinates to be used, the image must have a valid WCS.

    camera : `stellarphot.Camera`
        Camera object which has gain, read noise and dark current set.

    observatory_location : `astropy.coordinates.EarthLocation`
        Location of the observatory where the images were taken.  Used for calculating
        the BJD.

    aperture_radius : int
        Radius of aperture(s) in pixels.

    inner_annulus : int
        Inner radius of the annulus in pixels.

    outer_annulus : int
        Outer radius of the annulus in pixels.

    shift_tolerance : float
        If the image position needs to be computed using WCS, then computed
        impage positions are refined by afterward centroiding the sources.
        This reflects the tolerance in pixels for the shift between the
        the computed positions and the refined positions, in pixels.  Since we
        would not expect the shift to be more than the FWHM, a measured FWHM
        might be a good value to provide here.

    max_adu : float
        Maximum allowed pixel value before a source is considered saturated.

    reject_too_close : bool, optional (Default: True)
        If ``True``, any sources that are closer than twice the aperture radius
        are rejected.  If ``False``, all sources in field are used.

    include_dig_noise : bool, optional (Default: True)
        If ``True``, include the digitization noise in the calculation of the
        noise for each observation.  If ``False``, only the Poisson noise from
        the source and the sky will be included.

    passband_map: dict, optional (Default: None)
        A dictionary containing instrumental passband names as keys and
        AAVSO passband names as values. This is used to rename the passband
        entries in the output photometry table to be AAVSO standard versus
        whatever is in the source list.

    fwhm_by_fit : bool, optional (default: True)
        If ``True``, the FWHM will be calculated by fitting a Gaussian to
        the star. If ``False``, the FWHM will be calculated by finding the
        second moments of the light distribution. Default is ``True``.

    fname : str, optional
        Name of the image file on which photometry is being performed.


    Returns
    -------

    phot_table : `stellarphot.PhotometryData`
        Photometry data for all the locations in which aperture photometry was
        performed.  This may be a subset of the sources in the sourcelist if
        locations were too close to the edge of the image or to each other for
        successful apeture photometry.

    out_of_bounds : list
        This of the star_ids of the sources that fell outside the image and did not
        have photometry performed.
    """

    # Check that the input parameters are valid
    if not isinstance(ccd_image, CCDData):
        raise TypeError("ccd_image must be a CCDData object")
    if not isinstance(sourcelist, SourceListData):
        raise TypeError("sourcelist must be a SourceListData object")
    if not isinstance(camera, Camera):
        raise TypeError("camera must be a Camera object")
    if not isinstance(observatory_location, EarthLocation):
        raise TypeError("observatory_location must be a EarthLocation object")
    if inner_annulus >= outer_annulus:
        raise ValueError("outer_annulus must be greater than inner_annulus")
    if aperture_radius >= inner_annulus:
        raise ValueError("inner_radius must be greater than aperture_radius")
    if (shift_tolerance<=0):
        raise ValueError("shift_tolerance must be greater than 0 (should be on order "
                         "of FWHM)")
    if (max_adu<=0):
        raise ValueError("max_adu must be greater than 0")

    # Set hot pixels to NaN
    ccd_image.data[ccd_image.data > max_adu] = np.nan

    # Extract necessary values from sourcelist structure
    star_ids = sourcelist['star_id'].value
    xs = sourcelist['xcenter'].value
    ys = sourcelist['ycenter'].value
    ra = sourcelist['ra'].value
    dec = sourcelist['dec'].value
    src_cnt = len(sourcelist)

    # Reject sources that are within an aperture diameter of each other.
    too_close = find_too_close(sourcelist, aperture_radius,
                               pixel_scale=camera.pixel_scale)
    too_close_cnt = np.sum(too_close)
    non_overlap = ~too_close
    print(f"single_image_photometry: {too_close_cnt} of {src_cnt} sources within "
                "2 aperture radii of nearest neighbor", end="")
    if reject_too_close:
        # Remove sources too close together
        star_ids = star_ids[non_overlap]
        xs = xs[non_overlap]
        ys = ys[non_overlap]
        ra = ra[non_overlap]
        dec = dec[non_overlap]
        print(" ... removed them.")
    else:
        print(" ... keeping them.")

    # Remove all source positions too close to edges of image (where the annulus would
    # extend beyond the image boundaries).
    padding = outer_annulus
    out_of_bounds = ( (xs < padding) | (xs > (ccd_image.shape[1] - padding)) |
                    (ys < padding) | (ys  > (ccd_image.shape[0] - padding)) )
    in_bounds = ~out_of_bounds
    # Track the out of bounds sources
    out_of_bounds_sources = star_ids[out_of_bounds]
    # Remove sources too close to the edges
    star_ids = star_ids[in_bounds]
    xs = xs[in_bounds]
    ys = ys[in_bounds]
    ra = ra[in_bounds]
    dec = dec[in_bounds]
    in_cnt = np.sum(in_bounds)
    out_cnt = np.sum(out_of_bounds)
    print(f"single_image_photometry: {in_cnt} sources kept of {src_cnt} original "
          f"sources ({out_cnt} sources too close to image edge).")

    # If RA/Dec are available attempt to use them to determine the source positions
    if sourcelist.has_ra_dec and not sourcelist.has_x_y:
        try:
            imgpos = ccd_image.wcs.world_to_pixel(SkyCoord(ra, dec, unit=u.deg,
                                                           frame='icrs'))
            # Recentroid the sources using the WCS-derived positions (this is
            # particularly useful is processing multiple images of the same field
            # and just passing the same sourcelist when calling single_image_photometry
            # on each image.
            try:
                xcen, ycen = centroid_sources(ccd_image.data,
                                            imgpos[0],
                                            imgpos[1],
                                            box_size=2 * aperture_radius + 1)
                # Calculate offset between centroid in this image and the positions
                # based on input RA/Dec.
                center_diff = np.sqrt((xs - xcen)**2 + (ys - ycen)**2)

                # The center really shouldn't move more than about the fwhm, could
                # rework this in the future to use that instead.
                too_much_shift = center_diff > shift_tolerance

                # If the shift is too large, use the WCS-derived positions instead
                # (these sources are probably too faint for centroiding to work well)
                xcen[too_much_shift] = xs[too_much_shift]
                ycen[too_much_shift] = ys[too_much_shift]
                xs, ys = xcen, ycen
            except NoOverlapError:
                print("single_image_photometry: determining new centroid failed... "
                      "SKIPPING THIS IMAGE!")
                return None, None
        except AttributeError:
            # No WCS
            print("single_image_photometry: ccd_image must have a valid WCS to use "
                    "RA/Dec ... SKIPPING THIS IMAGE!")
            return None, None

    # Compute RA/Dec if not already provided
    if not sourcelist.has_ra_dec:
        try:
            skypos = ccd_image.wcs.pixel_to_world(xs, ys)
            ra = skypos.ra.value
            dec = skypos.dec.value
        except AttributeError:
            ra = [np.nan] * len(xs)
            dec = [np.nan] * len(ys)

    # Define apertures and annuli for the aperture photometry
    aper_locs = np.array([xs, ys]).T
    apers = CircularAperture(aper_locs, r=aperture_radius)
    anuls = CircularAnnulus(aper_locs, r_in=inner_annulus, r_out=outer_annulus)

    # Perform the aperture photometry
    photom = aperture_photometry(ccd_image.data, (apers, anuls),
                                mask=ccd_image.mask, method='center')

    # Add source ids to the photometry table
    photom['star_id'] = star_ids
    photom['ra'] = ra
    photom['ra'].unit = u.deg
    photom['dec'] = dec
    photom['dec'].unit = u.deg

    # Drop ID column from aperture_photometry()
    del photom['id']

    # Add various CCD image parameters to the photometry table
    if fname is not None:
        photom['file'] = fname
    photom['exposure'] = [ccd_image.header['exposure']] * len(photom) * u.second
    photom['date-obs'] = Column(data=Time([ccd_image.header['DATE-OBS']] * len(photom),
                                format='isot', scale='utc'), name='date-obs')
    metadata_to_add = ['AIRMASS', 'FILTER']
    for meta in metadata_to_add:
        try:
            photom[meta.lower()] = [ccd_image.header[meta]] * len(photom)
        except KeyError:
            print(f"single_image_photometry: {meta} not found in CCD image header ... "
                  "SKIPPING THIS IMAGE!")
            return None, None
    photom.rename_column('filter', 'passband')

    # Save aperture and annulus information
    photom.rename_column('aperture_sum_0', 'aperture_sum')
    photom.rename_column('aperture_sum_1', 'annulus_sum')
    photom['aperture'] = apers.r * u.pixel
    photom['aperture_area'] = apers.area * u.pixel * u.pixel
    photom['annulus_inner'] = anuls.r_in * u.pixel
    photom['annulus_outer'] = anuls.r_out * u.pixel
    photom['annulus_area'] = anuls.area  * u.pixel * u.pixel

    print('single_image_photometry: computing clipped sky stats ...')
    try:
        avg_sky_per_pix, med_sky_per_pix, std_sky_per_pix = \
                clipped_sky_per_pix_stats(ccd_image, anuls)
    except AttributeError:
        print('single_image_photometry: BAD ANNULUS!')
        avg_sky_per_pix, med_sky_per_pix, std_sky_per_pix = \
            np.nan, np.nan, np.nan
    photom['sky_per_pix_avg'] = avg_sky_per_pix
    photom['sky_per_pix_med'] = med_sky_per_pix
    photom['sky_per_pix_std'] = std_sky_per_pix
    print('                          ...DONE computing clipped sky stats')
    # Compute counts using clipped stats on sky per pixel
    photom['aperture_net_cnts'] = (photom['aperture_sum'].value -
                                    (photom['aperture_area'].value *
                                    photom['sky_per_pix_avg'].value))

    # Fit the FWHM of the sources
    print('single_image_photometry: fitting fwhm of all sources ...')
    fwhm_x, fwhm_y = compute_fwhm(ccd_image, photom, fit=fwhm_by_fit)
    print('                          ...DONE fitting fwhm of all sources')

    # Deal with bad FWHM values
    bad_fwhm = (fwhm_x<1) | (fwhm_y<1) # Set bad values to NaN now
    fwhm_x[bad_fwhm] = np.nan
    fwhm_y[bad_fwhm] = np.nan
    photom['fwhm_x'] = fwhm_x * u.pixel
    photom['fwhm_y'] = fwhm_y * u.pixel
    photom['width'] = ((fwhm_x + fwhm_y) / 2) * u.pixel

    # Flag sources with bad counts before computing noise
    # This can happen, for example, when the object is faint
    # and centroiding is bad.
    bad_cnts = photom['aperture_net_cnts'].value < 0
    all_bads = bad_cnts | bad_fwhm
    photom['aperture_net_cnts'][all_bads] = np.nan

    # Compute and save noise
    noise = calculate_noise(gain=camera.gain.value,
                            read_noise=camera.read_noise.value,
                            dark_current_per_sec=camera.dark_current.value,
                            counts=photom['aperture_net_cnts'].value,
                            sky_per_pix=photom['sky_per_pix_avg'].value,
                            aperture_area=photom['aperture_area'].value,
                            annulus_area=photom['annulus_area'].value,
                            exposure=photom['exposure'].value,
                            include_digitization=include_dig_noise)
    photom['noise_electrons'] = noise  # Noise in electrons
    photom['noise_cnts'] = noise / camera.gain.value  # Noise in counts

    # Compute and save SNR
    snr = camera.gain.value * photom['aperture_net_cnts'] / noise
    photom['snr'] = snr
    photom['mag_error'] = 1.085736205 / snr

    # Compute instrumental magnitudes
    photom['mag_inst'] = \
        (-2.5 * np.log10(camera.gain.value * photom['aperture_net_cnts'].value /
                            photom['exposure'].value))

    # Set units on photometry related terms
    photom['noise_electrons'].unit = u.electron
    photom['noise_cnts'].unit = photom['sky_per_pix_avg'].unit
    photom['aperture_sum'].unit = photom['sky_per_pix_avg'].unit
    photom['annulus_sum'].unit = photom['sky_per_pix_avg'].unit
    photom['aperture_net_cnts'].unit = photom['sky_per_pix_avg'].unit

    # Create PhotometryData object to return
    photom_data = PhotometryData(observatory=observatory_location, camera=camera,
                                input_data=photom, passband_map=passband_map)

    # TO DO
    # - filter for too close sources

    return photom_data, out_of_bounds_sources


def photutils_stellar_photometry(ccd_image, sources,
                                 aperture_radius, inner_annulus,
                                 outer_annulus, gain=1.0, N_R=0, N_dark_pp=0,
                                 reject_background_outliers=True):
    """
    Perform aperture photometry on an image, with a few options for estimating
    the local background from an annulus around the aperture.

    Parameters
    ----------
    ccd_image : `astropy.nddata.CCDData`
        Image on which to perform aperture photometry.

    sources : `astropy.table.Table`
        Table of extracted sources. Assumed to be the output of
        `photutils.daofind()` source extraction function.

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
    phot_table : `astropy.table.Table`
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
        ra, dec = ccd_image.wcs.all_pix2world(coords[0], coords[1], 0)
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


def find_too_close(sourcelist, aperture_rad, pixel_scale=None):
    """
    Identify sources that are closer together than twice the aperture radius.

    If 'xcenter' and 'ycenter' are available in the sourcelist (as determined by
    the value of sourcelist.has_x_y), they are used to determine separation of sources
    in units of pixels, otherwise if only ra/dec are available, the pixel_scale
    is necessary to determine the separation.

    Parameters
    ----------

    sourcelist : `stellarphot.SourceListData`
        A list of sources with x/y and/or RA/Dec coordinates.

    aperture_rad : int
        Radius of the aperture, in pixels.

    pixel_scale : float, optional (Default: None)
        Pixel scale of the image in arcsec/pixel. Only required
        if x/y coordinates are NOT provided.

    Returns
    -------

    numpy array of bool
        Array the same length as the RA/Dec that is ``True`` where the sources
        are closer than two aperture radii, ``False`` otherwise.
    """
    if not isinstance(sourcelist, SourceListData):
        raise TypeError("sourcelist must be of type SourceListData not "
                        f"{type(sourcelist)}")

    if sourcelist.has_x_y:
        x, y = sourcelist['xcenter'], sourcelist['ycenter']
        # Find the pixel distance to the nearest neighbor for each source
        dist_mat = cdist(np.array([x, y]).T, np.array([x, y]).T, metric='euclidean')
        np.fill_diagonal(dist_mat, np.inf)
        # Return array with True where the distance is less than twice the aperture 
        # radius
        return (dist_mat.min(0) < 2 * aperture_rad)
    elif sourcelist.has_ra_dec:
        star_coords = SkyCoord(ra=sourcelist['ra'], dec=sourcelist['dec'],
                           frame='icrs', unit='degree')
        idxc, d2d, d3d = star_coords.match_to_catalog_sky(star_coords,
                                                        nthneighbor=2)
        return (d2d < (aperture_rad * 2 * pixel_scale * u.arcsec))
    else:
        raise ValueError("sourcelist must have x/y or ra/dec coordinates")


def clipped_sky_per_pix_stats(data, annulus, sigma=5, iters=5):
    """
    Calculate sigma-clipped statistics on an annulus.

    Parameters
    ----------

    data : `astropy.nddata.CCDData`
        CCD image on which the annuli are defined.

    annulus : `photutils.CircularAnnulus`
        One or more annulus (of any shape) from photutils.

    sigma : float, optional
        Number of standard deviations from the central value a
        pixel must be to reject it.

    iters : int, optional
        Maximum number of sigma clip iterations to perform. Iterations stop
        automatically if no pixels are rejected.

    Returns
    -------

    avg_sky_per_pix, med_sky_per_pix, std_sky_per_pix : `astropy.units.Quantity`
        Average, median and standard deviation of the sky per pixel.

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
    Calculate several columns for photometry table which are appended to the
    input phot.

    Parameters
    ----------

    phot : `astropy.table.Table`
        An astropy Table with raw photometry data in it (generated by
        `photutils.aperture_photometry`).

    ccd : `astropy.nddata.CCDData`
        Image on which photometry is being done.

    annulus : `photutils.CircularAnnulus`
        One or more annulus (of any shape) from photutils.

    apertures : `photutils.CircularAperture`
        One or more apertures (of any shape) from photutils.

    fname : str, optional
        Name of the image file on which photometry is being performed.

    star_ids : str or int, optional
        ID for each of the sources.

    gain : float, optional
        Gain, in electrons/ADU, of the camera that took the image. The gain
        is used in calculating the instrumental magnitude.

    bjd_coords : `astropy.coordinates.SkyCoord`
        Coordinates of the object of interest in the Barycentric Julian Date
        frame. If not provided, the BJD column will not be added to the
        photometry table.

    observatory_location : str
        Name of the observatory where the images were taken. If not 'feder',
        the BJD column will not be added to the photometry table.

    fwhm_by_fit : bool, optional
        If ``True``, the FWHM will be calculated by fitting a Gaussian to
        the star. If ``False``, the FWHM will be calculated by finding the
        second moments of the light distribution. Default is ``True``.
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
    phot['night'] = int(np.floor(night.value - 0.5))
    phot['aperture_net_counts'] = (phot['aperture_sum'] -
                                 (phot['aperture_area'] *
                                  phot['sky_per_pix_avg']))

    # This can happen, for example, when the object is faint
    # and centroiding is bad.
    bad_flux = phot['aperture_net_counts'] < 0
    all_bads = bad_flux | bad_fwhm

    phot['aperture_net_counts'][all_bads] = np.nan

    if observatory_location.lower() == 'feder':
        phot['BJD'] = find_bjd(phot['date-obs'][0], phot['exposure'][0],
                                 ra=bjd_coords.ra, dec=bjd_coords.dec)

    if camera is not None:
        phot['mag_inst'] = \
            (-2.5 * np.log10(camera.gain * phot['aperture_net_counts'].value /
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

    aperture_rad : float
        Radius of the aperture to use when performing photometry.

    inner_annulus : float
        Inner radius of annulus to use in for performing local sky
        subtraction.

    outer_annulus : float
        Outer radius of annulus to use in for performing local sky
        subtraction.

    max_adu : float
        Maximum allowed pixel value before a source is considered
        saturated.

    star_ids : array-like
        Unique identifier for each source in ``star_locs``.

    camera : `~stellarphot.Camera`
        Camera object which has gain, read noise and dark current set.

    bjd_coords : `astropy.coordinates.SkyCoord`
        Coordinates of the object of interest in the Barycentric Julian Date
        frame. If not provided, the BJD column will not be added to the
        photometry table.

    observatory_location : str
        Name of the observatory where the images were taken. If not provided,
        the BJD column will not be added to the photometry table.

    fwhm_by_fit : bool, optional
        If ``True``, the FWHM will be calculated by fitting a Gaussian to
        the star. If ``False``, the FWHM will be calculated by finding the
        second order moments of the light distribution. Default is ``True``.

    Returns
    -------

    phot : `astropy.table.Table`
        Table containing the photometry results.

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

    # Compute and save noise
    noise = calculate_noise(gain=camera.gain, read_noise=camera.read_noise,
                            dark_current_per_sec=camera.dark_current,
                            counts=all_phot['aperture_net_counts'],
                            sky_per_pix=all_phot['sky_per_pix_avg'].value,
                            aperture_area=all_phot['aperture_area'],
                            annulus_area=all_phot['annulus_area'],
                            exposure=all_phot['exposure'].value,
                            include_digitization=False)
    all_phot['noise'] = noise  # Noise in electrons
    all_phot['noise-aij'] = noise / gain  # Noise in counts

    # Compute and save SNR
    snr = gain * all_phot['aperture_net_counts'] / noise
    all_phot['snr'] = snr
    all_phot['mag_error'] = 1.085736205 / snr

    return all_phot


def calculate_noise(gain=1.0, read_noise=0.0, dark_current_per_sec=0.0,
                    counts=0.0, sky_per_pix=0.0, aperture_area=0, annulus_area=0,
                    exposure=0, include_digitization=False):
    """
    Computes the noise in a photometric measurement.

    This function computes the noise (in units of electrons) in a photometric
    measurement using the revised CCD equation from Collins et al (2017) AJ, 153, 77
    who based their expression on the one originally proposed for SNR by 
    Merline, W. J., & Howell, S. B. 1995, Experimental Astronomy, 6, 163.
    
    The equation is:

    .. math::

        \\sigma = \\sqrt{G \\cdot N_C + A \\cdot \\left(1 + \\frac{A}{B}\\right)\\cdot \\left[ G\\cdot S + D \\cdot t + R^2 + (0.289 G)^2\\right]}

    where :math:`\sigma` is the noise, :math:`G` is the gain,
    :math:`N_C` is the source counts (which is photon/electron counts divided by gain), 
    :math:`A` is the aperture area in pixels,
    :math:`B` is the annulus area in pixels,
    :math:`S` is the sky counts per pixel,
    :math:`D` is the dark current in electrons per second,
    :math:`R` is the read noise in electrons per pixel per read,
    and :math:`t` is exposure time in seconds.

    Note: The :math:`(0.289 G)^2` term is "digitization noise" and is optional.

    Parameters
    ----------

    gain : float, optional
        Gain of the CCD. In units of electrons per count.

    read_noise : float, optional
        Read noise of the CCD in electrons.

    dark_current_per_sec : float, optional
        Dark current of the CCD in electrons per second.

    counts : float, optional
        Counts of the source.

    sky_per_pix : float, optional
        Sky per pixel in counts per pixel.

    aperture_area : int, optional
        Area of the aperture in pixels.

    annulus_area : int, optional
        Area of the annulus in pixels.

    exposure : int, optional
        Exposure time in seconds.

    include_digitization : bool, optional
        Whether to include the digitization noise. Defaults to False.

    Returns
    -------

    noise : float
        The noise in the photometric measurement in electrons.
    """

    try:
        no_annulus = (annulus_area == 0).all()
    except AttributeError:
        no_annulus = annulus_area == 0

    if no_annulus:
        area_ratio = aperture_area
    else:
        area_ratio = aperture_area * (1 + aperture_area / annulus_area)

    # Convert counts to electrons
    poisson_source = gain * counts

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


def find_bjd(dates_col, exposure, ra, dec,
               latitude=46.86678, longitude=263.54672):
    """
    Returns a numpy array of barycentric Julian date times

    Parameters
    ----------

    dates_col : `astropy.table.Column` or numpy array
        numpy array or column of observation dates from the photometry table

    exposure : float, optional
        exposure time in seconds

    ra : float
        Right ascension in degree units

    dec : float
        Declination  in degree units

    latitude : float, optional
        latitude of the observatory in degrees North, default is for Paul P. Feder
        Observatory

    longitude : float, optional
        longitude of the observatory in degree East of Greenwich (0 to 360), default
        is for Paul P. Feder Observatory


    Returns
    -------

    numpy array
        array of times in barycentric Julian date

    """
    location = EarthLocation(lat=latitude, lon=longitude)

    times = Time(dates_col, scale='utc', format='isot', location=location)
    ip_peg = SkyCoord(ra=ra, dec=dec, unit='degree')
    ltt_bary = times.light_travel_time(ip_peg)
    times_tdb = times.tdb
    time_barycenter = times_tdb + ltt_bary

    # Adjust to midpoint of exposure
    bary_time = time_barycenter + exposure / 2

    return bary_time.jd