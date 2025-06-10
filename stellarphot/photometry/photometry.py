import logging
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import CCDData, NoOverlapError
from astropy.stats import SigmaClip
from astropy.table import Column, vstack
from astropy.time import Time
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import FITSFixedWarning
from ccdproc import ImageFileCollection
from photutils.aperture import (
    ApertureStats,
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)
from photutils.centroids import centroid_sources
from pydantic import BaseModel, validate_call
from scipy.spatial.distance import cdist

from stellarphot import PhotometryData, SourceListData
from stellarphot.settings import (
    Camera,
    Observatory,
    PhotometryApertures,
    PhotometryOptionalSettings,
    PhotometrySettings,
)

from .source_detection import compute_fwhm, fast_fwhm_from_image

__all__ = [
    "AperturePhotometry",
    "single_image_photometry",
    "multi_image_photometry",
    "find_too_close",
    "clipped_sky_per_pix_stats",
    "calculate_noise",
]

# Allowed FITS header keywords for exposure values
EXPOSURE_KEYWORDS = ["EXPOSURE", "EXPTIME", "TELAPSE", "ELAPTIME", "ONTIME", "LIVETIME"]


class AperturePhotometry(BaseModel):
    """Class to perform aperture photometry on one or more images"""

    settings: PhotometrySettings

    @validate_call
    def __call__(
        self,
        file_or_directory: str | Path,
        **kwargs,
    ) -> PhotometryData:
        """
        Perform aperture photometry on a single image or a directory of images.

        Parameters
        ----------
        file_or_directory : str or Path
            The file or directory on which to perform aperture photometry.

        logline : str, optional (Default: "single_image_photometry:")
            String to prepend to all log messages, *only used for single image
            photometry*.

        reject_unmatched : bool, optional (Default: True)
            If ``True``, any sources that are not detected on all the images are
            rejected.  If you are interested in a source that can intermittently
            fall below your detection limits, we suggest setting this to ``False``
            so that all sources detected on each image are reported. *Only used for
            multi-image photometry*.

        object_of_interest : str, optional (Default: None)
            Name of the object of interest. The only files on which photometry
            will be done are those whose header contains the keyword ``OBJECT``
            whose value is ``object_of_interest``. *Only used for multi-image
            photometry* to select which files to perform photometry on.

        Returns
        -------
        photom_data : `stellarphot.PhotometryData`
            Photometry data for all the sources on which aperture photometry was
            performed in all the images. This may be a subset of the sources in
            the source list if locations were too close to the edge of any one
            image or to each other for successful aperture photometry.
        """
        # Make sure we have a Path object
        path = Path(file_or_directory)
        if path.is_dir():
            photom_data = multi_image_photometry(path, self.settings, **kwargs)
        elif path.is_file():
            image = CCDData.read(path)
            photom_data = single_image_photometry(
                image, self.settings, fname=str(path), **kwargs
            )
        else:
            raise ValueError(
                f"file_or_directory '{path}' is not a valid file or directory."
            )

        return photom_data


def single_image_photometry(
    ccd_image,
    photometry_settings,
    fname=None,
    logline="single_image_photometry:",
):
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
        Image on which to perform aperture photometry.  It's headers must contain
        DATE-OBS, FILTER, and exposure time in units of seconds (identified by one
        of the following keywords: EXPOSURE, EXPTIME, TELAPSE, ELAPTIME, ONTIME,
        or LIVETIME). If AIRMASS is available it will be added to `phot_table`.
        This image  must also have a WCS header associated with it if you want to
        use sky positions as inputs.

    photometry_settings : `stellarphot.settings.PhotometrySettings`
        Photometry settings to use for the photometry.  This includes the camera,
        observatory, the aperture and annulus radii to use for the photometry, a map
        of passbands from your filters to AAVSO filter names, and options for the
        photometry. See `stellarphot.settings.PhotometrySettings` for more information.

    fname : str, optional (Default: None)
        Name of the image file on which photometry is being performed, used for
        logging and tracking purposes, the file is not accessed or modified.

    logline : str, optional (Default: "single_image_photometry:")
        String to prepend to all log messages.

    Returns
    -------

    phot_table : `stellarphot.PhotometryData`
        Photometry data for all the locations in which aperture photometry was
        performed.  This may be a subset of the sources in the sourcelist if
        locations were too close to the edge of the image or to each other for
        successful aperture photometry.  If pixel (x/y) positions were used for
        the photometry, but a valid WCS header was not available for `ccd_image`,
        the output 'ra', 'dec', and 'bjd' columns will have np.nan values

    dropped_sources : list
        This of the star_ids of the sources that fell outside the image or were
        too close together and did not have photometry performed.

    Notes
    -----
    The default behavior (set by `use_coordinates`="pixel") for determining
    the placement of apertures is to use the x/y positions in the `sourcelist`.
    This is because in most cases, ra/dec positions are derived from the x/y
    positions and the WCS of the `ccd_image`.  This default reduces unnecessary
    computation and works if the WCS is not very accurate.

    When attempting to process multiple images of the same field, it makes sense
    to use the ra/dec positions in the sourcelist and the WCS of the `ccd_image`
    to compute the x/y positions on each image individually. In this scenario,
    the `use_coordinates` parameter should be set to "sky".
    """

    sourcelist = SourceListData.read(
        photometry_settings.source_location_settings.source_list_file
    )
    camera = photometry_settings.camera
    observatory = photometry_settings.observatory
    photometry_apertures = photometry_settings.photometry_apertures
    photometry_options = photometry_settings.photometry_optional_settings
    logging_options = photometry_settings.logging_settings
    passband_map = photometry_settings.passband_map
    use_coordinates = photometry_settings.source_location_settings.use_coordinates

    # Check that the input parameters are valid
    if not isinstance(ccd_image, CCDData):
        raise TypeError(
            "ccd_image must be a CCDData object, but it is " f"'{type(ccd_image)}'."
        )
    if not isinstance(sourcelist, SourceListData):
        raise TypeError(
            "sourcelist must be a SourceListData object, but it is "
            f"'{type(sourcelist)}'."
        )
    if not isinstance(camera, Camera):
        raise TypeError(f"camera must be a Camera object, but it is '{type(camera)}'.")
    if not isinstance(observatory, Observatory):
        raise TypeError(
            "observatory_location must be a EarthLocation object, but it "
            f"is '{type(observatory)}'."
        )

    if not isinstance(photometry_apertures, PhotometryApertures):
        raise TypeError(
            "photometry_apertures must be a PhotometryApertures object, but it is "
            f"'{type(photometry_apertures)}'."
        )

    if not isinstance(photometry_options, PhotometryOptionalSettings):
        raise TypeError(
            "photometry_options must be a PhotometryOptions object, but it is "
            f"'{type(photometry_options)}'."
        )

    # Set up logging
    logfile = logging_options.logfile
    console_log = logging_options.console_log

    # If we have no handlers, set up logging
    logger = logging.getLogger("single_image_photometry")

    # Use this to catch if logging was already set up by a call
    # from multi_image_photometry before creating new log file handler
    # (Warning: This actually just catches if the logger has any handlers.
    # Probably not the best way to do this.)
    fh_exists = False  # Default to filehandler not existing
    if logger.hasHandlers() is False:
        logger.setLevel(logging.INFO)
        # Set up logging to a file (in addition to any logging below)
        if logfile is not None:
            # by default this appends to existing logfile
            fh = logging.FileHandler(logfile)
            log_format = logging.Formatter("%(levelname)s - %(message)s")
            fh.setFormatter(log_format)
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)
            fh_exists = True

        # Set up logging to console if requested
        if console_log:
            ch = logging.StreamHandler()
        else:  # otherwise effectively suppress output
            ch = logging.NullHandler()
        console_format = logging.Formatter("%(message)s")
        ch.setFormatter(console_format)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    #
    # Check CCDData headers before proceeding
    #
    # Search for exposure keyword in header and set exposure if found
    matched_kw = None
    for kw in EXPOSURE_KEYWORDS:
        if kw in ccd_image.header:
            matched_kw = kw
            break

    if matched_kw is None:
        logger.warning(
            f"{logline} None of the accepted exposure keywords "
            f"({format(', '.join(EXPOSURE_KEYWORDS))}) found in the "
            "header ... SKIPPING THIS IMAGE!"
        )
        return None, None

    exposure = ccd_image.header[matched_kw]

    # Search for other keywords that are required
    try:
        date_obs = ccd_image.header["DATE-OBS"]
    except KeyError:
        logger.warning(
            f"{logline} 'DATE-OBS' not found in CCD image header "
            "... SKIPPING THIS IMAGE!"
        )
        return None, None
    try:
        filter = ccd_image.header["FILTER"]
    except KeyError:
        logger.warning(
            f"{logline} 'FILTER' not found in CCD image header ... "
            "SKIPPING THIS IMAGE!"
        )
        return None, None

    # Set high pixels to NaN (make sure ccd_image.data is a float array first)
    ccd_image.data = ccd_image.data.astype(float)
    ccd_image.data[ccd_image.data > camera.max_data_value.value] = np.nan

    # Extract necessary values from sourcelist structure
    star_ids = sourcelist["star_id"].value
    xs = sourcelist["xcenter"].value
    ys = sourcelist["ycenter"].value
    ra = sourcelist["ra"].value
    dec = sourcelist["dec"].value
    src_cnt = len(sourcelist)

    # If RA/Dec are available attempt to use them to determine the source positions
    if use_coordinates == "sky" and sourcelist.has_ra_dec:
        try:
            imgpos = ccd_image.wcs.world_to_pixel(
                SkyCoord(ra, dec, unit=u.deg, frame="icrs")
            )
            xs, ys = imgpos[0], imgpos[1]
        except AttributeError:
            # No WCS, skip this image
            msg = f"{logline} ccd_image must have a valid WCS to use RA/Dec!"
            logger.warning(msg)
            return None, None
    elif use_coordinates == "sky" and not sourcelist.has_ra_dec:
        raise ValueError(
            "use_coordinates='sky' but sourcelist does not have" "RA/Dec coordinates!"
        )

    if photometry_apertures.variable_aperture:
        # Get a fast, robust estimate of the FWHM of the sources for setting
        # the aperture size.
        fwhm = fast_fwhm_from_image(
            ccd_image,
            photometry_apertures.fwhm_estimate,
            noise=camera.read_noise.value,
            max_adu=camera.max_data_value.value,
        )
    else:
        # Use the FWHM from the settings
        fwhm = photometry_apertures.fwhm_estimate

    # Reject sources that are within an aperture diameter of each other.
    dropped_sources = []
    try:
        too_close = find_too_close(
            sourcelist,
            photometry_apertures.radius_pixels(fwhm),
            pixel_scale=camera.pixel_scale.value,
        )
    except Exception as err:
        # Any failure here is BAD, so raise an error
        raise RuntimeError(
            f"Call to find_too_close() returned {type(err).__name__}: {str(err)}"
        ) from err
    too_close_cnt = np.sum(too_close)
    non_overlap = ~too_close
    msg = (
        f"{logline} {too_close_cnt} of {src_cnt} sources within 2 aperture radii of "
        "nearest neighbor"
    )

    if photometry_options.reject_too_close:
        # Track dropped sources due to being too close together
        dropped_sources.extend(star_ids[too_close].tolist())
        # Remove sources too close together
        star_ids = star_ids[non_overlap]
        xs = xs[non_overlap]
        ys = ys[non_overlap]
        ra = ra[non_overlap]
        dec = dec[non_overlap]
        msg += " ... removed them."
    else:
        msg += " ... keeping them."
    logger.info(msg)

    # Remove all source positions too close to edges of image (where the annulus would
    # extend beyond the image boundaries).
    padding = photometry_apertures.outer_annulus
    out_of_bounds = (
        (xs < padding)
        | (xs > (ccd_image.shape[1] - padding))
        | (ys < padding)
        | (ys > (ccd_image.shape[0] - padding))
    )
    in_bounds = ~out_of_bounds
    # Track dropped sources due to out of bounds positions
    dropped_sources.extend(star_ids[out_of_bounds].tolist())
    # Remove sources too close to the edges
    star_ids = star_ids[in_bounds]
    xs = xs[in_bounds]
    ys = ys[in_bounds]
    ra = ra[in_bounds]
    dec = dec[in_bounds]
    in_cnt = np.sum(in_bounds)
    out_cnt = np.sum(out_of_bounds)
    logger.info(
        f"{logline} {out_cnt} sources too close to image edge ... removed " "them."
    )
    logger.info(
        f"{logline} {in_cnt} of {src_cnt} original sources to have photometry " "done."
    )

    # If we are using x/y positions previously obtained from the ra/dec positions and
    # WCS, then recentroid the sources to refine the positions. This is
    # particularly useful is processing multiple images of the same field
    # and just passing the same sourcelist when calling single_image_photometry
    # on each image.
    if use_coordinates == "sky":
        try:
            xcen, ycen = centroid_sources(
                ccd_image.data,
                xs,
                ys,
                box_size=2 * int(photometry_apertures.radius_pixels(fwhm)) + 1,
            )
        except NoOverlapError:
            logger.warning(
                f"{logline} Determining new centroids failed ... "
                "SKIPPING THIS IMAGE!"
            )
            return None, None
        else:  # Proceed
            # Calculate offset between centroid in this image and the positions
            # based on input RA/Dec.
            center_diff = np.sqrt((xs - xcen) ** 2 + (ys - ycen) ** 2)

            # The center really shouldn't move more than about the fwhm, could
            # rework this in the future to use that instead.
            too_much_shift = (
                center_diff
                > photometry_settings.source_location_settings.shift_tolerance
            )

            # If the shift is too large, use the WCS-derived positions instead
            # (these sources are probably too faint for centroiding to work well)
            xcen[too_much_shift] = xs[too_much_shift]
            ycen[too_much_shift] = ys[too_much_shift]
            xs, ys = xcen, ycen

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
    apers = CircularAperture(aper_locs, r=photometry_apertures.radius_pixels(fwhm))
    annuli = CircularAnnulus(
        aper_locs,
        r_in=photometry_apertures.inner_annulus,
        r_out=photometry_apertures.outer_annulus,
    )

    # Perform the aperture photometry
    photom = aperture_photometry(
        ccd_image.data,
        (apers, annuli),
        mask=ccd_image.mask,
        method=photometry_options.partial_pixel_method,
    )

    # photutils 2.1.0 stopped returning pixel units for x/y center
    # https://photutils.readthedocs.io/en/stable/changelog.html#id6
    if photom["xcenter"].unit is None:
        photom["xcenter"] = photom["xcenter"] * u.pixel
        photom["ycenter"] = photom["ycenter"] * u.pixel

    # Add source ids to the photometry table
    photom["star_id"] = star_ids
    photom["ra"] = ra * u.deg
    photom["dec"] = dec * u.deg

    # Drop ID column from aperture_photometry()
    del photom["id"]

    # Add various CCD image parameters to the photometry table
    if fname is not None:
        photom["file"] = fname
    else:
        photom["file"] = [""] * len(photom)

    # Set various columns based on CCDData headers (which we
    # checked for earlier)
    photom["exposure"] = [exposure] * len(photom) * u.second
    photom["date-obs"] = Time(Column(data=[date_obs]))
    photom["filter"] = [filter] * len(photom)
    photom.rename_column("filter", "passband")

    # Check for airmass keyword in header and set 'airmass' if found,
    # but accept it may not be available
    try:
        photom["airmass"] = [ccd_image.header["AIRMASS"]] * len(photom)
    except KeyError:
        logger.warning(
            f"{logline} 'AIRMASS' not found in CCD " "image header ... setting to NaN!"
        )
        photom["airmass"] = [np.nan] * len(photom)

    # Save aperture and annulus information
    photom.rename_column("aperture_sum_0", "aperture_sum")
    photom.rename_column("aperture_sum_1", "annulus_sum")
    photom["aperture_sum"].unit = ccd_image.unit
    photom["annulus_sum"].unit = ccd_image.unit
    photom["aperture"] = apers.r * u.pixel
    photom["annulus_inner"] = annuli.r_in * u.pixel
    photom["annulus_outer"] = annuli.r_out * u.pixel
    # By convention, area is in units of pixels (not pixels squared) in a digital image
    photom["aperture_area"] = apers.area * u.pixel
    photom["annulus_area"] = annuli.area * u.pixel

    if photometry_options.reject_background_outliers:
        msg = f"{logline} Computing clipped sky stats ... "
        try:
            (
                avg_sky_per_pix,
                med_sky_per_pix,
                std_sky_per_pix,
            ) = clipped_sky_per_pix_stats(ccd_image, annuli)
        except AttributeError:
            msg += "BAD ANNULUS ('sky_per_pix' stats set to np.nan) ... "
            avg_sky_per_pix, med_sky_per_pix, std_sky_per_pix = np.nan, np.nan, np.nan
        photom["sky_per_pix_avg"] = avg_sky_per_pix / u.pixel
        photom["sky_per_pix_med"] = med_sky_per_pix / u.pixel
        photom["sky_per_pix_std"] = std_sky_per_pix / u.pixel
        msg += "DONE."
        logger.info(msg)

    else:  # Don't reject outliers (but why would you do this?)
        logger.warning(
            f"{logline} SUGGESTION: You are computing sky per pixel "
            "without clipping (set reject_background_outliers=True "
            "to perform clipping)."
        )
        med_pp = []
        std_pp = []
        for mask in annuli.to_mask():
            annulus_data = mask.cutout(ccd_image)
            med_pp.append(np.median(annulus_data))
            std_pp.append(np.std(annulus_data))
        photom["sky_per_pix_avg"] = photom["annulus_sum"] / photom["annulus_area"]
        photom["sky_per_pix_med"] = np.array(med_pp) * ccd_image.unit / u.pixel
        photom["sky_per_pix_std"] = np.array(std_pp) * ccd_image.unit / u.pixel

    # Compute counts using clipped stats on sky per pixel
    photom["aperture_net_cnts"] = photom["aperture_sum"].value - (
        photom["aperture_area"].value * photom["sky_per_pix_avg"].value
    )
    photom["aperture_net_cnts"].unit = ccd_image.unit

    # Fit the FWHM of the sources (can result in many warnings due to
    # failed FWHM fitting, capture those warnings and print a summary)
    msg = f"{logline} Fitting FWHM of all sources (may take a few minutes) ... "
    with warnings.catch_warnings(record=True) as warned:
        warnings.filterwarnings("always", category=AstropyUserWarning)
        fwhm_x, fwhm_y = compute_fwhm(
            ccd_image,
            photom,
            fwhm_estimate=photometry_apertures.fwhm_estimate,
            fit_method=photometry_options.fwhm_method,
            sky_per_pix_column="sky_per_pix_avg",
        )
        num_warnings = len(warned)
        msg += f"fitting failed on {num_warnings} of {len(photom)} sources  ... "
    msg += "DONE."
    logger.info(msg)

    # Deal with bad FWHM values
    bad_fwhm = (fwhm_x < 1) | (fwhm_y < 1)  # Set bad values to NaN now
    fwhm_x[bad_fwhm] = np.nan
    fwhm_y[bad_fwhm] = np.nan
    photom["fwhm_x"] = fwhm_x * u.pixel
    photom["fwhm_y"] = fwhm_y * u.pixel
    photom["width"] = ((fwhm_x + fwhm_y) / 2) * u.pixel
    if np.sum(bad_fwhm) > 0:
        logger.info(
            f"{logline} Bad FWHM values (<1 pixel) for {np.sum(bad_fwhm)} " "sources."
        )

    # Flag sources with bad counts before computing noise.
    # This can happen, for example, when the object is faint and centroiding is
    # bad.  It can also happen when the sky background is low.
    bad_cnts = photom["aperture_net_cnts"].value < 0
    # This next line works because booleans are just 0/1 in numpy
    if np.sum(bad_cnts) > 0:
        logger.info(
            f"{logline} Aperture net counts negative for {np.sum(bad_cnts)} " "sources."
        )

    all_bads = bad_cnts | bad_fwhm

    photom["aperture_net_cnts"][all_bads] = np.nan
    logger.info(
        f"{logline} {np.sum(all_bads)} sources with either bad FWHM fit "
        "or bad aperture net counts had aperture_net_cnts set to NaN."
    )

    # Compute instrumental magnitudes
    photom["mag_inst"] = -2.5 * np.log10(
        camera.gain.value * photom["aperture_net_cnts"].value / photom["exposure"].value
    )

    # Compute and save noise
    msg = f"{logline} Calculating noise for all sources ... "
    noise = calculate_noise(
        camera=camera,
        counts=photom["aperture_net_cnts"].value,
        sky_per_pix=photom["sky_per_pix_avg"].value,
        aperture_area=photom["aperture_area"].value,
        annulus_area=photom["annulus_area"].value,
        exposure=photom["exposure"].value,
        include_digitization=photometry_options.include_dig_noise,
    )
    photom["noise_electrons"] = noise  # Noise in electrons
    photom["noise_electrons"].unit = u.electron
    photom["noise_cnts"] = noise / camera.gain.value  # Noise in counts
    photom["noise_cnts"].unit = ccd_image.unit

    # Compute and save SNR
    snr = camera.gain.value * photom["aperture_net_cnts"] / noise
    photom["snr"] = snr
    photom["mag_error"] = 1.085736205 / snr
    msg += "DONE."
    logger.info(msg)

    # Close logfile if it was created
    if fh_exists:
        fh.flush()
        fh.close()

    # Remove logger handler
    logger.handlers.clear()

    # Create PhotometryData object to return
    photom_data = PhotometryData(
        observatory=observatory,
        camera=camera,
        input_data=photom,
        passband_map=passband_map,
    )

    return photom_data, dropped_sources


def multi_image_photometry(
    directory_with_images,
    photometry_settings,
    reject_unmatched=True,
    object_of_interest=None,
):
    """
    Perform aperture photometry on a directory of images.

    Parameters
    ----------

    directory_with_images : str
        Folder containing the images on which to do photometry. Photometry
        will only be done on images that contain the ``object_of_interest``.
        All images *must* have WCS headers and the following headers: OBJECT,
        DATE-OBS, an exposure time header (which can be any of the following: EXPOSURE,
        EXPTIME, TELAPSE, ELAPTIME, ONTIME, or LIVETIME), and FILTER.  If AIRMASS is
        available it will be added to `phot_table`.

    photometry_settings : `stellarphot.settings.PhotometrySettings`
        Photometry settings to use for the photometry.  This includes the camera,
        observatory, the aperture and annulus radii to use for the photometry, a map
        of passbands from your filters to AAVSO filter names, and options for the
        photometry. See `stellarphot.settings.PhotometrySettings` for more information.

    reject_unmatched : bool, optional (Default: True)
        If ``True``, any sources that are not detected on all the images are
        rejected.  If you are interested in a source that can intermittently
        fall below your detection limits, we suggest setting this to ``False``
        so that all sources detected on each image are reported.

    object_of_interest : str, optional (Default: None)
        Name of the object of interest. The only files on which photometry
        will be done are those whose header contains the keyword ``OBJECT``
        whose value is ``object_of_interest``. *Only used for multi-image
        photometry* to select which files to perform photometry on.

    Returns
    -------

    phot_table : `stellarphot.PhotometryData`
        Photometry data for all the sources on which aperture photometry was
        performed in all the images. This may be a subset of the sources in
        the sourcelist if locations were too close to the edge of any one image
        or to each other for successful aperture photometry.

    """
    sourcelist = SourceListData.read(
        photometry_settings.source_location_settings.source_list_file
    )

    # Initialize lists to track all PhotometryData objects and all dropped sources
    phots = []
    missing_sources = []

    # Confirm sourcelist has ra/dec coordinates
    if not sourcelist.has_ra_dec:
        raise ValueError(
            "multi_image_photometry: sourcelist must have RA/Dec "
            "coordinates to use this function."
        )

    # Set up logging (retrieve a logger but purge any existing handlers)
    multilogger = logging.getLogger("multi_image_photometry")
    multilogger.setLevel(logging.INFO)
    console_format = logging.Formatter("%(message)s")
    # Remove all other existing handlers from the logger
    # (Kind of brute force, but works for our purposes
    for handler in multilogger.handlers[:]:
        multilogger.removeHandler(handler)

    logfile = photometry_settings.logging_settings.logfile
    console_log = photometry_settings.logging_settings.console_log

    # Set up logging:
    # Check if logfile is not None, set up logging to be written to the logfile.
    # Next check if console_log is True, if it is, set up logging to be written
    #    to the console.
    # If neither of these are true, set up logging to be written to NullHandler
    #    which effectively suppresses the logging output.

    # Set up logging to a file (in addition to any logging below)
    if logfile is not None:
        # Get the name of the logfile without the path
        logfile = Path(logfile).name
        # Redirect the logfile to the directory_with_images
        logfile = Path(directory_with_images) / logfile
        # Change the settings so when they are passed to single_image_photometry
        # the logging will be written to the same logfile
        photometry_settings.logging_settings.logfile = str(logfile)

        # Keep original name without path
        logfile_name = str(Path(logfile).name)

        # by default this appends to existing logfile
        fh = logging.FileHandler(logfile)
        log_format = logging.Formatter("%(levelname)s - %(message)s")
        fh.setFormatter(log_format)
        fh.setLevel(logging.INFO)
        multilogger.addHandler(fh)

    # Set up logging to console if requested
    if console_log:
        ch = logging.StreamHandler()
    else:  # otherwise effectively suppress output
        ch = logging.NullHandler()
    ch.setFormatter(console_format)
    ch.setLevel(logging.INFO)
    multilogger.addHandler(ch)

    ##
    ## Process all the individual files
    ##

    # Build image file collection
    ifc = ImageFileCollection(directory_with_images)

    # Disable any root logger handlers that are active before using
    # logging (must be done here because ImageFileCollection() creates
    # a logger)
    if logging.root.hasHandlers():
        logging.root.handlers.clear()

    n_files_processed = 0

    msg = f"Starting photometry of files in {directory_with_images} ... "
    if logfile is not None:
        msg += f"logging output to {logfile_name}"
        # If not logging to console, print message here
        if not console_log:
            print(msg)
    multilogger.info(msg)

    # Suppress the FITSFixedWarning that is raised when reading a FITS file header
    warnings.filterwarnings("ignore", category=FITSFixedWarning)

    # Process all the files
    for this_ccd, this_fname in ifc.ccds(object=object_of_interest, return_fname=True):
        multilogger.info(f"multi_image_photometry: Processing image {this_fname}")
        if this_ccd.wcs is None:
            multilogger.warning("                   .... SKIPPING THIS IMAGE (NO WCS)")
            continue

        # Call single_image_photometry on each image
        n_files_processed += 1
        multilogger.info("  Calling single_image_photometry ...")
        this_phot, this_missing_sources = single_image_photometry(
            this_ccd,
            photometry_settings,
            fname=this_fname,
            logline="    >",
        )
        if (this_phot is None) or (this_missing_sources is None):
            multilogger.info("  single_image_photometry failed for this image.")
        else:
            multilogger.info(
                f"  Done with single_image_photometry for {this_fname}\n\n"
            )

            # Extend the list of missing stars
            missing_sources.extend(this_missing_sources)

            # And add the final table to the list of tables
            phots.append(this_phot)

    if n_files_processed == 0:
        raise RuntimeError("No images were processed!")

    ##
    ## Done processing individual images, now combine them into one table
    ##

    # Combine all of the individual photometry tables into one.
    # Attributes should survive intact assume all have same camera and observatory.
    all_phot = vstack(phots)

    # If requested, eliminate source not detected on every image by building
    # a set of all the unique star_ids that were missing on at least one image.
    if reject_unmatched and len(missing_sources) > 0:
        if len(missing_sources) > 1:
            uniques = set(missing_sources)
        else:
            uniques = set([missing_sources])

        msg = f"  Removing {len(uniques)} sources not observed in every image ... "
        # Purge the photometry table of all sources that were eliminated
        # on at least one image
        starid_to_remove = sorted([u for u in uniques if u in all_phot["star_id"]])
        # add index to PhotometryData to speed up removal
        all_phot.add_index("star_id")
        # Remove the starid for objects not observed in every image
        if starid_to_remove:
            bad_rows = all_phot.loc_indices[starid_to_remove]
            try:
                bad_rows = list(bad_rows)
            except TypeError:
                bad_rows = [bad_rows]
            all_phot.remove_indices("star_id")
            all_phot.remove_rows(sorted(bad_rows))
        # Drop index from PhotometryData to save memory
        all_phot.remove_indices("star_id")
        msg += "DONE."
        multilogger.info(msg)

    multilogger.info(
        f"  DONE processing all matching images in {directory_with_images}"
    )
    if logfile is not None and not console_log:
        print(f"  DONE processing all matching images in {directory_with_images}")

    # Close the logfile if it is open
    if logfile is not None:
        fh.flush()
        fh.close()
    # Remove logger handler
    multilogger.handlers.clear()

    return all_phot


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

    pixel_scale : `float, optional` (Default: None)
        Pixel scale of the image in arcsec/pixel. Only required
        if x/y coordinates are NOT provided.

    Returns
    -------

    numpy array of bool
        Array the same length as the RA/Dec that is ``True`` where the sources
        are closer than two aperture radii, ``False`` otherwise.
    """
    if not isinstance(sourcelist, SourceListData):
        raise TypeError(
            "sourcelist must be of type SourceListData not " f"'{type(sourcelist)}'"
        )

    if not isinstance(pixel_scale, float):
        raise TypeError(f"pixel_scale must be a float not '{type(pixel_scale)}'")

    if sourcelist.has_x_y:
        x, y = sourcelist["xcenter"], sourcelist["ycenter"]
        # Find the pixel distance to the nearest neighbor for each source
        dist_mat = cdist(np.array([x, y]).T, np.array([x, y]).T, metric="euclidean")
        np.fill_diagonal(dist_mat, np.inf)
        # Return array with True where the distance is less than twice the aperture
        # radius
        return dist_mat.min(0) < 2 * aperture_rad
    elif sourcelist.has_ra_dec:
        if pixel_scale is None:
            raise ValueError(
                "pixel_scale must be provided if x/y coordinates are "
                "not available in the sourcelist."
            )
        star_coords = SkyCoord(
            ra=sourcelist["ra"], dec=sourcelist["dec"], frame="icrs", unit="degree"
        )
        idxc, d2d, d3d = star_coords.match_to_catalog_sky(star_coords, nthneighbor=2)
        return d2d < (aperture_rad * 2 * pixel_scale * u.arcsec)
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
    # Use ApertureStats from photutils to get what we need.
    sigclip = SigmaClip(sigma=sigma, maxiters=iters)
    ap_stats = ApertureStats(data, annulus, sigma_clip=sigclip, sum_method="center")

    return (
        ap_stats.mean,
        ap_stats.median,
        ap_stats.std,
    )


def calculate_noise(
    camera=None,
    counts=0.0,
    sky_per_pix=0.0,
    aperture_area=0,
    annulus_area=0,
    exposure=0,
    include_digitization=False,
):
    """
    Computes the noise in a photometric measurement.

    This function computes the noise (in units of electrons) in a photometric
    measurement using the revised CCD equation from Collins et al (2017) AJ, 153, 77
    who based their expression on the one originally proposed for SNR by
    Merline, W. J., & Howell, S. B. 1995, Experimental Astronomy, 6, 163.

    The equation is:

    .. math::

        \\sigma = \\sqrt{G \\cdot N_C + A \\cdot \\left(1 + \\frac{A}{B}\\right)\\cdot
        \\left[ G\\cdot S + D \\cdot t + R^2 + (0.289 G)^2\\right]}

    where :math:`\\sigma` is the noise, :math:`G` is the gain,
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

    camera : `stellarphot.Camera`
        The camera object that contains the gain, read noise and dark current
        of the CCD.

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
    if camera is None:
        raise ValueError("camera must be provided")
    elif not isinstance(camera, Camera):
        raise ValueError(f"camera must be of type Camera not '{type(camera)}'")

    # Extract values from camera object
    gain = camera.gain.value
    dark_current_per_sec = camera.dark_current.value
    read_noise = camera.read_noise.value

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
    rn_error = area_ratio * read_noise**2

    digitization = 0.0

    if include_digitization:
        digitization = area_ratio * (gain * 0.289) ** 2

    return np.sqrt(poisson_source + sky + dark + rn_error + digitization)
