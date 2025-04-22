import warnings

import numpy as np
from astropy import units as u
from astropy.nddata import CCDData, block_reduce
from astropy.nddata.utils import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning
from photutils.detection import DAOStarFinder
from photutils.morphology import data_properties
from photutils.profiles import RadialProfile
from photutils.psf import fit_2dgaussian, fit_fwhm

from stellarphot.core import SourceListData
from stellarphot.settings.models import FwhmMethods

__all__ = ["source_detection", "compute_fwhm", "fast_fwhm_from_image"]


def compute_fwhm(
    ccd,
    sources,
    fwhm_estimate=5,
    x_column="xcenter",
    y_column="ycenter",
    fit_method=FwhmMethods.FIT,  # This matches the old default
    sky_per_pix_avg=None,
    sky_per_pix_column=None,
):
    """
    Computes the FWHM in both x and y directions of sources in an image.

    Parameters
    ----------

    ccd : `astropy.nddata.CCDData` or `numpy.ndarray`
        The CCD Image array.

    sources : `astropy.table.Table`
        An astropy table of the positions of sources in the image.

    fwhm_estimate : float, optional (default=5)
        The initial guess for the FWHM of the sources in the image.

    x_column : str, optional (default='xcenter')
        The name of the column in `sources` that contains the x positions
        of the sources.

    y_column : str, optional (default='ycenter')
        The name of the column in `sources` that contains the y positions
        of the sources.

    fit : bool, optional (default=True)
        If ``True``, fit a 2D Gaussian to each source to estimate its FWHM. If
        ``False``, estimate the FWHM of each source by computing the second
        moments of its light distribution using photutils.

    sky_per_pix_avg : float or array-like of float, optional (default=0)
        Sky background to subtract before centroiding. Cannot specify both
        `sky_per_pix_avg` and `sky_per_pix_column`.

    sky_per_pix_column : str, optional
        The name of the column in `sources` that contains the sky
        background values for each source. This is used to subtract the
        sky background from the cutout before computing the FWHM. Cannot
        specify both `sky_per_pix_avg` and `sky_per_pix_column`.

    Returns
    -------

    fwhm_x, fwhm_y : tuple of np.array float
        The FWHM of each source in the x and y directions.

    """
    data = ccd.data if isinstance(ccd, CCDData) else ccd

    if sky_per_pix_avg is not None and sky_per_pix_column is not None:
        raise ValueError(
            "Cannot specify both `sky_per_pix_avg` and `sky_per_pix_column`."
        )

    if sky_per_pix_column is not None:
        if sky_per_pix_column not in sources.colnames:
            raise ValueError(f"Column {sky_per_pix_column} not found in sources table.")
        sky_values = sources[sky_per_pix_column]

    if sky_per_pix_avg is not None:
        # User gave a value for the sky background to subtract
        sky_values = [sky_per_pix_avg] * len(sources)
    elif sky_per_pix_column is None:
        # User didn't give a value for the sky background to subtract
        # so try an of estimate it from the image

        sky_values = [np.nanmedian(ccd.data)] * len(sources)

    fwhm_x = []
    fwhm_y = []
    for source, sky in zip(sources, sky_values, strict=True):
        x = source[x_column]
        y = source[y_column]

        # Cutout2D needs no units on the center position, so remove unit
        # if it is present.
        try:
            x = x.value
            y = y.value
            sky = sky.value
        except AttributeError:
            pass

        # SKY SUBTRACT STUFF!!
        cutout = Cutout2D(data - sky, (x, y), 5 * fwhm_estimate)

        # Mask any NaNs in the data
        nan_mask = np.isnan(cutout.data)
        inp_mask = getattr(ccd, "mask", None)
        if inp_mask is not None:
            inp_mask = inp_mask[cutout.slices_original]
            mask = inp_mask | nan_mask
        else:
            mask = nan_mask

        cutout_xy = cutout.to_cutout_position((x, y))
        match fit_method:
            case FwhmMethods.FIT:
                # Make sure we get an odd fits shape
                fit_shape = int(2 * ((5 * fwhm_estimate) // 2) + 1)

                # fit_fwhm is supposed to handle NaNs automatically but it doesn't
                # as of photutils 2.2.0
                # see https://github.com/astropy/photutils/issues/2029
                # For now replace any NaN with zero and hope for the best.
                cutout.data[nan_mask] = 0
                fit = fit_fwhm(
                    cutout.data,
                    xypos=cutout_xy,
                    fwhm=fwhm_estimate,
                    fit_shape=fit_shape,
                    mask=mask,
                )
                fit = fit[0]

                fwhm_x.append(fit)  # gaussian_sigma_to_fwhm * fit.x_stddev_1)
                fwhm_y.append(fit)  # gaussian_sigma_to_fwhm * fit.y_stddev_1)
                # print('Still fitting!!')
            case FwhmMethods.MOMENTS:
                sc = data_properties(cutout.data)

                fwhm_xm = sc.fwhm.value
                fwhm_ym = fwhm_xm
                fwhm_x.append(fwhm_xm)
                fwhm_y.append(fwhm_ym)
            case FwhmMethods.PROFILE:
                radii = np.arange(int(3 * fwhm_estimate))
                profile = RadialProfile(cutout.data, cutout_xy, radii)
                fwhm = profile.gaussian_fwhm
                fwhm_x.append(fwhm)
                fwhm_y.append(fwhm)
            case _:
                raise ValueError(f"Unknown fit method: {fit_method}")

    return np.array(fwhm_x), np.array(fwhm_y)


def source_detection(
    ccd,
    fwhm=8,
    sigma=3.0,
    iters=5,
    threshold=10.0,
    stddev=None,
    find_fwhm=True,
    sky_per_pix_avg=0,
    padding=0,
    verbose=False,
):
    """
    Returns an SourceListData object containing the position of sources
    within the image identified using `photutils.DAOStarFinder` algorithm.
    It can also compute the FWHM of each source by fitting a 2D Gaussian
    which can be useful for choosing a useful aperture size for photometry.

    Parameters
    ----------

    ccd : `numpy.ndarray` or `astropy.nddata.CCDData`
        This is the 2-D array of the image to be analyzed.  If
        a CCDData object is passed, its data attribute will be used.

    fwhm : float, optional (default=8)
        Initial estimate of full-width half-max of stars in the image,
        in pixels.

    sigma : float, optional. (default=3.0)
        The number of standard deviations to use as the lower and
        upper clipping limit when calculating the image background.

    iters : int, optional (default=5)
        The number of iterations to perform sigma clipping

    threshold : float, optional. (default=10.0)
        The number of standard deviations above which to select sources.

    stddev : float, optional
        If provided, this value will be used as the standard deviation
        of the image.  If not provided, the standard deviation will be
        calculated using `sigma_clipped_stats` on the image.

    find_fwhm : bool, optional (default=True)
        If ``True``, estimate the FWHM of each source by fitting a 2D Gaussian
        to it. If ``False``, the 'fwhm_x', 'fwhm_y', and 'width' columns of
        output table will be filled with NaN values.

    sky_per_pix_avg : float or None, optional (default=None)
        Sky background to subtract before centroiding. If set to ``None``,
        it will be estimated using the mean of the sigma_clipped_stats
        of the image.

    padding : int, optional (default=0)
        Distance from the edge of the image to ignore when searching for
        sources.

    verbose : bool, optional
        If ``True``, print additional information about the source
        detection process.

    Returns
    -------

    sources: `stellardev.SourceListData`
        A table of the positions of sources in the image.  If `find_fwhm` is
        ``True``, includes columns for `fwhm_x`, `fwhm_y, and `width`.  If
        the input CCDData object has WCS information, the columns `ra` and
        `dec` are also populated, otherwise they are set to NaN.
    """
    # if image is a CCDData object, extract the data array
    if not isinstance(ccd, CCDData) and not isinstance(ccd, np.ndarray):
        raise ValueError("ccd must be a numpy array or CCDData object")

    # If user uses units for fwhm or sky_per_pix_avg, convert to value
    if isinstance(sky_per_pix_avg, u.Quantity):
        sky_per_pix_avg = sky_per_pix_avg.value
    if isinstance(fwhm, u.Quantity):
        fwhm = fwhm.value

    if verbose:
        print(
            "source_detection: You may see a warning about invalid values in the "
            "input image.  This is expected if any pixels are saturated and can be "
            "ignored."
        )

    # Get statistics of the input image (and use them to estimate the sky background
    # if not provided).  Using clipped stats should hopefully get rid of any
    # bright stars that might be in the image. Only do this if stddev is not provided.
    if stddev is None:
        mean, median, stddev = sigma_clipped_stats(ccd, sigma=sigma, maxiters=iters)
    else:
        # Set median to None to indicate sigma clipped stats were not
        # calculated.
        median = None

    if sky_per_pix_avg is None:
        if median is not None:
            # We calculated sigma clipped stats, so use them
            sky_per_pix_avg = median
        else:
            # A median is a pretty good estimate of the sky background, so use it
            # for detection. For *photometry* a better estimate is needed, but for
            # detection, the median is good enough and much faster than sigma clipping.
            sky_per_pix_avg = np.nanmedian(ccd.data)
        if verbose:
            print(f"source_detection: sky_per_pix_avg set to {sky_per_pix_avg:.4f}")

    # Identify sources applying DAOStarFinder to a "sky subtracted"
    # image.
    if verbose:
        print(
            f"source_detection: threshold set to {threshold}* standard deviation "
            f"({stddev:.4f})"
        )
        print(f"source_detection: Assuming fwhm of {fwhm} for DAOStarFinder")

    # daofind should be run on background subtracted image
    # (fails, or at least returns garbage, if sky_per_pix_avg is too low)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * stddev)
    if isinstance(ccd, CCDData):
        sources = daofind(ccd.data - sky_per_pix_avg)
    else:
        sources = daofind(ccd - sky_per_pix_avg)

    # Identify sources near the edge of the image and remove them
    # from the source list.
    padding_smt = ""
    if padding > 0:
        src_cnt0 = len(sources)
        y_lim, x_lim = ccd.shape
        keep = (
            (sources["xcentroid"].value >= padding)
            & (sources["ycentroid"].value >= padding)
            & (sources["xcentroid"].value < x_lim - padding)
            & (sources["ycentroid"].value < y_lim - padding)
        )
        sources = sources[keep]
        padding_smt = f" (after removing {src_cnt0-len(sources)} sources near the edge)"

    src_cnt = len(sources)
    if verbose:
        print(f"source_detection: {src_cnt} sources identified{padding_smt}.")

    # If image as WCS, compute RA and Dec of each source
    try:
        # Retrieve the RA and Dec of each source as SKyCoord objects, then convert to
        # arrays of floats to add to table
        skypos = ccd.wcs.pixel_to_world(sources["xcentroid"], sources["ycentroid"])
        sources["ra"] = skypos.ra.value
        sources["dec"] = skypos.dec.value
    except AttributeError:
        # No WCS, so add empty columns
        sources["ra"] = np.nan * np.ones(src_cnt)
        sources["dec"] = np.nan * np.ones(src_cnt)

    # If requested, compute the FWHM of each source
    if find_fwhm:
        x, y = compute_fwhm(
            ccd,
            sources,
            fwhm_estimate=fwhm,
            x_column="xcentroid",
            y_column="ycentroid",
            sky_per_pix_avg=sky_per_pix_avg,
        )
        sources["fwhm_x"] = x
        sources["fwhm_y"] = y
        sources["width"] = (x + y) / 2  # Average of x and y FWHM

        # Flag bogus fwhm values returned from fitting (no objects
        # have a fwhm less than 1 pixel)
        bad_src = (sources["fwhm_x"] < 1) | (sources["fwhm_y"] < 1)
        sources["fwhm_x"][bad_src] = np.nan
        sources["fwhm_y"][bad_src] = np.nan
        sources["width"][bad_src] = np.nan
    else:  # add empty columns
        sources["fwhm_x"] = np.nan * np.ones(src_cnt)
        sources["fwhm_y"] = np.nan * np.ones(src_cnt)
        sources["width"] = np.nan * np.ones(src_cnt)

    # Convert sources to SourceListData object by adding
    # units to the columns
    units_dict = {
        "id": None,
        "ra": u.deg,
        "dec": u.deg,
        "xcentroid": u.pix,
        "ycentroid": u.pix,
        "fwhm_x": u.pix,
        "fwhm_y": u.pix,
        "width": u.pix,
    }
    sources = Table(data=sources, units=units_dict)
    # Rename columns to match SourceListData
    colnamemap = {"id": "star_id", "xcentroid": "xcenter", "ycentroid": "ycenter"}

    sl_data = SourceListData(input_data=sources, colname_map=colnamemap)
    return sl_data


def fast_fwhm_from_image(
    ccd,
    fwhm_estimate,
    noise=10,
    n_brightest_sources=30,
    max_adu=40000,
    block_size=8,
    min_block_fwhm=1,
    aggregate_by="mean",
):
    """
    Compute the FWHM of a CCD image by block reducing it, running source detection
    on the reduced image, and then computing the FWHM of the detected sources in the
    original image.

    Parameters
    ----------
    ccd : `numpy.ndarray` or `astropy.nddata.CCDData`
        The CCD image array.
    fwhm_estimate : float
        The initial estimate for the FWHM of the sources in the image.
    noise : float, optional
        The estimated noise in the image.
    n_brightest_sources : int, optional
        The number of brightest sources to use for the FWHM estimate.
    max_adu : float, optional
        The maximum ADU value for a pixel in the image. Sources with peak
        fluxes greater than this value will be ignored.
    block_size : int, optional
        The size of the blocks to use for block reduction. The image will be
        divided into blocks of this size, and source detection will be done on the
        reduced image.
    min_block_fwhm : float, optional
        The minimum FWHM to allow after the block reduction. If the estimated FWHM
        is smaller than this value, the block size will be changed.
    aggregate_by : str, optional
        The method to use for aggregating the FWHM estimates. Can be 'mean' or
        'median'. If None, the FWHM estimates will not be aggregated.

    Returns
    -------
    fwhm : float or np.ndarray
        The FWHM of the sources in the image. If `aggregate_by` is None, this will
        be an array of FWHM values for each source. Otherwise, it will be a single
        value.

    Notes
    -----

    The approach here is to detect sources in the image and measure their FWHM. To
    keep this reasonably fast, we block reduce the image and then run source detection
    on the reduced image. We then use the positions of the detected sources to
    estimate the FWHM of the sources in the original image. Only the brightest
    sources are used to estimate the FWHM, and sources that are too bright (i.e.,
    have a peak flux greater than `max_adu`) are ignored.
    """
    # Check whether the reduced fwhm is larger than the minimum and reset block_size
    # if it is too small.
    if fwhm_estimate / block_size < min_block_fwhm:
        block_size = int(fwhm_estimate / min_block_fwhm)

    # Get data and mask from CCDData object
    if isinstance(ccd, CCDData):
        data = ccd.data
        mask = ccd.mask
    else:
        data = ccd
        mask = None

    with warnings.catch_warnings():
        # block_reduce generates some warnings about things like unit, wcs, etc
        # that are set on ccd but not preserved in the reduced image.
        # That is expected, so ignore it.
        warnings.filterwarnings(
            "ignore",
            message="The following attributes were set on the data object",
            category=AstropyUserWarning,
        )
        reduced_data = block_reduce(data, block_size=block_size)

    # Pick a padding that is about 1% of the block reduced image size
    padding = int(0.01 * reduced_data.shape[0])

    block_sources = source_detection(
        reduced_data,
        fwhm=fwhm_estimate / block_size,
        stddev=noise * block_size,  # noise adds in quadrature
        sky_per_pix_avg=None,  # make source_detection do the sky subtraction
        find_fwhm=False,
        threshold=20,
        padding=padding,
    )

    # Reverse sort by peak flux
    block_sources.sort("peak", reverse=True)
    # To estimate whether the sources are too bright, divide the peak
    # flux by the area of the block.  If this is greater than max_adu,
    # then the sources are too bright.
    too_bright = block_sources["peak"] / block_size**2 > max_adu

    # Drop the sources that are too bright
    block_sources = block_sources[~too_bright]

    # Only keep the n_brightest sources (eventually)
    n_brightest = min(len(block_sources), int(n_brightest_sources))
    # Pad n_brightest a bit in case some of these end up being too bright
    fwhm_est_sources = block_sources[: int(1.2 * n_brightest)]

    # Estimate the x and y positions of the sources in the original image
    fwhm_est_sources["xcenter"] = block_size * (fwhm_est_sources["xcenter"].value + 0.5)
    fwhm_est_sources["ycenter"] = block_size * (fwhm_est_sources["ycenter"].value + 0.5)

    # Compute the FWHM of the sources in the original image
    # Make sure fit_shape is an odd number about 5 times the estimated fwhm
    fit_shape = int(2 * ((5 * fwhm_estimate) // 2) + 1)

    # This next bit is a little sneaky. We mask any values in the image larger than
    # max_adu. After we do the PSF fitting, we only keep results in which there were
    # no masked pixels in the fit. As a result, stars that are too bright in the image
    # will be ignored in the event they made it through the block reduction.
    mask = np.zeros_like(ccd.data, dtype=bool) if mask is None else mask
    mask |= data > max_adu

    with warnings.catch_warnings():
        # fit_2dgaussian generates some warnings if some of the fits do not converge.
        # This probably does not actually affect real images, but it affects tests.
        # We suppress the warnings here and check the fit results for flags that
        # indicate a bad fit.
        warnings.filterwarnings(
            "ignore",
            message=r"One or more fit\(s\) may not have converged. Please check the",
            category=AstropyUserWarning,
        )
        # This is faster than calling our own compute_fwhm, so do this.
        fit = fit_2dgaussian(
            data - np.nanmedian(data),
            xypos=list(
                zip(
                    fwhm_est_sources["xcenter"],
                    fwhm_est_sources["ycenter"],
                    strict=True,
                )
            ),
            fwhm=fwhm_estimate,
            fix_fwhm=False,
            fit_shape=fit_shape,
            mask=mask,
        )

    results = fit.results

    # Only keep good fits -- this is where saturated pixels get removed,
    # because they were masked in the input image and flags=1 means there
    # were masked pixels in the fit.
    results = results[results["flags"] == 0]

    # Estimates of the FWHM that are larger than the fit_shape are bad, so drop
    # any of those
    results = results[results["fwhm_fit"] < fit_shape]

    # Only keep the desired number of sources
    fwhm = results["fwhm_fit"][:n_brightest]

    if aggregate_by is not None:
        match aggregate_by:
            case "mean":
                fwhm = np.mean(fwhm)
            case "median":
                fwhm = np.median(fwhm)
            case _:
                raise ValueError(f"Unknown aggregate_by method: {aggregate_by}")

    return fwhm
