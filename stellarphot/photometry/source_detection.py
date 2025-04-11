import numpy as np
from astropy import units as u
from astropy.nddata import CCDData
from astropy.nddata.utils import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from photutils.detection import DAOStarFinder
from photutils.morphology import data_properties
from photutils.profiles import RadialProfile
from photutils.psf import fit_fwhm

from stellarphot.core import SourceListData
from stellarphot.settings.models import FwhmMethods

__all__ = ["source_detection", "compute_fwhm"]


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

    ccd : `astropy.nddata.CCDData`
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

        # SKY SUBTRACT SHIT!!
        cutout = Cutout2D(ccd.data - sky, (x, y), 5 * fwhm_estimate)

        # Mask any NaNs in the data
        nan_mask = np.isnan(cutout.data)
        inp_mask = getattr(ccd, "mask", False)
        if inp_mask:
            inp_mask = inp_mask[cutout.slices_original()]
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
    find_fwhm=True,
    sky_per_pix_avg=0,
    padding=0,
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

    # Get statistics of the input image (and use them to estimate the sky background
    # if not provided).  Using clipped stats should hopefully get rid of any
    # bright stars that might be in the image, so the mean should be a good
    # estimate of the sky background.
    print(
        "source_detection: You may see a warning about invalid values in the "
        "input image.  This is expected if any pixels are saturated and can be "
        "ignored."
    )
    mean, median, std = sigma_clipped_stats(ccd, sigma=sigma, maxiters=iters)
    print(
        f"source_detection: sigma_clipped_stats mean={mean:.4f}, "
        f"median={median:.4f}, std={std:.4f}"
    )
    if sky_per_pix_avg is None:
        sky_per_pix_avg = mean
        print(f"source_detection: sky_per_pix_avg set to {sky_per_pix_avg:.4f}")

    # Identify sources applying DAOStarFinder to a "sky subtracted"
    # image.
    print(
        f"source_detection: threshold set to {threshold}* standard deviation "
        f"({std:.4f})"
    )
    print(f"source_detection: Assuming fwhm of {fwhm} for DAOStarFinder")
    # daofind should be run on background subtracted image
    # (fails, or at least returns garbage, if sky_per_pix_avg is too low)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
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
