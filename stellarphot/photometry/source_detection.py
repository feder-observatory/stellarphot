import numpy as np
from astropy import units as u
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Const2D, Gaussian2D
from astropy.nddata import CCDData
from astropy.nddata.utils import Cutout2D
from astropy.stats import gaussian_sigma_to_fwhm, sigma_clipped_stats
from astropy.table import Table
from photutils.detection import DAOStarFinder
from photutils.morphology import data_properties

from stellarphot.core import SourceListData

__all__ = ['source_detection', 'compute_fwhm']


def _fit_2dgaussian(data):
    """
    Fit a 2D Gaussian to data.

    Written as a replacement for functionality that was removed from
    photutils.

    This function will be kept private so we don't have to support it.

    Copy/pasted from
    https://github.com/astropy/photutils/pull/1064/files#diff-9e64908ff7ac552845b4831870a749f397d73c681d218267bd9087c7757e6dd4R285

    Parameters
    ----------
    data : array-like
        The 2D array of data to fit.

    Returns
    -------
    gfit : `astropy.modeling.Model`
        The best-fit 2D Gaussian model.
    """
    mask = ~np.isfinite(data)

    # If there are non-finite pixels they need to be masked out
    # or many of the returned properties will be NaN.
    props = data_properties(data - np.min(data[~mask]), mask=mask)

    init_const = 0.  # subtracted data minimum above
    # ptp = peak-to-peak, i.e. max - min, need to also exclude non-finite
    # values here.
    init_amplitude = np.ptp(data[~mask])

    g_init = (Const2D(init_const)
                  + Gaussian2D(amplitude=init_amplitude,
                               x_mean=props.xcentroid,
                               y_mean=props.ycentroid,
                               x_stddev=props.semimajor_sigma.value,
                               y_stddev=props.semiminor_sigma.value,
                               theta=props.orientation.value))

    fitter = LevMarLSQFitter()
    y, x = np.indices(data.shape)
    # Call fitter, enabling filtering of NaN/inf values
    gfit = fitter(g_init, x, y, data, filter_non_finite=True)

    return gfit


def compute_fwhm(ccd, sources, fwhm_estimate=5,
                 x_column='xcenter', y_column='ycenter',
                 fit=True,
                 sky_per_pix_avg=0):
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
        Sky background to subtract before centroiding.

    Returns
    -------

    fwhm_x, fwhm_y : tuple of np.array float
        The FWHM of each source in the x and y directions.

    """

    fwhm_x = []
    fwhm_y = []
    for source in sources:
        x = source[x_column]
        y = source[y_column]
        sky = sky_per_pix_avg
        # Cutout2D needs no units on the center position, so remove unit
        # if it is present.
        try:
            x = x.value
            y = y.value
            sky = sky.value
        except AttributeError:
            pass

        cutout = Cutout2D(ccd.data, (x, y), 5 * fwhm_estimate)
        if fit:
            fit = _fit_2dgaussian(cutout.data)
            fwhm_x.append(gaussian_sigma_to_fwhm * fit.x_stddev_1)
            fwhm_y.append(gaussian_sigma_to_fwhm * fit.y_stddev_1)
            # print('Still fitting!!')
        else:
            sc = data_properties(cutout.data - sky)

            # dat = np.where(cutout.data - sky > 0, cutout.data - sky, 0)
            # mom1 = _moments(dat, order=1)
            # xc = mom1[0, 1] / mom1[0, 0]
            # yc = mom1[1, 0] / mom1[0, 0]
            # moments = _moments_central(dat,
            #                            center=(xc, yc), order=2)
            # mom_scale = (moments / mom1[0, 0])
            # fwhm_xm = 2 * np.sqrt(np.log(2) * mom_scale[0, 2])
            # fwhm_ym = 2 * np.sqrt(np.log(2) * mom_scale[2, 0])

            fwhm_xm = sc.fwhm.value
            fwhm_ym = fwhm_xm
            fwhm_x.append(fwhm_xm)
            fwhm_y.append(fwhm_ym)

    return np.array(fwhm_x), np.array(fwhm_y)


def source_detection(ccd, fwhm=8, sigma=3.0, iters=5,
                     threshold=10.0, find_fwhm=True,
                     sky_per_pix_avg=0, padding=0):
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
    print("source_detection: You may see a warning about invalid values in the "
          "input image.  This is expected if any pixels are saturated and can be "
          "ignored.")
    mean, median, std = sigma_clipped_stats(ccd, sigma=sigma, maxiters=iters)
    print(f"source_detection: sigma_clipped_stats mean={mean:.4f}, median={median:.4f}, std={std:.4f}")
    if sky_per_pix_avg is None:
        sky_per_pix_avg = mean
        print(f"source_detection: sky_per_pix_avg set to {sky_per_pix_avg:.4f}")

    # Identify sources applying DAOStarFinder to a "sky subtracted"
    # image.
    print(f"source_detection: threshold set to {threshold}* standard deviation "
          f"({std:.4f})")
    print(f"source_detection: Assuming fwhm of {fwhm} for DAOStarFinder")
    # daofind should be run on background subtracted image
    # (fails, or at least returns garbage, if sky_per_pix_avg is too low)
    daofind = DAOStarFinder(fwhm = fwhm, threshold = threshold * std)
    if isinstance(ccd, CCDData):
        sources = daofind(ccd.data - sky_per_pix_avg)
    else:
        sources = daofind(ccd - sky_per_pix_avg)

    # Identify sources near the edge of the image and remove them
    # from the source list.
    padding_smt = ""
    if (padding > 0):
        src_cnt0 = len(sources)
        y_lim, x_lim = ccd.shape
        keep = ((sources['xcentroid'].value >= padding) & (sources['ycentroid'].value >= padding) &
                (sources['xcentroid'].value < x_lim-padding) & (sources['ycentroid'].value < y_lim-padding))
        sources = sources[keep]
        padding_smt = f" (after removing {src_cnt0-len(sources)} sources near the edge)"

    src_cnt = len(sources)
    print(f"source_detection: {src_cnt} sources identified{padding_smt}.")

    # If image as WCS, compute RA and Dec of each source
    try:
        # Retrieve the RA and Dec of each source as SKyCoord objects, then convert to
        # arrays of floats to add to table
        skypos = ccd.wcs.pixel_to_world(sources['xcentroid'], sources['ycentroid'])
        sources['ra'] = skypos.ra.value
        sources['dec'] = skypos.dec.value
    except AttributeError:
        # No WCS, so add empty columns
        sources['ra'] = np.nan * np.ones(src_cnt)
        sources['dec'] = np.nan * np.ones(src_cnt)

    # If requested, compute the FWHM of each source
    if find_fwhm:
        x, y = compute_fwhm(ccd, sources, fwhm_estimate=fwhm,
                            x_column='xcentroid', y_column='ycentroid',
                            sky_per_pix_avg=sky_per_pix_avg)
        sources['fwhm_x'] = x
        sources['fwhm_y'] = y
        sources['width'] = (x + y) / 2 # Average of x and y FWHM

        # Flag bogus fwhm values returned from fitting (no objects
        # have a fwhm less than 1 pixel)
        bad_src = (sources['fwhm_x']<1) | (sources['fwhm_y']<1)
        sources['fwhm_x'][bad_src] = np.nan
        sources['fwhm_y'][bad_src] = np.nan
        sources['width'][bad_src] = np.nan
    else: # add empty columns
        sources['fwhm_x'] = np.nan * np.ones(src_cnt)
        sources['fwhm_y'] = np.nan * np.ones(src_cnt)
        sources['width'] = np.nan * np.ones(src_cnt)

    # Convert sources to SourceListData object by adding
    # units to the columns
    units_dict = {
        'id' : None,
        'ra' : u.deg,
        'dec' : u.deg,
        'xcentroid' : u.pix,
        'ycentroid' : u.pix,
        'fwhm_x' : u.pix,
        'fwhm_y' : u.pix,
        'width' : u.pix
    }
    sources = Table(data=sources, units=units_dict)
    # Rename columns to match SourceListData
    colnamemap = {'id' : 'star_id',
                  'xcentroid' : 'xcenter',
                  'ycentroid' : 'ycenter'}

    sl_data = SourceListData(input_data=sources, colname_map=colnamemap)
    return sl_data
