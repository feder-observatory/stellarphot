import numpy as np

from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from photutils import fit_2dgaussian
from astropy.nddata.utils import Cutout2D
from astropy.stats import gaussian_sigma_to_fwhm
from astropy import units as u

__all__ = ['source_detection', 'compute_fwhm']


def compute_fwhm(ccd, sources, fwhm_estimate=5,
                 x_column='xcenter', y_column='ycenter'):
    fwhm_x = []
    fwhm_y = []
    for source in sources:
        x = source[x_column] / u.pixel
        y = source[y_column] / u.pixel
        cutout = Cutout2D(ccd, (x, y), 5 * fwhm_estimate)
        fit = fit_2dgaussian(cutout.data)
        fwhm_x.append(gaussian_sigma_to_fwhm * fit.x_stddev)
        fwhm_y.append(gaussian_sigma_to_fwhm * fit.y_stddev)
    return np.array(fwhm_x), np.array(fwhm_y)


def source_detection(ccd, fwhm=8, sigma=3.0, iters=5,
                     threshold=10.0, find_fwhm=True):
    """
    Returns an astropy table containing the position of sources
    within the image.

    Parameters
    ----------------

    ccd : numpy.ndarray
        The CCD Image array.

    fwhm : float, optional
        Full-width half-max of stars in the image.

    sigma : float, optional.
        The number of standard deviations to use as the lower and
        upper clipping limit.

    iters : int, optional
        The number of iterations to perform sigma clipping

    threshold : float, optional.
        The absolute image value above which to select sources.

    find_fwhm : bool, optional
        If ``True``, estimate the FWHM of each source by fitting a 2D Gaussian
        to it.

    Returns
    -----------

    sources
        an astropy table of the positions of sources in the image.
        If `find_fwhm` is ``True``, includes a column called ``FWHM``.
    """
    mean, median, std = sigma_clipped_stats(ccd, sigma=sigma, maxiters=iters)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
    sources = daofind(ccd - median)
    if find_fwhm:
        x, y = compute_fwhm(ccd, sources, fwhm_estimate=fwhm)
        sources['FWHM'] = (x + y) / 2
    return sources
