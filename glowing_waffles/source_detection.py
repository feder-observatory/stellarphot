from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from photutils import fit_2dgaussian
from astropy.nddata.utils import Cutout2D
from astropy.stats import gaussian_sigma_to_fwhm

__all__ = ['source_detection']


def source_detection(ccd, fwhm=3.0, sigma=3.0, iters=5,
                     threshold=5.0, find_fwhm=True):
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
    data = ccd
    mean, median, std = sigma_clipped_stats(data, sigma=sigma, iters=iters)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
    sources = daofind(data - median)
    if find_fwhm:
        fwhm_fit = []
        for source in sources:
            x = source['xcentroid']
            y = source['ycentroid']
            cutout = Cutout2D(data, (x, y), 5 * fwhm)
            fit = fit_2dgaussian(cutout.data)
            fwhm_fit.append(gaussian_sigma_to_fwhm * (fit.x_stddev + fit.y_stddev) / 2)
        sources['FWHM'] = fwhm_fit
    return sources
