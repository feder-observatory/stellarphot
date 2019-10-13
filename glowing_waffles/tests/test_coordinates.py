import numpy as np

from astropy.wcs import WCS
from astropy.nddata import CCDData

from ..coordinates import convert_pixel_wcs


def test_coord_conversion():
    wcs = WCS(naxis=2)
    # Numbering of pixels for crpix starts at 1....
    wcs.wcs.crpix = [5., 5.]
    wcs.wcs.cdelt = [1, 1]
    wcs.wcs.crval = [10, 5]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    ccd = CCDData(np.ones([10, 10]), wcs=wcs, unit='adu')
    # Pixel values below should give back ra, dec values for crval
    ra, dec = convert_pixel_wcs(ccd, 4, 4)
    print(ra, dec)
    np.testing.assert_almost_equal(ra, wcs.wcs.crval[0])
    np.testing.assert_almost_equal(dec, wcs.wcs.crval[1])
    # These pixel values should be one larger than the crval since cdelt is 1
    ra, dec = convert_pixel_wcs(ccd, 5, 5)
    np.testing.assert_almost_equal(ra, wcs.wcs.crval[0] + 1, decimal=2)
    np.testing.assert_almost_equal(dec, wcs.wcs.crval[1] + 1, decimal=2)

    # Now transform back...
    x, y = convert_pixel_wcs(ccd, ra, dec, is_pix=False)
    np.testing.assert_almost_equal(x, 5)
    np.testing.assert_almost_equal(y, 5)
