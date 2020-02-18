from astropy.wcs import WCS


def make_wcs():
    wcs = WCS(naxis=2)
    # Numbering of pixels for crpix starts at 1....
    wcs.wcs.crpix = [5., 5.]
    wcs.wcs.cdelt = [1, 1]
    wcs.wcs.crval = [10, 5]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs
