
import astropy.io.fits as fits
import numpy as np
from astropy.nddata import CCDData
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from photutils.datasets import make_gaussian_sources_image, make_noise_image


class FakeImage:
    """
    Creates a fake image with a set of sources using the datafile stored
    at `data/test_sources.csv`.

    Parameters
    ----------

    noise_dev : float, optional
        The standard deviation of the noise in the image.

    seed : int, optional
        The seed to use for the random number generator. If not specified,
        the seed will be randomly generated.
    """
    def __init__(self, noise_dev=1.0, seed=None):
        self.image_shape = [400, 500]
        data_file = get_pkg_data_filename('data/test_sources.csv')
        self._sources = Table.read(data_file)
        self.mean_noise = self.sources['amplitude'].max() / 100
        self.noise_dev = noise_dev
        self._stars = make_gaussian_sources_image(self.image_shape,
                                                  self.sources)
        self._noise = make_noise_image(self._stars.shape,
                                       mean=self.mean_noise,
                                       stddev=noise_dev,
                                       seed=seed)
        # Sky background per pixel should be the mean level of the noise.
        self._sources['sky_per_pix_avg'] = self.mean_noise

    @property
    def sources(self):
        """
        Return the table of sources used to create the fake image.
        """
        return self._sources

    @property
    def image(self):
        """
        Return the fake image.
        """
        return self._stars + self._noise


class FakeCCDImage(CCDData):
    # Generates a fake CCDData object for testing purposes.
    def __init__(self, *args, **kwargs):
        # Pull off the seed argument if it exists.
        seed = kwargs.pop('seed', None)

        # If no arguments are passed, use the default FakeImage.
        # This dodge is necessary because otherwise we can't copy the CCDData
        # object apparently.
        if (len(args) == 0) and (len(kwargs) == 0):
            base_data = FakeImage(seed=seed)
            super().__init__(base_data.image.copy(), unit='adu')

            # Append attributes from the base data object.
            self.sources = base_data.sources.copy()
            self.noise_dev = base_data.noise_dev
            self.mean_noise = base_data.mean_noise
            self.image_shape = base_data.image_shape.copy()

            # Add some additional features to the CCDData object, like
            # a header and the sources used to create the image.abs
            self.header = fits.Header()
            self.header['OBJECT'] = 'Test Object'
            self.header['EXPOSURE'] = 1.0
            self.header['DATE-OBS'] = '2018-01-01T00:00:00.0'
            self.header['AIRMASS'] = 1.2
            self.header['FILTER'] = 'V'

            # Set up a  WCS header for the CCDData object.
            (size_y, size_x) = base_data.image_shape
            pixel_scale = 0.75 # arcseconds per pixel
            ra_center = 283.6165
            dec_center = 33.05857
            w = WCS(naxis=2)
            w.wcs.crpix = [size_x / 2, size_y / 2]  # Reference pixel (center of the image)
            w.wcs.cdelt = [-pixel_scale / 3600, pixel_scale / 3600]  # Pixel scale in degrees per pixel
            w.wcs.crval = [ra_center, dec_center]  # RA and Dec of the reference pixel in degrees
            w.wcs.ctype = ['RA---TAN', 'DEC--TAN']  # Coordinate type (TAN projection)
            # Rotate image to be slightly off of horizontal
            north_angle_deg = 8.4
            w.wcs.pc = [[np.cos(np.radians(north_angle_deg)), -np.sin(np.radians(north_angle_deg))],
                        [np.sin(np.radians(north_angle_deg)), np.cos(np.radians(north_angle_deg))]]

            self.wcs = w
            self.header.update(w.to_header())

    def drop_wcs(self):
        # Convienence function to remove WCS information from the CCDData object
        # for testing purposes.
        self.wcs = None
        wcs_keywords = ['CTYPE', 'CRPIX', 'CRVAL', 'CDELT','CUNIT',
                        'CD1_', 'CD2_', 'PC1_', 'PC2_']
        for keyword in wcs_keywords:
            matching_keys =  [key for key in self.header.keys() if keyword in key]
            for key in matching_keys:
                del self.header[key]


def shift_FakeCCDImage(ccd_data, x_shift, y_shift):
    """
    Create a test CCD image based on the first test image with the central
    positions shifted by the given amount.  To prevent any sources being
    shifted off the edge of the image, an error is thrown if this is
    happening.

    WCS information in the CCDData is also updated to reflect the shift.

    Parameters:
        ccd_data : `FakeCCDImage`
            The original CCDData object.

        x_shift : int
            The amount of shift in x direction (in pixels)

        y_shift : int
            The amount of shift in y direction (in pixels)

    Returns:
        `FakeCCDImage`: A new CCDData object.
    """
    # Copy WCS from original CCDData object
    shifted_ccd_data  = ccd_data.copy()
    for key, val in ccd_data.__dict__.items():
        try:
            shifted_ccd_data.__dict__[key] = val.copy()
        except AttributeError:
            shifted_ccd_data.__dict__[key] = val
    shifted_wcs = ccd_data.wcs.deepcopy()

    # Calculate the new RA and Dec center after shifting
    x_shift = int(x_shift)
    y_shift = int(y_shift)
    ra_center_shifted, dec_center_shifted = \
        shifted_wcs.all_pix2world((shifted_ccd_data.data.shape[1]) / 2 + x_shift,
                                  (shifted_ccd_data.data.shape[0]) / 2 + y_shift, 0)

    # Shift source positions
    shifted_ccd_data.sources['x_mean'] -= x_shift
    shifted_ccd_data.sources['y_mean'] -= y_shift

    # Check shifted sources are still on the image
    if ( (np.any(shifted_ccd_data.sources['x_mean'] < 0)) |
         (np.any(shifted_ccd_data.sources['x_mean'] > shifted_ccd_data.data.shape[1])) |
         (np.any(shifted_ccd_data.sources['y_mean'] < 0)) |
         (np.any(shifted_ccd_data.sources['y_mean'] > shifted_ccd_data.data.shape[0])) ):
        raise ValueError('Sources shifted off the edge of the image.')

    # Update the new RA and Dec center in the shifted WCS
    shifted_wcs.wcs.crval = [ra_center_shifted, dec_center_shifted]
    shifted_ccd_data.wcs = shifted_wcs

    # Make image
    srcs = make_gaussian_sources_image(shifted_ccd_data.image_shape,
                                       shifted_ccd_data.sources)
    background = make_noise_image(srcs.shape,
                                  mean=shifted_ccd_data.mean_noise,
                                  stddev=shifted_ccd_data.noise_dev)
    shifted_ccd_data.data = srcs + background

    return shifted_ccd_data
