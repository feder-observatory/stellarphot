
import astropy.io.fits as fits
from astropy.nddata import CCDData
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename

from photutils.datasets import make_gaussian_sources_image, make_noise_image


class FakeImage:
    """
    Creates a fake image with a set of sources using the datafile stored
    at `data/test_sources.csv`.

    Parameters
    ----------

    noise_dev : float, optional
        The standard deviation of the noise in the image.
    """
    def __init__(self, noise_dev=1.0):
        self.image_shape = [400, 500]
        data_file = get_pkg_data_filename('data/test_sources.csv')
        self._sources = Table.read(data_file)
        self.mean_noise = self.sources['amplitude'].max() / 100
        self.noise_dev = noise_dev
        self._stars = make_gaussian_sources_image(self.image_shape,
                                                  self.sources)
        self._noise = make_noise_image(self._stars.shape,
                                       mean=self.mean_noise,
                                       stddev=noise_dev)
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
    def __init__(self):
        base_data = FakeImage()
        super().__init__(base_data.image.copy(), unit='adu')
        # Add some additional features to the CCDData object, like
        # a header and the sources used to create the image.abs
        self.header = fits.Header()
        self.header['EXPOSURE'] = 1.0
        self.header['DATE-OBS'] = '2018-01-01T00:00:00.0'
        self.header['AIRMASS'] = 1.2
        self.header['FILTER'] = 'V'
        self.sources = base_data.sources.copy()
        self.noise_dev = base_data.noise_dev
        self.mean_noise = base_data.mean_noise
        self.image_shape = base_data.image_shape.copy()
