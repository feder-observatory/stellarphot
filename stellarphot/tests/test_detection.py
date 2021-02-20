import numpy as np
import pytest

from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table
from astropy.stats import gaussian_sigma_to_fwhm
from astropy import units as u

from photutils.datasets import make_gaussian_sources_image, make_noise_image

from stellarphot.source_detection import source_detection, compute_fwhm
from stellarphot.photometry import photutils_stellar_photometry


class FakeImage:
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
        self._sources['sky_per_pix_avg'] = noise_dev

    @property
    def sources(self):
        return self._sources

    @property
    def image(self):
        return self._stars + self._noise


@pytest.mark.parametrize('units', [u.pixel, None])
def test_compute_fwhm(units):
    fake_image = FakeImage()
    sources = fake_image.sources
    if units is not None:
        # It turns out having a unit on a column is not the same as
        # things in the column having units. The construct below ensures
        # that the source table values have units.
        # Do not try: sources['x_mean'] = sources['x_mean'] * units
        # Turns out individual values do NOT have units in that case.
        sources['x_mean'] = [v * units for v in sources['x_mean']]
        sources['y_mean'] = [v * units for v in sources['y_mean']]

    fwhm_x, fwhm_y = compute_fwhm(fake_image.image, sources,
                                  x_column='x_mean', y_column='y_mean')


def test_detect_source_number_location():
    """
    Make sure we detect the sources in the input table....
    """
    fake_image = FakeImage()
    sources = fake_image.sources
    print(sources)
    found_sources = source_detection(fake_image.image,
                                     fwhm=2 * sources['x_stddev'].mean(),
                                     threshold=10,
                                     sky_per_pix_avg=sources['sky_per_pix_avg'])
    # Sort by flux so we can reliably match them
    sources.sort('amplitude')
    found_sources.sort('flux')

    # Do we have the right number of sources?
    assert len(sources) == len(found_sources)

    for inp, out in zip(sources, found_sources):
        # Do the positions match?
        np.testing.assert_allclose(out['xcentroid'], inp['x_mean'],
                                   rtol=1e-5, atol=0.05)
        np.testing.assert_allclose(out['ycentroid'], inp['y_mean'],
                                   rtol=1e-5, atol=0.05)
        np.testing.assert_allclose(gaussian_sigma_to_fwhm * (inp['x_stddev'] + inp['y_stddev']) / 2,
                                   out['FWHM'],
                                   rtol=1e-5, atol=0.05)


def test_aperture_photometry_no_outlier_rejection():
    fake_image = FakeImage()
    sources = fake_image.sources
    aperture = sources['aperture'][0]
    found_sources = source_detection(fake_image.image,
                                     fwhm=sources['x_stddev'].mean(),
                                     threshold=10)
    phot = photutils_stellar_photometry(fake_image.image,
                                        found_sources, aperture,
                                        2 * aperture, 3 * aperture,
                                        reject_background_outliers=False)
    phot.sort('aperture_sum')
    sources.sort('amplitude')
    found_sources.sort('flux')

    for inp, out in zip(sources, phot):
        stdev = inp['x_stddev']
        expected_flux = (inp['amplitude'] * 2 * np.pi *
                         stdev**2 *
                         (1 - np.exp(-aperture**2 / (2 * stdev**2))))
        # This expected flux is correct IF there were no noise. With noise, the
        # standard deviation in the sum of the noise within in the aperture is
        # n_pix_in_aperture times the single-pixel standard deviation.
        #
        # We could require that the result be within some reasonable
        # number of those expected variations or we could count up the
        # actual number of background counts at each of the source
        # positions.

        # Here we just check whether any difference is consistent with
        # less than the expected one sigma deviation.
        assert (np.abs(expected_flux - out['net_flux']) <
                np.pi * aperture**2 * fake_image.noise_dev)


@pytest.mark.parametrize('reject', [True, False])
def test_aperture_photometry_with_outlier_rejection(reject):
    """
    Insert some really large pixel values in the annulus and check that
    the photometry is correct when outliers are rejected and is
    incorrect when outliers are not rejected.
    """
    fake_image = FakeImage()
    sources = fake_image.sources
    aperture = sources['aperture'][0]
    image = fake_image.image

    found_sources = source_detection(image,
                                     fwhm=sources['x_stddev'].mean(),
                                     threshold=10)

    inner_annulus = 2 * aperture
    outer_annulus = 3 * aperture
    # Add some large pixel values to the annulus for each source.
    # adding these moves the average pixel value by quite a bit,
    # so we'll only get the correct net flux if these are removed.
    for source in fake_image.sources:
        center_px = (np.int(source['x_mean']), np.int(source['y_mean']))
        begin = center_px[0] + inner_annulus + 1
        end = begin + (outer_annulus - inner_annulus - 1)
        # Yes, x and y are deliberately reversed below.
        image[center_px[1], begin:end] = 100 * fake_image.mean_noise

    phot = photutils_stellar_photometry(image,
                                        found_sources, aperture,
                                        2 * aperture, 3 * aperture,
                                        reject_background_outliers=reject)
    phot.sort('aperture_sum')
    sources.sort('amplitude')
    found_sources.sort('flux')

    for inp, out in zip(sources, phot):
        stdev = inp['x_stddev']
        expected_flux = (inp['amplitude'] * 2 * np.pi *
                         stdev**2 *
                         (1 - np.exp(-aperture**2 / (2 * stdev**2))))
        # This expected flux is correct IF there were no noise. With noise, the
        # standard deviation in the sum of the noise within in the aperture is
        # n_pix_in_aperture times the single-pixel standard deviation.
        #

        expected_deviation = np.pi * aperture**2 * fake_image.noise_dev
        # We could require that the result be within some reasonable
        # number of those expected variations or we could count up the
        # actual number of background counts at each of the source
        # positions.

        # Here we just check whether any difference is consistent with
        # less than the expected one sigma deviation.
        if reject:
            assert (np.abs(expected_flux - out['net_flux']) <
                    expected_deviation)
        else:
            with pytest.raises(AssertionError):
                assert (np.abs(expected_flux - out['net_flux']) <
                        expected_deviation)
