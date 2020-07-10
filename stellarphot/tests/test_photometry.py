import pytest
import numpy as np

from stellarphot.photometry import calculate_noise
from stellarphot.core import Camera

GAINS = [1.0, 1.5, 2.0]


def test_calc_noise_defaults():
    # If we put in nothing we should get zero back
    assert calculate_noise() == 0


@pytest.mark.parametrize('aperture_area', [5, 20])
@pytest.mark.parametrize('gain', GAINS)
def test_calc_noise_source_only(gain, aperture_area):
    # If the only source of noise is Poisson error in the source
    # then the noise should be the square root of the counts.
    counts = 100
    expected = np.sqrt(gain * counts)

    np.testing.assert_allclose(calculate_noise(gain=gain,
                                               flux=counts,
                                               aperture_area=aperture_area),
                               expected)


@pytest.mark.parametrize('aperture_area', [5, 20])
@pytest.mark.parametrize('gain', GAINS)
def test_calc_noise_dark_only(gain, aperture_area):
    # Gain should not affect this one. Dark current needs a couple other things,
    # but this is basically Poisson error.
    dark_current = 10
    exposure = 20
    expected = np.sqrt(dark_current * aperture_area * exposure)

    np.testing.assert_allclose(calculate_noise(gain=gain,
                                               dark_current_per_sec=dark_current,
                                               aperture_area=aperture_area,
                                               exposure=exposure),
                               expected)


@pytest.mark.parametrize('aperture_area', [5, 20])
@pytest.mark.parametrize('gain', GAINS)
def test_calc_read_noise_only(gain, aperture_area):
    # The read noise per pixel IS the noise. The only multiplier is
    # the number of pixels.
    read_noise = 10
    expected = np.sqrt(aperture_area * read_noise**2)

    np.testing.assert_allclose(calculate_noise(gain=gain,
                                               read_noise=read_noise,
                                               aperture_area=aperture_area),
                               expected)


@pytest.mark.parametrize('aperture_area', [5, 20])
@pytest.mark.parametrize('gain', GAINS)
def test_calc_sky_only(gain, aperture_area):
    # The sky noise per pixel is the poisson and per pixel.
    sky = 10
    expected = np.sqrt(gain * aperture_area * sky)

    np.testing.assert_allclose(calculate_noise(gain=gain,
                                               aperture_area=aperture_area,
                                               sky_per_pix=sky),
                               expected)


def test_annulus_area_term():
    # Test that noise is correct with an annulus
    aperture_area = 20

    # Annulus is typically quite a bit larger than aperture.
    annulus_area = 10 * aperture_area
    gain = 1.5
    sky = 10
    expected = np.sqrt(gain * aperture_area *
                       (1 + aperture_area / annulus_area) * sky)
    np.testing.assert_allclose(calculate_noise(gain=gain,
                                               aperture_area=aperture_area,
                                               annulus_area=annulus_area,
                                               sky_per_pix=sky),
                               expected)


@pytest.mark.parametrize('digit,expected',
                         ((False, 89.078616), (True, 89.10182)))
def test_calc_noise_messy_case(digit, expected):
    # Do a single test where all the parameters are set and compare with
    # what a calculator gave.
    counts = 1000

    aperture_area = 20
    annulus_area = 10 * aperture_area

    gain = 1.5
    sky = 15
    dark_current = 7
    exposure = 18
    read_noise = 12

    np.testing.assert_allclose(
        calculate_noise(flux=counts,
                        gain=gain,
                        dark_current_per_sec=dark_current,
                        read_noise=read_noise,
                        sky_per_pix=sky,
                        exposure=exposure,
                        aperture_area=aperture_area,
                        annulus_area=annulus_area,
                        include_digitization=digit),
        expected
    )
