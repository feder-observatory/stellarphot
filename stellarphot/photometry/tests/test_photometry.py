import pytest
import numpy as np
from astropy.io import ascii
from astropy.utils.data import get_pkg_data_filename

from stellarphot.photometry import calculate_noise, find_too_close
from stellarphot.core import SourceListData

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
                                               counts=counts,
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
        calculate_noise(counts=counts,
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


def test_find_too_close():
    # Load test sourcelist into memory
    test_sl_data = ascii.read(get_pkg_data_filename('data/test_corner.ecsv'),
                                format='ecsv',
                                fast_reader=False)

    # Create no sky position sourcelist
    test_sl_data_nosky = test_sl_data.copy()
    test_sl_data_nosky.remove_column('ra')
    test_sl_data_nosky.remove_column('dec')

    # Create no image position sourcelist
    test_sl_data_noimgpos = test_sl_data.copy()
    test_sl_data_noimgpos.remove_column('xcenter')
    test_sl_data_noimgpos.remove_column('ycenter')

    # Create SourceListData objects
    sl_test = SourceListData(input_data=test_sl_data, colname_map=None)
    sl_test_nosky = SourceListData(input_data=test_sl_data_nosky, colname_map=None)
    sl_test_noimgpos = SourceListData(input_data=test_sl_data_noimgpos, colname_map=None)

    assert sl_test.has_ra_dec == True
    assert sl_test.has_x_y == True
    assert sl_test_nosky.has_ra_dec == False
    assert sl_test_nosky.has_x_y == True
    assert sl_test_noimgpos.has_ra_dec == True
    assert sl_test_noimgpos.has_x_y == False

    # Test full positions available
    ap_in_asec = 5
    feder_scale = 0.563
    aperture_rad = ap_in_asec / feder_scale

    rejects = find_too_close(sl_test, aperture_rad, pixel_scale=feder_scale)
    assert np.sum(rejects) == 5

    # Test only sky positions available
    rejects = find_too_close(sl_test_noimgpos, aperture_rad, pixel_scale=feder_scale)
    assert np.sum(rejects) == 5

    # Test only image positions available
    rejects = find_too_close(sl_test_nosky, aperture_rad, pixel_scale=feder_scale)
    assert np.sum(rejects) == 5
