import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.io import ascii
from astropy.utils.data import get_pkg_data_filename
from fake_image import FakeCCDImage

from stellarphot.core import Camera, SourceListData
from stellarphot.photometry import (
    calculate_noise,
    find_too_close,
    single_image_photometry,
    source_detection,
)

GAINS = [1.0, 1.5, 2.0]

def test_calc_noise_defaults():
    # If we put in nothing we should get an error about is missing camera
    # instance.
    with pytest.raises(ValueError):
        assert calculate_noise() == 0


@pytest.mark.parametrize('aperture_area', [5, 20])
@pytest.mark.parametrize('gain', GAINS)
def test_calc_noise_source_only(gain, aperture_area):
    # If the only source of noise is Poisson error in the source
    # then the noise should be the square root of the counts.
    counts = 100
    expected = np.sqrt(gain * counts)

    # Create camera instance
    camera = Camera(gain=gain*u.electron/u.adu, read_noise=0*u.electron,
                    dark_current=0*u.electron/u.second,
                    pixel_scale=1*u.arcsec/u.pixel)

    np.testing.assert_allclose(calculate_noise(camera,
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

    # Create camera instance
    camera = Camera(gain=gain*u.electron/u.adu,
                    read_noise=0*u.electron,
                    dark_current=dark_current*u.electron/u.second,
                    pixel_scale=1*u.arcsec/u.pixel)

    expected = np.sqrt(dark_current * aperture_area * exposure)

    np.testing.assert_allclose(calculate_noise(camera,
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

    # Create camera instance
    camera = Camera(gain=gain*u.electron/u.adu,
                    read_noise=read_noise*u.electron,
                    dark_current=0*u.electron/u.second,
                    pixel_scale=1*u.arcsec/u.pixel)

    np.testing.assert_allclose(calculate_noise(camera,
                                               aperture_area=aperture_area),
                               expected)


@pytest.mark.parametrize('aperture_area', [5, 20])
@pytest.mark.parametrize('gain', GAINS)
def test_calc_sky_only(gain, aperture_area):
    # The sky noise per pixel is the poisson and per pixel.
    sky = 10
    expected = np.sqrt(gain * aperture_area * sky)

    # Create camera instance
    camera = Camera(gain=gain*u.electron/u.adu,
                    read_noise=0*u.electron,
                    dark_current=0*u.electron/u.second,
                    pixel_scale=1*u.arcsec/u.pixel)

    np.testing.assert_allclose(calculate_noise(camera,
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

    # Create camera instance
    camera = Camera(gain=gain*u.electron/u.adu,
                    read_noise=0*u.electron,
                    dark_current=0*u.electron/u.second,
                    pixel_scale=1*u.arcsec/u.pixel)

    np.testing.assert_allclose(calculate_noise(camera,
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

    # Create camera instance
    camera = Camera(gain=gain*u.electron/u.adu,
                    read_noise=read_noise*u.electron,
                    dark_current=dark_current*u.electron/u.second,
                    pixel_scale=1*u.arcsec/u.pixel)

    np.testing.assert_allclose(
        calculate_noise(camera,
                        counts=counts,
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
    sl_test_noimgpos = SourceListData(input_data=test_sl_data_noimgpos,
                                      colname_map=None)

    assert sl_test.has_ra_dec is True
    assert sl_test.has_x_y is True
    assert sl_test_nosky.has_ra_dec is False
    assert sl_test_nosky.has_x_y is True
    assert sl_test_noimgpos.has_ra_dec is True
    assert sl_test_noimgpos.has_x_y is False

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

# Constants for the following tests
shift_tolerance = 6
max_adu = 60000
fwhm_estimate = 5
fake_camera = Camera(gain = 1.0*u.electron/u.adu,
                     read_noise = 0*u.electron,
                     dark_current = 0.1*u.electron/u.second,
                     pixel_scale = 1*u.arcsec/u.pixel)
fake_obs = EarthLocation(lat = 0*u.deg,
                         lon = 0*u.deg,
                         height = 0*u.m)
coords2use='pixel'

def test_aperture_photometry_no_outlier_rejection():
    fake_CCDimage = FakeCCDImage()
    sources = fake_CCDimage.sources
    aperture = sources['aperture'][0]
    inner_annulus = 2 * aperture
    outer_annulus = 3 * aperture
    found_sources = source_detection(fake_CCDimage,
                                    fwhm=sources['x_stddev'].mean(),
                                    threshold=10)
    phot, missing_sources = single_image_photometry(fake_CCDimage,
                                                    found_sources,
                                                    fake_camera,
                                                    fake_obs,
                                                    aperture, inner_annulus,
                                                    outer_annulus,
                                                    shift_tolerance,
                                                    max_adu, fwhm_estimate,
                                                    use_coordinates=coords2use,
                                                    include_dig_noise=True,
                                                    reject_too_close=False,
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

        # We could require that the result be within some reasonable
        # number of those expected variations or we could count up the
        # actual number of background counts at each of the source
        # positions.

        # Here we just check whether any difference is consistent with
        # less than the expected one sigma deviation.
        assert (np.abs(expected_flux - out['aperture_net_cnts'].value) <
                np.pi * aperture**2 * fake_CCDimage.noise_dev)


@pytest.mark.parametrize('reject', [True, False])
def test_aperture_photometry_with_outlier_rejection(reject):
    """
    Insert some really large pixel values in the annulus and check that
    the photometry is correct when outliers are rejected and is
    incorrect when outliers are not rejected.
    """
    fake_CCDimage = FakeCCDImage()
    sources = fake_CCDimage.sources
    aperture = sources['aperture'][0]
    inner_annulus = 2 * aperture
    outer_annulus = 3 * aperture
    image = fake_CCDimage.data

    found_sources = source_detection(fake_CCDimage,
                                    fwhm=sources['x_stddev'].mean(),
                                    threshold=10)

    # Add some large pixel values to the annulus for each source.
    # adding these moves the average pixel value by quite a bit,
    # so we'll only get the correct net flux if these are removed.
    for source in fake_CCDimage.sources:
        center_px = (int(source['x_mean']), int(source['y_mean']))
        begin = center_px[0] + inner_annulus + 1
        end = begin + (outer_annulus - inner_annulus - 1)
        # Yes, x and y are deliberately reversed below.
        image[center_px[1], begin:end] = 100 * fake_CCDimage.mean_noise

    phot, missing_sources = single_image_photometry(fake_CCDimage,
                                                    found_sources,
                                                    fake_camera,
                                                    fake_obs,
                                                    aperture, inner_annulus,
                                                    outer_annulus,
                                                    shift_tolerance,
                                                    max_adu, fwhm_estimate,
                                                    use_coordinates=coords2use,
                                                    include_dig_noise=True,
                                                    reject_too_close=False,
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

        expected_deviation = np.pi * aperture**2 * fake_CCDimage.noise_dev
        # We could require that the result be within some reasonable
        # number of those expected variations or we could count up the
        # actual number of background counts at each of the source
        # positions.

        # Here we just check whether any difference is consistent with
        # less than the expected one sigma deviation.
        if reject:
            assert (np.abs(expected_flux - out['aperture_net_cnts'].value) <
                    expected_deviation)
        else:
            with pytest.raises(AssertionError):
                assert (np.abs(expected_flux - out['aperture_net_cnts'].value) <
                        expected_deviation)