import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.io import ascii
from astropy.utils.data import get_pkg_data_filename
from fake_image import FakeCCDImage, shift_FakeCCDImage

from stellarphot.core import Camera, SourceListData
from stellarphot.photometry import (calculate_noise, find_too_close,
                                    multi_image_photometry,
                                    single_image_photometry, source_detection)
from stellarphot.settings import ApertureSettings

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


# The True case below is a regression test for #157
@pytest.mark.parametrize('int_data', [True, False])
def test_aperture_photometry_no_outlier_rejection(int_data):
    fake_CCDimage = FakeCCDImage()

    sources = fake_CCDimage.sources
    aperture = sources['aperture'][0]
    inner_annulus = 2 * aperture
    outer_annulus = 3 * aperture
    aperture_settings = ApertureSettings(radius=aperture,
                                         gap=inner_annulus - aperture,
                                         width_annulus=outer_annulus - inner_annulus)

    found_sources = source_detection(fake_CCDimage,
                                    fwhm=sources['x_stddev'].mean(),
                                    threshold=10)

    # The scale_factor is used to rescale data to integers if needed. It
    # needs to be set later on when the net counts are "unscaled" in the
    # asserts that constitute the actual test.
    scale_factor = 1.0
    if int_data:
        scale_factor = 0.75 * max_adu / fake_CCDimage.data.max()
        # For the moment, ensure the integer data is NOT larger than max_adu
        # because until #161 is fixed then having NaN in the data will not succeed.
        data = scale_factor * fake_CCDimage.data
        fake_CCDimage.data = data.astype(int)

    phot, missing_sources = single_image_photometry(fake_CCDimage,
                                                    found_sources,
                                                    fake_camera,
                                                    fake_obs,
                                                    aperture_settings,
                                                    shift_tolerance,
                                                    max_adu, fwhm_estimate,
                                                    use_coordinates=coords2use,
                                                    include_dig_noise=True,
                                                    reject_too_close=False,
                                                    reject_background_outliers=False)

    phot.sort('aperture_sum')
    sources.sort('amplitude')

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

        # We need to remove any scaling that has been done of the data values.
        assert (np.abs(expected_flux - out['aperture_net_cnts'].value / scale_factor) <
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

    aperture_settings = ApertureSettings(radius=aperture,
                                         gap=inner_annulus - aperture,
                                         width_annulus=outer_annulus - inner_annulus)

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
                                                    aperture_settings,
                                                    shift_tolerance,
                                                    max_adu, fwhm_estimate,
                                                    use_coordinates=coords2use,
                                                    include_dig_noise=True,
                                                    reject_too_close=False,
                                                    reject_background_outliers=reject)

    phot.sort('aperture_sum')
    sources.sort('amplitude')

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


def list_of_fakes(num_files):
    # Generate fake CCDData objects for use in photometry_on_directory tests
    fake_images = [FakeCCDImage()]

    # Create additional images, each in a different position.
    for i in range(num_files-1):
        angle = 2*np.pi/(num_files-1) * i
        rad = 50
        dx, dy = rad*np.cos(angle), rad*np.sin(angle)
        fake_images.append(shift_FakeCCDImage(fake_images[0], dx, dy) )

    filters = ['U', 'B', 'V', 'R', 'I']
    for i in range(num_files):
        if (i < 5):
            fake_images[i].header['FILTER'] = filters[i]
        else:
            fake_images[i].header['FILTER'] = 'V'

    return fake_images


def test_photometry_on_directory():
    # Create list of fake CCDData objects
    num_files = 5
    fake_images = list_of_fakes(num_files)

    # Write fake images to temporary directory and test
    # multi_image_photometry on them.
    # NOTE: ignore_cleanup_errors=True is needed to avoid an error
    #       when the temporary directory is deleted on Windows.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        # Come up with Filenames
        temp_file_names = [Path(temp_dir) /
                            f"tempfile_{i:02d}.fit" for i in range(1, num_files + 1)]
        # Write the CCDData objects to files
        for i, image in enumerate(fake_images):
            image.write(temp_file_names[i])

        object_name = fake_images[0].header['OBJECT']
        sources = fake_images[0].sources
        aperture = sources['aperture'][0]
        inner_annulus = 2 * aperture
        outer_annulus = 3 * aperture
        aperture_settings = ApertureSettings(radius=aperture,
                                             gap=inner_annulus - aperture,
                                             width_annulus=outer_annulus - inner_annulus)

        # Generate the sourcelist
        found_sources = source_detection(fake_images[0],
                                        fwhm=fake_images[0].sources['x_stddev'].mean(),
                                        threshold=10)

        phot_data = multi_image_photometry(temp_dir,
                                object_name,
                                found_sources,
                                fake_camera,
                                fake_obs,
                                aperture_settings,
                                shift_tolerance, max_adu, fwhm_estimate,
                                include_dig_noise=True,
                                reject_too_close=True,
                                reject_background_outliers=True,
                                passband_map=None,
                                fwhm_by_fit=True)

    # For following assertion to be true, rad must be small enough that
    # no source lies within outer_annulus of the edge of an image.
    assert len(phot_data) == num_files*len(found_sources)

    # Sort all data by amount of signal
    sources.sort('amplitude')
    found_sources.sort('flux')

    # Get noise level from the first image
    noise_dev = fake_images[0].noise_dev

    for fnd, inp in zip(found_sources, sources):
        star_id_chk = fnd['star_id']
        # Select the rows in phot_data that correspond to the current star
        # and compute the average of the aperture sums.
        selected_rows = phot_data[phot_data['star_id'] == star_id_chk]
        obs_avg_net_cnts = np.average(selected_rows['aperture_net_cnts'].value)

        stdev = inp['x_stddev']
        expected_flux = (inp['amplitude'] * 2 * np.pi *
                            stdev**2 *
                            (1 - np.exp(-aperture**2 / (2 * stdev**2))))
        # This expected flux is correct IF there were no noise. With noise, the
        # standard deviation in the sum of the noise within in the aperture is
        # n_pix_in_aperture times the single-pixel standard deviation.
        #

        expected_deviation = np.pi * aperture**2 * noise_dev

        # We could require that the result be within some reasonable
        # number of those expected variations or we could count up the
        # actual number of background counts at each of the source
        # positions.

        # Here we just check whether any difference is consistent with
        # less than the expected one sigma deviation.
        assert (np.abs(expected_flux - obs_avg_net_cnts) <
                np.pi * aperture**2 * noise_dev)


def test_photometry_on_directory_with_no_ra_dec():
    # Create list of fake CCDData objects
    num_files = 5
    fake_images = list_of_fakes(num_files)

    # Write fake images to temporary directory and test
    # multi_image_photometry on them.
    # NOTE: ignore_cleanup_errors=True is needed to avoid an error
    #       when the temporary directory is deleted on Windows.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        # Come up with Filenames
        temp_file_names = [Path(temp_dir) /
                            f"tempfile_{i:02d}.fits" for i in range(1, num_files + 1)]
        # Write the CCDData objects to files
        for i, image in enumerate(fake_images):
            image.write(temp_file_names[i])

        object_name = fake_images[0].header['OBJECT']
        sources = fake_images[0].sources
        aperture = sources['aperture'][0]
        inner_annulus = 2 * aperture
        outer_annulus = 3 * aperture
        aperture_settings = ApertureSettings(radius=aperture,
                                             gap=inner_annulus - aperture,
                                             width_annulus=outer_annulus - inner_annulus)

        # Generate the sourcelist
        found_sources = source_detection(fake_images[0],
                                        fwhm=fake_images[0].sources['x_stddev'].mean(),
                                        threshold=10)

        # Damage the sourcelist by removing the ra and dec columns
        found_sources.drop_ra_dec()

        with pytest.raises(ValueError):
            phot_data = multi_image_photometry(temp_dir,
                                    object_name,
                                    found_sources,
                                    fake_camera,
                                    fake_obs,
                                    aperture_settings,
                                    shift_tolerance, max_adu, fwhm_estimate,
                                    include_dig_noise=True,
                                    reject_too_close=True,
                                    reject_background_outliers=True,
                                    passband_map=None,
                                    fwhm_by_fit=True)


def test_photometry_on_directory_with_bad_fits():
    # Create list of fake CCDData objects
    num_files = 5
    clean_fake_images = list_of_fakes(num_files)
    fake_images = list_of_fakes(num_files)

    # Write fake images (without WCS) to temporary directory and test
    # multi_image_photometry on them.
    # NOTE: ignore_cleanup_errors=True is needed to avoid an error
    #       when the temporary directory is deleted on Windows.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        # Come up with Filenames
        temp_file_names = [Path(temp_dir) /
                            f"tempfile_{i:02d}.fits" for i in range(1, num_files + 1)]
        # Write the CCDData objects to files
        for i, image in enumerate(fake_images):
            image.drop_wcs()
            image.write(temp_file_names[i])

        object_name = fake_images[0].header['OBJECT']
        sources = fake_images[0].sources
        aperture = sources['aperture'][0]
        inner_annulus = 2 * aperture
        outer_annulus = 3 * aperture

        aperture_settings = ApertureSettings(radius=aperture,
                                             gap=inner_annulus - aperture,
                                             width_annulus=outer_annulus - inner_annulus)

        # Generate the sourcelist with RA/Dec information from a clean image
        found_sources = source_detection(clean_fake_images[0],
                                        fwhm=clean_fake_images[0].sources['x_stddev'].mean(),
                                        threshold=10)

        # Since none of the images will be valid, it should raise a RuntimeError
        with pytest.raises(RuntimeError):
            phot_data = multi_image_photometry(temp_dir,
                                    object_name,
                                    found_sources,
                                    fake_camera,
                                    fake_obs,
                                    aperture_settings,
                                    shift_tolerance, max_adu, fwhm_estimate,
                                    include_dig_noise=True,
                                    reject_too_close=True,
                                    reject_background_outliers=True,
                                    passband_map=None,
                                    fwhm_by_fit=True)
