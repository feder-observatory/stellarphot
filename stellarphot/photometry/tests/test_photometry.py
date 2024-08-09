import logging
import tempfile
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.io import ascii
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.metadata.exceptions import MergeConflictWarning

from stellarphot.core import SourceListData
from stellarphot.photometry import (
    AperturePhotometry,
    calculate_noise,
    find_too_close,
    multi_image_photometry,
    single_image_photometry,
    source_detection,
)
from stellarphot.photometry.tests.fake_image import FakeCCDImage, shift_FakeCCDImage
from stellarphot.settings import (
    Camera,
    LoggingSettings,
    Observatory,
    PassbandMap,
    PhotometryApertures,
    PhotometryOptionalSettings,
    PhotometrySettings,
    SourceLocationSettings,
)

# Constants for the tests

GAINS = [1.0, 1.5, 2.0]
# Make sure the tests are deterministic by using a seed
SEED = 5432985


# The default coordinate system and shift tolerance to use for the tests
COORDS2USE = "pixel"
SHIFT_TOLERANCE = 6
DEFAULT_SOURCE_LOCATIONS = SourceLocationSettings(
    source_list_file="",
    shift_tolerance=SHIFT_TOLERANCE,
    use_coordinates=COORDS2USE,
)

PHOTOMETRY_OPTIONS = PhotometryOptionalSettings()
# This used to be the default; it has switched to exact
PHOTOMETRY_OPTIONS.method = "center"

# A camera with not unreasonable settings
FAKE_CAMERA = Camera(
    data_unit=u.adu,
    gain=1.0 * u.electron / u.adu,
    name="test camera",
    read_noise=0 * u.electron,
    dark_current=0.1 * u.electron / u.second,
    pixel_scale=1 * u.arcsec / u.pixel,
    max_data_value=40000 * u.adu,
)

# Camera with no read noise or dark current
ZERO_CAMERA = Camera(
    data_unit=u.adu,
    gain=1.0 * u.electron / u.adu,
    name="test camera",
    read_noise=0 * u.electron,
    dark_current=0.0 * u.electron / u.second,
    pixel_scale=1 * u.arcsec / u.pixel,
    max_data_value=40000 * u.adu,
)

# Fake observatory location
FAKE_OBS = Observatory(
    name="test observatory", latitude=0 * u.deg, longitude=0 * u.deg, elevation=0 * u.m
)

# The fake image used for testing
FAKE_CCD_IMAGE = FakeCCDImage(seed=SEED)

# Build default PhotometryOptions for the tests based on the fake image
DEFAULT_PHOTOMETRY_APERTURES = PhotometryApertures(
    radius=FAKE_CCD_IMAGE.sources["aperture"][0],
    gap=FAKE_CCD_IMAGE.sources["aperture"][0],
    annulus_width=FAKE_CCD_IMAGE.sources["aperture"][0],
    fwhm=FAKE_CCD_IMAGE.sources["x_stddev"].mean(),
)

# Passband map for the tests
PASSBAND_MAP = PassbandMap(
    name="Example Passband Map",
    your_filter_names_to_aavso={
        "B": "B",
        "rp": "SR",
    },
)

DEFAULT_LOGGING_SETTINGS = LoggingSettings()


@pytest.fixture
def photometry_settings():
    return PhotometrySettings(
        camera=FAKE_CAMERA,
        observatory=FAKE_OBS,
        photometry_apertures=DEFAULT_PHOTOMETRY_APERTURES,
        source_location_settings=DEFAULT_SOURCE_LOCATIONS,
        photometry_optional_settings=PHOTOMETRY_OPTIONS,
        passband_map=PASSBAND_MAP,
        logging_settings=DEFAULT_LOGGING_SETTINGS,
    )


class TestAperturePhotometry:
    @staticmethod
    def create_source_list():
        # This has X, Y
        sources = FAKE_CCD_IMAGE.sources.copy()

        # Rename to match the expected names
        sources.rename_column("x_mean", "xcenter")
        sources.rename_column("y_mean", "ycenter")

        # Calculate RA/Dec from image WCS
        coords = FAKE_CCD_IMAGE.wcs.pixel_to_world(
            sources["xcenter"], sources["ycenter"]
        )
        sources["ra"] = coords.ra
        sources["dec"] = coords.dec
        sources["star_id"] = list(range(len(sources)))
        sources["xcenter"] = sources["xcenter"] * u.pixel
        sources["ycenter"] = sources["ycenter"] * u.pixel

        return SourceListData(input_data=sources, colname_map=None)

    @staticmethod
    def list_of_fakes(num_files):
        # Generate fake CCDData objects for use in photometry_on_directory tests
        fake_images = [deepcopy(FAKE_CCD_IMAGE)]

        # Create additional images, each in a different position.
        for i in range(num_files - 1):
            angle = 2 * np.pi / (num_files - 1) * i
            rad = 50
            dx, dy = rad * np.cos(angle), rad * np.sin(angle)
            fake_images.append(shift_FakeCCDImage(fake_images[0], dx, dy))

        filters = ["U", "B", "V", "R", "I"]
        for i in range(num_files):
            if i < 5:
                fake_images[i].header["FILTER"] = filters[i]
            else:
                fake_images[i].header["FILTER"] = "V"

        return fake_images

    def test_create_aperture_photometry(self, tmp_path, photometry_settings):
        source_list = self.create_source_list()
        source_list_file = tmp_path / "source_list.ecsv"
        source_list.write(source_list_file, overwrite=True)

        # We are fine to modify photometry_settings here because pytest will
        # make a new one for each test.
        photometry_settings.source_location_settings.source_list_file = str(
            source_list_file
        )

        # Create an AperturePhotometry object
        ap_phot = AperturePhotometry(settings=photometry_settings)

        # Check that the object was created correctly
        assert ap_phot.settings.camera is FAKE_CAMERA
        assert ap_phot.settings.observatory is FAKE_OBS

    # The True case below is a regression test for #157
    @pytest.mark.parametrize("int_data", [True, False])
    def test_aperture_photometry_no_outlier_rejection(self, int_data, tmp_path):
        fake_CCDimage = deepcopy(FAKE_CCD_IMAGE)

        found_sources = source_detection(
            fake_CCDimage, fwhm=fake_CCDimage.sources["x_stddev"].mean(), threshold=10
        )

        # The scale_factor is used to rescale data to integers if needed. It
        # needs to be set later on when the net counts are "unscaled" in the
        # asserts that constitute the actual test.
        scale_factor = 1.0
        if int_data:
            scale_factor = (
                0.75 * FAKE_CAMERA.max_data_value.value / fake_CCDimage.data.max()
            )
            # For the moment, ensure the integer data is NOT larger than max_adu
            # because until #161 is fixed then having NaN in the data will not succeed.
            data = scale_factor * fake_CCDimage.data
            fake_CCDimage.data = data.astype(int)

        # Make a copy of photometry options
        phot_options = PhotometryOptionalSettings(**PHOTOMETRY_OPTIONS.model_dump())

        # Modify options to match test before we used phot_options
        phot_options.reject_background_outliers = False
        phot_options.reject_too_close = False
        phot_options.include_dig_noise = True

        source_list_file = tmp_path / "source_list.ecsv"
        found_sources.write(source_list_file, format="ascii.ecsv", overwrite=True)

        source_locations = DEFAULT_SOURCE_LOCATIONS.model_copy()
        source_locations.source_list_file = str(source_list_file)

        image_file = tmp_path / "fake_image.fits"
        fake_CCDimage.write(image_file, overwrite=True)

        photometry_settings = PhotometrySettings(
            camera=FAKE_CAMERA,
            observatory=FAKE_OBS,
            photometry_apertures=DEFAULT_PHOTOMETRY_APERTURES,
            source_location_settings=source_locations,
            photometry_optional_settings=phot_options,
            passband_map=PASSBAND_MAP,
            logging_settings=DEFAULT_LOGGING_SETTINGS,
        )

        ap_phot = AperturePhotometry(settings=photometry_settings)
        phot, missing_sources = ap_phot(image_file)

        phot.sort("aperture_sum")
        sources = fake_CCDimage.sources
        # Astropy tables sort in-place so we need to sort the sources table
        # after the fact.
        sources.sort("amplitude")
        aperture = DEFAULT_PHOTOMETRY_APERTURES.radius

        for inp, out in zip(sources, phot, strict=True):
            stdev = inp["x_stddev"]
            expected_flux = (
                inp["amplitude"]
                * 2
                * np.pi
                * stdev**2
                * (1 - np.exp(-(aperture**2) / (2 * stdev**2)))
            )
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
            assert (
                np.abs(expected_flux - out["aperture_net_cnts"].value / scale_factor)
                < np.pi * aperture**2 * fake_CCDimage.noise_dev
            )

    @pytest.mark.parametrize("reject", [True, False])
    def test_aperture_photometry_with_outlier_rejection(
        self, reject, tmp_path, photometry_settings
    ):
        """
        Insert some really large pixel values in the annulus and check that
        the photometry is correct when outliers are rejected and is
        incorrect when outliers are not rejected.
        """
        fake_CCDimage = deepcopy(FAKE_CCD_IMAGE)
        sources = fake_CCDimage.sources

        aperture_settings = DEFAULT_PHOTOMETRY_APERTURES
        aperture = aperture_settings.radius
        inner_annulus = aperture_settings.inner_annulus
        outer_annulus = aperture_settings.outer_annulus

        image = fake_CCDimage.data

        found_sources = source_detection(
            fake_CCDimage, fwhm=sources["x_stddev"].mean(), threshold=10
        )
        source_list_file = tmp_path / "source_list.ecsv"
        found_sources.write(source_list_file, format="ascii.ecsv", overwrite=True)

        # Add some large pixel values to the annulus for each source.
        # adding these moves the average pixel value by quite a bit,
        # so we'll only get the correct net flux if these are removed.
        for source in fake_CCDimage.sources:
            center_px = (int(source["x_mean"]), int(source["y_mean"]))
            begin = center_px[0] + inner_annulus + 1
            end = begin + (outer_annulus - inner_annulus - 1)
            # Yes, x and y are deliberately reversed below.
            image[center_px[1], begin:end] = 100 * fake_CCDimage.mean_noise

        # Make a copy of photometry options
        phot_options = PhotometryOptionalSettings(**PHOTOMETRY_OPTIONS.model_dump())

        # Modify options to match test before we used phot_options
        phot_options.reject_background_outliers = reject
        phot_options.reject_too_close = False
        phot_options.include_dig_noise = True

        # It is fine to modify photometry_settings here because pytest will
        # make a new one for each test.
        photometry_settings.source_location_settings.source_list_file = str(
            source_list_file
        )
        photometry_settings.photometry_optional_settings = phot_options

        image_file = tmp_path / "fake_image.fits"
        fake_CCDimage.write(image_file, overwrite=True)

        ap_phot = AperturePhotometry(settings=photometry_settings)
        phot, missing_sources = ap_phot(image_file)

        phot.sort("aperture_sum")
        sources.sort("amplitude")

        for inp, out in zip(sources, phot, strict=True):
            stdev = inp["x_stddev"]
            expected_flux = (
                inp["amplitude"]
                * 2
                * np.pi
                * stdev**2
                * (1 - np.exp(-(aperture**2) / (2 * stdev**2)))
            )
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
                assert (
                    np.abs(expected_flux - out["aperture_net_cnts"].value)
                    < expected_deviation
                )
            else:
                with pytest.raises(AssertionError):
                    assert (
                        np.abs(expected_flux - out["aperture_net_cnts"].value)
                        < expected_deviation
                    )

    def test_photometry_method_argument(self, tmp_path, photometry_settings):
        """
        Make sure that setting the method option has an effect on the photometry.
        """
        fake_CCDimage = deepcopy(FAKE_CCD_IMAGE)

        found_sources = source_detection(
            fake_CCDimage, fwhm=fake_CCDimage.sources["x_stddev"].mean(), threshold=10
        )

        source_list_file = tmp_path / "source_list.ecsv"
        found_sources.write(source_list_file, format="ascii.ecsv", overwrite=True)

        image_file = tmp_path / "fake_image.fits"
        fake_CCDimage.write(image_file, overwrite=True)

        # Make a copy of photometry options
        phot_options = PhotometryOptionalSettings(**PHOTOMETRY_OPTIONS.model_dump())

        # Do the photometry with method = "center" first
        phot_options.method = "center"

        photometry_settings.source_location_settings.source_list_file = str(
            source_list_file
        )
        photometry_settings.photometry_optional_settings = phot_options

        ap_phot = AperturePhotometry(settings=photometry_settings)
        phot_center, _ = ap_phot(image_file)

        # Now redo photometry with method="exact". Whether the flux is larger or smaller
        # in this case depends on the exact position of the sources in the image.
        # Here we just check that the flux is different.
        phot_options.method = "exact"
        photometry_settings.photometry_optional_settings = phot_options
        ap_phot_exact = AperturePhotometry(settings=photometry_settings)
        phot_exact, _ = ap_phot_exact(image_file)
        assert np.all(phot_exact["aperture_sum"] != phot_center["aperture_sum"])

    @pytest.mark.parametrize("coords", ["sky", "pixel"])
    def test_photometry_on_directory(self, coords, photometry_settings):
        # Create list of fake CCDData objects
        num_files = 5
        fake_images = list_of_fakes(num_files)

        # Write fake images to temporary directory and test
        # multi_image_photometry on them.
        # NOTE: ignore_cleanup_errors=True is needed to avoid an error
        #       when the temporary directory is deleted on Windows.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            # Come up with Filenames
            temp_file_names = [
                Path(temp_dir) / f"tempfile_{i:02d}.fit"
                for i in range(1, num_files + 1)
            ]
            # Write the CCDData objects to files
            for i, image in enumerate(fake_images):
                image.write(temp_file_names[i])

            object_name = fake_images[0].header["OBJECT"]
            sources = fake_images[0].sources
            aperture_settings = DEFAULT_PHOTOMETRY_APERTURES
            aperture = aperture_settings.radius

            # Generate the sourcelist
            found_sources = source_detection(
                fake_images[0],
                fwhm=fake_images[0].sources["x_stddev"].mean(),
                threshold=10,
            )

            source_list_file = Path(temp_dir) / "source_list.ecsv"
            found_sources.write(source_list_file, format="ascii.ecsv", overwrite=True)

            # Make a copy of photometry options
            phot_options = PhotometryOptionalSettings(**PHOTOMETRY_OPTIONS.model_dump())

            # Modify options to match test before we used phot_options
            phot_options.include_dig_noise = True
            phot_options.reject_too_close = True
            phot_options.reject_background_outliers = True
            phot_options.fwhm_by_fit = True

            photometry_settings.photometry_optional_settings = phot_options
            photometry_settings.source_location_settings.use_coordinates = coords
            photometry_settings.source_location_settings.source_list_file = str(
                source_list_file
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Cannot merge meta key",
                    category=MergeConflictWarning,
                )
                ap_phot = AperturePhotometry(settings=photometry_settings)
                phot_data = ap_phot(
                    temp_dir,
                    object_of_interest=object_name,
                )

        # For following assertion to be true, rad must be small enough that
        # no source lies within outer_annulus of the edge of an image.
        assert len(phot_data) == num_files * len(found_sources)

        # Sort all data by amount of signal
        sources.sort("amplitude")
        found_sources.sort("flux")

        # Get noise level from the first image
        noise_dev = fake_images[0].noise_dev

        for fnd, inp in zip(found_sources, sources, strict=True):
            star_id_chk = fnd["star_id"]
            # Select the rows in phot_data that correspond to the current star
            # and compute the average of the aperture sums.
            selected_rows = phot_data[phot_data["star_id"] == star_id_chk]
            obs_avg_net_cnts = np.average(selected_rows["aperture_net_cnts"].value)

            stdev = inp["x_stddev"]
            expected_flux = (
                inp["amplitude"]
                * 2
                * np.pi
                * stdev**2
                * (1 - np.exp(-(aperture**2) / (2 * stdev**2)))
            )
            # This expected flux is correct IF there were no noise. With noise, the
            # standard deviation in the sum of the noise within in the aperture is
            # n_pix_in_aperture times the single-pixel standard deviation.
            #

            expected_deviation = np.pi * aperture**2 * noise_dev

            # We have two cases to consider: use_coordinates="sky" and
            # use_coordinates="pixel".
            if coords == "sky":
                # In this case, the expected result is the test below.

                # We could require that the result be within some reasonable
                # number of those expected variations or we could count up the
                # actual number of background counts at each of the source
                # positions.

                # Here we just check whether any difference is consistent with
                # less than the expected one sigma deviation.
                assert np.abs(expected_flux - obs_avg_net_cnts) < expected_deviation
            else:
                # In this case we are trying to do photometry in pixel coordinates,
                # using the pixel location of the sources as found in the first image --
                # see the line where found_sources is defined.
                #
                # However, the images are shifted with respect to each other by
                # list_of_fakes, so there are no longer stars at those positions in the
                # other images.
                #
                # Because of that, the expected result is that either obs_avg_net_cnts
                # is nan or the difference is bigger than the expected_deviation.
                assert (
                    np.isnan(obs_avg_net_cnts)
                    or np.abs(expected_flux - obs_avg_net_cnts) > expected_deviation
                )

    def test_photometry_on_directory_with_no_ra_dec(self, photometry_settings):
        # Create list of fake CCDData objects
        num_files = 5
        fake_images = list_of_fakes(num_files)

        # Write fake images to temporary directory and test
        # multi_image_photometry on them.
        # NOTE: ignore_cleanup_errors=True is needed to avoid an error
        #       when the temporary directory is deleted on Windows.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            # Come up with Filenames
            temp_file_names = [
                Path(temp_dir) / f"tempfile_{i:02d}.fits"
                for i in range(1, num_files + 1)
            ]
            # Write the CCDData objects to files
            for i, image in enumerate(fake_images):
                image.write(temp_file_names[i])

            object_name = fake_images[0].header["OBJECT"]

            # Generate the sourcelist
            found_sources = source_detection(
                fake_images[0],
                fwhm=fake_images[0].sources["x_stddev"].mean(),
                threshold=10,
            )

            # Damage the sourcelist by removing the ra and dec columns
            found_sources.drop_ra_dec()

            source_list_file = Path(temp_dir) / "source_list.ecsv"
            found_sources.write(source_list_file, format="ascii.ecsv", overwrite=True)

            phot_options = PhotometryOptionalSettings(**PHOTOMETRY_OPTIONS.model_dump())

            # Modify options to match test before we used phot_options
            phot_options.include_dig_noise = True
            phot_options.reject_too_close = True
            phot_options.reject_background_outliers = True
            phot_options.fwhm_by_fit = True

            photometry_settings.photometry_optional_settings = phot_options
            photometry_settings.source_location_settings.source_list_file = str(
                source_list_file
            )
            # The setting below was implicit in the old default
            photometry_settings.source_location_settings.use_coordinates = "sky"

            ap_phot = AperturePhotometry(settings=photometry_settings)
            with pytest.raises(ValueError):
                ap_phot(
                    temp_dir,
                    object_of_interest=object_name,
                )

    def test_photometry_on_directory_with_bad_fits(self, photometry_settings):
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
            temp_file_names = [
                Path(temp_dir) / f"tempfile_{i:02d}.fits"
                for i in range(1, num_files + 1)
            ]
            # Write the CCDData objects to files
            for i, image in enumerate(fake_images):
                image.drop_wcs()
                image.write(temp_file_names[i])

            object_name = fake_images[0].header["OBJECT"]

            # Generate the sourcelist with RA/Dec information from a clean image
            found_sources = source_detection(
                clean_fake_images[0],
                fwhm=clean_fake_images[0].sources["x_stddev"].mean(),
                threshold=10,
            )

            source_list_file = Path(temp_dir) / "source_list.ecsv"
            found_sources.write(source_list_file, format="ascii.ecsv", overwrite=True)

            phot_options = PhotometryOptionalSettings(**PHOTOMETRY_OPTIONS.model_dump())

            # Modify options to match test before we used phot_options
            phot_options.include_dig_noise = True
            phot_options.reject_too_close = True
            phot_options.reject_background_outliers = True
            phot_options.fwhm_by_fit = True

            photometry_settings.photometry_optional_settings = phot_options
            photometry_settings.source_location_settings.source_list_file = str(
                source_list_file
            )
            # The settings below was implicit in the old default
            photometry_settings.source_location_settings.use_coordinates = "sky"

            ap_phot = AperturePhotometry(settings=photometry_settings)
            # Since none of the images will be valid, it should raise a RuntimeError
            with pytest.raises(RuntimeError):
                ap_phot(
                    temp_dir,
                    object_of_interest=object_name,
                )

    def test_invalid_path(self, photometry_settings):
        ap = AperturePhotometry(settings=photometry_settings)
        with pytest.raises(ValueError, match="is not a valid file or directory"):
            ap("invalid_path")

    # Checking logging for AperturePhotometry for single image photometry.
    @pytest.mark.parametrize("logfile", ["test.log", None])
    @pytest.mark.parametrize("console_log", [True, False])
    def test_logging_single_image(self, capsys, logfile, console_log, tmp_path):
        # Disable any root logger handlers that are active before using
        # logging since that is expectation of single_image_photometry.
        if logging.root.hasHandlers():
            logging.root.handlers.clear()

        # Create fake image
        fake_CCDimage = deepcopy(FAKE_CCD_IMAGE)
        image_file = tmp_path / "fake_image.fits"
        fake_CCDimage.write(image_file, overwrite=True)
        # Create source list from fake image
        found_sources = source_detection(
            fake_CCDimage, fwhm=fake_CCDimage.sources["x_stddev"].mean(), threshold=10
        )
        source_list_file = tmp_path / "source_list.ecsv"
        found_sources.write(source_list_file, format="ascii.ecsv", overwrite=True)

        # Make a copy of photometry options and modify them to match the
        # test_aperture_photometry_no_outlier_rejection settings
        phot_options = PhotometryOptionalSettings(**PHOTOMETRY_OPTIONS.model_dump())
        phot_options.reject_background_outliers = False
        phot_options.reject_too_close = False
        phot_options.include_dig_noise = True

        # Define the source locations settings
        source_locations = DEFAULT_SOURCE_LOCATIONS.model_copy()
        source_locations.source_list_file = str(source_list_file)

        # Define the logging settings
        logging_settings = DEFAULT_LOGGING_SETTINGS.model_copy()
        if logfile:
            # Define the log file and console log settings
            # and make sure to set full path of log file.
            logging_settings.logfile = str(tmp_path / logfile)
            logging_settings.console_log = console_log
            full_logfile = logging_settings.logfile

        photometry_settings = PhotometrySettings(
            camera=FAKE_CAMERA,
            observatory=FAKE_OBS,
            photometry_apertures=DEFAULT_PHOTOMETRY_APERTURES,
            source_location_settings=source_locations,
            photometry_optional_settings=phot_options,
            passband_map=PASSBAND_MAP,
            logging_settings=logging_settings,
        )

        # Call the AperturePhotometry class with a single image
        ap_phot = AperturePhotometry(settings=photometry_settings)
        phot, missing_sources = ap_phot(image_file)

        #
        # Test logging was consistent with settings
        #
        # Check and see if the output log file was created and contains the
        # expected messages.
        if logfile:
            assert Path(full_logfile).exists()
            with open(full_logfile) as f:
                log_content = f.read()
                # Confirm last log message written by single_image_photometry
                # present.
                assert "Calculating noise for all sources" in log_content

        # If console logging is enabled then the stderr should contain the
        # expected messages.
        if console_log:
            captured_stdout = capsys.readouterr()
            # Confirm last log message written by single_image_photometry
            # present.
            assert "Calculating noise for all sources" in captured_stdout.err

    # Checking logging for AperturePhotometry for multiple image photometry.
    @pytest.mark.parametrize("logfile", ["test.log", None])
    @pytest.mark.parametrize("console_log", [True, False])
    def test_logging_multiple_image(
        self, capsys, logfile, console_log, photometry_settings
    ):
        # Create list of fake CCDData objects
        num_files = 5
        fake_images = list_of_fakes(num_files)

        # Write fake images to temporary directory and test
        # multi_image_photometry on them.0
        # NOTE: ignore_cleanup_errors=True is needed to avoid an error
        #       when the temporary directory is deleted on Windows.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            # Come up with Filenames
            temp_file_names = [
                Path(temp_dir) / f"tempfile_{i:02d}.fit"
                for i in range(1, num_files + 1)
            ]
            # Write the CCDData objects to files
            for i, image in enumerate(fake_images):
                image.write(temp_file_names[i])
            object_name = fake_images[0].header["OBJECT"]

            # Generate the sourcelist
            found_sources = source_detection(
                fake_images[0],
                fwhm=fake_images[0].sources["x_stddev"].mean(),
                threshold=10,
            )
            source_list_file = Path(temp_dir) / "source_list.ecsv"
            found_sources.write(source_list_file, format="ascii.ecsv", overwrite=True)

            # Make a copy of photometry options based on those used in
            # successful test_photometry_on_directory
            phot_options = PhotometryOptionalSettings(**PHOTOMETRY_OPTIONS.model_dump())
            phot_options.include_dig_noise = True
            phot_options.reject_too_close = True
            phot_options.reject_background_outliers = True
            phot_options.fwhm_by_fit = True

            photometry_settings.photometry_optional_settings = phot_options
            photometry_settings.source_location_settings.use_coordinates = "sky"
            photometry_settings.source_location_settings.source_list_file = str(
                source_list_file
            )

            # Define the logging settings
            logging_settings = DEFAULT_LOGGING_SETTINGS.model_copy()
            if logfile:
                logging_settings.logfile = logfile
                logging_settings.console_log = console_log
                # log file should be written to image directory (tmp_dir)
                # automatically, this ensures we know the path to the log file.
                full_logfile = str(Path(temp_dir) / logfile)
                photometry_settings.logging_settings = logging_settings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Cannot merge meta key",
                    category=MergeConflictWarning,
                )

                ap_phot = AperturePhotometry(settings=photometry_settings)
                _ = ap_phot(temp_dir, object_of_interest=object_name)

                #
                # Test logging was consistent with settings
                #
                # Check and see if the output log file was created and contains the
                # expected messages.
                if logfile:
                    assert Path(full_logfile).exists()
                    with open(full_logfile) as f:
                        log_content = f.read()
                        # Check for log messages output by multi_image_photometry
                        assert "Starting photometry of files in" in log_content
                        assert "DONE processing all matching images" in log_content
                        # Check for last log message from single_image_photometry
                        assert "Calculating noise for all sources" in log_content

                # If console logging is enabled then the stderr should contain the
                # expected messages.
                if console_log:
                    captured_stdout = capsys.readouterr()
                    # Check for log messages output by multi_image_photometry
                    assert "Starting photometry of files in" in captured_stdout.err
                    assert "DONE processing all matching images" in captured_stdout.err
                    # Check for last log message from single_image_photometry
                    assert "Calculating noise for all sources" in captured_stdout.err


def test_calc_noise_defaults():
    # If we put in nothing we should get an error about is missing camera
    # instance.
    with pytest.raises(ValueError):
        assert calculate_noise() == 0


@pytest.mark.parametrize("aperture_area", [5, 20])
@pytest.mark.parametrize("gain", GAINS)
def test_calc_noise_source_only(gain, aperture_area):
    # If the only source of noise is Poisson error in the source
    # then the noise should be the square root of the counts.
    counts = 100
    expected = np.sqrt(gain * counts)

    # Create camera instance
    camera = ZERO_CAMERA.model_copy()
    camera.gain = gain * camera.gain.unit

    np.testing.assert_allclose(
        calculate_noise(camera, counts=counts, aperture_area=aperture_area), expected
    )


@pytest.mark.parametrize("aperture_area", [5, 20])
@pytest.mark.parametrize("gain", GAINS)
def test_calc_noise_dark_only(gain, aperture_area):
    # Gain should not affect this one. Dark current needs a couple other things,
    # but this is basically Poisson error.
    dark_current = 10
    exposure = 20

    # Create camera instance
    camera = ZERO_CAMERA.model_copy()
    # Set gain and dark current to values for test
    camera.dark_current = dark_current * camera.dark_current.unit
    camera.gain = gain * camera.gain.unit

    expected = np.sqrt(dark_current * aperture_area * exposure)

    np.testing.assert_allclose(
        calculate_noise(camera, aperture_area=aperture_area, exposure=exposure),
        expected,
    )


@pytest.mark.parametrize("aperture_area", [5, 20])
@pytest.mark.parametrize("gain", GAINS)
def test_calc_read_noise_only(gain, aperture_area):
    # The read noise per pixel IS the noise. The only multiplier is
    # the number of pixels.
    read_noise = 10
    expected = np.sqrt(aperture_area * read_noise**2)

    # Create camera instance
    camera = ZERO_CAMERA.model_copy()
    camera.read_noise = read_noise * camera.read_noise.unit
    camera.gain = gain * camera.gain.unit

    np.testing.assert_allclose(
        calculate_noise(camera, aperture_area=aperture_area), expected
    )


@pytest.mark.parametrize("aperture_area", [5, 20])
@pytest.mark.parametrize("gain", GAINS)
def test_calc_sky_only(gain, aperture_area):
    # The sky noise per pixel is the poisson and per pixel.
    sky = 10
    expected = np.sqrt(gain * aperture_area * sky)

    # Create camera instance
    camera = ZERO_CAMERA.model_copy()
    camera.gain = gain * camera.gain.unit

    np.testing.assert_allclose(
        calculate_noise(camera, aperture_area=aperture_area, sky_per_pix=sky), expected
    )


def test_annulus_area_term():
    # Test that noise is correct with an annulus
    aperture_area = 20

    # Annulus is typically quite a bit larger than aperture.
    annulus_area = 10 * aperture_area
    gain = 1.5
    sky = 10
    expected = np.sqrt(gain * aperture_area * (1 + aperture_area / annulus_area) * sky)

    # Create camera instance
    camera = ZERO_CAMERA.model_copy()
    camera.gain = gain * camera.gain.unit

    np.testing.assert_allclose(
        calculate_noise(
            camera,
            aperture_area=aperture_area,
            annulus_area=annulus_area,
            sky_per_pix=sky,
        ),
        expected,
    )


@pytest.mark.parametrize("digit,expected", ((False, 89.078616), (True, 89.10182)))
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
    camera = ZERO_CAMERA.model_copy()
    camera.gain = gain * camera.gain.unit
    camera.dark_current = dark_current * camera.dark_current.unit
    camera.read_noise = read_noise * camera.read_noise.unit

    np.testing.assert_allclose(
        calculate_noise(
            camera,
            counts=counts,
            sky_per_pix=sky,
            exposure=exposure,
            aperture_area=aperture_area,
            annulus_area=annulus_area,
            include_digitization=digit,
        ),
        expected,
    )


def test_find_too_close():
    # Load test sourcelist into memory
    test_sl_data = ascii.read(
        get_pkg_data_filename("data/test_corner.ecsv"), format="ecsv", fast_reader=False
    )

    # Create no sky position sourcelist
    test_sl_data_nosky = test_sl_data.copy()
    test_sl_data_nosky.remove_column("ra")
    test_sl_data_nosky.remove_column("dec")

    # Create no image position sourcelist
    test_sl_data_noimgpos = test_sl_data.copy()
    test_sl_data_noimgpos.remove_column("xcenter")
    test_sl_data_noimgpos.remove_column("ycenter")

    # Create SourceListData objects
    sl_test = SourceListData(input_data=test_sl_data, colname_map=None)
    sl_test_nosky = SourceListData(input_data=test_sl_data_nosky, colname_map=None)
    sl_test_noimgpos = SourceListData(
        input_data=test_sl_data_noimgpos, colname_map=None
    )

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


# The True case below is a regression test for #157
@pytest.mark.parametrize("int_data", [True, False])
def test_aperture_photometry_no_outlier_rejection(int_data, tmp_path):
    fake_CCDimage = deepcopy(FAKE_CCD_IMAGE)

    found_sources = source_detection(
        fake_CCDimage, fwhm=fake_CCDimage.sources["x_stddev"].mean(), threshold=10
    )

    # The scale_factor is used to rescale data to integers if needed. It
    # needs to be set later on when the net counts are "unscaled" in the
    # asserts that constitute the actual test.
    scale_factor = 1.0
    if int_data:
        scale_factor = (
            0.75 * FAKE_CAMERA.max_data_value.value / fake_CCDimage.data.max()
        )
        # For the moment, ensure the integer data is NOT larger than max_adu
        # because until #161 is fixed then having NaN in the data will not succeed.
        data = scale_factor * fake_CCDimage.data
        fake_CCDimage.data = data.astype(int)

    # Make a copy of photometry options
    phot_options = PhotometryOptionalSettings(**PHOTOMETRY_OPTIONS.model_dump())

    # Modify options to match test before we used phot_options
    phot_options.reject_background_outliers = False
    phot_options.reject_too_close = False
    phot_options.include_dig_noise = True

    source_list_file = tmp_path / "source_list.ecsv"
    found_sources.write(source_list_file, format="ascii.ecsv", overwrite=True)

    source_locations = DEFAULT_SOURCE_LOCATIONS.model_copy()
    source_locations.source_list_file = str(source_list_file)
    photometry_settings = PhotometrySettings(
        camera=FAKE_CAMERA,
        observatory=FAKE_OBS,
        photometry_apertures=DEFAULT_PHOTOMETRY_APERTURES,
        source_location_settings=source_locations,
        photometry_optional_settings=phot_options,
        passband_map=PASSBAND_MAP,
        logging_settings=DEFAULT_LOGGING_SETTINGS,
    )
    phot, missing_sources = single_image_photometry(
        fake_CCDimage,
        photometry_settings,
    )

    phot.sort("aperture_sum")
    sources = fake_CCDimage.sources
    # Astropy tables sort in-place so we need to sort the sources table
    # after the fact.
    sources.sort("amplitude")
    aperture = DEFAULT_PHOTOMETRY_APERTURES.radius

    for inp, out in zip(sources, phot, strict=True):
        stdev = inp["x_stddev"]
        expected_flux = (
            inp["amplitude"]
            * 2
            * np.pi
            * stdev**2
            * (1 - np.exp(-(aperture**2) / (2 * stdev**2)))
        )
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
        assert (
            np.abs(expected_flux - out["aperture_net_cnts"].value / scale_factor)
            < np.pi * aperture**2 * fake_CCDimage.noise_dev
        )


@pytest.mark.parametrize("reject", [True, False])
def test_aperture_photometry_with_outlier_rejection(
    reject, tmp_path, photometry_settings
):
    """
    Insert some really large pixel values in the annulus and check that
    the photometry is correct when outliers are rejected and is
    incorrect when outliers are not rejected.
    """
    fake_CCDimage = deepcopy(FAKE_CCD_IMAGE)
    sources = fake_CCDimage.sources

    aperture_settings = DEFAULT_PHOTOMETRY_APERTURES
    aperture = aperture_settings.radius
    inner_annulus = aperture_settings.inner_annulus
    outer_annulus = aperture_settings.outer_annulus

    image = fake_CCDimage.data

    found_sources = source_detection(
        fake_CCDimage, fwhm=sources["x_stddev"].mean(), threshold=10
    )
    source_list_file = tmp_path / "source_list.ecsv"
    found_sources.write(source_list_file, format="ascii.ecsv", overwrite=True)

    # Add some large pixel values to the annulus for each source.
    # adding these moves the average pixel value by quite a bit,
    # so we'll only get the correct net flux if these are removed.
    for source in fake_CCDimage.sources:
        center_px = (int(source["x_mean"]), int(source["y_mean"]))
        begin = center_px[0] + inner_annulus + 1
        end = begin + (outer_annulus - inner_annulus - 1)
        # Yes, x and y are deliberately reversed below.
        image[center_px[1], begin:end] = 100 * fake_CCDimage.mean_noise

    # Make a copy of photometry options
    phot_options = PhotometryOptionalSettings(**PHOTOMETRY_OPTIONS.model_dump())

    # Modify options to match test before we used phot_options
    phot_options.reject_background_outliers = reject
    phot_options.reject_too_close = False
    phot_options.include_dig_noise = True

    photometry_settings.source_location_settings.source_list_file = str(
        source_list_file
    )
    photometry_settings.photometry_optional_settings = phot_options

    phot, missing_sources = single_image_photometry(
        fake_CCDimage,
        photometry_settings,
    )

    phot.sort("aperture_sum")
    sources.sort("amplitude")

    for inp, out in zip(sources, phot, strict=True):
        stdev = inp["x_stddev"]
        expected_flux = (
            inp["amplitude"]
            * 2
            * np.pi
            * stdev**2
            * (1 - np.exp(-(aperture**2) / (2 * stdev**2)))
        )
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
            assert (
                np.abs(expected_flux - out["aperture_net_cnts"].value)
                < expected_deviation
            )
        else:
            with pytest.raises(AssertionError):
                assert (
                    np.abs(expected_flux - out["aperture_net_cnts"].value)
                    < expected_deviation
                )


def list_of_fakes(num_files):
    # Generate fake CCDData objects for use in photometry_on_directory tests
    fake_images = [deepcopy(FAKE_CCD_IMAGE)]

    # Create additional images, each in a different position.
    for i in range(num_files - 1):
        angle = 2 * np.pi / (num_files - 1) * i
        rad = 50
        dx, dy = rad * np.cos(angle), rad * np.sin(angle)
        fake_images.append(shift_FakeCCDImage(fake_images[0], dx, dy))

    filters = ["U", "B", "V", "R", "I"]
    for i in range(num_files):
        if i < 5:
            fake_images[i].header["FILTER"] = filters[i]
        else:
            fake_images[i].header["FILTER"] = "V"

    return fake_images


@pytest.mark.parametrize("coords", ["sky", "pixel"])
def test_photometry_on_directory(coords, photometry_settings):
    # Create list of fake CCDData objects
    num_files = 5
    fake_images = list_of_fakes(num_files)

    # Write fake images to temporary directory and test
    # multi_image_photometry on them.
    # NOTE: ignore_cleanup_errors=True is needed to avoid an error
    #       when the temporary directory is deleted on Windows.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        # Come up with Filenames
        temp_file_names = [
            Path(temp_dir) / f"tempfile_{i:02d}.fit" for i in range(1, num_files + 1)
        ]
        # Write the CCDData objects to files
        for i, image in enumerate(fake_images):
            image.write(temp_file_names[i])

        object_name = fake_images[0].header["OBJECT"]
        sources = fake_images[0].sources
        aperture_settings = DEFAULT_PHOTOMETRY_APERTURES
        aperture = aperture_settings.radius

        # Generate the sourcelist
        found_sources = source_detection(
            fake_images[0], fwhm=fake_images[0].sources["x_stddev"].mean(), threshold=10
        )

        source_list_file = Path(temp_dir) / "source_list.ecsv"
        found_sources.write(source_list_file, format="ascii.ecsv", overwrite=True)

        # Make a copy of photometry options
        phot_options = PhotometryOptionalSettings(**PHOTOMETRY_OPTIONS.model_dump())

        # Modify options to match test before we used phot_options
        phot_options.include_dig_noise = True
        phot_options.reject_too_close = True
        phot_options.reject_background_outliers = True
        phot_options.fwhm_by_fit = True

        photometry_settings.photometry_optional_settings = phot_options
        photometry_settings.source_location_settings.use_coordinates = coords
        photometry_settings.source_location_settings.source_list_file = str(
            source_list_file
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Cannot merge meta key", category=MergeConflictWarning
            )
            phot_data = multi_image_photometry(
                temp_dir,
                photometry_settings,
                object_of_interest=object_name,
            )

    # For following assertion to be true, rad must be small enough that
    # no source lies within outer_annulus of the edge of an image.
    assert len(phot_data) == num_files * len(found_sources)

    # Sort all data by amount of signal
    sources.sort("amplitude")
    found_sources.sort("flux")

    # Get noise level from the first image
    noise_dev = fake_images[0].noise_dev

    for fnd, inp in zip(found_sources, sources, strict=True):
        star_id_chk = fnd["star_id"]
        # Select the rows in phot_data that correspond to the current star
        # and compute the average of the aperture sums.
        selected_rows = phot_data[phot_data["star_id"] == star_id_chk]
        obs_avg_net_cnts = np.average(selected_rows["aperture_net_cnts"].value)

        stdev = inp["x_stddev"]
        expected_flux = (
            inp["amplitude"]
            * 2
            * np.pi
            * stdev**2
            * (1 - np.exp(-(aperture**2) / (2 * stdev**2)))
        )
        # This expected flux is correct IF there were no noise. With noise, the
        # standard deviation in the sum of the noise within in the aperture is
        # n_pix_in_aperture times the single-pixel standard deviation.
        #

        expected_deviation = np.pi * aperture**2 * noise_dev

        # We have two cases to consider: use_coordinates="sky" and
        # use_coordinates="pixel".
        if coords == "sky":
            # In this case, the expected result is the test below.

            # We could require that the result be within some reasonable
            # number of those expected variations or we could count up the
            # actual number of background counts at each of the source
            # positions.

            # Here we just check whether any difference is consistent with
            # less than the expected one sigma deviation.
            assert np.abs(expected_flux - obs_avg_net_cnts) < expected_deviation
        else:
            # In this case we are trying to do photometry in pixel coordinates,
            # using the pixel location of the sources as found in the first image --
            # see the line where found_sources is defined.
            #
            # However, the images are shifted with respect to each other by
            # list_of_fakes, so there are no longer stars at those positions in the
            # other images.
            #
            # Because of that, the expected result is that either obs_avg_net_cnts
            # is nan or the difference is bigger than the expected_deviation.
            assert (
                np.isnan(obs_avg_net_cnts)
                or np.abs(expected_flux - obs_avg_net_cnts) > expected_deviation
            )


def test_photometry_on_directory_with_no_ra_dec(photometry_settings):
    # Create list of fake CCDData objects
    num_files = 5
    fake_images = list_of_fakes(num_files)

    # Write fake images to temporary directory and test
    # multi_image_photometry on them.
    # NOTE: ignore_cleanup_errors=True is needed to avoid an error
    #       when the temporary directory is deleted on Windows.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        # Come up with Filenames
        temp_file_names = [
            Path(temp_dir) / f"tempfile_{i:02d}.fits" for i in range(1, num_files + 1)
        ]
        # Write the CCDData objects to files
        for i, image in enumerate(fake_images):
            image.write(temp_file_names[i])

        object_name = fake_images[0].header["OBJECT"]

        # Generate the sourcelist
        found_sources = source_detection(
            fake_images[0], fwhm=fake_images[0].sources["x_stddev"].mean(), threshold=10
        )

        # Damage the sourcelist by removing the ra and dec columns
        found_sources.drop_ra_dec()

        source_list_file = Path(temp_dir) / "source_list.ecsv"
        found_sources.write(source_list_file, format="ascii.ecsv", overwrite=True)

        phot_options = PhotometryOptionalSettings(**PHOTOMETRY_OPTIONS.model_dump())

        # Modify options to match test before we used phot_options
        phot_options.include_dig_noise = True
        phot_options.reject_too_close = True
        phot_options.reject_background_outliers = True
        phot_options.fwhm_by_fit = True

        photometry_settings.photometry_optional_settings = phot_options
        photometry_settings.source_location_settings.source_list_file = str(
            source_list_file
        )
        # The setting below was implicit in the old default
        photometry_settings.source_location_settings.use_coordinates = "sky"

        with pytest.raises(ValueError):
            multi_image_photometry(
                temp_dir,
                photometry_settings,
                object_of_interest=object_name,
            )


def test_photometry_on_directory_with_bad_fits(photometry_settings):
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
        temp_file_names = [
            Path(temp_dir) / f"tempfile_{i:02d}.fits" for i in range(1, num_files + 1)
        ]
        # Write the CCDData objects to files
        for i, image in enumerate(fake_images):
            image.drop_wcs()
            image.write(temp_file_names[i])

        object_name = fake_images[0].header["OBJECT"]

        # Generate the sourcelist with RA/Dec information from a clean image
        found_sources = source_detection(
            clean_fake_images[0],
            fwhm=clean_fake_images[0].sources["x_stddev"].mean(),
            threshold=10,
        )

        source_list_file = Path(temp_dir) / "source_list.ecsv"
        found_sources.write(source_list_file, format="ascii.ecsv", overwrite=True)

        phot_options = PhotometryOptionalSettings(**PHOTOMETRY_OPTIONS.model_dump())

        # Modify options to match test before we used phot_options
        phot_options.include_dig_noise = True
        phot_options.reject_too_close = True
        phot_options.reject_background_outliers = True
        phot_options.fwhm_by_fit = True

        photometry_settings.photometry_optional_settings = phot_options
        photometry_settings.source_location_settings.source_list_file = str(
            source_list_file
        )
        # The settings below was implicit in the old default
        photometry_settings.source_location_settings.use_coordinates = "sky"

        # Since none of the images will be valid, it should raise a RuntimeError
        with pytest.raises(RuntimeError):
            multi_image_photometry(
                temp_dir,
                photometry_settings,
                object_of_interest=object_name,
            )
