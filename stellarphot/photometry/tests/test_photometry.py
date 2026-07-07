import logging
import tempfile
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.io import ascii
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.metadata.exceptions import MergeConflictWarning

from stellarphot.core import SourceListData
from stellarphot.photometry import (
    AperturePhotometry,
    calculate_noise,
    find_too_close,
    source_detection,
)
from stellarphot.photometry.tests.fake_image import FakeCCDImage, shift_FakeCCDImage
from stellarphot.settings import (
    Camera,
    FwhmMethods,
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


@pytest.fixture
def source_locations():
    return SourceLocationSettings(
        source_list_file="",
        shift_tolerance=6,
        use_coordinates="pixel",
    )


@pytest.fixture
def photometry_optional_settings():
    pos = PhotometryOptionalSettings()
    # This used to be the default; it has switched to exact
    pos.partial_pixel_method = "center"
    return pos


@pytest.fixture
def fake_camera():
    # A camera with not unreasonable settings
    return Camera(
        data_unit=u.adu,
        gain=1.0 * u.electron / u.adu,
        name="test camera",
        read_noise=0 * u.electron,
        dark_current=0.1 * u.electron / u.second,
        pixel_scale=1 * u.arcsec / u.pixel,
        max_data_value=40000 * u.adu,
    )


@pytest.fixture
def zero_camera():
    # Camera with no read noise or dark current
    return Camera(
        data_unit=u.adu,
        gain=1.0 * u.electron / u.adu,
        name="test camera",
        read_noise=0 * u.electron,
        dark_current=0.0 * u.electron / u.second,
        pixel_scale=1 * u.arcsec / u.pixel,
        max_data_value=40000 * u.adu,
    )


@pytest.fixture
def fake_obs():
    # Fake observatory location
    return Observatory(
        name="test observatory",
        latitude=0 * u.deg,
        longitude=0 * u.deg,
        elevation=0 * u.m,
    )


# The fake image used for testing
FAKE_CCD_IMAGE = FakeCCDImage(seed=SEED)


# Build default PhotometryOptions for the tests based on the fake image
@pytest.fixture
def photometry_apertures():
    return PhotometryApertures(
        radius=FAKE_CCD_IMAGE.sources["aperture"][0],
        gap=FAKE_CCD_IMAGE.sources["aperture"][0],
        annulus_width=FAKE_CCD_IMAGE.sources["aperture"][0],
        fwhm_estimate=FAKE_CCD_IMAGE.sources["x_stddev"].mean(),
    )


# Passband map for the tests
@pytest.fixture
def passband_map():
    return PassbandMap(
        name="Example Passband Map",
        your_filter_names_to_aavso={
            "B": "B",
            "rp": "SR",
        },
    )


@pytest.fixture
def photometry_settings_for_test(
    fake_camera,
    fake_obs,
    photometry_apertures,
    source_locations,
    photometry_optional_settings,
    passband_map,
):
    return PhotometrySettings(
        camera=fake_camera,
        observatory=fake_obs,
        photometry_apertures=photometry_apertures,
        source_location_settings=source_locations,
        photometry_optional_settings=photometry_optional_settings,
        passband_map=passband_map,
        logging_settings=LoggingSettings(),
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

    def test_create_aperture_photometry(self, tmp_path, photometry_settings_for_test):
        source_list = self.create_source_list()
        source_list_file = tmp_path / "source_list.ecsv"
        source_list.write(source_list_file, overwrite=True)

        # We are fine to modify photometry_settings here because pytest will
        # make a new one for each test.
        photometry_settings_for_test.source_location_settings.source_list_file = str(
            source_list_file
        )

        # Create an AperturePhotometry object
        ap_phot = AperturePhotometry(settings=photometry_settings_for_test)

        # Check that the object was created correctly
        assert ap_phot.settings.camera is photometry_settings_for_test.camera
        assert ap_phot.settings.observatory is photometry_settings_for_test.observatory

    # The True case below is a regression test for #157
    @pytest.mark.parametrize("int_data", [True, False])
    def test_aperture_photometry_no_outlier_rejection(
        self, int_data, tmp_path, photometry_settings_for_test
    ):
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
                0.75
                * photometry_settings_for_test.camera.max_data_value.value
                / fake_CCDimage.data.max()
            )
            # Keep the integer data below max_adu so that no source is
            # saturated. Saturated sources (see #161/#591) are flagged and get
            # aperture_net_cnts set to NaN, which would break the flux
            # comparison below; saturation itself is covered by
            # test_aperture_photometry_flags_saturated_source.
            data = scale_factor * fake_CCDimage.data
            fake_CCDimage.data = data.astype(int)

        # Make a copy of photometry options
        phot_options = (
            photometry_settings_for_test.photometry_optional_settings.model_copy()
        )

        # Modify options to match test before we used phot_options
        phot_options.reject_background_outliers = False
        phot_options.reject_too_close = False
        phot_options.include_dig_noise = True

        # Set the camera noise to the value in the fake image
        unit = photometry_settings_for_test.camera.read_noise
        photometry_settings_for_test.camera.read_noise = fake_CCDimage.noise_dev * unit

        source_list_file = tmp_path / "source_list.ecsv"
        found_sources.write(source_list_file, format="ascii.ecsv", overwrite=True)

        source_locations = (
            photometry_settings_for_test.source_location_settings.model_copy()
        )
        source_locations.source_list_file = str(source_list_file)

        image_file = tmp_path / "fake_image.fits"
        fake_CCDimage.write(image_file, overwrite=True)

        # We can simply change the fields we want to change now that we use
        # a fixture for photometry_settings.
        photometry_settings_for_test.source_location_settings = source_locations
        photometry_settings_for_test.photometry_optional_settings = phot_options

        ap_phot = AperturePhotometry(settings=photometry_settings_for_test)
        phot, missing_sources = ap_phot(image_file)

        phot.sort("aperture_sum")
        sources = fake_CCDimage.sources
        # Astropy tables sort in-place so we need to sort the sources table
        # after the fact.
        sources.sort("amplitude")
        aperture = photometry_settings_for_test.photometry_apertures.radius

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
        # Finally, check the units on the magnitude columns
        assert phot["mag_inst"].unit is None
        assert phot["mag_error"].unit is None

    @pytest.mark.parametrize("reject", [True, False])
    def test_aperture_photometry_with_outlier_rejection(
        self, reject, tmp_path, photometry_settings_for_test
    ):
        """
        Insert some really large pixel values in the annulus and check that
        the photometry is correct when outliers are rejected and is
        incorrect when outliers are not rejected.
        """
        fake_CCDimage = deepcopy(FAKE_CCD_IMAGE)
        sources = fake_CCDimage.sources

        aperture_settings = photometry_settings_for_test.photometry_apertures
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
            begin = int(center_px[0] + inner_annulus + 1)
            end = begin + int(outer_annulus - inner_annulus - 1)
            # Yes, x and y are deliberately reversed below.
            image[center_px[1], begin:end] = 100 * fake_CCDimage.mean_noise

        # Make a copy of photometry options
        phot_options = (
            photometry_settings_for_test.photometry_optional_settings.model_copy()
        )

        # Modify options to match test before we used phot_options
        phot_options.reject_background_outliers = reject
        phot_options.reject_too_close = False
        phot_options.include_dig_noise = True

        # It is fine to modify photometry_settings here because pytest will
        # make a new one for each test.
        photometry_settings_for_test.source_location_settings.source_list_file = str(
            source_list_file
        )
        photometry_settings_for_test.photometry_optional_settings = phot_options

        image_file = tmp_path / "fake_image.fits"
        fake_CCDimage.write(image_file, overwrite=True)

        ap_phot = AperturePhotometry(settings=photometry_settings_for_test)
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

    def test_photometry_method_argument(self, tmp_path, photometry_settings_for_test):
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
        phot_options = (
            photometry_settings_for_test.photometry_optional_settings.model_copy()
        )

        # Do the photometry with method = "center" first
        phot_options.partial_pixel_method = "center"

        photometry_settings_for_test.source_location_settings.source_list_file = str(
            source_list_file
        )
        photometry_settings_for_test.photometry_optional_settings = phot_options

        ap_phot = AperturePhotometry(settings=photometry_settings_for_test)
        phot_center, _ = ap_phot(image_file)

        # Now redo photometry with method="exact". Whether the flux is larger or smaller
        # in this case depends on the exact position of the sources in the image.
        # Here we just check that the flux is different.
        phot_options.partial_pixel_method = "exact"
        photometry_settings_for_test.photometry_optional_settings = phot_options
        ap_phot_exact = AperturePhotometry(settings=photometry_settings_for_test)
        phot_exact, _ = ap_phot_exact(image_file)
        assert np.all(phot_exact["aperture_sum"] != phot_center["aperture_sum"])

    def test_aperture_photometry_flags_saturated_source(
        self, tmp_path, photometry_settings_for_test
    ):
        # Regression test for #591 (which extends #161): a source with
        # saturated pixels in its aperture should be explicitly flagged and
        # have its aperture_net_cnts set to NaN, while the photometry of the
        # other sources is unaffected.
        fake_CCDimage = deepcopy(FAKE_CCD_IMAGE)
        source_list = self.create_source_list()
        source_list_file = tmp_path / "source_list.ecsv"
        source_list.write(source_list_file, overwrite=True)

        max_adu = photometry_settings_for_test.camera.max_data_value.value

        # Saturate a few pixels at the center of the first source, i.e. well
        # inside its photometry aperture.
        saturated_source = source_list[0]
        x_sat = int(saturated_source["xcenter"].value)
        y_sat = int(saturated_source["ycenter"].value)
        fake_CCDimage.data[y_sat : y_sat + 2, x_sat : x_sat + 2] = 2 * max_adu

        photometry_settings_for_test.source_location_settings.source_list_file = str(
            source_list_file
        )

        image_file = tmp_path / "fake_image.fits"
        fake_CCDimage.write(image_file, overwrite=True)

        # Also do photometry on the image without saturated pixels for
        # comparison.
        clean_image_file = tmp_path / "clean_image.fits"
        deepcopy(FAKE_CCD_IMAGE).write(clean_image_file, overwrite=True)

        ap_phot = AperturePhotometry(settings=photometry_settings_for_test)
        phot, _ = ap_phot(image_file)
        clean_phot, _ = ap_phot(clean_image_file)

        saturated_row = phot["star_id"] == saturated_source["star_id"]
        assert saturated_row.sum() == 1

        # The saturated source is flagged and its net counts are NaN rather
        # than a silently wrong value.
        assert np.all(phot["saturated"][saturated_row])
        assert np.all(np.isnan(phot["aperture_net_cnts"][saturated_row].value))

        # The other sources are not flagged and their results match the
        # photometry of the image with no saturated pixels.
        assert not np.any(phot["saturated"][~saturated_row])
        assert np.all(np.isfinite(phot["aperture_net_cnts"][~saturated_row].value))
        clean_saturated_row = clean_phot["star_id"] == saturated_source["star_id"]
        np.testing.assert_allclose(
            phot["aperture_sum"][~saturated_row].value,
            clean_phot["aperture_sum"][~clean_saturated_row].value,
        )

    def test_saturated_source_recentroiding_falls_back_to_wcs(
        self, tmp_path, photometry_settings_for_test
    ):
        # Regression test for #592: a completely saturated source used to
        # produce a NaN centroid that escaped the shift-tolerance check and
        # then crashed the photometry with a ValueError when the NaN position
        # reached CircularAperture. Instead, the source position should fall
        # back to the WCS-derived position and the source should be flagged
        # as saturated.
        fake_CCDimage = deepcopy(FAKE_CCD_IMAGE)
        source_list = self.create_source_list()
        source_list_file = tmp_path / "source_list.ecsv"
        source_list.write(source_list_file, overwrite=True)

        max_adu = photometry_settings_for_test.camera.max_data_value.value

        # Completely saturate the first source over a region larger than the
        # box used for recentroiding so that its centroid cannot be computed.
        saturated_source = source_list[0]
        x_sat = int(saturated_source["xcenter"].value)
        y_sat = int(saturated_source["ycenter"].value)
        half_width = 15
        fake_CCDimage.data[
            y_sat - half_width : y_sat + half_width + 1,
            x_sat - half_width : x_sat + half_width + 1,
        ] = (
            2 * max_adu
        )

        photometry_settings_for_test.source_location_settings.source_list_file = str(
            source_list_file
        )
        photometry_settings_for_test.source_location_settings.use_coordinates = "sky"

        image_file = tmp_path / "fake_image.fits"
        fake_CCDimage.write(image_file, overwrite=True)

        ap_phot = AperturePhotometry(settings=photometry_settings_for_test)

        # This must not raise even though the saturated source has no usable
        # centroid.
        phot, _ = ap_phot(image_file)

        saturated_row = phot["star_id"] == saturated_source["star_id"]
        assert saturated_row.sum() == 1

        # The position of the saturated source falls back to the WCS-derived
        # position, which for this image is essentially the input position.
        assert np.all(
            np.abs(
                phot["xcenter"][saturated_row].value - saturated_source["xcenter"].value
            )
            < 0.01
        )
        assert np.all(
            np.abs(
                phot["ycenter"][saturated_row].value - saturated_source["ycenter"].value
            )
            < 0.01
        )

        # The saturated source is flagged, its net counts are NaN, and the
        # other sources are unaffected.
        assert np.all(phot["saturated"][saturated_row])
        assert np.all(np.isnan(phot["aperture_net_cnts"][saturated_row].value))
        assert not np.any(phot["saturated"][~saturated_row])
        assert np.all(np.isfinite(phot["aperture_net_cnts"][~saturated_row].value))

    @pytest.mark.parametrize("coords", ["sky", "pixel"])
    def test_photometry_on_directory(self, coords, photometry_settings_for_test):
        # Create list of fake CCDData objects
        num_files = 5
        fake_images = self.list_of_fakes(num_files)

        # Set the camera noise to the noise in the test images
        noise_unit = photometry_settings_for_test.camera.read_noise.unit
        photometry_settings_for_test.camera.read_noise = (
            fake_images[0].noise_dev * noise_unit
        )

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
            aperture_settings = photometry_settings_for_test.photometry_apertures
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
            phot_options = (
                photometry_settings_for_test.photometry_optional_settings.model_copy()
            )

            # Modify options to match test before we used phot_options
            phot_options.include_dig_noise = True
            phot_options.reject_too_close = True
            phot_options.reject_background_outliers = True
            phot_options.fwhm_method = FwhmMethods.FIT

            photometry_settings_for_test.photometry_optional_settings = phot_options
            photometry_settings_for_test.source_location_settings.use_coordinates = (
                coords
            )
            photometry_settings_for_test.source_location_settings.source_list_file = (
                str(source_list_file)
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Cannot merge meta key",
                    category=MergeConflictWarning,
                )
                ap_phot = AperturePhotometry(settings=photometry_settings_for_test)
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

    def test_multi_image_photometry_preserves_foreign_log_handlers(
        self, photometry_settings_for_test
    ):
        # Regression test for #153. multi_image_photometry used to clear the
        # root logger's handlers (and remove every handler from its own logger),
        # which destroyed logging configuration that stellarphot did not set up.
        # It must now leave handlers it did not add in place.
        #
        # The foreign root handler added below also guards against a subtler
        # regression: single_image_photometry must still write to the log file
        # even when the root logger already has handlers. It used to rely on
        # multi_image_photometry clearing the root handlers and skipped its own
        # setup (via logger.hasHandlers(), which walks up to root) when any were
        # present.
        num_files = 3
        fake_images = self.list_of_fakes(num_files)

        noise_unit = photometry_settings_for_test.camera.read_noise.unit
        photometry_settings_for_test.camera.read_noise = (
            fake_images[0].noise_dev * noise_unit
        )

        # Add "foreign" handlers -- handlers added by something other than
        # stellarphot -- to both the root logger and the multi_image_photometry
        # logger. These stand in for handlers a downstream app or user might have
        # installed. The asserts at the end of the test check that running
        # photometry leaves these handlers in place instead of nuking handlers it
        # did not create.
        root_logger = logging.getLogger()
        multilogger = logging.getLogger("multi_image_photometry")
        foreign_root_handler = logging.NullHandler()
        foreign_named_handler = logging.NullHandler()
        root_logger.addHandler(foreign_root_handler)
        multilogger.addHandler(foreign_named_handler)

        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
                temp_file_names = [
                    Path(temp_dir) / f"tempfile_{i:02d}.fit"
                    for i in range(1, num_files + 1)
                ]
                for i, image in enumerate(fake_images):
                    image.write(temp_file_names[i])
                object_name = fake_images[0].header["OBJECT"]

                found_sources = source_detection(
                    fake_images[0],
                    fwhm=fake_images[0].sources["x_stddev"].mean(),
                    threshold=10,
                )
                source_list_file = Path(temp_dir) / "source_list.ecsv"
                found_sources.write(
                    source_list_file, format="ascii.ecsv", overwrite=True
                )

                phot_options = (
                    photometry_settings_for_test.photometry_optional_settings.model_copy()
                )
                phot_options.fwhm_method = FwhmMethods.FIT
                photometry_settings_for_test.photometry_optional_settings = phot_options
                source_locations = photometry_settings_for_test.source_location_settings
                source_locations.use_coordinates = "sky"
                source_locations.source_list_file = str(source_list_file)

                logging_settings = (
                    photometry_settings_for_test.logging_settings.model_copy()
                )
                logging_settings.logfile = "test.log"
                photometry_settings_for_test.logging_settings = logging_settings
                # multi_image_photometry writes the log file into the image dir.
                full_logfile = Path(temp_dir) / "test.log"

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Cannot merge meta key",
                        category=MergeConflictWarning,
                    )
                    ap_phot = AperturePhotometry(settings=photometry_settings_for_test)
                    # Calling AperturePhotometry on a directory (rather than a
                    # single file) dispatches to multi_image_photometry, which in
                    # turn calls single_image_photometry once per image. That is
                    # how this test exercises the multi-image logging path.
                    ap_phot(temp_dir, object_of_interest=object_name)

                # single_image_photometry must still write to the log file even
                # though the root logger already has a handler.
                assert full_logfile.exists()
                log_content = full_logfile.read_text()
                assert "Calculating noise for all sources" in log_content

            # Handlers stellarphot did not add must still be present.
            assert foreign_root_handler in root_logger.handlers
            assert foreign_named_handler in multilogger.handlers
        finally:
            root_logger.removeHandler(foreign_root_handler)
            multilogger.removeHandler(foreign_named_handler)

    def test_reject_unmatched_single_missing_source(
        self, monkeypatch, tmp_path, photometry_settings_for_test
    ):
        # Regression test for #474. When exactly one source is missing from at
        # least one image, reject_unmatched must not raise. The old code did
        # ``set([missing_sources])`` in the length-one case, which wraps the
        # list and raises ``TypeError: unhashable type: 'list'``.
        from stellarphot.photometry import photometry as photometry_mod

        # Two images are enough: one source is forced missing on the first image
        # only (see ``fake_single`` below), so the accumulated missing_sources
        # list has length exactly one -- the case that triggered #474 -- while
        # the second image lets us check the source is rejected from every image.
        num_files = 2
        fake_images = self.list_of_fakes(num_files)

        noise_unit = photometry_settings_for_test.camera.read_noise.unit
        photometry_settings_for_test.camera.read_noise = (
            fake_images[0].noise_dev * noise_unit
        )

        temp_file_names = [
            tmp_path / f"tempfile_{i:02d}.fit" for i in range(1, num_files + 1)
        ]
        for i, image in enumerate(fake_images):
            image.write(temp_file_names[i])
        object_name = fake_images[0].header["OBJECT"]

        found_sources = source_detection(
            fake_images[0],
            fwhm=fake_images[0].sources["x_stddev"].mean(),
            threshold=10,
        )
        source_list_file = tmp_path / "source_list.ecsv"
        found_sources.write(source_list_file, format="ascii.ecsv", overwrite=True)

        photometry_settings_for_test.source_location_settings.source_list_file = str(
            source_list_file
        )

        # Force exactly one source to be reported missing, and only on the
        # first image processed, so the accumulated missing_sources list has
        # length exactly one -- the case that triggered #474. With the default
        # settings no sources are dropped naturally (see
        # test_photometry_on_directory), so this mock is the only source missing.
        missing_id = int(found_sources["star_id"][0])
        real_single = photometry_mod.single_image_photometry
        state = {"calls": 0}

        def fake_single(*args, **kwargs):
            phot, missing = real_single(*args, **kwargs)
            state["calls"] += 1
            if state["calls"] == 1:
                missing = list(missing) + [missing_id]
            return phot, missing

        monkeypatch.setattr(photometry_mod, "single_image_photometry", fake_single)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Cannot merge meta key",
                category=MergeConflictWarning,
            )
            ap_phot = AperturePhotometry(settings=photometry_settings_for_test)
            # Before the #474 fix this raised TypeError: unhashable type: 'list'
            phot_data = ap_phot(str(tmp_path), object_of_interest=object_name)

        # The single missing source must have been rejected from every image.
        assert missing_id not in set(phot_data["star_id"])

    def test_photometry_on_directory_with_no_ra_dec(self, photometry_settings_for_test):
        # Create list of fake CCDData objects
        num_files = 5
        fake_images = self.list_of_fakes(num_files)

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

            phot_options = (
                photometry_settings_for_test.photometry_optional_settings.model_copy()
            )

            # Modify options to match test before we used phot_options
            phot_options.include_dig_noise = True
            phot_options.reject_too_close = True
            phot_options.reject_background_outliers = True
            phot_options.fwhm_method = FwhmMethods.FIT

            photometry_settings_for_test.photometry_optional_settings = phot_options
            photometry_settings_for_test.source_location_settings.source_list_file = (
                str(source_list_file)
            )
            # The setting below was implicit in the old default
            photometry_settings_for_test.source_location_settings.use_coordinates = (
                "sky"
            )

            ap_phot = AperturePhotometry(settings=photometry_settings_for_test)
            with pytest.raises(ValueError):
                ap_phot(
                    temp_dir,
                    object_of_interest=object_name,
                )

    def test_photometry_on_directory_with_bad_fits(self, photometry_settings_for_test):
        # Create list of fake CCDData objects
        num_files = 5
        clean_fake_images = self.list_of_fakes(num_files)
        fake_images = self.list_of_fakes(num_files)

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

            phot_options = (
                photometry_settings_for_test.photometry_optional_settings.model_copy()
            )

            # Modify options to match test before we used phot_options
            phot_options.include_dig_noise = True
            phot_options.reject_too_close = True
            phot_options.reject_background_outliers = True
            phot_options.fwhm_method = FwhmMethods.FIT

            photometry_settings_for_test.photometry_optional_settings = phot_options
            photometry_settings_for_test.source_location_settings.source_list_file = (
                str(source_list_file)
            )
            # The settings below was implicit in the old default
            photometry_settings_for_test.source_location_settings.use_coordinates = (
                "sky"
            )

            ap_phot = AperturePhotometry(settings=photometry_settings_for_test)
            # Since none of the images will be valid, it should raise a RuntimeError
            with pytest.raises(RuntimeError):
                ap_phot(
                    temp_dir,
                    object_of_interest=object_name,
                )

    def test_photometry_variable_aperture(self, tmp_path, photometry_settings_for_test):
        # Create a series of images with sources of different FWHM and
        # run photometry on them with a variable aperture radius.
        fwhm_values = [5, 7.5, 10]

        # Set the camera noise and use this as the noise for the image
        noise = 1 * u.electron
        photometry_settings_for_test.camera.read_noise = noise

        fake_images = [
            FakeCCDImage(seed=SEED, fwhm=fwhm, noise_dev=noise.value)
            for fwhm in fwhm_values
        ]
        num_files = len(fake_images)
        # Write fake images to temporary directory and test
        # multi_image_photometry on them.
        # NOTE: ignore_cleanup_errors=True is needed to avoid an error
        #       when the temporary directory is deleted on Windows.

        # Come up with filenames
        temp_file_names = [
            Path(tmp_path) / f"tempfile_{i:02d}.fits" for i in range(1, num_files + 1)
        ]
        # Write the CCDData objects to files
        for i, image in enumerate(fake_images):
            image.write(temp_file_names[i])

        object_name = fake_images[0].header["OBJECT"]
        sources = fake_images[0].sources

        # Get the expected fwhm of the sources, and make sure we use that in
        # the aperture settings.
        aperture_settings = photometry_settings_for_test.photometry_apertures
        fwhm_est = gaussian_sigma_to_fwhm * sources["x_stddev"].mean()
        aperture_settings.fwhm_estimate = fwhm_est

        fwhm_multiplier = 1.5
        # Set the aperture radius to be a function of the FWHM
        aperture_settings.radius = fwhm_multiplier
        aperture_settings.variable_aperture = True

        # Generate the source list for photometry
        wcs = fake_images[0].wcs
        source_coords = wcs.pixel_to_world(
            sources["x_mean"].value, sources["y_mean"].value
        )
        sources["ra"] = source_coords.ra
        sources["dec"] = source_coords.dec
        sources["x_mean"] = sources["x_mean"] * u.pixel
        sources["y_mean"] = sources["y_mean"] * u.pixel
        sources["star_id"] = list(range(len(sources)))
        sources.rename_columns(
            ["x_mean", "y_mean"],
            ["xcenter", "ycenter"],
        )

        source_list_file = Path(tmp_path) / "source_list.ecsv"
        SourceListData(input_data=sources).write(
            source_list_file, format="ascii.ecsv", overwrite=True
        )

        # Make a copy of photometry options
        phot_options = (
            photometry_settings_for_test.photometry_optional_settings.model_copy()
        )

        # Modify options to match test before we used phot_options
        phot_options.include_dig_noise = True
        phot_options.reject_too_close = True
        phot_options.reject_background_outliers = True
        phot_options.fwhm_method = FwhmMethods.FIT

        photometry_settings_for_test.photometry_optional_settings = phot_options
        photometry_settings_for_test.source_location_settings.use_coordinates = "sky"
        photometry_settings_for_test.source_location_settings.source_list_file = str(
            source_list_file
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Cannot merge meta key",
                category=MergeConflictWarning,
            )
            ap_phot = AperturePhotometry(settings=photometry_settings_for_test)
            phot_data = ap_phot(
                tmp_path,
                object_of_interest=object_name,
            )

        grouped = phot_data.group_by("file")
        tolerance = 0.01
        for expected_fwhm, group in zip(fwhm_values, grouped.groups, strict=True):
            # Check that computed fhwm is close
            assert np.allclose(
                group["fwhm_x"].value,
                expected_fwhm,
                rtol=tolerance,
            )
            # Check that the aperture radius is set correctly; use the same tolerance
            # as the fwhm
            assert np.allclose(
                group["aperture"].value, fwhm_multiplier * expected_fwhm, rtol=tolerance
            )

    def test_invalid_path(self, photometry_settings_for_test):
        ap = AperturePhotometry(settings=photometry_settings_for_test)
        with pytest.raises(ValueError, match="is not a valid file or directory"):
            ap("invalid_path")

    # Checking logging for AperturePhotometry for single image photometry.
    @pytest.mark.parametrize("logfile", ["test.log", None])
    @pytest.mark.parametrize("console_log", [True, False])
    def test_logging_single_image(
        self, capsys, logfile, console_log, tmp_path, photometry_settings_for_test
    ):
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
        phot_options = (
            photometry_settings_for_test.photometry_optional_settings.model_copy()
        )
        phot_options.reject_background_outliers = False
        phot_options.reject_too_close = False
        phot_options.include_dig_noise = True

        # Define the source locations settings
        source_locations = (
            photometry_settings_for_test.source_location_settings.model_copy()
        )
        source_locations.source_list_file = str(source_list_file)

        # Define the logging settings
        logging_settings = photometry_settings_for_test.logging_settings.model_copy()
        if logfile:
            # Define the log file and console log settings
            # and make sure to set full path of log file.
            logging_settings.logfile = str(tmp_path / logfile)
            logging_settings.console_log = console_log
            full_logfile = logging_settings.logfile

        photometry_settings_for_test.source_location_settings = source_locations
        photometry_settings_for_test.photometry_optional_settings = phot_options
        photometry_settings_for_test.logging_settings = logging_settings

        # Call the AperturePhotometry class with a single image
        ap_phot = AperturePhotometry(settings=photometry_settings_for_test)
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
        self, capsys, logfile, console_log, photometry_settings_for_test
    ):
        # Create list of fake CCDData objects
        num_files = 5
        fake_images = self.list_of_fakes(num_files)

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
            phot_options = (
                photometry_settings_for_test.photometry_optional_settings.model_copy()
            )
            phot_options.include_dig_noise = True
            phot_options.reject_too_close = True
            phot_options.reject_background_outliers = True
            phot_options.fwhm_method = FwhmMethods.FIT

            photometry_settings_for_test.photometry_optional_settings = phot_options
            photometry_settings_for_test.source_location_settings.use_coordinates = (
                "sky"
            )
            photometry_settings_for_test.source_location_settings.source_list_file = (
                str(source_list_file)
            )

            # Define the logging settings
            logging_settings = (
                photometry_settings_for_test.logging_settings.model_copy()
            )
            if logfile:
                logging_settings.logfile = logfile
                logging_settings.console_log = console_log
                # log file should be written to image directory (tmp_dir)
                # automatically, this ensures we know the path to the log file.
                full_logfile = str(Path(temp_dir) / logfile)
                photometry_settings_for_test.logging_settings = logging_settings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Cannot merge meta key",
                    category=MergeConflictWarning,
                )

                ap_phot = AperturePhotometry(settings=photometry_settings_for_test)
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
def test_calc_noise_source_only(gain, aperture_area, zero_camera):
    # If the only source of noise is Poisson error in the source
    # then the noise should be the square root of the counts.
    counts = 100
    expected = np.sqrt(gain * counts)

    # Create camera instance
    camera = zero_camera.model_copy()
    camera.gain = gain * camera.gain.unit

    np.testing.assert_allclose(
        calculate_noise(camera, counts=counts, aperture_area=aperture_area), expected
    )


@pytest.mark.parametrize("aperture_area", [5, 20])
@pytest.mark.parametrize("gain", GAINS)
def test_calc_noise_dark_only(gain, aperture_area, zero_camera):
    # Gain should not affect this one. Dark current needs a couple other things,
    # but this is basically Poisson error.
    dark_current = 10
    exposure = 20

    # Create camera instance
    camera = zero_camera.model_copy()
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
def test_calc_read_noise_only(gain, aperture_area, zero_camera):
    # The read noise per pixel IS the noise. The only multiplier is
    # the number of pixels.
    read_noise = 10
    expected = np.sqrt(aperture_area * read_noise**2)

    # Create camera instance
    camera = zero_camera.model_copy()
    camera.read_noise = read_noise * camera.read_noise.unit
    camera.gain = gain * camera.gain.unit

    np.testing.assert_allclose(
        calculate_noise(camera, aperture_area=aperture_area), expected
    )


@pytest.mark.parametrize("aperture_area", [5, 20])
@pytest.mark.parametrize("gain", GAINS)
def test_calc_sky_only(gain, aperture_area, zero_camera):
    # The sky noise per pixel is the poisson and per pixel.
    sky = 10
    expected = np.sqrt(gain * aperture_area * sky)

    # Create camera instance
    camera = zero_camera.model_copy()
    camera.gain = gain * camera.gain.unit

    np.testing.assert_allclose(
        calculate_noise(camera, aperture_area=aperture_area, sky_per_pix=sky), expected
    )


def test_annulus_area_term(zero_camera):
    # Test that noise is correct with an annulus
    aperture_area = 20

    # Annulus is typically quite a bit larger than aperture.
    annulus_area = 10 * aperture_area
    gain = 1.5
    sky = 10
    expected = np.sqrt(gain * aperture_area * (1 + aperture_area / annulus_area) * sky)

    # Create camera instance
    camera = zero_camera.model_copy()
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
def test_calc_noise_messy_case(digit, expected, zero_camera):
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
    camera = zero_camera.model_copy()
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
