from copy import deepcopy

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Gaussian1D
from astropy.nddata import CCDData
from astropy.utils.data import get_pkg_data_filename
from photutils.datasets import make_noise_image

from stellarphot.photometry import CenterAndProfile, find_center
from stellarphot.photometry.tests.fake_image import make_gaussian_sources_image
from stellarphot.settings import Camera
from stellarphot.settings.constants import TEST_CAMERA_VALUES

SHAPE = (300, 300)
RANDOM_SEED = 1230971

TEST_CAMERA_VALUES = deepcopy(TEST_CAMERA_VALUES)


class TestCenter:
    # The scope="function" below is not required (it is the default option) but
    # being explicit seems good, and we want the scope to be function so that each test
    # gets its own copy of the table.
    # Setting autouse=True means that this fixture will be run for every test in this
    # class. To get the data from the fixture we need to store it somewhere, so we store
    # it in self.profile_stars.
    @pytest.fixture(scope="function", autouse=True)
    def make_star_table(self, profile_stars):
        self.profile_stars = profile_stars

    def test_find_center_no_noise_good_guess(self):
        image = make_gaussian_sources_image(SHAPE, self.profile_stars)
        # Good initial guess, no noise, should converge in one try
        cen1 = find_center(image, (31, 41), max_iters=1)
        np.testing.assert_allclose(cen1, [30, 40], rtol=1e-6)
        self.profile_stars = self.profile_stars[:-1]

    def test_find_center_noise_bad_guess(self):
        image = make_gaussian_sources_image(SHAPE, self.profile_stars)
        noise = make_noise_image(
            SHAPE, distribution="gaussian", mean=0, stddev=5, seed=RANDOM_SEED
        )
        cen2 = find_center(image + noise, [40, 50], max_iters=1)
        # Bad initial guess, noise, should take more than one try...
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(cen2, [30, 40])

    def test_find_center_noise_good_guess(self):
        image = make_gaussian_sources_image(SHAPE, self.profile_stars)
        noise = make_noise_image(
            SHAPE, distribution="gaussian", mean=0, stddev=5, seed=RANDOM_SEED
        )
        # Trying again with several iterations should work
        cen3 = find_center(image + noise, [31, 41], max_iters=20)
        # Tolerance chosen based on some trial and error
        np.testing.assert_allclose(cen3, [30, 40], atol=0.02)

    def test_find_center_no_noise_star_at_edge(self):
        # Trying to put the star at the edge of the initial guess
        image = make_gaussian_sources_image(SHAPE, self.profile_stars)
        cen = find_center(image, [45, 65], max_iters=20)

        np.testing.assert_allclose(cen, [30, 40], atol=0.02)

    def test_find_center_no_star(self):
        # No star anywhere near the original guess
        image = make_gaussian_sources_image(SHAPE, self.profile_stars)
        # Offset the mean from zero to avoid nan center
        noise = make_noise_image(
            SHAPE, distribution="gaussian", mean=1000, stddev=5, seed=RANDOM_SEED
        )

        with pytest.raises(RuntimeError, match="Centroid did not converge on a star"):
            find_center(image + noise, [50, 200], max_iters=10)

    def test_find_center_dim_star(self):
        # Regression test for #352, in which a dim star is improperly centered.
        # The cutout loaded below is from an image of the field of WASP-10, and the star
        # in question has Gaia DR3 ID that is stored in the header. The Gaia position
        # is also stored in the header, and is what is taken to be the "correct"
        # position of the star.
        #
        # There is only one star in this cutout.
        #
        # For the record, the Gaia DR3 ID is 1909763087978475392.

        name = get_pkg_data_filename(data_name="data/center_cutout_1.fits")
        ccd = CCDData.read(name)

        true_coordinate = SkyCoord(ccd.header["gaia_coord"])
        true_pixel_center = ccd.wcs.world_to_pixel(true_coordinate)

        # At least with a faint star and large cutout, the
        # centroid as determined by COM and the centroid determined by the Gaussian fit
        # are about 20 pixels apart (pre-bug-fix).
        with pytest.raises(RuntimeError, match="Centroid did not converge on a star"):
            center = find_center(
                ccd.data,
                (40, 42),
                cutout_size=80,
                max_iters=10,
            )

        # Now try with a much smaller cutout size
        center = find_center(ccd.data, (40, 42), cutout_size=20, max_iters=10)
        # Check that we got a good center...
        assert np.linalg.norm(center - true_pixel_center) < 2


class TestRadialProfile:
    # The scope="function" below is not required (it is the default option) but
    # being explicit seems good, and we want the scope to be function so that each test
    # gets its own copy of the table.
    # Setting autouse=True means that this fixture will be run for every test in this
    # class. To get the data from the fixture we need to store it somewhere, so we store
    # it in self.profile_stars.
    @pytest.fixture(scope="function", autouse=True)
    def make_star_table(self, profile_stars):
        self.profile_stars = profile_stars

    def test_radial_profile(self):
        # Test that both curve of growth and radial profile are correct

        image = make_gaussian_sources_image(SHAPE, self.profile_stars)
        for row in self.profile_stars:
            cen = find_center(image, (row["x_mean"], row["y_mean"]), max_iters=10)

            # The "stars" have FWHM around 9.5, so make the cutouts used for finding the
            # stars fairly big -- the bare minimum would be a radius of 3 FWHM, which is
            # a cutout size around 60.
            rad_prof = CenterAndProfile(
                image, cen, centering_cutout_size=60, profile_radius=30
            )

            # Test that the curve of growth is correct

            # Numerical value below is integral of input 2D gaussian, 2pi A sigma^2
            expected_integral = 2 * np.pi * row["amplitude"] * row["x_stddev"] ** 2
            np.testing.assert_allclose(
                rad_prof.curve_of_growth.profile[-1], expected_integral, atol=50
            )

            # Test that the radial profile is correct by comparing pixel values to a
            # gaussian fit to the profile.
            data_radii, data_counts = rad_prof.pixel_values_in_profile
            expected_profile = rad_prof.radial_profile.gaussian_fit(data_radii)

            np.testing.assert_allclose(data_counts, expected_profile, atol=20)

    def test_radial_profile_exposure_is_nan(self):
        # Check that using an exposure value of NaN returns NaN for the SNR and noise
        image = make_gaussian_sources_image(SHAPE, self.profile_stars)

        cen = find_center(image, (50, 50), max_iters=10)

        rad_prof = CenterAndProfile(
            image, cen, centering_cutout_size=60, profile_radius=30
        )

        c = Camera(**TEST_CAMERA_VALUES)
        assert np.isnan(rad_prof.snr(c, np.nan)[-1])
        assert np.isnan(rad_prof.noise(c, np.nan)[-1])
        assert all(np.isfinite(rad_prof.curve_of_growth.profile))
        assert all(np.isfinite(rad_prof.radial_profile.profile))

    def test_radial_profile_with_background(self):
        # Regression test for #328 -- image with a background level
        image = make_gaussian_sources_image(SHAPE, self.profile_stars)
        noise_stdev = 10
        image = image + make_noise_image(
            image.shape,
            distribution="gaussian",
            mean=100,
            stddev=noise_stdev,
            seed=20250404,
        )
        for row in self.profile_stars:
            cen = find_center(image, (row["x_mean"], row["y_mean"]), max_iters=10)

            # The "stars" have FWHM around 9.5, so make the cutouts used for finding the
            # stars fairly big -- the bare minimum would be a radius of 3 FWHM, which is
            # a cutout size around 60.
            rad_prof = CenterAndProfile(
                image, cen, centering_cutout_size=60, profile_radius=30
            )

            # Numerical value below is integral of input 2D gaussian, 2pi A sigma^2
            expected_integral = 2 * np.pi * row["amplitude"] * row["x_stddev"] ** 2

            # The standard deviation in the sum of N gaussian random variables with
            # standard deviation SD is
            #    σ = sqrt(N × SD^2)
            # The curve of growth includes the sum of a bunch of pixels which each have
            # a standard deviation of 10, so the standard deviation of the sum of those
            # pixels is given by the formula above, with N being the number of pixels
            # in the curve.

            expected_stddev = np.sqrt(
                rad_prof.curve_of_growth.area[-1] * noise_stdev**2
            )
            print(
                expected_stddev,
                rad_prof.curve_of_growth.profile[-1] - expected_integral,
            )

            # With the seed above the difference is just under 2.0 standard deviations.
            np.testing.assert_allclose(
                rad_prof.curve_of_growth.profile[-1],
                expected_integral,
                atol=2.0 * expected_stddev,
            )

            # Test that the radial profile is correct by comparing pixel values to a
            # gaussian fit to the profile.
            data_radii, data_counts = rad_prof.pixel_values_in_profile
            expected_profile = rad_prof.radial_profile.gaussian_fit(data_radii)

            # The test here is that the difference between the actual profile and the
            # expected is itself a Gaussian distribution with standard deviation very
            # roughly equal to the standard deviation of the noise we put in.
            differences = data_counts - expected_profile
            counts, bin_edges = np.histogram(differences)
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
            g1d_init = Gaussian1D()
            fitter = LevMarLSQFitter()
            g1d = fitter(g1d_init, bin_centers, counts)

            assert np.abs(g1d.stddev.value - noise_stdev) < 1.5
            assert np.abs(g1d.mean.value) < 1

    def test_radial_profile_bigger_profile_than_cutout(self):
        # Test that the cutout, used for finding the star, can be smaller than
        # the profile radius.
        image = make_gaussian_sources_image(SHAPE, self.profile_stars)

        # Just look at one star in this test, the last one, which is far from the edges.
        # Prior to a change in CenterAndProfile, this would have raised an error because
        # the same cutout size was used for the profile and the centering. The result if
        # the profile size was larger was that the profile eventually had only NaNs in
        # the outermost annuli.
        profile = CenterAndProfile(
            image,
            (self.profile_stars["x_mean"][-1], self.profile_stars["y_mean"][-1]),
            centering_cutout_size=20,
            profile_radius=50,
        )

        assert profile.profile_cutout.shape == (100, 100)

    def test_radial_profile_no_profile_size(self):
        # Test that when we do not provide a profile size it is half the cutout size
        image = make_gaussian_sources_image(SHAPE, self.profile_stars)
        profile = CenterAndProfile(
            image,
            (self.profile_stars["x_mean"][-1], self.profile_stars["y_mean"][-1]),
            centering_cutout_size=50,
        )

        # Last point should be the average of the last two bin edges, which is 24.5
        assert profile.radial_profile.radius[-1] == 24.5
