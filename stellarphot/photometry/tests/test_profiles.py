import numpy as np
import pytest
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Gaussian1D
from astropy.table import Table
from photutils.datasets import make_gaussian_sources_image, make_noise_image

from stellarphot.photometry import CenterAndProfile, find_center
from stellarphot.settings import Camera
from stellarphot.settings.tests.test_models import TEST_CAMERA_VALUES

# Make a few round stars
STARS = Table(
    dict(
        amplitude=[1000, 200, 300],
        x_mean=[30, 100, 150],
        y_mean=[40, 110, 160],
        x_stddev=[4, 4, 4],
        y_stddev=[4, 4, 4],
        theta=[0, 0, 0],
    )
)
SHAPE = (300, 300)
RANDOM_SEED = 1230971


def test_find_center_no_noise_good_guess():
    image = make_gaussian_sources_image(SHAPE, STARS)
    # Good initial guess, no noise, should converge in one try
    cen1 = find_center(image, (31, 41), max_iters=1)
    np.testing.assert_allclose(cen1, [30, 40])


def test_find_center_noise_bad_guess():
    image = make_gaussian_sources_image(SHAPE, STARS)
    noise = make_noise_image(
        SHAPE, distribution="gaussian", mean=0, stddev=5, seed=RANDOM_SEED
    )
    cen2 = find_center(image + noise, [40, 50], max_iters=1)
    # Bad initial guess, noise, should take more than one try...
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(cen2, [30, 40])


def test_find_center_noise_good_guess():
    image = make_gaussian_sources_image(SHAPE, STARS)
    noise = make_noise_image(
        SHAPE, distribution="gaussian", mean=0, stddev=5, seed=RANDOM_SEED
    )
    # Trying again with several iterations should work
    cen3 = find_center(image + noise, [31, 41], max_iters=20)
    # Tolerance chosen based on some trial and error
    np.testing.assert_allclose(cen3, [30, 40], atol=0.02)


def test_find_center_no_noise_star_at_edge():
    # Trying to put the star at the edge of the initial guess
    image = make_gaussian_sources_image(SHAPE, STARS)
    cen = find_center(image, [45, 65], max_iters=20)

    np.testing.assert_allclose(cen, [30, 40], atol=0.02)


def test_find_center_no_star():
    # No star anywhere near the original guess
    image = make_gaussian_sources_image(SHAPE, STARS)
    # Offset the mean from zero to avoid nan center
    noise = make_noise_image(
        SHAPE, distribution="gaussian", mean=1000, stddev=5, seed=RANDOM_SEED
    )

    with pytest.raises(RuntimeError, match="Centroid did not converge on a star"):
        find_center(image + noise, [50, 200], max_iters=10)


def test_radial_profile():
    # Test that both curve of growth and radial profile are correct

    image = make_gaussian_sources_image(SHAPE, STARS)
    for row in STARS:
        cen = find_center(image, (row["x_mean"], row["y_mean"]), max_iters=10)

        # The "stars" have FWHM around 9.5, so make the cutouts used for finding the
        # stars fairly big -- the bare minimum would be a radius of 3 FWHM, which is a
        # cutout size around 60.
        rad_prof = CenterAndProfile(image, cen, cutout_size=60, profile_radius=30)

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


def test_radial_profile_exposure_is_nan():
    # Check that using an exposure value of NaN returns NaN for the SNR and noise
    image = make_gaussian_sources_image(SHAPE, STARS)

    cen = find_center(image, (50, 50), max_iters=10)

    rad_prof = CenterAndProfile(image, cen, cutout_size=60, profile_radius=30)

    c = Camera(**TEST_CAMERA_VALUES)
    assert np.isnan(rad_prof.snr(c, np.nan)[-1])
    assert np.isnan(rad_prof.noise(c, np.nan)[-1])
    assert all(np.isfinite(rad_prof.curve_of_growth.profile))
    assert all(np.isfinite(rad_prof.radial_profile.profile))


def test_radial_profile_with_background():
    # Regression test for #328 -- image with a background level
    image = make_gaussian_sources_image(SHAPE, STARS)
    noise_stdev = 10
    image = image + make_noise_image(
        image.shape, distribution="gaussian", mean=100, stddev=noise_stdev, seed=43917
    )
    for row in STARS:
        cen = find_center(image, (row["x_mean"], row["y_mean"]), max_iters=10)

        # The "stars" have FWHM around 9.5, so make the cutouts used for finding the
        # stars fairly big -- the bare minimum would be a radius of 3 FWHM, which is a
        # cutout size around 60.
        rad_prof = CenterAndProfile(image, cen, cutout_size=60, profile_radius=30)

        # Numerical value below is integral of input 2D gaussian, 2pi A sigma^2
        expected_integral = 2 * np.pi * row["amplitude"] * row["x_stddev"] ** 2

        # The standard deviation in the sum of N gaussian random variables with
        # standard deviation SD is
        #    σ = sqrt(N × SD^2)
        # The curve of growth includes the sume of a bunch of pixels which each have a
        # standard deviation of 10, so the standard deviation of the sum of those pixels

        expected_stddev = np.sqrt(rad_prof.curve_of_growth.area[-1] * noise_stdev**2)

        # Allow for a 3-sigma tolerance
        np.testing.assert_allclose(
            rad_prof.curve_of_growth.profile[-1],
            expected_integral,
            atol=3 * expected_stddev,
        )

        # Test that the radial profile is correct by comparing pixel values to a
        # gaussian fit to the profile.
        data_radii, data_counts = rad_prof.pixel_values_in_profile
        expected_profile = rad_prof.radial_profile.gaussian_fit(data_radii)

        # The test here is that the difference between the actual profile and the
        # expected is itself a Gaussian distribution with standard deviation very
        # roughly equal to the standardof the noise we put in.
        differences = data_counts - expected_profile
        counts, bin_edges = np.histogram(differences)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        g1d_init = Gaussian1D()
        fitter = LevMarLSQFitter()
        g1d = fitter(g1d_init, bin_centers, counts)

        assert np.abs(g1d.stddev.value - noise_stdev) < 1.5
        assert np.abs(g1d.mean.value) < 1
