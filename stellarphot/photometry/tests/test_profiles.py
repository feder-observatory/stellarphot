import numpy as np

import pytest

from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning

from photutils.datasets import make_gaussian_sources_image, make_noise_image

from stellarphot.photometry import find_center, CenterAndProfile

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
        cen = find_center(image + noise, [50, 200], max_iters=10)


def test_radial_profile():
    image = make_gaussian_sources_image(SHAPE, STARS)
    for row in STARS:
        cen = find_center(image, (row["x_mean"], row["y_mean"]), max_iters=10)

        # The "stars" have FWHM around 9.5, so make the cutouts used for finding the
        # stars fairly big -- the bare minimum would be a radius of 3 FWHM, which is a
        # cutout size around 60.
        rad_prof = CenterAndProfile(image, cen, cutout_size=60, profile_radius=30)

        # Numerical value below is integral of input 2D gaussian, 2pi A sigma^2
        expected_integral = 2 * np.pi * row["amplitude"] * row["x_stddev"] ** 2

        np.testing.assert_allclose(
            rad_prof.curve_of_growth.profile[-1], expected_integral, atol=50
        )
