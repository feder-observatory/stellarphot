import numpy as np

import pytest

from photutils.datasets import make_gaussian_sources_image, make_noise_image
from astropy.table import Table
from astrowidgets import ImageWidget

from stellarphot.visualization import seeing_profile_functions as spf

# Make a few round stars
STARS = Table(dict(amplitude=[1000, 200, 300],
                   x_mean=[30, 100, 150],
                   y_mean=[40, 110, 160],
                   x_stddev=[4, 4, 4],
                   y_stddev=[4, 4, 4],
                   theta=[0, 0, 0]
                   )
)
SHAPE = (300, 300)
RANDOM_SEED = 1230971

def test_keybindings():
    def simple_bindmap(bindmap):
        bound_keys = {}
        # The keys of the event map are...messy. This converts them to strings
        for key in bindmap.keys():
            modifier = key[1]
            key_name = key[2]
            bound_keys[str(key[0]) + ''.join(modifier) + key_name] = key
        return bound_keys

    # This test assumes the ginga widget backend...
    iw = ImageWidget()
    original_bindings = iw._viewer.get_bindmap().eventmap

    bound_keys = simple_bindmap(original_bindings)
    # Spot check a couple of things before we run our function
    assert 'Nonekp_D' in bound_keys
    assert 'Nonekp_+' in bound_keys
    assert 'Nonekp_left' not in bound_keys

    # rebind
    spf.set_keybindings(iw)
    new_bindings = iw._viewer.get_bindmap().eventmap
    bound_keys = simple_bindmap(new_bindings)
    assert 'Nonekp_D' not in bound_keys
    assert 'Nonekp_+' in bound_keys
    # Yes, the line below is correct...
    assert new_bindings[bound_keys['Nonekp_left']]['name'] == 'pan_right'


def test_find_center_no_noise_good_guess():
    image = make_gaussian_sources_image(SHAPE, STARS)
    # Good initial guess, no noise, should converge in one try
    cen1 = spf.find_center(image, (31, 41), max_iters=1)
    np.testing.assert_allclose(cen1, [30, 40])


def test_find_center_noise_bad_guess():
    image = make_gaussian_sources_image(SHAPE, STARS)
    noise = make_noise_image(SHAPE, distribution='gaussian', mean=0, stddev=5,
                             seed=RANDOM_SEED)
    cen2 = spf.find_center(image + noise, [40, 50], max_iters=1)
    # Bad initial guess, noise, should take more than one try...
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(cen2, [30, 40])


def test_find_center_noise_good_guess():
    image = make_gaussian_sources_image(SHAPE, STARS)
    noise = make_noise_image(SHAPE, distribution='gaussian', mean=0, stddev=5,
                             seed=RANDOM_SEED)
    # Trying again with several iterations should work
    cen3 = spf.find_center(image + noise, [31, 41], max_iters=10)
    # Tolerance chosen based on some trial and error
    np.testing.assert_allclose(cen3, [30, 40], atol=0.02)


def test_find_center_no_noise_star_at_edge():
    # Trying to put the star at the edge of the initial guess
    image = make_gaussian_sources_image(SHAPE, STARS)
    cen = spf.find_center(image, [45, 65], max_iters=10)
    np.testing.assert_allclose(cen, [30, 40])


def test_find_center_no_star():
    # No star anywhere near the original guess
    image = make_gaussian_sources_image(SHAPE, STARS)
    # Offset the mean from zero to avoid nan center
    noise = make_noise_image(SHAPE, distribution='gaussian',
                             mean=1000, stddev=5, seed=RANDOM_SEED)
    cen = spf.find_center(image + noise, [50, 200], max_iters=10)
    assert (np.abs(cen[0] - 50) > 1) and (np.abs(cen[1] - 200) > 1)


def test_radial_profile():
    image = make_gaussian_sources_image(SHAPE, STARS)
    for row in STARS:
        cen = spf.find_center(image, (row['x_mean'], row['y_mean']),
                              max_iters=10)
        print(row)
        r_ex, r_a, radprof = spf.radial_profile(image, cen)
        r_exs, r_as, radprofs = spf.radial_profile(image, cen,
                                                   return_scaled=False)

        # Numerical value below is integral of input 2D gaussian, 2pi A sigma^2
        expected_integral = 2 * np.pi * row['amplitude'] * row['x_stddev']**2
        print(expected_integral, radprofs.sum())
        np.testing.assert_allclose(radprofs.sum(), expected_integral, atol=50)
