from math import erf

import numpy as np

from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table
from photutils.datasets import make_gaussian_sources, make_noise_image
from glowing_waffles.source_detection import source_detection

TEST_IMAGE_SHAPE = [400, 500]


def get_source_table():
    data_file = get_pkg_data_filename('data/test_sources.csv')
    return Table.read(data_file)


def fake_image():
    sources = get_source_table()
    source_image = make_gaussian_sources(TEST_IMAGE_SHAPE,
                                   sources)
    noise = make_noise_image(source_image.shape,
                       mean=sources['amplitude'].max() / 100,
                       stddev=1)
    return source_image + noise


test_image = fake_image()


def test_detection_number_sources():
    """
    Make sure we detect the sources in the input table....
    """
    sources = get_source_table()
    found_sources = source_detection(test_image,
                             fwhm=sources['x_stddev'].mean())
    print(found_sources.colnames)
    # Sort by flux so we can reliably match them
    sources.sort('amplitude')
    found_sources.sort('flux')

    # Do we have the right number of sources?
    assert len(sources) == len(found_sources)

    for inp, out in zip(sources, found_sources):
        # Do the positions match?
        assert np.round(out['xcentroid']) == inp['x_mean']
        assert np.round(out['ycentroid']) == inp['y_mean']
