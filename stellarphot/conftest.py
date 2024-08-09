# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure.
import os

import pytest
from astropy.table import Table
from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS

# astropy-specific-stuff


def pytest_configure(config):
    from astropy.utils.iers import conf as iers_conf

    # Disable IERS auto download for testing
    iers_conf.auto_download = False

    config.option.astropy_header = True
    PYTEST_HEADER_MODULES.pop("h5py", None)

    from .version import version

    packagename = os.path.basename(os.path.dirname(__file__))
    TESTED_VERSIONS[packagename] = version


def pytest_unconfigure():
    from astropy.utils.iers import conf as iers_conf

    # Undo IERS auto download setting for testing
    iers_conf.reset("auto_download")


# stellarphot fixtures


@pytest.fixture
def profile_stars():
    # Make a few round stars
    return Table(
        dict(
            amplitude=[1000, 200, 300],
            x_mean=[30, 100, 150],
            y_mean=[40, 110, 160],
            x_stddev=[4, 4, 4],
            y_stddev=[4, 4, 4],
            theta=[0, 0, 0],
        )
    )
