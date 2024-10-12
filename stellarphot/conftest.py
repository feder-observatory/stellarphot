# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure.
import os

import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS

from stellarphot import PhotometryData

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


@pytest.fixture
def tess_tic_expected_values():
    # Use this in tests where you need a valid TOI (or TIC) and a coordinate for it.
    return dict(
        tic_id=236158940,
        expected_coords=SkyCoord(ra=313.41953739, dec=34.35164717, unit="degree"),
    )


@pytest.fixture
def simple_photometry_data():
    # Grab the test photometry file and simplify it a bit.
    data_file = get_pkg_data_filename("tests/data/test_photometry_data.ecsv")
    pd_input = PhotometryData.read(data_file)

    # Keep stars 1, 6, 9, 12, first time slice only
    # These stars have no NaNs in them and we only need one image to generate
    # more test data from that.
    first_slice = pd_input["file"] == "wasp-10-b-S001-R001-C099-r.fit"
    ids = [1, 6, 9, 12]
    good_star = pd_input["star_id"] == ids[0]

    for an_id in ids[1:]:
        good_star = good_star | (pd_input["star_id"] == an_id)

    return pd_input[good_star & first_slice]


@pytest.fixture
def stellphotv1_photometry_data(two_filters):
    """
    Load photometry data form version 1 of stellarphot.

    By default the data has two filters, with the filter name part of the column name,
    e.g. "mag_inst_B" and "mag_inst_ip".

    Parameters
    ----------
    two_filters : bool
        If True, return data with two filters. If False, return data with only the "B"
        filter and column name "mag_inst_B".
    """
    # Grab the test photometry file and simplify it a bit.
    data_file = get_pkg_data_filename("utils/tests/data/sp1-data-two-filters.csv")
    data = Table.read(data_file)
    if not two_filters:
        data = data[data["filter"] == "B"]
        del data["mag_inst_ip"]
    return data
