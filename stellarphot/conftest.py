# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure.
import os
import urllib.error

import matplotlib
import matplotlib.pyplot as plt
import pytest
import requests
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS

from stellarphot import PhotometryData

# Errors that indicate one of the external services the remote-data tests
# depend on (ExoFOP, MAST, the Louisville GAIA aperture service) is down or
# misbehaving. Remote-data tests xfail rather than fail on these.
SERVER_DOWN_ERRORS = (
    # Server unreachable
    requests.exceptions.ConnectionError,
    # Server too slow; covers ReadTimeout and ConnectTimeout
    requests.exceptions.Timeout,
    # raise_for_status() on an error response
    requests.exceptions.HTTPError,
    # astropy/astroquery network calls go through urllib, not requests
    urllib.error.URLError,
    # Raw socket timeouts
    TimeoutError,
    # The GAIA aperture service returning HTTP 200 with no usable
    # download link surfaces as a ValueError
    ValueError,
)


@pytest.fixture(autouse=True)
def _catalog_sources_vizier_only(monkeypatch):
    # Catalog lookups in the tests must not depend on the developer's (or a
    # JupyterHub's) environment, and the non-VizieR backends would reach for
    # the network in tests that mock only VizieR. Tests that want another
    # backend pass ``backends=`` explicitly.
    monkeypatch.setenv("STELLARPHOT_CATALOG_SOURCES", "vizier")
    monkeypatch.delenv("STELLARPHOT_CATALOG_DIR", raising=False)
    monkeypatch.delenv("STELLARPHOT_VIZIER_SERVERS", raising=False)


# astropy-specific-stuff


def pytest_configure(config):
    from astropy.utils.iers import conf as iers_conf

    # The tests never display plots, so use a non-interactive backend.
    matplotlib.use("agg")

    # Disable IERS auto download for testing
    iers_conf.auto_download = False

    config.option.astropy_header = True
    PYTEST_HEADER_MODULES.pop("h5py", None)

    # Always display versions of these packages
    for package in [
        "astroquery",
        "photutils",
        "astrowidgets",
        "astro_image_display_api",
        "bqplot",
        "pydantic",
        "ipyautoui",
    ]:
        PYTEST_HEADER_MODULES[package] = package

    from .version import version

    packagename = os.path.basename(os.path.dirname(__file__))
    TESTED_VERSIONS[packagename] = version


def pytest_unconfigure():
    from astropy.utils.iers import conf as iers_conf

    # Undo IERS auto download setting for testing
    iers_conf.reset("auto_download")


# Guard against tests that change the working directory and do not change it
# back. Every later test then runs, and saves its files, somewhere it did not
# expect. These are hooks rather than a fixture so that the check happens
# after *all* of a test's fixtures (including monkeypatch) have been torn down,
# no matter what order the fixtures were set up in. The change has to be undone
# by the end of each test, so a class- or module-scoped fixture that changes
# the directory fails the guard too.

_cwd_before_test = pytest.StashKey[str]()


def _restore_cwd(item):
    """
    Change back to the directory the test started in. Returns the directory
    the test left behind, or `None` if the test did not leave it changed.
    """
    original_dir = item.stash.get(_cwd_before_test, None)
    if original_dir is None:
        return None
    current_dir = os.getcwd()
    if current_dir == original_dir:
        return None
    os.chdir(original_dir)
    return current_dir


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    item.stash[_cwd_before_test] = os.getcwd()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item):
    try:
        result = yield
    except BaseException:
        # A fixture's teardown failed. Still put the directory back, so later
        # tests are not affected, but report the original error.
        _restore_cwd(item)
        raise
    left_in = _restore_cwd(item)
    if left_in is not None:
        pytest.fail(
            f"Test left the working directory changed to {left_in}. Use "
            "the change_to_tmp_dir fixture (or monkeypatch.chdir) instead of "
            "os.chdir so that the change is undone when the test ends.",
            pytrace=False,
        )
    return result


# stellarphot fixtures


@pytest.fixture(autouse=True)
def close_figures():
    """
    Keep the figures the tests make from piling up.
    """
    yield
    plt.close("all")


@pytest.fixture
def change_to_tmp_dir(tmp_path, monkeypatch):
    # Change the working directory to the temporary directory for the duration
    # of the test; monkeypatch changes back to the original directory when the
    # test is done. This fixture is not autouse; request it explicitly, or
    # apply it to a whole class or module with
    # pytest.mark.usefixtures("change_to_tmp_dir").
    monkeypatch.chdir(tmp_path)


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
