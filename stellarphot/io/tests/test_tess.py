import os
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from requests import ConnectionError, ReadTimeout

from stellarphot.io.tess import (
    TOI,
    TessSubmission,
    TessTargetFile,
    tess_photometry_setup,
)

GOOD_HEADER = {
    "date-obs": "2022-06-04T05:44:28.010",
    "filter": "ip",
    "object": "TIC-237205154",
}

GOOD_HEADER_WITH_PLANET = {
    "date-obs": "2022-06-04T05:44:28.010",
    "filter": "ip",
    "object": "TIC-237205154.01",
}

BAD_HEADER = {}


def test_good_header_sucess():
    tsub = TessSubmission.from_header(GOOD_HEADER)
    assert tsub.utc_start == "20220604"
    assert tsub.tic_id == 237205154
    assert tsub.planet_number == 0
    assert tsub.telescope_code == ""


def test_good_header_planet_sucess():
    tsub = TessSubmission.from_header(GOOD_HEADER_WITH_PLANET)
    assert tsub.utc_start == "20220604"
    assert tsub.tic_id == 237205154
    assert tsub.planet_number == 1
    assert tsub.telescope_code == ""


def test_bad_header_fails():
    with pytest.raises(ValueError) as e:
        TessSubmission.from_header(BAD_HEADER)
    match_str = ".*UTC date of first image.*filter/passband.*TIC ID number.*"
    assert re.search(match_str, str(e))


def test_base_name():
    tsub = TessSubmission.from_header(GOOD_HEADER_WITH_PLANET, telescope_code="ABS")
    assert tsub.base_name == "TIC237205154-01_20220604_ABS_ip"


def test_seeing_profile():
    tsub = TessSubmission.from_header(GOOD_HEADER_WITH_PLANET, telescope_code="ABS")
    assert tsub.seeing_profile == "TIC237205154-01_20220604_ABS_ip_seeing-profile.png"


def test_valid_method():
    tsub = TessSubmission.from_header(GOOD_HEADER)

    # Missing telescope code, amopng other things...
    assert not tsub._valid()

    # Fix the code, should still not be valid because of planet number
    tsub.telescope_code = "ABC"
    assert not tsub._valid()

    # Set planet number and should be valid
    tsub.planet_number = 1
    assert tsub._valid()

    # Set invalid TIC number
    tsub.tic_id = 10_000_000_000
    assert not tsub._valid()


@pytest.mark.remote_data
def test_target_file():
    # Getting the target information failed on windows, so the
    # first point of this test is to simply succeed in creating the
    # object

    tic_742648307 = SkyCoord(ra=104.733225, dec=49.968739, unit="degree")
    server_down = False

    try:
        tess_target = TessTargetFile(tic_742648307, magnitude=12, depth=10)
    except (ConnectionError, ReadTimeout):
        server_down = True
        tess_target = None  # Assure tess_target is defined so that we can delete it
    except ValueError:
        # The server is technically back but producing garbage....
        server_down = True
        tess_target = None  # Assure tess_target is defined so that we can delete it
    else:
        # Check that the first thing in the list is the tick object
        check_coords = SkyCoord(
            ra=tess_target.table["RA"][0],
            dec=tess_target.table["Dec"][0],
            unit=("hour", "degree"),
        )
        assert tic_742648307.separation(check_coords).arcsecond < 1
    finally:
        # Make sure that regardless of whether the server is down we delete
        # the temporary file that gets created.

        # On windows the temporary file cannot be deleted because if it is
        # then the file immediately vanished. That means we get a warning
        # about having an open file handle. We can ignore that warning
        # because we don't care about it here.

        # The warning doesn't get generated until the last reference to
        # the object is deleted, so we need to do that here instead of
        # letting it happen at the end of the test.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="unclosed file", category=ResourceWarning
            )
            try:
                del tess_target
            except UnboundLocalError:
                pass

        if server_down:
            # The server at University of Louisville is down sometimes, so
            # xfail this test when that happens
            pytest.xfail("TESS/gaia Server down")


class TestTOI:
    @pytest.fixture
    def sample_toi(self):
        toi_dict = {
            "tic_id": 236158940,
            "coord": {
                "ra": "313d25m10.33459171s",
                "dec": "34d21m05.92981069s",
                "representation_type": "spherical",
                "frame": "icrs",
            },
            "depth_ppt": 3.31,
            "depth_error_ppt": 0.13131700000000002,
            "duration": "2.885 h",
            "duration_error": "0.286 h",
            "epoch": {
                "jd1": 2459817.0,
                "jd2": 0.28766999999061227,
                "format": "jd",
                "scale": "tdb",
                "precision": 3,
                "in_subfmt": "*",
                "out_subfmt": "*",
            },
            "epoch_error": "0.0018336 d",
            "period": "2.677573 d",
            "period_error": "1.8e-05 d",
            "tess_mag": 11.3283,
            "tess_mag_error": 0.015,
        }
        return TOI.model_validate(toi_dict)

    @pytest.mark.remote_data
    def test_from_tic_id(self, tess_tic_expected_values):
        # Nothing special about the TIC ID chosen here. It is one we happened
        # to be looking at when writing this test.
        tic_id = tess_tic_expected_values["tic_id"]
        toi_info = TOI.from_tic_id(tic_id)
        assert toi_info.tic_id == tic_id

        # Test the coordinate, but not other properties because those may change
        # over time as the planet candidate is better refined. The coordinate
        # should always be the same.
        expected_coord = tess_tic_expected_values["expected_coords"]
        assert toi_info.coord.separation(expected_coord).arcsecond < 1

        # Try round-tripping through json
        # This is a regression test for #427
        json_str = toi_info.model_dump_json()
        new_toi = TOI.model_validate_json(json_str)

        assert toi_info.coord.separation(new_toi.coord).arcsecond < 0.01

    @pytest.mark.parametrize("start_before_midpoint", [True, False])
    def test_transit_time_for_observation(self, sample_toi, start_before_midpoint):
        # For this test we are checking that the correct transit time is identified
        # for a given observation time.
        # For the sake of the test, we are going to use the 124th transit of the TOI
        # as the reference transit.
        reference_midpoint = sample_toi.epoch + 124 * sample_toi.period
        if start_before_midpoint:
            # Start the observation before the midpoint (and before the transit)
            test_time_start = Time(reference_midpoint - 0.8 * sample_toi.duration)
        else:
            # Start the observation after the midpoint
            test_time_start = Time(reference_midpoint + 0.1 * sample_toi.duration)
        obs_times = test_time_start + np.linspace(0, 2) * sample_toi.duration

        assert sample_toi.transit_time_for_observation(obs_times).jd == pytest.approx(
            reference_midpoint.jd
        )


class TestTessPhotometrySetup:
    # This auto-used fixture changes the working directory to the temporary directory
    # and then changes back to the original directory after the test is done.
    @pytest.fixture(autouse=True)
    def change_to_tmp_dir(self, tmp_path):
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        # Yielding here is important. It means that when the test is done, the remainder
        # of the function will be executed. This is important because the test is run in
        # a temporary directory and we want to change back to the original directory
        # when the test is done.
        yield
        os.chdir(original_dir)

    def test_creation_invalid_input_raises_error(self):
        with pytest.raises(
            ValueError, match="Must provide either TIC ID or TOI object"
        ):
            tess_photometry_setup()

    @pytest.mark.remote_data
    @pytest.mark.parametrize("creation_method", ["tic_id", "toi_object"])
    def test_creation(self, tess_tic_expected_values, creation_method):
        # Check that we can create the necessary files for TESS photometry
        # from a TIC ID.
        tic_id = tess_tic_expected_values["tic_id"]
        if creation_method == "tic_id":
            tess_photometry_setup(tic_id=tic_id)
        else:
            toi_info = TOI.from_tic_id(tic_id)
            tess_photometry_setup(TOI_object=toi_info)

        p_info = Path(f"TIC-{tic_id}-info.json")
        assert p_info.exists()

        p_source_list = Path(f"TIC-{tic_id}-source-list-input.ecsv")
        assert p_source_list.exists()

    @pytest.mark.remote_data
    def test_creation_with_overwrite(self, tess_tic_expected_values):
        # Check to see that an error is raised if the files already exist
        # and the overwrite flag is not set.
        tic_id = tess_tic_expected_values["tic_id"]
        tess_photometry_setup(tic_id=tic_id)

        # Try re-running with both files present, where we hit the error from
        # the source list file first
        with pytest.raises(FileExistsError):
            tess_photometry_setup(tic_id=tic_id)

        # Remove the source list table and try again, this time hitting the
        # error from the info file
        Path(f"TIC-{tic_id}-source-list-input.ecsv").unlink()
        with pytest.raises(FileExistsError):
            tess_photometry_setup(tic_id=tic_id)
