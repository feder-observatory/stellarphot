import re
import warnings

import pytest
from astropy.coordinates import SkyCoord
from requests import ConnectionError, ReadTimeout

from stellarphot.io.tess import TOI, TessSubmission, TessTargetFile

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
    @pytest.mark.remote_data
    def test_from_tic_id(self):
        # Nothing special about the TIC ID chosen here. It is one we happened
        # to be looking at when writing this test.
        tic_id = 236158940
        toi_info = TOI.from_tic_id(tic_id)
        assert toi_info.tic_id == tic_id

        # Test the coordinate, but not other properties because those may change
        # over time as the planet candidate is better refined. The coordinate
        # should always be the same.
        expected_coord = SkyCoord(ra=313.41953739, dec=34.35164717, unit="degree")
        assert toi_info.coord.separation(expected_coord).arcsecond < 1
