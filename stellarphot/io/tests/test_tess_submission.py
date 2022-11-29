import re

import pytest

from stellarphot.io.tess import TessSubmission

GOOD_HEADER = {
    "date-obs": "2022-06-04T05:44:28.010",
    "filter": "ip",
    "object": "TIC-237205154"
}

GOOD_HEADER_WITH_PLANET = {
    "date-obs": "2022-06-04T05:44:28.010",
    "filter": "ip",
    "object": "TIC-237205154.01"
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