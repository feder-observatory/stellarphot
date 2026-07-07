import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.utils.data import get_pkg_data_filename
from requests import HTTPError

from stellarphot.conftest import SERVER_DOWN_ERRORS
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
    except SERVER_DOWN_ERRORS:
        server_down = True
        tess_target = None  # Assure tess_target is defined so that we can delete it
    else:
        # Check that the first thing in the list is the TIC object
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


def test_target_file_no_download_link_raises(monkeypatch, tmp_path):
    # When the GAIA aperture service returns HTTP 200 but a body with no
    # download link (i.e. the server is "back but producing garbage"), we
    # should raise a clear, catchable ValueError instead of an opaque
    # ``TypeError: 'NoneType' object is not subscriptable``.
    class FakeResponse:
        status_code = 200
        text = "<html>The server is having a bad day -- no link here.</html>"

    def fake_get(*args, **kwargs):  # noqa: ARG001
        return FakeResponse()

    monkeypatch.setattr("stellarphot.io.tess.requests.get", fake_get)

    coord = SkyCoord(ra=104.733225, dec=49.968739, unit="degree")
    # Provide a file we manage ourselves so TessTargetFile does not create a
    # NamedTemporaryFile that would be left open when construction fails. A
    # leaked open handle raises an unclosed-file ResourceWarning, which is
    # promoted to an error on Windows CI.
    with open(tmp_path / "gaia_targets.dat", "w") as target_file:
        with pytest.raises(ValueError, match="no download link"):
            TessTargetFile(coord, magnitude=12, depth=10, file=target_file)


class FakeExoFOPResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise HTTPError(f"HTTP status {self.status_code}")


# Minimal planet entry with only the keys from_tic_id consumes; values are
# strings (some with leading-dot decimals) exactly as ExoFOP serves them.
TOI_PLANET_ENTRY = {
    "name": "TIC 236158940.01",
    "toi": "TOI 5868.01",
    "epoch": "2459817.28767",
    "epoch_e": ".0018336",
    "per": "2.677573",
    "per_e": ".000018",
    "dur": "2.885",
    "dur_e": ".286",
    "dep_p": "3310",
    "dep_p_e": "131.317",
}

TESS_MAGNITUDE_ENTRY = {"band": "TESS", "value": "11.3283", "value_e": "0.0145"}

TOI_COORDINATES_ENTRY = {"ra": "313.41952362637397", "dec": "+34.351622812290799"}


def make_exofop_payload(planet_parameters, magnitudes=None):
    if magnitudes is None:
        magnitudes = [TESS_MAGNITUDE_ENTRY]
    return {
        "coordinates": TOI_COORDINATES_ENTRY,
        "planet_parameters": planet_parameters,
        "magnitudes": magnitudes,
    }


class TestTOIFromTicIdOffline:
    """
    Test TOI.from_tic_id parsing of the ExoFOP single-target JSON endpoint
    without any network access. See #623.
    """

    @pytest.fixture
    def exofop_payload(self):
        # Payload captured from
        # https://exofop.ipac.caltech.edu/tess/target.php?id=236158940&json
        with open(get_pkg_data_filename("data/tic-236158940-exofop.json")) as f:
            return json.load(f)

    def patch_network(self, monkeypatch, payload):
        """
        Fake every network access from_tic_id makes and record the URL it
        requests. The full-TOI-table download and the MAST query must not
        happen at all.
        """
        calls = {}

        def fake_get(url, *args, **kwargs):  # noqa: ARG001
            calls["url"] = url
            return FakeExoFOPResponse(payload)

        def fail_get_tic_info(*args, **kwargs):  # noqa: ARG001
            raise AssertionError("from_tic_id must not query MAST (#623)")

        def fail_download(*args, **kwargs):  # noqa: ARG001
            raise AssertionError(
                "from_tic_id must not download the full TOI table (#623)"
            )

        monkeypatch.setattr("stellarphot.io.tess.requests.get", fake_get)
        monkeypatch.setattr("stellarphot.io.tess.get_tic_info", fail_get_tic_info)
        # raising=False because once #623 is fixed download_file is no longer
        # imported in stellarphot.io.tess and this setattr becomes a no-op.
        monkeypatch.setattr(
            "stellarphot.io.tess.download_file", fail_download, raising=False
        )
        return calls

    def test_from_tic_id_parses_exofop_json(
        self, monkeypatch, tess_tic_expected_values, exofop_payload
    ):
        tic_id = tess_tic_expected_values["tic_id"]
        calls = self.patch_network(monkeypatch, exofop_payload)

        toi = TOI.from_tic_id(tic_id)

        # The single-target endpoint was hit for this TIC ID
        assert "target.php" in calls["url"]
        assert str(tic_id) in calls["url"]
        assert "json" in calls["url"]

        assert toi.tic_id == tic_id
        expected_coord = tess_tic_expected_values["expected_coords"]
        assert toi.coord.separation(expected_coord).arcsecond < 1
        # Values below are from the captured payload for TIC 236158940
        assert toi.depth_ppt == pytest.approx(3.310)
        assert toi.depth_error_ppt == pytest.approx(0.131317)
        assert toi.duration == 2.885 * u.hour
        assert toi.duration_error == 0.286 * u.hour
        assert toi.epoch.scale == "tdb"
        assert toi.epoch.jd == pytest.approx(2459817.28767)
        assert toi.epoch_error == 0.0018336 * u.day
        assert toi.period == 2.677573 * u.day
        assert toi.period_error == 1.8e-05 * u.day
        assert toi.tess_mag == pytest.approx(11.3283)
        assert toi.tess_mag_error == pytest.approx(0.0145)

        # Offline version of the round-trip regression test for #427
        new_toi = TOI.model_validate_json(toi.model_dump_json())
        assert toi.coord.separation(new_toi.coord).arcsecond < 0.01

    def test_from_tic_id_only_uses_toi_provenance(
        self, monkeypatch, tess_tic_expected_values
    ):
        # A user-supplied CTOI with a different period follows the TOI
        # section; it must be ignored.
        user_entry = TOI_PLANET_ENTRY | {"per": "99.9"}
        payload = make_exofop_payload(
            [
                {"prov": "toi", "prov_title": "TOIs (TESS Project)", "prov_num": "1"},
                TOI_PLANET_ENTRY,
                {"prov": "user", "prov_title": "User", "prov_num": "2"},
                user_entry,
            ]
        )
        self.patch_network(monkeypatch, payload)
        toi = TOI.from_tic_id(tess_tic_expected_values["tic_id"])
        assert toi.period == 2.677573 * u.day

    @pytest.mark.parametrize(
        "planet_parameters,num_found",
        [
            # Only a user provenance -> no TOI entries
            ([{"prov": "user", "prov_num": "1"}, TOI_PLANET_ENTRY], 0),
            # Two planets in the TOI section
            (
                [{"prov": "toi", "prov_num": "1"}, TOI_PLANET_ENTRY, TOI_PLANET_ENTRY],
                2,
            ),
        ],
    )
    def test_from_tic_id_wrong_number_of_tois_raises(
        self, monkeypatch, tess_tic_expected_values, planet_parameters, num_found
    ):
        payload = make_exofop_payload(planet_parameters)
        self.patch_network(monkeypatch, payload)
        with pytest.raises(RuntimeError, match=f"Found {num_found}.*expected one"):
            TOI.from_tic_id(tess_tic_expected_values["tic_id"])

    def test_from_tic_id_missing_tess_mag_raises(
        self, monkeypatch, tess_tic_expected_values
    ):
        payload = make_exofop_payload(
            [{"prov": "toi", "prov_num": "1"}, TOI_PLANET_ENTRY],
            magnitudes=[{"band": "V", "value": "11.701", "value_e": "0.057"}],
        )
        self.patch_network(monkeypatch, payload)
        with pytest.raises(RuntimeError, match="TESS magnitude"):
            TOI.from_tic_id(tess_tic_expected_values["tic_id"])

    def test_from_tic_id_empty_value_raises(
        self, monkeypatch, tess_tic_expected_values
    ):
        # ExoFOP serves missing numbers as empty strings
        payload = make_exofop_payload(
            [{"prov": "toi", "prov_num": "1"}, TOI_PLANET_ENTRY | {"dur_e": ""}]
        )
        self.patch_network(monkeypatch, payload)
        with pytest.raises(RuntimeError, match="transit duration error"):
            TOI.from_tic_id(tess_tic_expected_values["tic_id"])


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
        try:
            toi_info = TOI.from_tic_id(tic_id)
        except SERVER_DOWN_ERRORS as e:
            pytest.xfail(f"ExoFOP server down or misbehaving: {e}")
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

    @pytest.mark.parametrize(
        "phase",
        [-2.6, -2.1, -1.9, -0.9, -0.1, 0.1, 0.9, 123.6],
    )
    def test_transit_time_for_observation_nearest_transit(self, sample_toi, phase):
        # Regression test for #594: the returned transit time should be the
        # transit nearest the first observation time even when the observation
        # is before the tabulated epoch (negative phase).
        expected_midpoint = sample_toi.epoch + round(phase) * sample_toi.period
        test_time_start = sample_toi.epoch + phase * sample_toi.period
        obs_times = test_time_start + np.linspace(0, 2) * sample_toi.duration

        # Observations starting far from a transit generate a warning; the
        # warning must be present in that case and absent otherwise.
        far_from_transit = (
            abs(phase - round(phase)) * sample_toi.period > 3 * sample_toi.duration
        )
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            transit_time = sample_toi.transit_time_for_observation(obs_times)

        far_warnings = [w for w in recorded if "far from a transit" in str(w.message)]
        assert bool(far_warnings) == far_from_transit

        # The identified transit should be the one nearest the start of the
        # observations, so it can be no more than half a period away.
        half_period = 0.5 * sample_toi.period
        assert abs((transit_time - test_time_start).to("day")) <= half_period
        assert transit_time.jd == pytest.approx(expected_midpoint.jd)


class TestTessPhotometrySetup:
    # Auto-use the shared change_to_tmp_dir fixture (defined in the top-level
    # conftest) so every test in this class runs in a temporary directory.
    @pytest.fixture(autouse=True)
    def _change_to_tmp_dir(self, change_to_tmp_dir):
        pass

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
        try:
            if creation_method == "tic_id":
                tess_photometry_setup(tic_id=tic_id)
            else:
                toi_info = TOI.from_tic_id(tic_id)
                tess_photometry_setup(TOI_object=toi_info)
        except SERVER_DOWN_ERRORS as e:
            pytest.xfail(f"TESS/GAIA server down or misbehaving: {e}")

        p_info = Path(f"TIC-{tic_id}-info.json")
        assert p_info.exists()

        p_source_list = Path(f"TIC-{tic_id}-source-list-input.ecsv")
        assert p_source_list.exists()

    @pytest.mark.remote_data
    def test_creation_with_overwrite(self, tess_tic_expected_values):
        # Check to see that an error is raised if the files already exist
        # and the overwrite flag is not set.
        tic_id = tess_tic_expected_values["tic_id"]
        try:
            tess_photometry_setup(tic_id=tic_id)
        except SERVER_DOWN_ERRORS as e:
            pytest.xfail(f"TESS/GAIA server down or misbehaving: {e}")

        # The first call succeeded, so the server is up and the remaining
        # calls below should reach the FileExistsError checks.
        # Try re-running with both files present, where we hit the error from
        # the source list file first
        with pytest.raises(FileExistsError):
            tess_photometry_setup(tic_id=tic_id)

        # Remove the source list table and try again, this time hitting the
        # error from the info file
        Path(f"TIC-{tic_id}-source-list-input.ecsv").unlink()
        with pytest.raises(FileExistsError):
            tess_photometry_setup(tic_id=tic_id)
