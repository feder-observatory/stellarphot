import os
import warnings
from copy import deepcopy

import ipywidgets as ipw
import matplotlib
import pytest
from astropy.nddata import CCDData
from photutils.datasets import make_noise_image

from stellarphot.gui import (
    seeing_profile_functions as spf,
)
from stellarphot.gui.seeing_profile_functions import (
    AP_SETTING_NEEDS_SAVE,
    AP_SETTING_SAVED,
)
from stellarphot.photometry.tests.fake_image import make_gaussian_sources_image
from stellarphot.photometry.tests.test_profiles import RANDOM_SEED, SHAPE
from stellarphot.settings import (
    Camera,
    Observatory,
    PhotometryApertures,
    PhotometryWorkingDirSettings,
    settings_files,  # This import is needed for mocking
)
from stellarphot.settings.constants import (
    TEST_CAMERA_VALUES,
)

TEST_CAMERA_VALUES = deepcopy(TEST_CAMERA_VALUES)


def make_click_event(x, y):
    """
    Make a bqplot mouse-click payload like the one the front end sends.
    """
    return {"event": "click", "domain": {"x": x, "y": y}}


def test_seeing_profile_object_creation():
    # This test simply makes sure we can create the object
    profile_widget = spf.SeeingProfileWidget()
    assert isinstance(profile_widget.box, ipw.Box)


@pytest.fixture(autouse=True)
def fake_settings_dir(mocker, tmp_path):
    # See test_settings_files.py for more information on this fixture.
    # It makes a fake settings directory for each test to use.

    # stellarphot is added to the name of the directory to make sure we start
    # without a stellarphot directory for each test.
    mocker.patch.object(
        settings_files.PlatformDirs, "user_data_dir", tmp_path / "stellarphot"
    )


def test_seeing_profile_properties(tmp_path, profile_stars):
    # Here we make a seeing profile then load an image.
    profile_widget = spf.SeeingProfileWidget(
        camera=Camera(**TEST_CAMERA_VALUES), _testing_path=tmp_path
    )

    # Make a fits file
    image = make_gaussian_sources_image(SHAPE, profile_stars) + make_noise_image(
        SHAPE, mean=10, stddev=100, seed=RANDOM_SEED
    )

    ccd = CCDData(image, unit="adu")
    ccd.header["exposure"] = 30.0
    ccd.header["object"] = "test"
    file_name = tmp_path / "test.fits"
    ccd.write(file_name)

    # Load the file
    profile_widget.fits_file.set_file("test.fits", tmp_path)
    profile_widget.load_fits()

    # Check that a couple of properties are set correctly
    assert profile_widget.object_name == "test"
    assert profile_widget.exposure == 30.0

    # Check that the photometry apertures have defaulted to the
    # default values in the model.
    assert profile_widget.aperture_settings.value == PhotometryApertures().model_dump()

    star_loc_x, star_loc_y = profile_stars["x_mean"][0], profile_stars["y_mean"][0]
    # Sending a mock event will generate plots that we don't want to see
    # so set the matplotlib backend to a non-interactive one
    matplotlib.use("agg")
    # matplotlib generates a warning that we are using a non-interactive backend
    # so filter that warning out for the remainder of the test. There are at least
    # a couple of times we generate this warning as values are changed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Simulate a click by sending the same message the bqplot front end
        # would send to the production click dispatcher.
        profile_widget._on_click_message(
            profile_widget.iw._astro_im.interaction,
            make_click_event(star_loc_x, star_loc_y),
            [],
        )

        # The FWHM should be close to 9.6
        assert 9 < profile_widget.aperture_settings.value["fwhm_estimate"] < 10

        # Get a copy of the current aperture settings
        phot_aps = dict(profile_widget.aperture_settings.value)
        new_radius = phot_aps["radius"] - 2
        # Change the radius by directly setting the value of the widget that holds
        # the value. That ends up being nested fairly deeply...
        profile_widget.aperture_settings.di_widgets["radius"].value = new_radius

        # Make sure the settings are updated
        phot_aps["radius"] = new_radius
        assert profile_widget.aperture_settings.value == phot_aps


def test_seeing_profile_save_apertures(tmp_path):
    # Make sure that saving partial photometery settings works
    os.chdir(tmp_path)

    phot_settings = PhotometryWorkingDirSettings()

    # There should be no saved settings...
    with pytest.raises(ValueError, match="does not exist"):
        phot_settings.load()

    profile_widget = spf.SeeingProfileWidget(
        camera=Camera(**TEST_CAMERA_VALUES), _testing_path=tmp_path
    )

    profile_widget.save()
    assert not profile_widget.aperture_settings.savebuttonbar.unsaved_changes

    settings = phot_settings.load()
    assert settings.camera == Camera(**TEST_CAMERA_VALUES)
    assert settings.photometry_apertures == PhotometryApertures(
        radius=1, annulus_width=1, gap=1
    )


def test_seeing_profile_save_box_title(tmp_path):
    # Save box title ends with different characters depending on whether values
    # need to be saved.

    # Change working directory since settings file is saved there.
    os.chdir(tmp_path)

    profile_widget = spf.SeeingProfileWidget(
        camera=Camera(**TEST_CAMERA_VALUES), _testing_path=tmp_path
    )
    profile_widget.aperture_settings.di_widgets["radius"].value = 3

    assert profile_widget.aperture_settings.savebuttonbar.unsaved_changes
    assert AP_SETTING_NEEDS_SAVE in profile_widget.ap_title.value

    profile_widget.save()
    assert AP_SETTING_SAVED in profile_widget.ap_title.value


def test_seeing_profile_error_messages_no_star(tmp_path, capsys):
    # Make sure the appropriate error message is displayed when a click happens on
    # a region with no star, and that the message only appears once.
    profile_widget = spf.SeeingProfileWidget(
        camera=Camera(**TEST_CAMERA_VALUES), _testing_path=tmp_path
    )
    # Make a fits file with no stars
    image = make_noise_image(SHAPE, mean=10, stddev=1, seed=RANDOM_SEED)

    ccd = CCDData(image, unit="adu")
    ccd.header["exposure"] = 30.0
    ccd.header["object"] = "test"
    file_name = tmp_path / "test.fits"
    ccd.write(file_name)

    # Load the file
    profile_widget.fits_file.set_file("test.fits", tmp_path)
    profile_widget.load_fits()

    star_loc_x, star_loc_y = SHAPE[0] // 2, SHAPE[1] // 2
    # Sending a mock event will generate plots that we don't want to see
    # so set the matplotlib backend to a non-interactive one
    matplotlib.use("agg")
    # matplotlib generates a warning that we are using a non-interactive backend
    # so filter that warning out for the remainder of the test. There are at least
    # a couple of times we generate this warning as values are changed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert len(profile_widget.error_console.outputs) == 0

        # Clicking once should generate an error...
        profile_widget._on_click_message(
            profile_widget.iw._astro_im.interaction,
            make_click_event(star_loc_x, star_loc_y),
            [],
        )
        assert len(profile_widget.error_console.outputs) == 1
        assert (
            "No star found at this location"
            in profile_widget.error_console.outputs[0]["data"]["text/plain"]
        )

        # Clicking a second time should also just have one error
        profile_widget._on_click_message(
            profile_widget.iw._astro_im.interaction,
            make_click_event(star_loc_x, star_loc_y),
            [],
        )
        assert len(profile_widget.error_console.outputs) == 1

    # The message reaches the user through error_console only; it should
    # not also be printed to stdout. (Other widgets write terminal control
    # sequences to stdout, so only check for the message itself.)
    assert "No star found" not in capsys.readouterr().out


def test_click_dispatcher_ignores_non_click_events(tmp_path, profile_stars):
    # Mouse messages other than clicks should not trigger the profile
    # calculation.
    profile_widget = spf.SeeingProfileWidget(
        camera=Camera(**TEST_CAMERA_VALUES), _testing_path=tmp_path
    )

    image = make_gaussian_sources_image(SHAPE, profile_stars) + make_noise_image(
        SHAPE, mean=10, stddev=100, seed=RANDOM_SEED
    )
    ccd = CCDData(image, unit="adu")
    ccd.header["exposure"] = 30.0
    ccd.header["object"] = "test"
    file_name = tmp_path / "test.fits"
    ccd.write(file_name)

    profile_widget.fits_file.set_file("test.fits", tmp_path)
    profile_widget.load_fits()

    star_loc_x, star_loc_y = profile_stars["x_mean"][0], profile_stars["y_mean"][0]
    profile_widget._on_click_message(
        profile_widget.iw._astro_im.interaction,
        {"event": "mousemove", "domain": {"x": star_loc_x, "y": star_loc_y}},
        [],
    )

    # No profile should have been computed
    assert profile_widget.rad_prof is None


def test_seeing_profile_no_observatory():
    # This test checks that with no observatory set, there is no TESS
    # related box displayed.
    profile_widget = spf.SeeingProfileWidget()
    assert profile_widget.save_toggle is None
    assert profile_widget.tess_box.layout.visibility == "hidden"


@pytest.mark.parametrize("tess_code", [None, "dummy"])
def test_seeing_profile_with_observatory(tess_code):
    # Test two cases here:
    #  1. The observatory has TESS_telescope_code set to a string
    #  2. The observatory has TESS_telescope_code set to None
    observatory = Observatory(
        name="test",
        latitude=0.0,
        longitude=0.0,
        elevation="0.0 m",
        TESS_telescope_code=tess_code,
    )
    profile_widget = spf.SeeingProfileWidget(observatory=observatory)
    if tess_code is None:
        assert profile_widget.save_toggle is None
    else:
        assert isinstance(profile_widget.save_toggle, ipw.ToggleButton)
