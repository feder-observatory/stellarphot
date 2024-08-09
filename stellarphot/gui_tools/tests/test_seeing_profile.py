import os
import warnings
from collections import namedtuple

import ipywidgets as ipw
import matplotlib
import pytest
from astropy.nddata import CCDData
from astrowidgets import ImageWidget
from photutils.datasets import make_noise_image

from stellarphot.gui_tools import (
    seeing_profile_functions as spf,
)
from stellarphot.gui_tools.seeing_profile_functions import (
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
from stellarphot.settings.tests.test_models import (
    TEST_CAMERA_VALUES,
)


def test_keybindings():
    def simple_bindmap(bindmap):
        bound_keys = {}
        # The keys of the event map are...messy. This converts them to strings
        for key in bindmap.keys():
            modifier = key[1]
            key_name = key[2]
            bound_keys[str(key[0]) + "".join(modifier) + key_name] = key
        return bound_keys

    # This test assumes the ginga widget backend...
    iw = ImageWidget()
    original_bindings = iw._viewer.get_bindmap().eventmap

    bound_keys = simple_bindmap(original_bindings)
    # Spot check a couple of things before we run our function
    assert "Nonekp_D" in bound_keys
    assert "Nonekp_+" in bound_keys
    assert "Nonekp_left" not in bound_keys

    # rebind
    spf.set_keybindings(iw)
    new_bindings = iw._viewer.get_bindmap().eventmap
    bound_keys = simple_bindmap(new_bindings)
    assert "Nonekp_D" not in bound_keys
    assert "Nonekp_+" in bound_keys
    # Yes, the line below is correct...
    assert new_bindings[bound_keys["Nonekp_left"]]["name"] == "pan_right"


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

    # Get the event handler that updates plots
    handler = profile_widget._make_show_event()

    # Make a mock event object
    Event = namedtuple("Event", ["data_x", "data_y"])
    star_loc_x, star_loc_y = profile_stars["x_mean"][0], profile_stars["y_mean"][0]
    # Sending a mock event will generate plots that we don't want to see
    # so set the matplotlib backend to a non-interactive one
    matplotlib.use("agg")
    # matplotlib generates a warning that we are using a non-interactive backend
    # so filter that warning out for the remainder of the test. There are at least
    # a couple of times we generate this warning as values are changed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        handler(profile_widget.iw, Event(star_loc_x, star_loc_y))

        # The FWHM should be close to 9.6
        assert 9 < profile_widget.aperture_settings.value["fwhm"] < 10

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


def test_seeing_profile_error_messages_no_star(tmp_path):
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

    # Get the event handler that updates plots
    handler = profile_widget._make_show_event()

    # Make a mock event object
    Event = namedtuple("Event", ["data_x", "data_y"])
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
        handler(profile_widget.iw, Event(star_loc_x, star_loc_y))
        assert len(profile_widget.error_console.outputs) == 1
        assert (
            "No star found at this location"
            in profile_widget.error_console.outputs[0]["data"]["text/plain"]
        )

        # Clicking a second time should also just have one error
        handler(profile_widget.iw, Event(star_loc_x, star_loc_y))
        assert len(profile_widget.error_console.outputs) == 1


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
