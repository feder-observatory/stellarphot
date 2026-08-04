import warnings
from copy import deepcopy
from pathlib import Path

import ipywidgets as ipw
import pytest
from astropy.units import Quantity
from ipyautoui.custom.iterable import ItemBox, ItemControl
from pydantic import ValidationError
from pydantic.alias_generators import to_snake

from stellarphot.gui.custom_widgets import (
    ChooseOrMakeNew,
    Confirm,
    ReviewSettings,
    SaveStatus,
    SettingWithTitle,
    Spinner,
    _add_saving_to_widget,
)
from stellarphot.gui.views import ui_generator
from stellarphot.settings import (
    Camera,
    LoggingSettings,
    Observatory,
    PartialPhotometrySettings,
    PassbandMap,
    PhotometryApertures,
    PhotometryOptionalSettings,
    PhotometrySettings,
    PhotometrySettingsWarning,
    PhotometryWorkingDirSettings,
    SavedSettings,
    SourceLocationSettings,
    settings_files,
)
from stellarphot.settings.constants import (
    TEST_CAMERA_VALUES,
    TEST_OBSERVATORY_SETTINGS,
    TEST_PASSBAND_MAP,
    TEST_PHOTOMETRY_SETTINGS,
)

# Make sure we only use copies of the test settings
TEST_CAMERA_VALUES = deepcopy(TEST_CAMERA_VALUES)
TEST_OBSERVATORY_SETTINGS = deepcopy(TEST_OBSERVATORY_SETTINGS)
TEST_PASSBAND_MAP = deepcopy(TEST_PASSBAND_MAP)
TEST_PHOTOMETRY_SETTINGS = deepcopy(TEST_PHOTOMETRY_SETTINGS)


# See test_settings_file.TestSavedSettings for a detailed description of what the
# following fixture does. In brief, it patches the settings_files.PlatformDirs class
# so that the user_data_dir method returns the temporary directory.
@pytest.fixture(autouse=True)
def fake_settings_dir(mocker, tmp_path):
    mocker.patch.object(
        settings_files.PlatformDirs, "user_data_dir", tmp_path / "stellarphot"
    )


class TestChooseOrMakeNew:
    """
    Class for testing the ChooseOrMakeNew widget.
    """

    def make_test_camera(self):
        """
        Make a camera with the default testing values and save it. This came
        up often enough to warrant its own method.
        """
        saved = SavedSettings()
        camera = Camera(**TEST_CAMERA_VALUES)
        saved.add_item(camera)
        # Return the camera that was saved in case the test wants to use it
        return camera

    def test_creation_without_type_raises_error(self):
        # Should raise an error if no type is provided
        with pytest.raises(TypeError):
            ChooseOrMakeNew()

    def test_creation_unknown_item_type_raises_error(self):
        # Should raise an error if an unknown item type is provided
        with pytest.raises(ValueError, match="Unknown item type unknown"):
            ChooseOrMakeNew("unknown")

    @pytest.mark.parametrize(
        "item_type",
        [
            "camera",
            "observatory",
            "passband_map",
            "Camera",
            "PassbandMap",
            "Observatory",
        ],
    )
    def test_creation(self, item_type):
        # Should create a widget with a dropdown and a button
        choose_or_make_new = ChooseOrMakeNew(item_type)
        assert choose_or_make_new._item_type_name == item_type

    def test_initial_configuration_with_no_items(self):
        # Should have a dropdown with one item, "Make new passband map"
        # using passband_map here to also test that underscores get converted to spaces
        choose_or_make_new = ChooseOrMakeNew("passband_map")
        assert len(choose_or_make_new._choose_existing.options) == 1
        assert choose_or_make_new._choose_existing.options[0] == (
            "Make new passband map",
            "none",
        )

    def test_make_new_makes_a_new_item(self):
        # We will test this with Camera, should work for the others too since the code
        # path is the same.

        choose_or_make_new = ChooseOrMakeNew("camera")
        # Set the values for the new item
        choose_or_make_new._item_widget.value = TEST_CAMERA_VALUES

        # Simulate a click on the save button...
        choose_or_make_new._item_widget.savebuttonbar.bn_save.click()
        # No need for the confirmation dialog because we are not overwriting
        # anything.

        # making sure the widget state is properly updated
        assert choose_or_make_new._making_new is False

        # Check what we created using SavedSettings...
        saved = SavedSettings()
        cameras = saved.get_items("camera")
        assert len(cameras.as_dict) == 1
        assert list(cameras.as_dict.values())[0].model_dump() == TEST_CAMERA_VALUES

    @pytest.mark.parametrize(
        "item_type,setting",
        [
            ("camera", TEST_CAMERA_VALUES),
            ("observatory", TEST_OBSERVATORY_SETTINGS),
            ("passband_map", TEST_PASSBAND_MAP),
        ],
    )
    def test_make_new_with_existing_item_resets_value(self, item_type, setting):
        # When "make a new" item is selected the value of the widget should be
        # the same as when a widget of that type is created.

        # Make a camera widget just to get the value for a new item.
        choose_or_make_new = ChooseOrMakeNew(item_type)
        value_when_new = choose_or_make_new._item_widget.value.copy()

        # Make a camera
        saved = SavedSettings()
        item = choose_or_make_new._item_widget.model(**setting)
        saved.add_item(item)

        # Make a camera widget and select "Make new"
        choose_or_make_new = ChooseOrMakeNew(item_type)
        choose_or_make_new._choose_existing.value = "none"
        assert choose_or_make_new._item_widget.value == value_when_new

    def test_edit_requires_confirmation(self):
        # Should require confirmation if the item already exists
        self.make_test_camera()
        choose_or_make_new = ChooseOrMakeNew("camera")
        # the edit button should be displayed and the confirm widget should be hidden
        # note: display typically start as None or an empty string, so we just check
        # that it is not "none", which is what it will be set to when it is hidden.
        assert choose_or_make_new._edit_delete_container.layout.display != "none"
        assert choose_or_make_new._confirm_edit_delete.layout.display == "none"

        # Simulate a click on the edit button...
        choose_or_make_new._edit_button.click()
        # The edit button should now be hidden
        assert choose_or_make_new._edit_delete_container.layout.display == "none"

        # The savebuttonbar should be displayed
        assert choose_or_make_new._item_widget.savebuttonbar.layout.display != "none"

        # Click on the save button
        choose_or_make_new._item_widget.savebuttonbar.bn_save.click()

        # The confirm dialog should be displayed
        assert choose_or_make_new._confirm_edit_delete.layout.display != "none"

        # The confirmation dialog should contain the word "replace"
        assert "replace" in choose_or_make_new._confirm_edit_delete.message.lower()

    def test_edit_item_saved_after_confirm(self):
        # Should save the item after confirmation
        self.make_test_camera()

        choose_or_make_new = ChooseOrMakeNew("camera")
        # Simulate a click on the edit button...
        choose_or_make_new._edit_button.click()

        # Change a value in the camera so we can check that the new value is saved.
        new_gain = 2 * Quantity(TEST_CAMERA_VALUES["gain"])
        choose_or_make_new._item_widget.di_widgets["gain"].value = str(new_gain)
        # Simulate a click on the save button...
        choose_or_make_new._item_widget.savebuttonbar.bn_save.click()
        # Simulate a click on the confirm button...
        choose_or_make_new._confirm_edit_delete._yes.click()

        saved = SavedSettings()
        cameras = saved.get_items("camera")
        assert cameras.as_dict[TEST_CAMERA_VALUES["name"]].gain == new_gain

    def test_edit_item_not_saved_after_cancel(self):
        # Should not save the item after clicking the No button
        camera = self.make_test_camera()

        # Make an additional test camera so that the result is the *second* camera
        # in the list, not the first. The workflow being test is that there are
        # already two cameras, then the second one is selected (so that something is
        # selected) and then that second one is edited, but "no" is clicked on the
        # confirmation dialog.
        saved = SavedSettings()
        camera.name = "zzzz" + camera.name
        saved.add_item(camera)

        choose_or_make_new = ChooseOrMakeNew("camera")

        # There should be three choices in the dropdown -- the two cameras and
        # "Make new"
        assert len(choose_or_make_new._choose_existing.options) == 3

        # Choose the second camera
        choose_or_make_new._choose_existing.value = camera

        assert camera == choose_or_make_new.value

        # Simulate a click on the edit button...
        choose_or_make_new._edit_button.click()

        # Change a value in the camera so we can check that the new value is saved.
        new_gain = 2 * Quantity(TEST_CAMERA_VALUES["gain"])
        choose_or_make_new._item_widget.di_widgets["gain"].value = str(new_gain)
        # Simulate a click on the save button...
        choose_or_make_new._item_widget.savebuttonbar.bn_save.click()
        # Simulate a click on the cancel button...
        choose_or_make_new._confirm_edit_delete._no.click()
        assert camera == choose_or_make_new.value

    def test_selecting_make_new_as_selection_works(self):
        # Should allow the user to select "Make new" as a selection

        # Make a camera so that we can test that selecting "Make new" works
        self.make_test_camera()

        choose_or_make_new = ChooseOrMakeNew("camera")
        # This is the value for the "Make new...." option
        choose_or_make_new._choose_existing.value = "none"

        # edit button should go away
        assert choose_or_make_new._edit_delete_container.layout.display == "none"

        # Widget should be enabled
        assert choose_or_make_new._item_widget.disabled is False

        # save button bar should be displayed
        assert choose_or_make_new._item_widget.savebuttonbar.layout.display != "none"

    def test_choosing_different_item_updates_display(self):
        # Should update the display when a different item is chosen
        saved = SavedSettings()
        observatory = Observatory(**TEST_OBSERVATORY_SETTINGS)
        saved.add_item(observatory)

        observatory2 = Observatory(**TEST_OBSERVATORY_SETTINGS)
        # Make sure this name sorts lower than the first observatory
        observatory2.name = "zzzz" + observatory2.name
        observatory2.elevation = "-200 m"

        saved.add_item(observatory2)

        choose_or_make_new = ChooseOrMakeNew("observatory")

        # There should be three choices in the dropdown
        assert len(choose_or_make_new._choose_existing.options) == 3

        # Make sure the correct first observatory is selected
        assert choose_or_make_new._choose_existing.value == observatory
        assert Observatory(**choose_or_make_new._item_widget.value) == observatory

        # Select the other observatory
        choose_or_make_new._choose_existing.value = observatory2

        # The item widget should now have the values of the second observatory
        assert Observatory(**choose_or_make_new._item_widget.value) == observatory2

    def test_passband_map_buttons_are_disabled_or_enabled(self):
        # When an existing PassbandMap is selected the add/remove buttons
        # for individual rows should not be displayed.
        saved = SavedSettings()
        passband_map = PassbandMap(**TEST_PASSBAND_MAP)
        saved.add_item(passband_map)

        choose_or_make_new = ChooseOrMakeNew("passband_map")

        # There is no great way to get to the ItemBox widget that contains and controls
        # the add/remove buttons, so we keep going down through widget children until we
        # get to an ItemBox and then check that the buttons are disabled.
        # Recursion is the easiest way to do that, so recurse we will..
        def find_item_box(top_widget):
            for kid in top_widget.children:
                if isinstance(kid, ItemBox):
                    return kid
                if hasattr(kid, "children"):
                    result = find_item_box(kid)
                    if result:
                        return result

        item_box = find_item_box(choose_or_make_new)
        assert item_box.add_remove_controls == ItemControl.none

        # Next, we will click the "Edit" button and check that the buttons are enabled.
        choose_or_make_new._edit_button.click()
        item_box = find_item_box(choose_or_make_new)
        assert item_box.add_remove_controls == ItemControl.add_remove

    def test_make_passband_map(self):
        # Make a passband map and save it, then check that it is in the dropdown
        saved = SavedSettings()
        passband_map = PassbandMap(**TEST_PASSBAND_MAP)
        saved.add_item(passband_map)

        # Should create a new passband map
        choose_or_make_new = ChooseOrMakeNew("passband_map")
        assert len(choose_or_make_new._choose_existing.options) == 2
        assert choose_or_make_new._choose_existing.options[0][0] == passband_map.name

    def test_no_edit_button_when_there_are_no_items(self):
        # Should not have an edit button when there are no items
        saved = SavedSettings()
        # Make sure there are no cameras
        assert len(saved.get_items("camera").as_dict) == 0

        choose_or_make_new = ChooseOrMakeNew("camera")
        assert choose_or_make_new._edit_delete_container.layout.display == "none"

    def test_edit_button_returns_after_making_new_item(self):
        # After making a new item the edit button should be displayed

        # There are no cameras so this puts us into the form to make a new one
        choose_or_make_new = ChooseOrMakeNew("camera")

        # Make sure the edit button is hidden
        assert choose_or_make_new._edit_delete_container.layout.display == "none"

        # Change a value in the camera so we can check that the new value is saved.
        choose_or_make_new._item_widget.value = TEST_CAMERA_VALUES
        # Simulate a click on the save button...
        choose_or_make_new._item_widget.savebuttonbar.bn_save.click()

        # The edit button should now be displayed
        assert choose_or_make_new._edit_delete_container.layout.display != "none"

        # Select Make a new camera again
        choose_or_make_new._choose_existing.value = "none"

        # The edit button should now be hidden
        assert choose_or_make_new._edit_delete_container.layout.display == "none"

        # Select the camera we made
        choose_or_make_new._choose_existing.value = Camera(**TEST_CAMERA_VALUES)

        # The edit button should now be displayed
        assert choose_or_make_new._edit_delete_container.layout.display != "none"

    def test_save_button_disabled_when_no_changes(self):
        # Immediately after the edit button has been clicked, the save button should be
        # disabled because no changes have been made yet.
        #
        # That should remain true even after the user makes a change, then saves it
        # and then edits again but has not yet made any changes.
        self.make_test_camera()

        choose_or_make_new = ChooseOrMakeNew("camera")
        # Simulate a click on the edit button...
        choose_or_make_new._edit_button.click()

        # The save button should be disabled
        assert choose_or_make_new._item_widget.savebuttonbar.bn_save.disabled

        # Edit a value in the camera so the save button is enabled
        new_gain = 2 * Quantity(TEST_CAMERA_VALUES["gain"])
        choose_or_make_new._item_widget.di_widgets["gain"].value = str(new_gain)

        # The save button should now be enabled
        assert not choose_or_make_new._item_widget.savebuttonbar.bn_save.disabled

        # Save the change
        choose_or_make_new._item_widget.savebuttonbar.bn_save.click()

        # Confirm the save
        choose_or_make_new._confirm_edit_delete._yes.click()

        # Edit again...
        choose_or_make_new._edit_button.click()

        # The save button should be disabled
        assert choose_or_make_new._item_widget.savebuttonbar.bn_save.disabled

    def test_revert_button_is_enabled_after_clicking_edit(self):
        # The revert button should be enabled after clicking the edit button so
        # the user can cancel the edit.
        self.make_test_camera()

        choose_or_make_new = ChooseOrMakeNew("camera")
        # Simulate a click on the edit button...
        choose_or_make_new._edit_button.click()

        # The revert button should be enabled
        assert not choose_or_make_new._item_widget.savebuttonbar.bn_revert.disabled

    def test_clicking_revert_button_cancels_edit(self):
        # Clicking the revert button should cancel the edit
        self.make_test_camera()

        choose_or_make_new = ChooseOrMakeNew("camera")
        # Simulate a click on the edit button...
        choose_or_make_new._edit_button.click()

        # Simulate a click on the revert button...
        choose_or_make_new._item_widget.savebuttonbar.bn_revert.click()

        # The camera should not have been changed
        assert choose_or_make_new._item_widget.value == TEST_CAMERA_VALUES

        # The edit button should be displayed
        assert choose_or_make_new._edit_delete_container.layout.display != "none"

        # We should not be in editing mode anymore
        assert not choose_or_make_new._editing

    def test_revert_button_remains_enabled_with_invalid_value_and_actually_reverts(
        self,
    ):
        # The revert button should remain enabled if the value is invalid and reverting
        # should actually revert the value.
        self.make_test_camera()

        choose_or_make_new = ChooseOrMakeNew("camera")
        # Simulate a click on the edit button...
        choose_or_make_new._edit_button.click()

        # Set an invalid value
        choose_or_make_new._item_widget.di_widgets["name"].value = ""

        # unsaved_changes should be true
        assert choose_or_make_new._item_widget.savebuttonbar.unsaved_changes

        # Make sure the change is really there
        assert choose_or_make_new._item_widget.value != TEST_CAMERA_VALUES

        # The revert button should be enabled
        assert not choose_or_make_new._item_widget.savebuttonbar.bn_revert.disabled

        # Save should still be disabled
        assert choose_or_make_new._item_widget.savebuttonbar.bn_save.disabled

        # Click the revert button
        choose_or_make_new._item_widget.savebuttonbar.bn_revert.click()

        # The camera should not have been changed
        assert choose_or_make_new._item_widget.value == TEST_CAMERA_VALUES

    def test_delete_button_click_displays_confirm_dialog(self):
        # Clicking "Delete" should display a confirmation dialog

        # Make an item to ensure an existing item is displayed when the widget is made
        self.make_test_camera()

        choose_or_make_new = ChooseOrMakeNew("camera")
        choose_or_make_new._delete_button.click()

        # Confirm dialog should be displayed
        assert choose_or_make_new._confirm_edit_delete.layout.display != "none"
        # edit/delete buttons should be hidden
        assert choose_or_make_new._edit_delete_container.layout.display == "none"
        # The confirm dialog should indicate you are deleting
        assert "delete" in choose_or_make_new._confirm_edit_delete.message.lower()

    @pytest.mark.parametrize("click_yes", [True, False])
    def test_delete_actions_after_confirmation(self, click_yes):
        # Clicking "Yes" in the confirmation dialog should delete the item
        self.make_test_camera()

        choose_or_make_new = ChooseOrMakeNew("camera")
        choose_or_make_new._delete_button.click()

        if click_yes:
            # Simulate a click on the "Yes" button in the confirm dialog
            choose_or_make_new._confirm_edit_delete._yes.click()

            # The camera should be gone
            saved = SavedSettings()
            cameras = saved.get_items("camera")
            assert len(cameras.as_dict) == 0
        else:
            # Simulate a click on the "No" button in the confirm dialog
            choose_or_make_new._confirm_edit_delete._no.click()

            # The camera should still be there
            saved = SavedSettings()
            cameras = saved.get_items("camera")
            assert len(cameras.as_dict) == 1
            assert list(cameras.as_dict.values())[0].model_dump() == TEST_CAMERA_VALUES

        # Regardless of the choice, the confirm dialog should be hidden
        assert choose_or_make_new._confirm_edit_delete.layout.display == "none"

        # We should also no longer be deleting
        assert not choose_or_make_new._deleting

        # If the user clicked "No" the edit/delete buttons should be displayed because
        # there is still an item to edit or delete.
        if not click_yes:
            assert choose_or_make_new._edit_delete_container.layout.display != "none"
        # If the user clicked "Yes" the edit/delete buttons should be hidden because
        # there is no item to edit or delete.
        else:
            assert choose_or_make_new._edit_delete_container.layout.display == "none"

    def test_correct_item_selected_after_delete_and_yes(self):
        # The correct item should be selected after an item is deleted and the user
        # clicks "Yes" in the confirmation dialog.
        # Make two cameras
        self.make_test_camera()
        saved = SavedSettings()
        camera2 = Camera(**TEST_CAMERA_VALUES)
        camera2.name = "zzzz" + camera2.name
        saved.add_item(camera2)

        choose_or_make_new = ChooseOrMakeNew("camera")
        # Select the first camera...
        choose_or_make_new._choose_existing.value = saved.cameras.as_dict[
            TEST_CAMERA_VALUES["name"]
        ]
        # ...and delete it
        choose_or_make_new._delete_button.click()
        choose_or_make_new._confirm_edit_delete._yes.click()
        # Are the edit/delete buttons being shown?
        assert choose_or_make_new._edit_delete_container.layout.display != "none"
        # Is the correct camera selected?
        assert choose_or_make_new._choose_existing.value == camera2

    @pytest.mark.parametrize("click_yes", [True, False])
    def test_making_new_with_same_name_as_existing(self, click_yes):
        # Making a new item with the same name as an existing item should require
        # confirmation
        self.make_test_camera()

        choose_or_make_new = ChooseOrMakeNew("camera")

        # Click the "Make new" button
        choose_or_make_new._choose_existing.value = "none"

        # set the item widget value to the existing camera with double the gain
        camera_values = deepcopy(TEST_CAMERA_VALUES)
        camera_values["gain"] = 2 * Quantity(camera_values["gain"])
        choose_or_make_new._item_widget.value = Camera(**camera_values)

        # Simulate a click on the save button...
        choose_or_make_new._item_widget.savebuttonbar.bn_save.click()

        # The confirm dialog should be displayed
        assert choose_or_make_new._confirm_edit_delete.layout.display != "none"
        # The confirm dialog should contain the word "replace"
        assert "replace" in choose_or_make_new._confirm_edit_delete.message.lower()

        if click_yes:
            # ...click confirm
            choose_or_make_new._confirm_edit_delete._yes.click()
            expected_camera = Camera(**camera_values)
        else:
            # ...click cancel
            choose_or_make_new._confirm_edit_delete._no.click()
            expected_camera = Camera(**TEST_CAMERA_VALUES)

        # Check the value of the camera
        chosen_cam = Camera(**choose_or_make_new._item_widget.value)
        assert chosen_cam == choose_or_make_new._choose_existing.value

        assert chosen_cam == expected_camera

    def test_weird_sequence_of_no_clicks(self):
        # This is a regression test for #320
        # Make a camera
        self.make_test_camera()

        # Make a new camera...
        choose_or_make_new = ChooseOrMakeNew("camera")
        choose_or_make_new._choose_existing.value = "none"
        choose_or_make_new._item_widget.value = TEST_CAMERA_VALUES

        # ...click save...
        choose_or_make_new._item_widget.savebuttonbar.bn_save.click()
        # ...but say no to confirm
        choose_or_make_new._confirm_edit_delete._no.click()

        # Check that we are back to our selected camera
        assert choose_or_make_new._choose_existing.value == Camera(**TEST_CAMERA_VALUES)

        # Click delete...
        choose_or_make_new._delete_button.click()
        # ...but say no to confirm...
        choose_or_make_new._confirm_edit_delete._no.click()

        # ...the edit/delete buttons should be displayed
        assert choose_or_make_new._edit_delete_container.layout.display != "none"

    @pytest.mark.parametrize("hideable", [True, False])
    def test_details_hideable_or_not(self, hideable):
        # The details should be hideable if requested

        # Make a camera so that the details can be hidden
        self.make_test_camera()
        choose_or_make_new = ChooseOrMakeNew("camera", details_hideable=hideable)

        # New UI element should be in the big box
        assert (
            choose_or_make_new._show_details_ui
            in choose_or_make_new._choose_detail_container.children
        )

        if hideable:
            # The "show/hide details" widget should be visible
            assert choose_or_make_new._show_details_ui.layout.display != "none"
        else:
            # The "show/hide details" widget should not be visible
            assert choose_or_make_new._show_details_ui.layout.display == "none"

        # Set "show details" box to unchecked
        choose_or_make_new._show_details_ui.value = False

        # If hideable then details should not be displayed
        if hideable:
            assert choose_or_make_new._details_box.layout.display == "none"
        else:
            assert choose_or_make_new._details_box.layout.display != "none"

        # Settings show_details back to true should display details again,
        # regardless of hideable
        choose_or_make_new._show_details_ui.value = True
        assert choose_or_make_new._details_box.layout.display != "none"

    def test_details_hideable_plays_nicely_with_new_item(self):
        # The details should be hideable and should play nicely with making a new item
        # Make an item so that selecting "Make new" will count as a value
        # change later.
        self.make_test_camera()

        choose_or_make_new = ChooseOrMakeNew("camera", details_hideable=True)

        assert choose_or_make_new._details_box.layout.display != "none"

        # Set "show details" box to unchecked so details are hidden
        choose_or_make_new._show_details_ui.value = False

        # Check that details are hidden
        assert choose_or_make_new._details_box.layout.display == "none"

        # Select make a new camera
        choose_or_make_new._choose_existing.value = "none"

        # The details should be shown now
        assert choose_or_make_new._details_box.layout.display != "none"

        # The "show/hide details" widget should be hidden
        assert choose_or_make_new._show_details_ui.layout.display == "none"

        # Making a new camera and saving it should bring back, and respect,
        # the hideable details
        new_camera = deepcopy(TEST_CAMERA_VALUES)
        new_camera["name"] = "new camera"
        choose_or_make_new._item_widget.value = new_camera
        choose_or_make_new._item_widget.savebuttonbar.bn_save.click()

        # "show details" should be visible
        assert choose_or_make_new._show_details_ui.layout.display != "none"

        # Details should be hidden
        assert choose_or_make_new._details_box.layout.display == "none"
        assert not choose_or_make_new._show_details_ui.value

    def test_details_hideable_not_set_preserves_old_behavior(self):
        # The details should not be hidden if hideable is false
        # Make a camera
        self.make_test_camera()

        choose_or_make_new = ChooseOrMakeNew("camera", details_hideable=False)

        # New UI element should not be in the big box
        assert choose_or_make_new._show_details_ui.layout.display == "none"

        # Details should be displayed
        assert choose_or_make_new._details_box.layout.display != "none"

        # Set "show details" box to unchecked
        choose_or_make_new._show_details_ui.value = False

        # Details should still be displayed
        assert choose_or_make_new._details_box.layout.display != "none"

    @pytest.mark.parametrize("show_detail_state", [True, False])
    def test_details_hideable_cancel_making_new_restores_state(self, show_detail_state):
        # The details should be hideable and should play nicely with making a new item
        # Make an item so that selecting "Make new" will count as a value
        # change later.
        self.make_test_camera()

        choose_or_make_new = ChooseOrMakeNew("camera", details_hideable=True)

        assert choose_or_make_new._details_box.layout.display != "none"

        # Note whether the "show details" box is checked or not
        show_state = show_detail_state
        choose_or_make_new._show_details_ui.value = show_state

        # Select make a new camera
        choose_or_make_new._choose_existing.value = "none"

        # The details should be shown now
        assert choose_or_make_new._details_box.layout.display != "none"

        # The "show/hide details" widget should be hidden
        assert choose_or_make_new._show_details_ui.layout.display == "none"

        # Making a new camera and then cancelling should restore the state
        choose_or_make_new._item_widget.value = TEST_CAMERA_VALUES
        choose_or_make_new._item_widget.savebuttonbar.bn_save.click()
        choose_or_make_new._confirm_edit_delete._no.click()

        # "show details" should be visible
        assert choose_or_make_new._show_details_ui.layout.display != "none"

        # Details should match prior state
        assert choose_or_make_new._show_details_ui.value == show_state

    def test_chooser_has_value(self):
        # The chooser should have a value
        self.make_test_camera()

        choose_or_make_new = ChooseOrMakeNew("camera")
        assert choose_or_make_new.value == Camera(**TEST_CAMERA_VALUES)

    def test_details_can_be_hidden(self):
        # Make sure that item details can be hidden programmatically.
        self.make_test_camera()

        choose_or_make_new = ChooseOrMakeNew("camera", details_hideable=True)

        # Check that the details start out displayed
        assert choose_or_make_new._details_box.layout.display != "none"
        assert choose_or_make_new.display_details

        # Hide the details
        choose_or_make_new.display_details = False

        # Check that the details are hidden
        assert choose_or_make_new._details_box.layout.display == "none"

    def test_details_visibilty_cannot_be_changed_when_not_hideable(self):
        # Make sure that item details cannot be hidden programmatically when hideable
        # is False.
        self.make_test_camera()

        choose_or_make_new = ChooseOrMakeNew("camera", details_hideable=False)

        # Check that the details start out displayed
        assert choose_or_make_new._details_box.layout.display != "none"
        # Value should be None in this case
        assert choose_or_make_new.display_details is None

        # Hide the details
        choose_or_make_new.display_details = False
        # Value should still be None in this case
        assert choose_or_make_new.display_details is None

        # Check that the details are still displayed
        assert choose_or_make_new._details_box.layout.display != "none"


class TestConfirm:
    def test_initial_value(self):
        # Should initialize with value of None
        confirm = Confirm()
        assert confirm.value is None

    def test_initial_display(self):
        # Should initialize not displayed
        confirm = Confirm()
        assert confirm.layout.display == "none"

    def test_message(self):
        # Should initialize with a message
        confirm = Confirm(message="Hello")
        assert confirm.message == "Hello"
        # message should be settable
        confirm.message = "Goodbye"
        assert confirm.message == "Goodbye"

    def test_widget_structure(self):
        # Should have a message, yes, and no buttons
        confirm = Confirm()
        assert isinstance(confirm.children[0], ipw.HTML)
        assert isinstance(confirm.children[1], ipw.Button)
        assert isinstance(confirm.children[2], ipw.Button)

    @pytest.mark.parametrize("other_widget", [None, ipw.Dropdown()])
    def test_show(self, other_widget):
        # Should display when show is called
        confirm = Confirm(widget_to_hide=other_widget)
        if other_widget:
            assert other_widget.layout.display != "none"
        confirm.show()
        assert confirm.layout.display == "flex"
        if other_widget:
            assert other_widget.layout.display == "none"

    @pytest.mark.parametrize("yes_or_no", ["yes", "no"])
    @pytest.mark.parametrize("other_widget", [None, ipw.Dropdown()])
    def test_clicking_yes_or_no(self, yes_or_no, other_widget):
        # Should set value to True when yes is "clicked"
        confirm = Confirm(widget_to_hide=other_widget)
        # These are the handlers for a click on yes or no -- it takes an argument but we
        #  don't use it. The widget API requires that the function be able to take an
        # argument.
        if yes_or_no == "yes":
            confirm._yes.click()
        else:
            confirm._no.click()

        assert confirm.value == (yes_or_no == "yes")
        assert confirm.layout.display == "none"
        if other_widget:
            assert other_widget.layout.display != "none"


class TestSettingWithTitle:
    def test_title_format(self):
        # Check that the title is formatted with the requested heading
        # level and that there is, initially, no decoration.
        # Make a Camera with default testing settings
        camera = ui_generator(Camera)

        # Choosing something other than the default
        header_level = 5

        # Make a SettingWithTitle with the camera
        plain_title = "I am a camera"
        camera_title = SettingWithTitle(plain_title, camera, header_level=header_level)

        assert (
            camera_title.title.value
            == f"<h{header_level}>{plain_title}</h{header_level}>"
        )

    def test_title_decoration_plain_autoui(self):
        # Check that the title has the correct decoration given the
        # state of the settings widget.

        # Make a camera ui
        camera = ui_generator(Camera)
        plain_title = "I am a camera"

        # Make a SettingWithTitle with the camera
        camera_title = SettingWithTitle(plain_title, camera)

        # At the moment the title should not be decorated
        assert SaveStatus.SETTING_IS_SAVED not in camera_title.title.value
        assert SaveStatus.SETTING_NOT_SAVED not in camera_title.title.value

        # There should also be no unsaved changes at the moment
        assert not camera_title._autoui_widget.savebuttonbar.unsaved_changes

        # Manually set the value of the gain, which should make the trait
        # unsaved_changes True...
        camera_title._autoui_widget.di_widgets["gain"].value = str(
            2 * TEST_CAMERA_VALUES["gain"]
        )

        # ...and if unsaved_changes is True the "not saved" indicator should be present
        assert SaveStatus.SETTING_NOT_SAVED in camera_title.title.value

        # Click save...
        camera_title._autoui_widget.savebuttonbar.bn_save.click()

        # ... and check that the title is decorated with the "saved" indicator
        assert SaveStatus.SETTING_IS_SAVED in camera_title.title.value

        # Finally, click the save button and the title should be decorated with
        # the saved indication.
        camera_title._autoui_widget.savebuttonbar.bn_save.click()
        assert SaveStatus.SETTING_IS_SAVED in camera_title.title.value

    @pytest.mark.parametrize("accept_edits", [True, False])
    def test_title_decoration_choose_or_make_new_editing(self, accept_edits):
        # Check that the title has the correct decoration given the
        # state of the settings widget.

        # Make an item so there is something to select
        saved_setting = SavedSettings()
        saved_setting.add_item(Camera(**TEST_CAMERA_VALUES))

        camera = ChooseOrMakeNew(Camera.__name__)
        plain_title = camera._title.value

        # Make a SettingWithTitle with the camera
        camera_title = SettingWithTitle(plain_title, camera)

        # At the moment the title should not be decorated
        assert SaveStatus.SETTING_IS_SAVED not in camera_title.title.value
        assert SaveStatus.SETTING_NOT_SAVED not in camera_title.title.value

        # There should also be no unsaved changes at the moment
        assert not camera_title._autoui_widget.savebuttonbar.unsaved_changes

        # Click the edit button
        camera_title._widget._edit_button.click()

        # Edit a value in the camera so the save button is enabled
        new_gain = 2 * Quantity(TEST_CAMERA_VALUES["gain"])
        camera_title._widget._item_widget.di_widgets["gain"].value = str(new_gain)

        assert SaveStatus.SETTING_NOT_SAVED in camera_title.title.value

        # Click the save button....
        camera_title._widget._item_widget.savebuttonbar.bn_save.click()

        # ...should still be unsaved pending confirmation...
        assert SaveStatus.SETTING_NOT_SAVED in camera_title.title.value

        # ...now confirm or deny the save...
        if accept_edits:
            camera_title._widget._confirm_edit_delete._yes.click()
        else:
            camera_title._widget._confirm_edit_delete._no.click()

        # ...and the title should be decorated with the saved indication.
        assert SaveStatus.SETTING_IS_SAVED in camera_title.title.value


SETTING_CLASSES = [
    Camera,
    Observatory,
    PassbandMap,
    PhotometryApertures,
    PhotometryOptionalSettings,
    SourceLocationSettings,
    LoggingSettings,
]


def _to_space(name: str) -> str:
    return name.replace("_", " ")


class TestReviewSettings:
    """
    Test of the magical ReviewSettings widget.
    """

    # Auto-use the shared change_to_tmp_dir fixture (defined in the top-level
    # conftest) so every test in this class runs in a temporary directory.
    @pytest.fixture(autouse=True)
    def _change_to_tmp_dir(self, change_to_tmp_dir):
        pass

    @pytest.mark.parametrize("container_type", ["tabs", "accordion"])
    def test_creation_no_saved_settings(self, container_type):
        # Check creation and names of tab when there are no saved settings
        # and just one type of setting.
        for setting_class in SETTING_CLASSES:
            wd_settings = PhotometryWorkingDirSettings()

            # Remove any existing settings files that may have been saved in earlier
            # iterations of the loop.
            p = Path(".")
            (p / wd_settings.partial_settings_file).unlink(missing_ok=True)
            (p / wd_settings.settings_file).unlink(missing_ok=True)
            review_settings = ReviewSettings([setting_class], style=container_type)

            assert (
                _to_space(to_snake(setting_class.__name__))
                in review_settings._container.titles[0]
            )
            if container_type == "tabs":
                assert isinstance(review_settings._container, ipw.Tab)
            else:
                assert isinstance(review_settings._container, ipw.Accordion)

            # What happens next depends on whether the setting can be created from
            # default values or not.
            try:
                setting_class()
            except ValidationError:
                created_from_defaults = False
            else:
                created_from_defaults = True

            if created_from_defaults:
                # Test whether the setting, which was able to be created from default
                # values of the fields, is in the partial_settings.
                wd_settings.load()
                snake_name = to_snake(setting_class.__name__)
                assert (
                    getattr(wd_settings.partial_settings, snake_name) == setting_class()
                )
            else:
                with pytest.raises(
                    ValueError,
                    match="Settings file photometry_settings.json does not exist",
                ):
                    wd_settings.load()

    def test_creation_with_saved_settings(self):
        # Check creation and names of tab when there are saved settings
        # and just one type of setting.

        saveable_types = ChooseOrMakeNew._known_types
        for setting_class in SETTING_CLASSES:
            if setting_class.__name__ not in saveable_types:
                # The rest of this only makes sense if the setting class is saveable
                continue

            # Make a setting
            saved = SavedSettings()
            # Make an instance of the class
            snake_name = to_snake(setting_class.__name__)
            item = setting_class.model_validate(TEST_PHOTOMETRY_SETTINGS[snake_name])
            # Save the instance to saved settings file
            saved.add_item(item)

            # Make the review widget. It should auto-populate with the item with just
            # created because that is the only item of that type.
            review_settings = ReviewSettings([setting_class])
            model_dict = review_settings._container.children[0]._autoui_widget.value
            assert setting_class(**model_dict) == item

            # This setting was made automatically by the widget, so the title of the
            # container should end with SaveStatus.SETTING_SHOULD_BE_REVIEWED
            assert (
                SaveStatus.SETTING_SHOULD_BE_REVIEWED
                in review_settings._container.titles[0]
            )

            # The item setting should also have been saved to the working directory
            wd_settings = PhotometryWorkingDirSettings()
            loaded_settings = wd_settings.load()
            assert getattr(loaded_settings, snake_name) == item

    def test_creation_with_saved_working_dir_settings(self):
        # Check that when there is a saved item in the working directory
        # the badge attached to the name is "needs review"
        photometry_apertures = TEST_PHOTOMETRY_SETTINGS["photometry_apertures"]
        wk_dir = PhotometryWorkingDirSettings()
        partial_settings = PartialPhotometrySettings(
            photometry_apertures=photometry_apertures
        )
        wk_dir.save(partial_settings)

        # Create the review widget
        review_settings = ReviewSettings(
            [PhotometryOptionalSettings, PhotometryApertures]
        )

        assert (
            SaveStatus.SETTING_SHOULD_BE_REVIEWED
            in review_settings._container.titles[1]
        )

    def test_selecting_tab_updates_title(self):
        # Check that selecting a tab updates the title of the widget to end with the
        # appropriate indicator.
        # Save a camera and an observatory
        saved = SavedSettings()
        camera = Camera(**TEST_CAMERA_VALUES)
        saved.add_item(camera)
        observatory = Observatory(**TEST_OBSERVATORY_SETTINGS)
        saved.add_item(observatory)

        # Create the review widget
        review_settings = ReviewSettings([Camera, Observatory])

        # Select the Observatory tab
        review_settings._container.selected_index = 1

        assert SaveStatus.SETTING_IS_SAVED in review_settings._container.titles[1]

    def test_conflict_between_saved_and_working_dir(self):
        # Check that the expected error is raised if the value of a saved setting
        # like a camera conflicts with the value of the same setting in the working
        # directory.

        # Make a camera for the saved settings (i.e. those in user's settings directory)
        saved = SavedSettings()
        camera = Camera(**TEST_CAMERA_VALUES)
        saved.add_item(camera)

        # Make a camera for the working directory with different gain than the first
        # camera
        wd_settings = PhotometryWorkingDirSettings()
        wd_camera = Camera(**TEST_CAMERA_VALUES)
        wd_camera.gain = 2 * wd_camera.gain
        wd_settings.save(PartialPhotometrySettings(camera=wd_camera))

        # Create the review widget and check that the error is raised
        with pytest.raises(
            ValueError, match="The camera setting saved in the working directory is not"
        ):
            ReviewSettings([Camera])

    def test_error_when_setting_has_no_saved_or_default_setting(self):
        # Make a review widget with a setting that has no saved or default setting
        # like Camera and check that an error is raised.

        review_settings = ReviewSettings([Camera])

        assert SaveStatus.SETTING_NOT_SAVED in review_settings._container.titles[0]

    def test_setting_selected_item_to_none_does_not_save(self):
        # The intent of this is to test an edge case where the user selects an item
        # from the dropdown and then selects "None" from the dropdown. The selected
        # item should be saved, but there should not be another save when None is
        # selected.
        saved = SavedSettings()
        camera = Camera(**TEST_CAMERA_VALUES)
        saved.add_item(camera)
        wd_settings = PhotometryWorkingDirSettings()
        wd_set_path = Path(wd_settings.partial_settings_file)
        review_settings = ReviewSettings([Camera])

        # Get the modification time of the working directory settings file
        mtime = wd_set_path.stat().st_mtime

        assert (
            review_settings._setting_widgets[0]._widget._choose_existing.value == camera
        )
        review_settings._setting_widgets[0]._widget._choose_existing.value = "none"

        # Check that the working directory settings file has not been modified
        assert wd_set_path.stat().st_mtime == mtime

    def test_clicking_tab_with_already_saved_settings_updates_badge(self):
        # Check that if there are already settings saved to the working directory
        # and the user clicks on the tab, the badge is updated to reflect that the
        # setting has been saved.

        # To set this up we need to save settings to the working directory AND to the
        # saved user settings.
        wd_settings = PhotometryWorkingDirSettings()
        camera = Camera(**TEST_CAMERA_VALUES)
        observatory = Observatory(**TEST_OBSERVATORY_SETTINGS)
        passbands = PassbandMap(**TEST_PASSBAND_MAP)
        wd_settings.save(
            PartialPhotometrySettings(
                camera=camera, observatory=observatory, passband_map=passbands
            )
        )
        saved = SavedSettings()
        saved.add_item(camera)
        saved.add_item(observatory)
        saved.add_item(passbands)

        # Make the review widget
        review_settings = ReviewSettings([Camera, Observatory, PassbandMap])
        for title in review_settings._container.titles:
            assert SaveStatus.SETTING_SHOULD_BE_REVIEWED in title

        # Click on each tab, starting with the last one, to make sure a change in
        # the selected value happens to trigger the observers.
        for i in range(2, -1, -1):
            review_settings._container.selected_index = i

        for title in review_settings._container.titles:
            assert SaveStatus.SETTING_IS_SAVED in title

    def test_clicking_tab_marks_green_when_all_saved(self):
        # Check that if there are already settings saved to the working directory
        # and the user clicks on the tab, the badge is updated to reflect that the
        # setting has been saved.

        # To set this up we need to save settings to the working directory AND to the
        # saved user settings.
        wd_settings = PhotometryWorkingDirSettings()
        camera = Camera(**TEST_CAMERA_VALUES)
        observatory = Observatory(**TEST_OBSERVATORY_SETTINGS)
        passbands = PassbandMap(**TEST_PASSBAND_MAP)
        wd_settings.save(
            PartialPhotometrySettings(
                camera=camera, observatory=observatory, passband_map=passbands
            )
        )
        saved = SavedSettings()
        saved.add_item(camera)
        saved.add_item(observatory)
        saved.add_item(passbands)

        review_settings = ReviewSettings(
            [
                Camera,
                Observatory,
                PassbandMap,
                PhotometryApertures,
                SourceLocationSettings,
                PhotometryOptionalSettings,
                LoggingSettings,
            ]
        )

        num_tabs = len(review_settings._container.children)

        for title in review_settings._container.titles:
            assert SaveStatus.SETTING_SHOULD_BE_REVIEWED in title

        # Click on each tab, starting with the last one, to make sure a change in
        # the selected value happens to trigger the observers.
        for i in range(num_tabs - 1, -1, -1):
            review_settings._container.selected_index = i

        for title in review_settings._container.titles:
            assert SaveStatus.SETTING_IS_SAVED in title

    def test_loading_saved_source_locations_to_ui(self):
        # Check that saved source locations are loaded into the UI
        wd_set = PhotometryWorkingDirSettings()
        source_locations = SourceLocationSettings()
        wd_set.save(
            PartialPhotometrySettings(source_location_settings=source_locations)
        )

        review_settings = ReviewSettings([SourceLocationSettings])

        # Check that the saved source locations are in the UI
        assert (
            review_settings._setting_widgets[0]._autoui_widget.value
            == source_locations.model_dump()
        )
        assert (
            review_settings._setting_widgets[0]
            ._autoui_widget.di_widgets["source_list_file"]
            .selected_filename
            == source_locations.source_list_file
        )

    def test_getting_settings_with_nothing_saved(self):
        # Check that when we create the object with no saved settings we get an empty
        # partial settings object.
        review_settings = ReviewSettings([Camera])
        assert review_settings.current_settings == PartialPhotometrySettings()

    def test_selecting_table_without_saved_setting_sets_proper_badge(self):
        # Test that selecting a tab with a saved setting sets a proper "NOT SAVED" badge
        review_settings = ReviewSettings([PhotometryApertures, Camera])
        review_settings._container.selected_index = 1
        assert SaveStatus.SETTING_NOT_SAVED in review_settings._container.titles[1]

    def test_banner_hidden_when_no_settings_files(self):
        # With no saved settings there is nothing to report, so the banner
        # is empty and hidden.
        review_settings = ReviewSettings([Camera])
        assert review_settings._banner_html.value == ""
        assert review_settings._banner.layout.display == "none"

    def test_banner_reports_unreadable_settings_file(self):
        # A settings file in the working directory that exists but cannot be
        # read used to be silently ignored. The widget should still come up,
        # starting from defaults, with a banner explaining what happened, and
        # the unreadable file should be left in place.
        wd_settings = PhotometryWorkingDirSettings()
        bad_content = '{"pasta": "carbonara"}'
        wd_settings.settings_file.write_text(bad_content)

        # PhotometryApertures can be created from default values, so this also
        # exercises the automatic save that happens during widget creation.
        review_settings = ReviewSettings([Camera, PhotometryApertures])

        banner_text = review_settings._banner_html.value
        assert "There is a problem with the saved settings" in banner_text
        assert review_settings._banner.layout.display == "flex"
        assert wd_settings.settings_file.read_text() == bad_content

        # The message names the unreadable file and points out that a save
        # will not overwrite it, and that a save that needs to replace it
        # will first preserve it with a .bak extension.
        assert "photometry_settings.json.bak" in banner_text
        assert "will not be overwritten" in banner_text

        # Only the first line of the error is in the visible message; the
        # rest of the validation error is inside a collapsed details element,
        # formatted in a pre block that preserves pydantic's line breaks.
        visible_text = banner_text.split("<details>")[0]
        assert "validation error" in visible_text
        assert "Field required" not in visible_text
        assert "<details><summary>Full error</summary><pre>" in banner_text
        assert "Field required" in banner_text.split("<details>")[1]

    def test_banner_reports_both_files_unreadable(self):
        # When both settings files exist but neither can be read, the
        # banner names both files and both .bak names in one message.
        wd_settings = PhotometryWorkingDirSettings()
        wd_settings.settings_file.write_text('{"pasta": "carbonara"}')
        wd_settings.partial_settings_file.write_text('{"pasta": "rigatoni"}')

        review_settings = ReviewSettings([Camera])

        banner_text = review_settings._banner_html.value
        assert review_settings._banner.layout.display == "flex"
        assert "None of the values below" in banner_text
        assert (
            "photometry_settings.json and partial_photometry_settings.json"
            in banner_text
        )
        assert (
            "photometry_settings.json.bak and partial_photometry_settings.json.bak"
            in banner_text
        )
        assert "will not be overwritten" in banner_text

    def test_banner_single_message_when_error_identity_changes(self):
        # With both files unreadable, the first refresh raises an error
        # whose first line comes from the partial file; the construction
        # autosave (PhotometryApertures saves its defaults) renames the bad
        # partial file to .bak, so later refreshes raise the full-file
        # error instead. That is still the same incident, so the banner
        # must show exactly one problem message, updated in place, not one
        # per error wording.
        wd_settings = PhotometryWorkingDirSettings()
        wd_settings.settings_file.write_text('{"pasta": "carbonara"}')
        wd_settings.partial_settings_file.write_text('{"pasta": "rigatoni"}')

        review_settings = ReviewSettings([Camera, PhotometryApertures])
        banner_text = review_settings._banner_html.value
        assert banner_text.count("There is a problem with the saved settings") == 1

        # A tab selection triggers another refresh; still one message.
        review_settings._container.selected_index = 1
        banner_text = review_settings._banner_html.value
        assert banner_text.count("There is a problem with the saved settings") == 1

    def test_banner_dismiss_button_stays_dismissed(self):
        # Clicking the dismiss button hides the banner, and the message does
        # not come back when the settings are reloaded (which happens on
        # every tab selection). The automatic save of PhotometryApertures
        # during construction writes a readable partial settings file, which
        # changes the wording composed around the error on later refreshes;
        # dismissal must stick anyway because it is keyed on the load
        # error's fixed identity plus a fingerprint of its full text, and
        # that text stays stable across these refreshes even though the
        # wording composed around it changes.
        wd_settings = PhotometryWorkingDirSettings()
        wd_settings.settings_file.write_text('{"pasta": "carbonara"}')

        review_settings = ReviewSettings([Camera, PhotometryApertures])
        assert review_settings._banner.layout.display == "flex"

        review_settings._banner_dismiss.click()
        assert review_settings._banner.layout.display == "none"
        assert review_settings._banner_html.value == ""

        # Selecting a different tab reloads the settings from disk; the
        # dismissed message should stay dismissed.
        review_settings._container.selected_index = 1
        assert review_settings._banner.layout.display == "none"

    def test_banner_message_stays_after_underlying_problem_resolves(self):
        # A banner message must not silently disappear once the user has
        # had a chance to see it, even if a later refresh no longer
        # reproduces the underlying problem. This matters because the
        # automatic save that happens during widget construction (for a
        # setting like PhotometryApertures that can be built from defaults)
        # can itself cure the problem -- e.g. by renaming an unreadable
        # partial settings file to .bak -- before the user has dismissed
        # anything.
        wd_settings = PhotometryWorkingDirSettings()
        wd_settings.partial_settings_file.write_text('{"pasta": "carbonara"}')

        # The automatic save for PhotometryApertures fires during
        # construction and renames the bad partial settings file to .bak,
        # so the refresh at the end of construction already finds the
        # problem cured -- the message is shown, in its resolved wording.
        review_settings = ReviewSettings([Camera, PhotometryApertures])
        assert review_settings._banner.layout.display == "flex"
        banner_text = review_settings._banner_html.value
        assert "A problem with the saved settings" in banner_text

        # Selecting the PhotometryApertures tab triggers another refresh.
        # The load succeeds now, but the message should still be shown
        # because the user never dismissed it -- now in its past-tense
        # resolved wording instead of asserting the (no longer true)
        # original claim or making a future-tense .bak promise about a
        # rename that already happened.
        review_settings._container.selected_index = 1
        assert review_settings._banner.layout.display == "flex"
        banner_text = review_settings._banner_html.value
        assert "A problem with the saved settings" in banner_text
        assert "no longer detected as of the latest reload" in banner_text
        assert "There is a problem" not in banner_text
        assert "will not be overwritten" not in banner_text
        assert "was preserved as" in banner_text
        assert "partial_photometry_settings.json.bak" in banner_text

        # Dismissing hides it...
        review_settings._banner_dismiss.click()
        assert review_settings._banner.layout.display == "none"
        assert review_settings._banner_html.value == ""

        # ... and it stays hidden across further refreshes.
        review_settings._container.selected_index = 0
        assert review_settings._banner.layout.display == "none"

    def test_dismissed_messages_pruned_once_no_longer_produced(self):
        # The set of dismissed messages should not grow without bound: once
        # a dismissed message is no longer produced by a refresh, it should
        # be dropped so that an unrelated, differently-worded problem later
        # is shown rather than being mistaken for something dismissed.
        wd_settings = PhotometryWorkingDirSettings()
        wd_settings.partial_settings_file.write_text('{"pasta": "carbonara"}')

        review_settings = ReviewSettings([Camera, PhotometryApertures])
        assert review_settings._banner.layout.display == "flex"

        review_settings._banner_dismiss.click()
        assert review_settings._dismissed

        # The bad partial settings file was renamed to .bak during
        # construction, so this refresh succeeds and no longer produces the
        # dismissed message; it should be pruned from the dismissed set.
        review_settings._container.selected_index = 1
        assert review_settings._dismissed == {}

    def test_dismissed_load_error_reappears_for_new_error(self):
        # Dismissing a load error must not silently suppress a later, freshly
        # arisen load error that happens to reuse the same fixed banner key.
        wd_settings = PhotometryWorkingDirSettings()
        wd_settings.settings_file.write_text('{"pasta": "carbonara"}')

        review = ReviewSettings([Camera])

        review._banner_dismiss.click()
        assert review._banner.layout.display == "none"

        # A second, previously fine file now also becomes corrupt. This
        # changes the error's first line from "Error loading settings: ..."
        # to "Error loading partial settings: ...", i.e. a genuinely new
        # incident under the same _LOAD_ERROR_KEY.
        partial_settings_file = wd_settings.partial_settings_file
        partial_settings_file.write_text('{"pasta": "rigatoni"}')
        review._refresh()

        assert review._banner.layout.display == "flex"
        assert "There is a problem with the saved settings" in review._banner_html.value

    def test_dismissed_load_error_reappears_when_error_detail_changes(self):
        # Two different corruptions of the same file can share the same
        # generic first line ("Error loading settings: N validation errors
        # for ..."); a dismissal of one must not suppress the other, so the
        # fingerprint is the full error text.
        wd_settings = PhotometryWorkingDirSettings()
        wd_settings.settings_file.write_text('{"camera": 12}')

        review = ReviewSettings([Camera])
        review._banner_dismiss.click()
        assert review._banner.layout.display == "none"

        # A different corruption with the same number of validation errors,
        # so the first line of the error is unchanged but the details are
        # not. Verify the premise so this test fails loudly if pydantic's
        # error formatting changes.
        wd_settings.settings_file.write_text('{"observatory": 17}')
        with pytest.raises(ValueError) as exc_info:
            PhotometryWorkingDirSettings().load()
        assert "validation error" in str(exc_info.value).splitlines()[0]

        review._refresh()
        assert review._banner.layout.display == "flex"

    def test_tab_selection_refreshes_banner_even_with_positive_badge(self):
        # A problem that arises after construction must surface in the
        # banner on the next tab selection even when the selected tab's
        # badge is positive, i.e. when no badge re-derivation from disk
        # would otherwise happen.
        wd_settings = PhotometryWorkingDirSettings()
        review = ReviewSettings([Camera, PhotometryApertures])
        assert review._banner.layout.display == "none"

        review._setting_widgets[1].badge = SaveStatus.SETTING_IS_SAVED
        wd_settings.settings_file.write_text('{"pasta": "carbonara"}')

        review._container.selected_index = 1
        assert review._banner.layout.display == "flex"

    def test_banner_marks_resolved_problem(self):
        # When an autosave triggered by a tab selection cures the underlying
        # problem, the sticky message stays but gets a "resolved" marker
        # rather than continuing to assert the (no longer true) claim.
        wd_settings = PhotometryWorkingDirSettings()
        wd_settings.partial_settings_file.write_text('{"pasta": "carbonara"}')

        # The PhotometryApertures autosave during construction renames the
        # bad partial settings file to .bak, curing the problem.
        review = ReviewSettings([Camera, PhotometryApertures])

        # Selecting the PhotometryApertures tab triggers another refresh.
        review._container.selected_index = 1

        assert review._banner.layout.display == "flex"
        assert "no longer detected as of the latest reload" in review._banner_html.value

    def test_banner_message_updates_instead_of_duplicating(self):
        # The wording composed around a load error can change between
        # refreshes for the same underlying problem: here the corrupt full
        # settings file first has no readable companion ("None of the values
        # below..."), then the automatic save of PhotometryApertures during
        # construction writes a readable partial settings file, so later
        # refreshes report the partial file as the fallback. The banner must
        # show exactly one message, updated in place to the latest wording,
        # not one near-duplicate per wording.
        wd_settings = PhotometryWorkingDirSettings()
        wd_settings.settings_file.write_text('{"pasta": "carbonara"}')

        review_settings = ReviewSettings([Camera, PhotometryApertures])

        banner_text = review_settings._banner_html.value
        assert banner_text.count("There is a problem with the saved settings") == 1

        # Selecting a tab triggers another refresh; still just one message.
        review_settings._container.selected_index = 1
        banner_text = review_settings._banner_html.value
        assert banner_text.count("There is a problem with the saved settings") == 1

        # The message reflects the latest wording: the autosaved partial
        # settings file is now the readable fallback, and the original
        # "no readable settings" sentence has been replaced in place.
        assert "come from the partial settings file" in banner_text
        assert "None of the values below" not in banner_text

    def test_current_settings_fresh_after_construction(self):
        # Widget construction itself saves settings (PhotometryApertures can
        # be built from defaults and is autosaved), so the snapshot taken at
        # the top of construction goes stale before the constructor returns.
        # The refresh at the end of construction must pick up those saves.
        review_settings = ReviewSettings([Camera, PhotometryApertures])
        assert review_settings.current_settings.photometry_apertures is not None

    def test_banner_uses_readable_settings_as_fallback(self):
        # If the partial settings file cannot be read but the full settings
        # file is fine, the widget should still be built from the readable
        # full settings rather than falling all the way back to empty
        # settings, and the banner should say so.
        saved = SavedSettings()
        saved.add_item(Camera(**TEST_CAMERA_VALUES))

        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        wd_settings = PhotometryWorkingDirSettings()
        wd_settings.save(full_settings)
        wd_settings.partial_settings_file.write_text('{"pasta": "carbonara"}')

        review_settings = ReviewSettings([Camera])

        # The Camera save observer fires during construction and sets the
        # unreadable partial file aside, curing the problem before
        # construction finishes. Re-corrupt the file and refresh so the
        # active wording -- the point of this test -- can be observed.
        wd_settings.partial_settings_file.write_text('{"pasta": "carbonara"}')
        review_settings._refresh()

        banner_text = review_settings._banner_html.value
        assert review_settings._banner.layout.display == "flex"
        assert "come from the full settings file" in banner_text
        assert review_settings.current_settings.camera == full_settings.camera

    def test_banner_conflict_wording_when_both_files_readable(self):
        # When both the full and partial settings files are individually
        # readable but disagree with each other, the "this file could not
        # be read" wording would be misleading -- both files were read
        # just fine. The banner should instead say that the values shown
        # come from the (preferred) full settings file and that saving full
        # settings resolves the conflict, keeping the conflicting (losing)
        # partial settings file as a .bak.
        saved = SavedSettings()
        saved.add_item(Camera(**TEST_CAMERA_VALUES))

        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        wd_settings = PhotometryWorkingDirSettings()
        wd_settings.settings_file.write_text(full_settings.model_dump_json())

        conflicting_camera = Camera(**TEST_CAMERA_VALUES)
        conflicting_camera.gain = 2 * conflicting_camera.gain
        wd_settings.partial_settings_file.write_text(
            PartialPhotometrySettings(camera=conflicting_camera).model_dump_json()
        )

        review_settings = ReviewSettings([Camera])

        # The Camera save observer fires during construction and resolves
        # the conflict, so recreate it and refresh to observe the active
        # conflict wording -- the point of this test.
        wd_settings.partial_settings_file.write_text(
            PartialPhotometrySettings(camera=conflicting_camera).model_dump_json()
        )
        review_settings._refresh()

        banner_text = review_settings._banner_html.value
        assert review_settings._banner.layout.display == "flex"
        assert "will not be overwritten" not in banner_text
        assert "come from the full settings file" in banner_text
        assert "resolve the conflict" in banner_text
        assert "conflicting partial settings file will be preserved" in banner_text
        assert "partial_photometry_settings.json.bak" in banner_text

    def test_construction_autosave_preserves_conflicting_partial(self):
        # The save observers fire during construction with values taken
        # from the full settings file (the fallback when the two settings
        # files conflict), so the save that resolves the conflict is just
        # echoing the full file's own camera back. It must still preserve
        # the losing partial settings file -- whose camera value exists
        # nowhere else -- as .bak, as the banner promises; it used to be
        # deleted outright because the echoed camera counted as an
        # explicit replacement.
        saved = SavedSettings()
        saved.add_item(Camera(**TEST_CAMERA_VALUES))

        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        wd_settings = PhotometryWorkingDirSettings()
        wd_settings.settings_file.write_text(full_settings.model_dump_json())

        conflicting_camera = Camera(**TEST_CAMERA_VALUES)
        conflicting_camera.gain = 2 * conflicting_camera.gain
        partial_content = PartialPhotometrySettings(
            camera=conflicting_camera
        ).model_dump_json()
        wd_settings.partial_settings_file.write_text(partial_content)

        ReviewSettings([Camera])

        assert not wd_settings.partial_settings_file.exists()
        backup = wd_settings.partial_settings_file.with_name(
            wd_settings.partial_settings_file.name + ".bak"
        )
        assert backup.read_text() == partial_content

    def test_banner_handles_invalid_utf8_settings_file(self):
        # Bytes that are not valid UTF-8 raise a UnicodeDecodeError while
        # reading the settings file; that must be handled the same as any
        # other unreadable settings file rather than propagating out of
        # widget construction.
        wd_settings = PhotometryWorkingDirSettings()
        wd_settings.settings_file.write_bytes(b"\xff\xfe not utf8")

        review_settings = ReviewSettings([Camera])

        assert review_settings._banner.layout.display == "flex"

    def test_banner_shows_warnings_from_loading_settings(self, mocker):
        # Warnings generated while loading settings (e.g. a settings-format
        # migration) should be displayed in the banner.
        def fake_load(_self):
            warnings.warn(
                "Settings were migrated", PhotometrySettingsWarning, stacklevel=2
            )
            return PartialPhotometrySettings()

        mocker.patch.object(
            settings_files.PhotometryWorkingDirSettings, "load", fake_load
        )
        review_settings = ReviewSettings([Camera])

        assert "Settings were migrated" in review_settings._banner_html.value
        assert review_settings._banner.layout.display == "flex"

    def test_banner_escapes_html_from_settings_file(self):
        # The banner message includes content that came from the settings
        # file, so markup in a corrupt file must be escaped rather than
        # rendered as HTML.
        wd_settings = PhotometryWorkingDirSettings()
        wd_settings.settings_file.write_text(
            '{"camera": "<img src=x onerror=alert(1)>"}'
        )

        review_settings = ReviewSettings([Camera])

        banner_text = review_settings._banner_html.value
        assert "<img" not in banner_text
        assert "&lt;img" in banner_text

    def test_banner_shows_warnings_when_load_also_fails(self, mocker):
        # Warnings emitted while loading settings should reach the banner
        # even when the load then fails, alongside the error message itself.
        def fake_load(_self):
            warnings.warn(
                "Settings were migrated", PhotometrySettingsWarning, stacklevel=2
            )
            raise ValueError("Error loading settings: broken")

        mocker.patch.object(
            settings_files.PhotometryWorkingDirSettings, "load", fake_load
        )
        # The error is only reported when a settings file actually exists.
        PhotometryWorkingDirSettings().settings_file.write_text("{}")

        review_settings = ReviewSettings([Camera])

        banner_text = review_settings._banner_html.value
        assert "Error loading settings: broken" in banner_text
        assert "Settings were migrated" in banner_text
        assert review_settings._banner.layout.display == "flex"

    @pytest.mark.parametrize("category", [DeprecationWarning, UserWarning])
    def test_banner_ignores_other_warning_categories(self, mocker, category):
        # Only PhotometrySettingsWarning is user-actionable; anything else
        # raised on the load path -- a DeprecationWarning from a library
        # internal, or a plain UserWarning such as a pydantic serializer
        # warning or astropy's AstropyUserWarning -- should stay out of the
        # banner.
        def fake_load(_self):
            warnings.warn("some library warning", category, stacklevel=2)
            return PartialPhotometrySettings()

        mocker.patch.object(
            settings_files.PhotometryWorkingDirSettings, "load", fake_load
        )
        review_settings = ReviewSettings([Camera])

        assert review_settings._banner_html.value == ""
        assert review_settings._banner.layout.display == "none"

    def test_unreadable_settings_file_preserved_as_bak(self):
        # An unreadable partial settings file in the working directory should
        # be reported in the banner, and the automatic save that happens
        # during widget creation should preserve the original content with a
        # .bak extension instead of overwriting it.
        wd_settings = PhotometryWorkingDirSettings()
        bad_content = '{"pasta": "carbonara"}'
        wd_settings.partial_settings_file.write_text(bad_content)

        # PhotometryApertures can be created from default values, so the
        # automatic save fires during widget creation.
        review_settings = ReviewSettings([Camera, PhotometryApertures])

        banner_text = review_settings._banner_html.value
        # The autosave cures the problem during construction, so the message
        # is already in its past-tense resolved wording by this point.
        assert "problem with the saved settings" in banner_text
        backup = wd_settings.partial_settings_file.with_name(
            wd_settings.partial_settings_file.name + ".bak"
        )
        assert backup.read_text() == bad_content

    def test_autosave_does_not_print_settings_warnings(self, mocker):
        # A settings-migration warning raised while loading should reach the
        # banner exactly once. Without the fix, it would also be printed to
        # the console once per autosaved setting: the widget's own explicit
        # loads (which feed the banner) route the warning through their own
        # recording context, but the autosave triggered during construction
        # calls save(), whose internal bookkeeping load deliberately lets
        # PhotometrySettingsWarning through.
        def fake_load(_self):
            warnings.warn(
                "Settings were migrated", PhotometrySettingsWarning, stacklevel=2
            )
            return PartialPhotometrySettings()

        mocker.patch.object(
            settings_files.PhotometryWorkingDirSettings, "load", fake_load
        )

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            # PhotometryApertures can be built from defaults, so it is
            # autosaved during construction.
            review_settings = ReviewSettings([PhotometryApertures])

        assert "Settings were migrated" in review_settings._banner_html.value
        assert not any(
            issubclass(w.category, PhotometrySettingsWarning) for w in recorded
        )

    def test_autosave_reemits_non_settings_warnings(self, mocker):
        # The autosave suppression's recording context swallows every
        # warning raised during the save. Anything that is not a
        # PhotometrySettingsWarning (e.g. a serialization or file warning)
        # must be re-emitted so the suppression does not eat unrelated
        # warnings.
        def fake_save(_self, _settings, update=False):  # noqa: ARG001
            warnings.warn("unrelated library warning", UserWarning, stacklevel=2)

        mocker.patch.object(
            settings_files.PhotometryWorkingDirSettings, "save", fake_save
        )

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            # PhotometryApertures can be built from defaults, so it is
            # autosaved during construction.
            ReviewSettings([PhotometryApertures])

        assert "unrelated library warning" in [str(w.message) for w in recorded]

    def test_autosave_reemits_unshown_settings_warnings(self, mocker):
        # The autosave suppression exists because the banner already shows
        # the PhotometrySettingsWarnings raised by the widget's own loads.
        # A PhotometrySettingsWarning the banner has *not* shown -- e.g.
        # one raised on the save path rather than by load() -- must be
        # re-emitted rather than silenced.
        def fake_save(_self, _settings, update=False):  # noqa: ARG001
            warnings.warn(
                "Problem while saving", PhotometrySettingsWarning, stacklevel=2
            )

        mocker.patch.object(
            settings_files.PhotometryWorkingDirSettings, "save", fake_save
        )

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            ReviewSettings([PhotometryApertures])

        assert "Problem while saving" in [str(w.message) for w in recorded]

    def test_accordion_collapse_does_not_error(self):
        # An accordion with every section collapsed has selected_index None,
        # which used to raise a TypeError in the selection observer.
        review_settings = ReviewSettings(
            [Camera, PhotometryApertures], style="accordion"
        )
        review_settings._container.selected_index = 0
        review_settings._container.selected_index = None
        # Reaching this point without an exception is the test

    def test_tab_selection_marks_unsaved_setting(self):
        # Selecting a tab whose setting is not saved on disk should set that
        # tab's badge to SETTING_NOT_SAVED. This used to silently do nothing
        # because the selection observer rebound a local variable instead of
        # setting the widget's badge.
        wd_settings = PhotometryWorkingDirSettings()

        # PhotometryApertures can be created from default values, so it is
        # saved automatically when the widget is created; deleting the
        # settings file afterwards leaves its tab in the not-saved state.
        review_settings = ReviewSettings([PhotometryApertures, Camera])
        wd_settings.partial_settings_file.unlink(missing_ok=True)

        # Switch away from the apertures tab and back so the selection
        # observer fires for it.
        review_settings._container.selected_index = 1
        review_settings._container.selected_index = 0

        # Setting the widget badge triggers the observer that copies it into
        # the badge list and the tab title.
        assert review_settings.badges[0] == SaveStatus.SETTING_NOT_SAVED
        assert SaveStatus.SETTING_NOT_SAVED in review_settings._container.titles[0]

    def test_tab_selection_picks_up_external_save(self):
        # A badge latched at SETTING_NOT_SAVED must recover when the setting
        # is saved by something other than this widget's own observers, e.g.
        # another widget or a notebook writing the settings file. Selecting
        # the tab used to trust a non-None badge unconditionally, so the
        # not-saved badge stuck forever.
        wd_settings = PhotometryWorkingDirSettings()

        # PhotometryApertures autosaves its defaults during construction;
        # deleting the settings file afterwards leaves its tab unsaved.
        review = ReviewSettings([PhotometryApertures, Camera])
        wd_settings.partial_settings_file.unlink(missing_ok=True)

        # Switch away and back so the selection observer latches the badge
        # at not-saved.
        review._container.selected_index = 1
        review._container.selected_index = 0
        assert review.badges[0] == SaveStatus.SETTING_NOT_SAVED

        # Save the value displayed in the widget externally, i.e. through a
        # fresh PhotometryWorkingDirSettings rather than the widget's own
        # save machinery.
        apertures = PhotometryApertures.model_validate(
            review._setting_widgets[0]._autoui_widget.value
        )
        PhotometryWorkingDirSettings().save(
            PartialPhotometrySettings(photometry_apertures=apertures), update=True
        )

        # Selecting the tab again re-derives the badge from disk instead of
        # trusting the latched not-saved badge.
        review._container.selected_index = 1
        review._container.selected_index = 0
        assert review.badges[0] == SaveStatus.SETTING_IS_SAVED
        assert SaveStatus.SETTING_IS_SAVED in review._container.titles[0]


def test_add_saving_with_unrecognized_widget():
    # Check that an error is raised if a widget is added to the saving list
    # that is not recognized.
    with pytest.raises(ValueError, match="is not a recognized type of widget"):
        _add_saving_to_widget(ipw.Dropdown())


class TestSpinner:
    def test_create_spinner(self):
        # Test we can create a spinner and set a message
        spinner = Spinner()
        assert spinner.layout.display == "none"

        message = "Hello world"
        spinner = Spinner(message=message)
        assert spinner.message == message

        # Just make sure there is some spinner file
        assert spinner.spinner_file != ""

    def test_start_stop_spinner(self):
        # Test we can start and stop the spinner
        spinner = Spinner()
        spinner.start()
        assert spinner.layout.display == "flex"

        spinner.stop()
        assert spinner.layout.display == "none"

    def test_settings_spinner_file(self, tmp_path):
        # Test that we can set the spinner file. The file we
        # create here is not valid svg. We just make sure that
        # the file is read.

        spinner_file = tmp_path / "fake_spinner.svg"
        spinner_contents = "This is not a valid svg file"
        spinner_file.write_text(spinner_contents)

        spinner = Spinner(spinner_file=spinner_file)
        assert spinner.spinner_file == spinner_file
        assert spinner.children[1].value == spinner_contents
