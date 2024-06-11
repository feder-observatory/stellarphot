import ipywidgets as ipw
import pytest
from ipyautoui.custom.iterable import ItemBox, ItemControl

from stellarphot.settings import (
    Camera,
    Observatory,
    PassbandMap,
    SavedSettings,
    ui_generator,
)
from stellarphot.settings.custom_widgets import (
    ChooseOrMakeNew,
    Confirm,
    SettingWithTitle,
)
from stellarphot.settings.tests.test_models import (
    DEFAULT_OBSERVATORY_SETTINGS,
    DEFAULT_PASSBAND_MAP,
    TEST_CAMERA_VALUES,
)


class TestChooseOrMakeNew:
    """
    Class for testing the ChooseOrMakeNew widget.
    """

    def make_test_camera(self, path):
        """
        Make a camera with the default testing values and save it. This came
        up often enough to warrant its own method.
        """
        saved = SavedSettings(_testing_path=path)
        camera = Camera(**TEST_CAMERA_VALUES)
        saved.add_item(camera)

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

    def test_initial_configuration_with_no_items(self, tmp_path):
        # Should have a dropdown with one item, "Make new passband map"
        # using passband_map here to also test that underscores get converted to spaces
        choose_or_make_new = ChooseOrMakeNew("passband_map", _testing_path=tmp_path)
        assert len(choose_or_make_new._choose_existing.options) == 1
        assert choose_or_make_new._choose_existing.options[0] == (
            "Make new passband map",
            "none",
        )

    def test_make_new_makes_a_new_item(self, tmp_path):
        # We will test this with Camera, should work for the others too since the code
        # path is the same.

        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
        # Set the values for the new item
        choose_or_make_new._item_widget.value = TEST_CAMERA_VALUES

        # Simulate a click on the save button...
        choose_or_make_new._item_widget.savebuttonbar.bn_save.click()
        # No need for the confirmation dialog because we are not overwriting
        # anything.

        # making sure the widget state is properly updated
        assert choose_or_make_new._making_new is False

        # Check what we created using SavedSettings...
        saved = SavedSettings(_testing_path=tmp_path)
        cameras = saved.get_items("camera")
        assert len(cameras.as_dict) == 1
        assert list(cameras.as_dict.values())[0].model_dump() == TEST_CAMERA_VALUES

    @pytest.mark.parametrize(
        "item_type,setting",
        [
            ("camera", TEST_CAMERA_VALUES),
            ("observatory", DEFAULT_OBSERVATORY_SETTINGS),
            ("passband_map", DEFAULT_PASSBAND_MAP),
        ],
    )
    def test_make_new_with_existing_item_resets_value(
        self, tmp_path, item_type, setting
    ):
        # When "make a new" item is selected the value of the widget should be
        # the same as when a widget of that type is created.

        # Make a camera widget just to get the value for a new item.
        choose_or_make_new = ChooseOrMakeNew(item_type, _testing_path=tmp_path)
        value_when_new = choose_or_make_new._item_widget.value.copy()

        # Make a camera
        saved = SavedSettings(_testing_path=tmp_path)
        item = choose_or_make_new._item_widget.model(**setting)
        saved.add_item(item)

        # Make a camera widget and select "Make new"
        choose_or_make_new = ChooseOrMakeNew(item_type, _testing_path=tmp_path)
        choose_or_make_new._choose_existing.value = "none"
        assert choose_or_make_new._item_widget.value == value_when_new

    def test_edit_requires_confirmation(self, tmp_path):
        # Should require confirmation if the item already exists
        self.make_test_camera(tmp_path)
        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
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

    def test_edit_item_saved_after_confirm(self, tmp_path):
        # Should save the item after confirmation
        self.make_test_camera(tmp_path)

        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
        # Simulate a click on the edit button...
        choose_or_make_new._edit_button.click()

        # Change a value in the camera so we can check that the new value is saved.
        new_gain = 2 * TEST_CAMERA_VALUES["gain"]
        choose_or_make_new._item_widget.di_widgets["gain"].value = str(new_gain)
        # Simulate a click on the save button...
        choose_or_make_new._item_widget.savebuttonbar.bn_save.click()
        # Simulate a click on the confirm button...
        choose_or_make_new._confirm_edit_delete._yes.click()

        saved = SavedSettings(_testing_path=tmp_path)
        cameras = saved.get_items("camera")
        assert cameras.as_dict[TEST_CAMERA_VALUES["name"]].gain == new_gain

    def test_edit_item_not_saved_after_cancel(self, tmp_path):
        # Should not save the item after clicking the No button
        self.make_test_camera(tmp_path)

        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
        # Simulate a click on the edit button...
        choose_or_make_new._edit_button.click()

        # Change a value in the camera so we can check that the new value is saved.
        new_gain = 2 * TEST_CAMERA_VALUES["gain"]
        choose_or_make_new._item_widget.di_widgets["gain"].value = str(new_gain)
        # Simulate a click on the save button...
        choose_or_make_new._item_widget.savebuttonbar.bn_save.click()
        # Simulate a click on the cancel button...
        choose_or_make_new._confirm_edit_delete._no.click()

        saved = SavedSettings(_testing_path=tmp_path)
        cameras = saved.get_items("camera")
        assert (
            cameras.as_dict[TEST_CAMERA_VALUES["name"]].gain
            == TEST_CAMERA_VALUES["gain"]
        )

    def test_selecting_make_new_as_selection_works(self, tmp_path):
        # Should allow the user to select "Make new" as a selection

        # Make a camera so that we can test that selecting "Make new" works
        self.make_test_camera(tmp_path)

        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
        # This is the value for the "Make new...." option
        choose_or_make_new._choose_existing.value = "none"

        # edit button should go away
        assert choose_or_make_new._edit_delete_container.layout.display == "none"

        # Widget should be enabled
        assert choose_or_make_new._item_widget.disabled is False

        # save button bar should be displayed
        assert choose_or_make_new._item_widget.savebuttonbar.layout.display != "none"

    def test_choosing_different_item_updates_display(self, tmp_path):
        # Should update the display when a different item is chosen
        saved = SavedSettings(_testing_path=tmp_path)
        observatory = Observatory(**DEFAULT_OBSERVATORY_SETTINGS)
        saved.add_item(observatory)

        observatory2 = Observatory(**DEFAULT_OBSERVATORY_SETTINGS)
        # Make sure this name sorts lower than the first observatory
        observatory2.name = "zzzz" + observatory2.name
        observatory2.elevation = "-200 m"

        saved.add_item(observatory2)

        choose_or_make_new = ChooseOrMakeNew("observatory", _testing_path=tmp_path)

        # There should be three choices in the dropdown
        assert len(choose_or_make_new._choose_existing.options) == 3

        # Make sure the correct first observatory is selected
        assert choose_or_make_new._choose_existing.value == observatory
        assert Observatory(**choose_or_make_new._item_widget.value) == observatory

        # Select the other observatory
        choose_or_make_new._choose_existing.value = observatory2

        # The item widget should now have the values of the second observatory
        assert Observatory(**choose_or_make_new._item_widget.value) == observatory2

    def test_passband_map_buttons_are_disabled_or_enabled(self, tmp_path):
        # When an existing PassbandMap is selected the add/remove buttons
        # for individual rows should not be displayed.
        saved = SavedSettings(_testing_path=tmp_path)
        passband_map = PassbandMap(**DEFAULT_PASSBAND_MAP)
        saved.add_item(passband_map)

        choose_or_make_new = ChooseOrMakeNew("passband_map", _testing_path=tmp_path)

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

    def test_make_passband_map(self, tmp_path):
        # Make a passband map and save it, then check that it is in the dropdown
        saved = SavedSettings(_testing_path=tmp_path)
        passband_map = PassbandMap(**DEFAULT_PASSBAND_MAP)
        saved.add_item(passband_map)

        # Should create a new passband map
        choose_or_make_new = ChooseOrMakeNew("passband_map", _testing_path=tmp_path)
        assert len(choose_or_make_new._choose_existing.options) == 2
        assert choose_or_make_new._choose_existing.options[0][0] == passband_map.name

    def test_no_edit_button_when_there_are_no_items(self, tmp_path):
        # Should not have an edit button when there are no items
        saved = SavedSettings(_testing_path=tmp_path)
        # Make sure there are no cameras
        assert len(saved.get_items("camera").as_dict) == 0

        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
        assert choose_or_make_new._edit_delete_container.layout.display == "none"

    def test_edit_button_returns_after_making_new_item(self, tmp_path):
        # After making a new item the edit button should be displayed

        # There are no cameras so this puts us into the form to make a new one
        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)

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

    def test_save_button_disabled_when_no_changes(self, tmp_path):
        # Immediately after the edit button has been clicked, the save button should be
        # disabled because no changes have been made yet.
        #
        # That should remain true even after the user makes a change, then saves it
        # and then edits again but has not yet made any changes.
        self.make_test_camera(tmp_path)

        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
        # Simulate a click on the edit button...
        choose_or_make_new._edit_button.click()

        # The save button should be disabled
        assert choose_or_make_new._item_widget.savebuttonbar.bn_save.disabled

        # Edit a value in the camera so the save button is enabled
        new_gain = 2 * TEST_CAMERA_VALUES["gain"]
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

    def test_revert_button_is_enabled_after_clicking_edit(self, tmp_path):
        # The revert button should be enabled after clicking the edit button so
        # the user can cancel the edit.
        self.make_test_camera(tmp_path)

        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
        # Simulate a click on the edit button...
        choose_or_make_new._edit_button.click()

        # The revert button should be enabled
        assert not choose_or_make_new._item_widget.savebuttonbar.bn_revert.disabled

    def test_clicking_revert_button_cancels_edit(self, tmp_path):
        # Clicking the revert button should cancel the edit
        self.make_test_camera(tmp_path)

        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
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
        self, tmp_path
    ):
        # The revert button should remain enabled if the value is invalid and reverting
        # should actually revert the value.
        self.make_test_camera(tmp_path)

        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
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

    def test_delete_button_click_displays_confirm_dialog(self, tmp_path):
        # Clicking "Delete" should display a confirmation dialog

        # Make an item to ensure an existing item is displayed when the widget is made
        self.make_test_camera(tmp_path)

        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
        choose_or_make_new._delete_button.click()

        # Confirm dialog should be displayed
        assert choose_or_make_new._confirm_edit_delete.layout.display != "none"
        # edit/delete buttons should be hidden
        assert choose_or_make_new._edit_delete_container.layout.display == "none"
        # The confirm dialog should indicate you are deleting
        assert "delete" in choose_or_make_new._confirm_edit_delete.message.lower()

    @pytest.mark.parametrize("click_yes", [True, False])
    def test_delete_actions_after_confirmation(self, tmp_path, click_yes):
        # Clicking "Yes" in the confirmation dialog should delete the item
        self.make_test_camera(tmp_path)

        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
        choose_or_make_new._delete_button.click()

        if click_yes:
            # Simulate a click on the "Yes" button in the confirm dialog
            choose_or_make_new._confirm_edit_delete._yes.click()

            # The camera should be gone
            saved = SavedSettings(_testing_path=tmp_path)
            cameras = saved.get_items("camera")
            assert len(cameras.as_dict) == 0
        else:
            # Simulate a click on the "No" button in the confirm dialog
            choose_or_make_new._confirm_edit_delete._no.click()

            # The camera should still be there
            saved = SavedSettings(_testing_path=tmp_path)
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

    def test_correct_item_selected_after_delete_and_yes(self, tmp_path):
        # The correct item should be selected after an item is deleted and the user
        # clicks "Yes" in the confirmation dialog.
        # Make two cameras
        self.make_test_camera(tmp_path)
        saved = SavedSettings(_testing_path=tmp_path)
        camera2 = Camera(**TEST_CAMERA_VALUES)
        camera2.name = "zzzz" + camera2.name
        saved.add_item(camera2)

        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
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
    def test_making_new_with_same_name_as_existing(self, tmp_path, click_yes):
        # Making a new item with the same name as an existing item should require
        # confirmation
        self.make_test_camera(tmp_path)

        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)

        # Click the "Make new" button
        choose_or_make_new._choose_existing.value = "none"

        # set the item widget value to the existing camera with double the gain
        camera_values = TEST_CAMERA_VALUES.copy()
        camera_values["gain"] = 2 * camera_values["gain"]
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

    def test_weird_sequence_of_no_clicks(self, tmp_path):
        # This is a regression test for #320
        # Make a camera
        self.make_test_camera(tmp_path)

        # Make a new camera...
        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
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
    def test_details_hideable_or_not(self, tmp_path, hideable):
        # The details should be hideable if requested

        # Make a camera so that the details can be hidden
        self.make_test_camera(tmp_path)
        choose_or_make_new = ChooseOrMakeNew(
            "camera", details_hideable=hideable, _testing_path=tmp_path
        )

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

    def test_details_hideable_plays_nicely_with_new_item(self, tmp_path):
        # The details should be hideable and should play nicely with making a new item
        # Make an item so that selecting "Make new" will count as a value
        # change later.
        self.make_test_camera(tmp_path)

        choose_or_make_new = ChooseOrMakeNew(
            "camera", details_hideable=True, _testing_path=tmp_path
        )

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
        new_camera = TEST_CAMERA_VALUES.copy()
        new_camera["name"] = "new camera"
        choose_or_make_new._item_widget.value = new_camera
        choose_or_make_new._item_widget.savebuttonbar.bn_save.click()

        # "show details" should be visible
        assert choose_or_make_new._show_details_ui.layout.display != "none"

        # Details should be hidden
        assert choose_or_make_new._details_box.layout.display == "none"
        assert not choose_or_make_new._show_details_ui.value

    def test_details_hideable_not_set_preserves_old_behavior(self, tmp_path):
        # The details should not be hidden if hideable is false
        # Make a camera
        self.make_test_camera(tmp_path)

        choose_or_make_new = ChooseOrMakeNew(
            "camera", details_hideable=False, _testing_path=tmp_path
        )

        # New UI element should not be in the big box
        assert choose_or_make_new._show_details_ui.layout.display == "none"

        # Details should be displayed
        assert choose_or_make_new._details_box.layout.display != "none"

        # Set "show details" box to unchecked
        choose_or_make_new._show_details_ui.value = False

        # Details should still be displayed
        assert choose_or_make_new._details_box.layout.display != "none"

    @pytest.mark.parametrize("show_detail_state", [True, False])
    def test_details_hideable_cancel_making_new_restores_state(
        self, tmp_path, show_detail_state
    ):
        # The details should be hideable and should play nicely with making a new item
        # Make an item so that selecting "Make new" will count as a value
        # change later.
        self.make_test_camera(tmp_path)

        choose_or_make_new = ChooseOrMakeNew(
            "camera", details_hideable=True, _testing_path=tmp_path
        )

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

    def test_chooser_has_value(self, tmp_path):
        # The chooser should have a value
        self.make_test_camera(tmp_path)

        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
        assert choose_or_make_new.value == Camera(**TEST_CAMERA_VALUES)

    def test_details_can_be_hidden(self, tmp_path):
        # Make sure that item details can be hidden programmatically.
        self.make_test_camera(tmp_path)

        choose_or_make_new = ChooseOrMakeNew(
            "camera", details_hideable=True, _testing_path=tmp_path
        )

        # Check that the details start out displayed
        assert choose_or_make_new._details_box.layout.display != "none"
        assert choose_or_make_new.display_details

        # Hide the details
        choose_or_make_new.display_details = False

        # Check that the details are hidden
        assert choose_or_make_new._details_box.layout.display == "none"

    def test_details_visibilty_cannot_be_changed_when_not_hideable(self, tmp_path):
        # Make sure that item details cannot be hidden programmatically when hideable
        # is False.
        self.make_test_camera(tmp_path)

        choose_or_make_new = ChooseOrMakeNew(
            "camera", details_hideable=False, _testing_path=tmp_path
        )

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

    def test_title_decoration(self):
        # Check that the title has the correct decoration given the
        # state of the settings widget.

        # Make a camera ui
        camera = ui_generator(Camera)

        # Make a SettingWithTitle with the camera
        plain_title = "I am a camera"
        camera_title = SettingWithTitle(plain_title, camera)

        # At the moment the title should not be decorated
        assert camera_title.SETTING_IS_SAVED not in camera_title.title.value
        assert camera_title.SETTING_NOT_SAVED not in camera_title.title.value

        # There should also be no unsaved changes at the moment
        assert not camera.savebuttonbar.unsaved_changes

        # Manually set unsaved_changes then call the change handler, which should
        # add an indication that there are unsaved changes.
        camera.savebuttonbar.unsaved_changes = True
        camera_title.decorate_title()
        assert camera_title.SETTING_NOT_SAVED in camera_title.title.value

        # Go back to unsaved_changes being False
        camera.savebuttonbar.unsaved_changes = False

        # Manually call the change handler, which should add an indication that
        # saves have been done.
        camera_title.decorate_title()
        assert camera_title.SETTING_IS_SAVED in camera_title.title.value

        # Now change the camera value, which should trigger the change handler
        camera.value = TEST_CAMERA_VALUES

        assert camera_title.SETTING_NOT_SAVED in camera_title.title.value

        # Finally, click the save button and the title should be decorated with
        # the saved indication.
        camera.savebuttonbar.bn_save.click()
        assert camera_title.SETTING_IS_SAVED in camera_title.title.value
