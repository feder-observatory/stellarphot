import ipywidgets as ipw
import pytest
from ipyautoui.custom.iterable import ItemBox, ItemControl

from stellarphot.settings import Camera, Observatory, PassbandMap, SavedSettings
from stellarphot.settings.custom_widgets import ChooseOrMakeNew, Confirm
from stellarphot.settings.tests.test_models import (
    DEFAULT_OBSERVATORY_SETTINGS,
    DEFAULT_PASSBAND_MAP,
    TEST_CAMERA_VALUES,
)


class TestChooseOrMakeNew:
    """
    Class for testing the ChooseOrMakeNew widget.
    """

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

        # Check what we created using SavedSettings...
        saved = SavedSettings(_testing_path=tmp_path)
        cameras = saved.get_items("camera")
        assert len(cameras.as_dict) == 1
        assert list(cameras.as_dict.values())[0].model_dump() == TEST_CAMERA_VALUES

    def test_edit_requires_confirmation(self, tmp_path):
        # Should require confirmation if the item already exists
        saved = SavedSettings(_testing_path=tmp_path)
        camera = Camera(**TEST_CAMERA_VALUES)
        saved.add_item(camera)
        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
        # the edit button should be displayed and the confirm widget should be hidden
        # note: display typically start as None or an empty string, so we just check
        # that it is not "none", which is what it will be set to when it is hidden.
        assert choose_or_make_new._edit_button.layout.display != "none"
        assert choose_or_make_new._confirm_edit.layout.display == "none"

        # Simulate a click on the edit button...
        choose_or_make_new._edit_button.click()
        # The edit button should now be hidden
        assert choose_or_make_new._edit_button.layout.display == "none"

        # The savebuttonbar should be displayed
        assert choose_or_make_new._item_widget.savebuttonbar.layout.display != "none"

        # Click on the save button
        choose_or_make_new._item_widget.savebuttonbar.bn_save.click()

        # The confirm dialog should be displayed
        assert choose_or_make_new._confirm_edit.layout.display != "none"

    def test_edit_item_saved_after_confirm(self, tmp_path):
        # Should save the item after confirmation
        saved = SavedSettings(_testing_path=tmp_path)
        camera = Camera(**TEST_CAMERA_VALUES)
        saved.add_item(camera)
        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
        # Simulate a click on the edit button...
        choose_or_make_new._edit_button.click()

        # Change a value in the camera so we can check that the new value is saved.
        new_gain = 2 * TEST_CAMERA_VALUES["gain"]
        choose_or_make_new._item_widget.di_widgets["gain"].value = str(new_gain)
        # Simulate a click on the save button...
        choose_or_make_new._item_widget.savebuttonbar.bn_save.click()
        # Simulate a click on the confirm button...
        choose_or_make_new._confirm_edit._yes.click()

        cameras = saved.get_items("camera")
        assert cameras.as_dict[camera.name].gain == new_gain

    def test_edit_item_not_saved_after_cancel(self, tmp_path):
        # Should not save the item after clicking the No button
        saved = SavedSettings(_testing_path=tmp_path)
        camera = Camera(**TEST_CAMERA_VALUES)
        saved.add_item(camera)
        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
        # Simulate a click on the edit button...
        choose_or_make_new._edit_button.click()

        # Change a value in the camera so we can check that the new value is saved.
        new_gain = 2 * TEST_CAMERA_VALUES["gain"]
        choose_or_make_new._item_widget.di_widgets["gain"].value = str(new_gain)
        # Simulate a click on the save button...
        choose_or_make_new._item_widget.savebuttonbar.bn_save.click()
        # Simulate a click on the cancel button...
        choose_or_make_new._confirm_edit._no.click()

        cameras = saved.get_items("camera")
        assert cameras.as_dict[camera.name].gain == TEST_CAMERA_VALUES["gain"]

    def test_selecting_make_new_as_selection_works(self, tmp_path):
        # Should allow the user to select "Make new" as a selection

        # Make a camera so that we can test that selecting "Make new" works
        saved = SavedSettings(_testing_path=tmp_path)
        camera = Camera(**TEST_CAMERA_VALUES)
        saved.add_item(camera)

        choose_or_make_new = ChooseOrMakeNew("camera", _testing_path=tmp_path)
        # This is the value for the "Make new...." option
        choose_or_make_new._choose_existing.value = "none"

        # edit button should go away
        assert choose_or_make_new._edit_button.layout.display == "none"

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
        print(f"{choose_or_make_new._item_widget.value=}")
        print(f"{observatory.model_dump()=}")
        assert Observatory(**choose_or_make_new._item_widget.value) == observatory

        # Select the other observatory
        choose_or_make_new._choose_existing.value = observatory2

        # The item widget should now have the values of the second observatory
        assert Observatory(**choose_or_make_new._item_widget.value) == observatory2

    def test_make_passband_map(self, tmp_path):
        # Make a passband map and save it, then check that it is in the dropdown
        saved = SavedSettings(_testing_path=tmp_path)
        passband_map = PassbandMap(**DEFAULT_PASSBAND_MAP)
        saved.add_item(passband_map)

        # Should create a new passband map
        choose_or_make_new = ChooseOrMakeNew("passband_map", _testing_path=tmp_path)
        assert len(choose_or_make_new._choose_existing.options) == 2
        assert choose_or_make_new._choose_existing.options[0][0] == passband_map.name

    def test_passband_map_buttons_are_disabled(self, tmp_path):
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
        print(f"{item_box.add_remove_controls=}")
        assert item_box.add_remove_controls == ItemControl.none


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
