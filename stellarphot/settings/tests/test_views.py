from stellarphot.settings import Camera, ui_generator
from stellarphot.settings.tests.test_models import TEST_CAMERA_VALUES


class TestUiGenerator:
    def test_camera(self):
        ui = ui_generator(Camera)
        # The description should be the beginning of the docstring
        assert Camera.__doc__.strip().startswith(ui.description.split()[0])

        # We always want to show nullable fields
        assert ui.show_null

        # Which means we don't need the button to show/hide them
        assert ui.bn_shownull.layout.display == "none"

        # For now we do not show the validation output because it is painfully
        # verbose and not very helpful.
        assert not ui.show_validation

        # We always display nested models
        assert ui.open_nested

        # Finally, we don't want the widget to update continuously because it
        # will overwrite the value the user entered.
        for widget in ui.di_widgets.values():
            if hasattr(widget, "continuous_update"):
                assert not widget.continuous_update

    def test_disabled_state_save_revert_button(self):
        # We want the save and revert buttons to be enabled only when
        # 1. The user has made a change, AND
        # 2. The value in the widget is a valid pydantic model

        ui = ui_generator(Camera)
        # The save button should be disabled
        assert ui.savebuttonbar.bn_save.disabled
        assert ui.savebuttonbar.bn_revert.disabled

        # Set one field to a valid value....
        ui.value["name"] = "test"

        # ...and the save button should still be disabled
        assert ui.savebuttonbar.bn_save.disabled
        assert ui.savebuttonbar.bn_revert.disabled

        # Set a valid value
        ui.value = TEST_CAMERA_VALUES

        # So it turns out that the validation stuff only updates when changes are made
        # in the UI rather than programmatically. Since we know we've set a valid value,
        # and that we've made changes we just manually set the relevant values.
        ui.savebuttonbar.unsaved_changes = True
        ui.is_valid.value = True

        # The save button should now be enabled
        assert not ui.savebuttonbar.bn_save.disabled
        assert not ui.savebuttonbar.bn_revert.disabled

        # Click on save
        ui.savebuttonbar.bn_save.click()

        # Unsaved changes should be False
        assert not ui.savebuttonbar.unsaved_changes

        # The save and revert buttons should now be disabled
        assert ui.savebuttonbar.bn_save.disabled
        assert ui.savebuttonbar.bn_revert.disabled
