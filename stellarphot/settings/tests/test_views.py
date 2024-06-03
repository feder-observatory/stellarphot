from stellarphot.settings import Camera, SourceLocationSettings, ui_generator
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

    def test_disabled_state_save_button(self):
        # We want the save button to be enabled only when
        # 1. The user has made a change, AND
        # 2. The value in the widget is a valid pydantic model

        ui = ui_generator(Camera)
        # The save button should be disabled
        assert ui.savebuttonbar.bn_save.disabled

        # Set one field to a valid value....
        ui.value["name"] = "test"

        # We need to manually indicate a change has been made
        ui.savebuttonbar.unsaved_changes = True

        # ...and the save button should still be disabled, because
        # the rest of the fields are not valid.
        assert ui.savebuttonbar.bn_save.disabled

        # Set a valid value
        ui.value = TEST_CAMERA_VALUES

        # So it turns out that the validation stuff only updates when changes are made
        # in the UI rather than programmatically. Since we know we've set a valid value,
        # and that we've made changes we just manually set the relevant values.
        ui.savebuttonbar.unsaved_changes = True
        ui.is_valid.value = True

        # The save button should now be enabled
        assert not ui.savebuttonbar.bn_save.disabled

        # Click on save
        ui.savebuttonbar.bn_save.click()

        # Unsaved changes should be False
        assert not ui.savebuttonbar.unsaved_changes

        # The save and revert buttons should now be disabled
        assert ui.savebuttonbar.bn_save.disabled

    def test_disabled_state_revert_button(self):
        # We want the revert button to be enabled whenever the user has made a change,
        # regardless of whether the value is valid or not.

        ui = ui_generator(Camera)
        # The revert button should be disabled
        assert ui.savebuttonbar.bn_revert.disabled

        # Set one field to a valid value -- the rest of the fields are still invalid
        ui.value["name"] = "test"

        # We need to manually indicate a change has been made
        ui.savebuttonbar.unsaved_changes = True

        # ...and the revert button should be enabled
        assert not ui.savebuttonbar.bn_revert.disabled

        # Click on revert
        ui.savebuttonbar.bn_revert.click()

        # The revert button should now be disabled
        assert ui.savebuttonbar.bn_revert.disabled

        # Unsaved changes should be False
        assert not ui.savebuttonbar.unsaved_changes

    def test_field_width(self):
        # We want to set the width of all except FileChooser fields to a fixed value

        # Nothing special about the value below except that it is unliekly to be
        # a default...
        width = "142px"
        # Use SourceLocationSettings because it has a FileChooser field.
        ui = ui_generator(SourceLocationSettings, max_field_width=width)

        for widget in ui.di_widgets.values():
            if widget.__class__.__name__ == "FileChooser":
                # FileChooser widgets have a different layout object
                assert widget.layout.max_width != width
            else:
                assert widget.layout.max_width == width

    def test_file_chooser_width(self):
        # We want to set the width of FileChooser fields, but no toher fields,
        # to a fixed value.

        # Nothing special about the value below except that it is unliekly to be
        # a default...
        width = "142px"
        # Use SourceLocationSettings because it has a FileChooser field.
        ui = ui_generator(SourceLocationSettings, file_chooser_max_width=width)

        for widget in ui.di_widgets.values():
            if widget.__class__.__name__ == "FileChooser":
                assert widget.layout.max_width == width
            else:
                assert widget.layout.max_width != width
