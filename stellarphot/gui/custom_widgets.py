# Some settings require custom widgets to be displayed in the GUI. These are defined in
# this module.

import html
import warnings
from dataclasses import dataclass
from enum import StrEnum

import ipywidgets as ipw
import papermill as pm
import traitlets as tr
from astropy.utils.data import get_pkg_data_filename
from ipyautoui.autoobject import AutoObject
from ipyautoui.custom.iterable import ItemBox
from pydantic import ValidationError
from pydantic.alias_generators import to_snake

from stellarphot.gui.fits_opener import FitsOpener
from stellarphot.gui.views import ui_generator
from stellarphot.io.tess import TIC_regex, tess_photometry_setup
from stellarphot.settings import (
    Camera,
    Observatory,
    PartialPhotometrySettings,
    PassbandMap,
    PhotometryRunSettings,
    PhotometrySettingsWarning,
    PhotometryWorkingDirSettings,
    SavedSettings,
)

__all__ = ["ChooseOrMakeNew", "Confirm", "SettingWithTitle"]

DEFAULT_BUTTON_WIDTH = "300px"


class ChooseOrMakeNew(ipw.VBox):
    """
    Widget to present a list of existing items or the option to make a new one.

    Parameters
    ----------

    item_type_name : str
        Name of the item type to be displayed in the widget. Must be one of
        "camera", "observatory", "passband_map", "Camera", "Observatory"
        or "PassbandMap".
    """

    _known_types = [
        "camera",
        "observatory",
        "passband_map",
        Camera.__name__,
        Observatory.__name__,
        PassbandMap.__name__,
    ]

    def __init__(self, item_type_name, *arg, details_hideable=False, **kwargs):
        if item_type_name not in self._known_types:
            raise ValueError(
                f"Unknown item type {item_type_name}. Must "
                f"be {', '.join(self._known_types)}"
            )
        # Get the widgety goodness from the parent class
        super().__init__(*arg, **kwargs)

        self._saved_settings = SavedSettings()
        self._item_type_name = item_type_name

        # keep track of whether we are editing an existing item
        self._editing = False

        # also track whether we are in the midst of a delete confirmation
        self._deleting = False

        # and track if we are making a new item
        self._making_new = False

        # keep track of whether there is a "show details" checkbox
        self._show_details_shown = details_hideable

        self._display_name = item_type_name.replace("_", " ")

        # Create the child widgets

        # Descriptive title
        self._title = ipw.HTML(
            value=(f"Choose a {self._display_name} or make a new one")
        )

        self._choose_detail_container = ipw.HBox(layout={"width": DEFAULT_BUTTON_WIDTH})

        # Selector for existing items or to make a new one
        self._choose_existing = ipw.Dropdown(description="")
        choose_width = 75  # percent, the details checkbox takes up the rest
        self._choose_existing.layout.width = (
            f"{choose_width}%" if details_hideable else "100%"
        )

        # Option to show/hide details, only displayed if user wants it.
        self._show_details_ui = ipw.Checkbox(description="Details", value=True)
        self._show_details_ui.layout.display = "flex" if details_hideable else "none"
        # Removes unused whitespace before the checkbox
        self._show_details_ui.style.description_width = "0px"

        if details_hideable:
            self._show_details_ui.layout.width = f"{100 - choose_width}%"
        self._show_details_cached_value = self._show_details_ui.value

        self._choose_detail_container.children = [
            self._choose_existing,
            self._show_details_ui,
        ]

        self._edit_delete_container = ipw.HBox(
            # width below was chosen to match the dropdown...would prefer to
            # determine this programmatically but don't know how.
            layout={"width": DEFAULT_BUTTON_WIDTH}
        )

        self._edit_button = ipw.Button(
            description=f"Edit this {self._display_name}",
        )

        self._delete_button = ipw.Button(
            description=f"Delete this {self._display_name}",
        )

        # Put almost everything into a VBox
        self._details_box = ipw.VBox()

        self._edit_delete_container.children = [self._edit_button, self._delete_button]

        self._confirm_edit_delete = Confirm()

        self._item_widget, self._widget_value_new_item = self._make_new_widget()

        # Put all of the details into a box that can be easily hidden
        self._details_box.children = [
            self._edit_delete_container,
            self._confirm_edit_delete,
            self._item_widget,
        ]

        # Build the main widget
        self.children = [
            self._title,
            self._choose_detail_container,
            self._details_box,
        ]

        # Set up the dropdown widget
        self._construct_choices()
        # Set the selection to the first choice if there is one
        self._choose_existing.value = self._choose_existing.options[0][1]

        # An observer has not been set up yet, so manually call the handler
        if len(self._choose_existing.options) == 1:
            # There are no items, so we are making a new one
            self._handle_selection({"new": "none"})
        else:
            self._handle_selection({"new": self._choose_existing.value})

        # A couple of styling choices for the way existing objects appear
        # in this UI. The title/description is clear from the title of this
        # widget.
        self._item_widget.show_title = False
        self._item_widget.show_description = False
        # Really only applies to PassbandMap, which has nested models,
        # but does no harm in the other cases (true of both lines below)
        self._item_widget.open_nested = True

        # Set up some observers

        # Respond to user clicking the edit button
        self._edit_button.on_click(self._edit_button_action)

        # Respond to user clicking the delete button
        self._delete_button.on_click(self._delete_button_action)

        # Respond to user clicking the save button
        self._item_widget.savebuttonbar.fns_onsave_add_action(self._save_confirmation())

        # Respond to user interacting with a confirmation widget
        # Hide the save button bar so the user gets the confirmation instead
        self._confirm_edit_delete.widget_to_hide = self._item_widget.savebuttonbar
        # Add the observer
        self._confirm_edit_delete.observe(self._handle_confirmation(), names="value")

        # Respond when user wants to make a new thing
        self._choose_existing.observe(self._handle_selection, names="value")

        # Set up an observer to show/hide the details box if the check box
        # is clicked
        self._show_details_ui.observe(self._show_details_handler, names="value")

    @property
    def value(self):
        """
        The value of the widget.
        """
        return self._item_widget.model(**self._item_widget.value)

    @property
    def is_mid_interaction(self):
        """
        True while the user is making a new item or editing an existing one,
        i.e. while the displayed value may not match anything saved.
        """
        return self._making_new or self._editing

    @property
    def display_details(self):
        """
        Whether the details box is displayed. Returns the value of the details checkbox
        if the details are hideable, otherwise returns None.
        """
        if self._show_details_shown:
            return self._show_details_ui.value
        else:
            return None

    @display_details.setter
    def display_details(self, value):
        """
        Set the value of the details checkbox if the details are hideable.
        """
        if self._show_details_shown:
            self._show_details_ui.value = value

    def _save_confirmation(self):
        """
        Function to attach to the save button to show the confirmation widget if
        the save button was clicked while editing an existing item.
        """

        def f():
            # This function will be run every time the save button is clicked but
            # we only want to ask for confirmation if we are editing an existing item
            # rather than saving a new one.
            if self._editing:
                self._set_confirm_message()
                self._confirm_edit_delete.show()

        return f

    def _construct_choices(self):
        """
        Set up the choices for the selection widget.
        """
        saved_items = self._saved_settings.get_items(self._item_type_name)
        existing_choices = [(k, v) for k, v in saved_items.as_dict.items()]
        existing_choices = sorted(existing_choices, key=lambda x: x[0].lower())
        choices = existing_choices + [(f"Make new {self._display_name}", "none")]
        # self._choose_existing.value = None
        # This sets the options but doesn't select any of them
        self._choose_existing.options = choices

    def _handle_selection(self, change):
        if change["new"] is None:
            return
        if change["new"] == "none":
            # We are making a new item...

            # Hide the edit button
            self._edit_delete_container.layout.display = "none"

            # Make sure details are shown and hide the "show details" checkbox
            if self._show_details_shown:
                self._show_details_cached_value = self._show_details_ui.value
                self._show_details_ui.value = True
                self._show_details_ui.layout.display = "none"

            # This sets the ui back to its original state when created, i.e.
            # everything is empty.
            self._item_widget._init_ui()

            # Fun fact: _init_ui does not reset the value of the widget. Also,
            # setting the value fails if you try to set it to an empty dict because that
            # is not a valid value for the pydantic model for the widget.
            # So we have to set each of the values individually.
            for key, value in self._widget_value_new_item.items():
                self._item_widget.value[key] = value

            self._item_widget.show_savebuttonbar = True
            self._item_widget.disabled = False

            # Set the validation status to invalid since the user must
            # fill in the fields, and display the validation status
            self._item_widget.is_valid.value = False
            self._item_widget.is_valid.layout.display = "flex"

            # Really only applies to PassbandMap, which has nested models,
            # but does no harm in the other cases (true of both lines below)
            # (and yes, both lines below are needed...this is a bug in ipyautoui,
            #  I think, because open_nested=True isn't respected when we _init_ui.
            #  Forcing a *change* in the value triggers the behavior we want.)
            self._item_widget.open_nested = False
            self._item_widget.open_nested = True

            # Note that we are making a new item
            self._making_new = True

        else:
            # Display the selected item...
            self._item_widget.show_savebuttonbar = False
            self._item_widget.disabled = True
            self._item_widget.is_valid.layout.display = "none"
            self._item_widget.value = self._get_item(change["new"].name)

            # Display the edit button
            self._edit_delete_container.layout.display = "flex"

            # Really only applies to PassbandMap, which has nested models,
            # but does no harm in the other cases
            self._set_disable_state_nested_models(self._item_widget, True)

            # We may have arrived here by choosing a different item while
            # making a new one, so we restore the state of the "show details"
            # checkbox.
            if self._show_details_shown:
                self._show_details_ui.layout.display = "flex"
                self._show_details_ui.value = self._show_details_cached_value

    def _edit_button_action(self, _):
        """
        Handle the edit button being clicked.
        """
        # Replace the display of the edit button with the save button bar...
        self._edit_delete_container.layout.display = "none"
        self._item_widget.show_savebuttonbar = True
        # ...enable the widget...
        self._item_widget.disabled = False
        # ...and show the validation status
        self._item_widget.is_valid.layout.display = "flex"

        # Enable the nested model components
        self._set_disable_state_nested_models(self._item_widget, False)

        # disable the name control, since the whole point of this
        # is to be able to replace the values for a particular name
        self._item_widget.di_widgets["name"].disabled = True

        # This really only applies to PassbandMap, which has nested models,
        # but does no harm in the other cases (true of both lines below)
        # (and yes, both lines below are needed...this is a bug in ipyautoui,
        #  I think, because open_nested=True isn't respected when we _init_ui.
        #  Forcing a *change* in the value triggers the behavior we want.)
        self._item_widget.open_nested = False
        self._item_widget.open_nested = True

        # Update the current state of the widget
        self._editing = True

        # Enable the revert button so that the user can cancel the edit
        self._item_widget.savebuttonbar.bn_revert.disabled = False

    def _delete_button_action(self, _):
        """
        Handle the delete button being clicked.
        """
        # Change our state
        self._deleting = True

        # Hide the edit/delete buttons
        self._edit_delete_container.layout.display = "none"

        # Show the confirmation widget
        self._set_confirm_message()
        self._confirm_edit_delete.show()

    def _show_details_handler(self, change):
        """
        Show or hide the details box based on the value of the checkbox.
        """
        if self._show_details_ui.layout.display == "none":
            # The element is hidden, so just return
            return

        self._details_box.layout.display = "flex" if change["new"] else "none"

    def _set_disable_state_nested_models(self, top, value):
        """
        When a one model contains another and the top-level model widget
        sets disabled=True that does not actually disable the nested model.
        This method handles that in a crude way by walking the tree of
        widgets in the top-level model widget and disabling them all.

        Parameters
        ----------

        top : `ipyautoui.AutoUi`
            Top-level widget that may have nested models.

        value : bool
            State that ``disabled`` should be set to.
        """

        if isinstance(top, AutoObject):
            top.disabled = value
        elif isinstance(top, ItemBox):
            if value:
                # Disabled, so do not show the add/remove buttons
                top.add_remove_controls = "none"
            else:
                # Enabled, so show the add/remove buttons
                top.add_remove_controls = "add_remove"

        try:
            for child in top.children:
                self._set_disable_state_nested_models(child, value)
        except AttributeError:
            # No children...
            pass

    def _set_confirm_message(self):
        """
        Set the message for the confirmation widget.
        """
        if self._editing or self._making_new:
            self._confirm_edit_delete.message = (
                f"Replace value of this {self._display_name}?"
            )
        elif self._deleting:
            self._confirm_edit_delete.message = f"Delete this {self._display_name}?"

    def _make_new_widget(self):
        """
        Make a new widget for the item type and set up actions for the save button.

        Also returns the initial value of the widget for resetting the widget value.
        """
        match self._item_type_name:
            case "camera" | Camera.__name__:
                new_widget = ui_generator(Camera)
            case "observatory" | Observatory.__name__:
                new_widget = ui_generator(Observatory)
            case "passband_map" | PassbandMap.__name__:
                new_widget = ui_generator(PassbandMap)

        def saver():
            """
            Tries to save the new item, and if it fails, shows the confirmation widget.
            """
            try:
                self._saved_settings.add_item(new_widget.model(**new_widget.value))
            except ValueError:
                # This will happen in two circumstances if the item already exists:
                # 1. User is editing an existing item
                # 2. User is making a new item with the same name as an existing one
                self._set_confirm_message()
                self._confirm_edit_delete.show()
            else:
                # If saving works, we update the choices and select the new item
                self._making_new = False
                if self._show_details_shown:
                    self._show_details_ui.layout.display = "flex"
                    self._show_details_ui.value = self._show_details_cached_value
                update_choices_and_select_new()

        def update_choices_and_select_new():
            """
            Update the choices after a new item is saved, update the choices
            and select the new item.
            """
            if not (self._editing or self._making_new):
                value_to_select = new_widget.model(**new_widget.value)
                self._construct_choices()
                self._choose_existing.value = value_to_select
                # Make sure the edit button is displayed
                self._edit_delete_container.layout.display = "flex"

        def revert_to_saved_value():
            """
            Revert the widget to the saved value and end editing.

            This should only apply while editing. If you are making a new
            item you can either select a different item (if there are any) or
            you really need to make a new one.
            """
            if self._editing:
                # We have a selection so we need to stop editing...
                self._editing = False

                # ...and trigger the selection handler.
                self._handle_selection({"new": self._choose_existing.value})

        # This is the mechanism for adding callbacks to the save button.
        new_widget.savebuttonbar.fns_onsave_add_action(saver)
        new_widget.savebuttonbar.fns_onsave_add_action(update_choices_and_select_new)
        new_widget.savebuttonbar.fns_onrevert_add_action(revert_to_saved_value)

        return new_widget, new_widget.value.copy()

    def _handle_confirmation(self):
        """
        Handle the confirmation of a save operation.
        """

        # Use a closure here to capture the current state of the widget
        def confirm_handler(change):
            """
            This handles interactions with the confirmation widget, which is displayed
            when the user has done any of these things:

            + tried to save a new item with the same name as an existing one
            + tried to save an existing item they have edited
            + tried to delete an existing item.

            The widget has three possible values: True (yes), False (no), and None

            This widget is called when the widget value changes, which can happen two
            ways:

            1. The user clicks the "yes" or "no" button, in which case the value will
                be True or False, respectively.
            2. This handler sets the value to None after the user has clicked Yes or No.

            The second case is the reason most of the handler is wrapped in an
            if statement.
            """
            was_editing = self._editing
            # value of None means the widget has been reset to not answered
            if change["new"] is not None:
                item = self._item_widget.model(**self._item_widget.value)
                if self._editing or self._making_new:
                    # We are done editing/making new regardless of
                    # the confirmation outcome
                    self._making_new = False
                    self._editing = False
                    if change["new"]:
                        # User has said yes to updating the item, which we do by
                        # deleting the old one and adding the new one.
                        self._saved_settings.delete_item(item, confirm=True)
                        self._saved_settings.add_item(item)
                        # Rebuild the dropdown list
                        self._construct_choices()
                        # Select the edited item
                        # To make 100% sure the observer is triggered, we set the value
                        # to None first.
                        self._choose_existing.value = None
                        self._choose_existing.value = item
                    else:
                        # User has said no to updating the item, so we just
                        # act as though the user has selected this item.
                        if was_editing:
                            # The user has presumably changed the value in the UI, so
                            # get the correct value from disk.
                            item = self._get_item(item.name)

                            # To make 100% sure the observer is triggered, we set the
                            # value to None first.
                            self._choose_existing.value = None
                            self._choose_existing.value = item
                        else:
                            # Set the selection to the first choice if there is one
                            # To make 100% sure the observer is triggered, we set the
                            # value to None first.
                            self._choose_existing.value = None
                            self._choose_existing.value = self._choose_existing.options[
                                0
                            ][1]

                elif self._deleting:
                    if change["new"]:
                        # User has confirmed the deletion
                        self._saved_settings.delete_item(item, confirm=True)
                        # Rebuild the dropdown list
                        self._construct_choices()

                        # Select the first item...
                        self._choose_existing.value = self._choose_existing.options[0][
                            1
                        ]
                        # ...but if there is only one option, the line above doesn't
                        # trigger the _choose_existing observer because the value is set
                        # when the options are set. So we need to trigger it manually.
                        if len(self._choose_existing.options) == 1:
                            self._handle_selection({"new": self._choose_existing.value})
                    else:
                        # User has decided not to delete the item
                        self._handle_selection({"new": item})
                    self._deleting = False

                # Reset the confirmation widget to unanswered
                self._confirm_edit_delete.value = None

        return confirm_handler

    def _get_item(self, item_name):
        """
        Get an item from the saved settings by name.
        """
        match self._item_type_name:
            case "camera" | Camera.__name__:
                container = self._saved_settings.cameras
            case "observatory" | Observatory.__name__:
                container = self._saved_settings.observatories
            case "passband_map" | PassbandMap.__name__:
                container = self._saved_settings.passband_maps

        return container.as_dict[item_name]


class Confirm(ipw.HBox):
    """
    Widget to confirm a choice.

    The value of this widget will be ``True`` if the user confirms the choice, ``False``
    if they do not, and ``None`` if they have not yet answered.
    """

    def __init__(self, message="", widget_to_hide=None, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        # Hide this widget until it is needed
        self.layout.display = "none"
        self._other_widget = widget_to_hide
        self._message = ipw.HTML(value=message)
        button_layout = ipw.Layout(width="50px")
        self._yes = ipw.Button(
            description="Yes", button_style="success", layout=button_layout
        )
        self._no = ipw.Button(
            description="No", button_style="danger", layout=button_layout
        )
        self._yes.on_click(self._handle_yes)
        self._no.on_click(self._handle_no)
        self.children = [self._message, self._yes, self._no]
        # Value van be either True (yes), False (no), or None (not yet answered)
        self.add_traits(value=tr.Bool(allow_none=True))
        self.value = None

    @property
    def message(self):
        return self._message.value

    @message.setter
    def message(self, value):
        self._message.value = value

    def show(self):
        """
        Display the confirmation widget and, if desired, hide the widget it replaces.
        """
        self.layout.display = "flex"
        if self._other_widget is not None:
            self._other_widget.layout.display = "none"

    # THere ought to be a way to refactor these two, but this works for now.
    def _handle_yes(self, _):
        self.layout.display = "none"
        if self._other_widget is not None:
            self._other_widget.layout.display = "flex"
        self.value = True

    def _handle_no(self, _):
        self.layout.display = "none"
        if self._other_widget is not None:
            self._other_widget.layout.display = "flex"
        self.value = False


class SaveStatus(StrEnum):
    """
    Class to define the symbols used to represent a save status.
    """

    SETTING_NOT_SAVED = "❗️"
    SETTING_IS_SAVED = "✅"
    SETTING_SHOULD_BE_REVIEWED = "🔆"


# Shown beside the save button of a setting that is not in the saved settings
# for this directory, so that a tab badged SETTING_NOT_SAVED says how to fix
# itself instead of just reporting a problem.
SAVE_PROMPT_MESSAGE = "<i>not saved in this directory — click save to add it</i>"


class SettingWithTitle(ipw.VBox):
    """
    Class that adds a title to a setting widget made by ipyautoui and
    styles the title based on whether the settings need to be saved.

    Parameters
    ----------

    plain_title : str
        Title of the setting widget without any decoration.

    widget : ipyautoui.AutoUi
        The setting widget to be displayed.
    """

    badge = tr.UseEnum(SaveStatus, default=None, allow_none=True)

    def __init__(self, plain_title, widget, header_level=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._header_level = header_level
        self._plain_title = plain_title
        self._widget = widget

        if isinstance(widget, ChooseOrMakeNew):
            self.title = self._widget._title
            self.children = [self._widget]
            observer = self._choose_existing_observer
            self._widget._choose_existing.observe(observer, names="value")
            self._autoui_widget = self._widget._item_widget
            # In case a value gets set programmatically....
            # self._autoui_widget.observe(self._title_observer, names="_value")
        else:
            self.title = ipw.HTML()
            self._format_title(None)
            self.children = [self.title, self._widget]
            # Set up an observer to update title decoration when the settings
            # change.
            observer = self._title_observer
            # Also update after the save button is clicked
            # self._widget.savebuttonbar.fns_onsave_add_action(self._title_observer)
            self._autoui_widget = self._widget
        self._autoui_widget.savebuttonbar.observe(observer, names="unsaved_changes")

    def _choose_existing_observer(self, _=None):
        """
        Observer for the ChooseOrMakeNew widget.
        """
        # Unless we are making a new item or editing an item then what is displayed
        # is saved.
        if not self._widget.is_mid_interaction:
            self.badge = SaveStatus.SETTING_IS_SAVED
        else:
            self.badge = SaveStatus.SETTING_NOT_SAVED

    @tr.observe("badge")
    def _format_title(self, _=None):
        badge = self.badge or ""
        badge = badge + " " if badge else ""
        self.title.value = (
            f"<h{self._header_level}>{badge}{self._plain_title}</h{self._header_level}>"
        )

    def decorate_title(self):
        """
        Public interface for forcing a title update.
        """
        self._format_title()

    def _title_observer(self, change):
        """
        Observer for the title of the widget, triggered when unsaved_changes
        changes.
        """
        if change["new"]:
            # i.e. unsaved_changes is True
            self.badge = SaveStatus.SETTING_NOT_SAVED
        else:
            self.badge = SaveStatus.SETTING_IS_SAVED

    def prompt_save(self):
        """
        Ask the save button bar to advertise that this setting needs saving.

        The status light next to the save button reports whether the *form*
        has been edited since the last save, so it happily says "SAFE" while
        the setting is missing from the working directory entirely -- which
        contradicts a SETTING_NOT_SAVED badge and makes the enabled save
        button look inert. Setting ``unsaved_changes`` puts the light in its
        "changes since last save" state and points at the button that fixes
        things.
        """
        if isinstance(self._widget, ChooseOrMakeNew):
            # A ChooseOrMakeNew is fixed by picking or making an item in its
            # dropdown, not by a save button -- one it keeps hidden while an
            # existing item is displayed. Its ``unsaved_changes`` observer is
            # _choose_existing_observer, which derives the badge from
            # is_mid_interaction and so could report SAVED here.
            return
        self._autoui_widget.savebuttonbar.unsaved_changes = True
        self._autoui_widget.savebuttonbar.message.value = SAVE_PROMPT_MESSAGE

    def clear_save_prompt(self):
        """
        Undo `prompt_save`, for a setting that is on disk again.
        """
        if isinstance(self._widget, ChooseOrMakeNew):
            return
        self._autoui_widget.savebuttonbar.unsaved_changes = False
        # Only our own prompt is cleared; a genuine "changes saved: ..." or
        # "UI reverted to last save" from ipyautoui stays put.
        if self._autoui_widget.savebuttonbar.message.value == SAVE_PROMPT_MESSAGE:
            self._autoui_widget.savebuttonbar.message.value = ""


# Fixed identity key for the (at most one) load-error banner message, so
# that a refresh which reworks the message for the same incident updates it
# in place rather than appending a near-duplicate. Not plausible warning
# text, since a warning is keyed on its own text in the same dict.
_LOAD_ERROR_KEY = "__settings-load-error__"

# CSS class (and the style sheet that gives it meaning) applied to the
# settings container while a load error is active: the tabs are greyed out
# and inert, making the banner's buttons the only affordance. The greying
# is an affordance only -- programmatic changes bypass CSS -- so the
# selection observer separately refuses to re-derive badges while an error
# is active.
_INERT_CLASS = "stellarphot-settings-inert"
_INERT_CSS = f"<style>.{_INERT_CLASS} {{pointer-events: none; opacity: 0.5;}}</style>"

# Banner border colors: amber while a load error is unresolved, a calm
# light green once every problem is resolved or only warnings remain.
_ERROR_BORDER = "2px solid #ffc107"
_CALM_BORDER = "2px solid #a3cfbb"


@dataclass
class _BannerEntry:
    """
    One message shown in the `ReviewSettings` banner.
    """

    # What kind of incident this entry reports; drives which banner
    # buttons are shown while the entry is active and how its resolved
    # wording is composed. One of "unreadable", "conflict", "oserror",
    # or "warning".
    kind: str
    # Composed, already-escaped HTML for the message, in its active wording.
    message: str
    # Paths (relative to the working directory) the incident is about;
    # used to look up -- and purge -- recorded backups when composing the
    # resolved wording.
    files: tuple = ()
    # True when the most recent refresh no longer reproduces the problem;
    # the message is then shown in a resolved wording instead of being
    # removed, so the user still learns e.g. that a file was set aside
    # as .bak.
    resolved: bool = False


class ReviewSettings(ipw.VBox):
    """
    Widget to preview the saved settings in the working directory. It displays one
    tab or accordion for each type of setting being reviewed.

    This widget does a bunch of automatic saving and loading behind the scenes:

    1. When the widget is created, it loads the settings from the working directory, if
       there are any. Settings loaded this way are marked as "need review" to remind the
       user they might want to take a look.
    2. When the widget is created, any of the saveable settings are set to the default
       for that setting and then saved to the working directory, with the tab markked as
       "needs review".
    2. When the user clicks the save button for a setting that displays one,
       the settings are saved to the working directory settings.
    3. When the user selects a setting from the settings that have a dropdown, the
       selected setting is saved. Currently those settings are Camera, Observatory, and
       PassbandMap, but the definitive list is given in
       `stellarphot.settings.ChooseOrMakeNew._known_types`.
    4. Creating a new one of those saveable settings also saves it to the working
       directory settings.

    If the saved settings in the working directory cannot be read, or reading them
    generates warnings, a banner describing the problem is displayed above the
    settings. While a load error is active -- which, since construction and
    every save cure such problems automatically, means the settings files
    changed outside this widget -- the settings below the banner are greyed
    out and the banner's buttons are the only way forward; once the problem
    is resolved (or when there are only warnings) the banner turns a calm
    green and can be dismissed with its close button.
    """

    def __init__(self, settings, style="tabs", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Banner for messages about problems loading saved settings. It is
        # hidden unless there is a message to show. Which of its buttons
        # are visible -- and whether it can be dismissed at all -- depends
        # on the state of the entries; see _update_banner.
        self._banner_html = ipw.HTML(layout=ipw.Layout(flex="1 1 auto"))
        self._banner_dismiss = ipw.Button(
            description="✕",
            tooltip="Dismiss these messages",
            # Sizing the button to its label rather than to a fixed width:
            # JupyterLab's button padding eats most of a 2.5em box, so the
            # glyph overflows and CSS text-overflow appends an ellipsis,
            # making the ✕ look like it has a period after it. flex keeps the
            # button from shrinking when the message beside it is long.
            layout=ipw.Layout(width="auto", flex="0 0 auto"),
        )
        self._banner_dismiss.on_click(self._dismiss_banner)
        # One-click fix for an unreadable settings file: rename it to .bak
        # and re-save the values the widget is showing. Only shown while an
        # unreadable-file load error is active -- visibility is managed by
        # _update_banner.
        self._banner_fix = ipw.Button(
            description="Set aside broken file(s) and keep the values shown",
            tooltip=(
                "Rename the unreadable settings file(s) to a .bak backup "
                "and save the values currently displayed below"
            ),
            button_style="warning",
            layout=ipw.Layout(width="auto", flex="0 0 auto"),
        )
        self._banner_fix.layout.display = "none"
        self._banner_fix.on_click(self._fix_unreadable_files)
        # Resolution for a full/partial conflict: save the displayed values
        # (which come from the full settings file), preserving the losing
        # partial file as a .bak backup. Only shown while a conflict is
        # active.
        self._banner_keep = ipw.Button(
            description="Keep the values shown",
            tooltip=(
                "Save the values currently displayed below, resolving the "
                "conflict; the conflicting partial settings file is kept "
                "as a .bak backup"
            ),
            button_style="warning",
            layout=ipw.Layout(width="auto", flex="0 0 auto"),
        )
        self._banner_keep.layout.display = "none"
        self._banner_keep.on_click(self._keep_displayed_values)
        # Reload the settings from disk -- the repair-by-hand path, and the
        # only resolution for a load failure nothing in the banner can fix
        # (e.g. a read-only directory). Shown while any load error is
        # active.
        self._banner_reload = ipw.Button(
            description="Reload",
            tooltip="Reload the settings from the working directory",
            button_style="warning",
            layout=ipw.Layout(width="auto", flex="0 0 auto"),
        )
        self._banner_reload.layout.display = "none"
        self._banner_reload.on_click(self._reload)
        self._banner = ipw.HBox(
            [
                self._banner_html,
                self._banner_fix,
                self._banner_keep,
                self._banner_reload,
                self._banner_dismiss,
            ],
            layout=ipw.Layout(
                align_items="center",
                border=_ERROR_BORDER,
                padding="0.25em 1em",
            ),
        )
        self._banner.layout.display = "none"
        # ``_banner_entries`` maps a stable key (``_LOAD_ERROR_KEY``, or a
        # warning's own text) to a `_BannerEntry`; insertion order is
        # display order. ``_dismissed`` is the set of dismissed keys.
        self._banner_entries = {}
        self._dismissed = set()
        # ``_backups_made`` accumulates, across refreshes, the backups the
        # save machinery reports having made (original path -> backup
        # path), so resolved banner messages can name the actual .bak
        # file. ``_saver`` is the PhotometryWorkingDirSettings instance
        # every save goes through; created below, after the early refresh.
        self._backups_made = {}
        self._saver = None

        # Get a copy of whatever settings may have already been saved.
        self._refresh()

        self._setting_widgets = []
        self._plain_names = []

        self._settings = settings

        # One shared instance for every save this widget makes, so the
        # backups those saves create are recorded somewhere the banner can
        # find them (see _collect_backups).
        self._saver = PhotometryWorkingDirSettings()

        for setting in settings:
            # Track whether we are using the ChooseOrMakeNew or not
            is_choose_or_make_new = False
            if setting.__name__ in ChooseOrMakeNew._known_types:
                widget = ChooseOrMakeNew(setting.__name__)
                val_to_set = widget._choose_existing
                is_choose_or_make_new = True
            else:
                widget = ui_generator(setting)
                val_to_set = widget

            _add_saving_to_widget(
                widget,
                wd_settings=self._saver,
                on_save_error=self._note_save_failure,
            )
            name = to_snake(setting.__name__)
            plain_name = " ".join(name.split("_"))
            self._plain_names.append(plain_name)
            self._setting_widgets.append(SettingWithTitle(plain_name, widget))

            # This should be either a valid object or None
            saved_value = getattr(self._current_settings, name)

            if saved_value is not None:
                try:
                    if is_choose_or_make_new:
                        # Set to None first to ensure there is a change in the value
                        # when we set it to saved_value.
                        val_to_set.value = None
                    val_to_set.value = saved_value
                except tr.TraitError as e:
                    # It can happen, while testing, that a setting gets saved to a local
                    # directory but is no longer in the saved settings for Camera, etc.
                    # We cannot fix that here, so raise a clearer error.
                    raise ValueError(
                        f"The {name} setting saved in the working directory is not "
                        f"consistent with the list of {name} items that are saved "
                        "in your permanent settings. Please fix this manually "
                        f"by editing your saved {name} settings or by deleting the "
                        "working directory settings."
                    ) from e
                # Add symbol to title to indicate that the setting needs
                # review. The widget's badge is the single source of truth;
                # this explicit write lands after the save side effects
                # above, so it wins over whatever those observers set.
                self._setting_widgets[-1].badge = SaveStatus.SETTING_SHOULD_BE_REVIEWED

            elif is_choose_or_make_new:
                if len(widget._choose_existing.options) > 1:
                    # There is also one already-saved choice, so we load it.
                    # If we are using the ChooseOrMakeNew widget, we need to set the
                    # value of the widget to the default item to trigger the save.
                    # To do that, first set the value to None and then set the value
                    # back to the default item.
                    default_item = val_to_set.value
                    val_to_set.value = None
                    val_to_set.value = default_item
                    self._setting_widgets[-1].badge = (
                        SaveStatus.SETTING_SHOULD_BE_REVIEWED
                    )
                else:
                    # This setting needs to be made, not reviewed
                    self._setting_widgets[-1].badge = SaveStatus.SETTING_NOT_SAVED
            else:
                # We got here because there was not a setting saved in the working
                # directory, and this is not a ChooseOrMakeNew, which might have
                # settings saved at the user level.

                # Two possibilities:
                # 1. The setting can be made from default values but needs to be
                #    reviewed. Status should be "needs review".
                # 2. The setting cannot be made from default values. Status should be
                #    "not saved".
                try:
                    val_to_set.model()
                except ValidationError:  # pragma: no cover
                    # This should never happen with the code base as of 2024-06-27 but
                    # might in the future.
                    self._setting_widgets[-1].badge = SaveStatus.SETTING_NOT_SAVED
                else:
                    # This setting can be made from default values, so we save it to the
                    # working directory.
                    val_to_set.savebuttonbar.bn_save.click()
                    self._setting_widgets[-1].badge = (
                        SaveStatus.SETTING_SHOULD_BE_REVIEWED
                    )

        if style == "tabs":
            self._container = ipw.Tab()
        else:
            self._container = ipw.Accordion()

        self._container.children = self._setting_widgets
        self._container.titles = self._make_titles()

        # The style sheet that lets _update_banner grey out and disable the
        # settings container while a load error is active.
        self._inert_style = ipw.HTML(_INERT_CSS)
        self.children = [self._inert_style, self._banner, self._container]

        # Set up an observer to run when a tab is selected
        self._container.observe(self._observe_tab_selection, names="selected_index")

        # Set up observer for each of the widget badges
        for widget in self._setting_widgets:
            widget.observe(self._observe_badge_change, names="badge")

        # Construction itself may have changed what is on disk -- default-
        # constructible settings were autosaved above, and setting a
        # ChooseOrMakeNew value triggers its save observer -- so refresh
        # once more so that current_settings (and the banner) reflect the
        # post-construction state instead of the pre-save snapshot taken at
        # the top of this method. This refresh also harvests the backups
        # those construction-time saves recorded on self._saver, so a
        # problem cured during construction is reported with the actual
        # .bak name.
        self._refresh()

    def _make_titles(self):
        """
        Make titles from badges and plain titles.
        """
        return [
            f"{badge} {plain}"
            for badge, plain in zip(self.badges, self._plain_names, strict=True)
        ]

    @property
    def badges(self):
        """
        The badge of each setting widget, in display order. The widgets
        themselves are the single source of truth; this is a read-only
        view of their badges.
        """
        return [widget.badge for widget in self._setting_widgets]

    @property
    def current_settings(self):
        """
        The settings as of the last refresh; a snapshot rather than a live
        read from disk.
        """
        return self._current_settings

    def _refresh(self):
        """
        Reload the settings from the working directory, updating the banner
        with any problems the load encounters.

        Note that the underlying load is not a pure read: it deletes a
        partial settings file that exactly duplicates the full settings
        file, so a refresh can tidy the working directory as a side
        effect.
        """
        self._current_settings = self._load_working_dir_settings()

    def _collect_backups(self):
        """
        Merge the backups recorded by the load-side and save-side
        `PhotometryWorkingDirSettings` instances into ``_backups_made``,
        which survives the per-refresh rebinding of ``_wd_settings``.
        Every instance builds its paths relative to ``Path(".")``, so the
        keys from different instances merge correctly.
        """
        for source in (
            getattr(self, "_wd_settings", None),
            getattr(self, "_saver", None),
        ):
            if source is not None:
                self._backups_made.update(source.backups_made)
                # Drain the source once harvested: ``_backups_made`` is the
                # single accumulator. Re-merging an already-harvested record
                # on a later refresh would resurrect entries the
                # new-incident purge dropped and let an old record from one
                # instance overwrite a newer backup made through the other.
                source._backups_made.clear()

    @property
    def _load_error_active(self):
        """
        True while the banner holds an unresolved load-error entry, i.e.
        while the on-disk settings are broken rather than merely missing.
        """
        entry = self._banner_entries.get(_LOAD_ERROR_KEY)
        return entry is not None and not entry.resolved

    def _load_working_dir_settings(self):
        """
        Load settings from the working directory, routing any warnings the load
        generates, and any failure to read an existing settings file, into the
        banner instead of silently discarding them.

        At most one load-error message can exist per refresh, so it is keyed
        by the fixed ``_LOAD_ERROR_KEY`` rather than by anything derived
        from the error text; a warning is keyed by its own (stable) text.
        Keying the load error on a fixed value, rather than on the error
        text, matters because the sentences composed around it -- and even
        which file is reported as unreadable -- can change between refreshes
        for the same underlying incident (e.g. an autosave during widget
        construction renames one bad file to ``.bak``, so a later refresh
        raises an error about the *other* file instead). When a refresh
        reproduces an already-shown key, its wording is updated in place
        instead of a near-duplicate being appended.

        Banner messages are also sticky: once shown, an entry stays in
        ``self._banner_entries`` until the user dismisses it, even if a
        later refresh no longer produces its key (e.g. an automatic save
        during widget construction renames the offending file to ``.bak``,
        curing the problem before the user ever saw the banner). Such
        entries are marked resolved instead of being removed -- see
        ``_update_banner`` -- and only ``_dismiss_banner`` removes them.
        """
        # Harvest backup records before this refresh rebinds
        # self._wd_settings below, or they would be lost with the old
        # instance.
        self._collect_backups()
        wd_settings = PhotometryWorkingDirSettings()
        current = []
        try:
            with warnings.catch_warnings(record=True) as recorded:
                # Only PhotometrySettingsWarning -- the category for
                # user-actionable problems with saved settings files -- is
                # recorded for display. Plain UserWarnings from libraries on
                # the load path (e.g. pydantic serializer warnings, astropy's
                # AstropyUserWarning) deliberately stay out of the banner,
                # along with deprecation and other internal noise.
                warnings.simplefilter("ignore")
                warnings.simplefilter("always", PhotometrySettingsWarning)
                loaded = wd_settings.load()
        except (ValueError, OSError) as e:
            # load() parses both files before raising, so whichever of these
            # is not None was actually readable and can be used as a
            # fallback instead of falling all the way back to empty settings.
            # OSError can escape load() when tidying a duplicate partial
            # settings file fails (e.g. in a read-only directory); it is
            # reported like any other load problem instead of crashing
            # widget construction.
            loaded = (
                wd_settings.settings
                or wd_settings.partial_settings
                or PartialPhotometrySettings()
            )
            settings_file_exists = (
                wd_settings.settings_file.exists()
                or wd_settings.partial_settings_file.exists()
            )
            if settings_file_exists:
                error_lines = str(e).splitlines()

                # A file is unreadable, as opposed to merely conflicting with
                # its counterpart, exactly when it existed on disk but could
                # not be parsed -- which load() records in these properties.
                unreadable_full = wd_settings.full_settings_unreadable
                unreadable_partial = wd_settings.partial_settings_unreadable
                if unreadable_full or unreadable_partial:
                    files = []
                    if unreadable_full:
                        files.append(wd_settings.settings_file)
                    if unreadable_partial:
                        files.append(wd_settings.partial_settings_file)
                    joined_names = " and ".join(html.escape(f.name) for f in files)
                    file_word = "files" if len(files) > 1 else "file"
                    # Construction and every save cure unreadable files
                    # automatically, so an *active* error provably means
                    # the file changed outside this widget.
                    message = (
                        f"The settings {file_word} {joined_names} changed "
                        "outside this widget and could not be read."
                    )
                    kind = "unreadable"
                elif isinstance(e, ValueError):
                    # Both files were readable, so this is the
                    # conflicting-values case. The partial settings file is
                    # the one preserved as .bak when a save resolves the
                    # conflict, so it is the file this incident is about.
                    full_name = html.escape(wd_settings.settings_file.name)
                    partial_name = html.escape(wd_settings.partial_settings_file.name)
                    message = (
                        f"The files {full_name} and {partial_name} changed "
                        "outside this widget and now disagree; the values "
                        f"shown come from {full_name}."
                    )
                    kind = "conflict"
                    files = [wd_settings.partial_settings_file]
                else:
                    # OSError with both files readable: the load failed
                    # while deleting the partial settings file, which it
                    # only does when that file is an exact duplicate of the
                    # full settings file (e.g. in a read-only directory).
                    full_name = html.escape(wd_settings.settings_file.name)
                    partial_name = html.escape(wd_settings.partial_settings_file.name)
                    message = (
                        f"The file {partial_name} is an exact duplicate of "
                        f"{full_name} but could not be deleted. It is safe "
                        "to delete by hand; make this directory writable "
                        "or delete the file, then click Reload."
                    )
                    kind = "oserror"
                    files = []

                # The full error is long -- dozens of lines for a pydantic
                # ValidationError -- so all of it, first line included,
                # goes into a collapsed details element.
                if detail := "\n".join(error_lines):
                    message += (
                        "<details><summary>Full error</summary>"
                        f"<pre>{html.escape(detail)}</pre></details>"
                    )
                current.append((_LOAD_ERROR_KEY, kind, message, tuple(files)))
        # Add the warnings outside the try/except so they are reported even
        # when load() raises after emitting them; ``recorded`` stays bound
        # because catch_warnings exits normally during exception propagation.
        # A warning's text is stable across refreshes, so it is its own key.
        for warning in recorded:
            warning_text = html.escape(str(warning.message))
            current.append((warning_text, "warning", warning_text, ()))

        # Retain the instance that performed this load: its unreadable
        # flags identify the files an active load error is about, which
        # the banner's fix button consults, and its backup records are
        # harvested by the next refresh.
        self._wd_settings = wd_settings

        current_keys = {key for key, *_ in current}
        # A dismissal is remembered only for as long as its key keeps being
        # produced, so it cannot grow without bound with stale entries; if a
        # problem goes away and later recurs, showing it again is correct.
        self._dismissed &= current_keys
        for key, kind, msg, files in current:
            if key in self._dismissed:
                # Only warnings can be dismissed while still active, and a
                # warning's text is stable, so a dismissed key that is
                # still produced is the same incident. Keep it dismissed.
                continue
            previous = self._banner_entries.get(key)
            if key == _LOAD_ERROR_KEY and (previous is None or previous.resolved):
                # A fresh incident: a backup recorded before this problem
                # was detected cannot be what cures it, so drop any stale
                # records for the files involved lest a resolved message
                # later name a backup that predates the problem.
                for path in files:
                    self._backups_made.pop(path, None)
            # Assignment updates an already-shown key in place instead of
            # appending a duplicate.
            self._banner_entries[key] = _BannerEntry(
                kind=kind,
                message=msg,
                files=files,
            )

        # A sticky entry whose key this refresh no longer produces is marked
        # resolved rather than removed; a recurrence un-resolves it (the
        # assignment above recreates the entry with resolved=False).
        for key, entry in self._banner_entries.items():
            entry.resolved = key not in current_keys

        self._update_banner()
        return loaded

    def _compose_resolved_text(self, entry):
        """
        Compose the wording for a resolved entry from what is actually
        known: backups recorded for the entry's files are named exactly;
        with no recorded backup the problem is reported as no longer
        detected, with no speculation about how it was cured (the user may
        simply have repaired or removed the file by hand).
        """
        if entry.kind == "unreadable":
            moved = {
                path: self._backups_made[path]
                for path in entry.files
                if path in self._backups_made
            }
            if moved:
                originals = " and ".join(html.escape(p.name) for p in moved)
                backups = " and ".join(html.escape(p.name) for p in moved.values())
                was_were = "originals were" if len(moved) > 1 else "original was"
                return (
                    f"There was a problem with {originals}; "
                    f"the {was_were} saved as {backups}."
                )
            names = " and ".join(html.escape(p.name) for p in entry.files)
            return f"A problem with {names} is no longer detected."
        if entry.kind == "conflict":
            full_name = html.escape(self._wd_settings.settings_file.name)
            partial_name = html.escape(self._wd_settings.partial_settings_file.name)
            text = (
                f"The files {full_name} and {partial_name} disagreed; "
                "the conflict has been resolved"
            )
            backup = next(
                (
                    self._backups_made[path]
                    for path in entry.files
                    if path in self._backups_made
                ),
                None,
            )
            if backup is not None:
                text += (
                    ", and the conflicting partial file was saved as "
                    f"{html.escape(backup.name)}."
                )
            else:
                text += "."
            return text
        # Warnings and OSError entries resolve generically.
        return (
            "<em>No longer detected as of the latest reload:</em> " f"{entry.message}"
        )

    def _update_banner(self):
        """
        Render the banner from the current entries.

        While a load error is active -- which, since construction and every
        save cure such problems automatically, means the settings changed
        outside this widget -- the banner is modal: the settings below are
        greyed out and inert, the dismiss button is hidden, and the
        banner's own buttons (which of them depends on the kind of error)
        are the only affordance. Once every entry is resolved, or when only
        warnings remain, the banner turns a calm green and is dismissable.

        A resolved message -- one whose problem the latest reload no longer
        reproduces -- is kept visible in a resolved wording, so the banner
        does not keep asserting a stale problem while the user still
        learns e.g. that a file was set aside as ``.bak``.
        """
        error_entry = self._banner_entries.get(_LOAD_ERROR_KEY)
        error_active = error_entry is not None and not error_entry.resolved

        if self._banner_entries:
            # A div rather than a p because a message may contain a details
            # element, which is not allowed inside a p.
            parts = []
            for entry in self._banner_entries.values():
                if entry.resolved:
                    parts.append(f"<div>✓ {self._compose_resolved_text(entry)}</div>")
                else:
                    parts.append(f"<div>⚠️ {entry.message}</div>")
            self._banner_html.value = "".join(parts)
            self._banner.layout.display = "flex"
            self._banner.layout.border = _ERROR_BORDER if error_active else _CALM_BORDER
        else:
            self._banner_html.value = ""
            self._banner.layout.display = "none"

        # Which buttons apply depends on the kind of active error; none of
        # them apply to a resolved entry or a warnings-only banner, and an
        # active error cannot be dismissed -- resolving it is the only way
        # forward.
        active_kind = error_entry.kind if error_active else None
        self._banner_fix.layout.display = "" if active_kind == "unreadable" else "none"
        self._banner_keep.layout.display = "" if active_kind == "conflict" else "none"
        self._banner_reload.layout.display = "" if error_active else "none"
        self._banner_dismiss.layout.display = "none" if error_active else ""

        # Grey out and disable the settings while an error is active. The
        # container does not exist yet during the refresh at the top of
        # __init__; the refresh at the end of construction re-runs this.
        container = getattr(self, "_container", None)
        if container is not None:
            if error_active:
                container.add_class(_INERT_CLASS)
            else:
                container.remove_class(_INERT_CLASS)

    def _dismiss_banner(self, _=None):
        """
        Dismiss every message currently shown, hiding the banner. There is
        no per-message dismissal: the single close button clears all
        messages at once, including any that arrived after the one the
        user meant to dismiss.

        Only an entry that is still active -- with the dismiss button
        hidden while a load error is active, that means a warning -- is
        recorded as dismissed. A resolved entry is simply removed, so a
        recurrence of the underlying problem is always shown fresh.
        """
        for key, entry in self._banner_entries.items():
            if not entry.resolved:
                self._dismissed.add(key)
            # The entry's story is over; its backup records must not leak
            # into some future incident's resolved wording.
            for path in entry.files:
                self._backups_made.pop(path, None)
        self._banner_entries = {}
        self._update_banner()

    def _note_save_failure(self, e, lead):
        """
        Route a failed save or rename (e.g. a read-only directory) into
        the banner instead of letting it raise out of a click handler or
        observer.

        Parameters
        ----------
        e : OSError
            The exception describing the failure.
        lead : str
            Plain-text lead-in for the message, e.g. "The displayed
            values could not be saved".
        """
        first_line = html.escape(str(e).splitlines()[0])
        message = (
            f"{lead} ({first_line}); check that this directory and its "
            "files are writable."
        )
        entry = self._banner_entries.get(_LOAD_ERROR_KEY)
        if entry is None:
            self._banner_entries[_LOAD_ERROR_KEY] = _BannerEntry(
                kind="oserror", message=message
            )
        else:
            # Reuse the existing entry (keeping its kind, so e.g. the fix
            # button stays available for a retry after the user makes the
            # directory writable). Transient by design: the next refresh
            # recomposes the standard active wording.
            entry.message = message
            entry.resolved = False
        self._update_banner()

    def _fix_unreadable_files(self, _=None):
        """
        Set aside the settings file(s) the last load found unreadable, save
        the values currently displayed so the directory loads cleanly
        again, and refresh so the banner reports the outcome -- naming the
        actual backups from the records the save machinery keeps.
        """
        try:
            self._wd_settings.set_aside_unreadable()
        except OSError as e:
            self._note_save_failure(
                e, "The broken settings file(s) could not be set aside"
            )
            return
        self._save_displayed_values()
        self._refresh()

    def _keep_displayed_values(self, _=None):
        """
        Resolve a full/partial conflict by saving the values currently
        displayed -- which came from the full settings file -- so the
        losing partial settings file is preserved as a .bak backup.
        """
        self._save_displayed_values()
        self._refresh()

    def _reload(self, _=None):
        """
        Reload the settings from disk. When the user has repaired or
        removed the problem file(s) by hand, this resolves the banner and
        un-greys the settings.
        """
        self._refresh()

    def _save_displayed_values(self):
        """
        Save the values currently displayed in the setting widgets to the
        working directory in one aggregate save, skipping any widget whose
        value is incomplete or invalid.
        """
        values = {}
        for setting_widget in self._setting_widgets:
            widget = setting_widget._widget
            if hasattr(widget, "_choose_existing"):
                name = to_snake(widget._item_type_name)
            else:
                name = to_snake(widget.model.__name__)
            try:
                PartialPhotometrySettings(**{name: widget.value})
            except ValidationError:
                # Mirrors the per-widget autosave: an in-progress or
                # incomplete tab is skipped.
                continue
            values[name] = widget.value
        if not values:
            return
        # The banner is the display channel for PhotometrySettingsWarning,
        # so the warning save() re-emits from its bookkeeping load is
        # silenced rather than printed.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PhotometrySettingsWarning)
            try:
                self._wd_settings.save(PartialPhotometrySettings(**values), update=True)
            except OSError as e:
                self._note_save_failure(e, "The displayed values could not be saved")

    def _resave_chooser_value(self, setting_widget):
        """
        Re-save a `ChooseOrMakeNew`'s displayed value when its setting has
        disappeared from disk. A chooser shows no save button while an
        existing item is displayed, so stamping its tab not-saved would
        leave the user with no way to fix it from the widget; instead the
        displayed value is saved again, mirroring the save that
        construction performs. A chooser whose displayed value is
        incomplete falls back to the not-saved stamp.
        """
        widget = setting_widget._widget
        name = to_snake(widget._item_type_name)
        try:
            pps = PartialPhotometrySettings(**{name: widget.value})
        except ValidationError:
            # No complete value to save; prompt_save is a no-op for a
            # chooser, so this is just the not-saved stamp.
            setting_widget.prompt_save()
            setting_widget.badge = SaveStatus.SETTING_NOT_SAVED
            return
        # The banner is the display channel for PhotometrySettingsWarning,
        # so the warning save() re-emits from its bookkeeping load is
        # silenced rather than printed.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PhotometrySettingsWarning)
            try:
                self._saver.save(pps, update=True)
            except OSError as e:
                self._note_save_failure(e, "The displayed values could not be saved")
                return
        self._refresh()
        setting_widget.badge = SaveStatus.SETTING_IS_SAVED

    def _observe_tab_selection(self, change):
        """
        Observer for the tab or accordion selection.
        """
        # Once the user has clicked on the tab, the status badge for the tab
        # should stay as-is when the widget reports the setting saved and it
        # still exists on disk. Any other badge is re-derived by comparing
        # the widget value to the saved value, so a badge latched at
        # NOT_SAVED can recover when the setting gets saved by something
        # other than this widget's own observers, a needs-review badge
        # settles once the user has looked at the tab -- and a positive
        # badge whose setting has disappeared from disk drops back to
        # NOT_SAVED.

        new_selected = change["new"]

        # Reload from disk on every selection change so the snapshot and
        # the banner stay current even when the selected tab's badge needs
        # no re-deriving below (e.g. a settings file corrupted after
        # construction while every tab reads as saved).
        self._refresh()

        # An accordion reports a selection of None when every section is
        # collapsed; there is nothing to update in that case.
        if new_selected is None:
            return

        # While a load error is active, the on-disk state is broken rather
        # than missing: the banner is the sole indicator of that, and the
        # settings below it are greyed out. Badges are never re-derived
        # from the broken (fallback) snapshot. The CSS inertness is an
        # affordance only -- programmatic selection bypasses it -- so this
        # early return is the real guard.
        if self._load_error_active:
            return

        setting_widget = self._setting_widgets[new_selected]
        setting_badge = setting_widget.badge

        # A ChooseOrMakeNew that is making a new item or editing one manages
        # its own badge through _choose_existing_observer; a disk comparison
        # here would report the still-saved on-disk value as SAVED while the
        # user is mid-edit, so trust its badge unconditionally. The chooser
        # sets its badge on the next value change anyway.
        chooser = setting_widget._widget
        mid_interaction = (
            isinstance(chooser, ChooseOrMakeNew) and chooser.is_mid_interaction
        )
        if not mid_interaction:
            snake_name = to_snake(setting_widget._autoui_widget.model.__name__)
            disk_value = getattr(self.current_settings, snake_name)
            if disk_value is None and isinstance(chooser, ChooseOrMakeNew):
                # A chooser is fixed by re-saving its displayed value, not
                # by a save button it keeps hidden.
                self._resave_chooser_value(setting_widget)
            elif disk_value is None:
                # No saved value on disk: even a positive badge is stale
                # here (e.g. the settings file was deleted outside this
                # widget) -- the mirror image of the latched-NOT_SAVED
                # recovery below.
                setting_widget.prompt_save()
                setting_widget.badge = SaveStatus.SETTING_NOT_SAVED
            elif setting_badge == SaveStatus.SETTING_IS_SAVED:
                # Saved, and the setting exists on disk: nothing to
                # re-derive.
                pass
            else:
                # Any other badge (None, NOT_SAVED, or needs-review) is
                # re-derived from the snapshot refreshed above rather than
                # trusted, so a save made outside this widget's own
                # observers (e.g. by another widget writing the settings
                # file) is picked up, and a needs-review badge settles to
                # SAVED once the user has looked at the tab.
                try:
                    value_from_widget = (
                        setting_widget._autoui_widget.model.model_validate(
                            setting_widget._autoui_widget.value
                        )
                    )
                except ValidationError:
                    # The widget holds an incomplete/invalid value; whatever
                    # is on disk, it does not match what is displayed. Also
                    # prompt for a save, so a red badge never coexists with
                    # a green save light -- the invalid value keeps the
                    # save button disabled until it is fixed.
                    setting_widget.prompt_save()
                    setting_widget.badge = SaveStatus.SETTING_NOT_SAVED
                else:
                    # A valid widget value that differs from disk deliberately
                    # leaves the badge unchanged, as before this re-derivation
                    # existed.
                    if disk_value == value_from_widget:
                        # The setting is on disk again, so drop any prompt a
                        # previous selection left on the save bar.
                        setting_widget.clear_save_prompt()
                        setting_widget.badge = SaveStatus.SETTING_IS_SAVED

        self._container.titles = self._make_titles()

    def _observe_badge_change(self, _=None):
        """
        Observer for the badge of any setting widget: the titles are
        re-derived from the badges, which live on the widgets themselves.
        """
        self._container.titles = self._make_titles()


def _add_saving_to_widget(setting_widget, wd_settings=None, on_save_error=None):
    """
    Add an observer to a widget that autosaves the settings for that widget to
    the working directory.

    Parameters
    ----------
    setting_widget : ChooseOrMakeNew
        The widget to add the observer to.

    wd_settings : `~stellarphot.settings.PhotometryWorkingDirSettings`, optional
        The instance to save through. By default a fresh instance is
        constructed, which preserves standalone use of this function;
        `ReviewSettings` passes its shared instance so that the backups
        these saves make are recorded where the banner can report them.

    on_save_error : callable, optional
        Called as ``on_save_error(exception, lead_text)`` when the save
        raises `OSError` (e.g. in a read-only directory). When not
        provided, the error propagates.
    """
    if wd_settings is None:
        wd_settings = PhotometryWorkingDirSettings()

    # Define name here so that it is available in the save_wd function. Its
    # value will be set in the if/elif block below.
    name = ""

    def save_wd(_=None):
        try:
            pps = PartialPhotometrySettings(**{name: setting_widget.value})
        except ValidationError:
            # This can happen while making a new item, or while in the process of
            # editing one
            return
        # We have a validated setting so save it. The ReviewSettings banner
        # is the display channel for PhotometrySettingsWarning, and these
        # autosaves fire once per setting, so the warning save() re-emits
        # from its bookkeeping load is silenced here rather than printed
        # once per autosaved setting. Other warning categories are
        # unaffected.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PhotometrySettingsWarning)
            try:
                wd_settings.save(pps, update=True)
            except OSError as e:
                if on_save_error is None:
                    raise
                on_save_error(e, "The displayed values could not be saved")

    if hasattr(setting_widget, "_choose_existing"):
        setting_widget._choose_existing.observe(save_wd, "value")
        name = to_snake(setting_widget._item_type_name)
    elif hasattr(setting_widget, "savebuttonbar"):
        setting_widget.savebuttonbar.fns_onsave_add_action(save_wd)
        name = to_snake(setting_widget.model.__name__)
    else:
        raise ValueError(
            f"The widget {setting_widget} is not a recognized type of widget."
        )


class PhotometryRunner(ipw.VBox):
    """
    Class to run the photometry notebook on a folder of images.

    Parameters
    ----------
    photometry_notebook_name : str, optional
        Name of the photometry notebook to run. Default is "photometry_run.ipynb".

    Notes
    -----

    When this widget is run, it will created a new notebook in the current directory
    called ``photometry_notebook_name``, which will perform the photometry.
    """

    def __init__(
        self, photometry_notebook_name="photometry_run.ipynb", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.photometry_notebook_name = photometry_notebook_name
        self.fitsopen = FitsOpener(
            title=(
                "Choose an image of the object of interest to you.</br>Photometry "
                "will be performed on all images of that object in the folder "
                "containing the selected image."
            )
        )
        self.info_box = ipw.HTML()
        self.run_output = ipw.Output()
        self.confirm = Confirm(message="Is this correct?")
        self.children = (
            self.fitsopen.file_chooser,
            self.info_box,
            self.confirm,
            self.run_output,
        )
        self.fitsopen.file_chooser.observe(self._file_chosen, "_value")
        self.confirm.observe(self._confirmation, "value")
        self.run_settings = None

    def _file_chosen(self, _):
        self.run_settings = PhotometryRunSettings(
            directory_with_images=self.fitsopen.path.parent,
            object_of_interest=self.fitsopen.header["object"],
        )
        self.info_box.value = (
            "<h2>" + self.info_message + "</br>Is this correct?" + "</h2>"
        )
        self.confirm.show()

    @property
    def info_message(self):
        return (
            f"Photometry will be performed on all images of the object "
            f"'<code>{self.run_settings.object_of_interest}</code>' in the "
            f"folder '<code>{self.run_settings.directory_with_images}</code>'"
        )

    def _confirmation(self, change=None):
        if change["new"]:
            # User said yes

            # Update informational message
            self.info_box.value = (
                "<h2>" + self.info_message + "</br>Photometry is running..." + "</h2>"
            )
            template_nb = get_pkg_data_filename(
                "photometry_runner.ipynb", package="stellarphot.notebooks"
            )

            with self.run_output:
                pm.execute_notebook(
                    template_nb,
                    self.photometry_notebook_name,
                    parameters=self.run_settings.model_dump(mode="json"),
                )
        else:
            # User said no, so reset to initial state.
            self.fitsopen.file_chooser.reset()
            self.info_box.value = ""
            self.run_settings = None


class TessPhotometrySetup(ipw.VBox):
    """
    Widget for getting some photometry inputs for a TESS Object of Interest (TOI).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define the children of self
        self.header = ipw.HTML("<h2>Enter the TIC ID of the exoplanet candidate</h2>")
        self.drop_label = ipw.HTML("How would you like to specify the TIC ID?")
        self.drop = ipw.Dropdown(
            options=(("Type in TIC ID", 0), ("Choose an image", 1))
        )
        self.tic_id_entry = ipw.IntText(description="TIC ID")
        self.fits_opener = FitsOpener()
        self.confirm = Confirm()
        self.spinner = Spinner(message="<h3>Downloading TIC info...</h3>")
        self.all_done = ipw.HTML("<h2>Done! Files have been written.</h2>")

        # Set up observers
        self.drop.observe(self.watch_drop(), names="value")
        self.tic_id_entry.observe(self.watch_tic_id_text_box(), names="value")
        self.fits_opener.file_chooser.observe(self.watch_fits_opener(), names="_value")
        self.confirm.observe(self.watch_confirmation(), names="value")

        # Initialize the widget

        # Hide the "all done" message until it is needed
        self.all_done.layout.display = "none"

        # Initialize the TIC ID
        self.tic_id = 0

        # This is correct -- watch_drop returns a function that takes a change dict
        # Option 0 is to type in the TIC ID
        self.watch_drop()({"new": 0})

        # Define the children
        self.children = (
            self.header,
            self.drop_label,
            self.drop,
            self.tic_id_entry,
            self.fits_opener.file_chooser,
            self.confirm,
            self.spinner,
            self.all_done,
        )

    def watch_drop(self):
        def observer(change):
            # Change whether a text box for entering the TIC ID is visible or a
            # file chooser for selecting an image is visible.
            if change["new"] == 0:
                self.tic_id_entry.layout.display = "flex"
                self.fits_opener.file_chooser.layout.display = "none"
            else:
                self.tic_id_entry.layout.display = "none"
                self.fits_opener.file_chooser.layout.display = "flex"

        return observer

    def set_tic_id(self, an_id):
        # The actual setting of the TIC ID is here so that all observers can
        # call the same method.
        self.tic_id = int(an_id)
        # Display the confirmation widget
        self.confirm.message = f"Is the TIC ID {self.tic_id} correct?"
        self.confirm.show()

    def watch_tic_id_text_box(self):
        def observer(change):
            # Just set the TIC ID....
            self.set_tic_id(change["new"])

        return observer

    def watch_fits_opener(self):
        def observer(_):
            # The FitsOpener handles loading the header and setting the object.
            # Here we just need to extract the TIC ID from the object.
            match = TIC_regex.match(self.fits_opener.object)
            self.set_tic_id(match.group("star"))

        return observer

    def watch_confirmation(self):
        def observer(change):
            # Confirm has a bool value, True if the user says yes, False if no.
            # The Confirm widget closes itself, so we don't need to do anything
            # here except to start the spinner, run the setup function, and stop
            # the spinner.
            if change["new"]:
                self.spinner.start()
                tess_photometry_setup(self.tic_id, overwrite=True)
                self.spinner.stop()
                self.all_done.layout.display = "flex"

        return observer


class Spinner(ipw.VBox):
    """
    A spinner widget.
    """

    def __init__(self, *args, spinner_file=None, message="", **kwargs):
        if spinner_file is None:
            spinner_file = get_pkg_data_filename(
                "data/star_spinner.svg", package="stellarphot"
            )
        self._spinner_file = spinner_file
        super().__init__(*args, **kwargs)
        with open(spinner_file) as f:
            self._spinner = ipw.HTML(f.read())
        self._message = ipw.HTML(message)
        self.children = [self._message, self._spinner]
        self.layout.display = "none"
        self.layout.width = "200px"

    @property
    def message(self):
        return self._message.value

    @property
    def spinner_file(self):
        return self._spinner_file

    def start(self):
        self.layout.display = "flex"

    def stop(self):
        self.layout.display = "none"
