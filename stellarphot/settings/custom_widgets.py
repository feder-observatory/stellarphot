# Some settings require custom widgets to be displayed in the GUI. These are defined in
# this module.

# This workaround is for Python < 3.11. It is not needed in Python 3.11 and later.
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        """
        A class that allows for string values in an Enum pre-Python 3.11.
        """


import ipywidgets as ipw
import papermill as pm
import traitlets as tr
from astropy.utils.data import get_pkg_data_filename
from camel_converter import to_snake
from ipyautoui.autoobject import AutoObject
from ipyautoui.custom.iterable import ItemBox
from pydantic import ValidationError

from stellarphot.settings import (
    Camera,
    Observatory,
    PartialPhotometrySettings,
    PassbandMap,
    PhotometryRunSettings,
    PhotometryWorkingDirSettings,
    SavedSettings,
    ui_generator,
)
from stellarphot.settings.fits_opener import FitsOpener

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
            value=(f"Choose a {self._display_name} " "or make a new one")
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
        if not self._widget._making_new and not self._widget._editing:
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

    """

    def __init__(self, settings, style="tabs", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Get a copy of whatever settings may have already been saved.
        try:
            self._current_settings = PhotometryWorkingDirSettings().load()
        except ValueError:
            self._current_settings = PartialPhotometrySettings()

        self._setting_widgets = []
        self._plain_names = []
        self.badges = []

        self._settings = settings

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

            _add_saving_to_widget(widget)
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
                # Add symbol to title to indicate that the setting needs review
                self.badges.append(SaveStatus.SETTING_SHOULD_BE_REVIEWED)

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
                    self.badges.append(SaveStatus.SETTING_SHOULD_BE_REVIEWED)
                else:
                    # This setting needs to be made, not reviewed
                    self.badges.append(SaveStatus.SETTING_NOT_SAVED)
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
                    self.badges.append(SaveStatus.SETTING_NOT_SAVED)
                else:
                    # This setting can be made from default values, so we save it to the
                    # working directory.
                    val_to_set.savebuttonbar.bn_save.click()
                    self.badges.append(SaveStatus.SETTING_SHOULD_BE_REVIEWED)

        # Check that everything is consistent....
        assert len(self.badges) == len(self._plain_names)

        if style == "tabs":
            self._container = ipw.Tab()
        else:
            self._container = ipw.Accordion()

        self._container.children = self._setting_widgets
        self._container.titles = self._make_titles()

        self.children = [self._container]

        # Set up an observer to run when a tab is selected
        self._container.observe(self._observe_tab_selection, names="selected_index")

        # Set up observer for each of the widget badges
        for idx, widget in enumerate(self._setting_widgets):
            widget.observe(self._observe_badge_change(idx), names="badge")

    def _make_titles(self):
        """
        Make titles from badges and plain titles.
        """
        return [
            f"{badge} {plain}"
            for badge, plain in zip(self.badges, self._plain_names, strict=True)
        ]

    @property
    def current_settings(self):
        """
        The current settings in the widget.
        """
        try:
            self._current_settings = PhotometryWorkingDirSettings().load()
        except ValueError:
            self._current_settings = PartialPhotometrySettings()

        return self._current_settings

    def _observe_tab_selection(self, change):
        """
        Observer for the tab or accordion selection.
        """
        # Once the user has clicked on the tab, the status badge for the
        # tab should just be the badge for the widget it holds, if the
        # widget has a badge. Otherwise, compare the widget value to the
        # saved value to determine the badge.

        # Get the index
        new_selected = change["new"]

        setting_badge = self._setting_widgets[new_selected].badge
        if setting_badge is not None:
            self.badges[new_selected] = setting_badge
        else:
            # Check whether the setting is saved or not
            setting_widget = self._setting_widgets[new_selected]

            snake_name = to_snake(setting_widget._autoui_widget.model.__name__)

            disk_value = getattr(self.current_settings, snake_name)
            if disk_value is None:
                # The setting is not saved
                setting_widget = SaveStatus.SETTING_NOT_SAVED
            else:
                # Set the badge to saved if it has been saved.
                value_from_widget = setting_widget._autoui_widget.model.model_validate(
                    setting_widget._autoui_widget.value
                )

                if disk_value == value_from_widget:
                    setting_widget.badge = SaveStatus.SETTING_IS_SAVED

        self._container.titles = self._make_titles()

    def _observe_badge_change(self, index):
        """
        Observer for the badge of a setting widget.
        """

        def observer(change):
            self.badges[index] = change["new"]

            # This should only be called when a badge changes, so we can just
            # update the titles.
            self._container.titles = self._make_titles()

        return observer


def _add_saving_to_widget(setting_widget):
    """
    Add an observer to a widget that autosaves the settings for that widget to
    the working directory.

    Parameters
    ----------
    setting_widget : ChooseOrMakeNew
        The widget to add the observer to.
    """
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
        # We have a validated setting so save it.
        wd_settings.save(pps, update=True)

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
    def __init__(
        self, photometry_notebook_name="photometry_run.ipynb", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.photometry_notebook_name = photometry_notebook_name
        self.fo = FitsOpener(
            title=(
                "Choose any image in the folder of images to do photometry on that "
                "contains the object of interest"
            )
        )
        self.info_box = ipw.HTML()
        self.run_output = ipw.Output()
        self.confirm = Confirm(message="Is this correct?")
        self.children = (
            self.fo.file_chooser,
            self.info_box,
            self.confirm,
            self.run_output,
        )
        self.fo.file_chooser.observe(self._file_chosen, "_value")
        self.confirm.observe(self._confirmation, "value")
        self.run_settings = None

    def _file_chosen(self, _):
        self.run_settings = PhotometryRunSettings(
            directory_with_images=self.fo.path.parent,
            object_of_interest=self.fo.header["object"],
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
            print(template_nb)
            with self.run_output:
                pm.execute_notebook(
                    template_nb,
                    self.photometry_notebook_name,
                    parameters=self.run_settings.model_dump(mode="json"),
                )
        else:
            # User said no, so reset to initial state.
            self.fo.file_chooser.reset()
            self.info_box.value = ""
            self.run_settings = None
