# Some settings require custom widgets to be displayed in the GUI. These are defined in
# this module.

import ipywidgets as ipw
import traitlets as tr
from ipyautoui.autoobject import AutoObject

from stellarphot.settings import (
    Camera,
    Observatory,
    PassbandMap,
    SavedSettings,
    ui_generator,
)


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

    def __init__(self, item_type_name, *arg, _testing_path=None, **kwargs):
        if item_type_name not in self._known_types:
            raise ValueError(
                f"Unknown item type {item_type_name}. Must "
                f"be {', '.join(self._known_types)}"
            )
        # Get the widgety goodness from the parent class
        super().__init__(*arg, **kwargs)

        self._saved_settings = SavedSettings(_testing_path=_testing_path)
        self._item_type_name = item_type_name

        # keep track of whether we are editing an existing item
        self._editing = False

        self._display_name = item_type_name.replace("_", " ")

        # Create the child widgets
        self._title = ipw.HTML(
            value=(f"<h2>Choose a {self._display_name} " "or make a new one</h2>")
        )

        self._choose_existing = ipw.Dropdown(description="")

        self._edit_button = ipw.Button(
            description=f"Edit this {self._display_name}",
            # width below was chosen to match the dropdown...would prefer to
            # determine this programmatically but don't know how.
            layout={"width": "300px"},
        )

        self._confirm_edit = Confirm(
            message=f"Replace value of this {self._display_name}?",
        )

        self._item_widget = self._make_new_widget()

        self._edit_button.on_click(self._edit_button_action)

        # Build the main widget
        self.children = [
            self._title,
            self._choose_existing,
            self._edit_button,
            self._confirm_edit,
            self._item_widget,
        ]

        # Set up the dropdown widget
        self._construct_choices()
        # Set the selection to the first choice if there is one
        self._choose_existing.value = self._choose_existing.options[0][1]

        if len(self._choose_existing.options) == 1:
            # There are no items, so we are making a new one
            self._choose_existing.disabled = True
            self._choose_existing.layout.display = "none"
            self._title.value = f"<h2>Make a new {item_type_name}</h2>"
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

        self._item_widget.savebuttonbar.fns_onsave_add_action(self._save_confirmation())

        # Link validation value to save button. The use of directional link
        # is important to make sure the validation state controls
        # the save button state and not the other way around.
        ipw.dlink(
            (self._item_widget.is_valid, "value"),
            (self._item_widget.savebuttonbar.bn_save, "disabled"),
            transform=lambda x: not x,
        )

        # Set up some observers

        # Respond to user interacting with a confirmation widget
        # Hide the save button bar so the user gets the confirmation instead
        self._confirm_edit.widget_to_hide = self._item_widget.savebuttonbar
        # Add the observer
        self._confirm_edit.observe(self._handle_confirmation(), names="value")

        # Respond when user wants to make a new thing
        self._choose_existing.observe(self._handle_selection, names="value")

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
                self._confirm_edit.show()

        return f

    def _construct_choices(self):
        """
        Set up the choices for the selection widget.
        """
        saved_items = self._saved_settings.get_items(self._item_type_name)
        existing_choices = [(k, v) for k, v in saved_items.as_dict.items()]
        existing_choices = sorted(existing_choices, key=lambda x: x[0].lower())
        choices = existing_choices + [(f"Make new {self._display_name}", "none")]
        # This sets the options but doesn't select any of them
        self._choose_existing.options = choices

    def _handle_selection(self, change):
        if change["new"] == "none":

            # We are making a new item...

            # Hide the edit button
            self._edit_button.layout.display = "none"

            # This sets the ui back to its original state when created, i.e.
            # everything is empty.
            self._item_widget._init_ui()
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

        else:
            # Display the selected item...
            self._item_widget.show_savebuttonbar = False
            self._item_widget.disabled = True
            self._item_widget.is_valid.layout.display = "none"
            self._item_widget.value = self._get_item(change["new"].name)

            # Really only applies to PassbandMap, which has nested models,
            # but does no harm in the other cases
            self._set_disable_state_nested_models(self._item_widget, True)

    def _edit_button_action(self, _):
        """
        Handle the edit button being clicked.
        """
        # Replace the display of the edit button with the save button bar...
        self._edit_button.layout.display = "none"
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

        # Thie really only applies to PassbandMap, which has nested models,
        # but does no harm in the other cases (true of both lines below)
        # (and yes, both lines below are needed...this is a bug in ipyautoui,
        #  I think, because open_nested=True isn't respected when we _init_ui.
        #  Forcing a *change* in the value triggers the behavior we want.)
        self._item_widget.open_nested = False
        self._item_widget.open_nested = True

        # Update the current state of the widget
        self._editing = True

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

        if hasattr(top, "disabled") or isinstance(top, AutoObject):
            top.disabled = value
        try:
            for child in top.children:
                self._set_disable_state_nested_models(child, value)
        except AttributeError:
            # No children...
            pass

    def _make_new_widget(self):
        """
        Make a new widget for the item type and set up actions for the save button.
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
                # This will happen if the item already exists
                self._confirm_edit.show()

        def update_choices_and_select_new():
            """
            Update the choices after a new item is saved, update the choices
            and select the new item.
            """
            if not self._editing:
                value_to_select = new_widget.model(**new_widget.value)
                self._construct_choices()
                self._choose_existing.value = value_to_select

        # This is the mechanism for adding callbacks to the save button.
        new_widget.savebuttonbar.fns_onsave_add_action(saver)
        new_widget.savebuttonbar.fns_onsave_add_action(update_choices_and_select_new)
        return new_widget

    def _handle_confirmation(self):
        """
        Handle the confirmation of a save operation.
        """

        # Use a closure here to capture the current state of the widget
        def save_confirm_handler(change):
            """
            This handles interactions with the confirmation widget, which is displayed
            when the user either tries to save a new item with the same name as an
            existing one or tries to save an edited item with the same name as an
            existing one.

            The widget has three possible values: True (yes), False (no), and None

            This widget is called whent the widget value changes, which can happen two
            ways:

            1. The user clicks the "yes" or "no" button, in which case the value will
                be True or False, respectively.
            2. This handler sets the value to None after the user has clicked Yes or No.

            The second case is the reason most of the handler is wrapped in an
            if statement.
            """
            # value of None means the widget has been reset to not answered
            if change["new"] is not None:
                item = self._item_widget.model(**self._item_widget.value)
                if change["new"]:
                    # Use has said yes to updating the item, which we do by
                    # deleting the old one and adding the new one.
                    self._saved_settings.delete_item(item, confirm=True)
                    self._saved_settings.add_item(item)
                    # Rebuild the dropdown list
                    self._construct_choices()
                    # Select the edited item
                    self._choose_existing.value = item
                else:
                    # Use has said no to updating the item, so we just
                    # act as though the user has selected this item.
                    self._handle_selection({"new": item})

                # Reset the confirmation widget to unanswered
                self._confirm_edit.value = None
            # We are done editing regardless of the confirmation outcome
            self._editing = False
            # Bring the edit button back
            self._edit_button.layout.display = "flex"

        return save_confirm_handler

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
        Display the confrimation widget and, if desired, hide the widget it replaces.
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
