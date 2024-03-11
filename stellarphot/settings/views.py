from ipyautoui import AutoUi

from .models import _extract_short_description

__all__ = ["ui_generator"]


def ui_generator(model):
    """
    Generate a user interface with ipyautoui with a few default settings.

    Parameters
    ----------
    model : `pydantic.BaseModel` subclass
        The model to generate the user interface for.
    """
    ui = AutoUi(model)

    # validation is really messy looking right now, so suppress display of
    # the validation errors. A green check or red x will still be displayed.
    ui.show_validation = False

    # By default nullable are not shown at all but it seems much easier for
    # the user to understand if they are shown but disabled.
    ui.show_null = True

    # Set the description to the first sentence of the docstring. The default is
    # to use the entire docstring, which is often too long.
    ui.description = _extract_short_description(model.__doc__)

    # Validation is checked every time a value is changed, and the contents of each
    # field are written to the widget if validation passes. In at least one case,
    # the "Observatory" model, this is not helpful because the format of lat/lon is
    # not necessarily what the user entered. Furthermore, you can't edit the value
    # in the widget because it is overwritten every time validation passes. So, for
    # now, we will disable this feature by turning continuous update off for most
    # fields.
    for widget in ui.di_widgets.values():
        if hasattr(widget, "continuous_update"):
            widget.continuous_update = False

    return ui
