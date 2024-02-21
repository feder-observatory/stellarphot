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
    ui.show_validation = False
    ui.show_null = True
    ui.description = _extract_short_description(model.__doc__)
    return ui
