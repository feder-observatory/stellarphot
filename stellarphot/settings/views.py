from ipyautoui import AutoUi

__all__ = ["ui_generator"]


def ui_generator(model):
    """
    Silly function to generate an AutoUi object from a model.
    """
    return AutoUi(model)
