from ipyautoui import AutoUi

__all__ = ['ui_generator']


def ui_generator(model):
    return AutoUi(model)
