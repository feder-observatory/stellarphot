# Some classes for ipyautoui that really belong there, not here

import ipywidgets as w
from ipyautoui.autowidgets import create_widget_caller

__all__ = [
    'CustomBoundedIntTex'
]


class CustomBoundedIntTex(w.BoundedIntText):
    """
    A BoundedIntText widget adapted for use in ipyautoui.
    """
    def __init__(self, schema):
        self.schema = schema
        self.caller = create_widget_caller(schema)
        super().__init__(**self.caller)
