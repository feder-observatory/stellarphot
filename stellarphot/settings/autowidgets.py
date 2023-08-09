# Some classes for ipyautoui that really belong there, not here

import ipywidgets as w
from ipyautoui.autowidgets import create_widget_caller


class CustomBoundedIntTex(w.BoundedIntText):
    def __init__(self, schema):
        self.schema = schema
        self.caller = create_widget_caller(schema)
        super().__init__(**self.caller)
