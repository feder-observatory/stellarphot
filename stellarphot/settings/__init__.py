from .aavso_submission import *
from .models import *
from .settings_files import *

# For conveniences, provide the JSON schema and an example of PhotometrySettings.
photometry_settings_schema = PhotometrySettings.model_json_schema()  # noqa: F405

from .constants import TEST_PHOTOMETRY_SETTINGS

photometry_settings_example = PhotometrySettings(  # noqa: F405
    **TEST_PHOTOMETRY_SETTINGS
)

# Clean up the namespace
del TEST_PHOTOMETRY_SETTINGS


def __getattr__(name):
    # Lazily expose the widget builder so importing the data models does not pull
    # in ipywidgets/ipyautoui at package import time (see views.py). This keeps
    # ``from stellarphot.settings import ui_generator`` working for notebooks.
    if name == "ui_generator":
        from .views import ui_generator

        return ui_generator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
