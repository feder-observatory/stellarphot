from .models import *
from .settings_files import *
from .views import *

# For conveniences, provide the JSON schema and an example of PhotometrySettings.
photometry_settings_schema = PhotometrySettings.model_json_schema()  # noqa: F405

from .tests.test_models import TEST_PHOTOMETRY_SETTINGS

photometry_settings_example = PhotometrySettings(  # noqa: F405
    **TEST_PHOTOMETRY_SETTINGS
)

# Clean up the namespace
del TEST_PHOTOMETRY_SETTINGS
