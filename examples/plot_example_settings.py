"""
Example Photometry Settings
===========================

A sample set of photometry settings.
"""

from stellarphot.settings import photometry_settings_example

model_json = photometry_settings_example.model_dump_json(indent=4)
print(model_json)
