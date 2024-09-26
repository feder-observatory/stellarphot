"""
JSON Schema for Photometry Settings
===================================

The JSON schema for photometry settings is here.
"""

import json

from stellarphot.settings import photometry_settings_schema

json_schema = json.dumps(photometry_settings_schema, indent=4)
print(json_schema)
