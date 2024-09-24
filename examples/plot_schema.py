"""
"This" is my example-script
===========================

This example doesn't do much, it just makes a simple plot
"""

import json

from stellarphot.settings import photometry_settings_schema

json_schema = json.dumps(photometry_settings_schema, indent=4)
print(json_schema)
