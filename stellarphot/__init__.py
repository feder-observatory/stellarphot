# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
try:
    from .version import version as __version__
except ImportError:
    __version__ = ""

from .core import *

# We load this for its side effect of adding YAML representations for the models
from .table_representations import *
