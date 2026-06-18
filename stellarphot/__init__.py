# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
try:
    from .version import version as __version__
except ImportError:
    __version__ = ""

# ``catalogs`` provides the catalog-fetcher factory functions (apass_dr9,
# vsx_vizier, refcat2) as public, un-deprecated top-level names, e.g.
# ``from stellarphot import apass_dr9``.
from .catalogs import *
from .core import *

# We load this for its side effect of adding YAML representations for the models
from .table_representations import *
