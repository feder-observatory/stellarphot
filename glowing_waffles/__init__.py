# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is a package for doing stellar photometry that relies on astropy
"""

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

if not _ASTROPY_SETUP_:
    from .core import *
    from .source_detection import *
    from .photometry import photutils_stellar_photometry
