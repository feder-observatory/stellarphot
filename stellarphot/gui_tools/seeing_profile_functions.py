# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Backwards-compatibility shim: moved to
# ``stellarphot.gui.seeing_profile_functions``.
import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

from stellarphot.gui import seeing_profile_functions as _moved

warnings.warn(
    "stellarphot.gui_tools.seeing_profile_functions has moved to "
    "stellarphot.gui.seeing_profile_functions; update your imports. Deprecated "
    "since stellarphot 2.1.0; this compatibility shim will be removed in "
    "stellarphot 3.0.0.",
    AstropyDeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    return getattr(_moved, name)


def __dir__():
    return dir(_moved)
