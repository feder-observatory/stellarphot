# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Backwards-compatibility shim: moved to ``stellarphot.gui.profile_and_comps``.
import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

from stellarphot.gui import profile_and_comps as _moved

warnings.warn(
    "stellarphot.gui_tools.profile_and_comps has moved to "
    "stellarphot.gui.profile_and_comps; update your imports. Deprecated since "
    "stellarphot 2.1.0; this compatibility shim will be removed in stellarphot "
    "3.0.0.",
    AstropyDeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    return getattr(_moved, name)


def __dir__():
    return dir(_moved)
