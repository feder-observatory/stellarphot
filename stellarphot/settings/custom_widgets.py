# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Backwards-compatibility shim: the widgets that used to live here moved to
# ``stellarphot.gui.custom_widgets`` so that ``stellarphot.settings`` stays pure
# Pydantic (no GUI imports). Importing from this old location still works but
# emits an AstropyDeprecationWarning; this shim will be removed in stellarphot
# 3.0.0.
import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

from stellarphot.gui import custom_widgets as _moved

warnings.warn(
    "stellarphot.settings.custom_widgets has moved to "
    "stellarphot.gui.custom_widgets; update your imports. Deprecated since "
    "stellarphot 2.1.0; this compatibility shim will be removed in stellarphot "
    "3.0.0.",
    AstropyDeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    return getattr(_moved, name)


def __dir__():
    return dir(_moved)
