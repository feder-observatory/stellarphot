# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Backwards-compatibility shim: moved to ``stellarphot.gui.transit_fitting_gui``.
import warnings

from stellarphot.gui import transit_fitting_gui as _moved

warnings.warn(
    "stellarphot.transit_fitting.gui has moved to "
    "stellarphot.gui.transit_fitting_gui; update your imports. This "
    "compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    return getattr(_moved, name)


def __dir__():
    return dir(_moved)
