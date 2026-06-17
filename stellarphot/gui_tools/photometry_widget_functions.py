# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Backwards-compatibility shim: moved to
# ``stellarphot.gui.photometry_widget_functions``.
import warnings

from stellarphot.gui import photometry_widget_functions as _moved

warnings.warn(
    "stellarphot.gui_tools.photometry_widget_functions has moved to "
    "stellarphot.gui.photometry_widget_functions; update your imports. This "
    "compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    return getattr(_moved, name)


def __dir__():
    return dir(_moved)
