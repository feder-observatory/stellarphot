# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Backwards-compatibility shim: the widget code that used to live in
# ``stellarphot.gui_tools`` now lives in ``stellarphot.gui``. Importing from the
# old location still works but emits an AstropyDeprecationWarning; this shim will
# be removed in stellarphot 3.0.0.
import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

from stellarphot import gui as _moved

warnings.warn(
    "stellarphot.gui_tools has moved to stellarphot.gui; update your imports. "
    "Deprecated since stellarphot 2.1.0; this compatibility shim will be removed "
    "in stellarphot 3.0.0.",
    AstropyDeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    return getattr(_moved, name)


def __dir__():
    return dir(_moved)
