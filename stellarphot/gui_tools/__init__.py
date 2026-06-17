# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Backwards-compatibility shim: the widget code that used to live in
# ``stellarphot.gui_tools`` now lives in ``stellarphot.gui``. Importing from the
# old location still works but emits a DeprecationWarning; this shim will be
# removed in a future release.
import warnings

from stellarphot import gui as _moved

warnings.warn(
    "stellarphot.gui_tools has moved to stellarphot.gui; update your imports. "
    "This compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    return getattr(_moved, name)


def __dir__():
    return dir(_moved)
