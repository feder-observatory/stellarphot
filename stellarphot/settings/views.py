# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Backwards-compatibility shim: moved to ``stellarphot.gui.views`` so that
# ``stellarphot.settings`` stays pure Pydantic (no GUI imports).
import warnings

from stellarphot.gui import views as _moved

warnings.warn(
    "stellarphot.settings.views has moved to stellarphot.gui.views; update your "
    "imports. This compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    return getattr(_moved, name)


def __dir__():
    return dir(_moved)
