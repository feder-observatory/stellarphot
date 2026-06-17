# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Backwards-compatibility shim: moved to ``stellarphot.gui.fits_opener`` so that
# ``stellarphot.settings`` stays pure Pydantic (no GUI imports).
import warnings

from stellarphot.gui import fits_opener as _moved

warnings.warn(
    "stellarphot.settings.fits_opener has moved to stellarphot.gui.fits_opener; "
    "update your imports. This compatibility shim will be removed in a future "
    "release.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    return getattr(_moved, name)


def __dir__():
    return dir(_moved)
