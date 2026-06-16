# Licensed under a 3-clause BSD style license - see LICENSE.rst

import importlib

from .aavso import *
from .aij import *


# tess imports stellarphot.core; expose its public names lazily so that importing
# io (e.g. from core.py) does not create a core <-> io import cycle. We import the
# submodule via importlib rather than ``from . import tess`` to avoid re-entering
# this __getattr__ through the from-import machinery.
def __getattr__(name):
    # Short-circuit private/dunder probes (e.g. doctest's ``hasattr(mod,
    # "__test__")``) so they do not import tess and mutate this package's
    # namespace mid-iteration in tools that walk ``__dict__``.
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    tess = importlib.import_module(".tess", __name__)
    if name in tess.__all__:
        return getattr(tess, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
