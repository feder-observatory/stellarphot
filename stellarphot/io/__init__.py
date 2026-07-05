# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Backwards-compatibility shim. The public names in this package's submodules used
# to be importable directly from ``stellarphot.io`` (e.g. ``from stellarphot.io
# import TOI``). As of stellarphot 2.1.0 the canonical location is the submodule
# itself (``stellarphot.io.aavso`` / ``.aij`` / ``.tess``); the old top-level
# access still works via the lazy ``__getattr__`` below but emits an
# ``AstropyDeprecationWarning``. This shim will be removed in stellarphot 3.0.0.
#
# The lookup is lazy for two reasons: (1) ``stellarphot.io.tess`` imports
# ``stellarphot.core`` while ``core`` imports ``stellarphot.io.aavso``, so eagerly
# importing tess here would create a core <-> io import cycle -- keeping tess out
# of module import time means it only loads once ``core`` is already available; and
# (2) private/dunder probes (doctest's ``hasattr(mod, "__test__")``, pickling,
# etc.) are short-circuited so they never force a tess import or spurious warning.
# Note: ``from stellarphot.io import *`` is intentionally unsupported (no
# ``__all__``) -- import the specific submodule instead.

import importlib
import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

# Submodules whose public (``__all__``) names used to be exposed here. aavso and
# aij are cycle-safe; tess pulls in stellarphot.core, so it is imported lazily.
_MOVED_SUBMODULES = ("aavso", "aij", "tess")


def __getattr__(name):
    # Short-circuit private/dunder probes so they neither import tess nor warn.
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    for submodule in _MOVED_SUBMODULES:
        module = importlib.import_module(f".{submodule}", __name__)
        if name in getattr(module, "__all__", ()):
            warnings.warn(
                f"Importing {name!r} from stellarphot.io is deprecated; import it "
                f"from stellarphot.io.{submodule} instead. Deprecated since "
                "stellarphot 2.1.0; this compatibility shim will be removed in "
                "stellarphot 3.0.0.",
                AstropyDeprecationWarning,
                stacklevel=2,
            )
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    names = set(globals())
    for submodule in _MOVED_SUBMODULES:
        module = importlib.import_module(f".{submodule}", __name__)
        names.update(getattr(module, "__all__", ()))
    return sorted(names)
