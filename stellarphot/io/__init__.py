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
# Note: ``from stellarphot.io import *`` is a deliberate no-op -- ``__all__`` is
# empty (see below), so star-import exposes nothing and never leaks this shim's
# own imports. Import the specific submodule instead.

import importlib
import warnings

# Keep ``from stellarphot.io import *`` a no-op. Without an explicit ``__all__``,
# star-import would fall back to this module's non-underscore globals and leak the
# shim internals (importlib, warnings); an empty list restores the pre-2.1.0
# empty-package behavior. The deprecated names remain reachable by explicit access
# through ``__getattr__`` below.
__all__ = []

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
            # Imported here (not at module level) to mirror the sibling shim in
            # stellarphot/core.py and keep the deprecation machinery out of the
            # module namespace.
            from astropy.utils.exceptions import AstropyDeprecationWarning

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


# NOTE: intentionally no custom ``__dir__``. The moved names stay reachable via
# ``__getattr__`` for back-compat, but they are deliberately not advertised by
# ``dir(stellarphot.io)``. A ``__dir__`` that enumerated them would (a) force an
# eager import of every submodule -- including ``tess``, which pulls in
# ``stellarphot.core`` -- defeating the laziness above, and (b) let documentation
# tools (automodapi/autodoc) rediscover and re-document them under their old,
# deprecated location, firing a deprecation warning that fails the ``-W`` docs
# build. This mirrors the shim in ``stellarphot/core.py``.
