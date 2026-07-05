# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Backwards-compatibility shim. The public names in this package's submodules used
# to be importable directly from ``stellarphot.io`` (e.g. ``from stellarphot.io
# import TOI``). As of stellarphot 2.1.0 the canonical location is the submodule
# itself (``stellarphot.io.aavso`` / ``.aij`` / ``.tess``); the old top-level
# access still works via the lazy ``__getattr__`` below but emits an
# ``AstropyDeprecationWarning``. This shim will be removed in stellarphot 3.0.0.
#
# The lookup is lazy and table-driven for two reasons: (1) ``stellarphot.io.tess``
# imports ``stellarphot.core`` while ``core`` imports ``stellarphot.io.aavso``, so
# eagerly importing tess here would create a core <-> io import cycle -- keeping
# tess out of module import time means it only loads once ``core`` is already
# available; and (2) resolving each name through a static map means an unknown
# attribute (a typo, or a private/dunder probe from doctest/pickling) fails fast
# without importing any submodule at all, and a known name imports exactly the one
# submodule it lives in. Note: ``from stellarphot.io import *`` is a deliberate
# no-op -- ``__all__`` is empty (see below), so star-import exposes nothing and
# never leaks this shim's own imports. Import the specific submodule instead.

import importlib
import warnings

# Keep ``from stellarphot.io import *`` a no-op. Without an explicit ``__all__``,
# star-import would fall back to this module's non-underscore globals and leak the
# shim internals (importlib, warnings); an empty list restores the pre-2.1.0
# empty-package behavior. The deprecated names remain reachable by explicit access
# through ``__getattr__`` below.
__all__ = []

# Map of each public name that used to be exposed here to the submodule it now
# lives in. This is the union of the ``aavso``/``aij``/``tess`` ``__all__`` lists
# frozen at the 2.1.0 split. It is deliberately static rather than rebuilt from the
# live ``__all__`` lists: a name added to one of those submodules after the split
# must NOT silently reappear at this deprecated top-level location. A test asserts
# this map stays in sync with the submodules' ``__all__`` so drift fails loudly.
_MOVED_NAMES = {
    "write_aavso_extended": "aavso",
    "ApertureAIJ": "aij",
    "MultiApertureAIJ": "aij",
    "ApertureFileAIJ": "aij",
    "generate_aij_table": "aij",
    "parse_aij_table": "aij",
    "Star": "aij",
    "tess_photometry_setup": "tess",
    "TessSubmission": "tess",
    "TOI": "tess",
    "TessTargetFile": "tess",
}


def __getattr__(name):
    submodule = _MOVED_NAMES.get(name)
    if submodule is None:
        # Unknown names -- typos and private/dunder probes alike -- fail fast
        # without importing any submodule (in particular never pulling in tess).
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Imported here (not at module level) to mirror the sibling shim in
    # stellarphot/core.py and keep the deprecation machinery out of the module
    # namespace.
    from astropy.utils.exceptions import AstropyDeprecationWarning

    module = importlib.import_module(f".{submodule}", __name__)
    warnings.warn(
        f"Importing {name!r} from stellarphot.io is deprecated; import it "
        f"from stellarphot.io.{submodule} instead. Deprecated since "
        "stellarphot 2.1.0; this compatibility shim will be removed in "
        "stellarphot 3.0.0.",
        AstropyDeprecationWarning,
        stacklevel=2,
    )
    return getattr(module, name)


# NOTE: intentionally no custom ``__dir__``. The moved names stay reachable via
# ``__getattr__`` for back-compat, but they are deliberately not advertised by
# ``dir(stellarphot.io)``. A ``__dir__`` that enumerated them would (a) force an
# eager import of every submodule -- including ``tess``, which pulls in
# ``stellarphot.core`` -- defeating the laziness above, and (b) let documentation
# tools (automodapi/autodoc) rediscover and re-document them under their old,
# deprecated location, firing a deprecation warning that fails the ``-W`` docs
# build. This mirrors the shim in ``stellarphot/core.py``.
