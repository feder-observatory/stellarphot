# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Public names that used to be importable directly from ``stellarphot.io`` now live
in the ``aavso``/``aij``/``tess`` submodules. The old top-level access still works
via a lazy compatibility shim that emits an ``AstropyDeprecationWarning``. These
tests check that the shim forwards attributes and warns.
"""

import importlib
import sys
import warnings

import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning

# (attribute name that used to live on stellarphot.io, submodule it now lives in)
SHIMS = [
    ("write_aavso_extended", "aavso"),
    ("ApertureAIJ", "aij"),
    ("generate_aij_table", "aij"),
    ("TOI", "tess"),
    ("TessSubmission", "tess"),
    ("TessTargetFile", "tess"),
    ("tess_photometry_setup", "tess"),
]


@pytest.mark.parametrize("name, submodule", SHIMS)
def test_old_io_access_warns_and_forwards(name, submodule):
    io = importlib.import_module("stellarphot.io")
    with pytest.warns(AstropyDeprecationWarning, match=r"stellarphot\.io"):
        obj = getattr(io, name)
    real = getattr(importlib.import_module(f"stellarphot.io.{submodule}"), name)
    assert obj is real


def test_unknown_io_attribute_raises_attribute_error():
    io = importlib.import_module("stellarphot.io")
    missing = "definitely_not_a_real_name"
    with pytest.raises(AttributeError):
        getattr(io, missing)


def test_private_probe_does_not_warn_or_import_tess():
    # Dunder/private probes (doctest's ``hasattr(mod, "__test__")``, pickling, ...)
    # must not trigger the shim: no deprecation warning and no import of the heavy
    # ``tess`` submodule. An unknown name misses the ``_MOVED_NAMES`` map and fails
    # fast before touching a submodule, so this holds regardless of test order.
    io = importlib.import_module("stellarphot.io")
    private = "_some_private_probe"
    # Clear both sys.modules and the parent-package attribute so the "not imported"
    # check keys off a genuine (re)import, not a module left over from an earlier
    # test, and so we don't leave the two in an inconsistent state on teardown.
    sys.modules.pop("stellarphot.io.tess", None)
    io.__dict__.pop("tess", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(AttributeError):
            getattr(io, private)
    assert not caught
    assert "stellarphot.io.tess" not in sys.modules


def test_moved_names_map_every_forward_is_valid():
    # ``_MOVED_NAMES`` is the frozen 2.1.0 snapshot of the deprecated surface,
    # intentionally NOT rebuilt from the live ``__all__`` lists: a name added to a
    # submodule after the split must not silently reappear at the deprecated
    # top-level location. So we do NOT assert the map equals the current union of
    # ``__all__`` (that would force the frozen surface to track later additions).
    # We only assert the weaker, still-important property that every forward the
    # map promises resolves -- each name really lives in the submodule it points at
    # -- so a rename/removal that would break a deprecated forward fails loudly.
    io = importlib.import_module("stellarphot.io")
    for name, submodule in io._MOVED_NAMES.items():
        mod = importlib.import_module(f"stellarphot.io.{submodule}")
        assert name in mod.__all__


def test_miss_does_not_import_any_submodule():
    # An unknown attribute must fail fast without importing any moved submodule --
    # in particular it must not drag in ``tess`` (and therefore ``stellarphot.core``)
    # just to raise AttributeError.
    io = importlib.import_module("stellarphot.io")
    names = ("aavso", "aij", "tess")
    for name in names:
        sys.modules.pop(f"stellarphot.io.{name}", None)
        io.__dict__.pop(name, None)
    missing = "definitely_not_a_real_name"
    with pytest.raises(AttributeError):
        getattr(io, missing)
    assert not any(f"stellarphot.io.{name}" in sys.modules for name in names)


def test_star_import_is_a_noop():
    # ``__all__ = []`` keeps ``from stellarphot.io import *`` a no-op so it does
    # not leak the shim's own imports (importlib, warnings, ...). Without the
    # explicit empty ``__all__`` star-import falls back to module globals.
    io = importlib.import_module("stellarphot.io")
    assert io.__all__ == []
    namespace = {}
    exec("from stellarphot.io import *", namespace)
    leaked = [name for name in namespace if not name.startswith("__")]
    assert leaked == []
