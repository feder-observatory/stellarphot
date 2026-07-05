# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Public names that used to be importable directly from ``stellarphot.io`` now live
in the ``aavso``/``aij``/``tess`` submodules. The old top-level access still works
via a lazy compatibility shim that emits an ``AstropyDeprecationWarning``. These
tests check that the shim forwards attributes and warns.
"""

import importlib

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
    # must not trigger the shim, which would import tess and emit a warning.
    io = importlib.import_module("stellarphot.io")
    private = "_some_private_probe"
    with pytest.raises(AttributeError):
        getattr(io, private)
