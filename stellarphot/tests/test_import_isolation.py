# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Guard tests for the headless-import guarantee.

``import stellarphot`` (and the core data structures / compute layers) must not
pull in the ipywidgets/ipyautoui/astrowidgets/ginga GUI stack. All GUI code lives
in ``stellarphot.gui``; nothing else in the package may import the GUI libraries
at the source level.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import stellarphot

GUI_LIBRARIES = frozenset(
    {"ipywidgets", "ipyautoui", "astrowidgets", "ginga", "bqplot", "ipyfilechooser"}
)

# Source subtrees that are allowed to import the GUI libraries.
ALLOWED_GUI_PARTS = ("gui", "notebooks")

PACKAGE_ROOT = Path(stellarphot.__file__).parent


def _iter_non_gui_source_files():
    for path in PACKAGE_ROOT.rglob("*.py"):
        rel_parts = path.relative_to(PACKAGE_ROOT).parts
        # Skip the GUI package and the launcher notebooks (allowed to use widgets)
        # and skip test modules (GUI tests legitimately import the widget stack).
        if any(part in ALLOWED_GUI_PARTS for part in rel_parts):
            continue
        if "tests" in rel_parts:
            continue
        yield path


def _gui_imports_in(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in GUI_LIBRARIES:
                    hits.add(top)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            top = node.module.split(".")[0]
            if top in GUI_LIBRARIES:
                hits.add(top)
    return hits


def test_no_gui_imports_outside_gui_package():
    """No non-test module outside ``stellarphot.gui`` imports a GUI library."""
    offenders = {}
    for path in _iter_non_gui_source_files():
        hits = _gui_imports_in(path)
        if hits:
            offenders[str(path.relative_to(PACKAGE_ROOT))] = sorted(hits)
    assert not offenders, (
        "These non-GUI modules import GUI libraries (move them into "
        f"stellarphot.gui): {offenders}"
    )


def test_importing_stellarphot_loads_no_gui_libraries():
    """In a fresh interpreter, importing the engine pulls in no GUI library."""
    code = (
        "import sys, stellarphot\n"
        "from stellarphot import PhotometryData, CatalogData, SourceListData\n"
        "from stellarphot.photometry import AperturePhotometry\n"
        "loaded = sorted("
        f"  {set(GUI_LIBRARIES)!r} & {{m.split('.')[0] for m in sys.modules}})\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = result.stdout.strip()
    assert loaded == "", f"import stellarphot loaded GUI libraries: {loaded}"


def test_import_gui_works():
    """The GUI package itself still imports (with the [gui] extra installed)."""
    pytest.importorskip("ipywidgets")
    import stellarphot.gui  # noqa: F401
    from stellarphot.gui import ComparisonAndSeeing  # noqa: F401
