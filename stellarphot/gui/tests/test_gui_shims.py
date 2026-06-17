# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The widget code moved from ``stellarphot.gui_tools`` / ``stellarphot.settings`` /
``stellarphot.transit_fitting.gui`` into ``stellarphot.gui``. The old import paths
still work via compatibility shims that emit a ``DeprecationWarning``. These tests
check that the shims forward attributes and warn.
"""

import importlib

import pytest

# (old module path, attribute that should resolve through the shim)
SHIMS = [
    ("stellarphot.gui_tools", "ComparisonAndSeeing"),
    ("stellarphot.gui_tools.comparison_functions", "ComparisonViewer"),
    ("stellarphot.gui_tools.seeing_profile_functions", "SeeingProfileWidget"),
    ("stellarphot.gui_tools.profile_and_comps", "ComparisonAndSeeing"),
    ("stellarphot.gui_tools.photometry_widget_functions", "TessAnalysisInputControls"),
    ("stellarphot.settings.custom_widgets", "ChooseOrMakeNew"),
    ("stellarphot.settings.views", "ui_generator"),
    ("stellarphot.settings.fits_opener", "FitsOpener"),
    ("stellarphot.transit_fitting.gui", "exotic_settings_widget"),
]


@pytest.mark.parametrize("module_path, attribute", SHIMS)
def test_old_import_path_warns_and_forwards(module_path, attribute):
    pytest.importorskip("ipywidgets")
    with pytest.warns(DeprecationWarning, match="has moved to stellarphot.gui"):
        module = importlib.import_module(module_path)
        # Force a fresh import so the warning fires even if cached.
        module = importlib.reload(module)
    assert hasattr(module, attribute)
