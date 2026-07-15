"""
Tests for the workarounds stellarphot applies for bugs in astrowidgets
0.5.0. Each widget class applies the workarounds independently in its
constructor, so the tests are parametrized over the widget classes.
"""

import warnings

import numpy as np
import pytest
from astropy.table import Table
from astrowidgets.bqplot import ImageWidget

from stellarphot.gui import comparison_functions as cf
from stellarphot.gui import seeing_profile_functions as spf
from stellarphot.gui.astrowidgets_workarounds import load_catalog


@pytest.mark.parametrize(
    "widget_class",
    [spf.SeeingProfileWidget, cf.ComparisonViewer],
    ids=["seeing-profile", "comparison-viewer"],
)
def test_builtin_click_handler_is_neutralized(widget_class):
    # astrowidgets 0.5.0 has a bug in which the bqplot ImageWidget's built-in
    # _mouse_click handler references attributes (click_center and is_marking)
    # that are never initialized, so any click raises AttributeError and,
    # because ipywidgets runs on_msg callbacks in registration order without
    # exception isolation, blocks our click handler too. Each widget works
    # around this by setting both attributes to False.
    widget = widget_class()
    iw = widget.iw
    assert iw.click_center is False
    assert iw.is_marking is False

    # With an image loaded, the built-in click handler should be a no-op
    # rather than raising AttributeError.
    iw.load_image(np.zeros((10, 10)))
    iw._mouse_click({"domain": {"x": 3, "y": 3}})


def test_load_catalog_does_not_warn_about_dtype():
    # astrowidgets passes the catalog table's Column objects straight to
    # bqplot Array traits; np.asarray copies a Column even when the dtype
    # already matches, so traittypes warns 'Given trait value dtype "float64"
    # does not match required type "float64"' on every catalog load and the
    # nonsense message lands in the app's log console. The load_catalog
    # wrapper suppresses that warning.
    iw = ImageWidget()
    iw.load_image(np.zeros((10, 10)))
    catalog = Table({"x": [2.0, 5.0], "y": [3.0, 7.0]})

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        load_catalog(
            iw,
            catalog,
            catalog_label="test",
            catalog_style={"shape": "cross", "color": "red", "size": 20},
        )

    dtype_warnings = [
        w for w in recorded if "Given trait value dtype" in str(w.message)
    ]
    assert not dtype_warnings

    # The catalog should still have been loaded.
    assert "test" in iw.catalog_labels


def test_load_catalog_applies_catalog_style_size():
    # astrowidgets 0.5.x plot_named_markers hard-codes default_size=100 on
    # the ScatterGL mark and never uses the size from catalog_style, so all
    # catalog markers render at the same (large) size. The load_catalog
    # wrapper sets the mark's default_size from the requested size, using
    # the same size**2 convention as astrowidgets' set_catalog_style.
    iw = ImageWidget()
    iw.load_image(np.zeros((10, 10)))
    catalog = Table({"x": [2.0, 5.0], "y": [3.0, 7.0]})

    load_catalog(
        iw,
        catalog,
        catalog_label="test",
        catalog_style={"shape": "cross", "color": "red", "size": 20},
    )

    mark = iw._astro_im._scatter_marks["test"]
    assert mark.default_size == 400
