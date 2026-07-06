"""
Tests for the workarounds stellarphot applies for bugs in astrowidgets
0.5.0. Each widget class applies the workarounds independently in its
constructor, so the tests are parametrized over the widget classes.
"""

import numpy as np
import pytest

from stellarphot.gui import comparison_functions as cf
from stellarphot.gui import seeing_profile_functions as spf


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
