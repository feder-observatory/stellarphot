"""
Workarounds for bugs in the astrowidgets 0.5.x bqplot backend that cannot
be fixed from stellarphot's side. Others, which need widget state, are
applied in the widget constructors (see e.g. ``ComparisonViewer._viewer``).
"""

import warnings

__all__ = ["load_catalog"]


def load_catalog(image_widget, *args, **kwargs):
    """
    Call ``image_widget.load_catalog`` with the traittypes dtype-coercion
    warning suppressed and the requested marker size actually applied.

    astrowidgets passes the catalog table's `~astropy.table.Column` objects
    straight to bqplot ``Array`` traits, and ``np.asarray`` copies a
    ``Column`` even when the dtype already matches, so traittypes warns
    'Given trait value dtype "float64" does not match required type
    "float64"' on every catalog load. The copy is harmless, but the
    nonsense message lands in the app's log console, so hide it.

    astrowidgets' ``plot_named_markers`` also hard-codes ``default_size=100``
    on the ``ScatterGL`` mark it creates, ignoring the ``size`` in
    ``catalog_style``, so every catalog marker renders at the same (large)
    size. Fix the mark up after the load, using the same ``size**2``
    convention as astrowidgets' ``set_catalog_style``.

    Parameters
    ----------
    image_widget : `astrowidgets.bqplot.ImageWidget`
        The widget whose ``load_catalog`` method to call.

    *args, **kwargs
        Passed through to ``image_widget.load_catalog``.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Given trait value dtype",
            category=UserWarning,
        )
        result = image_widget.load_catalog(*args, **kwargs)

    style = kwargs.get("catalog_style")
    if style and "size" in style:
        label = str(kwargs.get("catalog_label"))
        mark = image_widget._astro_im._scatter_marks.get(label)
        if mark is not None:
            mark.default_size = style["size"] ** 2

    return result
