# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from matplotlib.figure import Figure

from stellarphot.plotting import seeing_plot
from stellarphot.settings import PhotometryApertures

HWHM = 3.0


def fake_profile_data(hwhm=HWHM):
    """
    Make a small, smooth, fake radial profile suitable for passing to
    `~stellarphot.plotting.seeing_plot`.

    Parameters
    ----------
    hwhm : float, optional
        Half width at half maximum of the fake profile, in pixels.

    Returns
    -------
    dict
        Keyword arguments for `~stellarphot.plotting.seeing_plot`: the raw and
        binned radius/counts arrays and the HWHM.
    """
    raw_radius = np.linspace(0, 10 * hwhm, 100)
    raw_counts = np.exp(-0.5 * (raw_radius / hwhm) ** 2)
    binned_radius = raw_radius[::10]
    binned_counts = raw_counts[::10]

    return dict(
        raw_radius=raw_radius,
        raw_counts=raw_counts,
        binned_radius=binned_radius,
        binned_counts=binned_counts,
        HWHM=hwhm,
    )


def annotation_texts(fig):
    """
    All of the annotation/label strings in the (single-axes) figure.
    """
    return [text.get_text() for text in fig.axes[0].texts]


def test_seeing_plot_no_photometry_settings():
    # Regression test for #667: the documented default, in which no
    # photometry settings are passed in, used to raise a pydantic
    # ValidationError because the fallback settings were constructed from
    # read-only properties instead of fields.
    fig = seeing_plot(**fake_profile_data())

    assert isinstance(fig, Figure)

    # The documented defaults are radius = 4 * HWHM, inner annulus =
    # radius + 10 and outer annulus = radius + 25.
    radius = 4 * HWHM
    texts = annotation_texts(fig)
    assert f"Radius {radius:2.1f}" in texts
    assert f"Back> {radius + 10:2.1f}" in texts
    assert f"<Back {radius + 25:2.1f}" in texts


def test_seeing_plot_with_photometry_settings():
    settings = PhotometryApertures(
        radius=5, gap=7, annulus_width=11, fwhm_estimate=2 * HWHM
    )
    fig = seeing_plot(**fake_profile_data(), photometry_settings=settings)

    assert isinstance(fig, Figure)

    texts = annotation_texts(fig)
    assert "Radius 5.0" in texts
    assert "Back> 12.0" in texts
    assert "<Back 23.0" in texts
