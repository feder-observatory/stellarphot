Performing Aperture Photometry
##############################

Variable-Aperture Photometry
=============================

When enabled, variable-aperture photometry adapts the aperture to the seeing
in each image. The aperture radius, gap between aperture and sky annulus, and
annulus width are all interpreted as multiples of the per-image FWHM, estimated
from the stars in that image at photometry time (by
`~stellarphot.photometry.fast_fwhm_from_image`). The default values are
``radius=1.5``, ``gap=2.0``, and ``annulus_width=1.5`` (all in FWHM units).
This keeps a consistent fraction of each star's light inside the aperture and
maintains proper sky-annulus geometry when seeing varies across a night. Only
the ``fwhm_estimate`` setting is always in pixels; it seeds the per-image FWHM
measurement.

Note: For differential photometry, a radius of 1.5×FWHM optimizes
signal-to-noise. For absolute or all-sky photometry, use a radius of roughly
4×FWHM to capture nearly all the star's light (at the cost of higher sky
noise).

To enable variable-aperture photometry, check the ``variable_aperture`` box in
the aperture settings shown by the seeing-profile widget, or set
``variable_aperture=True`` in your
`~stellarphot.settings.PhotometryApertures` settings.
