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
measurement. The estimate only needs to be within roughly a factor of a few
of the actual FWHM -- the measurement is insensitive to it from about one
fifth of to four times the true value -- but outside that range the
measurement fails, and an image whose FWHM cannot be measured is skipped
with a warning.

Note: For differential photometry, a radius of about 1.5×FWHM is a good
compromise: the formal signal-to-noise optimum is smaller (near 0.7×FWHM for
sky-limited images), but such small apertures are much more sensitive to
seeing changes and centroiding errors. For absolute or all-sky photometry,
either use a much larger aperture or apply an aperture correction derived
from a growth curve (see, e.g., Howell's *Handbook of CCD Astronomy*).

To enable variable-aperture photometry, check the ``variable_aperture`` box in
the aperture settings shown by the seeing-profile widget, or set
``variable_aperture=True`` in your
`~stellarphot.settings.PhotometryApertures` settings.
