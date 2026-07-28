Performing Aperture Photometry
##############################

Variable-Aperture Photometry
=============================

When enabled, variable-aperture photometry adapts the aperture radius to the
seeing in each image. The aperture radius used for an image is
``radius`` × (image FWHM), where the FWHM is estimated from the stars in that
image at photometry time (by `~stellarphot.photometry.fast_fwhm_from_image`).
This helps keep a consistent fraction of each star's light inside the aperture
when the seeing varies across a night.

The scaling applies only to the aperture radius: the ``gap`` between the
aperture and the sky annulus and the ``annulus_width`` are still given in
pixels, not in multiples of the FWHM.

To enable variable-aperture photometry, check the ``variable_aperture`` box in
the aperture settings shown by the seeing-profile widget, or set
``variable_aperture=True`` in your
`~stellarphot.settings.PhotometryApertures` settings.
