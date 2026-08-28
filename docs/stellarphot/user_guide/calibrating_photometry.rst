Calibrating Photometry
######################

Exporting to AAVSO
==================

Once you have calibrated magnitudes for your target and a check star, you can
write a file in the `AAVSO Extended File Format
<https://www.aavso.org/aavso-extended-file-format>`_ that the AAVSO WebObs
loader accepts. Stellarphot's writer produces an *ensemble* submission
(``CNAME=ENSEMBLE``, ``CMAG=na``) with one target star and one check star,
paired observation-by-observation by ``(date-obs, passband)``.

The writer expects the ``passband`` column to already contain valid AAVSO
filter names. If you started from instrumental filter names, use a
:class:`~stellarphot.settings.PassbandMap` when constructing
:class:`~stellarphot.PhotometryData` and the column will be remapped in place.

Example
-------

.. code-block:: python

    from stellarphot.settings import AAVSOSubmissionHeader

    header = AAVSOSubmissionHeader(
        type="EXTENDED",
        obscode="ABC",
        software="stellarphot 1.4",
        delim="comma",
        date_format="JD",
    )

    phot_data.write_aavso_extended(
        "submission.csv",
        header=header,
        target_star_id=1,
        target_name="V0533 Her",
        check_star_id=6,
        check_name="000-BLS-123",
        chart="X12345",
        mag_column="mag_cal",
        mag_error_column="mag_cal_error",
        trans=False,
    )

The ``mag_column`` and ``mag_error_column`` arguments name the calibrated
magnitude column and its uncertainty column to read for the target and the
check star.

``mag_cal_error`` combines the input uncertainties as quoted, the
uncertainty of the transform fit and, in quadrature, the image's
``fit_excess_scatter``, so it reflects the scatter actually seen in the
data. Before submitting, check the ``fit_redchi`` and ``fit_excess_scatter``
columns produced by
:func:`~stellarphot.utils.magnitude_transforms.transform_to_catalog`,
described under :ref:`calibration-fit-quality` below.

.. note::

   This release supports ``DATE=JD`` only. The AAVSO spec also allows ``HJD``
   and ``EXCEL`` dates; using either currently raises ``NotImplementedError``.
   The writer always emits ``#OBSTYPE=CCD`` and ``MTYPE=STD`` (consistent with
   ensemble photometry using standardized magnitudes).

.. _calibration-fit-quality:

Weighting and uncertainties in the transform fit
================================================

:func:`~stellarphot.utils.magnitude_transforms.transform_to_catalog` fits
the transform to each image separately, weighted by the errors of the stars
in it, and adds columns that describe the fit alongside the calibrated
magnitudes. All of those columns are properties of an image rather than of
a star, so they are repeated down every row of the image.

How the fit is weighted
-----------------------

The fit is weighted by the observed errors and the catalog's own errors
combined in quadrature, ``1 / sqrt(obs_error**2 + cat_error**2)``, wherever
the catalog has a ``mag_error_<cat_filter>`` column, and by the observed
errors alone, with a log message naming the band, wherever it does not. A
star whose catalog error is missing, masked, NaN or not positive is not left
out of the fit for it -- the catalog simply does not know its own uncertainty
for that star, so its weight falls back to the observed error alone, exactly
as if the catalog had no error column at all.

Which bands have an error depends on the catalog.
:class:`~stellarphot.CatalogData` builds ``mag_error_<band>`` for the bands a
catalog measured itself -- B, V, SG, SR and SI for APASS DR9, the Sloan bands
for refcat2 -- while the Johnson-Cousins R and I are added afterwards by
:func:`~stellarphot.utils.magnitude_system_transforms.transform_apass_bands`
and
:func:`~stellarphot.utils.magnitude_system_transforms.transform_refcat2_bands`,
which propagate the catalog's errors through the band transforms into
``mag_error_R`` and ``mag_error_I``. Those two are a floor rather than a
full error, because the USNO'-to-SDSS-DR7 step of the transform has no
published residual to add. So the fallback is the exception: it applies only
where a catalog lacks an error for a star or a band. When the fallback is in
use, ``fit_redchi`` is the column to watch: an image whose stars scatter
about the transform by more than they claim to be uncertain reports a
``fit_redchi`` well above one,
and where the catalog's uncertainty is the reason, weighting cannot know that
but the scatter still shows up there.

Whichever way it is weighted, the total uncertainty the fit weights each
star by is bounded below by ``min_fit_sigma``, 0.01 mag by default and ``0``
for no floor. Without one, a single star claiming a tiny uncertainty can
hold most of a fit's weight; see issue #694. The floor applies to the
weighting but not to the alarms: ``fit_redchi`` and ``fit_excess_scatter``
are measured against the errors as quoted, so an image whose quoted errors
are far too small still raises the alarm those columns exist for. The
measurement term of ``mag_cal_error`` is the star's own error exactly as
quoted, never raised to the floor. The floor is otherwise silent -- a star
it raised looks like one that quoted the floor -- so ``fit_sigma_floor_frac``
reports the fraction of each image's fitted stars it decided. Near zero, the
floor was a safeguard that did not fire; at one, every star weighed the same
and the fit was in effect unweighted, so the ``*_error`` columns are sized
by the floor rather than by anything measured.

Reading ``fit_redchi``
----------------------

``fit_redchi`` is the reduced chi-square: the summed squared residuals per
degree of freedom, i.e. divided by the number of stars fit minus the number
of terms varied. It is a chi-square only when ``obs_error_column`` is given,
because only then is anything dividing the residuals: the residuals are in
units of the errors that were supplied, and a value near one says the model
misses the stars by about as much as they claim to be uncertain. Without an
error column the fit is unweighted and the same column holds the summed
squared residuals per degree of freedom in **mag squared** -- the same name
and a completely different scale, so values from a weighted and an
unweighted fit must not be compared with each other.

``fit_redchi`` is a ratio, and an image whose errors are wrong in the right
way reports a healthy-looking one. Three more columns describe how the fit
was weighted, which ``fit_redchi`` cannot.

The weighting diagnostics
-------------------------

``fit_cat_error_missing_frac`` is the fraction of the stars in the fit whose
catalog error the fit could not use. It is 1.0 when the catalog has no error
column for the band and when the fit is unweighted, both of which are the
same statement -- nothing was known about any of them. A value near one says
``fit_redchi`` is not comparable with another band's: APASS DR9 reports an
error of exactly zero for most of its B stars in a typical field and almost
none of its V stars, and the stars it does that for are the faint ones, so
without the sigma floor the fit would weight the worst-measured stars the
most heavily.

``fit_max_weight_share`` is the largest share of the fit's total statistical
weight held by any one star, so a fit spread evenly over N stars reports
``1/N`` and one star running the fit reports a number near one. It is the
column that catches the case above, which nothing else in the output
reveals.

``fit_excess_scatter`` is the scatter, in magnitudes, that would have to be
added in quadrature to every star's uncertainty to bring ``fit_redchi`` to
one: how far the stars sit from the model over and above what they claim. It
is zero when the residuals are already no larger than the errors claim, and
NaN for an unweighted fit, which has no errors for its residuals to be
excessive with respect to. It is reported rather than folded into the
weights, because a fit that absorbs its own excess scatter has a reduced
chi-square of one by construction and can no longer report that anything is
wrong. Real contributors seen in it include flat-field gradients across the
field and the catalog's own photometry -- neither of which any weighting
scheme can repair. Note that it understates the excess when
``fit_cat_error_missing_frac`` is large, since the sigmas it is measured
against are then missing a term rather than merely being small.

What the uncertainties believe
------------------------------

The coefficient uncertainties, ``a_error`` through ``z_error``, believe the
errors the fit was weighted by -- the quoted errors, floored at
``min_fit_sigma`` -- not the scatter observed about the fit. That keeps
``fit_redchi`` and the coefficient ``*_error`` columns independent: stars
scattering beyond their quoted errors leave those columns alone and instead
raise ``fit_redchi``, with ``fit_excess_scatter`` giving the size of what the
quoted errors missed. When the quoted errors are wrong, the coefficient
``*_error`` columns are wrong with them, and ``fit_redchi`` far from one is
the alarm that says so -- they are then off by roughly its square root. The
one exception is quoted errors below the floor: there the coefficient
uncertainties take the floor's scale, overstated by about floor/quoted, and
that rule of thumb overcorrects; pass ``min_fit_sigma=0`` when errors that
small are genuine. An unweighted fit quotes no errors to believe, so its
uncertainties are scaled to the observed scatter, the only scale it has.

``mag_cal_error`` combines the star's own measurement error with the
uncertainty of the fitted transform, correlations between the terms
included, and then adds the image's ``fit_excess_scatter`` in quadrature.
The transform term is a significant contribution -- the part a plain
measurement-error column omits entirely -- that grows as the number of
fitted stars shrinks, and it is worked out per star because a fit predicts
best at the centroid of the stars it was fit to. The excess term is what
makes the column reflect the scatter actually observed about the transform
rather than only the quoted errors. The transform term itself still believes
the quoted errors, so when those are too small it remains understated by
about ``sqrt(fit_redchi)`` -- but it is the ``O(1/N)`` part of the total.

What the catalog contributes
----------------------------

A star's catalog entry decides whether it was matched. When a color term
(``c`` or ``d``) is fit, though, the model also applies that star's catalog
color, so the color's uncertainty is a real, knowingly omitted contribution
-- roughly ``c * sigma_color`` -- tracked as issue #691. What the catalog
reliably contributes is its field-wide scatter about the transform, which
lands in ``fit_excess_scatter``; and its systematic tie to the standard
system -- about 0.02 mag for APASS DR9 -- which is identical for every star
in every image, so a per-star column would mislead, appearing to average
down by the square root of the number of stars. Most of
``fit_excess_scatter``, measured against two catalogs, turns out to be
per-star observational scatter rather than catalog noise, which is why it is
added to every star's ``mag_cal_error`` directly; see the Notes of
:func:`~stellarphot.utils.magnitude_transforms.transform_to_catalog`.

A target whose ``mag_cal`` and ``mag_cat`` are NaN while the comparison
stars around it are fine is usually more than ``match_radius`` -- 2.0
arcsec by default -- from its catalog entry, which a VSX position need not
agree with that closely. Pass a larger ``match_radius`` to
:func:`~stellarphot.utils.magnitude_transforms.transform_to_catalog`; its
Notes explain why the fit's own, tighter limit is not a keyword.
