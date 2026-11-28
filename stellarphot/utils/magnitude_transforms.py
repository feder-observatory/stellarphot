import logging
import warnings
from contextlib import contextmanager

import lmfit
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, UnitSphericalRepresentation
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning
from numpy.lib.stride_tricks import sliding_window_view
from numpy.linalg import LinAlgError
from uncertainties import correlated_values, unumpy

from ..catalogs import apass_dr9, refcat2
from ..settings.aavso_models import AAVSOFilters
from .fit_diagnostics import excess_scatter, quoted_redchi
from .magnitude_system_transforms import transform_apass_bands, transform_refcat2_bands

logger = logging.getLogger(__name__)

__all__ = [
    "UnsupportedPassbandError",
    "calibrated_from_instrumental",
    "estimate_catalog_colors",
    "filter_transform",
    "transform_to_catalog",
]


class UnsupportedPassbandError(ValueError):
    """
    A passband cannot be calibrated against the catalog that was asked for.

    Raised by `transform_to_catalog` when the passband has no conventionally
    used color and none was given, and when the catalog has no magnitude for a
    passband the fit needs.

    Notes
    -----

    A caller transforming every passband in a table -- the shipped "Calibrate
    magnitudes" notebook loops over each unique value of the ``passband``
    column -- can catch this to skip the bands the catalog cannot calibrate
    without also swallowing the other `ValueError` exceptions that
    `transform_to_catalog` raises when the call itself is wrong.
    """


# Coefficients of the transform model, in the order
# `calibrated_from_instrumental` takes them. Keep the two in sync.
_COEFF_NAMES = ("a", "b", "c", "d", "z")

# Terms fit by default. The quadratic terms ``b`` and ``d`` are held at zero
# unless the caller asks for them.
_DEFAULT_VARY = ("a", "c", "z")

# Ranges a term is expected to fall in. These are *not* constraints on the
# fit -- a value outside the range is reported, not clamped.
_DEFAULT_EXPECTED = {"z": (18.0, 22.0)}

# Names of the columns of calibrated magnitudes that `transform_to_catalog`
# adds to the table it is given.
_CAL_MAG_COLUMN = "mag_cal"
_CAL_MAG_ERROR_COLUMN = "mag_cal_error"

# Name of the column of estimated colors that `estimate_catalog_colors` adds,
# and of the entry in the table's meta recording which two passbands each
# passband's estimate is the color of. The uncertainty of the estimate goes in
# the column of the same name with `_COEFF_ERROR_SUFFIX` added.
_COLOR_EST_COLUMN = "color_est"
_COLOR_EST_META_KEY = "color_est_bands"

# What joins the two passband names of a color in that record, and in
# messages: R-I. No AAVSO passband name contains it.
_COLOR_BAND_SEPARATOR = "-"

# Fewest stars, beyond the number of terms being fit, that the relation
# between instrumental and catalog color is fit to. With fewer there is no
# telling a good relation from one drawn through its own points.
_MIN_EXTRA_COLOR_STARS = 3

# How far, in robust standard deviations, a star may sit from the color
# relation and still help define it.
_COLOR_RELATION_CLIP = 4.0

# A fit star is left out of the color relation when the variance of its
# instrumental color, judged from the scatter of its own images, is more than
# this many times the median for the stars nearest it in magnitude. Such a
# star -- a variable, or one hit by a cosmic ray -- would otherwise be
# corrected for all of that noise at its full weight and steepen the slope
# for everyone. The yardstick is other stars, not the quoted errors: those
# can understate the scatter of every bright star several times over. With
# two images in each band an ordinary star's variance estimate has a long
# tail, and at 20 about one star in a thousand is left out by chance.
_SCATTER_OUTLIER_FACTOR = 20.0

# Number of stars, nearest in magnitude and counting the star itself, whose
# median variance is the yardstick above.
_SCATTER_NEIGHBOURS = 9

# Fewest fit stars for which any is left out as above. With fewer, what is
# ordinary for the night cannot be told.
_MIN_SCATTER_STARS = 8

# Largest fraction of the fit stars' weighted spread in instrumental color --
# what is left after the constant and any magnitude term -- that the
# correction for their noise may remove. Past this the slope rests on the
# small difference of two noisy sums and swings wildly from night to night;
# the night gets no relation instead.
_MAX_NOISE_FRACTION = 0.5

# A fractional flux error times this is a magnitude error, and a magnitude
# error is this divided by the signal-to-noise ratio.
_MAG_PER_FRACTIONAL_FLUX = 2.5 / np.log(10)

# Suffix of the column reporting the uncertainty of a fit coefficient, so that
# ``a`` is reported alongside ``a_error``. Matches the mag/mag_error and
# mag_cal/mag_cal_error pairs already in the table.
_COEFF_ERROR_SUFFIX = "_error"

# Name of the column reporting how well one image's transform fit its stars.
_FIT_REDCHI_COLUMN = "fit_redchi"

# Default of the ``min_fit_sigma`` parameter: the smallest uncertainty, in
# magnitudes, that a star is allowed to claim when the fit weights it. A
# least-squares weight goes as one over the variance, so a star claiming a
# sigma far below this does not merely count for a little more -- it can run
# the fit; see issue #694. The value is a statement about what is credible: a
# single ground-based measurement tied to a survey catalog is not good below
# about 0.01 mag whatever the catalog says.
#
# Applied to the total sigma the fit weights by, so it bounds the weight of a
# star however its sigma got small -- a zero catalog error, a tiny observed
# error, or a band whose catalog has no error column at all. It floors the
# weighting, and with it the covariance the reported uncertainties come from
# (see issue #690); the reported ``fit_redchi`` and ``fit_excess_scatter``
# are still measured against the errors as quoted.
_MIN_FIT_SIGMA = 0.01

# Names of the columns describing how the fit was weighted, as opposed to where
# it landed. Each answers a question ``fit_redchi`` cannot: whether the catalog
# knew its own errors at all, whether one star ran the fit, how far the
# residuals sit from the errors that were quoted, and how much of the
# weighting the ``min_fit_sigma`` floor decided.
_FIT_CAT_ERROR_MISSING_COLUMN = "fit_cat_error_missing_frac"
_FIT_MAX_WEIGHT_SHARE_COLUMN = "fit_max_weight_share"
_FIT_EXCESS_SCATTER_COLUMN = "fit_excess_scatter"
_FIT_SIGMA_FLOOR_COLUMN = "fit_sigma_floor_frac"

# The four as one tuple, mirroring `_COEFF_NAMES` above: `_fit_diagnostics`
# returns a dict keyed by exactly these names, and the per-image write loop
# and the output-column block both iterate over them, so adding a diagnostic
# is an edit here and in that return dict rather than a hunt for every site.
_FIT_DIAGNOSTIC_COLUMNS = (
    _FIT_CAT_ERROR_MISSING_COLUMN,
    _FIT_MAX_WEIGHT_SHARE_COLUMN,
    _FIT_EXCESS_SCATTER_COLUMN,
    _FIT_SIGMA_FLOOR_COLUMN,
)

# How close an observed star must be to a catalog entry. Two limits rather
# than one: a star must match tightly to be allowed to influence the fit, and
# less tightly to be given values derived from its match. Which stars
# *define* the transform is a different and riskier decision from which stars
# *receive* a calibrated magnitude, so only the second limit is a keyword
# (``match_radius``, whose default this is); the fit limit stays fixed. The
# Notes of ``transform_to_catalog`` say why the band between them exists,
# and issue #707 is where the default comes from.
_FIT_MATCH_RADIUS = 1.0 * u.arcsec
_CAL_MATCH_RADIUS = 2.0 * u.arcsec

# Added to the radius of the cone the calibration catalog is fetched from, so
# that the cone is a little larger than the field the observations cover: a
# star at the very edge of the field still has to find its catalog counterpart
# inside the cone. ``match_radius`` may not exceed this margin, so a
# counterpart close enough to match is always close enough to have been
# fetched. The margin is also what gives a field whose positions span nothing
# at all -- a single star, or one image of one target -- a cone of a sensible
# size rather than of no size.
_CATALOG_RADIUS_MARGIN = 1 * u.arcmin

# Derived cone size above which the caller is told what is about to be asked
# of Vizier. One degree because that is exactly what the radius used to be
# fixed at: a cone wider than this is one no call could previously produce,
# and a field that large is usually a mosaic, or ra/dec columns holding more
# than one field's worth of stars, rather than a single image. Only a cone
# that came out this wide on its own is worth a warning -- an explicit
# `search_radius` is a deliberate choice and passes without comment.
_WIDE_FIELD_RADIUS = 1 * u.degree

# The catalogs `transform_to_catalog` knows how to fetch. Named here so that
# the check of the caller's `cat_name`, which happens up front alongside the
# other argument checks, and the fetch itself, which happens much further
# down, cannot drift apart.
_CATALOG_NAMES = ("apass_dr9", "refcat2")

# The color conventionally used with each canonical passband, for a caller
# who names a band but no color. Which color goes with a band is a convention
# rather than something derivable, so a band that is not here has to be asked
# about rather than guessed at. Limited to the bands the two catalog fetches
# below can actually supply -- apass_dr9 fetches B, V, R, I, SR, SG and SI,
# refcat2 fetches B, V, R and I -- so U, SU and SZ are left out even though
# AAVSO recognizes them: naming a color built from a band the catalog cannot
# return would only replace this lookup's raise with a `KeyError` further on.
_CONVENTIONAL_COLOR = {
    "B": ("B", "V"),
    "V": ("B", "V"),
    "R": ("R", "I"),
    "I": ("R", "I"),
    "SG": ("SG", "SR"),
    "SR": ("SR", "SI"),
    "SI": ("SR", "SI"),
}


def _canonical_passband(name):
    """
    The AAVSO-standard spelling of a passband name.

    `~stellarphot.settings.aavso_models.AAVSOFilters` is the definitive list
    of AAVSO passbands, and its members carry both the name a caller is
    likely to type -- ``Rc`` and ``Ic`` for Cousins R and I -- and the value
    that is AAVSO's own preferred spelling for it -- ``"R"`` and ``"I"``. This
    matches ``name`` case-insensitively against both, so ``"Rc"``, ``"rc"``,
    ``"RC"`` and ``"R"`` all resolve to the one member and come back as
    ``"R"``. A name that matches nothing is not necessarily wrong -- callers
    can and do use catalog-specific bands this function has never heard of --
    so it is returned upper-cased rather than rejected, upper case being the
    convention AAVSO passband names follow.

    Parameters
    ----------

    name : str
        Passband name as a caller wrote it.

    Returns
    -------
    str
        The canonical spelling of the same band.
    """
    upper = name.upper()
    for member in AAVSOFilters:
        if member.name.upper() == upper or member.value.upper() == upper:
            return member.value
    return upper


# How strongly the terms of the fit may be correlated with each other before
# their individual values stop meaning anything. See `_underdetermined_reason`,
# which explains how this was arrived at.
_MAX_DESIGN_CONDITION = 3e3


def calibrated_from_instrumental(X, a, b, c, d, z):
    """
    Calculate the calibrated magnitudes from the instrumental magnitudes and colors.

    Parameters
    ----------

    X : tuple of numpy.ndarray
        The first element is an array of instrumental magnitudes,
        the second is an array of colors.

    a, b, c, d, z : float
        Parameters of the fit.

    Returns
    -------
    `numpy.ndarray`
        Array of calibrated magnitudes.
    """
    mag_inst, color = X

    return a * mag_inst + b * mag_inst**2 + c * color + d * color**2 + z


def _transform_residual(params, mag_inst, color, data, weights):
    """
    Residual of the transform model, in the form `lmfit.minimize` expects.

    Parameters
    ----------

    params : `lmfit.Parameters`
        Coefficients of the transform model.

    mag_inst, color : `numpy.ndarray`
        Instrumental magnitude and color of each star.

    data : `numpy.ndarray`
        Values being fit.

    weights : `numpy.ndarray` or float
        Weight of each residual, i.e. one over the uncertainty.

    Returns
    -------
    `numpy.ndarray`
        Weighted difference between the model and the data.
    """
    values = params.valuesdict()
    model = calibrated_from_instrumental(
        (mag_inst, color), *(values[name] for name in _COEFF_NAMES)
    )
    return (model - data) * weights


def _to_float_array(values):
    """
    Convert values to a plain float array, turning masked entries into NaN.

    Catalog columns are masked where the catalog has no value, and neither
    `scipy.optimize.least_squares` nor the model handles masked input
    sensibly. NaN carries the same information and propagates the way the
    calibrated magnitudes need it to.

    A bare ``values.filled(np.nan)`` will not do, which is why this looks more
    roundabout than it needs to: a plain `~astropy.table.Column` has no
    ``filled`` method at all, and depending on the catalog either flavor can
    turn up here; and an integer `~astropy.table.MaskedColumn` refuses NaN as
    a fill value, so the conversion to float has to happen first. Going
    through `numpy.ma` handles both -- unmasked input passes through
    `numpy.ma.asarray` without a mask, which makes `numpy.ma.filled` a no-op.

    The final `~numpy.asarray` drops any array subclass that survives the
    trip through `numpy.ma`. A `~astropy.units.Quantity` does survive it,
    unit and all -- and error columns in real photometry files can carry a
    unit -- which would otherwise crash the weighting arithmetic downstream:
    ``np.hypot`` of the observed and catalog errors and the ``np.maximum``
    sigma floor both refuse to mix a unit-bearing array with plain floats.
    Dropping the unit unconditionally is not the fix, though: an error
    column quoted in mmag would then enter the fit as if it were mag,
    silently wrong by a factor of 1000. So the unit, if there is one, is
    read off *before* the trip through `numpy.ma` and used afterward --
    converting to magnitudes when it is convertible to them, passing a
    dimensionless unit through unchanged, and raising rather than guessing
    for anything else.

    Parameters
    ----------

    values : array-like
        Values to convert. May be a `~astropy.table.MaskedColumn` or a
        `~astropy.units.Quantity`, with or without a unit.

    Returns
    -------
    `numpy.ndarray`
        Float array with NaN wherever the input was masked, converted to
        magnitudes if the input carried a unit convertible to them.

    Raises
    ------
    `~astropy.units.UnitConversionError`
        If ``values`` has a unit that is neither dimensionless nor
        convertible to magnitudes. Raised rather than stripped silently, the
        same complaint `numpy.hypot` used to raise mixing a unit-bearing
        array with plain floats before this function handled units at all.
    """
    unit = getattr(values, "unit", None)
    filled = np.asarray(
        np.ma.filled(np.ma.asarray(values, dtype=float), np.nan), dtype=float
    )

    if unit is None or unit.is_equivalent(u.dimensionless_unscaled):
        return filled

    if not unit.is_equivalent(u.mag):
        raise u.UnitConversionError(
            f"Cannot use values with unit {unit!r} here: it is not "
            "convertible to magnitudes and is not dimensionless."
        )

    return (filled * unit).to_value(u.mag)


def _to_degree_array(values):
    """
    Convert a coordinate column to plain degrees, masked entries becoming NaN.

    The sibling of `_to_float_array` for angles rather than magnitudes: that
    one raises for a unit like ``deg``, since a magnitude column carrying one
    is a mistake, and coordinate columns carry one routinely.

    Parameters
    ----------

    values : array-like
        Values to convert. May be a `~astropy.table.MaskedColumn` or a
        `~astropy.units.Quantity`. Values with no unit are taken to be in
        degrees, which is the convention the ``ra`` and ``dec`` columns of a
        photometry table follow.

    Returns
    -------
    `numpy.ndarray`
        Float array of degrees, NaN wherever the input was masked.
    """
    unit = getattr(values, "unit", None)
    filled = np.asarray(
        np.ma.filled(np.ma.asarray(values, dtype=float), np.nan), dtype=float
    )

    return filled if unit is None else (filled * unit).to_value(u.degree)


def _validate_angle(name, value, hint):
    """
    Check that a keyword argument is a single positive, finite angle.

    Parameters
    ----------

    name : str
        Name of the keyword, for the error message.

    value : `astropy.units.Quantity` or anything that can be made one
        The value the caller passed.

    hint : str
        Sentence appended to the message for a value of the right kind but
        the wrong size, telling the caller what to do instead.

    Returns
    -------

    `astropy.units.Quantity`
        The value, as a Quantity.

    Raises
    ------

    ValueError
        If the value cannot be made a Quantity, is not an angle, is not a
        scalar, or is not positive and finite. The message begins with
        ``name``.
    """
    # Quantity raises TypeError for None and ValueError for a string it
    # cannot parse; neither message names the keyword, so both become the
    # not-an-angle error.
    try:
        value = u.Quantity(value)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"{name} must be an angle, such as 5 * u.arcmin, got {value!r}."
        ) from err
    if not value.unit.is_equivalent(u.degree):
        raise ValueError(
            f"{name} must be an angle, such as 5 * u.arcmin, got {value!r}."
        )
    if not value.isscalar:
        raise ValueError(f"{name} must be a single angle, got {value!r}.")
    if not np.isfinite(value.value) or value.value <= 0:
        raise ValueError(
            f"{name} must be a positive, finite angle, got {value!r}. {hint}"
        )
    return value


def _observed_field(ra, dec):
    """
    Find the patch of sky the observed positions cover.

    Parameters
    ----------

    ra, dec : array-like
        Right ascension and declination of the observed positions, in the
        form `_to_degree_array` accepts.

    Returns
    -------
    center : `astropy.coordinates.SkyCoord`
        Centroid of the positions.

    radius : `astropy.units.Quantity`
        Angular radius about ``center`` enclosing every position, plus
        `_CATALOG_RADIUS_MARGIN`.

    Raises
    ------
    `ValueError`
        If no position has both a finite right ascension and a finite
        declination.

    Notes
    -----

    The center is the mean of the positions' unit cartesian vectors rather
    than the mean of their right ascensions and declinations. The latter is
    wrong across the RA = 0 wrap, and wrong by half the sky: a field
    straddling it averages to the antipode, and a cone drawn there holds none
    of the stars it was meant to hold.

    A row is dropped unless both of its coordinates are usable, since half a
    position places nothing, and a masked or NaN entry left in would poison
    the mean rather than merely be ignored by it.
    """
    ra_degree = _to_degree_array(ra)
    dec_degree = _to_degree_array(dec)
    usable = np.isfinite(ra_degree) & np.isfinite(dec_degree)
    if not usable.any():
        raise ValueError(
            "No observed position has both a finite ra and a finite dec, so "
            "there is no field to search a catalog around."
        )

    coords = SkyCoord(ra_degree[usable], dec_degree[usable], unit="degree")
    direction = coords.cartesian.mean().represent_as(UnitSphericalRepresentation)
    center = SkyCoord(
        direction.lon, direction.lat, frame=coords.frame.replicate_without_data()
    )

    return center, coords.separation(center).max() + _CATALOG_RADIUS_MARGIN


def _underdetermined_reason(fit_result, vary):
    """
    Explain why a fit could not determine its terms, if it could not.

    `lmfit` reports success whether or not the data can actually tell the
    terms being fit apart from each other, so a successful fit still has to be
    checked before its coefficients are used. What is needed is already in the
    result: ``jac``, how much each term moves the residual at each star, which
    is what decides whether the terms can be distinguished.

    That is used rather than the covariance matrix, which answers the same
    question but is an inverse, so on hopeless data it comes back with
    negative variances or does not come back at all, and every one of those
    would need a case of its own here. ``jac`` is evaluated rather than
    inverted, so it is always finite: a term the data says nothing about
    leaves its column zero, and the condition number then comes out infinite
    on its own rather than by special arrangement.

    The columns are scaled to unit length first, so the measure does not
    depend on the wildly different scales of the terms -- ``mag_inst`` runs
    around -10 while ``color`` runs around 0.5.

    The threshold is a judgement call, because degeneracy is a continuum
    rather than a state. Measured over synthetic fits, a healthy three-term
    fit stays below about 400 even when every star has nearly the same
    brightness; a five-term fit over a three magnitude range reaches about
    750, and over a one magnitude range about 7000 -- and at that point it
    really does return coefficients wrong by several magnitudes, so rejecting
    it is right. Data whose color is exactly a linear function of
    instrumental magnitude lands around 3e7.

    Note that this cannot catch every fit whose coefficients are useless. A
    field correlated tightly enough to give coefficients wrong by a magnitude
    can still sit just below the threshold. Reporting the per-coefficient
    uncertainties the fit already works out would let a caller judge the
    cases this necessarily lets through.

    Parameters
    ----------

    fit_result : `lmfit.minimizer.MinimizerResult`
        Result of fitting one image.

    vary : sequence of str
        Names of the terms that were fit, used only in the message.

    Returns
    -------
    str or None
        Description of the problem, or `None` if the fit is determined.
    """
    jacobian = fit_result.jac

    # Leaving a zero-length column alone keeps it zero, which is what makes
    # the condition number infinite for a term the data cannot see at all.
    lengths = np.linalg.norm(jacobian, axis=0)
    scaled = jacobian / np.where(lengths > 0, lengths, 1.0)

    if np.linalg.cond(scaled) > _MAX_DESIGN_CONDITION:
        return (
            f"the terms {list(vary)} cannot be told apart from each other in "
            "this data"
        )

    return None


def _fit_diagnostics(fit_result, sigma, weights, cat_error_usable, min_fit_sigma):
    """
    Describe how one image's fit was weighted, as opposed to where it landed.

    Four numbers that ``fit_redchi`` cannot supply on its own, all of them
    from issue #694: a reduced chi-square is a ratio, so an image whose errors
    are wrong in the right way reports a healthy-looking one.

    Parameters
    ----------

    fit_result : `lmfit.minimizer.MinimizerResult`
        The fit these describe.

    sigma : `numpy.ndarray` or None
        Uncertainty of each star that was fit, as quoted -- before the
        ``min_fit_sigma`` floor, so the excess-scatter diagnostic measures
        the residuals against what the errors claim rather than against the
        floor. `None` for an unweighted fit, which has no uncertainties.

    weights : `numpy.ndarray` or float
        Weight the fit gave each residual, floor included -- the caller's
        own ``weights``, not something re-derived from ``sigma``. The scalar
        ``1.0`` for an unweighted fit.

    cat_error_usable : `numpy.ndarray` or None
        Boolean mask over the stars that were fit: whether each one's
        catalog error was usable, as decided where the errors were prepared.
        `None` when the catalog had no error column for the band, or when
        the fit was unweighted and never consulted one.

    min_fit_sigma : float
        The floor the caller applied to ``sigma`` to get ``weights``.

    Returns
    -------
    dict
        Keyed by `_FIT_DIAGNOSTIC_COLUMNS`, ready to be written to the rows
        of this image.

    Notes
    -----
    The floor fraction counts the stars whose quoted sigma is at or below
    ``min_fit_sigma`` -- ``<=`` rather than ``<`` because that is what
    ``np.maximum(sigma, min_fit_sigma)`` does: a star quoting exactly the
    floor gets the same weight as one raised to it, and this column should
    not tell the two apart. It is NaN for an unweighted fit, which had no
    sigmas to floor, rather than 0.0, which would claim they were all
    above it.
    """
    # A catalog with no error column for the band knows nothing about any of
    # these stars, which is the same statement as a column of zeros rather than
    # a question that does not apply, so it reads as "all of them missing".
    if cat_error_usable is None:
        missing_fraction = 1.0
    else:
        missing_fraction = float(np.mean(~cat_error_usable))

    # What a least-squares fit actually apportions among the stars is one over
    # the variance, so that is what a share of the fit means. An unweighted fit
    # divides it evenly, which `numpy.broadcast_to` gets right for free.
    statistical_weight = np.broadcast_to(
        np.asarray(weights) ** 2,
        np.shape(fit_result.residual),
    )
    max_weight_share = float(statistical_weight.max() / statistical_weight.sum())

    if sigma is None:
        floor_fraction = np.nan
    else:
        floor_fraction = float(np.mean(sigma <= min_fit_sigma))

    return {
        _FIT_CAT_ERROR_MISSING_COLUMN: missing_fraction,
        _FIT_MAX_WEIGHT_SHARE_COLUMN: max_weight_share,
        _FIT_EXCESS_SCATTER_COLUMN: excess_scatter(fit_result, sigma, weights),
        _FIT_SIGMA_FLOOR_COLUMN: floor_fraction,
    }


def _coefficient_uncertainties(fit_result):
    """
    Uncertainty of each coefficient of a fit, when the fit can supply them.

    Parameters
    ----------

    fit_result : `lmfit.minimizer.MinimizerResult`
        Result of fitting one image.

    Returns
    -------
    uncertainties : dict
        Standard error of each term of the transform model, keyed by term
        name. A term that was not varied was held at exactly zero and is
        therefore known exactly, so its uncertainty is ``0.0``. Every value is
        NaN if the covariance cannot be used.

    covar : `numpy.ndarray` or None
        The covariance matrix, or `None` if it cannot be used. Anything that
        needs the correlations between the terms as well as their individual
        uncertainties should take the matrix from here rather than asking the
        fit result again, so that there is only ever one verdict.

    Notes
    -----

    This is the single place the question "does this fit have a usable
    covariance?" is answered, so that everything derived from the covariance
    agrees about whether there is one to derive anything from.

    ``fit_result.errorbars`` is deliberately *not* what is checked. This code
    consumes the covariance matrix itself, so the matrix is what it checks;
    ``errorbars`` is a summary flag `lmfit` derives from that same object, and
    checking the artifact actually used is one less indirection.
    """
    covar = fit_result.covar

    if covar is None or not np.all(np.isfinite(covar)):
        return {name: np.nan for name in _COEFF_NAMES}, None

    # lmfit's own `create_uvars` wraps this same decomposition in
    # `except (LinAlgError, ValueError): pass` and falls back silently to
    # uncorrelated, stderr-based ufloats. Not known to happen with real data;
    # checked only so a covariance that fails to decompose (e.g. a tiny
    # negative eigenvalue from round-off) is never turned into nonsense.
    varied = [name for name in _COEFF_NAMES if fit_result.params[name].vary]
    try:
        correlated_values([fit_result.params[name].value for name in varied], covar)
    except (LinAlgError, ValueError):
        return {name: np.nan for name in _COEFF_NAMES}, None

    return {
        name: (
            np.nan
            if fit_result.params[name].stderr is None
            else float(fit_result.params[name].stderr)
        )
        for name in _COEFF_NAMES
    }, covar


@contextmanager
def _nan_is_an_expected_input():
    """
    Let NaN through arithmetic without numpy calling it invalid.

    A star with no instrumental magnitude, no color, or no usable error is
    meant to come out of the model with a NaN beside it rather than to be
    dropped, so NaN going in is part of the contract here rather than a sign
    that something has gone wrong.

    numpy 2.5 disagrees when the arithmetic runs elementwise over an array of
    uncertain values, which is what propagating the fit's covariance makes it
    do: it reports the NaN as an invalid-value `RuntimeWarning`, on Linux and
    Windows but not macOS. This package's test suite turns warnings into
    errors, so that would fail on some platforms and not others. Both layers
    are quieted because the two spellings of the complaint arrive by different
    routes -- one through the floating-point error state, one raised directly
    by the vectorized call numpy makes over the object array.

    Nothing else is silenced: a genuinely invalid operation on a real number
    is not what this covers, and NaN in the output is still NaN in the output.
    """
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
        yield


def _calibrated_with_uncertainty(
    fit_result,
    covar,
    mag_inst,
    color,
    errors,
    fit_diff,
    excess_scatter=0.0,
    color_errors=None,
):
    """
    Calibrated magnitude of every star, and how uncertain each one is.

    The returned error carries the star's own measurement error plus the
    uncertainty of the fitted transform, correlations between the transform's
    terms included, plus the image's excess scatter.

    Parameters
    ----------

    fit_result : `lmfit.minimizer.MinimizerResult`
        Result of fitting one image.

    covar : `numpy.ndarray` or None
        Covariance matrix of the fit, as vetted by
        `_coefficient_uncertainties`, or `None` if it cannot be used. Nothing
        is propagated when it is `None`: the calibrated magnitudes are still
        worked out, but there is no honest uncertainty to report for them.

    mag_inst, color : `numpy.ndarray`
        Instrumental magnitude and color of every star in the image, not just
        the ones the fit used.

    errors : `numpy.ndarray` or None
        Uncertainty of each instrumental magnitude, or `None` if none was
        supplied. Must already have been checked: `uncertainties` refuses
        outright to build a value whose standard deviation is negative. NaN it
        accepts, and carries through to a NaN uncertainty beside a perfectly
        good calibrated magnitude.

    fit_diff : bool
        Whether the fit was of the difference between the catalog and
        instrumental magnitudes, in which case the instrumental magnitude is
        added back here.

    excess_scatter : float, optional
        The image's ``fit_excess_scatter``, added in quadrature to every
        star's uncertainty. Non-positive or NaN values add nothing.

    color_errors : `numpy.ndarray`, optional
        Uncertainty of each color, propagated through the color terms. Must
        be non-negative or NaN, for the same reason ``errors`` must. Leave
        out for colors taken as exact.

    Returns
    -------
    calibrated : `numpy.ndarray`
        Calibrated magnitude of each star.

    uncertainty : `numpy.ndarray` or None
        Uncertainty of each calibrated magnitude, or `None` when there was
        nothing to propagate.

    Notes
    -----

    The coefficients arrive as `uncertainties` ufloats carrying the fit's
    full covariance matrix, and ``mag_inst`` is itself made a ufloat before
    `calibrated_from_instrumental` -- the same model function the fit used --
    evaluates on them. That way the sensitivity of the calibrated magnitude to
    the instrumental one, including the extra term from the ``fit_diff``
    add-back, is never written down by hand.

    This slightly over-counts the uncertainty of a star that was itself in
    the fit, since its own noise also helped move the coefficients; erring
    high there is deliberate.

    The excess scatter is added after the propagation, not to the star's
    measurement error before it.
    `~stellarphot.utils.fit_diagnostics.excess_scatter` measures it in
    calibrated-magnitude residual space -- against ``hypot(errors,
    cat_error)`` with no ``1 + a`` factor -- so adding it before the
    propagation would scale it by a ``1 + a`` sensitivity it already
    carries. It is the same number for every star in the image, so it does
    not bend the star-to-star shape of the transform term. See issue #698.
    """
    if covar is None or errors is None:
        # NaN inputs propagate through plain float arithmetic without
        # tripping numpy's invalid-value warning, so there is nothing to
        # suppress here.
        calibrated = calibrated_from_instrumental(
            (mag_inst, color),
            *(fit_result.params[name].value for name in _COEFF_NAMES),
        )
        if fit_diff:
            calibrated = calibrated + mag_inst
        return calibrated, None

    uvars = fit_result.params.create_uvars(covar)

    with _nan_is_an_expected_input():
        # Building the uncertain values is itself elementwise over the array,
        # so a NaN error complains here rather than in the model below.
        star_mag = unumpy.uarray(mag_inst, errors)
        if color_errors is not None:
            color = unumpy.uarray(color, color_errors)

        calibrated = calibrated_from_instrumental(
            (star_mag, color), *(uvars[name] for name in _COEFF_NAMES)
        )
        if fit_diff:
            calibrated = calibrated + star_mag

        uncertainty = unumpy.std_devs(calibrated)

    if excess_scatter > 0:
        uncertainty = np.hypot(uncertainty, excess_scatter)

    return unumpy.nominal_values(calibrated), uncertainty


def _check_known_terms(terms, argument_name):
    """
    Raise if any of ``terms`` is not a coefficient of the transform model.

    Parameters
    ----------

    terms : iterable of str
        Term names to check.

    argument_name : str
        Name of the argument the terms came from, used in the error message.
    """
    unknown = sorted(set(terms) - set(_COEFF_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown term(s) in {argument_name}: {unknown}. "
            f"The terms of the transform model are {list(_COEFF_NAMES)}."
        )


def _merge_with_existing(table, column_name, new_values, in_passband):
    """
    Combine newly fit values with any already in the table.

    Only the rows in the passband that was just fit are updated, so calling
    `transform_to_catalog` once per passband accumulates results in a single
    table instead of each call overwriting the one before it.

    Parameters
    ----------

    table : `astropy.table.Table`
        Table the column will be written to.

    column_name : str
        Name of the column being written.

    new_values : `numpy.ndarray`
        Values from this call, NaN outside the passband that was fit.

    in_passband : `numpy.ndarray`
        Boolean mask, `True` for the rows this call is responsible for.

    Returns
    -------
    `numpy.ndarray`
        Values to write to the column.
    """
    if column_name not in table.colnames:
        return new_values

    try:
        old_values = _to_float_array(table[column_name])
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Column {column_name!r} is already in the table with a "
            f"non-numeric dtype ({table[column_name].dtype}), so the values "
            "for rows in other passbands cannot be kept. Remove or rename "
            "the column and try again."
        ) from e

    return np.where(in_passband, new_values, old_values)


def _write_output_columns(table, columns, in_passband):
    """
    Write a set of output columns to the table, keeping other passbands' values.

    Each column goes through `_merge_with_existing`, so the rows this call is
    not responsible for keep whatever they already had. Columns are written in
    the order they appear in ``columns``, which is the order a new column is
    added to the table in.

    Parameters
    ----------

    table : `astropy.table.Table`
        Table the columns will be written to.

    columns : dict
        Values to write, as ``{column_name: values}``. Each array is the
        length of the whole table, NaN outside the passband that was fit.

    in_passband : `numpy.ndarray`
        Boolean mask, `True` for the rows this call is responsible for.
    """
    for column_name, new_values in columns.items():
        table[column_name] = _merge_with_existing(
            table, column_name, new_values, in_passband
        )


# Keys under which `transform_to_catalog` records, in the meta of the table it
# writes to, how each passband's call was made. Each holds a dict keyed by
# ``obs_filter``.
_META_KEYS = ("transform_weighting", "transform_match_radius")


def _record_call_in_meta(table, obs_filter, weighting_mode, match_radius):
    """
    Record how this call was made in the table's meta, keyed by passband.

    Keyed by ``obs_filter`` rather than a flat entry because
    `transform_to_catalog` is called once per passband on the same table, and
    a flat entry would be overwritten by the next call, losing the first
    passband's record.

    Entries for passbands with no rows in the table are dropped, since they
    describe nothing in it -- the same rule `_merge_with_existing` applies to
    the columns. Without this, ``group_by("passband")`` on a table that has
    been transformed before would hand every group a copy of the other
    passbands' entries from the earlier run, and stacking the re-transformed
    groups would merge those stale entries back in (issue #711).

    Parameters
    ----------

    table : `astropy.table.Table`
        Table whose meta the record is written to.

    obs_filter : str
        Passband this call transformed, exactly as the caller passed it.

    weighting_mode : str
        How the fit was weighted; see `transform_to_catalog`.

    match_radius : `astropy.units.Quantity`
        The match radius the call used.
    """
    present = set(table["passband"])
    for key in _META_KEYS:
        entries = table.meta.setdefault(key, {})
        for passband in list(entries):
            if passband not in present:
                del entries[passband]

    table.meta["transform_weighting"][obs_filter] = weighting_mode
    table.meta["transform_match_radius"][obs_filter] = match_radius.to(u.arcsec)


# For jester, using the transform for "all stars with Rc-Ic < 1.15"
# from
# http://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php#Jester2005
# Each entry is the coefficients of g, r and i and a constant term.
_JESTER_TRANSFORMS = {
    "B": [1.39, -0.39, 0, 0.21],
    "V": [0.41, 0.59, 0, -0.01],
    "R": [0.41, -0.5, 1.09, -0.23],
    "I": [0.41, -1.5, 2.09, -0.44],
}

# RMS residuals of the same tabulated transformations. B and V are direct
# fits; R and I are compositions of them (R = V - (V-R), with rms 0.01 and
# 0.03, and I = R - (Rc-Ic), with rms 0.01 more), so their residuals combine
# in quadrature.
_JESTER_RMS = {
    "B": 0.03,
    "V": 0.01,
    "R": np.sqrt(0.01**2 + 0.03**2),
    "I": np.sqrt(0.01**2 + 0.03**2 + 0.01**2),
}


def filter_transform(mag_data, output_filter, g=None, r=None, i=None, transform=None):
    """
    Transform SDSS magnitudes to BVRI using either the transforms from
    Jester et al or Ivezic et al.

    Parameters
    ----------

    mag_data : `astropy.table.Table`
        Table containing ``g``, ``r`` and ``i`` magnitudes (or at least)
        those required to transform to the desired output filter.

    output_filter : 'B', 'V', 'R' or 'I'
        Filter for which magnitude should be calculated. Note that
        *case matters* here.

    g, r, i : str
        Name of column in table for that magnitude.

    transform : 'jester' or 'ivezic'
        Transform equations to use.

    Returns
    -------

    `astropy.table.Column`
        Output transformed magnitudes as a table column

    Notes
    -----

    The transforms implemented in this function are taken from:

    Jester, et al, *The Sloan Digital Sky Survey View of the Palomar-Green Bright
    Quasar Survey*, AJ 130, p. 873 (2005)
    http://iopscience.iop.org/article/10.1086/432466/meta

    Ivezić et al, *A Comparison of SDSS Standard Star Catalog for Stripe 82 with
    Stetson's Photometric Standards*,
    The Future Of Photometric, Spectrophotometric And Polarimetric Standardization,
    ASP Conference Series 364, p. 165 (2007)
    http://aspbooks.org/custom/publications/paper/364-0165.html

    """
    supported_transforms = ["jester", "ivezic"]
    if transform not in supported_transforms:
        raise ValueError(
            f"Transform {transform} is not known. Must be one of {supported_transforms}"
        )
    transform_ivezic = {
        "B": [0.2628, -0.7952, 1.0544, 0.0268],
        "V": [0.0688, -0.2056, -0.3838, -0.0534],
        "R": [-0.0107, 0.0050, -0.2689, -0.1540],
        "I": [-0.0307, 0.1163, -0.3341, -0.3584],
    }
    base_mag_ivezic = {"B": g, "V": g, "R": r, "I": i}
    if output_filter not in base_mag_ivezic.keys():
        raise ValueError("the desired filter must be a string R B V or I")

    if transform == "ivezic":
        if output_filter == "R" or output_filter == "I":
            # This will throw a KeyError if the column is missing
            c = mag_data[r] - mag_data[i]

        if output_filter == "B" or output_filter == "V":
            # This will throw a KeyError if the column is missing
            c = mag_data[g] - mag_data[r]

        transform_poly = np.poly1d(transform_ivezic[output_filter])
        out_mag = transform_poly(c) + mag_data[base_mag_ivezic[output_filter]]
        # poly1d  ignores masks. Add masks back in here if necessary.
        try:
            input_mask = c.mask
        except AttributeError:
            pass
        else:
            out_mag = np.ma.array(out_mag, mask=input_mask)
    elif transform == "jester":
        coeff = _JESTER_TRANSFORMS[output_filter]
        out_mag = (
            coeff[0] * mag_data[g]
            + coeff[1] * mag_data[r]
            + coeff[2] * mag_data[i]
            + coeff[3]
        )

    out_mag.name = f"{output_filter}_mag"
    out_mag.description = f"{output_filter}-band magnitude transformed from gri"
    return out_mag


def _group_means(values, group, n_groups):
    """
    Plain mean of the finite values in each group.

    Parameters
    ----------

    values : `numpy.ndarray`
        Values to average.

    group : `numpy.ndarray`
        Integer index, for each value, of the group it belongs to.

    n_groups : int
        Number of groups.

    Returns
    -------
    mean : `numpy.ndarray`
        Mean of each group, NaN for a group with no finite value in it.

    count : `numpy.ndarray`
        Number of finite values in each group.
    """
    finite = np.isfinite(values)
    count = np.bincount(group[finite], minlength=n_groups)
    total = np.bincount(group[finite], weights=values[finite], minlength=n_groups)
    mean = np.full(n_groups, np.nan)
    populated = count > 0
    mean[populated] = total[populated] / count[populated]
    return mean, count


def _group_medians(values, group, n_groups):
    """
    Median of the values in each group.

    Parameters
    ----------

    values : `numpy.ndarray`
        Values, all finite.

    group : `numpy.ndarray`
        Integer index, for each value, of the group it belongs to.

    n_groups : int
        Number of groups.

    Returns
    -------
    `numpy.ndarray`
        Median of each group, NaN for an empty group.
    """
    # Sorted by group and, within a group, by value, so that each group's
    # median sits in the middle of its own run of the sorted values.
    ordered = values[np.lexsort((values, group))]
    count = np.bincount(group, minlength=n_groups)
    start = np.cumsum(count) - count
    populated = count > 0
    low = start[populated] + (count[populated] - 1) // 2
    high = start[populated] + count[populated] // 2
    median = np.full(n_groups, np.nan)
    median[populated] = (ordered[low] + ordered[high]) / 2
    return median


def _night_means(mags, errors, group, n_groups):
    """
    Magnitude of the mean flux in each group, and its uncertainty.

    Parameters
    ----------

    mags, errors : `numpy.ndarray`
        Magnitudes to average and their uncertainties. A magnitude takes
        part only if it is finite and its uncertainty is a positive, finite
        number.

    group : `numpy.ndarray`
        Integer index, for each magnitude, of the group it belongs to.

    n_groups : int
        Number of groups.

    Returns
    -------
    mean, error : `numpy.ndarray`
        Magnitude of the unweighted mean flux of each group, and the
        uncertainty of that magnitude.

    quoted_error : `numpy.ndarray`
        Uncertainty of that magnitude propagated from the quoted errors
        alone.

    noise_variance : `numpy.ndarray`
        Unbiased estimate of the variance of that magnitude, from the
        scatter of the fluxes alone; for a group with one value, the square
        of ``quoted_error``.

    Notes
    -----

    All four returned arrays are NaN for a group with nothing usable in it.

    Each uncertainty is that of the mean flux, converted to magnitudes.
    ``error`` is the larger of two estimates of it: the sample standard
    deviation of the fluxes divided by the square root of their number, and
    ``quoted_error``. A few images can agree better than their noise by
    chance, and the second keeps the uncertainty from following them down;
    with one image it is the only estimate there is.

    The larger of two estimates is biased high, which is harmless in a
    reported error but not in the correction `_fit_color_relation` makes
    for noise in the instrumental colors. That correction uses
    ``noise_variance``, which does not trust the quoted errors either: the
    read noise they assume can be wrong for the night.
    """
    usable = np.isfinite(mags) & np.isfinite(errors) & (errors > 0)
    members = group[usable]
    flux = 10 ** (-0.4 * mags[usable])
    flux_error = flux * errors[usable] / _MAG_PER_FRACTIONAL_FLUX

    mean_flux, count = _group_means(flux, members, n_groups)
    populated = count > 0
    count = count[populated]

    propagated = np.sqrt(
        np.bincount(members, weights=flux_error**2, minlength=n_groups)[populated]
    )
    propagated /= count
    squared_deviations = np.bincount(
        members, weights=(flux - mean_flux[members]) ** 2, minlength=n_groups
    )[populated]
    scatter = np.zeros_like(propagated)
    several = count > 1
    scatter[several] = np.sqrt(
        squared_deviations[several] / (count[several] - 1) / count[several]
    )

    def magnitudes(flux_error):
        # Error of the mean flux, as an error of its magnitude.
        result = np.full(n_groups, np.nan)
        result[populated] = _MAG_PER_FRACTIONAL_FLUX * flux_error / mean_flux[populated]
        return result

    mean = np.full(n_groups, np.nan)
    mean[populated] = -2.5 * np.log10(mean_flux[populated])
    error = magnitudes(np.maximum(propagated, scatter))
    quoted_error = magnitudes(propagated)
    noise_variance = magnitudes(np.where(several, scatter, propagated)) ** 2

    return mean, error, quoted_error, noise_variance


def _instrumental_color(first, second):
    """
    Instrumental color from the nightly means of its two bands.

    Parameters
    ----------

    first, second : tuple of `numpy.ndarray`
        What `_night_means` returns for each band of the color.

    Returns
    -------
    color, error : `numpy.ndarray`
        Instrumental color of each group and its uncertainty.

    weighting : dict
        The ``weight_error`` and ``noise_variance`` keyword arguments of
        `_fit_color_relation`: the uncertainty of the color from the quoted
        errors, and the variance of the color from the scatter of the
        images.
    """
    first_mean, first_error, first_quoted, first_variance = first
    second_mean, second_error, second_quoted, second_variance = second
    weighting = {
        "weight_error": np.hypot(first_quoted, second_quoted),
        "noise_variance": first_variance + second_variance,
    }
    return first_mean - second_mean, np.hypot(first_error, second_error), weighting


def _well_measured(in_band, mags, errors, group, n_groups, min_snr, min_fraction):
    """
    Whether each group is measured well enough in one band to be used.

    Parameters
    ----------

    in_band : `numpy.ndarray`
        Whether each row is in the band.

    mags, errors : `numpy.ndarray`
        Instrumental magnitude of each row and its uncertainty.

    group : `numpy.ndarray`
        Integer index, for each row, of the group it belongs to.

    n_groups : int
        Number of groups.

    min_snr : float
        Smallest median signal-to-noise ratio of a group's usable rows in
        the band.

    min_fraction : float
        Smallest fraction of a group's rows in the band that must be
        usable.

    Returns
    -------
    `numpy.ndarray`
        For each group, ``True`` if it meets both limits.

    Notes
    -----

    A row is usable if its magnitude is finite and its uncertainty is a
    positive, finite number. Its signal-to-noise ratio is
    ``2.5 / ln(10)`` divided by its uncertainty.
    """
    usable = in_band & np.isfinite(mags) & np.isfinite(errors) & (errors > 0)
    n_rows = np.bincount(group[in_band], minlength=n_groups)
    n_usable = np.bincount(group[usable], minlength=n_groups)
    snr = _group_medians(
        _MAG_PER_FRACTIONAL_FLUX / errors[usable], group[usable], n_groups
    )
    fraction = np.zeros(n_groups)
    seen = n_rows > 0
    fraction[seen] = n_usable[seen] / n_rows[seen]
    # A NaN median, for a group with nothing usable, compares as False.
    return (n_usable > 0) & (fraction >= min_fraction) & (snr >= min_snr)


def _scatter_outliers(noise_variance, mag, candidates):
    """
    Which stars' images scatter far more than those of stars as bright.

    Parameters
    ----------

    noise_variance : `numpy.ndarray`
        Variance of each star's instrumental color, from the scatter of its
        images.

    mag : `numpy.ndarray`
        Magnitude of each star.

    candidates : `numpy.ndarray`
        Whether each star is to be judged; the rest are neither judged nor
        part of the yardstick.

    Returns
    -------
    `numpy.ndarray`
        ``True`` for each candidate whose ``noise_variance`` is more than
        `_SCATTER_OUTLIER_FACTOR` times the median of the
        `_SCATTER_NEIGHBOURS` candidates nearest it in magnitude. All
        ``False`` when there are fewer than `_MIN_SCATTER_STARS` candidates
        with a magnitude.
    """
    wild = np.zeros(len(noise_variance), dtype=bool)
    judged = np.flatnonzero(candidates & np.isfinite(mag))
    if len(judged) < _MIN_SCATTER_STARS:
        return wild

    order = judged[np.argsort(mag[judged], kind="stable")]
    variance = noise_variance[order]
    width = min(_SCATTER_NEIGHBOURS, len(order))
    medians = np.median(sliding_window_view(variance, width), axis=1)
    # Each star's window is centred on it where the ends allow.
    start = np.clip(np.arange(len(order)) - width // 2, 0, len(order) - width)
    wild[order] = variance > _SCATTER_OUTLIER_FACTOR * medians[start]
    return wild


def _fit_color_relation(
    color_inst,
    color_inst_error,
    mag,
    color_cat,
    magnitude_term,
    *,
    weight_error,
    noise_variance,
):
    """
    Fit catalog color as a linear function of instrumental color.

    Parameters
    ----------

    color_inst, color_inst_error : `numpy.ndarray`
        Instrumental color of each star and the uncertainty to report for
        it.

    mag : `numpy.ndarray`
        Zero-pointed instrumental magnitude of each star in the first band
        of the color.

    color_cat : `numpy.ndarray`
        Catalog color of each star, NaN where the catalog has none.

    magnitude_term : bool
        Whether the relation also has a term linear in ``mag``.

    weight_error : `numpy.ndarray`
        Uncertainty of each instrumental color that the fit is weighted by.

    noise_variance : `numpy.ndarray`
        Estimate of the variance of each instrumental color, which the fit
        corrects its slope for.

    Returns
    -------
    estimate : `numpy.ndarray` or None
        Color the relation gives for every star, whether or not it was
        fit. `None` if no relation could be fit: too few stars, stars
        whose colors or magnitudes do not differ enough to define one, or
        stars whose colors differ by too little next to their noise.

    error : `numpy.ndarray` or None
        Uncertainty of each estimate.

    Notes
    -----

    The fit is weighted by ``weight_error``, floored at `_MIN_FIT_SIGMA`
    for the reason the transform fit floors its own, and is repeated after
    dropping stars more than `_COLOR_RELATION_CLIP` robust standard
    deviations away -- a variable, or a star whose catalog color is wrong
    -- until none are dropped. On each pass a star whose ``noise_variance``
    is far above that of the stars nearest it in magnitude is left out
    (see `_scatter_outliers`): its mean color may sit on the relation, so
    the clip keeps it, but the correction below would remove its noise at
    its full weight and steepen the slope for every star.

    Noise in the instrumental colors, which are the independent variable
    here, flattens an ordinary least-squares slope, and a flattened slope
    biases the estimate for a star whose color is outside the range of the
    fitted stars -- which a supernova's often is. The weighted sum of
    ``noise_variance`` is therefore subtracted from the color term of the
    normal equations, the standard moment correction for a regressor
    measured with known error. It has to be an unbiased estimate of the
    variance: one biased high over-corrects and steepens the slope. The
    noise's covariance with the magnitude term, whose regressor shares the
    first band's noise, is left alone: it is small next to the spread of
    the stars in magnitude.

    When the correction would remove more than `_MAX_NOISE_FRACTION` of
    the weighted spread in color that the other terms leave -- the colors
    differ by little more than their noise -- the slope is the small
    difference of two noisy sums and can be wildly wrong, so no relation
    is returned.

    The uncertainty has two parts, added in quadrature: the star's own
    ``color_inst_error`` scaled by the slope, and the uncertainty of
    the fitted relation evaluated at the star, which grows away from the
    stars that define it. The scatter of the fitted stars about the
    relation enters only through the second, not by being added whole:
    much of it is the catalog's own color errors, which the estimate for
    another star does not suffer.
    """
    columns = [np.ones_like(color_inst), color_inst]
    if magnitude_term:
        # About its median, so that the constant and the magnitude term can
        # be told apart numerically.
        columns.append(mag - np.nanmedian(mag))
    design = np.column_stack(columns)
    n_terms = design.shape[1]

    fit = (
        np.isfinite(design).all(axis=1)
        & np.isfinite(color_cat)
        & np.isfinite(color_inst_error)
        & np.isfinite(weight_error)
        & np.isfinite(noise_variance)
    )
    weights = 1 / np.maximum(weight_error, _MIN_FIT_SIGMA)

    # The columns other than the color.
    others = [0, *range(2, n_terms)]

    # Each pass either drops at least one star or ends the loop, so this
    # cannot run for longer than there are stars.
    while True:
        # Judged afresh on each pass, against the stars still in the fit.
        used = fit & ~_scatter_outliers(noise_variance, mag, fit)
        if used.sum() < n_terms + _MIN_EXTRA_COLOR_STARS:
            return None, None

        weighted_design = design[used] * weights[used, np.newaxis]
        normal = weighted_design.T @ weighted_design
        # The noise in each color adds its variance, times the square of
        # its weight, to this element of the weighted normal matrix.
        correction = np.sum(weights[used] ** 2 * noise_variance[used])

        # The weighted spread in color that the other terms do not account
        # for: the color element of the normal matrix less what the other
        # columns explain of it.
        try:
            explained = normal[1, others] @ np.linalg.solve(
                normal[np.ix_(others, others)], normal[others, 1]
            )
        except LinAlgError:
            return None, None
        if correction > _MAX_NOISE_FRACTION * (normal[1, 1] - explained):
            return None, None

        normal[1, 1] -= correction

        # Stars whose colors differ by no more than their noise, or whose
        # magnitudes are all the same, leave a matrix that is not positive
        # definite or is nearly singular. Either way there is no relation
        # in the data to find.
        try:
            np.linalg.cholesky(normal)
        except LinAlgError:
            return None, None
        if np.linalg.cond(normal) > _MAX_DESIGN_CONDITION**2:
            return None, None

        inverse = np.linalg.inv(normal)
        coefficients = inverse @ (weighted_design.T @ (color_cat[used] * weights[used]))

        residuals = color_cat - design @ coefficients
        center = np.median(residuals[used])
        # Scaled so that it estimates the standard deviation of a Gaussian,
        # and floored so that stars following the relation almost exactly
        # are not dropped over rounding error.
        spread = max(
            1.4826 * np.median(np.abs(residuals[used] - center)), _MIN_FIT_SIGMA
        )
        keep = fit & (np.abs(residuals - center) <= _COLOR_RELATION_CLIP * spread)
        if keep.sum() == fit.sum():
            break
        fit = keep

    degrees_of_freedom = used.sum() - n_terms

    # The weights are not the true uncertainties of the residuals -- they
    # know nothing of the catalog's errors -- so the covariance is scaled
    # by the reduced chi-square to the scatter actually seen.
    redchi = np.sum((residuals[used] * weights[used]) ** 2) / degrees_of_freedom
    relation_variance = np.einsum("ij,jk,ik->i", design, inverse * redchi, design)

    estimate = design @ coefficients
    error = np.sqrt((coefficients[1] * color_inst_error) ** 2 + relation_variance)

    return estimate, error


def estimate_catalog_colors(
    observed_mags,
    obs_mag_col="mag_inst",
    obs_error_column="mag_error",
    cat_name="apass_dr9",
    vary=_DEFAULT_VARY,
    search_radius=None,
    in_place=True,
    min_snr=5.0,
    min_image_fraction=0.8,
    error_floor=0.0,
):
    """
    Estimate a catalog-system color for every star from its instrumental
    magnitudes in two passbands, one estimate per night.

    The estimate is for stars a catalog has no color for -- a supernova is
    the usual case -- so that `transform_to_catalog` can still apply a color
    term to them: pass the ``color_est`` column this adds as its
    ``color_fallback_column``. Call this once, on a table holding every
    passband, before transforming each passband.

    Parameters
    ----------

    observed_mags : `astropy.table.Table`
        Observations in every passband. Must have ``file``, ``passband``,
        ``ra``, ``dec``, ``star_id`` and ``night`` columns in addition to the
        magnitude and error columns named below. Need not be grouped.

    obs_mag_col : str, optional
        Name of the column of instrumental magnitudes.

    obs_error_column : str, optional
        Name of the column of errors in the instrumental magnitudes.

    cat_name : str, optional
        Name of the catalog whose colors are being estimated, either
        ``"apass_dr9"`` or ``"refcat2"``. Use the catalog the transform will
        be fit against.

    vary : sequence of str, optional
        The terms the transform will be fit with; see `transform_to_catalog`.
        Only whether a magnitude term (``"a"`` or ``"b"``) is among them is
        used here.

    search_radius : `astropy.units.Quantity`, optional
        Radius of the cone the catalog is fetched from; see
        `transform_to_catalog`.

    in_place : bool, optional
        If ``True``, add the columns to the input table, otherwise return a
        copy.

    min_snr : float, optional
        Smallest median signal-to-noise ratio for which a star gets an
        estimate, in each of the two bands that night. The median is over
        the images on which the star has a usable measurement: a finite
        magnitude and a positive, finite error. Zero turns the limit off.

    min_image_fraction : float, optional
        Smallest fraction of a star's images in each of the two bands that
        night that must have a usable measurement for it to get an
        estimate. Greater than zero and at most one. The fraction is of the
        images in which the star has a row, so a star with rows in only a
        few of a night's images is judged on those. On a night with 2 to 4
        images in a band, one unusable image drops the star at the default.

    error_floor : float, optional
        Uncertainty, in magnitudes, added in quadrature to every
        ``color_est_error``.

    Returns
    -------

    `astropy.table.Table`
        The table with two columns added. ``color_est`` is the estimated
        color and ``color_est_error`` its uncertainty. Each row holds the
        color conventionally used to transform that row's own passband --
        R-I on R and I rows, B-V on B and V rows -- and the same value on
        every row of one star, one night and one color. Both are NaN for a
        star not observed in both passbands of the color that night, for a
        star that misses ``min_snr`` or ``min_image_fraction`` in either
        passband that night, and for a passband whose color could not be
        estimated at all.

        They are also NaN for a night on which too few stars had both an
        instrumental and a catalog color to relate the two, or those stars
        were too alike in color to define the relation, or the noise in
        their instrumental colors was too large next to their spread in
        color.

        ``result.meta["color_est_bands"]`` records, for each passband that
        got an estimate, the color the estimate is, as a string such as
        ``"R-I"``.

    Notes
    -----

    For each color, say R-I, the steps are:

    1. Each R and each I image is put on a common scale by fitting its zero
       point alone against the catalog, which takes out the changes in
       extinction and transparency through a night.
    2. Each star's zero-pointed magnitudes are turned into fluxes, the
       fluxes are averaged, without weights, over the night in each band,
       and the magnitudes of the two means are differenced to give an
       instrumental color. Only stars that meet ``min_snr`` and
       ``min_image_fraction`` in both bands get one.
    3. Each night, the catalog color of the stars that have one is fit as a
       linear function of instrumental color -- and of magnitude, when
       ``vary`` has a magnitude term -- and the fit is evaluated for every
       star.

    The fluxes are averaged rather than the magnitudes because the quoted
    magnitude errors are computed from the measured flux: weighting by them
    favours the images on which noise made a star brighter, and a weighted
    mean of the magnitudes of a faint star comes out too bright -- by about
    0.05 mag over many images that each have an error of 0.2 mag. An image
    on which a star's net counts were negative has no magnitude and cannot
    enter the mean, which biases it the same way; ``min_image_fraction`` is what keeps
    such stars from getting an estimate, and ``min_snr`` keeps out stars
    faint enough for that to happen often. The signal-to-noise ratio is
    judged from the star's own measured flux, so a faint star whose
    aperture holds light from a bright neighbour can pass the cut and get
    a wrong color.

    The relation in step 3 is exact for a transform with linear color and
    magnitude terms, which is the default; with the quadratic terms ``"b"``
    or ``"d"`` it is an approximation, and a poor one for a star well
    outside the color range of the stars that define it. Only stars within
    1.0 arcsec of their catalog entry define it, and a star whose images
    scatter far more than those of stars as bright -- a variable, say -- is
    left out of it. The fit is weighted by
    the quoted errors and allows for the noise in the instrumental colors,
    which would otherwise flatten the relation and bias the estimate for a
    star bluer or redder than the rest. It judges that noise from the
    scatter of each star's own images rather than from the quoted errors,
    whose assumed read noise can be wrong for the night. The estimate is
    on the catalog's own color scale, so any offset of the catalog's colors
    in the field carries over to it.

    Only the two passbands of the color itself are used. The relation
    between a color in one pair of bands and a color in another holds for
    stars and not for a supernova, so no other pair is substituted when one
    of the two is missing.

    ``color_est_error`` combines three things in quadrature: the
    uncertainty of the instrumental color scaled by the slope of the
    relation; the uncertainty of the relation itself at the star's color
    and magnitude, which grows away from the stars that define it; and
    ``error_floor``. The uncertainty of the instrumental color combines
    those of the two bands' means, and each of those is the uncertainty of
    the mean flux -- the larger of the scatter of the star's own fluxes
    that night divided by the square root of their number, and what the
    quoted errors give -- divided by the mean flux. The scatter of the
    fitted stars about the relation is not added on its own: much of it is
    the catalog's color errors, which say nothing about another star's
    estimate.

    ``error_floor`` exists because an error computed from the images
    leaves out whatever is the same on every image: flat-field and other
    position-dependent errors, and light from neighbouring stars. For B-V
    on the Feder Observatory camera in the field of NGC 7790, with
    ``error_floor=0`` the reported error was about 1.5 to 2.4 times too
    small, worst for bright stars, whose images scatter by only a few
    mmag. With ``error_floor=0.010`` it was about right (0.8 to 1.05 times
    the actual error) for stars without close neighbours; for stars with
    bright neighbours within about 25 arcsec it stayed about 1.5 to 2.4
    times too small. The default of zero builds no camera's value into the
    result. The error does not include any real scatter of objects about
    the relation, so it is too small for an object whose spectrum is not
    like a star's, such as a supernova, by an amount this function cannot
    know.

    When `transform_to_catalog` propagates ``color_est_error`` into
    ``mag_cal_error`` it treats the color as independent of the star's own
    instrumental magnitude in that image, though that magnitude helped set
    the color. The effect is the size of the color coefficient: with one
    image per band and a coefficient of 0.3 the magnitude error is
    understated by about 30 percent when the coefficient is positive and
    overstated when it is negative, and by less with more images or a
    smaller coefficient.

    The two bands of a color are not observed at the same moment, so a star
    whose brightness changes within a night gets a color blurred by that
    change. An object whose spectrum is unlike a star's -- a supernova --
    does not follow a transform derived from stars exactly even given its
    true color; the estimate removes the first-order color dependence, not
    that.
    """
    required = (
        "file",
        "passband",
        "ra",
        "dec",
        "star_id",
        "night",
        obs_mag_col,
        obs_error_column,
    )
    missing = [name for name in required if name not in observed_mags.colnames]
    if missing:
        raise ValueError(
            f"The table has no column(s) {missing}, needed to estimate colors. "
            "A night is what the instrumental magnitudes are averaged over, "
            "and star_id is what identifies a star from one image to the next."
        )

    if isinstance(vary, str):
        raise TypeError(
            f"vary must be a sequence of term names, not the string {vary!r}. "
            f"Did you mean {(vary,)!r}?"
        )
    _check_known_terms(vary, "vary")
    magnitude_term = bool({"a", "b"} & set(vary))

    # Written so that NaN fails each test.
    if not min_snr >= 0:
        raise ValueError(f"min_snr must be zero or more, not {min_snr}.")
    if not 0 < min_image_fraction <= 1:
        raise ValueError(
            "min_image_fraction must be more than zero and at most one, "
            f"not {min_image_fraction}."
        )
    if not (np.isfinite(error_floor) and error_floor >= 0):
        raise ValueError(
            f"error_floor must be a finite number, zero or more, not {error_floor}."
        )

    n_rows = len(observed_mags)
    passbands = np.asarray(observed_mags["passband"]).astype(str)
    mags = _to_float_array(observed_mags[obs_mag_col])
    errors = _to_float_array(observed_mags[obs_error_column])

    # The spellings in the table that mean each canonical passband, and the
    # spellings whose transform uses each color.
    spellings = {}
    users = {}
    # Plain `str`, not `numpy.str_`: the spellings become keys of the meta
    # record, which has to survive being written to a file.
    for spelling in sorted(str(passband) for passband in set(passbands)):
        band = _canonical_passband(spelling)
        spellings.setdefault(band, []).append(spelling)
        if band in _CONVENTIONAL_COLOR:
            users.setdefault(_CONVENTIONAL_COLOR[band], []).append(spelling)

    # One group per star per night.
    nights, night_index = np.unique(
        np.asarray(observed_mags["night"]), return_inverse=True
    )
    stars, star_index = np.unique(
        np.asarray(observed_mags["star_id"]), return_inverse=True
    )
    n_nights, n_stars = len(nights), len(stars)
    group = night_index * n_stars + star_index
    n_groups = n_nights * n_stars

    estimates = np.full(n_rows, np.nan)
    estimate_errors = np.full(n_rows, np.nan)
    recorded = {}

    for color_bands, user_spellings in users.items():
        color_name = _COLOR_BAND_SEPARATOR.join(color_bands)
        absent = [band for band in color_bands if band not in spellings]
        if absent:
            logger.warning(
                f"The color {color_name} cannot be estimated for "
                f"passband(s) {user_spellings}: there are no observations in "
                f"{absent}."
            )
            continue

        in_color = np.isin(passbands, [s for b in color_bands for s in spellings[b]])

        # A plain table of just what the zero-point fits need, so that
        # nothing they write reaches the caller's table. The rows are in
        # the caller's order: grouping happens on a further copy below, and
        # the row numbers carry the results back.
        work = Table(
            {
                "file": np.asarray(observed_mags["file"])[in_color],
                "passband": passbands[in_color],
                "ra": _to_degree_array(observed_mags["ra"])[in_color],
                "dec": _to_degree_array(observed_mags["dec"])[in_color],
                "mag": mags[in_color],
                "error": errors[in_color],
                "row": np.flatnonzero(in_color),
            }
        ).group_by("file")

        try:
            for band in color_bands:
                for spelling in spellings[band]:
                    transform_to_catalog(
                        work,
                        spelling,
                        obs_mag_col="mag",
                        obs_error_column="error",
                        cat_name=cat_name,
                        cat_color=color_bands,
                        vary=("z",),
                        # The zero point is checked against the caller's
                        # expectations when the transform itself is fit.
                        expected={},
                        search_radius=search_radius,
                        # So that color_cat is filled only for stars matched
                        # well enough to help define the relation.
                        match_radius=_FIT_MATCH_RADIUS,
                    )
        except UnsupportedPassbandError as e:
            logger.warning(
                f"The color {color_name} cannot be estimated for "
                f"passband(s) {user_spellings}: {e}"
            )
            continue

        rows = np.asarray(work["row"])
        zero_pointed = np.full(n_rows, np.nan)
        zero_pointed[rows] = np.asarray(work["mag"]) + np.asarray(work["z"])
        color_cat = np.full(n_rows, np.nan)
        color_cat[rows] = np.asarray(work["color_cat"])

        band_means = []
        well_measured = np.ones(n_groups, dtype=bool)
        for band in color_bands:
            in_band = np.isin(passbands, spellings[band])
            band_means.append(
                _night_means(
                    np.where(in_band, zero_pointed, np.nan), errors, group, n_groups
                )
            )
            # Judged on the instrumental magnitudes, so that an image whose
            # zero point could not be fit does not count against its stars.
            well_measured &= _well_measured(
                in_band,
                mags,
                errors,
                group,
                n_groups,
                min_snr,
                min_image_fraction,
            )
        # NaN, for a star that is not well measured, keeps it out of the fit
        # as well as giving it no estimate.
        band_means = [
            tuple(np.where(well_measured, part, np.nan) for part in means)
            for means in band_means
        ]
        first = band_means[0][0]
        color_inst, color_inst_error, weighting = _instrumental_color(*band_means)

        # The catalog color of a star is the same on every one of its rows,
        # so any average of them will do.
        star_color_cat, _ = _group_means(color_cat, group, n_groups)

        color_est = np.full(n_groups, np.nan)
        color_est_error = np.full(n_groups, np.nan)
        any_night_fit = False
        for night_number, night in enumerate(nights):
            this_night = slice(night_number * n_stars, (night_number + 1) * n_stars)
            estimate, error = _fit_color_relation(
                color_inst[this_night],
                color_inst_error[this_night],
                first[this_night],
                star_color_cat[this_night],
                magnitude_term,
                **{name: value[this_night] for name, value in weighting.items()},
            )
            if estimate is None:
                logger.warning(
                    f"The color {color_name} cannot be estimated "
                    f"for night {night}: too few stars have both an "
                    "instrumental and a catalog color, or the ones that do "
                    "are too alike, or too noisy for how much their colors "
                    "differ, to define a relation between the two."
                )
                continue

            any_night_fit = True
            color_est[this_night] = estimate
            color_est_error[this_night] = np.hypot(error, error_floor)

        if not any_night_fit:
            continue

        uses_color = np.isin(passbands, user_spellings)
        estimates[uses_color] = color_est[group[uses_color]]
        estimate_errors[uses_color] = color_est_error[group[uses_color]]
        for spelling in user_spellings:
            # A string rather than a list of the two bands: stacking tables
            # merges their meta, and two equal lists are merged by joining
            # them end to end, where two equal strings are left alone.
            recorded[spelling] = color_name

    result = observed_mags if in_place else observed_mags.copy()
    result[_COLOR_EST_COLUMN] = estimates
    result[_COLOR_EST_COLUMN + _COEFF_ERROR_SUFFIX] = estimate_errors
    result.meta[_COLOR_EST_META_KEY] = recorded

    return result


def transform_to_catalog(
    observed_mags_grouped,
    obs_filter,
    obs_mag_col="mag_inst",
    obs_error_column=None,
    cat_name="apass_dr9",
    cat_filter=None,
    cat_color=None,
    vary=_DEFAULT_VARY,
    expected=None,
    in_place=True,
    fit_diff=True,
    min_fit_sigma=_MIN_FIT_SIGMA,
    search_radius=None,
    match_radius=_CAL_MATCH_RADIUS,
    calibrate_unmatched=False,
    color_fallback_column=None,
):
    """
    Transform a set of instrumental magnitudes to a standard system using either
    instrumental colors or catalog colors.

    The transform is fit separately for each group of
    ``observed_mags_grouped``, using the model in
    `calibrated_from_instrumental`. Only rows whose ``passband`` is
    ``obs_filter`` take part; rows in other passbands keep whatever values
    they already have, so calling this function once per passband builds up a
    single table.

    Parameters
    ----------

    observed_mags_grouped : `astropy.table.Table`
        An astropy table, grouped by whatever separates the data into
        data from just one image. Must have ``ra``, ``dec`` and ``passband``
        columns in addition to the magnitude columns named below.

    obs_filter : str
        Name of the filter in which observations were done. Should be one of the names
        at https://www.aavso.org/filters

    obs_mag_col : str, optional
        Name of the column in ``observed_mags_grouped`` that contains instrumental
        magnitudes.

    obs_error_column : str, optional
        Name of the column in ``observed_mags_grouped`` that contains the error in
        the magnitude. The fit is weighted by these errors, combined with the
        catalog's own errors where it has them, so leaving this out is rarely
        what you want and warns. Stars whose error is not a positive, finite
        number are left out of the fit.

    cat_name : str, optional
        Name of the catalog to calibrate against, either ``"apass_dr9"`` or
        ``"refcat2"``.

    cat_filter : str, optional
        Name of the passband in the catalog that should be matched to the
        instrumental magnitudes, e.g. ``"R"`` or ``"SR"``. This is a passband
        name, not a column name: the column used is ``mag_<cat_filter>``.
        Passband spellings are canonicalized case-insensitively against
        `~stellarphot.settings.aavso_models.AAVSOFilters`, so ``"R"`` and
        ``"Rc"`` are one band rather than two. Defaults to ``obs_filter``,
        and the default can never mismatch; naming a different band
        explicitly is taken as deliberate -- unfiltered or DSLR observations
        (AAVSO's CV and TG) are routinely calibrated against a catalog's V,
        for instance, because there is no observed V to match against
        instead -- but it warns, because calibrating observations in one band
        against catalog magnitudes in another folds the color difference
        between the two bands into the fit, where it comes back as a
        transform coefficient and is applied to every star.

    cat_color : tuple of two strings, optional
        Names of the two passbands whose difference is the color. The color is
        calculated in the order the passbands are given, so ``("R", "I")``
        means the ``mag_R`` column minus the ``mag_I`` column. As with
        ``cat_filter``, these are passband names rather than column names, and
        are canonicalized the same way. Only used, and therefore only
        defaulted or checked, when a color term (``"c"`` or ``"d"``) is being
        fit; see ``vary``. When it is, this defaults to the color
        conventionally used with ``cat_filter``, which is ``("R", "I")`` for R
        and I, ``("B", "V")`` for B and V, and the neighboring Sloan band for
        the Sloan ones. A band with no convention recorded raises rather than
        being guessed at.

    vary : sequence of str, optional
        Which terms of the transform model to fit. Any term not named here is
        held at exactly zero. The terms are ``"a"`` and ``"b"``, the linear
        and quadratic dependence on instrumental magnitude, ``"c"`` and
        ``"d"``, the linear and quadratic dependence on color, and ``"z"``,
        the zero point.

        A star is required to have a catalog color only when ``"c"`` or
        ``"d"`` is being fit. With neither of them in ``vary`` the color does
        not enter the model at all, so stars the catalog has no color for take
        part in the fit and are calibrated like any other; their ``color_cat``
        is still NaN, because their color really is unknown.

    expected : dict, optional
        Range each term is expected to fall in, as ``{term: (low, high)}``.
        These are **not** constraints on the fit: a value outside its range is
        reported in a log message, not clamped. Terms that are not in ``vary`` are
        checked too -- a term held at zero when it should be near 20 is the
        most useful thing this can tell you. Pass an empty dict to check
        nothing. The default is ``{"z": (18, 22)}``.

    in_place : bool, optional
        If ``True``, add the calibrated magnitude to the input table, otherwise return
        a copy.

    fit_diff : bool, optional
        If ``True``, fit the difference between the instrumental and catalog magnitude
        instead of the treating the catalog mag as the dependent variable.

    min_fit_sigma : float, optional
        Floor, in magnitudes, on the per-star uncertainty the fit weights
        by; pass ``0`` to apply no floor. The default of 0.01 is about the
        smallest error credible for a single ground-based measurement tied
        to a survey catalog; see issue #694. Must not be negative, and has
        no effect on an unweighted fit. ``fit_sigma_floor_frac`` reports
        the fraction of each image's stars it applied to. See
        :ref:`calibration-fit-quality` for how the floor reaches the
        reported uncertainties.

    search_radius : `astropy.units.Quantity`, optional
        Angular radius of the cone the calibration catalog is fetched from,
        which must be a single positive, finite angle. Defaults to the radius
        that encloses every position observed in ``obs_filter``, plus a one
        arcminute margin, about the centroid of those positions. See Notes.

    match_radius : `astropy.units.Quantity`, optional
        How far an observed star may be from its nearest catalog entry and
        still be given values derived from that match, which must be a
        single positive, finite angle no smaller than the 1.0 arcsec limit
        for being used in the fit and no larger than the one arcminute
        margin of the catalog cone. Defaults to 2.0 arcsec; ``None`` means
        the default. See Notes.

    calibrate_unmatched : bool, optional
        If ``True``, a star with no catalog entry within ``match_radius`` is
        still given a ``mag_cal`` and ``mag_cal_error``, as long as its
        calibrated magnitude needs nothing from the catalog: either no color
        term (``"c"`` or ``"d"``) is being fit, or ``color_fallback_column``
        supplies its color. The default ``vary`` fits ``"c"``, so on its own
        this has no effect unless ``vary`` is changed too. The star's
        ``mag_cat`` and ``color_cat`` stay NaN. See Notes.

    color_fallback_column : str, optional
        Name of a column of colors to calibrate a star with when the catalog
        has no color for it, such as the ``color_est`` column written by
        `estimate_catalog_colors`. If the table also has a column of that
        name with ``_error`` added, it is taken as the uncertainty of those
        colors. Only used when a color term is being fit. See Notes.

    Returns
    -------

    `astropy.table.Table`
        Table containing the calibrated magnitudes and the fit parameters. The
        columns added are ``mag_cal`` and, if ``obs_error_column`` was given,
        ``mag_cal_error``; the fit coefficients ``a``, ``b``, ``c``, ``d`` and
        ``z``, with their uncertainties ``a_error`` through ``z_error``, the
        goodness of fit ``fit_redchi`` and the four weighting diagnostics
        ``fit_cat_error_missing_frac``, ``fit_max_weight_share``,
        ``fit_excess_scatter`` and ``fit_sigma_floor_frac``; and ``mag_cat``
        and ``color_cat``, the matched catalog magnitude and color.
        ``mag_cal`` is NaN for rows with no usable catalog match, as are all
        of the columns for rows in an image that could not be fit and for
        rows in other passbands that had no value already.

        Every column that comes from the catalog match -- ``mag_cal``,
        ``mag_cal_error``, ``mag_cat`` and ``color_cat`` -- is NaN for a star
        whose nearest catalog entry is more than ``match_radius`` away, so
        none of them can hold the values of an unrelated star. With
        ``calibrate_unmatched=True`` that holds for ``mag_cat`` and
        ``color_cat`` only; see Notes.
        ``mag_cal_error`` is NaN wherever ``mag_cal`` is, for whatever
        reason, so a finite calibrated error always has a calibrated
        magnitude to go with it.

        A star is allowed to *influence the fit*, as opposed to be given
        values derived from its match, only if its nearest catalog entry is
        closer than 1.0 arcsec. So the stars behind the coefficients are a
        subset of those with a finite ``mag_cat``, and the fit cannot be
        reproduced from the output table by selecting on ``mag_cat`` alone.

        ``result.meta["transform_weighting"]`` records how the fit for this
        call was weighted, keyed by ``obs_filter`` exactly as it was passed:
        ``"combined"`` when the catalog had an error column for the band,
        ``"observed"`` when it did not, and ``"unweighted"`` when
        ``obs_error_column`` was not given. Keyed rather than flat because
        this function is called once per passband on the same table, and a
        flat entry would be overwritten by the next call.
        ``result.meta["transform_match_radius"]`` records, keyed the same
        way, the ``match_radius`` the call used, as a Quantity in arcsec.
        Both records describe only the rows of the table: each call drops
        any entry for a passband the table has no rows in, so a table that
        is split by passband, re-transformed and stacked again carries the
        new calls' entries rather than the old ones.

    Notes
    -----

    The coefficients, their uncertainties, ``fit_redchi`` and the four
    weighting diagnostics describe the fit for one image rather than any one
    star, so each of them is the same on every row of that image. A term
    that is not in ``vary`` is held at exactly zero and is therefore known
    exactly, so its uncertainty is exactly zero rather than a small number.

    ``fit_redchi`` is the reduced chi-square of the fit: the summed squared
    residuals, in units of the errors the fit was given, per degree of
    freedom. For an unweighted fit it is the summed squared residuals per
    degree of freedom in **mag squared** instead -- the same name on a
    completely different scale. ``fit_cat_error_missing_frac`` is the
    fraction of the fitted stars whose catalog error the fit could not use;
    ``fit_max_weight_share`` is the largest share of the fit's statistical
    weight held by any one star; and ``fit_excess_scatter`` is the scatter,
    in magnitudes, that would have to be added in quadrature to every star's
    uncertainty to bring ``fit_redchi`` to one -- zero when the residuals
    are already no larger than the errors claim, NaN for an unweighted fit.
    ``fit_sigma_floor_frac`` is the fraction of the fitted stars whose
    quoted uncertainty sat at or below ``min_fit_sigma``, so whose weight
    the floor set rather than their errors -- NaN for an unweighted fit.

    The coefficient uncertainties believe the errors the fit was weighted by
    -- the quoted errors, floored at ``min_fit_sigma`` -- rather than the
    scatter observed about the fit, so stars scattering beyond their quoted
    errors raise ``fit_redchi`` and ``fit_excess_scatter`` instead of
    growing the ``*_error`` columns. The floor reaches the weighting only:
    ``fit_redchi``, ``fit_excess_scatter`` and the measurement term of
    ``mag_cal_error`` are measured against the errors as quoted. An
    unweighted fit quotes no errors to believe, so its uncertainties are
    scaled to the observed scatter, the only scale it has.

    ``mag_cal_error`` combines the star's own measurement error with the
    uncertainty of the fitted transform, correlations between the terms
    included, and then adds the image's ``fit_excess_scatter`` in
    quadrature. The excess term is what makes the column reflect the scatter
    actually observed about the transform rather than only the quoted
    errors, which on real data understated it by several times; the value
    without it is ``sqrt(mag_cal_error**2 - fit_excess_scatter**2)``. Once
    the excess is above zero, ``mag_cal_error`` no longer scales with the
    quoted errors the way ``a_error`` through ``z_error`` do. It is added
    per star, rather than only through the coefficient covariance, because
    measured across many images of one field against two catalogs most of
    it is per-star observational scatter that a target suffers too; the
    catalog's own noise, which reaches a target only through the
    coefficients, is the minority, so the column over-counts that part by
    of order 20 percent and under-counts by about as much when a catalog
    over-quotes its errors.
    ``mag_cal_error`` is NaN, rather than falling back to the measurement
    error alone, for an image whose fit left no usable covariance behind.

    The catalog is fetched once per call, from a cone centered on the
    centroid of the positions observed in ``obs_filter`` and just large
    enough to hold them all. Rows in other passbands are left out of it:
    they are never matched against the catalog, so sizing the cone from them
    only costs sky -- and the field of view can shift a little between bands
    even when the images are of one target. It used to be a fixed degree
    around whichever star happened to be first in the table, so a call asked
    Vizier for every row of a degree-wide cone however small the field really
    was; see issue #686.
    ``search_radius`` is therefore the way to make the cone deliberately
    *larger* than the field -- to calibrate against stars outside it, say --
    rather than the way to keep a query small, which the default already
    does. A derived cone wider than a degree warns, since a table covering
    that much sky is usually more than one field; passing ``search_radius``
    says the size was intended and silences it.

    Two match limits, rather than one, because a star needs to be matched
    far more surely to help *define* the transform than to *receive* a
    calibrated magnitude from it. Every star within 1.0 arcsec of its
    catalog entry is a candidate for the fit; every star within
    ``match_radius`` is calibrated. The band between them exists for
    variables: the position a target is found at, from VSX say, need not
    agree with its position in the calibration catalog to within an arcsec
    -- V2480 Cyg's VSX position is 1.73 arcsec from its APASS DR9 entry --
    and a star that far off is matched well enough to be calibrated, just
    not well enough to calibrate everything else against. So if the
    comparison stars have calibrated magnitudes and the target does not,
    its catalog entry is probably further away than ``match_radius``, and
    raising ``match_radius`` is the remedy. The fit limit is not a keyword:
    widening it changes which stars the coefficients come from, which is a
    different decision.

    Some objects have no catalog entry to match at all -- a supernova, a
    nova, an asteroid. With no color term in the fit, a star's calibrated
    magnitude is its instrumental magnitude run through the image's
    transform and uses nothing from its own catalog entry, so
    ``calibrate_unmatched=True`` gives such an object a ``mag_cal`` and
    ``mag_cal_error`` like any other star in the image. It is opt-in
    because an unmatched detection is far more often a mismatched or
    spurious source than a real object the catalog lacks. When a color term
    is being fit the calibrated magnitude does need the star's color, which
    an unmatched star does not have, so it stays NaN unless
    ``color_fallback_column`` supplies one.

    The colors in ``color_fallback_column`` are used only for stars the
    catalog has no color for: a star with no entry within ``match_radius``,
    or one whose entry lacks a magnitude in one of the ``cat_color`` bands.
    A catalog color always wins where there is one. Because the fit uses
    only stars with a catalog color, fallback colors never influence the
    coefficients; they are applied, not fit to. ``color_cat`` remains the
    catalog's color and so stays NaN for a star calibrated this way. The
    uncertainty of a fallback color, when the table gives one, reaches
    ``mag_cal_error`` through the color terms; a catalog color is still
    taken as exact (issue #691). The two keywords are independent: the
    fallback supplies a color, and ``calibrate_unmatched`` decides whether
    a star with no catalog entry is calibrated at all, so a supernova needs
    both.

    How the fit is weighted, what the floor does and does not reach, how to
    read the diagnostics and the coefficient uncertainties, and what the
    catalog contributes that none of these columns carry are described in
    the user guide under :ref:`calibration-fit-quality`.
    """
    if obs_error_column is None:
        warnings.warn(
            "No error column was given, so the fit will be unweighted. Error "
            "weighting is important in this fit; pass obs_error_column unless "
            "you are sure you want an unweighted fit.",
            AstropyUserWarning,
            stacklevel=2,
        )

    if isinstance(vary, str):
        raise TypeError(
            f"vary must be a sequence of term names, not the string {vary!r}. "
            f"Did you mean {(vary,)!r}?"
        )

    if not np.isfinite(min_fit_sigma) or min_fit_sigma < 0:
        raise ValueError(
            "min_fit_sigma must be a finite, non-negative number, got "
            f"{min_fit_sigma}. Pass 0 to apply no floor to the uncertainties "
            "the fit weights by."
        )

    # Checked here rather than where the cone is actually drawn, so that an
    # argument the caller can fix without leaving their keyboard does not cost
    # them a Vizier round trip first.
    if search_radius is not None:
        search_radius = _validate_angle(
            "search_radius",
            search_radius,
            "Leave it out to search the field the observations cover.",
        )

    if match_radius is None:
        match_radius = _CAL_MATCH_RADIUS
    match_radius = _validate_angle(
        "match_radius",
        match_radius,
        f"Leave it out to use the default of {_CAL_MATCH_RADIUS}.",
    )
    # A star in the fit is promised a finite mag_cat, which a match radius
    # tighter than the fit limit would break. Equality is fine: the fit cut
    # is strict and the calibration cut is not.
    if match_radius < _FIT_MATCH_RADIUS:
        raise ValueError(
            f"match_radius must be at least {_FIT_MATCH_RADIUS}, the "
            "distance within which a star is used in the fit, got "
            f"{match_radius!r}."
        )
    # The catalog cone reaches only _CATALOG_RADIUS_MARGIN beyond the observed
    # field, so a wider match radius could match an edge star whose true
    # counterpart is outside the cone to the nearest fetched entry instead,
    # with nothing in the output to say so.
    if match_radius > _CATALOG_RADIUS_MARGIN:
        raise ValueError(
            f"match_radius must be no more than {_CATALOG_RADIUS_MARGIN}, the "
            "margin by which the catalog cone extends beyond the observed "
            f"field, got {match_radius!r}."
        )

    if cat_name not in _CATALOG_NAMES:
        raise ValueError(
            f"Unknown catalog name {cat_name}. Must be one of "
            f"{' or '.join(repr(name) for name in _CATALOG_NAMES)}."
        )

    # Preserve the order the caller gave, minus any duplicates.
    vary = tuple(dict.fromkeys(vary))
    _check_known_terms(vary, "vary")
    if not vary:
        raise ValueError(
            "vary must name at least one term of the transform model to fit; "
            f"the terms are {list(_COEFF_NAMES)}."
        )

    # Whether a star needs a catalog color at all. Only the color terms use
    # one, and ``vary`` is fixed for the whole call, so this is settled once
    # rather than per image.
    fitting_color = bool({"c", "d"} & set(vary))

    # Checked against every term, varied or not. A term that is not fit sits
    # at exactly zero, and a fixed term sitting outside the range it is
    # expected in is precisely what the caller needs to hear about -- leaving
    # ``z`` out of ``vary`` pins the zero point at zero, which is never right.
    expected = dict(_DEFAULT_EXPECTED if expected is None else expected)
    _check_known_terms(expected, "expected")

    base_params = lmfit.Parameters()
    for name in _COEFF_NAMES:
        # No min or max: the expected ranges above are checked after the fit,
        # never imposed on it, and a term that is not varied is fixed outright
        # rather than boxed into a narrow range.
        base_params.add(name, value=0.0, vary=name in vary)

    n_rows = len(observed_mags_grouped)
    in_passband = np.asarray(observed_mags_grouped["passband"] == obs_filter)
    if not in_passband.any():
        raise ValueError(
            f"No rows with passband {obs_filter!r} in the table; it contains "
            f"{sorted(set(observed_mags_grouped['passband']))}."
        )

    # After the check above, not before it: a caller who got the observed
    # passband name wrong should hear about the name rather than about the
    # mismatch that follows from it, and checking the table before commenting
    # on the arguments is the better order in any case.
    cat_filter = _canonical_passband(obs_filter if cat_filter is None else cat_filter)
    if _canonical_passband(obs_filter) != cat_filter:
        # The default can never land here -- it is always obs_filter itself
        # -- so a mismatch only happens when the caller named a different
        # catalog band on purpose. That is a real workflow: unfiltered and
        # DSLR observations (AAVSO's CV and TG) have no observed V to match,
        # so they are calibrated against a catalog's V instead. What has to
        # be said out loud is the cost of doing that -- the color difference
        # between the two bands becomes part of the fit, comes back as a
        # transform coefficient, and is applied to every star -- not whether
        # it is allowed.
        warnings.warn(
            f"Observed passband {obs_filter!r} and catalog passband "
            f"{cat_filter!r} are different bands. The color difference "
            "between the two bands becomes part of the fit, which reports it "
            "as a transform coefficient and applies it to every star in the "
            "image. This is fine for a deliberate choice, such as "
            "calibrating unfiltered or DSLR observations against a catalog's "
            "V, but if that was not the intent, leave cat_filter out to use "
            "the observed passband, or transform the observations to the "
            "catalog's band first.",
            AstropyUserWarning,
            stacklevel=2,
        )

    if cat_color is not None:
        cat_color = tuple(_canonical_passband(band) for band in cat_color)
    elif cat_filter in _CONVENTIONAL_COLOR:
        cat_color = _CONVENTIONAL_COLOR[cat_filter]
    elif fitting_color:
        raise UnsupportedPassbandError(
            "No color is conventionally used with passband "
            f"{cat_filter!r}, so cat_color must be given explicitly. The "
            f"passbands with a default are {sorted(_CONVENTIONAL_COLOR)}."
        )
    # else: no color term is being fit, so the model never evaluates a color
    # and cat_color is left `None` -- nothing needs it, so nothing is guessed.

    # A fallback color only matters when a color term is being fit, so that
    # is the only time it is looked at -- or checked.
    use_fallback_column = color_fallback_column is not None and fitting_color
    fallback_error_column = None
    if use_fallback_column:
        if color_fallback_column not in observed_mags_grouped.colnames:
            raise ValueError(
                f"color_fallback_column {color_fallback_column!r} is not a "
                "column of the table. Run estimate_catalog_colors first to "
                f"add {_COLOR_EST_COLUMN!r}, or leave the keyword out."
            )
        if (
            color_fallback_column + _COEFF_ERROR_SUFFIX
            in observed_mags_grouped.colnames
        ):
            fallback_error_column = color_fallback_column + _COEFF_ERROR_SUFFIX

        # `estimate_catalog_colors` records which color it estimated for each
        # passband. A color in other bands than the ones this fit uses would
        # be applied without complaint and be wrong, so it is refused here.
        # A column that came from anywhere else has no record and is trusted.
        recorded = observed_mags_grouped.meta.get(_COLOR_EST_META_KEY, {})
        if color_fallback_column == _COLOR_EST_COLUMN and obs_filter in recorded:
            estimated = tuple(
                _canonical_passband(band)
                for band in recorded[obs_filter].split(_COLOR_BAND_SEPARATOR)
            )
            if estimated != cat_color:
                raise ValueError(
                    f"The {_COLOR_EST_COLUMN} column holds the color "
                    f"{estimated} for passband {obs_filter!r}, but this fit "
                    f"uses the color {cat_color}. Leave cat_color out to use "
                    "the conventional color, or leave color_fallback_column "
                    "out."
                )

    # Output is built at the length of the whole table, not of the rows in
    # this passband, and is written by row index below. Rows that are never
    # written keep the NaN they start with.
    coefficients = {name: np.full(n_rows, np.nan) for name in _COEFF_NAMES}
    coefficient_errors = {name: np.full(n_rows, np.nan) for name in _COEFF_NAMES}
    fit_redchis = np.full(n_rows, np.nan)
    fit_diagnostics = {
        name: np.full(n_rows, np.nan) for name in _FIT_DIAGNOSTIC_COLUMNS
    }
    cal_mags = np.full(n_rows, np.nan)
    cal_errors = np.full(n_rows, np.nan)
    cat_mags = np.full(n_rows, np.nan)
    cat_colors = np.full(n_rows, np.nan)

    # Only the rows in this passband: they are the only ones ever matched
    # against the catalog, so they are the only ones the cone has to cover.
    field_center, field_radius = _observed_field(
        observed_mags_grouped["ra"][in_passband],
        observed_mags_grouped["dec"][in_passband],
    )
    if search_radius is None:
        search_radius = field_radius
        if search_radius > _WIDE_FIELD_RADIUS:
            warnings.warn(
                "The catalog will be fetched from a cone of radius "
                f"{field_radius.to(u.degree):.2f} about the center of the "
                "observed positions, and a query that wide may be slow. That "
                "is usually a table holding more than one field; if it is "
                "not, pass search_radius to set the cone size yourself.",
                AstropyUserWarning,
                stacklevel=2,
            )

    if cat_name == "apass_dr9":
        cat = apass_dr9(
            field_center, radius=search_radius, clip_by_frame=False, padding=0
        )
        cat = cat.passband_columns(
            passbands=["B", "V", "R", "I", "SR", "SG", "SI"],
            transformer=transform_apass_bands,
        )
    elif cat_name == "refcat2":
        cat = refcat2(
            field_center, radius=search_radius, clip_by_frame=False, padding=0
        )
        cat = cat.passband_columns(
            # Catalog native passbands will be automatically
            # made to.
            passbands=["B", "V", "R", "I"],
            transformer=transform_refcat2_bands,
        )
    else:  # pragma: no cover
        # Unreachable: `cat_name` was checked against `_CATALOG_NAMES` above.
        # Kept so that adding a name to that tuple without adding a branch
        # here fails loudly rather than leaving `cat` unbound.
        raise ValueError(f"No catalog fetch is implemented for {cat_name}.")

    # cat_filter and cat_color are only ever strings until this point, so a
    # band the catalog cannot supply -- "U" against apass_dr9 or refcat2,
    # neither of which fetches it, is the obvious way -- would otherwise
    # survive every check above and die in a bare `KeyError` only after the
    # Vizier round trip just spent had already paid for it. Checked here,
    # right after the columns exist to check against, so the message can name
    # both what is missing and what the catalog actually has.
    needed_bands = {cat_filter} if cat_color is None else {cat_filter, *cat_color}
    available_bands = sorted(
        name.removeprefix("mag_")
        for name in cat.colnames
        if name.startswith("mag_") and not name.startswith("mag_error_")
    )
    missing_bands = sorted(needed_bands - set(available_bands))
    if missing_bands:
        raise UnsupportedPassbandError(
            f"The {cat_name} catalog has no magnitude for passband(s) "
            f"{missing_bands}, needed for cat_filter or cat_color. It has "
            f"magnitudes for {available_bands}."
        )

    cat_coords = SkyCoord(cat["ra"], cat["dec"], unit="degree")
    if cat_color is None:
        # No color term is being fit -- see fitting_color above -- so the
        # model never evaluates a color. An all-NaN column keeps color_cat
        # honestly reporting "unknown" for every star, rather than a
        # difference nobody asked for and the catalog was never checked to
        # support.
        cat["color"] = np.full(len(cat), np.nan)
    else:
        cat["color"] = cat[f"mag_{cat_color[0]}"] - cat[f"mag_{cat_color[1]}"]

    # Grouping a table reorders its rows, and the result below is either the
    # grouped table itself or a straight copy of it, so group number ``i``
    # is always rows ``group_bounds[i]:group_bounds[i + 1]`` of the output.
    group_bounds = observed_mags_grouped.groups.indices

    # Made before the loop rather than after it so that the results can be
    # written to it. The loop reads only ``observed_mags_grouped`` and never
    # this table, and the invariant documented just above is what makes
    # writing by row index legal.
    result = observed_mags_grouped if in_place else observed_mags_grouped.copy()

    # What is fit is the difference between an observed magnitude and a catalog
    # one, so both of them contribute to how well each star constrains the fit,
    # and the catalog says which of its stars it knows poorly. Weighting by the
    # two errors combined is what replaced the magnitude cut this function used
    # to make: a faint star is down-weighted by its errors rather than dropped
    # at a limit drawn across the whole field.
    #
    # The column is decided once here rather than per image, since it depends
    # only on the catalog and the band, and so is the log message when there
    # is none -- it is a fact about the call rather than about any one image.
    cat_error_column = f"mag_error_{cat_filter}"
    if obs_error_column is None:
        # The fit is unweighted at the caller's request, and was warned about
        # above. Weighting it by the catalog errors alone would make it a
        # weighted fit nobody asked for, and would quietly change what
        # ``fit_redchi`` means.
        cat_error_column = None
        weighting_mode = "unweighted"
    elif cat_error_column in cat.colnames:
        weighting_mode = "combined"
    else:
        logger.warning(
            f"The {cat_name} catalog has no {cat_error_column} column, so the "
            "fit is weighted by the errors of the observations alone and how "
            "well the catalog knows each star is not accounted for. Expect "
            "this for the Johnson-Cousins R and I bands, which are transformed "
            "from the catalog's native bands by a transform that does not yet "
            "propagate errors; see issue #685."
        )
        cat_error_column = None
        weighting_mode = "observed"

    for group_number, (file, one_image_all_bands) in enumerate(
        zip(
            observed_mags_grouped.groups.keys,
            observed_mags_grouped.groups,
            strict=True,
        )
    ):
        group_start = group_bounds[group_number]
        in_this_group = in_passband[group_start : group_bounds[group_number + 1]]
        rows = group_start + np.flatnonzero(in_this_group)

        if rows.size == 0:
            # Nothing in this image was taken in this passband.
            continue

        one_image = one_image_all_bands[in_this_group]
        our_coords = SkyCoord(one_image["ra"], one_image["dec"], unit="degree")

        cat_idx, d2d, _ = our_coords.match_to_catalog_sky(cat_coords)

        # Masked catalog entries become NaN here, which keeps them out of the
        # fit. A star's own calibrated magnitude never involves its catalog
        # magnitude, and a missing color only makes ``mag_cal`` NaN when a
        # color term is being fit -- ``model_color`` below covers the case
        # where none is.
        mag_inst = _to_float_array(one_image[obs_mag_col])
        cat_mag = _to_float_array(cat[f"mag_{cat_filter}"][cat_idx])
        color = _to_float_array(cat["color"][cat_idx])

        # The color the model is evaluated with. When no color term is
        # being fit the color is multiplied by zero, but ``0.0 * np.nan``
        # is NaN, which would poison both the residual and the calibrated
        # magnitude of every star the catalog has no color for. A neutral
        # zero stands in for the missing colors instead. `numpy.where`
        # rather than `numpy.nan_to_num`, which would also turn an
        # infinite color into a very large finite one. The raw ``color``
        # is what goes into the ``color_cat`` output column, so a star
        # with no color still reports none.
        model_color = (
            color if fitting_color else np.where(np.isfinite(color), color, 0.0)
        )

        # Whether each star's color really is that star's: its catalog entry
        # is close enough to be the same star, and has a color. Any other
        # star takes the fallback color if there is one. A star closer than
        # the fit limit with a catalog color always has its own color, so
        # nothing the fit uses changes here.
        matched = d2d <= match_radius
        has_catalog_color = matched & np.isfinite(color)
        color_errors = None
        if use_fallback_column:
            fallback_color = _to_float_array(one_image[color_fallback_column])
            from_fallback = ~has_catalog_color & np.isfinite(fallback_color)
            model_color = np.where(from_fallback, fallback_color, model_color)
            if fallback_error_column is not None:
                fallback_error = _to_float_array(one_image[fallback_error_column])
                # An uncertainty that is not a non-negative number is not
                # known, and NaN is how that is carried through; see the
                # same treatment of ``star_errors`` below.
                fallback_error = np.where(
                    np.isfinite(fallback_error) & (fallback_error >= 0),
                    fallback_error,
                    np.nan,
                )
                color_errors = np.where(from_fallback, fallback_error, 0.0)
        else:
            from_fallback = np.zeros(len(color), dtype=bool)

        # Impose some constraints on what is included in the fit
        good_cat = np.isfinite(cat_mag) & (d2d < _FIT_MATCH_RADIUS)
        if fitting_color:
            good_cat = good_cat & np.isfinite(color)
        good_dat = (mag_inst < -3) & (mag_inst > -20) & np.isfinite(mag_inst)

        if obs_error_column is not None:
            errors = _to_float_array(one_image[obs_error_column])
            # A non-positive or non-finite error gives a meaningless weight.
            good_dat = good_dat & np.isfinite(errors) & (errors > 0)

        if cat_error_column is not None:
            cat_error = _to_float_array(cat[cat_error_column][cat_idx])
            # Unlike the observed error just above, a catalog error the fit
            # cannot use -- missing, masked, NaN, zero or negative -- does
            # not mean the star's catalog magnitude is unusable, only that
            # the catalog does not know its own uncertainty for it. Zero
            # stands in for it instead of excluding the star: the weight
            # below is ``1 / hypot(errors, cat_error)``, and ``hypot(x, 0)``
            # is exactly ``x``, so the star ends up weighted by its observed
            # error alone -- the same fallback used when the catalog has no
            # error column for the band at all, just decided star by star
            # instead of for the whole image. This is what keeps a catalog's
            # single-observation stars, reported with zero error, in the fit
            # exactly as they were before catalog-error weighting existed.
            # The mask is kept as well as applied: it is the one place this
            # decision is made, and `_fit_diagnostics` reports it below.
            cat_error_usable = np.isfinite(cat_error) & (cat_error > 0)
            cat_error = np.where(cat_error_usable, cat_error, 0.0)

        # Both halves have to be good for the *same* star. Checking each on
        # its own is not enough: the two sets can be disjoint, which leaves
        # nothing to take the median of below.
        usable = good_dat & good_cat

        if not usable.any():
            logger.warning(f"No good data in {file[0]}")
            continue

        # Drop stars more than a magnitude away from the median difference
        # between the catalog and instrumental magnitudes -- either a bad
        # match or a bad measurement.
        mag_diff = cat_mag - mag_inst
        good = usable & (np.abs(mag_diff - np.median(mag_diff[usable])) < 1)

        if not good.any():
            logger.warning(f"No good data in {file[0]}")
            continue

        # Prep for fitting
        fit_mag = mag_inst[good]
        fit_color = model_color[good]
        offset = fit_mag if fit_diff else 0.0
        fit_data = cat_mag[good] - offset

        if obs_error_column is None:
            # An unweighted fit divides its residuals by nothing, so there is
            # no sigma to floor here and none for the excess-scatter
            # diagnostic below to measure the residuals against.
            sigma = None
            weights = 1.0
        else:
            if cat_error_column is None:
                sigma = errors[good]
            else:
                # In quadrature, because the residual being fit is the
                # difference between a measured magnitude and a catalog one and
                # is uncertain by both of them.
                sigma = np.hypot(errors[good], cat_error[good])

            # Bounded below so that no one star can take the fit over; see
            # ``min_fit_sigma`` and issue #694. The floor goes on the total
            # the fit divides by rather than on either error separately,
            # because it is the total that decides the star's weight -- and
            # on the weights alone: ``sigma`` stays as quoted, because the
            # reported ``fit_redchi`` and ``fit_excess_scatter`` are measured
            # against it below, and a statistic measured against the floor
            # would call badly under-quoted errors healthy.
            weights = 1.0 / np.maximum(sigma, min_fit_sigma)

        # Fewer stars than terms has to be caught before the fit rather than
        # after it: with nothing left to fit, lmfit takes the square root of
        # a negative number working out the uncertainties, and the
        # RuntimeWarning that produces is an outright exception for anyone
        # running with warnings as errors. Whether the terms can be told
        # apart from *each other* is a question about the fit and is asked
        # of it below.
        if fit_mag.size <= len(vary):
            logger.warning(
                f"Fit for {file[0]} is underdetermined: {fit_mag.size} usable "
                f"star(s) cannot determine {len(vary)} term(s) {list(vary)}"
            )
            continue

        params = base_params.copy()
        if params["z"].vary:
            # With every other term at its starting value of zero the model
            # is just the zero point, so the median of the data being fit is
            # the best starting guess available.
            params["z"].set(value=float(np.median(fit_data)))

        # The covariance is reported as the fit was weighted rather than
        # rescaled by the reduced chi-square: the quoted errors are believed,
        # and their disagreement with the observed scatter is reported once,
        # in ``fit_redchi`` and ``fit_excess_scatter``, instead of also being
        # folded silently into the coefficient uncertainties. An unweighted
        # fit quotes no errors to believe, so the observed scatter is the
        # only scale its covariance can take and the rescaling stays on.
        # See issue #690.
        fit_result = lmfit.minimize(
            _transform_residual,
            params,
            method="least_squares",
            args=(fit_mag, fit_color, fit_data, weights),
            scale_covar=sigma is None,
        )

        if not fit_result.success:
            logger.warning(f"Fit did not succeed for {file[0]}: {fit_result.message}")
            continue

        # A fit reports success whether or not the data can tell the terms
        # being fit apart from each other, and the coefficients it lands on
        # are applied to every star in the image, so a successful fit still
        # has to be checked before its results are used.
        underdetermined = _underdetermined_reason(fit_result, vary)
        if underdetermined is not None:
            logger.warning(f"Fit for {file[0]} is underdetermined: {underdetermined}")
            continue

        values = {name: fit_result.params[name].value for name in _COEFF_NAMES}

        # How well the fit pinned each term down, and how well the model it
        # landed on describes the stars. Both are answers about the image
        # rather than about any one star, and both are reported even when
        # they are the only thing that would tell a caller the coefficients
        # below are not worth applying.
        # The covariance matrix comes back too, and ``covar is None`` is
        # this function's single verdict on whether there is a usable one:
        # the calibrated errors below are propagated through it, or are
        # not reported at all.
        uncertainties, covar = _coefficient_uncertainties(fit_result)

        # The expected ranges are a check on the answer, not a constraint on
        # the fit, so a value outside its range is reported and kept.
        out_of_range = [
            f"{term}={values[term]:.4f} is outside the expected range "
            f"{tuple(expected[term])}"
            for term in expected
            if not expected[term][0] <= values[term] <= expected[term][1]
        ]
        if out_of_range:
            logger.warning(f"Fit for {file[0]}: " + "; ".join(out_of_range))

        if covar is None:
            message = (
                f"Fit for {file[0]} left no usable covariance behind, so its "
                "coefficient *_error columns are NaN for this image"
            )
            if obs_error_column is not None:
                message += (
                    ", and so is mag_cal_error, rather than reporting the "
                    "measurement error on its own, which would understate it"
                )
            logger.warning(message + ".")

        if obs_error_column is None:
            star_errors = None
        else:
            # An error that is not a positive, finite number is kept out of
            # the fit above, and must not come back out as a calibrated
            # error either: the AAVSO writer turns only non-finite errors
            # into "na", so a zero would be submitted as a real
            # uncertainty. Done here, before the propagation rather than
            # after it, because `uncertainties` refuses outright to build a
            # value with a negative standard deviation.
            star_errors = np.where(np.isfinite(errors) & (errors > 0), errors, np.nan)

        fit_redchis[rows] = quoted_redchi(fit_result, sigma, weights)

        # How the fit was weighted, rather than where it landed. Written
        # alongside ``fit_redchi`` because they are the numbers that say
        # whether that one can be taken at face value. Before the calibrated
        # errors, which need the excess scatter too.
        diagnostics = _fit_diagnostics(
            fit_result,
            sigma,
            weights,
            None if cat_error_column is None else cat_error_usable[good],
            min_fit_sigma,
        )
        for name, value in diagnostics.items():
            fit_diagnostics[name][rows] = value

        # Calculate calibrated magnitudes for every star in the image, not
        # just the ones the fit used, propagating the uncertainty of the
        # fit into them when there is one to propagate.
        cal_mag, cal_error = _calibrated_with_uncertainty(
            fit_result,
            covar,
            mag_inst,
            model_color,
            star_errors,
            fit_diff,
            diagnostics[_FIT_EXCESS_SCATTER_COLUMN],
            color_errors=color_errors,
        )

        # One cut for everything that comes from the match: a star matched
        # closely enough to be given a calibrated magnitude keeps the
        # catalog magnitude and color it was calibrated against, and a
        # star matched no better than this keeps none of them rather than
        # reporting an unrelated star's values. The comparison is ``<=`` so
        # that a match_radius equal to the fit limit still gives every fitted
        # star a finite mag_cat: the fit cut above is strict, this one is not.
        # ``matched`` was worked out above, where the colors were chosen.
        #
        # An unmatched star can keep its mag_cal only when it needs no color
        # from the match: no color term is fit, or its color came from the
        # fallback column; see Notes.
        if not calibrate_unmatched:
            cal_mag[~matched] = np.nan
        elif fitting_color:
            cal_mag[~(matched | from_fallback)] = np.nan

        for name in _COEFF_NAMES:
            coefficients[name][rows] = values[name]
            coefficient_errors[name][rows] = uncertainties[name]

        cal_mags[rows] = cal_mag
        cat_mags[rows] = np.where(matched, cat_mag, np.nan)
        cat_colors[rows] = np.where(matched, color, np.nan)

        if cal_error is not None:
            # A star with no calibrated magnitude has no calibrated error
            # either, whatever the reason it has no magnitude. That is
            # stronger than repeating the distance cut -- it also covers a
            # star with no instrumental magnitude -- and it is what makes a
            # finite mag_cal_error safe to select rows on.
            cal_error[~np.isfinite(cal_mag)] = np.nan

            cal_errors[rows] = cal_error

    # The keys are inserted in the order the columns are added to the table in.
    output_columns = {}

    # The calibrated errors are worked out in the loop, where the per-star
    # distances and calibrated magnitudes they depend on live. Only the
    # column order is decided here, and it puts them first.
    if obs_error_column is not None:
        output_columns[_CAL_MAG_ERROR_COLUMN] = cal_errors

    output_columns[_CAL_MAG_COLUMN] = cal_mags
    output_columns.update(coefficients)
    output_columns["mag_cat"] = cat_mags
    output_columns["color_cat"] = cat_colors

    # The fit's own account of how it did. Last, because the columns above
    # are what a caller reaches for first and their order predates these.
    output_columns.update(
        {
            name + _COEFF_ERROR_SUFFIX: errors_for_name
            for name, errors_for_name in coefficient_errors.items()
        }
    )
    output_columns[_FIT_REDCHI_COLUMN] = fit_redchis
    for name in _FIT_DIAGNOSTIC_COLUMNS:
        output_columns[name] = fit_diagnostics[name]

    _write_output_columns(result, output_columns, in_passband)

    # Written here, with the columns, so that an exception in the loop above
    # leaves an in-place table without a record of a call that did not
    # complete.
    _record_call_in_meta(result, obs_filter, weighting_mode, match_radius)

    return result
