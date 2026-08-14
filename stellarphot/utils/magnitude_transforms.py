import logging
import warnings
from contextlib import contextmanager

import lmfit
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyUserWarning
from numpy.linalg import LinAlgError
from scipy.optimize import brentq
from uncertainties import correlated_values, unumpy

from ..catalogs import apass_dr9, refcat2
from ..settings.aavso_models import AAVSOFilters
from .magnitude_system_transforms import transform_apass_bands, transform_refcat2_bands

logger = logging.getLogger(__name__)

__all__ = [
    "calibrated_from_instrumental",
    "filter_transform",
    "transform_to_catalog",
]

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
# knew its own errors at all, whether one star ran the fit, and how far the
# residuals sit from the errors that were quoted.
_FIT_CAT_ERROR_MISSING_COLUMN = "fit_cat_error_missing_frac"
_FIT_MAX_WEIGHT_SHARE_COLUMN = "fit_max_weight_share"
_FIT_EXCESS_SCATTER_COLUMN = "fit_excess_scatter"

# The three as one tuple, mirroring `_COEFF_NAMES` above: `_fit_diagnostics`
# returns a dict keyed by exactly these names, and the per-image write loop
# and the output-column block both iterate over them, so adding a diagnostic
# is an edit here and in that return dict rather than a hunt for every site.
_FIT_DIAGNOSTIC_COLUMNS = (
    _FIT_CAT_ERROR_MISSING_COLUMN,
    _FIT_MAX_WEIGHT_SHARE_COLUMN,
    _FIT_EXCESS_SCATTER_COLUMN,
)

# How close an observed star must be to a catalog entry, in arcseconds. Two
# limits rather than one: a star must match tightly to be allowed to influence
# the fit, and less tightly to be given values derived from its match.
#
# The band between the two is deliberate. A limit of 1 arcsec everywhere can
# stop a variable that really is in the catalog from matching at all -- V2480
# Cyg's VSX position is about 1.3 arcsec from its APASS DR9 position -- and a
# star in that band is well enough matched to calibrate, just not well enough
# to calibrate everything else against.
_FIT_MATCH_ARCSEC = 1.0
_CAL_MATCH_ARCSEC = 1.5

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

    Parameters
    ----------

    values : array-like
        Values to convert. May be a `~astropy.table.MaskedColumn`.

    Returns
    -------
    `numpy.ndarray`
        Float array with NaN wherever the input was masked.
    """
    return np.ma.filled(np.ma.asarray(values, dtype=float), np.nan)


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


def _fit_diagnostics(fit_result, sigma, weights, cat_error_usable):
    """
    Describe how one image's fit was weighted, as opposed to where it landed.

    Three numbers that ``fit_redchi`` cannot supply on its own, all of them
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

    Returns
    -------
    dict
        Keyed by `_FIT_DIAGNOSTIC_COLUMNS`, ready to be written to the rows
        of this image.
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

    return {
        _FIT_CAT_ERROR_MISSING_COLUMN: missing_fraction,
        _FIT_MAX_WEIGHT_SHARE_COLUMN: max_weight_share,
        _FIT_EXCESS_SCATTER_COLUMN: _excess_scatter(fit_result, sigma, weights),
    }


def _excess_scatter(fit_result, sigma, weights):
    """
    Scatter that would have to be added to every sigma to explain the residuals.

    The value ``s`` at which adding ``s`` in quadrature to each star's
    uncertainty brings the reduced chi-square to one: how far the stars sit
    from the model over and above what they claim to be uncertain by. Real
    contributors seen in it include flat-field gradients and the catalog's
    own photometry, neither of which any weighting scheme can fix; see
    issue #694.

    Reported rather than folded into the weights. A fit that absorbs its own
    excess scatter has a reduced chi-square of one by construction, which
    destroys the one diagnostic that revealed any of this.

    The best-fit residuals are held fixed rather than the model being refit
    with the widened sigmas. Refitting would move them, but a scatter term
    that is the same for every star barely changes where a fit lands -- it
    rescales the weights nearly uniformly -- and holding them fixed keeps this
    a description of the fit that was actually reported.

    Parameters
    ----------

    fit_result : `lmfit.minimizer.MinimizerResult`
        The fit to describe. Its ``residual`` is the weighted residual, i.e.
        already multiplied by ``weights``.

    sigma : `numpy.ndarray` or None
        Uncertainty of each star as quoted, before the ``min_fit_sigma``
        floor, so the excess is measured against what the errors claim
        rather than against the floor. `None` for an unweighted fit.

    weights : `numpy.ndarray` or float
        Weight the fit gave each residual, floor included. Unused when
        ``sigma`` is `None`.

    Returns
    -------
    float
        The excess scatter in magnitudes; zero when the residuals are already
        no larger than the errors claim, and NaN for an unweighted fit, whose
        residuals have no errors to be excessive with respect to.
    """
    if sigma is None or fit_result.nfree <= 0:
        # Nothing was divided by anything, so "how far the residuals sit from
        # what the stars claim" has no meaning. NaN rather than zero, which
        # would say the errors were checked and found adequate.
        return np.nan

    # Undo the weighting: lmfit's residual is the model minus the data times
    # the weights, and what is needed here is the difference itself, so that
    # it can be divided by the sigmas as quoted rather than as floored.
    residual = np.asarray(fit_result.residual) / weights

    def reduced_chi_square_less_one(excess):
        return np.sum(residual**2 / (sigma**2 + excess**2)) / fit_result.nfree - 1.0

    if reduced_chi_square_less_one(0.0) <= 0.0:
        # The stars are already no further from the model than they claim to
        # be uncertain, so no excess is needed and none is invented. The gate
        # is the bracket function itself, not a reduced chi-square computed
        # some other way: any other formulation of the same sum can disagree
        # with this one by a few ULPs around 1.0, and a gate that passes
        # while both bracket endpoints are negative is a `ValueError` from
        # `~scipy.optimize.brentq`.
        return 0.0

    # A bracket rather than a guess. The function falls monotonically from a
    # positive value at zero -- the branch above ruled out the alternative --
    # and at this upper bound every ``sigma**2 + excess**2`` is at least
    # ``excess**2``, so the sum is at most ``nfree`` and the function is at
    # most zero. So a root lies between them.
    upper = np.sqrt(np.sum(residual**2) / fit_result.nfree)

    return float(brentq(reduced_chi_square_less_one, 0.0, upper))


def _reported_redchi(fit_result, sigma, weights):
    """
    Reduced chi-square measured against the uncertainties as quoted.

    `lmfit`'s ``redchi`` is measured against the sigmas the fit actually
    divided by, which are floored at ``min_fit_sigma``. Below the floor that
    understates how far the residuals sit from the errors that were quoted
    -- an image whose quoted errors are far too small would report a small,
    healthy-looking value for exactly the case the statistic exists to
    catch. Undoing the weighting and dividing by the raw sigma instead keeps
    the floor where it belongs: on each star's leverage in the fit, and
    nowhere in the reporting.

    Parameters
    ----------

    fit_result : `lmfit.minimizer.MinimizerResult`
        The fit to describe.

    sigma : `numpy.ndarray` or None
        Uncertainty of each star that was fit, as quoted -- before the
        floor. `None` for an unweighted fit, whose ``redchi`` involves no
        sigmas and is reported as `lmfit` computed it.

    weights : `numpy.ndarray` or float
        Weight the fit gave each residual, floor included. Unused when
        ``sigma`` is `None`.

    Returns
    -------
    float
        The reduced chi-square of the reported fit against the quoted
        uncertainties.
    """
    if sigma is None:
        return fit_result.redchi

    residual = np.asarray(fit_result.residual) / weights
    return float(np.sum((residual / sigma) ** 2) / fit_result.nfree)


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


def _calibrated_with_uncertainty(fit_result, covar, mag_inst, color, errors, fit_diff):
    """
    Calibrated magnitude of every star, and how uncertain each one is.

    The returned error carries the star's own measurement error plus the
    uncertainty of the fitted transform, correlations between the transform's
    terms included.

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

        calibrated = calibrated_from_instrumental(
            (star_mag, color), *(uvars[name] for name in _COEFF_NAMES)
        )
        if fit_diff:
            calibrated = calibrated + star_mag

        return unumpy.nominal_values(calibrated), unumpy.std_devs(calibrated)


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
        Smallest uncertainty, in magnitudes, a star is allowed to claim when
        the fit weights it; a star whose combined uncertainty is smaller is
        weighted as if it were this. A least-squares weight goes as one over
        the variance, so without a floor a single star claiming a tiny
        uncertainty can hold most of a fit's weight; see issue #694. The
        default of 0.01 is about the smallest uncertainty credible for a
        single ground-based measurement tied to a survey catalog. Pass ``0``
        to apply no floor -- legitimate when the quoted errors really are
        believable below the default, e.g. stacked photometry of bright
        stars against a catalog with trustworthy millimagnitude errors. Must
        not be negative, and has no effect on an unweighted fit. The floor
        applies to the fit's weighting, and through it to the covariance
        behind ``a_error``...``z_error`` and the transform half of
        ``mag_cal_error`` (see issue #690): a fit whose quoted errors sit
        under the floor reports uncertainties at the floor's scale, which
        overstates them when those tiny errors are genuine -- the case
        passing ``0`` exists for. ``fit_redchi`` and ``fit_excess_scatter``
        are still measured against the errors as quoted.

    Returns
    -------

    `astropy.table.Table`
        Table containing the calibrated magnitudes and the fit parameters. The
        columns added are ``mag_cal`` and, if ``obs_error_column`` was given,
        ``mag_cal_error``; the fit coefficients ``a``, ``b``, ``c``, ``d`` and
        ``z``, with their uncertainties ``a_error`` through ``z_error``, the
        goodness of fit ``fit_redchi`` and the three weighting diagnostics
        ``fit_cat_error_missing_frac``, ``fit_max_weight_share`` and
        ``fit_excess_scatter``; and ``mag_cat`` and ``color_cat``, the matched
        catalog magnitude and color. ``mag_cal`` is NaN for rows with no
        usable catalog match, as are all of the columns for rows in an image
        that could not be fit and for rows in other passbands that had no
        value already.

        Every column that comes from the catalog match -- ``mag_cal``,
        ``mag_cal_error``, ``mag_cat`` and ``color_cat`` -- is NaN for a star
        whose nearest catalog entry is more than 1.5 arcsec away, so none of
        them can hold the values of an unrelated star. ``mag_cal_error`` is
        NaN wherever ``mag_cal`` is, for whatever reason, so a finite
        calibrated error always has a calibrated magnitude to go with it.

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

    Notes
    -----

    The coefficients, their uncertainties and ``fit_redchi`` describe the fit
    for one image rather than any one star, so each of them is the same on
    every row of that image. A term that is not in ``vary`` is held at exactly
    zero and is therefore known exactly, so its uncertainty is exactly zero
    rather than a small number.

    ``fit_redchi`` is the reduced chi-square: the summed squared
    residuals per degree of freedom, i.e. divided by the number of stars fit
    minus the number of terms varied. It is a chi-square only when
    ``obs_error_column`` is given, because only then is anything dividing the
    residuals: the residuals are in units of the errors that were supplied,
    and a value near one says the model misses the stars by about as much as
    they claim to be uncertain. Without an error column the fit is unweighted
    and the same column holds the summed squared residuals per degree of
    freedom in **mag squared** -- the same name and a completely different
    scale, so values from a weighted and an unweighted fit must not be
    compared with each other.

    The fit is weighted by the observed errors and the catalog's own errors
    combined in quadrature, ``1 / sqrt(obs_error**2 + cat_error**2)``, wherever
    the catalog has a ``mag_error_<cat_filter>`` column; by the observed errors
    alone, with a log message naming the band, wherever it does not. A star
    whose catalog error is missing, masked, NaN or not positive is not left
    out of the fit for it -- the catalog simply does not know its own
    uncertainty for that star, so its weight falls back to the observed error
    alone, exactly as if the catalog had no error column at all. Which bands
    have an error depends on the catalog rather than on this function:
    `~stellarphot.CatalogData` builds
    ``mag_error_<band>`` for the bands a catalog measured itself -- B, V, SG,
    SR and SI for APASS DR9, the Sloan bands for refcat2 -- while the
    Johnson-Cousins R and I are added afterwards by
    `~stellarphot.utils.magnitude_system_transforms.transform_apass_bands` and
    `~stellarphot.utils.magnitude_system_transforms.transform_refcat2_bands`,
    which return magnitudes and no errors. So the fallback is the normal case
    for R and I, the bands most often calibrated here, until issue #685
    teaches those transforms to propagate errors. When the fallback is in use,
    ``fit_redchi`` is the column to watch: an image whose stars scatter about
    the transform by more than they claim to be uncertain reports a
    ``fit_redchi`` well above one, and where the catalog's uncertainty is the
    reason, weighting cannot know that but the scatter still shows up there.

    Whichever way it is weighted, the total uncertainty the fit weights each
    star by is bounded below by ``min_fit_sigma``, 0.01 mag by default and
    ``0`` for no floor. Without one, a single star claiming a tiny
    uncertainty can hold most of a fit's weight; see issue #694. The floor
    applies to the weighting -- and therefore to the covariance the
    reported uncertainties come from, which is the fit's as it was actually
    weighted (see below) -- but not to the alarms: ``fit_redchi`` and
    ``fit_excess_scatter`` are measured against the errors as quoted, so an
    image whose quoted errors are far too small still raises the alarm
    those columns exist for. The measurement half of ``mag_cal_error`` is
    the star's own error exactly as quoted, never raised to the floor.

    The three diagnostic columns describe how the fit was weighted, which
    ``fit_redchi`` cannot: it is a ratio, and an image whose errors are wrong
    in the right way reports a healthy-looking one.

    ``fit_cat_error_missing_frac`` is the fraction of the stars in the fit
    whose catalog error the fit could not use. It is 1.0 when the catalog has
    no error column for the band and when the fit is unweighted, both of which
    are the same statement -- nothing was known about any of them. A value
    near one says ``fit_redchi`` is not comparable with another band's:
    APASS DR9 reports an error of exactly zero for most of its B stars in a
    typical field and almost none of its V stars, and the stars it does that
    for are the faint ones, so without the sigma floor the fit would weight
    the worst-measured stars the most heavily.

    ``fit_max_weight_share`` is the largest share of the fit's total
    statistical weight held by any one star, so a fit spread evenly over N
    stars reports ``1/N`` and one star running the fit reports a number near
    one. It is the column that catches the case above, which nothing else in
    the output reveals.

    ``fit_excess_scatter`` is the scatter, in magnitudes, that would have to be
    added in quadrature to every star's uncertainty to bring ``fit_redchi`` to
    one: how far the stars sit from the model over and above what they claim.
    It is zero when the residuals are already no larger than the errors claim,
    and NaN for an unweighted fit, which has no errors for its residuals to be
    excessive with respect to. It is reported rather than folded into the
    weights, because a fit that absorbs its own excess scatter has a reduced
    chi-square of one by construction and can no longer report that anything
    is wrong. Real contributors seen in it include flat-field gradients across
    the field and the catalog's own photometry -- neither of which any
    weighting scheme can repair. Note that it understates the excess when
    ``fit_cat_error_missing_frac`` is large, since the sigmas it is measured
    against are then missing a term rather than merely being small.

    The reported uncertainties believe the errors the fit was weighted by --
    the quoted errors, floored at ``min_fit_sigma`` -- rather than the
    scatter actually observed about the fit: `lmfit`'s ``scale_covar``
    rescaling, which multiplies the covariance by the reduced chi-square so
    that the reported errors track the observed scatter whatever the quoted
    errors said, is turned off. That makes ``fit_redchi`` and the
    ``*_error`` columns independent diagnostics: an image whose stars
    scatter twice as far from the model as they claim reports the
    quoted-error uncertainties and a ``fit_redchi`` near four, where the
    rescaling reported that one observation twice, dressed as two agreeing
    ones. The excess the rescaling used to fold in silently is reported
    instead, explicitly, in ``fit_excess_scatter``, for the caller to act
    on. The price is stated rather than hidden: when the quoted errors are
    systematically wrong, the ``*_error`` columns are wrong with them, and
    ``fit_redchi`` far from one is the alarm that says so. An unweighted
    fit quotes no errors to believe, so it keeps the rescaling: its
    uncertainties are scaled to the observed scatter, the only scale it
    has.

    ``mag_cal_error`` combines the star's own measurement error with the
    uncertainty of the fitted transform, correlations between the terms
    included. The transform term is a significant contribution -- the part a
    plain measurement-error column omits entirely -- that grows as the number
    of fitted stars shrinks, and it is worked out per star because a fit
    predicts best at the centroid of the stars it was fit to. It is NaN,
    rather than falling back to the measurement error alone, for an image
    whose fit left no usable covariance behind.

    A star's catalog entry decides whether it was matched. When a color term
    (``c`` or ``d``) is fit, though, the model also applies that star's
    catalog color, so the color's uncertainty is a real, knowingly omitted
    contribution -- roughly ``c * sigma_color`` -- tracked as issue #691. What
    the catalog reliably contributes is its field-wide scatter about the
    transform, which lands in ``fit_excess_scatter``; and its systematic tie
    to the standard system -- about 0.02 mag for APASS DR9 -- which is
    identical for every star in every image, so a per-star column would
    mislead, appearing to average down by the square root of the number of
    stars.

    Both halves of ``mag_cal_error`` therefore believe the errors as quoted:
    the measurement half is the star's own quoted error propagated, and the
    transform half comes from a covariance weighted by everyone's. So an
    image whose ``fit_redchi`` is far from one is reporting a
    ``mag_cal_error`` that is wrong by roughly the square root of that
    factor, and ``fit_excess_scatter`` is the size of what its quoted inputs
    missed -- which is the reason to read those columns before believing
    this one.
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

    if min_fit_sigma < 0:
        raise ValueError(
            f"min_fit_sigma must not be negative, got {min_fit_sigma}. Pass 0 "
            "to apply no floor to the uncertainties the fit weights by."
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
        raise ValueError(
            "No color is conventionally used with passband "
            f"{cat_filter!r}, so cat_color must be given explicitly. The "
            f"passbands with a default are {sorted(_CONVENTIONAL_COLOR)}."
        )
    # else: no color term is being fit, so the model never evaluates a color
    # and cat_color is left `None` -- nothing needs it, so nothing is guessed.

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

    one_coord = SkyCoord(
        observed_mags_grouped["ra"][0], observed_mags_grouped["dec"][0], unit="degree"
    )
    if cat_name == "apass_dr9":
        cat = apass_dr9(one_coord, radius=1 * u.degree, clip_by_frame=False, padding=0)
        cat = cat.passband_columns(
            passbands=["B", "V", "R", "I", "SR", "SG", "SI"],
            transformer=transform_apass_bands,
        )
    elif cat_name == "refcat2":
        cat = refcat2(one_coord, radius=1 * u.degree, clip_by_frame=False, padding=0)
        cat = cat.passband_columns(
            # Catalog native passbands will be automatically
            # made to.
            passbands=["B", "V", "R", "I"],
            transformer=transform_refcat2_bands,
        )
    else:
        raise ValueError(
            f"Unknown catalog name {cat_name}. Must be one of 'apass_dr9' or 'refcat2'."
        )

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
        raise ValueError(
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

    # Recorded on the result rather than returned separately, keyed by
    # obs_filter exactly as the caller passed it rather than as a flat entry,
    # because this function is called once per passband on the same table --
    # a flat entry would be overwritten by the next call, losing the first
    # passband's mode.
    result.meta.setdefault("transform_weighting", {})[obs_filter] = weighting_mode

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

        # Impose some constraints on what is included in the fit
        good_cat = np.isfinite(cat_mag) & (d2d.arcsecond < _FIT_MATCH_ARCSEC)
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

        # Calculate calibrated magnitudes for every star in the image, not
        # just the ones the fit used, propagating the uncertainty of the
        # fit into them when there is one to propagate.
        cal_mag, cal_error = _calibrated_with_uncertainty(
            fit_result, covar, mag_inst, model_color, star_errors, fit_diff
        )

        # One cut for everything that comes from the match: a star matched
        # closely enough to be given a calibrated magnitude keeps the
        # catalog magnitude and color it was calibrated against, and a
        # star matched no better than this keeps none of them rather than
        # reporting an unrelated star's values. The comparison is ``<=``
        # so that the boundary stays where it has always been.
        matched = d2d.arcsecond <= _CAL_MATCH_ARCSEC
        cal_mag[~matched] = np.nan

        for name in _COEFF_NAMES:
            coefficients[name][rows] = values[name]
            coefficient_errors[name][rows] = uncertainties[name]

        fit_redchis[rows] = _reported_redchi(fit_result, sigma, weights)

        # How the fit was weighted, rather than where it landed. Written
        # alongside ``fit_redchi`` because they are the numbers that say
        # whether that one can be taken at face value.
        for name, value in _fit_diagnostics(
            fit_result,
            sigma,
            weights,
            None if cat_error_column is None else cat_error_usable[good],
        ).items():
            fit_diagnostics[name][rows] = value

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

    return result
