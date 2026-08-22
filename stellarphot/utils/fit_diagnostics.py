"""
Diagnostics of a weighted least-squares fit: how the residuals compare to the
uncertainties the points were quoted with.

Shared by `~stellarphot.utils.magnitude_transforms.transform_to_catalog` and
`~stellarphot.transit_fitting.TransitModelFit`, both of which fit with
``scale_covar=False`` (issues #690 and #699) and report these numbers instead
of letting `lmfit` rescale the covariance to make the reduced chi-square one.
"""

import numpy as np
from scipy.optimize import brentq

__all__ = ["excess_scatter", "quoted_redchi"]


def quoted_redchi(fit_result, sigma, weights):
    """
    Reduced chi-square measured against the uncertainties as quoted.

    Parameters
    ----------

    fit_result : `lmfit.minimizer.MinimizerResult`
        The fit to describe.

    sigma : `numpy.ndarray` or None
        Uncertainty of each point that was fit, as quoted -- before the
        floor, if any. `None` for an unweighted fit, whose ``redchi`` involves no
        sigmas and is reported as `lmfit` computed it.

    weights : `numpy.ndarray` or float
        Weight the fit gave each residual, floor included. Unused when
        ``sigma`` is `None`.

    Returns
    -------
    float
        The reduced chi-square of the reported fit against the quoted
        uncertainties.

    Notes
    -----
    `lmfit`'s ``redchi`` is measured against the sigmas the fit actually
    divided by. A caller that floors those sigmas -- as
    `~stellarphot.utils.magnitude_transforms.transform_to_catalog` does with
    ``min_fit_sigma`` -- would otherwise see that floor understate how far
    the residuals sit from the errors that were quoted: a fit whose quoted
    errors are far too small would report a small, healthy-looking value for
    exactly the case the statistic exists to catch. Undoing the weighting
    and dividing by the raw sigma instead keeps the floor where it belongs:
    on each point's leverage in the fit, and nowhere in the reporting. A
    caller with no floor passes the same sigmas the fit used and gets
    ``fit_result.redchi`` back, to rounding.

    This is also the value `excess_scatter` gates on: whether any scatter
    needs inventing at all is exactly the question of whether this reduced
    chi-square already sits at or below one, so the two share this one
    computation rather than each summing the same quantity independently.
    """
    if sigma is None:
        return fit_result.redchi

    residual = np.asarray(fit_result.residual) / weights
    # The same operand order as `excess_scatter`'s bracket function at zero
    # excess -- ``residual**2 / sigma**2``, not ``(residual / sigma)**2``.
    # The gate there compares this value to one, and a value that disagreed
    # with the bracket by one ULP around it could hand
    # `~scipy.optimize.brentq` two negative endpoints -- a `ValueError`.
    return float(np.sum(residual**2 / sigma**2) / fit_result.nfree)


def excess_scatter(fit_result, sigma, weights, redchi_quoted=None):
    """
    Scatter that would have to be added to every sigma to explain the residuals.

    The value ``s`` at which adding ``s`` in quadrature to each point's
    uncertainty brings the reduced chi-square to one: how far the points sit
    from the model over and above what they claim to be uncertain by. Real
    contributors seen in it, in a magnitude transform, include flat-field
    gradients and the catalog's own photometry, neither of which any
    weighting scheme can fix; see issue #694.

    Reported rather than folded into the weights. A fit that absorbs its own
    excess scatter has a reduced chi-square of one by construction, which
    destroys the one diagnostic that revealed any of this.

    The best-fit residuals are held fixed rather than the model being refit
    with the widened sigmas. Refitting would move them, but a scatter term
    that is the same for every point barely changes where a fit lands -- it
    rescales the weights nearly uniformly -- and holding them fixed keeps this
    a description of the fit that was actually reported.

    Parameters
    ----------

    fit_result : `lmfit.minimizer.MinimizerResult`
        The fit to describe. Its ``residual`` is the weighted residual, i.e.
        already multiplied by ``weights``.

    sigma : `numpy.ndarray` or None
        Uncertainty of each point as quoted, before any floor the caller
        applied, so the excess is measured against what the errors claim
        rather than against the floor. `None` for an unweighted fit.

    weights : `numpy.ndarray` or float
        Weight the fit gave each residual, floor included. Unused when
        ``sigma`` is `None`.

    redchi_quoted : float, optional
        ``quoted_redchi(fit_result, sigma, weights)`` for this fit, if the
        caller already has it -- computed here otherwise. Ignored when
        ``sigma`` is `None`.

    Returns
    -------
    float
        The excess scatter in magnitudes; zero when the residuals are already
        no larger than the errors claim, and NaN for an unweighted fit, whose
        residuals have no errors to be excessive with respect to.

    Notes
    -----
    Callers such as
    `~stellarphot.utils.magnitude_transforms.transform_to_catalog` pass
    ``redchi_quoted`` in so that the value gating this function is the
    exact float they report as ``fit_redchi``,
    rather than a second sum of the same quantity that could round
    differently by a few ULPs around 1.0.
    """
    if sigma is None or fit_result.nfree <= 0:
        # Nothing was divided by anything, so "how far the residuals sit from
        # what the points claim" has no meaning. NaN rather than zero, which
        # would say the errors were checked and found adequate.
        return np.nan

    if redchi_quoted is None:
        redchi_quoted = quoted_redchi(fit_result, sigma, weights)

    if redchi_quoted - 1.0 <= 0.0:
        # The points are already no further from the model than they claim to
        # be uncertain, so no excess is needed and none is invented. The gate
        # is the same computation reported as fit_redchi, not a second sum
        # of the same quantity that could disagree with it by a few ULPs
        # around 1.0 -- and a gate that passed while both bracket endpoints
        # below evaluate negative would be a `ValueError` from
        # `~scipy.optimize.brentq`.
        return 0.0

    # Undo the weighting: lmfit's residual is the model minus the data times
    # the weights, and what is needed here is the difference itself, so that
    # it can be divided by the sigmas as quoted rather than as floored.
    residual = np.asarray(fit_result.residual) / weights

    def reduced_chi_square_less_one(excess):
        return np.sum(residual**2 / (sigma**2 + excess**2)) / fit_result.nfree - 1.0

    # A bracket rather than a guess. The function falls monotonically from a
    # positive value at zero -- the branch above ruled out the alternative --
    # and at this upper bound every ``sigma**2 + excess**2`` is at least
    # ``excess**2``, so the sum is at most ``nfree`` and the function is at
    # most zero. So a root lies between them.
    upper = np.sqrt(np.sum(residual**2) / fit_result.nfree)
    upper_value = reduced_chi_square_less_one(upper)

    if upper_value >= 0.0:
        # Mathematically ``upper_value`` is at most zero, by the argument
        # above -- so a positive value here only means it landed close
        # enough to zero that rounding pushed it over, which happens when
        # ``sigma`` sits many orders of magnitude below the residuals.
        # ``upper`` is then the root already, to within that same rounding,
        # and handing `~scipy.optimize.brentq` two endpoints that evaluate
        # to the same sign would raise `ValueError` instead of finding it.
        return float(upper)

    return float(brentq(reduced_chi_square_less_one, 0.0, upper))
