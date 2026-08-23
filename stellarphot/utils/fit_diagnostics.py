"""
Diagnostics of a weighted least-squares fit: how the residuals compare to the
uncertainties the points were quoted with.

Shared by `~stellarphot.utils.magnitude_transforms.transform_to_catalog` and
`~stellarphot.transit_fitting.TransitModelFit`, both of which turn off the
rescaling for a weighted fit (issues #690 and #699) and report these numbers
instead of letting `lmfit` rescale the covariance to make the reduced
chi-square one.
"""

import numpy as np
from scipy.optimize import brentq

__all__ = ["excess_scatter", "quoted_redchi"]


def _unweighted_residual(fit_result, sigma, weights):
    """
    Undo the weighting of ``fit_result.residual`` and drop zero-weight points.

    A weight of zero is `lmfit`'s idiom for excluding a point: its weighted
    residual is zero and its sigma (``1 / weight``) is infinite, so it would
    divide out to ``0 / 0``. Those points contribute nothing to chi-square,
    so they are dropped from the sums instead of divided by. ``weights`` may
    be a scalar.
    """
    residual = np.asarray(fit_result.residual, dtype=float)
    w = np.broadcast_to(np.asarray(weights, dtype=float), residual.shape)
    good = w != 0
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), w.shape)[good]
    return residual[good] / w[good], sigma


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
    on each point's leverage in the fit, and nowhere in the reporting.
    """
    if sigma is None:
        return fit_result.redchi

    residual, sigma = _unweighted_residual(fit_result, sigma, weights)
    # Divide by max(1, nfree), matching lmfit's own redchi convention (its
    # `_calculate_statistics`) so a fit with no degrees of freedom left
    # reports chisqr rather than raising or returning inf.
    return float(np.sum((residual / sigma) ** 2) / max(1, fit_result.nfree))


def excess_scatter(fit_result, sigma, weights):
    """
    Scatter that would have to be added to every sigma to explain the residuals.

    The value ``s`` at which adding ``s`` in quadrature to each point's
    uncertainty brings the reduced chi-square to one: how far the points sit
    from the model over and above what they claim to be uncertain by. In
    `~stellarphot.utils.magnitude_transforms.transform_to_catalog`'s case,
    real contributors seen in it include flat-field gradients and the
    catalog's own photometry, neither of which any weighting scheme can fix;
    see issue #694.

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

    Returns
    -------
    float
        The excess scatter, in the units of ``sigma``; zero when the
        residuals are already no larger than the errors claim, and NaN for
        an unweighted fit, whose residuals have no errors to be excessive
        with respect to.

    Notes
    -----
    Reported rather than folded into the weights. A fit that absorbs its own
    excess scatter has a reduced chi-square of one by construction, which
    destroys the one diagnostic that revealed any of this.

    The best-fit residuals are held fixed rather than the model being refit
    with the widened sigmas. Refitting would move them, but a scatter term
    that is the same for every point barely changes where a fit lands -- it
    rescales the weights nearly uniformly -- and holding them fixed keeps this
    a description of the fit that was actually reported.
    """
    if sigma is None or fit_result.nfree <= 0:
        # Nothing was divided by anything, so "how far the residuals sit from
        # what the points claim" has no meaning. NaN rather than zero, which
        # would say the errors were checked and found adequate.
        return np.nan

    # Undo the weighting: lmfit's residual is the model minus the data times
    # the weights, and what is needed here is the difference itself, so that
    # it can be divided by the sigmas as quoted rather than as floored.
    residual, sigma = _unweighted_residual(fit_result, sigma, weights)

    def reduced_chi_square_less_one(excess):
        return np.sum(residual**2 / (sigma**2 + excess**2)) / fit_result.nfree - 1.0

    if reduced_chi_square_less_one(0.0) <= 0.0:
        # The points are already no further from the model than they claim to
        # be uncertain, so no excess is needed and none is invented.
        return 0.0

    # The function falls monotonically from a positive value at zero. At
    # ``upper`` every ``sigma**2 + excess**2`` is at least ``4 * rms**2``, so
    # the sum is at most ``nfree / 4`` and the function is at most -3/4:
    # comfortably negative, so a root lies between the two.
    upper = 2 * np.sqrt(np.sum(residual**2) / fit_result.nfree)
    return float(brentq(reduced_chi_square_less_one, 0.0, upper))
