import warnings

import lmfit
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyUserWarning

from ..catalogs import apass_dr9, refcat2
from .magnitude_system_transforms import transform_apass_bands, transform_refcat2_bands

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
    # For jester, using the transform for "all stars with Rc-Ic < 1.15"
    # from
    # http://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php#Jester2005
    jester_transforms = {
        "B": [1.39, -0.39, 0, 0.21],
        "V": [0.41, 0.59, 0, -0.01],
        "R": [0.41, -0.5, 1.09, -0.23],
        "I": [0.41, -1.5, 2.09, -0.44],
    }

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
        coeff = jester_transforms[output_filter]
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
    cat_filter="R",
    cat_color=("R", "I"),
    vary=_DEFAULT_VARY,
    expected=None,
    in_place=True,
    fit_diff=True,
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
        the magnitude. The fit is weighted by these errors, so leaving this
        out is rarely what you want and warns. Stars whose error is not a
        positive, finite number are left out of the fit.

    cat_name : str, optional
        Name of the catalog to calibrate against, either ``"apass_dr9"`` or
        ``"refcat2"``.

    cat_filter : str, optional
        Name of the passband in the catalog that should be matched to the
        instrumental magnitudes, e.g. ``"R"`` or ``"SR"``. This is a passband
        name, not a column name: the column used is ``mag_<cat_filter>``.

    cat_color : tuple of two strings, optional
        Names of the two passbands whose difference is the color. The color is
        calculated in the order the passbands are given, so ``("R", "I")``
        means the ``mag_R`` column minus the ``mag_I`` column. As with
        ``cat_filter``, these are passband names rather than column names.

    vary : sequence of str, optional
        Which terms of the transform model to fit. Any term not named here is
        held at exactly zero. The terms are ``"a"`` and ``"b"``, the linear
        and quadratic dependence on instrumental magnitude, ``"c"`` and
        ``"d"``, the linear and quadratic dependence on color, and ``"z"``,
        the zero point.

    expected : dict, optional
        Range each term is expected to fall in, as ``{term: (low, high)}``.
        These are **not** constraints on the fit: a value outside its range is
        reported in a warning, not clamped. Terms that are not in ``vary`` are
        checked too -- a term held at zero when it should be near 20 is the
        most useful thing this can tell you. Pass an empty dict to check
        nothing. The default is ``{"z": (18, 22)}``.

    in_place : bool, optional
        If ``True``, add the calibrated magnitude to the input table, otherwise return
        a copy.

    fit_diff : bool, optional
        If ``True``, fit the difference between the instrumental and catalog magnitude
        instead of the treating the catalog mag as the dependent variable.

    Returns
    -------

    `astropy.table.Table`
        Table containing the calibrated magnitudes and the fit parameters. The
        columns added are ``mag_cal`` and, if ``obs_error_column`` was given,
        ``mag_cal_error``; the fit coefficients ``a``, ``b``, ``c``, ``d`` and
        ``z``; and ``mag_cat`` and ``color_cat``, the matched catalog
        magnitude and color. ``mag_cal`` is NaN for rows with no usable
        catalog match, as are all of the columns for rows in an image that
        could not be fit and for rows in other passbands that had no value
        already.

        ``mag_cat`` and ``color_cat`` are taken from the nearest catalog entry
        whatever its distance, so they can hold the values of an unrelated
        star on rows whose ``mag_cal`` is NaN.

    Notes
    -----

    The values in ``mag_cal_error`` are the instrumental errors scaled by how
    much the calibrated magnitude moves when the instrumental one does, which
    is ``1 + a`` when ``fit_diff`` is ``True`` and ``a`` when it is ``False``.
    That is not a propagation of the uncertainty in the fit itself, and so
    understates the true uncertainty.
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

    # Preserve the order the caller gave, minus any duplicates.
    vary = tuple(dict.fromkeys(vary))
    _check_known_terms(vary, "vary")
    if not vary:
        raise ValueError(
            "vary must name at least one term of the transform model to fit; "
            f"the terms are {list(_COEFF_NAMES)}."
        )

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

    # Output is built at the length of the whole table, not of the rows in
    # this passband, and is written by row index below. Rows that are never
    # written keep the NaN they start with.
    coefficients = {name: np.full(n_rows, np.nan) for name in _COEFF_NAMES}
    cal_mags = np.full(n_rows, np.nan)
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

    cat_coords = SkyCoord(cat["ra"], cat["dec"], unit="degree")
    cat["color"] = cat[f"mag_{cat_color[0]}"] - cat[f"mag_{cat_color[1]}"]

    # Grouping a table reorders its rows, and the result below is either the
    # grouped table itself or a straight copy of it, so group number ``i``
    # is always rows ``group_bounds[i]:group_bounds[i + 1]`` of the output.
    group_bounds = observed_mags_grouped.groups.indices

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

        # Masked catalog entries become NaN here, which both keeps them out of
        # the fit and makes the calibrated magnitudes that depend on them NaN.
        mag_inst = _to_float_array(one_image[obs_mag_col])
        cat_mag = _to_float_array(cat[f"mag_{cat_filter}"][cat_idx])
        color = _to_float_array(cat["color"][cat_idx])

        # Impose some constraints on what is included in the fit
        good_cat = np.isfinite(cat_mag) & np.isfinite(color) & (d2d.arcsecond < 1)
        good_dat = (mag_inst < -3) & (mag_inst > -20) & np.isfinite(mag_inst)

        if obs_error_column is not None:
            errors = _to_float_array(one_image[obs_error_column])
            # A non-positive or non-finite error gives a meaningless weight.
            good_dat = good_dat & np.isfinite(errors) & (errors > 0)

        # Both halves have to be good for the *same* star. Checking each on
        # its own is not enough: the two sets can be disjoint, which leaves
        # nothing to take the median of below.
        usable = good_dat & good_cat

        if not usable.any():
            warnings.warn(
                f"No good data in {file[0]}", AstropyUserWarning, stacklevel=2
            )
            continue

        # Drop stars more than a magnitude away from the median difference
        # between the catalog and instrumental magnitudes -- either a bad
        # match or a bad measurement.
        mag_diff = cat_mag - mag_inst
        good = usable & (np.abs(mag_diff - np.median(mag_diff[usable])) < 1)

        if not good.any():
            warnings.warn(
                f"No good data in {file[0]}", AstropyUserWarning, stacklevel=2
            )
            continue

        # Prep for fitting
        fit_mag = mag_inst[good]
        fit_color = color[good]
        offset = fit_mag if fit_diff else 0.0
        fit_data = cat_mag[good] - offset
        weights = 1.0 / errors[good] if obs_error_column is not None else 1.0

        # Fewer stars than terms has to be caught before the fit rather than
        # after it: with nothing left to fit, lmfit takes the square root of a
        # negative number working out the uncertainties, and the RuntimeWarning
        # that produces is an outright exception for anyone running with
        # warnings as errors. Whether the terms can be told apart from *each
        # other* is a question about the fit and is asked of it below.
        if fit_mag.size <= len(vary):
            warnings.warn(
                f"Fit for {file[0]} is underdetermined: {fit_mag.size} usable "
                f"star(s) cannot determine {len(vary)} term(s) {list(vary)}; "
                "the fit has no degrees of freedom left",
                AstropyUserWarning,
                stacklevel=2,
            )
            continue

        params = base_params.copy()
        if params["z"].vary:
            # With every other term at its starting value of zero the model is
            # just the zero point, so the median of the data being fit is the
            # best starting guess available.
            params["z"].set(value=float(np.median(fit_data)))

        fit_result = lmfit.minimize(
            _transform_residual,
            params,
            method="least_squares",
            args=(fit_mag, fit_color, fit_data, weights),
        )

        if not fit_result.success:
            warnings.warn(
                f"Fit did not succeed for {file[0]}: {fit_result.message}",
                AstropyUserWarning,
                stacklevel=2,
            )
            continue

        # A fit reports success whether or not the data can tell the terms
        # being fit apart from each other, and the coefficients it lands on
        # are applied to every star in the image, so a successful fit still
        # has to be checked before its results are used.
        underdetermined = _underdetermined_reason(fit_result, vary)
        if underdetermined is not None:
            warnings.warn(
                f"Fit for {file[0]} is underdetermined: {underdetermined}",
                AstropyUserWarning,
                stacklevel=2,
            )
            continue

        values = {name: fit_result.params[name].value for name in _COEFF_NAMES}

        # The expected ranges are a check on the answer, not a constraint on
        # the fit, so a value outside its range is reported and kept.
        out_of_range = [
            f"{term}={values[term]:.4f} is outside the expected range "
            f"{tuple(expected[term])}"
            for term in expected
            if not expected[term][0] <= values[term] <= expected[term][1]
        ]
        if out_of_range:
            warnings.warn(
                f"Fit for {file[0]}: " + "; ".join(out_of_range),
                AstropyUserWarning,
                stacklevel=2,
            )

        # Calculate calibrated magnitudes for every star in the image, not
        # just the ones the fit used.
        model_coefficients = [values[name] for name in _COEFF_NAMES]
        cal_mag = calibrated_from_instrumental((mag_inst, color), *model_coefficients)
        if fit_diff:
            cal_mag = cal_mag + mag_inst

        # A limit of 1 arcsec here can cause a variable that really is in APASS to not
        # match. An example is V2480 Cyg, whose VSX position is about 1.3 arcsec from
        # its APASS DR9 position.
        cal_mag[d2d.arcsecond > 1.5] = np.nan

        for name in _COEFF_NAMES:
            coefficients[name][rows] = values[name]

        cal_mags[rows] = cal_mag
        cat_mags[rows] = cat_mag
        cat_colors[rows] = color

    result = observed_mags_grouped if in_place else observed_mags_grouped.copy()

    if obs_error_column is not None:
        # How much the calibrated magnitude moves when the instrumental one
        # does. With fit_diff the instrumental magnitude is added back to the
        # model result, so that sensitivity is 1 + a; without it the model is
        # the calibrated magnitude outright and a itself is the factor -- it
        # fits to about 1 rather than about 0. Using the same factor for both
        # would double the reported uncertainty in one of the two modes.
        sensitivity = 1 + coefficients["a"] if fit_diff else coefficients["a"]

        # Scaled from the raw per-call coefficients rather than the merged
        # column, so that rows in other passbands are handled by the merge
        # below instead of being recomputed from another passband's fit.
        raw_errors = _to_float_array(result[obs_error_column])
        scaled_errors = sensitivity * raw_errors

        # An error that is not a positive, finite number is kept out of the
        # fit above, and must not come back out as a calibrated error either:
        # the AAVSO writer turns only non-finite errors into "na", so a zero
        # would be submitted as a real uncertainty.
        scaled_errors[~(np.isfinite(raw_errors) & (raw_errors > 0))] = np.nan

        result[_CAL_MAG_ERROR_COLUMN] = _merge_with_existing(
            result, _CAL_MAG_ERROR_COLUMN, scaled_errors, in_passband
        )

    result[_CAL_MAG_COLUMN] = _merge_with_existing(
        result, _CAL_MAG_COLUMN, cal_mags, in_passband
    )

    for name in _COEFF_NAMES:
        result[name] = _merge_with_existing(
            result, name, coefficients[name], in_passband
        )

    result["mag_cat"] = _merge_with_existing(result, "mag_cat", cat_mags, in_passband)
    result["color_cat"] = _merge_with_existing(
        result, "color_cat", cat_colors, in_passband
    )

    return result
