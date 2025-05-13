from collections import namedtuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.modeling import fitting, models
from astropy.stats import sigma_clip
from astropy.table import MaskedColumn
from scipy.optimize import curve_fit

from ..core import apass_dr9, refcat2
from .magnitude_system_transforms import transform_apass_bands, transform_refcat2_bands

__all__ = [
    "calibrated_from_instrumental",
    "opts_to_str",
    "calc_residual",
    "filter_transform",
    "calculate_transform_coefficients",
    "transform_magnitudes",
    "transform_to_catalog",
]


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


def opts_to_str(opts):
    """
    Convert the options from a fit to a string.

    Parameters
    ----------

    opts : tuple of float
        Options from a fit.

    Returns
    -------

    str
        String representation of the options.
    """
    opt_names = ["a", "b", "c", "d", "z"]
    names = []
    for name, value in zip(opt_names, opts, strict=True):
        names.append(f"{name}={value:.4f}")
    return ", ".join(names)


def calc_residual(new_cal, catalog):
    """
    Calculate the standard deviations of the residuals between
    the new calibrated magnitude and the catalog magnitude.

    Parameters
    ----------

    new_cal : `numpy.ndarray`
        New calibrated magnitudes.

    catalog : `numpy.ndarray`
        Catalog magnitudes.

    Returns
    -------

    float
        Standard deviation of the residual.

    """
    resid = new_cal - catalog
    return resid.std()


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
            f"Transform {transform} is not known. Must be one of "
            f"{supported_transforms}"
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
    out_mag.description = f"{output_filter}-band magnitude transformed " "from gri"
    return out_mag


def calculate_transform_coefficients(
    input_mag,
    catalog_mag,
    color,
    input_mag_error=None,
    catalog_mag_error=None,
    faintest_mag=None,
    order=1,
    sigma=2.0,
    gain=None,
):
    """
    Calculate linear transform coefficients from input magnitudes to catalog
    magnitudes.

    Parameters
    ----------

    input_mag : `numpy.ndarray` or `astropy.table.Column`
        Input magnitudes; for example, instrumental magnitudes.

    catalog_mag : `numpy.ndarray` or `astropy.table.Column`
        Catalog (or reference) magnitudes; the magnitudes to which the
        input_mag will eventually be transformed.

    color : `numpy.ndarray` or `astropy.table.Column`
        Colors to use in determining transform coefficients.

    input_mag_error : `numpy.ndarray` or `astropy.table.Column`, optional
        Error in input magnitudes. Default is zero.

    catalog_mag_error : `numpy.ndarray` or `astropy.table.Column`, optional
        Error in catalog magnitudes. Default is zero.

    faintest_mag_for_transform : float, optional
        If this is not ``None``, the magnitude of the faintest catalog stars
        to use in computing transform coefficients.

    order : int, optional
        Order of the polynomial fit to use in correcting for color.

    sigma : float, optional
        Value of sigma to use to reject outliers while fitting using
        sigma clipping.

    gain : float, optional
        If not ``None``, adjust the instrumental magnitude by
        -2.5 * log10(gain), i.e. gain correct the magnitude.

    verbose : bool, optional
        If ``True``, print some diagnostic information.

    extended_output : bool, optional
        If ``True``, return additional information.

    Returns
    -------

    filtered_data : `numpy.ma.core.MaskedArray`
        The data, with the mask set ``True`` for the data that was *omitted*
        from the fit.

    model : `astropy.modeling.FittableModel`
        Entries in the model are the coefficients in the fit made to the
        data. Since the model is always a polynomial, these are terms in
        a polynomial in the order of ascending power. In other words, the
        coefficient ``ci`` is the coefficient of the term ``x**i``.

    If ``extended_output=True``, then also return:

    fit_input : tuple
        A tuple of color, magnitude for only the stars brighter than
        ``faintest_mag_for_transform``. These are input to the sigma-clipping
        fitter.

    used_in_fit : tuple
        A tuple of color, magnitude for only the stars brighter than
        ``faintest_mag_for_transform`` that were not sigma-cliped out.

    Notes
    -----

    This function has some pretty serious limitations right now:

    + Errors in the independent variable are ignored.
    + Outliers are rejected using a modified loss function (Huber loss) that
      cannot be modified.
    + No errors are estimated in the calculated transformation coefficients.

    And there is all the stuff that is not listed here...
    """

    if input_mag_error is None:
        input_mag_error = np.zeros_like(input_mag)
    if catalog_mag_error is None:
        catalog_mag_error = np.zeros_like(catalog_mag)

    if gain is None:
        gain = 1.0

    # Independent variable is the color, dependent variable is the
    # difference between the measured (input) magnitude and the catalog
    # magnitude.

    mag_diff = catalog_mag - (input_mag - 2.5 * np.log10(gain))

    # The error is the errors of those combined in quadrature.
    # combined_error = np.sqrt(input_mag_error**2 + catalog_mag_error**2)

    # If both errors are zero then the error is omitted.
    # if (combined_error == 0).all():
    #     dy = None
    # else:
    #     dy = combined_error

    g_init = models.Polynomial1D(order)
    fit = fitting.LinearLSQFitter()
    or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=2, sigma=sigma)

    if faintest_mag is not None:
        bright = catalog_mag < faintest_mag
        try:
            bright = bright.filled(False)
        except AttributeError:
            # Might not have had a masked array...
            pass
    else:
        bright = np.ones_like(mag_diff, dtype="bool")

    bright_index = np.nonzero(bright)

    # get fitted model and filtered data
    or_fitted_model, filtered_data_mask = or_fit(
        g_init, color[bright], mag_diff[bright]
    )

    # Restore the filtered_data to the same size as the input
    # magnitudes. Unmasked values were included in the fit,
    # masked were not, either because they were too faint
    # or because they were sigma clipped out.
    restored_mask = np.zeros_like(mag_diff, dtype="bool")
    restored_mask[bright_index] = filtered_data_mask
    restored_mask[~bright] = True

    restored_filtered = MaskedColumn(mag_diff.copy())
    restored_filtered.mask = restored_mask

    return (restored_filtered, or_fitted_model)


def transform_magnitudes(
    input_mags,
    catalog,
    transform_catalog,
    input_mag_colum="mag_inst_r",
    catalog_mag_column="r_mag",
    catalog_color_column="B-V",
    faintest_mag_for_transform=14,
    sigma=2,
    order=1,
    gain=None,
):
    """
    Calculate catalog magnitudes and transform coefficients
    from instrumental magnitudes.

    Parameters
    ----------

    input_mags : `astropy.table.Table`
        Table which contains a column with instrumental magnitudes, i.e.
        -2.5 * log10(net_counts / exposure_time).

    catalog : `astropy.table.Table`
        Table containing reference catalog of magnitudes and colors.

    transform_catalog : `astropy.table.Table`
        Table containing the reference catalog of magnitudes and colors
        to use in determining the transform coefficients. Can be the
        same table as ``catalog`` if desired.

    input_mag_column : str, optional
        Name of the column in ``input_mags`` with the magnitudes to be
        transformed.

    catalog_mag_column : str, optional
        Name of the column in ``catalog`` with the reference magnitude.

    catalog_color_column : str, optional
        Name of the column in ``catalog`` with color for each star in the
        catalog.

    faintest_mag_for_transform : float, optional
        If this is not ``None``, the magnitude of the faintest catalog stars
        to use in computing transform coefficients.

    sigma : float, optional
        Number of standard deviations to use in rejecting outliers when fitting
        using sigma clipping.

    order : int, optional
        Order of the polynomial to use in fitting magnitude difference/color
        relationship.

    gain : float, optional
        If not ``None``, adjust the instrumental magnitude by
        -2.5 * log10(gain), i.e. gain correct the magnitude.

    Returns
    -------

    our_cat_mags : `astropy.table.Column`
        The calculated catalog magnitudes for the stars in ``input_mags``.

    good_match_all : `numpy.ndarray`
        Boolean array indicating which stars in ``input_mags`` have a match
        in the catalog.

    transforms : `astropy.modeling.FittableModel`
        The coefficients of the transform. The coefficients are in the order
        of ascending power, i.e. the coefficient ``ci`` is the coefficient
        of the term ``x**i``.  Warning: This returns a namedtuple if the fit
        fails.
    """
    catalog_all_coords = SkyCoord(catalog["RAJ2000"], catalog["DEJ2000"], unit="deg")

    transform_catalog_coords = SkyCoord(
        transform_catalog["RAJ2000"], transform_catalog["DEJ2000"], unit="deg"
    )
    input_coords = SkyCoord(input_mags["ra"], input_mags["dec"])

    transform_catalog_index, d2d, _ = match_coordinates_sky(
        input_coords, transform_catalog_coords
    )

    # create a boolean of all of the matches that have a discrepancy of less
    # than 5 arcseconds
    good_match_for_transform = d2d < 2 * u.arcsecond

    catalog_index, d2d, _ = match_coordinates_sky(input_coords, catalog_all_coords)

    good_match_all = d2d < 5 * u.arcsecond

    catalog_all_indexes = catalog_index[good_match_all]

    input_match_mags = input_mags[input_mag_colum][good_match_for_transform]

    catalog_match_indexes = transform_catalog_index[good_match_for_transform]

    catalog_match_mags = transform_catalog[catalog_mag_column][catalog_match_indexes]
    catalog_match_color = transform_catalog[catalog_color_column][catalog_match_indexes]

    good_mags = ~np.isnan(input_match_mags)

    input_match_mags = input_match_mags[good_mags]
    catalog_match_mags = catalog_match_mags[good_mags]
    catalog_match_color = catalog_match_color[good_mags]

    try:
        matched_data, transforms = calculate_transform_coefficients(
            input_match_mags,
            catalog_match_mags,
            catalog_match_color,
            sigma=sigma,
            faintest_mag=faintest_mag_for_transform,
            order=order,
            gain=gain,
        )
    except np.linalg.LinAlgError as e:
        print(f"Danger! LinAlgError: {str(e)}")
        Transform = namedtuple("Transform", ["parameters"])
        transforms = Transform(parameters=(np.nan,) * (order + 1))

    our_cat_mags = input_mags[input_mag_colum][good_match_all] + transforms(
        catalog[catalog_color_column][catalog_all_indexes]
    )

    return our_cat_mags, good_match_all, transforms


def transform_to_catalog(
    observed_mags_grouped,
    obs_filter,
    obs_mag_col="mag_inst",
    obs_error_column=None,
    cat_name="apass_dr9",
    cat_filter="r_mag",
    cat_color=("r_mag", "i_mag"),
    a_delta=0.5,
    a_cen=0,
    b_delta=1e-6,
    c_delta=0.5,
    d_delta=1e-6,
    zero_point_range=(18, 22),
    in_place=True,
    fit_diff=True,
    verbose=True,
):
    """
    Transform a set of instrumental magnitudes to a standard system using either
    instrumental colors or catalog colors.

    Parameters
    ----------

    observed_magnitudes_grouped : `astropy.table.Table`
        An astropy table, grouped by whatever you want that separates the data into
        data from just one image and just one filter.

    obs_mag_col : str
        Name of the column in `observed_magnitudes_grouped` that contains instrumental
        magnitudes.

    obs_filter : str
        Name of the filter in which observations were done. Should be one of the names
        at https://www.aavso.org/filters

    obs_error_column : str, optional
        Name of the column in `observed_magnitudes_grouped` that contains the error in
        the magnitude.

    cat_filter : str
        Name of the filter/passband in catalog that should be matched to the
        instrumental magnitudes.

    cat_color : tuple of two strings
        Names of the two columns in the catalog that should be used to calculate color.
        The magnitude difference will be calculated in the order the filters are given.
        For example, if the value is ``('r_mag', 'i_mag')`` then the calculated color
        will be the ``r_mag`` column minus the ``i_mag`` column.

    a_delta, b_delta, c_delta, d_delta : float, optional
        Range allowed in fitting for each of the parameters ``a``, ``b``, ``c``,
        and ``d``. Use ``1E-6`` to fix a parameter.

    a_cen : float, optional
        Center of range for the fitting parameter ``a``.

    zero_point_range : tuple of float, optional
        Range to which the value of the zero point is restricted in fitting to
        observed magnitudes.

    in_place : bool, optional
        If ``True``, add the calibrated magnitude to the input table, otherwise return
        a copy.

    fit_diff : bool, optional
        If ``True``, fit the difference between the instrumental and catalog magnitude
        instead of the treating the catalog mag as the dependent variable.

    verbose: bool optional
        If ``True``, print additional output.

    Returns
    -------

    `astropy.table.Table`
        Table containing the calibrated magnitudes and the fit parameters.

    """

    if obs_error_column is None and verbose:
        print("are you sure you want to do that? Error weighting is important!")

    fit_bounds_lower = [
        a_cen - a_delta,  # a
        -b_delta,  # b
        -c_delta,  # c
        -d_delta,  # d
        zero_point_range[0],  # z
    ]

    fit_bounds_upper = [
        a_cen + a_delta,  # a
        b_delta,  # b
        c_delta,  # c
        d_delta,  # d
        zero_point_range[1],  # z
    ]

    fit_bounds = (fit_bounds_lower, fit_bounds_upper)
    zero_point_mid = sum(zero_point_range) / 2

    a = []
    b = []
    c = []
    d = []
    z = []

    all_params = [a, b, c, d, z]

    cal_mags = []
    resids = []
    cat_mags = []
    cat_colors = []
    cat = None
    cat_coords = None

    def fake_it(one_image):
        # Accumulate the parameters
        popt = [np.nan] * len(all_params)
        for param, value in zip(all_params, popt, strict=True):
            param.extend([value] * len(one_image))
        nana = [np.nan] * len(one_image)
        cal_mags.extend(nana)
        cat_mags.extend(nana)
        cat_colors.extend(nana)
        resids.extend(nana)

    one_coord = SkyCoord(
        observed_mags_grouped["ra"][0], observed_mags_grouped["dec"][0], unit="degree"
    )
    if cat is None or cat_coords is None:
        if cat_name == "apass_dr9":
            cat = apass_dr9(
                one_coord, radius=1 * u.degree, clip_by_frame=False, padding=0
            )
            cat = cat.passband_columns(
                passbands=["B", "V", "R", "I", "SR", "SG", "SI"],
                transformer=transform_apass_bands,
            )
        elif cat_name == "refcat2":
            cat = refcat2(
                one_coord, radius=1 * u.degree, clip_by_frame=False, padding=0
            )
            cat = cat.passband_columns(
                # Catalog native passbands will be automatically
                # made to.
                passbands=["B", "V", "R", "I"],
                transformer=transform_refcat2_bands,
            )
        else:
            raise ValueError(
                f"Unknown catalog name {cat_name}. "
                "Must be one of 'apass_dr9' or 'refcat2'."
            )

        cat_coords = SkyCoord(cat["ra"], cat["dec"], unit="degree")
        cat["color"] = cat[f"mag_{cat_color[0]}"] - cat[f"mag_{cat_color[1]}"]

    for file, one_image_inp in zip(
        observed_mags_grouped.groups.keys, observed_mags_grouped.groups, strict=True
    ):
        one_image = one_image_inp[one_image_inp["passband"] == obs_filter]
        our_coords = SkyCoord(one_image["ra"], one_image["dec"], unit="degree")

        cat_idx, d2d, _ = our_coords.match_to_catalog_sky(cat_coords)

        mag_inst = one_image[obs_mag_col]
        cat_mag = cat[f"mag_{cat_filter}"][cat_idx]
        color = cat["color"][cat_idx]

        # Impose some constraints on what is included in the fit
        good_cat = ~(
            getattr(color, "mask", False)
            | getattr(cat_mag, "mask", False) & (d2d.arcsecond < 1)
        )
        good_dat = (
            (one_image[obs_mag_col] < -3)
            & (one_image[obs_mag_col] > -20)
            & ~np.isnan(one_image[obs_mag_col])
        )

        if not (any(good_dat) or any(good_cat)):
            print(f"No good data in {file[0]}")
            fake_it(one_image)
            continue

        mag_diff = cat_mag - mag_inst

        if (
            (not (any(good_dat) or any(good_cat)))
            or all(np.isnan(mag_diff))
            or all(getattr(mag_diff, "mask", [False]))
        ):
            print(f"No good data in {file[0]}")
            fake_it(one_image)
            continue

        mag_diff_mask = getattr(mag_diff, "mask", np.array([False]))
        good_data = good_dat & (
            np.abs(mag_diff - np.nanmedian(mag_diff[good_dat & ~mag_diff_mask])) < 1
        )
        try:
            good_data = good_data & ~one_image[obs_mag_col].mask
        except AttributeError:
            pass

        good = good_cat & good_data

        if not any(good):
            print(f"No good data in {file[0]}")
            fake_it(one_image)
            continue

        # Prep for fitting
        init_guess = (a_cen, 0, 0, 0, zero_point_mid)
        X = (mag_inst[good], color[good])
        catm = cat_mag[good]
        catm_mask = getattr(catm, "mask", np.array([False]))
        X1_mask = getattr(X[1], "mask", np.array([False]))
        if catm_mask.any() or X1_mask.any() or np.isnan(X[0]).any():
            print("heck")
        if obs_error_column is not None:
            errors = one_image[obs_error_column][good]
        else:
            errors = None

        if fit_diff:
            offset = X[0]
        else:
            offset = 0

        # Do the fit
        popt, pcov = curve_fit(
            calibrated_from_instrumental,
            X,
            catm - offset,
            p0=init_guess,
            bounds=fit_bounds,
            sigma=errors,
        )

        # Accumulate the parameters
        for param, value in zip(all_params, popt, strict=True):
            param.extend([value] * len(one_image))

        # Calculate and accumulate residual
        residual = calc_residual(calibrated_from_instrumental(X, *popt) + offset, catm)
        resids.extend([residual] * len(one_image))

        # Calculate calibrated magnitudes and accumulate, settings ones with
        # no catalog match to NaN
        X = (mag_inst, color)
        cal_mag = calibrated_from_instrumental(X, *popt)
        if fit_diff:
            cal_mag = cal_mag + X[0]
        bad_match = d2d.arcsecond > 1
        cal_mag[bad_match] = np.nan
        cal_mags.extend(cal_mag)
        cat_mags.extend(cat_mag)
        cat_colors.extend(color)

        # Keep the user entertained....
        print(f"{file[0]} has fit {opts_to_str(popt)} with {residual=:.4f}")

    mag_col_name = obs_mag_col + "_cal"
    if not in_place:
        result = observed_mags_grouped.copy()
    else:
        result = observed_mags_grouped

    result[mag_col_name] = cal_mags
    if obs_error_column is not None:
        result[mag_col_name + "_error"] = (1 + np.asarray(all_params[0])) * result[
            obs_error_column
        ]
    opt_names = ["a", "b", "c", "d", "z"]

    for name, values in zip(opt_names, all_params, strict=True):
        result[name] = values

    result["mag_cat"] = cat_mags
    result["color_cat"] = cat_colors
    return result
