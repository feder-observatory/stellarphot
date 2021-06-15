from collections import namedtuple

import numpy as np

from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from astropy.table import MaskedColumn
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u

__all__ = [
    'filter_transform',
    'calculate_transform_coefficients',
    'transform_magnitudes'
]


def filter_transform(mag_data, output_filter,
                     g=None, r=None, i=None,
                     transform=None):
    '''
    Transform SDSS magnitudes to BVRI using either the transforms from
    Jester et al or Ivezic et al.

    Parameters
    ----------

    mag_data : astropy.table.Table
        Table containing ``g``, ``r`` and ``i`` magnitudes (or at least)
        those required to transform to the desired output filter.
    output_filter : 'B', 'V', 'R' or 'I'
        Filter for which magnitude should be calculated. Note that
        *case matters* here.
    g, r, i : str
        Name of column in table for that magnitude.
    transform : 'jester' or 'ivezic'
        Transform equations to use.
    Description: This function impliments the transforms in
        'A Comparison of SDSS Standard

    Returns
    -------

    astropy.table.Column
        Output transformed magnitudes as a table column

    Notes
    -----

    The transforms implemented in this function are taken from:

    Jester, et al, *The Sloan Digital Sky Survey View of the Palomar-Green Bright Quasar Survey*, AJ 130, p. 873 (2005)
    http://iopscience.iop.org/article/10.1086/432466/meta

    Ivezić et al, *A Comparison of SDSS Standard Star Catalog for Stripe 82 with Stetson's Photometric Standards*,
    The Future Of Photometric, Spectrophotometric And Polarimetric Standardization, ASP Conference Series 364, p. 165 (2007)
    http://aspbooks.org/custom/publications/paper/364-0165.html

    '''
    supported_transforms = ['jester', 'ivezic']
    if transform not in supported_transforms:
        raise ValueError('Transform {} is not known. Must be one of '
                         '{}'.format(transform, supported_transforms))
    transform_ivezic = {
        'B': [0.2628, -0.7952, 1.0544, 0.0268],
        'V': [0.0688, -0.2056, -0.3838, -0.0534],
        'R': [-0.0107, 0.0050, -0.2689, -0.1540],
        'I': [-0.0307, 0.1163, -0.3341, -0.3584]
    }
    base_mag_ivezic = {
        'B': g,
        'V': g,
        'R': r,
        'I': i
    }
    # For jester, using the transform for "all stars with Rc-Ic < 1.15"
    # from
    # http://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php#Jester2005
    jester_transforms = {
        'B': [1.39, -0.39, 0, 0.21],
        'V': [0.41, 0.59, 0, -0.01],
        'R': [0.41, -0.5, 1.09, -0.23],
        'I': [0.41, -1.5, 2.09, -0.44]
    }

    if output_filter not in base_mag_ivezic.keys():
        raise ValueError('the desired filter must be a string R B V or I')

    if transform == 'ivezic':
        if output_filter == 'R' or output_filter == 'I':
            # This will throw a KeyError if the column is missing
            c = mag_data[r] - mag_data[i]

        if output_filter == 'B' or output_filter == 'V':
            # This will throw a KeyError if the column is missing
            c = mag_data[g] - mag_data[r]

        transform_poly = np.poly1d(transform_ivezic[output_filter])
        out_mag = transform_poly(c) + \
            mag_data[base_mag_ivezic[output_filter]]
        # poly1d  ignores masks. Add masks back in here if necessary.
        try:
            input_mask = c.mask
        except AttributeError:
            pass
        else:
            out_mag = np.ma.array(out_mag, mask=input_mask)
    elif transform == 'jester':
        coeff = jester_transforms[output_filter]
        out_mag = (coeff[0] * mag_data[g] + coeff[1] * mag_data[r] +
                   coeff[2] * mag_data[i] + coeff[3])

    out_mag.name = '{}_mag'.format(output_filter)
    out_mag.description = ('{}-band magnitude transformed '
                           'from gri'.format(output_filter))
    return out_mag


def calculate_transform_coefficients(input_mag, catalog_mag, color,
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

    input_mag : numpy array or astropy Table column
        Input magnitudes; for example, instrumental magnitudes.
    catalog_mag : numpy array or astropy Table column
        Catalog (or reference) magnitudes; the magnitudes to which the
        input_mag will eventually be transformed.
    color : numpy array or astropy Table column
        Colors to use in determining transform coefficients.
    input_mag_error : numpy array or astropy Table column, optional
        Error in input magnitudes. Default is zero.
    catalog_mag_error : numpy array or astropy Table column, optional
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

    filtered_data : `~numpy.ma.core.MaskedArray`
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
    combined_error = np.sqrt(input_mag_error**2 + catalog_mag_error**2)

    # If both errors are zero then the error is omitted.
    if (combined_error == 0).all():
        dy = None
    else:
        dy = combined_error

    g_init = models.Polynomial1D(order)
    fit = fitting.LinearLSQFitter()
    or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip,
                                               niter=2, sigma=sigma)

    if faintest_mag is not None:
        bright = catalog_mag < faintest_mag
        try:
            bright = bright.filled(False)
        except AttributeError:
            # Might not have had a masked array...
            pass
    else:
        bright = np.ones_like(mag_diff, dtype='bool')

    bright_index = np.nonzero(bright)

    # get fitted model and filtered data
    or_fitted_model, filtered_data_mask = or_fit(g_init,
                                                 color[bright],
                                                 mag_diff[bright])

    # Restore the filtered_data to the same size as the input
    # magnitudes. Unmasked values were included in the fit,
    # masked were not, either because they were too faint
    # or because they were sigma clipped out.
    restored_mask = np.zeros_like(mag_diff, dtype='bool')
    restored_mask[bright_index] = filtered_data_mask
    restored_mask[~bright] = True

    restored_filtered = MaskedColumn(mag_diff.copy())
    restored_filtered.mask = restored_mask

    return (restored_filtered, or_fitted_model)


def transform_magnitudes(input_mags, catalog,
                         transform_catalog,
                         input_mag_colum='mag_inst_r',
                         catalog_mag_column='r_mag',
                         catalog_color_column='B-V',
                         faintest_mag_for_transform=14,
                         sigma=2,
                         order=1,
                         gain=None):
    """
    Calculate catalog magnitudes and transform coefficients
    from instrumental magnitudes.

    Parameters
    ----------

    input_mags : astropy Table
        Table which contains a column with instrumental magnitudes, i.e.
        -2.5 * log10(net_counts / exposure_time).

    catalog : astropy Table
        Table containing reference catalog of magnitudes and colors.


    transform_catalog : astropy Table
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
    """
    catalog_all_coords = SkyCoord(catalog['RAJ2000'],
                                  catalog['DEJ2000'],
                                  unit='deg')

    transform_catalog_coords = SkyCoord(transform_catalog['RAJ2000'],
                                        transform_catalog['DEJ2000'],
                                        unit='deg')
    input_coords = SkyCoord(input_mags['RA'], input_mags['Dec'])

    transform_catalog_index, d2d, _ = \
        match_coordinates_sky(input_coords, transform_catalog_coords)

    # create a boolean of all of the matches that have a discrepancy of less
    # than 5 arcseconds
    good_match_for_transform = d2d < 2 * u.arcsecond

    catalog_index, d2d, _ = match_coordinates_sky(input_coords,
                                                  catalog_all_coords)

    good_match_all = d2d < 5 * u.arcsecond

    catalog_all_indexes = catalog_index[good_match_all]

    input_match_mags = input_mags[input_mag_colum][good_match_for_transform]

    catalog_match_indexes = transform_catalog_index[good_match_for_transform]

    catalog_match_mags = \
        transform_catalog[catalog_mag_column][catalog_match_indexes]
    catalog_match_color = \
        transform_catalog[catalog_color_column][catalog_match_indexes]

    good_mags = ~np.isnan(input_match_mags)

    input_match_mags = input_match_mags[good_mags]
    catalog_match_mags = catalog_match_mags[good_mags]
    catalog_match_color = catalog_match_color[good_mags]

    try:
        matched_data, transforms = \
            calculate_transform_coefficients(input_match_mags,
                                             catalog_match_mags,
                                             catalog_match_color,
                                             sigma=sigma,
                                             faintest_mag=faintest_mag_for_transform,
                                             order=order,
                                             gain=gain)
    except np.linalg.LinAlgError as e:
        print('Danger! LinAlgError: {}'.format(str(e)))
        Transform = namedtuple('Transform', ['parameters'])
        transforms = Transform(parameters=(np.nan,) * (order + 1))

    our_cat_mags = (input_mags[input_mag_colum][good_match_all] +
                    transforms(catalog[catalog_color_column][catalog_all_indexes]))

    return our_cat_mags, good_match_all, transforms
