from __future__ import print_function, division

import numpy as np
from scipy import optimize

import astropy.units as u
from astropy.coordinates import SkyCoord

__all__ = [
    'filter_transform',
    'standard_magnitude_transform',
    'calculate_transform_coefficients',
]


def filter_transform(mag_data, output_filter, R=None, B=None,
                     V=None, I=None, g=None, r=None, i=None):
    '''
    Transform SDSS magnitudes to BVRI using Ivezic et all (2007).

    Description: This function impliments the transforms in
        'A Comparison of SDSS Standard
    Preconditions: mag_data must be an astropy.table object
        consisting of numerical values,
    output_filter must be a string
        'R', 'B', 'V', or 'I' and for any output filter must be passed a
        corresponding key (arguemnts R, B, V...) to access the necissary
        filter information from mag_data
    Postconditions: returns a

    # #Basic filter transforms from Ivezic et all (2007)

    '''
    if output_filter == 'R':
        if r and i is not None:
            try:
                r_mags = mag_data[r]
            except:
                raise KeyError('key', str(
                    r), 'not found in mag data for r mags')
            try:
                i_mags = mag_data[i]
            except:
                raise KeyError('key', str(
                    i), 'not found in mag data for i mags')
            A = -0.0107
            B = 0.0050
            C = -0.2689
            D = -0.1540
            c = r_mags - i_mags
            R_mag = (A * (c**3)) + (B * (c**2)) + (C * c) + D + r_mags
            R_mag.name = 'R_mag'
            R_mag.description = ('R-band magnitude transformed '
                                 'from r-band and i-band')
            return R_mag
        else:
            raise KeyError(
                'arguemnts r and i must be defined to transform to I filter')

    if output_filter == 'I':
        if r and i is not None:
            try:
                r_mags = mag_data[r]
            except KeyError:
                raise KeyError('key', str(
                    r), 'not found in mag data for r mags')
            try:
                i_mags = mag_data[i]
            except KeyError:
                raise KeyError('key', str(
                    i), 'not found in mag data for i mags')
            A = -0.0307
            B = 0.1163
            C = -0.3341
            D = -0.3584
            c = r_mags - i_mags
            I_mag = (A * (c**3)) + (B * (c**2)) + (C * c) + D + r_mags
            I_mag.name = 'I_mag'
            I_mag.description = ('I-band magnitude transformed '
                                 'from r-band and i-band')
            return I_mag
        else:
            raise KeyError(
                'arguments r and i must be defined to transform to I filter')

    if output_filter == 'B':
        if r and g is not None:
            try:
                r_mags = mag_data[r]
            except KeyError:
                raise KeyError('key', str(
                    r), 'not found in mag data for r mags')
            try:
                g_mags = mag_data[g]
            except KeyError:
                raise KeyError('key', str(
                    i), 'not found in mag data for g mags')
            A = 0.2628
            B = -0.7952
            C = 1.0544
            D = 0.02684
            c = g_mags - r_mags
            B_mag = (A * (c**3)) + (B * (c**2)) + (C * c) + D + r_mags
            B_mag.name = 'B_mag'
            B_mag.description = ('B-band magnitude transformed '
                                 'from r-band and g-band')
            return B_mag
        else:
            raise KeyError(
                'arguemnts r and g must be defined to transform to B filter')

    if output_filter == 'V':
        if r and g is not None:
            try:
                r_mags = mag_data[r]
            except KeyError:
                raise KeyError('key', str(
                    r), 'not found in mag data for r mags')
            try:
                g_mags = mag_data[g]
            except KeyError:
                raise KeyError('key', str(
                    i), 'not found in mag data for g mags')
            A = 0.0688
            B = -0.2056
            C = -0.3838
            D = -0.0534
            c = g_mags - r_mags
            V_mag = (A * (c**3)) + (B * (c**2)) + (C * c) + D + r_mags
            V_mag.name = 'V_mag'
            V_mag.description = ('V-band magnitude transformed '
                                 'from r-band and g-band')
            return V_mag
        else:
            raise KeyError(
                'arguments r and g must be defined to transform to B filter')
    else:
        raise ValueError('the desired filter must be a string R B V or I')


# Define the log-likelihood via the Huber loss function
def huber_loss(m, b, x, y, dy, c=2):
    y_fit = m * x + b
    t = abs((y - y_fit) / dy)
    flag = t > c
    return np.sum((~flag) * (0.5 * t ** 2) - (flag) * c * (0.5 * c - t), -1)


def standard_magnitude_transform(instrumental,
                                 catalog,
                                 for_filter=None,
                                 match_radius=0.5 * u.arcsec,
                                 color_from='catalog',
                                 color_1='B',
                                 color_2='V',):
    """
    Given catalog magnitudes and instrumental magnitudes, calculate transform
    from instrumental to catalog. Colors can be drawn from the catalog or
    from instrumental magnitudes.

    Parameters
    ----------

    instrumental: astropy Table
        Table containing instrumental magnitudes from which transform
        coefficients are to be determined. Each filter should be a column
        whose name is the filter. In addition, there should be a column
        called 'RA' and a column called 'Dec'.
    catalog : astropy Table
        Table containing catalog magnitudes from which transform
        coefficients are to be determined. Each filter should be a column
        whose name is the filter. In addition there should be a column
        called 'RA' and a column called 'Dec'.
    for_filter : str
        Filter for which to calculate the transform. Filter name must be a
        column name in instrumental and catalog.
    match_radius : astropy Quantity  or float
        Determines cutoff for matching coordinates of instrumental
        magnitudes to catalog magnitudes. An on-sky separation of
        less than match_radius counts as a match. If no units are provided
        it is assumed the unit is degrees.
    color_from : str
        Either 'catalog' or 'instrumental', indicates which catalog should be
        used for calculating the color in the transform.
    color_1 : str
        First filter to use for calculating color.
    color_2 : str
        Second color to use for calculating the color.

    """
    tables = {
        'catalog': catalog,
        'instrumental': instrumental
    }

    if not for_filter:
        raise ValueError('Must provide a value for for_filter.')

    for name, table in tables.iteritems():
        if for_filter not in table.colnames:
            raise ValueError('Filter {} not found in {} '
                             'table'.format(for_filter, name))

    instrumental_coords = SkyCoord(ra=instrumental['RA'],
                                   dec=instrumental['Dec'])
    catalog_coords = SkyCoord(ra=catalog['RA'], dec=catalog['Dec'])

    catalog_index, d2d, d3d = \
        instrumental_coords.match_to_catalog_sky(catalog_coords)

    good_match = np.array(d2d < match_radius)

    if not good_match.sum():
        raise ValueError('No matches found between instrumental'
                         ' and catalog tables.')
    if color_from == 'catalog':
        indexes = catalog_index[good_match]
    else:
        indexes = good_match
    color = (tables[color_from][color_1][indexes] -
             tables[color_from][color_2][indexes])

    transforms = {}

    mag_diff = (catalog[for_filter][catalog_index[good_match]] -
                instrumental[for_filter][good_match])

    # dy is error...which is simply added in quadrature, if present
    try:
        catalog_error = catalog['e_' + for_filter]
    except KeyError:
        catalog_error = np.zeros_like(good_match)

    try:
        instrumental_error = instrumental['e_' + for_filter]
    except KeyError:
        instrumental_error = np.zeros_like(good_match)
    dy = np.sqrt(catalog_error[good_match] ** 2 +
                 instrumental_error[good_match] ** 2)

    def f_huber(beta):
        return huber_loss(beta[0], beta[1],
                          x=np.array(color),
                          y=np.array(mag_diff),
                          dy=dy,
                          c=1)

    beta0 = (0.1, 20)
    beta_huber = optimize.minimize(f_huber, beta0, method='Nelder-Mead')
    # This grabs the solution the optimizer found, or raises an error
    # if no solution is found.
    if beta_huber.success:
        transforms[for_filter] = beta_huber.x
    else:
        raise RuntimeError('Optimizer did not converge')

    return transforms


def calculate_transform_coefficients(input_mag, catalog_mag, color,
                                     input_mag_error=None,
                                     catalog_mag_error=None):
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

    Returns
    -------

    zero_point, slope : float
        Zero point and linear color term in transform equation (see notes
        below).

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

    # Independent variable is the color, dependent variable is the
    # difference between the measured (input) magnitude and the catalog
    # magnitude.

    mag_diff = catalog_mag - input_mag

    # The error that is the errors of those combined in quadrature.
    combined_error = np.sqrt(input_mag_error**2 + catalog_mag_error**2)

    # If both errors are zero then the error is omitted.
    if (combined_error == 0).all():
        dy = None
    else:
        dy = combined_error

    def f_huber(beta):
        return huber_loss(beta[0], beta[1],
                          x=np.array(color),
                          y=np.array(mag_diff),
                          dy=dy,
                          c=1)

    # Fairly arbitrary initial guess.
    beta0 = (0.1, 20)

    beta_huber = optimize.minimize(f_huber, beta0, method='Nelder-Mead')
    # This grabs the solution the optimizer found, or raises an error
    # if no solution is found.
    if beta_huber.success:
        transforms = beta_huber.x
    else:
        raise RuntimeError('Optimizer did not converge')

    return transforms
