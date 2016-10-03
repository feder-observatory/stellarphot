from __future__ import print_function, division

import numpy as np
from scipy import optimize

import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord

__all__ = ['filter_transform']


def filter_transform(mag_data, output_filter, R=None, B=None,
                     V=None, I=None, g=None, r=None, i=None):
    '''
    Transform SDSS magnitudes to BVRI using Ivezic et all (2007).

    Description: This function impliments the transforms in 'A Comparison of SDSS Standard
    Preconditions: mag_data must be an astropy.table object consisting of numerical values,
    output_filter must be a string 'R', 'B', 'V', or 'I' and for any output filter must be passed a
    corresponding key (arguemnts R, B, V...) to access the necissary filter information from
    mag_data
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
            R_mag.description = 'R-band magnitude transformed from r-band and i-band'
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
            I_mag.description = 'I-band magnitude transformed from r-band and i-band'
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
            B_mag.description = 'B-band magnitude transformed from r-band and g-band'
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
            V_mag.description = 'V-band magnitude transformed from r-band and g-band'
            return V_mag
        else:
            raise KeyError(
                'arguments r and g must be defined to transform to B filter')
    else:
        raise ValueError('the desired filter must be a string R B V or I')


def standard_magnitude_transform(instrumental,
                                 catalog,
                                 match_radius=0.5 * u.arcsec,
                                 color_from='catalog',
                                 color_1='B',
                                 color_2='V',
                                 for_filter=None):
    """
    Given catalog magnitudes and instrumental magnitudes, calculate transform
    from instrumental to catalog. Colors can be drawn from the catalog or
    from instrumental magnitudes.

    Parameters
    ----------

    instrumental: astropy Table
        Table containing instrumental magnitudes from which transform
        coefficients are to be determined. Each filter should be a column
        whose name is the filter. In addition there should be a column
        called 'RA' and a column called 'Dec'.
    catalog : astropy Table
        Table containing catalog magnitudes from which transform
        coefficients are to be determined. Each filter should be a column
        whose name is the filter. In addition there should be a column
        called 'RA' and a column called 'Dec'.
    match_radius : astropy Quantity  or float
        Determines cutoff for matching coordinates of instrumental
        magnitudes to catalog magnitudes. An on-sky separation of
        less than match_radius counts as a match. If no units are provided
        it is assumed the unit is degrees.
    """
    i_filters = [name for name in instrumental.colnames
                 if name not in ['RA', 'Dec']]

    c_filters = [name for name in catalog.colnames
                 if name not in ['RA', 'Dec']]

    if not set(i_filters).issubset(set(c_filters)):
        raise ValueError('Some filters in instrumental table not found in '
                         'catalog table.')
    instrumental_coords = SkyCoord(ra=instrumental['RA'],
                                   dec=instrumental['Dec'])
    catalog_coords = SkyCoord(ra=catalog['RA'], dec=catalog['Dec'])

    catalog_index, d2d, d3d = \
        catalog_coords.match_to_catalog_sky(instrumental_coords)

    good_match = d2d < match_radius

    if not good_match.sum():
        raise ValueError('No matches found between instrumental'
                         ' and catalog tables.')


    # Create empty list for the corrections (slope and intercept)
    corrections = []
    # Create empy list for the error in the corrections
    # loop over all images
    for idx in range(aij_mags.shape[1]):
        # create BminusV list
        BminusV = []
        # Create Rminusr list
        Rminusr = []
        # loop over the apass_index and the placement in the apass_index
        for aij_star, el in enumerate(apass_index):
            # check if the aij index corresponds to 18 or 23 (I don't know why)
            if aij_star in [18, 23]:
                # Breakes out of the apass_index loop?
                continue
            # check if the aij star has a corresponding apass match
            if good_match[aij_star]:
                # Make sure the aij stars magnitude in the image isn't friggin
                # huge
                if aij_stars[aij_star].magnitude[idx] < 100:
                    # Add the color of that star (according to apass) to the
                    # bminusv list
                    BminusV.append(apass_color[el])
                    # add the difference between aijs magnitude and apass
                    # transformed magnitude to the Rminusr list
                    Rminusr.append(
                        apass_R_mags[el] - aij_stars[aij_star].magnitude[idx])
        # findes the slope and the slope intercept (and error in) for a plot of Rminusr vs BminusV for that image
        # non huber way of getting slope... slope_intercept, cov =
        # np.polyfit(BminusV, Rminusr, 1, cov=True)
        dy = np.zeros(len(BminusV)) + 0.01
        f_huber = lambda beta: huber_loss(beta[0], beta[1], x=np.array(
            BminusV), y=np.array(Rminusr), dy=dy, c=1)
        beta0 = (0.1, 20)
        beta_huber = optimize.fmin(f_huber, beta0)
        # append corrections data to list
        corrections.append(beta_huber)

