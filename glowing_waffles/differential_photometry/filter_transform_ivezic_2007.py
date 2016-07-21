from __future__ import print_function, division


def filter_transform(mag_data, output_filter, R=None, B=None,
                     V=None, I=None, g=None, r=None, i=None):
    '''
    Description: This function impliments the transforms in 'A Comparison of SDSS Standard
    Star Catalog for Stripe 82 with Stetson’s Photometric Standards' by Ivezic et all (2007).
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
