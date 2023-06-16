import numpy as np


__all__ = ['plot_many_factors', 'bin_data', 'scale_and_shift']


def plot_many_factors(photometry, low, high, shift, scale):
    """ Plots many factors of photometry against each other.

    Parameters
    ----------

    photometry : `astropy.table.Table`
        The photometry table to plot.

    low : float
        The lower bound of the y-axis.

    high : float
        The upper bound of the y-axis.

    shift : float
        The amount to shift the data by.

    scale : float
        The amount to scale the data by.

    Returns
    -------

    None
        Added features to the plot directly.
    """
    airmass = photometry['airmass'] / np.mean(photometry['airmass'])
    x = photometry['xcenter'] / np.mean(photometry['xcenter'])
    y = photometry['ycenter'] / np.mean(photometry['ycenter'])
    comp_counts = photometry['comparison counts'] / np.mean(photometry['comparison counts'])
    sky_per_pix = photometry['sky_per_pix_avg'] / np.mean(photometry['sky_per_pix_avg'])
    width = photometry['width'] / np.mean(photometry['width'])

    scale_airmass = scale_and_shift(airmass, scale, 0.75 * shift, pos=False)
    scale_x = scale_and_shift(x, scale, shift, pos=True)
    scale_y = scale_and_shift(y, scale, shift, pos=True)
    scale_sky_pix = scale_and_shift(sky_per_pix, scale, shift, pos=True)
    scale_counts = scale_and_shift(comp_counts, scale, shift, pos=True)
    scale_width = scale_and_shift(width, scale, shift, pos=True)

    grid_y_ticks = np.arange(low, high, 0.02)


def bin_data(data_set, num=3, error_set=None):
    """ Bins data into groups of num.

    Parameters
    ----------

    data_set : array-like
        The data to bin.

    num : int, optional
        The number of data points to bin together. Default is 3.

    error_set : array-like
        The error on the data set. Default is None.

    Returns
    -------

    binned_set : array-like
        The binned data set.

    error : array-like, optional
        The error on the binned data set.
    """
    binned_set = []
    error = []
    for i in range(0, len(data_set), num):
        binned_set.append(data_set[i:i+num].mean())
        if error_set is not None:
            error_bin = error_set[i:i+num]**2
            error.append(error_bin.sum()/num)
    return np.array(binned_set), np.array(error)


def scale_and_shift(data_set, scale, shift, pos=True):
    """ Scales and shifts data set passed in.

    Parameters
    ----------

    data_set : array-like
        The data to scale and shift.

    scale : float
        The amount to scale the data by.

    shift : float
        The amount to shift the data by (in scaled units).

    pos : bool, optional
        Is data displayed in positive or negative direction? Default is True.

    Returns
    -------

    data_set : array-like
        The scaled and shifted data.
    """
    if not pos:
        data_set = 1 - scale * (data_set - data_set.min()) / (data_set.max() - data_set.min())
    else:
        data_set = 1 + scale * (data_set - data_set.min()) / (data_set.max() - data_set.min())

    data_set += shift

    return data_set
