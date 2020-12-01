import numpy as np


def plot_many_factors(photometry, low, high, shift, scale):
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
    binned_set = []
    error = []
    for i in range(0, len(data_set), num):
        binned_set.append(data_set[i:i+num].mean())
        if error_set is not None:
            error_bin = error_set[i:i+num]**2
            error.append(error_bin.sum()/num)
    return np.array(binned_set), np.array(error)


def scale_and_shift(data_set, scale, shift, pos=True):
    if not pos:
        data_set = 1 - scale * (data_set - data_set.min()) / (data_set.max() - data_set.min())

    else:
        data_set = 1 + scale * (data_set - data_set.min()) / (data_set.max() - data_set.min())
    data_set += shift

    return data_set
