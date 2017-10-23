import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.stats import mad_std
from astropy.time import Time

from gatspy.periodic import LombScargleFast


__all__ = ['plot_magnitudes', 'multi_night']


def plot_magnitudes(mags=None, errors=None, times=None,
                    source=None, night=None, ref_mag=0,
                    alpha=0.25, y_range=None):
    """
    Plot one night of magnitude data for one source, overlaying a rolling
    mean and indication of mean/deviation.
    """
    mean = np.nanmean(mags)
    std = np.nanstd(mags)

    working_times = times
    plt.errorbar(working_times, mags, yerr=errors, fmt='o', alpha=alpha,
                 label='night: {}'.format(night))
    plt.xlim(working_times.min(), working_times.max())
    plt.plot(plt.xlim(), [mean, mean], 'k--', )
    plt.plot(plt.xlim(), [mean + std, mean + std], 'k:')
    plt.plot(plt.xlim(), [mean - std, mean - std], 'k:')
    plt.plot(pd.rolling_mean(working_times, 20, center=True),
             pd.rolling_mean(mags, 20, center=True),
             color='gray', linewidth=3)

    if y_range:
        plt.ylim(y_range)
    else:
        # Make sure plot range is at least 0.1 mag...
        min_range = 0.1
        ylim = plt.ylim()
        if ylim[1] - ylim[0] < min_range:
            plt.ylim(mean - min_range / 2, mean + min_range / 2)

    ylim = plt.ylim()
    # Reverse vertical axis so brighter is higher
    plt.ylim((ylim[1], ylim[0]))

    # # Add marker indicating brightness of star.
    # size = 1000./(mean - ref_mag + 0.1)**2
    # plt.scatter([0.8*(plt.xlim()[1]-plt.xlim()[0]) + plt.xlim()[0]],
    #             [0.8*(plt.ylim()[1] - plt.ylim()[0]) + plt.ylim()[0]],
    #             c='red', marker='o', s=size)

    # plt.legend()
    calendar_date = Time(night, format='jd', out_subfmt='date')
    calendar_date.format = 'iso'

    plt.title('night {}'.format(calendar_date))
    plt.xlabel('time (days)')
    return mean, std


def multi_night(sources, unique_nights, night,
                brightest_mag, mags, mag_err,
                uniform_ylim=True):
    """
    Plot magnitude vs time data for several sources over several nights
    """
    number_of_nights = len(unique_nights)

    for source in sources:
        f = plt.figure(figsize=(5 * number_of_nights, 5))

        night_means = []
        night_stds = []
        night_bins = []
        source_mags = mags[source.id - 1]
        if uniform_ylim:
            # Use median to handle outliers.
            source_median = np.median(source_mags[np.isfinite(source_mags)])
            # Use median absolute deviation to get measure of scatter.
            # Helps avoid extremely points.
            source_variation = \
                3 * mad_std(source_mags[np.isfinite(source_mags)])

            # Ensure y range will be at least 0.2 magnitudes
            if source_variation < 0.1:
                half_range = 0.1
            else:
                half_range = source_variation

            y_range = (source_median - half_range, source_median + half_range)
        else:
            # Empty if this option wasn't chosen so that automatic limits
            # will be used.
            y_range = []

        last_axis = None
        for i, this_night in enumerate(unique_nights):
            last_axis = plt.subplot(1, number_of_nights + 1, i + 1,
                                    sharey=last_axis)
            night_mask = (night == this_night)
            night_mean, night_std = \
                plot_magnitudes(mags=mags[source.id - 1][night_mask],
                                errors=mag_err[source.id - 1][night_mask],
                                times=source.bjd_tdb[night_mask],
                                source=source.id,
                                night=this_night,
                                ref_mag=brightest_mag,
                                y_range=y_range)
            night_means.append(night_mean)
            night_stds.append(night_std)
            night_bins.append(this_night)

        plt.subplot(1, number_of_nights + 1, number_of_nights + 1)

        if uniform_ylim:
            f.subplots_adjust(wspace=0)
            plt.setp([a.get_yticklabels() for a in f.axes[1:]], visible=False)

        # Plot indicators of variation, and information about this source.
        # For simplicity, make the x and y range of this plot be 0 to 1.
        x = np.array([0., 1])
        y = x

        # Add invisible line to make plot.
        plt.plot(x, y, alpha=0, label='source {}'.format(source.id))
        night_means = np.array(night_means)

        # Plot bar proportional to Lomb-Scargle power.
        bad_mags = (np.isnan(mags[source.id - 1]) |
                    np.isinf(mags[source.id - 1]))
        bad_errs = (np.isnan(mag_err[source.id - 1]) |
                    np.isinf(mag_err[source.id - 1]))
        bads = bad_mags | bad_errs
        good_mags = ~bads
        model = LombScargleFast().fit(source.bjd_tdb[good_mags],
                                      mags[source.id - 1][good_mags],
                                      mag_err[source.id - 1][good_mags])
        periods, power = model.periodogram_auto(nyquist_factor=100,
                                                oversampling=3)
        max_pow = power.max()

        # print(source, max_pow)
        if max_pow > 0.5:
            color = 'green'
        elif max_pow > 0.4:
            color = 'cyan'
        else:
            color = 'gray'

        bar_x = 0.25
        plt.plot([bar_x, bar_x], [0, max_pow],
                 color=color, linewidth=10, label='LS power')

        plt.legend()

        # Add dot for magnitude of star.
        size = 10000. / np.abs(10**((source_median - brightest_mag) / 2.5))
        plt.scatter([0.8], [0.2], c='red', marker='o', s=size)
        plt.ylim(0, 1)
