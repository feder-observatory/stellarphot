import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

__all__ = [
    "plot_many_factors",
    "bin_data",
    "scale_and_shift",
    "plot_transit_lightcurve",
]


def plot_many_factors(photometry, shift, scale, ax=None):
    """Plots many factors of photometry against each other.

    Parameters
    ----------

    photometry : `stellarphot.PhotometryData`
        The photometry table to plot.

    shift : float
        The amount to shift the data by.

    scale : float
        The amount to scale the data by.

    ax : `matplotlib.axes.Axes`, optional
        The axes to plot on.

    Returns
    -------

    None
        Added features to the plot directly.
    """
    airmass = photometry["airmass"] / np.mean(photometry["airmass"])
    x = photometry["xcenter"] / np.mean(photometry["xcenter"])
    y = photometry["ycenter"] / np.mean(photometry["ycenter"])
    comp_counts = photometry["comparison counts"] / np.mean(
        photometry["comparison counts"]
    )
    sky_per_pix = photometry["sky_per_pix_avg"] / np.mean(photometry["sky_per_pix_avg"])
    width = photometry["width"] / np.mean(photometry["width"])

    scale_airmass = scale_and_shift(airmass, scale, 0.75 * shift, pos=False)
    scale_x = scale_and_shift(x, scale, shift, pos=True)
    scale_y = scale_and_shift(y, scale, shift, pos=True)
    scale_sky_pix = scale_and_shift(sky_per_pix, scale, shift, pos=True)
    scale_counts = scale_and_shift(comp_counts, scale, shift, pos=True)
    scale_width = scale_and_shift(width, scale, shift, pos=True)

    x_times = (photometry["bjd"] - 2400000 * u.day).jd

    if ax is None:
        ax = plt.gca()

    ax.plot(
        x_times,
        scale_counts,
        ".",
        c="brown",
        label="tot_C_cnts (arbitrarily scaled and shifted)",
        alpha=0.5,
        ms=4,
    )
    ax.plot(
        x_times,
        scale_airmass,
        "c-",
        label="AIRMASS (arbitrarily scaled and shifted)",
        ms=4,
    )
    ax.plot(
        x_times,
        scale_sky_pix,
        c="gold",
        label="Sky/Pixel_T1 (arbitrarily scaled and shifted)",
        ms=4,
    )
    ax.plot(
        x_times,
        scale_width,
        "-",
        c="gray",
        label="Width_T1 (arbitrarily scaled and shifted)",
        ms=4,
    )
    ax.plot(
        x_times,
        scale_x,
        "-",
        c="pink",
        label="X(FITS)_T1 (arbitrarily scaled and shifted)",
        ms=4,
    )
    ax.plot(
        x_times,
        scale_y,
        "-",
        c="lightblue",
        label="Y(FITS)_T1 (arbitrarily scaled and shifted)",
        ms=4,
    )


def bin_data(data_set, num=3, error_set=None):
    """Bins data into groups of num.

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
        binned_set.append(data_set[i : i + num].mean())
        if error_set is not None:
            error_bin = error_set[i : i + num] ** 2
            error.append(error_bin.sum() / num)
    return np.array(binned_set), np.array(error)


def scale_and_shift(data_set, scale, shift, pos=True):
    """Scales and shifts data set passed in.

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
        data_set = 1 - scale * (data_set - data_set.min()) / (
            data_set.max() - data_set.min()
        )
    else:
        data_set = 1 + scale * (data_set - data_set.min()) / (
            data_set.max() - data_set.min()
        )

    data_set += shift

    return data_set


def plot_transit_lightcurve(
    photometry,
    mod,
    tess_info,
    bin_size,
    low=0.82,
    high=1.06,
):
    # These affect spacing of lines on final plot
    scale = 0.15 * (high - low)
    shift = -0.72 * (high - low)
    # (RMS={rel_flux_rms:.5f})
    grid_y_ticks = np.arange(low, high, 0.02)

    date_obs = photometry["date-obs"][0]
    phot_filter = photometry["passband"][0]
    exposure_time = photometry["exposure"][0]

    midpoint = tess_info.transit_time_for_observation(photometry["bjd"])
    start = midpoint - 0.5 * tess_info.duration
    end = midpoint + 0.5 * tess_info.duration

    detrended_by = []
    if not mod.model.airmass_trend.fixed:
        detrended_by.append("Airmass")
    if not mod.model.spp_trend.fixed:
        detrended_by.append("SPP")
    if not mod.model.width_trend.fixed:
        detrended_by.append("Width")

    flux_full_detrend = mod.data_light_curve(detrend_by="all")
    flux_full_detrend_model = mod.model_light_curve(detrend_by="all")
    rel_detrended_flux = flux_full_detrend / np.mean(flux_full_detrend)

    rel_detrended_flux_rms = np.std(rel_detrended_flux)
    rel_model_rms = np.std(flux_full_detrend_model - rel_detrended_flux)

    rel_flux_rms = np.std(mod.data)

    fig, ax = plt.subplots(1, 1, figsize=(8, 11))

    plt.plot(
        (photometry["bjd"] - 2400000 * u.day).jd,
        photometry["normalized_flux"],
        "b.",
        label=f"rel_flux_T1 (RMS={rel_flux_rms:.5f})",
        ms=4,
    )

    plt.plot(
        mod.times,
        flux_full_detrend - 0.04,
        ".",
        c="r",
        ms=4,
        label=f"rel_flux_T1 ({detrended_by})(RMS={rel_detrended_flux_rms:.5f}), "
        f"(bin size={bin_size} min)",
    )

    plt.plot(
        mod.times,
        flux_full_detrend - 0.08,
        ".",
        c="g",
        ms=4,
        label=f"rel_flux_T1 ({detrended_by} with transit fit)(RMS={rel_model_rms:.5f}),"
        f" (bin size={bin_size})",
    )
    plt.plot(
        mod.times,
        flux_full_detrend_model - 0.08,
        c="g",
        ms=4,
        label=f"rel_flux_T1 Transit Model ([P={mod.model.period.value:.4f}], "
        f"(Rp/R*)^2={(mod.model.rp.value)**2:.4f}, \na/R*={mod.model.a.value:.4f}, "
        f"[Tc={mod.model.t0.value + 2400000:.4f}], "
        f"[u1={mod.model.limb_u1.value:.1f}, u2={mod.model.limb_u2.value:.1f})",
    )

    plot_many_factors(photometry, shift, scale)

    plt.vlines(start.jd - 2400000, low, 1.025, colors="r", linestyle="--", alpha=0.5)
    plt.vlines(end.jd - 2400000, low, 1.025, colors="r", linestyle="--", alpha=0.5)
    plt.text(
        start.jd - 2400000,
        low + 0.0005,
        f"Predicted\nIngress\n{start.jd-2400000-int(start.jd - 2400000):.3f}",
        horizontalalignment="center",
        c="r",
    )
    plt.text(
        end.jd - 2400000,
        low + 0.0005,
        f"Predicted\nEgress\n{end.jd-2400000-int(end.jd - 2400000):.3f}",
        horizontalalignment="center",
        c="r",
    )

    plt.ylim(low, high)
    plt.xlabel("Barycentric Julian Date (TDB)", fontname="Arial")
    plt.ylabel("Relative Flux (normalized)", fontname="Arial")
    plt.title(
        f"TIC {tess_info.tic_id}.01   UT{date_obs}\nPaul P. Feder Observatory 0.4m "
        f"({phot_filter} filter, {exposure_time} exp, "
        f"fap {photometry['aperture'][0].value:.0f}"
        f"-{photometry['annulus_inner'][0].value:.0f}"
        f"-{photometry['annulus_outer'][0].value:.0f})\n",
        fontsize=14,
        fontname="Arial",
    )
    plt.legend(loc="upper center", frameon=False, fontsize=8, bbox_to_anchor=(0.6, 1.0))
    ax.set_yticks(grid_y_ticks)
    plt.grid()

    plt.savefig(
        f"TIC{tess_info.tic_id}-01_{date_obs}_Paul-P-Feder-0.4m_{phot_filter}_lightcurve.png",
        facecolor="w",
    )
