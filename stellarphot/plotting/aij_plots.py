import matplotlib.pyplot as plt

from ..settings import PhotometryApertures

__all__ = ["seeing_plot"]


def seeing_plot(
    raw_radius,
    raw_counts,
    binned_radius,
    binned_counts,
    HWHM,
    plot_title="",
    file_name="",
    photometry_settings=None,
    figsize=(20, 10),
):
    """
    Show a seeing plot for data from an image with radius on the x axis and
    counts (ADU) on the y axis.

    Parameters
    ----------
    raw_radius : array
        the distance of each pixel from the object of interest

    raw_counts : array
        the counts of each pixel

    binned_radius : array
        pixels grouped by distance from object of interest

    binned_counts : array
        average counts of each group of pixels

    HWHM : number
        half width half max, 1/2 * FWHM

    plot_title : optional, string
        title of plot

    file_name : optional, string
        if entered, file will save as png with this name

    photometry_settings : optional, `stellarphot.settings.PhotometryApertures`
        The aperture settings used to create the plot. If not provided, the
        aperture radius will be set to 4 * HWHM, the inner annulus will be set
        to radius + 10, and the outer annulus will be set to radius + 25, and the
        FWHM will be set to 2 * HWHM.

    figsize : tuple of int, optional
        Size of figure.

    Returns
    -------

    `matplotlib.pyplot.figure`
        The figure object containing the seeing plot.
    """
    if photometry_settings is None:
        radius = 4 * HWHM
        photometry_settings = PhotometryApertures(
            radius=radius,
            inner_annulus=radius + 10,
            outer_annulus=radius + 25,
            fwhm=2 * HWHM,
        )

    radius = photometry_settings.radius
    inner_annulus = photometry_settings.inner_annulus
    outer_annulus = photometry_settings.outer_annulus

    fig = plt.figure(figsize=figsize)
    plt.grid(True)

    # plot the raw radius and raw counts
    plt.plot(
        raw_radius,
        raw_counts,
        linestyle="none",
        marker="s",
        markerfacecolor="none",
        color="blue",
    )

    # plot the binned radius and binned counts
    plt.plot(binned_radius, binned_counts, color="magenta", linewidth="1.0")

    # draw vertical line at HWHM and label it
    plt.vlines(HWHM, -0.2, 1.2, linestyle=(0, (5, 10)), color="#00cc00")
    plt.annotate(
        f"HWHM {HWHM:2.1f}",
        (HWHM, -0.25),
        color="#00cc00",
        horizontalalignment="center",
    )

    # label axis
    plt.xlabel("Radius (pixels)")
    plt.ylabel("ADU")

    # draw vertical line at the radius and label it
    plt.vlines(radius, -0.2, binned_counts[0], color="red")
    plt.annotate(
        f"Radius {radius:2.1f}",
        (radius, -0.25),
        color="red",
        horizontalalignment="center",
    )
    plt.hlines(binned_counts[0], binned_counts[0], radius, color="red")

    # label the source
    plt.annotate(
        "SOURCE",
        (radius, binned_counts[0] + 0.02),
        color="red",
        horizontalalignment="center",
    )

    # draw vertical lines at the background and label it
    plt.vlines(inner_annulus, -0.2, binned_counts[0], color="red")
    plt.vlines(outer_annulus, -0.2, binned_counts[0], color="red")
    plt.hlines(binned_counts[0], inner_annulus, outer_annulus, color="red")
    plt.annotate("BACKGROUND", (inner_annulus, binned_counts[0] + 0.02), color="red")
    plt.annotate(
        f"Back> {inner_annulus:2.1f}",
        (inner_annulus, -0.25),
        color="red",
        horizontalalignment="center",
    )
    plt.annotate(
        f"<Back {outer_annulus:2.1f}",
        (outer_annulus, -0.25),
        color="red",
        horizontalalignment="center",
    )

    # title the plot
    title_string = [f"{plot_title}", f"FWHM:{HWHM*2:.1f} pixels"]
    plt.title("\n".join(title_string))

    # save plot as png
    if file_name:
        safe_name = file_name.replace(" ", "-")
        plt.savefig(f"{safe_name + '-seeing-profile'}.png")
    return fig
