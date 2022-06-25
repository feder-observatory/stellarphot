import matplotlib.pyplot as plt

__all__ = ['seeing_plot', 'plot_predict_ingress_egress']


def seeing_plot(raw_radius, raw_counts, binned_radius, binned_counts, HWHM,
                plot_title='', file_name='', gap=6, annulus_width=13):
    """
    Show a seeing plot for data from an image with radius on the x axis and counts (ADU) on the y axis.

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
    gap : number
        the distance between the aperture and the inner annulus
    annulus_width : number
        the distance between the inner and outer annulus
    """
    radius = HWHM * 4
    plt.figure(figsize=(20, 10))
    plt.grid(True)
    inner_annulus = radius + gap
    outer_annulus = radius + annulus_width

    # plot the raw radius and raw counts
    plt.plot(raw_radius, raw_counts, linestyle='none',
             marker="s", markerfacecolor='none', color='blue')

    # plot the binned radius and binned counts
    plt.plot(binned_radius, binned_counts, color='magenta', linewidth='1.0')

    # draw vertical line at HWHM and label it
    plt.vlines(HWHM, -0.2, 1.2, linestyle=(0, (5, 10)), color='#00cc00')
    plt.annotate(f"HWHM {HWHM:2.1f}", (HWHM - 1, -0.25),
                 fontsize=15, color='#00cc00')

    # label axis
    plt.xlabel('Radius (pixels)', fontsize=20)
    plt.ylabel('ADU', fontsize=20)

    # draw vertical line at the radius and label it
    plt.vlines(radius, -0.2, binned_counts[0], color='red')
    plt.annotate(f"Radius {radius:2.1f}", (radius - 1, -0.25),
                 fontsize=15, color='red')
    plt.hlines(binned_counts[0], binned_counts[0], radius, color='red')

    # label the source
    plt.annotate(
        'SOURCE', (radius - 2, binned_counts[0] + 0.02), fontsize=15, color='red')

    # draw vertical lines at the background and label it
    plt.vlines(inner_annulus, -0.2, binned_counts[0], color='red')
    plt.vlines(outer_annulus, -0.2, binned_counts[0], color='red')
    plt.hlines(binned_counts[0], inner_annulus, outer_annulus, color='red')
    plt.annotate('BACKGROUND', (inner_annulus,
                                binned_counts[0] + 0.02), fontsize=15, color='red')
    plt.annotate(f"Back> {inner_annulus:2.1f}",
                 (inner_annulus - 1, -0.25), fontsize=15, color='red')
    plt.annotate(f"<Back {outer_annulus:2.1f}",
                 (outer_annulus - 1, -0.25), fontsize=15, color='red')

    # title the plot
    title_string = [f"{plot_title}", f"FWHM:{HWHM*2:.1f} pixels"]
    plt.title('\n'.join(title_string))

    # save plot as png
    if file_name:
        plt.savefig(f"{file_name}.png")


def plot_predict_ingress_egress(ingress_time, egress_time, end_line=1,
                                ingress_x_pos=1, egress_x_pos=1, labels_y_pos=1):
    """
    Parameters
    ----------
    ingress_time : number
        the beginning of an exoplanet transit

    egress_time : number
        the end of an exoplanet transit

    ingress_x_pos : number
        offset to center ingress label

    egress_x_pos : number
        offset to center egress label

    labels_y_pos : number
        offset to move ingress and egress labels
    """
    ymin, ymax = plt.ylim()

    # create a vertical line at the ingress time and label it
    plt.vlines(ingress_time, ymin - end_line, ymax,
               linestyle=(0, (5, 10)), color='red')
    plt.annotate("Predicted Ingress", (ingress_time - ingress_x_pos,
                                       ymin - labels_y_pos), fontsize=10, color='red')

    # create a vertical line at the egress time and label it
    plt.vlines(egress_time, ymin - end_line, ymax,
               linestyle=(0, (5, 10)), color='red')
    plt.annotate("Predicted Egress", (egress_time - egress_x_pos,
                                      ymin - labels_y_pos), fontsize=10, color='red')
