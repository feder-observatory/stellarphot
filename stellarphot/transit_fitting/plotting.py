import matplotlib.pyplot as plt

__all__ = ["plot_predict_ingress_egress"]


def plot_predict_ingress_egress(
    ingress_time,
    egress_time,
    end_line=1,
    ingress_x_pos=1,
    egress_x_pos=1,
    labels_y_pos=1,
):
    """
    Plot vertical lines at the ingress and egress times and label them.

    Parameters
    ----------
    ingress_time : float
        the beginning of an exoplanet transit

    egress_time : float
        the end of an exoplanet transit

    end_line : float
        offset to move the vertical lines

    ingress_x_pos : float
        offset to center ingress label

    egress_x_pos : float
        offset to center egress label

    labels_y_pos : float
        offset to move ingress and egress labels

    Returns
    -------

    None
        Directly adds lines and labels to the current plot.
    """
    ymin, ymax = plt.ylim()

    # create a vertical line at the ingress time and label it
    plt.vlines(ingress_time, ymin - end_line, ymax, linestyle=(0, (5, 10)), color="red")
    plt.annotate(
        "Predicted Ingress",
        (ingress_time - ingress_x_pos, ymin - labels_y_pos),
        color="red",
    )

    # create a vertical line at the egress time and label it
    plt.vlines(egress_time, ymin - end_line, ymax, linestyle=(0, (5, 10)), color="red")
    plt.annotate(
        "Predicted Egress",
        (egress_time - egress_x_pos, ymin - labels_y_pos),
        fontsize=10,
        color="red",
    )
