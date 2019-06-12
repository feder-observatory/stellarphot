import matplotlib.pyplot as plt


__all__ = ['seeing_plot']

def seeing_plot(raw_radius, raw_counts, binned_radius, binned_counts, HWHM, plot_title=''):
    radius = HWHM * 4
    plt.figure(figsize=(20,10))
    plt.plot(raw_radius, raw_counts, linestyle='none', marker="s", markerfacecolor='none', color='blue')
    plt.plot(binned_radius, binned_counts, color = 'magenta',linewidth = '4.0')
    plt.vlines(HWHM,-0.2,1.2, linestyle = (0, (5, 10)), color = '#00cc00')
    plt.annotate(f"HWHM {HWHM}", (HWHM - 1 ,-0.25), fontsize = 15, color = '#00cc00')
    plt.grid(True)
    plt.xlabel('Radius (pixels)', fontsize = 20)
    plt.ylabel('ADU', fontsize = 20)
    plt.vlines(radius,-0.2, binned_counts[0], color = 'red')
    plt.annotate(f"Radius {radius}", (radius - 1 ,-0.25), fontsize = 15, color = 'red')
    plt.hlines(binned_counts[0], binned_counts[0], radius, color = 'red')
    plt.annotate('SOURCE', (radius - 2, binned_counts[0] + 0.02), fontsize = 15, color = 'red')
    plt.vlines(radius + 6,-0.2, binned_counts[0], color = 'red')
    plt.vlines(radius + 13,-0.2, binned_counts[0], color = 'red')
    plt.hlines(binned_counts[0], radius + 6, radius + 13, color = 'red')
    plt.annotate('BACKGROUND', (radius +6, binned_counts[0] + 0.02), fontsize = 15, color = 'red')
    plt.annotate(f"Back> {radius + 6}", (radius + 5 ,-0.25), fontsize = 15, color = 'red')
    plt.annotate(f"<Back {radius + 13}", (radius + 12 ,-0.25), fontsize = 15, color = 'red')
    title_string = [f"{plot_title}", f"FWHM:{HWHM*2} pixels"]
    plt.title('\n'.join(title_string))