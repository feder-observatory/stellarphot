import numpy as np
from photutils.centroids import centroid_com
import ipywidgets as ipw

from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.nddata import Cutout2D

import matplotlib.pyplot as plt
from stellarphot.visualization import seeing_plot

__all__ = ['set_keybindings', 'box', 'make_show_event']

out = ipw.Output()
out2 = ipw.Output()
out3 = ipw.Output()


def set_keybindings(image_widget, scroll_zoom=False):
    """
    Set image widget keyboard bindings. The bindings are:

    + Pan by click-and-drag or with arrow keys.
    + Zoom by scrolling or using the ``+``/``-`` keys.
    + Adjust contrast by Ctrl-left click and drag; reset with
      shift-right-click.

    Any existing key bindings are removed.

    Parameters
    ----------

    image_widget : astrowidgets.ImageWidget
        Image widget on which to set the key bindings.
    """
    bind_map = image_widget._viewer.get_bindmap()
    # Displays the event map...
    # bind_map.eventmap
    bind_map.clear_event_map()
    bind_map.map_event(None, (), 'ms_left', 'pan')
    if scroll_zoom:
        bind_map.map_event(None, (), 'pa_pan', 'zoom')

    # bind_map.map_event(None, (), 'ms_left', 'cursor')
    # contrast with right mouse
    bind_map.map_event(None, (), 'ms_right', 'contrast')

    # shift-right mouse to reset contrast
    bind_map.map_event(None, ('shift',), 'ms_right', 'contrast_restore')
    bind_map.map_event(None, ('ctrl',), 'ms_left', 'cursor')

    # Bind +/- to zoom in/out
    bind_map.map_event(None, (), 'kp_+', 'zoom_in')
    bind_map.map_event(None, (), 'kp_=', 'zoom_in')
    bind_map.map_event(None, (), 'kp_-', 'zoom_out')
    bind_map.map_event(None, (), 'kp__', 'zoom_out')

    # Bind arrow keys to panning
    # There is NOT a typo below. I want the keys to move the image in the
    # direction of the arrow
    bind_map.map_event(None, (), 'kp_left', 'pan_right')
    bind_map.map_event(None, (), 'kp_right', 'pan_left')
    bind_map.map_event(None, (), 'kp_up', 'pan_down')
    bind_map.map_event(None, (), 'kp_down', 'pan_up')


def find_center(image, center_guess, cutout_size=30, max_iters=10):
    """
    Find the centroid of a star from an initial guess of its position. Originally
    written to find star from a mouse click.

    Parameters
    ----------

    image : numpy array or CCDData
        Image containing the star.

    center_guess : array or tuple
        The position, in pixels, of the initial guess for the position of
        the star. The coordinates should be horizontal first, then vertical,
        i.e. opposite the usual Python convention for a numpy array.

    cutout_size : int, optional
        The default width of the cutout to use for finding the star.

    max_iters : int, optional
        Maximum number of iterations to go through in finding the center.
    """
    pad = cutout_size // 2
    x, y = center_guess

    # Keep track of iterations
    cnt = 0

    # Grab the cutout...
    sub_data = image[y - pad:y + pad, x - pad:x + pad]  # - med

    # ...do stats on it...
    _, sub_med, _ = sigma_clipped_stats(sub_data)
    # sub_med = 0

    # ...and centroid.
    x_cm, y_cm = centroid_com(sub_data - sub_med)

    # Translate centroid back to original image (maybe use Cutout2D instead)
    cen = np.array([x_cm + x - pad, y_cm + y - pad])

    # ceno is the "original" center guess, set it to something nonsensical here
    ceno = np.array([-100, -100])

    while (cnt <= max_iters and
           (np.abs(np.array([x_cm, y_cm]) - pad).max() > 3
            or np.abs(cen - ceno).max() > 0.1)):

        # Update x, y positions for subsetting
        x = int(np.floor(x_cm)) + x - pad
        y = int(np.floor(y_cm)) + y - pad
        sub_data = image[y - pad:y + pad, x - pad:x + pad]  # - med
        _, sub_med, _ = sigma_clipped_stats(sub_data)
        # sub_med = 0
        mask = (sub_data - sub_med) < 0
        x_cm, y_cm = centroid_com(sub_data - sub_med, mask=mask)
        ceno = cen
        cen = np.array([x_cm + x - pad, y_cm + y - pad])
        if not np.all(~np.isnan(cen)):
            raise RuntimeError('Centroid finding failed, '
                               'previous was {}, current is {}'.format(ceno, cen))
        cnt += 1

    return cen


def radial_profile(data, center, size=30, return_scaled=True):
    """
    Construct a radial profile of a chunk of width ``size`` centered
    at ``center`` from image ``data`.

    Parameters
    ----------

    data : numpy array or CCDData
        Image data
    center : list-like
        x, y position of the center in pixel coordinates, i.e. horizontal
        coordinate then vertical.
    size : int, optional
        Width of the rectangular cutout to use in constructing the profile.
    return_scaled : bool, optional
        If ``True``, return an average radius and profile, otherwise
        it is cumulative. Not at all clear what a "cumulative" radius
        means, tbh.

    Returns
    -------

    r_exact : numpy array
        Exact radius of center of each pixels from profile center.
    ravg : numpy array
        Average radius of pixels in each bin.
    radialprofile : numpy array
        Radial profile.
    """
    yd, xd = np.indices((size, size))

    sub_image = Cutout2D(data, center, size, mode='strict')
    sub_center = sub_image.center_cutout

    r = np.sqrt((xd - sub_center[0])**2 + (yd - sub_center[1])**2)
    r_exact = r.copy()
    r = r.astype(np.int)

    sub_data = sub_image.data

    tbin = np.bincount(r.ravel(), sub_data.ravel())
    rbin = np.bincount(r.ravel(), r_exact.ravel())
    nr = np.bincount(r.ravel())
    if return_scaled:
        radialprofile = tbin / nr
        ravg = rbin / nr
    else:
        radialprofile = tbin
        ravg = rbin

    return r_exact, ravg, radialprofile


def find_hwhm(r, intensity):
    """
    Estimate HWHM from normalized, angle-averaged intensity profile.
    """
    # Make the bold assumption that intensity decreases monotonically
    # so that we just need to find the first place where intensity is
    # less than 0.5 to estimate the HWHM.
    less_than_half = intensity < 0.5
    half_index = np.arange(len(less_than_half))[less_than_half][0]
    before_half = half_index - 1

    # Do linear interpolation to find the radius at which the intensity
    # is 0.5.
    r_more = r[before_half]
    r_less = r[half_index]
    I_more = intensity[before_half]
    I_less = intensity[half_index]

    I_half = 0.5

    r_half = r_less - (I_less - I_half) / (I_less - I_more) * (r_less - r_more)

    return r_half


def make_show_event(iw):
    def show_event(viewer, event, datax, datay):

        i = iw._viewer.get_image()
        data = i.get_data()

        fig_size = (10, 5)
        # Rough location of click in original image
        x = int(np.floor(event.data_x))
        y = int(np.floor(event.data_y))

        cen = find_center(data, (x, y), cutout_size=30)

        # ADD MARKER WHERE CLICKED
        iw.add_markers(Table(data=[[cen[0]], [cen[1]]], names=['x', 'y']))

        # ----> MOVE PROFILE CONSTRUCTION INTO FUNCTION <----

        # CONSTRUCT RADIAL PROFILE OF PATCH AROUND STAR
        profile_size = 60
        # NOTE: THIS IS NOT BACKGROUND SUBTRACTED
        r_exact, ravg, radialprofile = radial_profile(data, cen, size=profile_size)
        sub_data = Cutout2D(data, cen, size=profile_size).data
        sub_med = np.median(sub_data)
        adjust_max = radialprofile.max() - sub_med
        scaled_profile = (radialprofile - sub_med) / adjust_max
        scaled_exact_counts = (sub_data - sub_med) / adjust_max

        # DISPLAY THE SCALED PROFILE
        out.clear_output(wait=True)
        with out:
            plt.clf()
            # sub_med += med
            HWHM = find_hwhm(ravg, scaled_profile)
            seeing_plot(r_exact, scaled_exact_counts, ravg, scaled_profile, HWHM,
                        'Some Image Name', file_name='some_name', gap=6, annulus_width=13)
            plt.show();

        # CALCULATE AND DISPLAY NET COUNTS INSIDE RADIUS
        out2.clear_output(wait=True)
        with out2:
            sub_blot = sub_data.copy()
            FWHM = int(np.round(2 * HWHM))
            min_idx = profile_size // 2 - 2 * FWHM
            max_idx = profile_size // 2 + 2 * FWHM
            sub_blot[min_idx:max_idx, min_idx:max_idx] = np.nan
            sub_std = np.nanstd(sub_blot)
            new_sub_med = np.nanmedian(sub_blot)
            r_exact, ravg, tbin2 = radial_profile(data - new_sub_med, cen,
                                                  size=profile_size,
                                                  return_scaled=False)
            r_exact_s, ravg_s, tbin2_s = radial_profile(data - new_sub_med, cen,
                                                  size=profile_size,
                                                  return_scaled=True)
            #tbin2 = np.bincount(r.ravel(), (sub_data - sub_med).ravel())
            counts = np.cumsum(tbin2)
            plt.figure(figsize=fig_size)
            plt.plot(range(len(radialprofile)), counts)
            plt.xlim(0, 40)
            plt.grid()

            plt.title('Net counts in aperture std {:.2f} med {:.2f} new {:.2f}'.format(
                sub_std, sub_med, new_sub_med))
            e_sky = np.sqrt(new_sub_med)
            plt.xlabel('Aperture radius')
            plt.show()

        # CALCULATE And DISPLAY SNR AS A FUNCTION OF RADIUS
        out3.clear_output(wait=True)
        with out3:
            read_noise = 10  # electrons
            gain = 1.5  # electrons/count
            # Poisson error is square root of the net number of counts enclosed
            poisson = np.sqrt(np.cumsum(tbin2))

            # This is obscure, but correctly calculated the number of pixels at
            # each radius
            nr = tbin2 / tbin2_s

            # This ignores dark current
            error = np.sqrt(poisson ** 2 + np.cumsum(nr)
                            * (e_sky ** 2 + (read_noise / gain)** 2))

            snr = np.cumsum(tbin2) / error
            plt.figure(figsize=fig_size)
            plt.plot(np.arange(len(radialprofile)) + 1, snr)

            plt.title(f'Signal to noise ratio max {snr.max():.1f} '
                      f'at radius {snr.argmax() + 1}')
            plt.xlim(0, 40)

            plt.xlabel('Aperture radius')
            plt.grid()
            plt.show()
    return show_event


def box(imagewidget):
    big_box = ipw.HBox()
    big_box = ipw.GridspecLayout(1, 2)
    layout = ipw.Layout(width='20ch')
    hb = ipw.HBox()
    ap_t = ipw.IntText(description='Aperture', value=5, layout=layout)
    in_t = ipw.IntText(description='Inner annulus', value=10, layout=layout)
    out_t = ipw.IntText(description='Outer annulus', value=20, layout=layout)
    hb.children = [ap_t, in_t, out_t]

    lil_box = ipw.VBox()
    lil_box.children = [out, out2, out3]
    # big_box.children = [imagewidget, lil_box]
    big_box[0, 0] = imagewidget
    big_box[0, 1] = lil_box
    big_box.layout.width = '100%'
    # Line below puts space between the image and the plots so the plots
    # don't jump around as the image value changes.
    big_box.layout.justify_content = 'space-between'

    return big_box
