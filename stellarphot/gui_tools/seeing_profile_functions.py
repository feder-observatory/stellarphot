from pathlib import Path
import warnings

import numpy as np
from photutils.centroids import centroid_com, centroid_quadratic, centroid_2dg
from photutils.profiles import RadialProfile, CurveOfGrowth
import ipywidgets as ipw

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.utils import lazyproperty

try:
    from astrowidgets import ImageWidget
except ImportError:
    from astrowidgets.ginga import ImageWidget

import matplotlib.pyplot as plt

from stellarphot.io import TessSubmission
from stellarphot.gui_tools.fits_opener import FitsOpener
from stellarphot.photometry import calculate_noise
from stellarphot.plotting import seeing_plot
from stellarphot.settings import ApertureSettings, ui_generator

__all__ = [
    "set_keybindings",
    "find_center",
    "CenterAndProfile",
    "box",
    "SeeingProfileWidget",
]

desc_style = {"description_width": "initial"}


# TODO: maybe move this into SeeingProfileWidget unless we anticipate
# other widgets using this.
def set_keybindings(image_widget, scroll_zoom=False):
    """
    Set image widget keyboard bindings. The bindings are:

    + Pan by click-and-drag or with arrow keys.
    + Zoom by scrolling or using the ``+``/``-`` keys.
    + Adjust contrast by Ctrl-right click and drag
    + Reset contrast with shift-right-click.

    Any existing key bindings are removed.

    Parameters
    ----------

    image_widget : `astrowidgets.ImageWidget`
        Image widget on which to set the key bindings.

    scroll_zoom : bool, optional
        If True, zooming can be done by scrolling the mouse wheel.
        Default is False.

    Returns
    -------

    None
        Adds key bindings to the image widget.
    """
    bind_map = image_widget._viewer.get_bindmap()
    # Displays the event map...
    # bind_map.eventmap
    bind_map.clear_event_map()
    bind_map.map_event(None, (), "ms_left", "pan")
    if scroll_zoom:
        bind_map.map_event(None, (), "pa_pan", "zoom")

    # bind_map.map_event(None, (), 'ms_left', 'cursor')
    # contrast with right mouse
    bind_map.map_event(None, (), "ms_right", "contrast")

    # shift-right mouse to reset contrast
    bind_map.map_event(None, ("shift",), "ms_right", "contrast_restore")
    bind_map.map_event(None, ("ctrl",), "ms_left", "cursor")

    # Bind +/- to zoom in/out
    bind_map.map_event(None, (), "kp_+", "zoom_in")
    bind_map.map_event(None, (), "kp_=", "zoom_in")
    bind_map.map_event(None, (), "kp_-", "zoom_out")
    bind_map.map_event(None, (), "kp__", "zoom_out")

    # Bind arrow keys to panning
    # There is NOT a typo below. I want the keys to move the image in the
    # direction of the arrow
    bind_map.map_event(None, (), "kp_left", "pan_right")
    bind_map.map_event(None, (), "kp_right", "pan_left")
    bind_map.map_event(None, (), "kp_up", "pan_down")
    bind_map.map_event(None, (), "kp_down", "pan_up")


# TODO: Can this be replaced by a properly masked call to centroid_com?
def find_center(image, center_guess, cutout_size=30, max_iters=10, match_limit=3):
    """
    Find the centroid of a star from an initial guess of its position. Originally
    written to find star from a mouse click.

    Parameters
    ----------

    image : `astropy.nddata.CCDData` or numpy array
        Image containing the star.

    center_guess : array or tuple
        The position, in pixels, of the initial guess for the position of
        the star. The coordinates should be horizontal first, then vertical,
        i.e. opposite the usual Python convention for a numpy array.

    cutout_size : int, optional
        The default width of the cutout to use for finding the star.

    max_iters : int, optional
        Maximum number of iterations to go through in finding the center.

    match_limit : int, optional
        Maximum number of pixels to allow the COM centroid and Gaussian center to
        differ.

    Returns
    -------

    cen : array
        The position of the star, in pixels, as found by the centroiding
        algorithm.
    """
    pad = cutout_size // 2
    x, y = center_guess

    # Keep track of iterations
    cnt = 0

    # Grab the cutout...
    sub_data = Cutout2D(image, center_guess, (cutout_size, cutout_size), mode="trim")
    # ...do stats on it...
    _, sub_med, _ = sigma_clipped_stats(sub_data.data)
    # ...and centroid.
    mask = (sub_data.data - sub_med) < 0
    x_cm, y_cm = centroid_com(sub_data.data - sub_med, mask=mask)

    # Translate centroid back to original image (maybe use Cutout2D instead)
    cen = np.array(sub_data.to_original_position((x_cm, y_cm)))
    print(f"{cen=} {x_cm=} {y_cm=}")
    # ceno is the "original" center guess, set it to something nonsensical here
    ceno = np.array([-100, -100])

    while cnt <= max_iters and (
        np.abs(np.array([x_cm, y_cm]) - pad).max() > 3 or np.abs(cen - ceno).max() > 0.1
    ):
        try:
            sub_data = Cutout2D(image, cen, (cutout_size, cutout_size), mode="trim")
        except NoOverlapError:
            raise RuntimeError(
                f"Centroid finding failed, previous was {ceno}, current is {cen}"
            )
        _, sub_med, _ = sigma_clipped_stats(sub_data.data)

        mask = (sub_data.data - sub_med) < 0
        x_cm, y_cm = centroid_com(sub_data.data - sub_med, mask=mask)
        ceno = cen
        cen = np.array(sub_data.to_original_position((x_cm, y_cm)))
        if not np.all(~np.isnan(cen)):
            raise RuntimeError(
                f"Centroid finding failed, previous was {ceno}, current is {cen}"
            )
        cnt += 1

    # Is we hit the max number of iterations, raise an error
    if max_iters > 1 and cnt > max_iters:
        raise RuntimeError(f"Centroid finding did not converge")

    # Get the final centroid position by fitting a gaussian
    # We may not have converged on a star, so capture any warning about a
    # bad fit.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ceng = centroid_2dg(sub_data.data - sub_med)
    ceng = np.array(sub_data.to_original_position(ceng))

    # Confirm that we actually found a star by comparing the two centroiding
    # methods
    if np.linalg.norm(cen - ceng) > match_limit:
        raise RuntimeError(
            "Centroid did not converge on a star. "
            f"Got {cen} from centroid_com and {ceng} from centroid_2dg."
        )

    return ceng


class CenterAndProfile:
    """
    Class to dentermine center of and hold radial profile information for a star.

    Parameters
    ----------
    data : `astropy.nddata.CCDData` or numpy array
        Image data

    center_approx : list-like
        x, y position of the center in pixel coordinates, i.e. horizontal
        coordinate then vertical.

    cutout_size : int, optional
        Width of the rectangular image cutout to use in looking for a star.

    profile_radius : int, optional
        Maximum radius to use in constructing the profile.
    """

    def __init__(self, data, center_approx, cutout_size=30, profile_radius=None):
        self._cen = find_center(data, center_approx, cutout_size=cutout_size)
        self._data = data
        self._cutout = Cutout2D(data, self._cen, (cutout_size, cutout_size))
        if profile_radius is None:
            profile_radius = cutout_size // 2

        radii = np.linspace(0, profile_radius, profile_radius + 1)
        # Get a rough profile without background subtraction
        self._radial_profile = RadialProfile(self._data, self._cen, radii)

        self._sky_area = None

        # Do proper background subtraction for the final radial profile
        self._radial_profile = RadialProfile(
            data - self.sky_pixel_value, self._cen, radii
        )

    @property
    def center(self):
        """
        x, y position of the center of the star.
        """
        return self._cen

    @property
    def FWHM(self):
        """
        Full-width half-max of the radial profile.
        """
        return self.radial_profile.gaussian_fwhm

    @property
    def HWHM(self):
        """
        Half-width half-max of the radial profile.
        """
        return self.FWHM / 2

    @property
    def cutout(self):
        """
        Cutout image around the star.
        """
        return self._cutout

    @property
    def radial_profile(self):
        """
        Radial profile of the star.
        """
        return self._radial_profile

    @lazyproperty
    def normalized_profile(self):
        """
        Radial profile scaled to have a maximum of 1.
        """
        # This seems to be what photutils does under the hood
        return self.radial_profile.profile / self.radial_profile.profile.max()

    @lazyproperty
    def pixel_values_in_profile(self):
        """
        Pixel values in the radial profile.
        """
        radii = []
        pixel_values = []
        for rad, ap in zip(self.radial_profile.radius, self.radial_profile.apertures):
            ap_mask = ap.to_mask(method="center")
            ap_data = ap_mask.multiply(self._data)
            good_data = ap_data != 0
            pixel_values.extend(ap_data[good_data].flatten())
            radii.extend([rad] * good_data.sum())
        radii = np.array(radii)
        pixel_values = np.array(pixel_values)
        return radii, pixel_values

    @lazyproperty
    def curve_of_growth(self):
        """
        Curve of growth for the star.
        """
        self._cog = CurveOfGrowth(
            self._data - self.sky_pixel_value,
            self.center,
            self.radial_profile.radii + 1,
        )
        return self._cog

    @lazyproperty
    def sky_pixel_value(self):
        """
        Pixel values for the sky, i.e. outside the star.
        """
        grid_x, grid_y = np.mgrid[: self.cutout.shape[0], : self.cutout.shape[1]]
        x_s, y_s = self.cutout.to_cutout_position(self.center)
        dist_from_star = np.sqrt((grid_x - x_s) ** 2 + (grid_y - y_s) ** 2)
        mask = dist_from_star > self.FWHM * 3
        self._sky_area = mask.sum()
        _, median, _ = sigma_clipped_stats(self.cutout.data[mask])
        return median

    @lazyproperty
    def sky_area(self):
        """
        Area of the sky annulus.
        """
        if self._sky_area is None:
            # sky area is set as a side effect of this....
            _ = self.sky_pixel_value

        return self._sky_area

    def noise(self, camera, exposure):
        """
        Noise in the star.
        """
        return calculate_noise(
            camera,
            counts=self.curve_of_growth.profile,
            sky_per_pix=self.sky_pixel_value,
            aperture_area=self.curve_of_growth.area,
            annulus_area=0,
            exposure=exposure,
        )

    def snr(self, camera, exposure):
        """
        Signal to noise ratio of the star.
        """
        return camera.gain * self.curve_of_growth.profile / self.noise(camera, exposure)


def box(imagewidget):
    """
    Compatibility layer for older versions of the photometry notebooks.

    Parameters
    ----------

    imagewidget : `astrowidgets.ImageWidget`
        ImageWidget instance to use for the seeing profile.

    Returns
    -------

    box : `ipywidgets.VBox`
        Box containing the seeing profile widget.
    """
    return SeeingProfileWidget(imagewidget=imagewidget).box


class SeeingProfileWidget:
    """
    A class for storing an instance of a widget displaying the seeing profile
    of stars in an image.

    Parameters
    ----------
    imagewidget : `astrowidgets.ImageWidget`, optional
        ImageWidget instance to use for the seeing profile.

    width : int, optional
        Width of the seeing profile widget. Default is 500 pixels.

    Attributes
    ----------

    ap_t : `ipywidgets.IntText`
        Text box for the aperture radius.

    box : `ipywidgets.VBox`
        Box containing the seeing profile widget.

    container : `ipywidgets.VBox`
        Container for the seeing profile widget.

    object_name : str
        Name of the object in the FITS file.

    in_t : `ipywidgets.IntText`
        Text box for the inner annulus.

    iw : `astrowidgets.ImageWidget`
        ImageWidget instance used for the seeing profile.

    object_name : str
        Name of the object in the FITS file.

    out : `ipywidgets.Output`
        Output widget for the seeing profile.

    out2 : `ipywidgets.Output`
        Output widget for the integrated counts.

    out3 : `ipywidgets.Output`
        Output widget for the SNR.

    out_t : `ipywidgets.IntText`
        Text box for the outer annulus.

    rad_prof : `RadialProfile`
        Radial profile of the star.

    save_aps : `ipywidgets.Button`
        Button to save the aperture settings.

    tess_box : `ipywidgets.VBox`
        Box containing the TESS settings.

    """

    def __init__(self, imagewidget=None, width=500, camera=None):
        if not imagewidget:
            imagewidget = ImageWidget(
                image_width=width, image_height=width, use_opencv=True
            )

        self.iw = imagewidget

        self.camera = camera
        # Do some set up of the ImageWidget
        set_keybindings(self.iw, scroll_zoom=False)
        bind_map = self.iw._viewer.get_bindmap()
        bind_map.map_event(None, ("shift",), "ms_left", "cursor")
        gvc = self.iw._viewer.get_canvas()
        self._mse = self._make_show_event()
        gvc.add_callback("cursor-down", self._mse)

        # Outputs to hold the graphs
        self.seeing_profile_plot = ipw.Output()
        self.curve_growth_plot = ipw.Output()
        self.snr_plot = ipw.Output()
        # Build the larger widget
        self.container = ipw.VBox()
        self.fits_file = FitsOpener(title="Choose an image")
        big_box = ipw.HBox()
        big_box = ipw.GridspecLayout(1, 2)
        layout = ipw.Layout(width="60ch")
        vb = ipw.VBox()
        self.aperture_settings_file_name = ipw.Text(
            description="Aperture settings file name",
            style={"description_width": "initial"},
            value="aperture_settings.json",
            layout=layout,
        )
        self.aperture_settings = ui_generator(ApertureSettings)
        self.aperture_settings.show_savebuttonbar = True
        self.aperture_settings.path = Path(self.aperture_settings_file_name.value)

        vb.children = [
            self.aperture_settings_file_name,
            self.aperture_settings,
        ]

        lil_box = ipw.VBox()
        lil_tabs = ipw.Tab()
        lil_tabs.children = [
            self.snr_plot,
            self.seeing_profile_plot,
            self.curve_growth_plot,
        ]
        lil_tabs.set_title(0, "SNR")
        lil_tabs.set_title(1, "Seeing profile")
        lil_tabs.set_title(2, "Integrated counts")
        self.tess_box = self._make_tess_box()
        lil_box.children = [lil_tabs, self.tess_box]

        imbox = ipw.VBox()
        imbox.children = [imagewidget, vb]
        big_box[0, 0] = imbox
        big_box[0, 1] = lil_box
        big_box.layout.width = "100%"

        # Line below puts space between the image and the plots so the plots
        # don't jump around as the image value changes.
        big_box.layout.justify_content = "space-between"
        self.big_box = big_box
        self.container.children = [self.fits_file.file_chooser, self.big_box]
        self.box = self.container
        self._aperture_name = "aperture"

        self._tess_sub = None

        # Fill these in later with name of object from FITS file
        self.object_name = ""
        self.exposure = 0
        self._set_observers()
        self.aperture_settings.description = ""

    def load_fits(self, file):
        """
        Load a FITS file into the image widget.

        Parameters
        ----------

        file : str
            Filename to open.
        """
        self.fits_file.load_in_image_widget(self.iw)
        self.object_name = self.fits_file.object
        self.exposure = self.fits_file.header["EXPOSURE"]

    def _update_file(self, change):
        self.load_fits(change.selected)

    def _construct_tess_sub(self):
        file = self.fits_file.path
        self._tess_sub = TessSubmission.from_header(
            fits.getheader(file),
            telescope_code=self.setting_box.telescope_code.value,
            planet=self.setting_box.planet_num.value,
        )

    def _set_seeing_profile_name(self, change):  # noqa: ARG002
        """
        Widget callbacks need to accept a single argument, even if it is not used.
        """
        self._construct_tess_sub()
        self.seeing_file_name.value = self._tess_sub.seeing_profile

    def _save_toggle_action(self, change):
        activated = change["new"]

        if activated:
            self.setting_box.layout.visibility = "visible"
            self._set_seeing_profile_name("")
        else:
            self.setting_box.layout.visibility = "hidden"

    def _save_seeing_plot(self, button):  # noqa: ARG002
        """
        Widget button callbacks need to accept a single argument.
        """
        self._seeing_plot_fig.savefig(self.seeing_file_name.value)

    def _change_aperture_save_location(self, change):
        new_name = change["new"]
        new_path = Path(new_name)
        self.aperture_settings.path = new_path
        self.aperture_settings.savebuttonbar.unsaved_changes = True

    def _set_observers(self):
        def aperture_obs(change):
            self._update_plots()
            ape = ApertureSettings(**change["new"])
            self.aperture_settings.description = (
                f"Inner annulus: {ape.inner_annulus}, "
                f"outer annulus: {ape.outer_annulus}"
            )

        self.aperture_settings.observe(aperture_obs, names="_value")
        self.aperture_settings_file_name.observe(
            self._change_aperture_save_location, names="value"
        )
        self.fits_file.register_callback(self._update_file)
        self.save_toggle.observe(self._save_toggle_action, names="value")
        self.save_seeing.on_click(self._save_seeing_plot)
        self.setting_box.planet_num.observe(self._set_seeing_profile_name)
        self.setting_box.telescope_code.observe(self._set_seeing_profile_name)

    def _make_tess_box(self):
        box = ipw.VBox()
        setting_box = ipw.HBox()
        self.save_toggle = ipw.ToggleButton(
            description="TESS seeing profile...", disabled=True
        )
        scope_name = ipw.Text(
            description="Telescope code", value="Paul-P-Feder-0.4m", style=desc_style
        )
        planet_num = ipw.IntText(description="Planet", value=1)
        self.save_seeing = ipw.Button(description="Save")
        self.seeing_file_name = ipw.Label(value="Moo")
        setting_box.children = (
            scope_name,
            planet_num,
            self.seeing_file_name,
            self.save_seeing,
        )
        # for kid in setting_box.children:
        #     kid.disabled = True
        box.children = (self.save_toggle, setting_box)
        setting_box.telescope_code = scope_name
        setting_box.planet_num = planet_num
        setting_box.layout.flex_flow = "row wrap"
        setting_box.layout.visibility = "hidden"
        self.setting_box = setting_box
        return box

    def _update_ap_settings(self, value):
        self.aperture_settings.value = value

    def _make_show_event(self):
        def show_event(
            viewer, event=None, datax=None, datay=None, aperture=None
        ):  # noqa: ARG001
            """
            ginga callbacks require the function signature above.
            """
            profile_size = 60
            default_gap = 5  # pixels
            default_annulus_width = 15  # pixels
            self.save_toggle.disabled = False

            update_aperture_settings = False
            if event is not None:
                # User clicked on a star, so generate profile
                i = self.iw._viewer.get_image()
                data = i.get_data()

                # Rough location of click in original image
                x = int(np.floor(event.data_x))
                y = int(np.floor(event.data_y))

                rad_prof = CenterAndProfile(data, (x, y), profile_radius=profile_size)

                try:
                    try:  # Remove previous marker
                        self.iw.remove_markers(marker_name=self._aperture_name)
                    except AttributeError:
                        self.iw.remove_markers_by_name(marker_name=self._aperture_name)
                except ValueError:
                    # No markers yet, keep going
                    pass

                # ADD MARKER WHERE CLICKED
                self.iw.add_markers(
                    Table(
                        data=[[rad_prof.center[0]], [rad_prof.center[1]]],
                        names=["x", "y"],
                    ),
                    marker_name=self._aperture_name,
                )

                # Default is 1.5 times FWHM
                aperture_radius = np.round(1.5 * rad_prof.FWHM, 0)
                self.rad_prof = rad_prof

                # Make an aperture settings object, but don't update it's widget yet.
                ap_settings = ApertureSettings(
                    radius=aperture_radius,
                    gap=default_gap,
                    annulus_width=default_annulus_width,
                )
                update_aperture_settings = True
            else:
                # User changed aperture
                aperture_radius = aperture["radius"]
                ap_settings = ApertureSettings(
                    **aperture
                )  # Make an ApertureSettings object

            if update_aperture_settings:
                self._update_ap_settings(ap_settings.dict())

            self._update_plots()

        return show_event

    def _update_plots(self):
        # DISPLAY THE SCALED PROFILE
        fig_size = (10, 5)
        profile_size = 60

        rad_prof = self.rad_prof
        self.seeing_profile_plot.clear_output(wait=True)
        ap_settings = ApertureSettings(**self.aperture_settings.value)
        with self.seeing_profile_plot:
            # sub_med += med
            r_exact, individual_counts = rad_prof.pixel_values_in_profile
            scaled_exact_counts = (
                individual_counts / rad_prof.radial_profile.profile.max()
            )
            self._seeing_plot_fig = seeing_plot(
                r_exact,
                scaled_exact_counts,
                rad_prof.radial_profile.radius,
                rad_prof.normalized_profile,
                rad_prof.HWHM,
                self.object_name,
                aperture_settings=ap_settings,
                figsize=fig_size,
            )
            plt.show()

        # CALCULATE AND DISPLAY NET COUNTS INSIDE RADIUS
        self.curve_growth_plot.clear_output(wait=True)
        with self.curve_growth_plot:
            cog = rad_prof.curve_of_growth

            plt.figure(figsize=fig_size)
            plt.plot(cog.radius, cog.profile)
            plt.xlim(0, 40)
            ylim = plt.ylim()
            plt.vlines(ap_settings.radius, *plt.ylim(), colors=["red"])
            plt.ylim(*ylim)
            plt.grid()

            plt.title("Net counts in aperture")

            plt.xlabel("Aperture radius (pixels)")
            plt.ylabel("Net counts")
            plt.show()

        # CALCULATE And DISPLAY SNR AS A FUNCTION OF RADIUS
        self.snr_plot.clear_output(wait=True)
        with self.snr_plot:
            plt.figure(figsize=fig_size)
            snr = rad_prof.snr(self.camera, self.exposure)
            plt.plot(rad_prof.curve_of_growth.radius, snr)

            plt.title(
                f"Signal to noise ratio max {snr.max():.1f} "
                f"at radius {snr.argmax() + 1}"
            )
            plt.xlim(0, 40)
            ylim = plt.ylim()
            plt.vlines(ap_settings.radius, *plt.ylim(), colors=["red"])
            plt.ylim(*ylim)
            plt.xlabel("Aperture radius (pixels)")
            plt.ylabel("SNR")
            plt.grid()
            plt.show()
