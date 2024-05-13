import warnings
from pathlib import Path

import ipywidgets as ipw
import numpy as np
from astropy.io import fits
from astropy.table import Table

try:
    from astrowidgets import ImageWidget
except ImportError:
    from astrowidgets.ginga import ImageWidget

import matplotlib.pyplot as plt

from stellarphot.gui_tools.fits_opener import FitsOpener
from stellarphot.io import TessSubmission
from stellarphot.photometry import CenterAndProfile
from stellarphot.photometry.photometry import EXPOSURE_KEYWORDS
from stellarphot.plotting import seeing_plot
from stellarphot.settings import PhotometryApertures, ui_generator
from stellarphot.settings.custom_widgets import ChooseOrMakeNew

__all__ = [
    "set_keybindings",
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

    camera : `stellarphot.settings.Camera`, optional
        Camera instance to use for calculating the signal to noise ratio. If
        ``None``, the signal to noise ratio will not be calculated.

    observatory : `stellarphot.settings.Observatory`, optional
        Observatory instance to use for setting the TESS telescope information.
        If `None`, or if the `~stellarphot.settings.Observatory.TESS_telescope_code`
        is `None`, the TESS settings will not be displayed.

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

    def __init__(self, imagewidget=None, width=500, camera=None, observatory=None):
        if not imagewidget:
            imagewidget = ImageWidget(
                image_width=width, image_height=width, use_opencv=True
            )

        self.iw = imagewidget

        self.observatory = observatory
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
        self.error_console = ipw.Output()

        # Build the larger widget
        self.container = ipw.VBox()
        self.fits_file = FitsOpener(title="Choose an image")
        self.camera_chooser = ChooseOrMakeNew("camera", details_hideable=True)
        if camera is not None:
            self.camera_chooser.value = camera

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
        self.aperture_settings = ui_generator(PhotometryApertures)
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
        self.container.children = [
            self.fits_file.file_chooser,
            self.camera_chooser,
            self.error_console,
            self.big_box,
        ]
        self.box = self.container
        self._aperture_name = "aperture"

        self._tess_sub = None

        # Fill these in later with name of object from FITS file
        self.object_name = ""
        self.exposure = 0
        self._set_observers()
        self.aperture_settings.description = ""

    @property
    def camera(self):
        return self.camera_chooser.value

    def load_fits(self):
        """
        Load a FITS file into the image widget.
        """
        self.fits_file.load_in_image_widget(self.iw)
        self.object_name = self.fits_file.object
        for key in EXPOSURE_KEYWORDS:
            if key in self.fits_file.header:
                self.exposure = self.fits_file.header[key]
                break
        else:
            # apparently setting a higher stacklevel is better, see
            # https://docs.astral.sh/ruff/rules/no-explicit-stacklevel/
            warnings.warn(
                "No exposure time keyword found in FITS header. "
                "Setting exposure to NaN",
                stacklevel=2,
            )
            self.exposure = np.nan

    def _update_file(self, change):  # noqa: ARG002
        # Widget callbacks need to accept a single argument, even if it is not used.
        self.load_fits()

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
            ape = PhotometryApertures(**change["new"])
            self.aperture_settings.description = (
                f"Inner annulus: {ape.inner_annulus}, "
                f"outer annulus: {ape.outer_annulus}"
            )

        self.aperture_settings.observe(aperture_obs, names="_value")
        self.aperture_settings_file_name.observe(
            self._change_aperture_save_location, names="value"
        )
        self.fits_file.file_chooser.observe(self._update_file, names="_value")
        if self.save_toggle:
            self.save_toggle.observe(self._save_toggle_action, names="value")
            self.save_seeing.on_click(self._save_seeing_plot)
            self.setting_box.planet_num.observe(self._set_seeing_profile_name)
            self.setting_box.telescope_code.observe(self._set_seeing_profile_name)

    def _make_tess_box(self):
        box = ipw.VBox()

        if self.observatory is None or self.observatory.TESS_telescope_code is None:
            """
            No TESS information, so definitely don't display this group of settings.
            """
            box.layout.flex_flow = "row wrap"
            box.layout.visibility = "hidden"

            self.save_toggle = None
            return box

        setting_box = ipw.HBox()
        self.save_toggle = ipw.ToggleButton(
            description="TESS seeing profile...", disabled=True
        )
        scope_name = ipw.Text(
            description="Telescope code",
            value=self.observatory.TESS_telescope_code,
            style=desc_style,
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
            viewer, event=None, datax=None, datay=None, aperture=None  # noqa: ARG001
        ):
            """
            ginga callbacks require the function signature above.
            """
            profile_size = 60
            centering_cutout_size = 20
            default_gap = 5  # pixels
            default_annulus_width = 15  # pixels
            if self.save_toggle:
                self.save_toggle.disabled = False

            update_aperture_settings = False
            if event is not None:
                # User clicked on a star, so generate profile
                i = self.iw._viewer.get_image()
                data = i.get_data()
                with self.error_console:
                    # Rough location of click in original image
                    x = int(np.floor(event.data_x))
                    y = int(np.floor(event.data_y))
                    print(f"Clicked at {x}, {y}")
                    rad_prof = CenterAndProfile(
                        data,
                        (x, y),
                        profile_radius=profile_size,
                        cutout_size=profile_size,
                        match_limit=100,
                    )
                    print(f"Center: {rad_prof.center} FWHM: {rad_prof.FWHM}")

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
                ap_settings = PhotometryApertures(
                    radius=aperture_radius,
                    gap=default_gap,
                    annulus_width=default_annulus_width,
                    fwhm=rad_prof.FWHM,
                )
                update_aperture_settings = True
            else:
                # User changed aperture
                aperture_radius = aperture["radius"]
                ap_settings = PhotometryApertures(
                    **aperture
                )  # Make an aperture settings object, but don't update it's widget yet.

            if update_aperture_settings:
                self._update_ap_settings(ap_settings.dict())

            self._update_plots()

        return show_event

    def _update_plots(self):
        # DISPLAY THE SCALED PROFILE
        fig_size = (10, 5)

        rad_prof = self.rad_prof
        self.seeing_profile_plot.clear_output(wait=True)
        ap_settings = PhotometryApertures(**self.aperture_settings.value)
        with self.seeing_profile_plot:
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
                photometry_settings=ap_settings,
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
            plt.ylabel(f"Net counts ({self.camera.data_unit})")
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
