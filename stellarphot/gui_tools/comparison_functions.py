import functools

import ipywidgets as ipw
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError
from astropy.table import Table

try:
    from astrowidgets import ImageWidget
except ImportError:
    from astrowidgets.ginga import ImageWidget

from stellarphot import SourceListData
from stellarphot.gui_tools.fits_opener import FitsOpener
from stellarphot.gui_tools.seeing_profile_functions import set_keybindings
from stellarphot.settings import (
    PartialPhotometrySettings,
    PhotometryWorkingDirSettings,
    SourceLocationSettings,
    ui_generator,
)
from stellarphot.settings.custom_widgets import SettingWithTitle
from stellarphot.utils.comparison_utils import (
    crossmatch_APASS2VSX,
    in_field,
    mag_scale,
    set_up,
)

__all__ = ["make_markers", "wrap", "ComparisonViewer"]

DESC_STYLE = {"description_width": "initial"}


def make_markers(iw, ccd, RD, vsx, ent, name_or_coord=None):
    """
    Add markers for APASS, TESS targets, VSX.  Also center on object/coordinate.

    Parameters
    ----------

    iw : `astrowidgets.ImageWidget`
        Ginga widget.

    ccd : `astropy.nddata.CCDData`
        Sample image.

    RD : `astropy.table.Table`
        Table with target information, including a
        `astropy.coordinates.SkyCoord` column.

    vsx : `astropy.table.Table`
        Table with known variables in the field of view.

    ent : `astropy.table.Table`
        Table with APASS stars in the field of view.

    name_or_coord : str or `astropy.coordinates.SkyCoord`, optional
        Name or coordinates of the target.

    Returns
    -------

    None
        Markers are added to the image in Ginga widget.
    """
    iw.load_nddata(ccd)
    iw.zoom_level = "fit"

    try:
        iw.reset_markers()
    except AttributeError:
        iw.remove_all_markers()

    if RD:
        iw.marker = {"type": "circle", "color": "green", "radius": 10}
        iw.add_markers(
            RD, skycoord_colname="coords", use_skycoord=True, marker_name="TESS Targets"
        )

    if name_or_coord is not None:
        if isinstance(name_or_coord, str):
            iw.center_on(SkyCoord.from_name(name_or_coord))
        else:
            iw.center_on(name_or_coord)

    if vsx:
        iw.marker = {"type": "circle", "color": "blue", "radius": 10}
        iw.add_markers(
            vsx, skycoord_colname="coords", use_skycoord=True, marker_name="VSX"
        )
    iw.marker = {"type": "circle", "color": "red", "radius": 10}
    iw.add_markers(
        ent,
        skycoord_colname="coords",
        use_skycoord=True,
        marker_name="APASS comparison",
    )
    iw.marker = {"type": "cross", "color": "red", "radius": 6}


def wrap(imagewidget, outputwidget):
    """
    Utility function to let you click to select/deselect comparisons.

    Parameters
    ----------

    imagewidget : `astrowidgets.ImageWidget`
        Ginga widget.

    outputwidget : `ipywidgets.Output`
        Output widget for printing information.

    """

    def cb(viewer, event, data_x, data_y):  # noqa: ARG001
        """
        The signature of this function must have the four arguments above.
        """
        i = imagewidget._viewer.get_image()

        try:
            imagewidget.next_elim += 1
        except AttributeError:
            imagewidget.next_elim = 1

        ra, dec = i.wcs.wcs.all_pix2world(event.data_x, event.data_y, 0)
        out_skycoord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree))

        try:
            all_table = imagewidget.get_markers(marker_name="all")
        except AttributeError:
            all_table = imagewidget.get_all_markers()

        with outputwidget:
            index, d2d, d3d = out_skycoord.match_to_catalog_sky(all_table["coord"])
            if d2d < 10 * u.arcsec:
                mouse = all_table["coord"][index].separation(all_table["coord"])
                rat = mouse < 1 * u.arcsec
                elims = [
                    name
                    for name in all_table["marker name"][rat]
                    if name.startswith("elim")
                ]
                if not elims:
                    imagewidget.add_markers(
                        all_table[rat],
                        skycoord_colname="coord",
                        use_skycoord=True,
                        marker_name=f"elim{imagewidget.next_elim}",
                    )
                else:
                    for elim in elims:
                        try:
                            imagewidget.remove_markers_by_name(marker_name=elim)
                        except AttributeError:
                            imagewidget.remove_markers(marker_name=elim)

            else:
                print("sorry try again")
                imagewidget._viewer.onscreen_message("Click closer to a star")

    return cb


class ComparisonViewer:
    """
    A class to store an instance of the comparison viewer.

    Parameters
    ----------

    file : str, optional
        File to open. Defaults to "".

    directory : str, optional
        Directory to open file from. Defaults to '.'.

    target_mag : float, optional
        Magnitude of the target. Defaults to 10.

    bright_mag_limit : float, optional
        Bright magnitude limit for APASS stars. Defaults to 8.

    dim_mag_limit : float, optional
        Dim magnitude limit for APASS stars.  Defaults to 17.

    targets_from_file : str, optional
        File with target information.  Defaults to None.

    object_coordinate : `astropy.coordinates.SkyCoord`, optional
        Coordinates of the target. Defaults to None.

    photom_apertures_file : str, optional
        File to save photometry aperture information to.  Defaults to None.

    overwrite_outputs: bool, optional
        Whether to overwrite existing output files. Defaults to True.

    observatory : `stellarphot.settings.Observatory`, optional
        Observatory information.

    Attributes
    ----------

    photom_apertures_file : str
        File to save photometry aperture information to.

    box : `ipywidgets.Box`
        Box containing the widgets.

    bright_mag_limit : float
        Bright magnitude limit for APASS stars.

    dim_mag_limit : float
        Dim magnitude limit for APASS stars.

    iw : `ginga.util.grc.RemoteClient`
        Ginga widget.

    overwrite_outputs: bool
        Whether to overwrite existing output files. Defaults to True.

    target_coord : `astropy.coordinates.SkyCoord`
        Coordinates of the target.

    targets_from_file : str
        File with target information.

    target_mag : float
        Magnitude of the target.

    variables : `astropy.table.Table`
    """

    def __init__(
        self,
        file="",
        directory=".",
        target_mag=10,
        bright_mag_limit=8,
        dim_mag_limit=17,
        targets_from_file=None,
        object_coordinate=None,
        photom_apertures_file=None,
        overwrite_outputs=True,
        observatory=None,
    ):
        self._label_name = "labels"
        self._circle_name = "target circle"
        self._file_chooser = FitsOpener()

        self._directory = directory
        self.target_mag = target_mag
        self.bright_mag_limit = bright_mag_limit
        self.dim_mag_limit = dim_mag_limit
        self.targets_from_file = targets_from_file
        self.target_coord = object_coordinate
        self.observatory = observatory

        self.box, self.iw = self._viewer()

        self.photometry_settings = PhotometryWorkingDirSettings()

        if photom_apertures_file is not None:
            # Set the source location file name to the name passed in
            self._set_source_location_file_to_value(photom_apertures_file)

        self.overwrite_outputs = overwrite_outputs

        if file:
            self._file_chooser.set_file(file, directory=directory)
            self._set_file(None)

        self._make_observers()

    def _init(self):
        """
        Handles aspects of initialization that need to be deferred until
        a file is chosen.
        """
        self.ccd, self.vsx = set_up(
            self._file_chooser.path.name,
            directory_with_images=self._file_chooser.path.parent,
        )

        apass, vsx_apass_angle, targets_apass_angle = crossmatch_APASS2VSX(
            self.ccd, self.targets_from_file, self.vsx
        )

        apass_good_coord, good_stars = mag_scale(
            self.target_mag,
            apass,
            vsx_apass_angle,
            targets_apass_angle,
            brighter_dmag=self.target_mag - self.bright_mag_limit,
            dimmer_dmag=self.dim_mag_limit - self.target_mag,
        )

        apass_comps = in_field(apass_good_coord, self.ccd, apass, good_stars)
        make_markers(
            self.iw,
            self.ccd,
            self.targets_from_file,
            self.vsx,
            apass_comps,
            name_or_coord=self.target_coord,
        )

        # Save the initial source list.
        if self.source_locations.value["source_list_file"] is not None:
            self._save_aperture_to_file(None)

    @property
    def photom_apertures_file(self):
        return self.source_locations.value["source_list_file"]

    @property
    def variables(self):
        """
        An `astropy.table.Table` of the variables in the class.
        """
        comp_table = self.generate_table()
        new_vsx_mark = comp_table["marker name"] == "VSX"
        idx, _, _ = comp_table["coord"][new_vsx_mark].match_to_catalog_sky(
            self.vsx["coords"]
        )
        our_vsx = self.vsx[idx]
        our_vsx["star_id"] = comp_table["star_id"][new_vsx_mark]
        return our_vsx

    def _set_object(self):
        """
        Try to automatically set object name immediately after file is chosen.
        """
        try:
            self.object_name.value = (
                f"<h2> Object: {self._file_chooser.header['object']}</h2>"
            )
            self._object = self._file_chooser.header["object"]
        except KeyError:
            # No object, will show placeholder
            self.object_name.value = "<h2>Object unknown</h2>"
            self.object_name.disabled = False
            self._object = ""

        # We have a name, try to get coordinates from it
        try:
            self.target_coord = SkyCoord.from_name(self._object)
        except NameResolveError:
            pass

    def _set_file(self, change):  # noqa: ARG002
        """
        Widget callbacks need to accept a change argument, even if not used.
        """
        self._set_object()
        self._init()

    def _make_observers(self):
        self._show_labels_button.observe(self._show_label_button_handler, names="value")
        self._save_var_info.on_click(self._save_variables_to_file)
        self._file_chooser.file_chooser.observe(self._set_file, names="_value")

    def _save_variables_to_file(self, button=None, filename=""):  # noqa: ARG002
        """
        Widget button callbacks need to be able to take an argument. It is called
        button above the the button will be passed as the first positional argument.
        """
        if not filename:
            filename = "variables.csv"
        # Export variables as CSV (overwrite existing file if it exists)
        try:
            self.variables.write(filename, overwrite=self.overwrite_outputs)
        except OSError as err:
            raise OSError(
                f"Existing file ({filename}) can not be overwritten. "
                "Set overwrite_outputs=True to address this."
            ) from err

    def _show_label_button_handler(self, change):
        value = change["new"]
        if value:
            # Showing labels can take a bit, disable while in progress
            self._show_labels_button.disabled = True
            self.show_labels()
            self._show_labels_button.disabled = False
        else:
            self.remove_labels()
        self._show_labels_button.description = self._show_labels_button.descriptions[
            value
        ]

    def _save_aperture_to_file(self, button=None, filename=""):  # noqa: ARG002
        """
        Widget button callbacks need to be able to take an argument. It is called
        button above the the button will be passed as the first positional argument.
        """
        if not filename:
            filename = self.photom_apertures_file

        # Convert aperture table into a SourceList objects and output
        targets_table = self.generate_table()
        # Assign units to columns
        targets_table["ra"] = targets_table["ra"] * u.deg
        targets_table["dec"] = targets_table["dec"] * u.deg
        targets_table["x"] = targets_table["x"] * u.pixel
        targets_table["y"] = targets_table["y"] * u.pixel
        # Drop redundant sky position column
        targets_table.remove_columns(["coord"])
        # Build sources list
        targets2sourcelist = {"x": "xcenter", "y": "ycenter"}
        sources = SourceListData(
            input_data=targets_table, colname_map=targets2sourcelist
        )

        # Export aperture file as CSV (overwrite existing file if it exists)
        try:
            sources.write(filename, overwrite=self.overwrite_outputs)
        except OSError as err:
            raise OSError(
                f"Existing file ({filename}) can not be overwritten."
                "Set overwrite_outputs=True to address this."
            ) from err

    def _make_control_bar(self):
        self._show_labels_button = ipw.ToggleButton(description="Click to show labels")
        self._show_labels_button.descriptions = {
            True: "Click to hide labels",
            False: "Click to show labels",
        }

        self._save_var_info = ipw.Button(description="Save variable info")

        controls = ipw.HBox(
            children=[
                self._show_labels_button,
                self._save_var_info,
            ]
        )

        return controls

    def save(self):
        """
        Save all of the settings we have to a partial settings file.
        """
        self.photometry_settings.save(
            PartialPhotometrySettings(
                source_locations=self.source_locations.value,
            ),
            update=True,
        )

        # For some reason the value of unsaved_changes is not updated until after this
        # function executes, so we force its value here.
        self.source_locations.savebuttonbar.unsaved_changes = False
        # Update the save box title to reflect the save
        self.source_and_title.decorate_title()

    def _set_source_location_file_to_value(self, name=None):
        # Right now the source location file name will not be set to the default name
        # properly because of a bug in ipyautoui, so we set it manually here.
        # Easier once/if https://github.com/maxfordham/ipyautoui/pull/323 is merged
        # and released. Then we can just set the value of the widget directly like this:
        #
        # self.source_locations.value = {'source_list_file': name, ...rest of settings}
        file_chooser = self.source_locations.di_widgets["source_list_file"]
        if name is None:
            # Use the default name
            name = self.source_locations.model.model_fields["source_list_file"].default
        # Make sure ipyautoui knows the value has changed
        file_chooser.value = name

        # Set the value to what we actually want
        file_chooser.reset(".", name)
        file_chooser._apply_selection()

        # Because we have updated the value outside of ipyautoui, also force an update
        # of the widget value.
        current_value = self.source_locations.value.copy()
        current_value["source_list_file"] = file_chooser.value
        self.source_locations.value = current_value

    def _viewer(self):
        header = ipw.HTML(
            value="""
        <h2>Click and drag or use arrow keys to pan, use +/- keys to zoom</h2>
        <h3>Shift-left click (or Crtl-left click)to exclude star as target
        or comp. Click again to include.</h3>
        """
        )

        legend = ipw.HTML(
            value="""
        <ul>
        <li>Green circles -- Gaia stars within 2.5 arcmin of target</li>
        <li>Red circles -- APASS stars within 1 mag of target</li>
        <li>Blue circles -- VSX variables</li>
        <li>Red × -- Exclude as target or comp</li>
        </ul>
        """
        )

        iw = ImageWidget()
        out = ipw.Output()
        set_keybindings(iw)
        bind_map = iw._viewer.get_bindmap()
        gvc = iw._viewer.get_canvas()
        bind_map.map_event(None, ("shift",), "ms_left", "cursor")
        gvc.add_callback("cursor-down", wrap(iw, out))

        self.object_name = ipw.HTML(value="<h2>Object: </h2>")
        self._object = None
        controls = self._make_control_bar()

        self.source_locations = ui_generator(
            SourceLocationSettings, max_field_width="75px"
        )

        self._set_source_location_file_to_value()
        self.source_and_title = SettingWithTitle(
            "Source location settings", self.source_locations
        )

        self.source_locations.savebuttonbar.fns_onsave_add_action(self.save)
        self.help_stuff = ipw.Accordion(children=[header, legend])
        self.help_stuff.titles = ["Help", "Legend"]
        box = ipw.VBox()
        inner_box = ipw.HBox()
        source_legend_box = ipw.VBox()
        source_legend_box.children = [
            self.object_name,
            self.source_and_title,
            self.help_stuff,
        ]
        inner_box.children = [iw, source_legend_box]  # legend]

        box.children = [
            self._file_chooser.file_chooser,
            inner_box,
            controls,
        ]

        return box, iw

    def generate_table(self):
        """
        Generate the table of stars to use for the aperture file.

        Returns
        -------
        comp_table : `astropy.table.Table`
            Table of stars to use for the aperture file.
        """
        try:
            all_table = self.iw.get_all_markers()
        except AttributeError:
            all_table = self.iw.get_markers(marker_name="all")

        elims = np.array([name.startswith("elim") for name in all_table["marker name"]])
        elim_table = all_table[elims]
        comp_table = all_table[~elims]

        index, d2d, d3d = elim_table["coord"].match_to_catalog_sky(comp_table["coord"])
        comp_table.remove_rows(index)

        # Add separate RA and Dec columns for ease in processing later
        comp_table["ra"] = comp_table["coord"].ra.degree
        comp_table["dec"] = comp_table["coord"].dec.degree

        # Calculate how far each is from target
        comp_table["separation"] = self.target_coord.separation(comp_table["coord"])

        # Add dummy column for sorting in the order we want
        comp_table["sort"] = np.zeros(len(comp_table))

        # Set sort order
        apass_mark = comp_table["marker name"] == "APASS comparison"
        vsx_mark = comp_table["marker name"] == "VSX"
        tess_mark = (comp_table["marker name"] == "TESS Targets") | (
            comp_table["separation"] < 0.3 * u.arcsec
        )

        comp_table["sort"][apass_mark] = 2
        comp_table["sort"][vsx_mark] = 1
        comp_table["sort"][tess_mark] = 0

        # Sort the table
        comp_table.sort(["sort", "separation"])

        # Assign the IDs
        comp_table["star_id"] = range(1, len(comp_table) + 1)

        return comp_table

    def show_labels(self):
        """
        Show the labels for the stars.

        Returns
        -------

        None
            Labels for the stars are shown.
        """
        plot_names = []
        comp_table = self.generate_table()

        original_mark = self.iw._marker
        for star in comp_table:
            star_id = star["star_id"]
            if star["marker name"] == "TESS Targets":
                label = f"T{star_id}"
                self.iw._marker = functools.partial(
                    self.iw.dc.Text,
                    text=label,
                    fontsize=20,
                    fontscale=False,
                    color="green",
                )
                self.iw.add_markers(
                    Table(data=[[star["x"] + 20], [star["y"] - 20]], names=["x", "y"]),
                    marker_name=self._label_name,
                )

            elif star["marker name"] == "APASS comparison":
                label = f"C{star_id}"
                self.iw._marker = functools.partial(
                    self.iw.dc.Text,
                    text=label,
                    fontsize=20,
                    fontscale=False,
                    color="red",
                )
                self.iw.add_markers(
                    Table(data=[[star["x"] + 20], [star["y"] - 20]], names=["x", "y"]),
                    marker_name=self._label_name,
                )

            elif star["marker name"] == "VSX":
                label = f"V{star_id}"
                self.iw._marker = functools.partial(
                    self.iw.dc.Text,
                    text=label,
                    fontsize=20,
                    fontscale=False,
                    color="blue",
                )
                self.iw.add_markers(
                    Table(data=[[star["x"] + 20], [star["y"] - 20]], names=["x", "y"]),
                    marker_name=self._label_name,
                )
            else:
                label = f"U{star_id}"
                print(f"Unrecognized marker name: {star['marker name']}")
            plot_names.append(label)
        self.iw._marker = original_mark

    def remove_labels(self):
        """
        Remove the labels for the stars.

        Returns
        -------

        None
            Labels for the stars are removed.
        """
        try:
            try:
                self.iw.remove_markers(marker_name=self._label_name)
            except AttributeError:
                self.iw.remove_markers_by_name(marker_name=self._label_name)
        except ValueError:
            # No labels, keep going
            pass

    def show_circle(self, radius=2.5 * u.arcmin, pixel_scale=0.56 * u.arcsec / u.pixel):
        """
        Show a circle around the target.

        Parameters
        ----------

        radius : `astropy.units.Quantity`, optional
            Radius of circle. The default is ``2.5*u.arcmin``.

        pixel_scale : `astropy.units.Quantity`, optional
            Pixel scale of image. The default is ``0.56*u.arcsec/u.pixel``.

        Returns
        -------

        None
            Circle is shown around the target.
        """
        radius_pixels = np.round((radius / pixel_scale).to(u.pixel).value, decimals=0)
        orig_marker = self.iw.marker
        self.iw.marker = {"color": "yellow", "radius": radius_pixels, "type": "circle"}
        self.iw.add_markers(
            Table(data=[[self.target_coord]], names=["coords"]),
            skycoord_colname="coords",
            use_skycoord=True,
            marker_name=self._circle_name,
        )
        self.iw.marker = orig_marker

    def remove_circle(self):
        """
        Remove the circle around the target.

        Returns
        -------

        None
            Circle is removed from the image.
        """
        try:
            self.iw.remove_markers(marker_name=self._circle_name)
        except AttributeError:
            self.iw.remove_markers_by_name(marker_name=self._circle_name)

    def tess_field_view(self):
        """
        Show the whole TESS field of view including circle around target,
        but hide labels.

        Returns
        -------

        None
            Shows image as described above.
        """

        # Show whole field of view
        self.iw.zoom_level = "fit"

        # Show the circle
        self.show_circle()

        # Turn off labels -- too cluttered
        self.remove_labels()

    def tess_field_zoom_view(self, width=6 * u.arcmin):
        """
        Zoom in on the TESS field of view.

        Parameters
        ----------

        width : `astropy.units.Quantity`, optional
            Width of field of view. The default is ``6*u.arcmin``.

        Returns
        -------

        None
            Zooms in on the image as described above.
        """

        # Turn off labels -- too cluttered
        self.remove_labels()

        left_side = self.ccd.wcs.pixel_to_world(0, self.ccd.shape[1] / 2)
        right_side = self.ccd.wcs.pixel_to_world(
            self.ccd.shape[0], self.ccd.shape[1] / 2
        )
        fov = left_side.separation(right_side)

        view_ratio = width / fov
        # fit first to get zoom level at full field of view
        self.iw.zoom_level = "fit"

        # Then set it to what we actually want...
        self.iw.zoom_level = self.iw.zoom_level / view_ratio

        # Show the circle
        self.show_circle()
