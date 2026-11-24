import html
import logging
import shutil
from pathlib import Path

import ipyautoui
import ipywidgets as ipw
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError
from astropy.table import Table, vstack
from astropy.utils.masked import Masked
from astrowidgets.bqplot import ImageWidget
from bqplot import Label, Lines
from ipyautoui.custom import FileChooser
from ipyautoui.custom.buttonbars import SaveButtonBar

from stellarphot import SourceListData
from stellarphot.gui.custom_widgets import Confirm, SettingWithTitle, Spinner
from stellarphot.gui.fits_opener import FitsOpener
from stellarphot.gui.views import ui_generator
from stellarphot.settings import (
    PartialPhotometrySettings,
    PhotometryWorkingDirSettings,
    SourceLocationSettings,
)
from stellarphot.utils.comparison_utils import (
    crossmatch_APASS2VSX,
    in_field,
    mag_scale,
    set_up,
)

__all__ = ["make_markers", "wrap", "ComparisonViewer"]

DESC_STYLE = {"description_width": "initial"}

# Markers this close together are one star; catalog positions of a star can
# differ by tenths of an arcsecond.
TARGET_MATCH_RADIUS = 2 * u.arcsec

# At most this many star_ids are listed in a message.
MAX_IDS_LISTED = 10

# What the help, the legend and the tooltips say depends on whether an
# existing source list is shown or one is generated.
_PAN_ZOOM_HELP = "<h3>Click and drag to pan, use the scroll wheel to zoom.</h3>"
_HELP = {
    "existing": _PAN_ZOOM_HELP
    + """
        <h3>An existing source list is view only: clicking a star does not
        change the list. To choose stars, generate a source list.</h3>
        """,
    "generate": _PAN_ZOOM_HELP
    + """
        <h3>Click on a star to mark it with a red +; click again to unmark
        it. Marked stars are still saved in the source list. Once a star is
        marked the labels are renumbered without it, so they no longer match
        the star_id values in the saved source list.</h3>
        """,
}
_LEGEND = {
    "existing": """
        <ul>
        <li>Green circles -- Targets in the source list</li>
        <li>Red diamonds -- Comparison stars in the source list</li>
        <li>Blue squares -- Variables in the source list</li>
        </ul>
        """,
    "generate": """
        <ul>
        <li>Green circles -- Gaia stars within 2.5 arcmin of target</li>
        <li>Red diamonds -- Comparison stars from the chosen catalog</li>
        <li>Blue squares -- VSX variables</li>
        <li>Red + -- Marked by clicking (still saved in the source list)</li>
        </ul>
        """,
}
_SAVE_TOOLTIP = {
    "existing": (
        "Copy the chosen source list to the source list file named in the "
        "source location settings"
    ),
    "generate": (
        "Save the source list shown to the source list file named in the "
        "source location settings"
    ),
}
_NO_VARIABLES_TOOLTIP = (
    "Not available for an existing source list: the variable star catalog "
    "is not queried"
)


def _coord_catalog_table(table):
    """
    Return a copy of ``table`` whose "coords" column is renamed to "coord",
    the SkyCoord column name astro-image-display-api uses by default, so
    that catalogs read back from the viewer combine cleanly whatever table
    they were loaded from.

    Parameters
    ----------

    table : `astropy.table.Table`
        Table with a "coords" (or already-renamed "coord") column of
        `astropy.coordinates.SkyCoord`.

    Returns
    -------

    `astropy.table.Table`
        Copy of the table with the column renamed.
    """
    out = Table(table)
    if "coord" not in out.colnames:
        if "coords" not in out.colnames:
            raise ValueError(
                "table must have a 'coords' (or 'coord') column of SkyCoord"
            )
        out.rename_column("coords", "coord")

    # Catalogs fetched through astroquery are masked tables, so a SkyCoord
    # built from their columns holds Masked arrays, which the high-level WCS
    # API used by the image widget refuses to transform. Rebuild the
    # coordinates from unmasked data.
    coord_col = out["coord"]
    ra = coord_col.ra.deg
    dec = coord_col.dec.deg
    if isinstance(ra, Masked) or isinstance(dec, Masked):
        out["coord"] = SkyCoord(
            ra=getattr(ra, "unmasked", ra),
            dec=getattr(dec, "unmasked", dec),
            unit="deg",
            frame=coord_col.frame.name,
        )
    return out


def _all_catalog_entries(iw):
    """
    Collect the entries of every catalog loaded in an image widget into a
    single table, tagged with the catalog each entry came from.

    Parameters
    ----------

    iw : `astrowidgets.bqplot.ImageWidget`
        Image widget to collect catalog entries from.

    Returns
    -------

    `astropy.table.Table`
        Table with "x", "y", "coord" and "marker name" columns, where
        "marker name" is the catalog label.
    """
    tables = []
    for label in iw.catalog_labels:
        # Keep only the columns the viewer manages. The catalogs also hold
        # every column of the table they were loaded from, and those columns
        # can collide with ones the viewer adds later (an input source list,
        # for example, already has xcenter/ycenter columns).
        entries = iw.get_catalog(catalog_label=label)["x", "y", "coord"]

        # Coordinates computed from a WCS are typically FK5 while catalog
        # coordinates are ICRS; vstack refuses to combine mismatched frames.
        coords = entries["coord"]
        if coords.frame.name != "icrs":
            entries["coord"] = coords.icrs

        entries["marker name"] = label
        tables.append(entries)

    if not tables:
        return Table(names=["x", "y", "coord", "marker name"])

    # The catalogs keep the metadata of the tables they were loaded from,
    # which typically conflicts between catalogs (e.g. the Vizier catalog
    # name). The metadata is not used here, so drop it silently.
    return vstack(tables, metadata_conflicts="silent")


def _read_source_list(path):
    """
    Read a source list made by the comparison viewer, adding a "coord" column
    of star positions (NaN where masked). Raise `ValueError`, saying why, if
    the file is not a source list the viewer can show and photometry can use.
    """
    try:
        sources = SourceListData.read(path)
    except Exception as err:
        raise ValueError(f"it could not be read as a source list ({err})") from err

    # SourceListData.read does not check the columns (issue #756). Photometry
    # reads the pixel columns and has_x_y even when it uses sky positions.
    missing = [
        name
        for name in ("star_id", "marker name", "ra", "dec", "xcenter", "ycenter")
        if name not in sources.colnames
    ]
    if missing:
        raise ValueError(f"it has no {' or '.join(missing)} column")
    if not sources.meta.get("has_ra_dec"):
        raise ValueError(
            "it was not saved as a stellarphot source list with sky "
            "positions (has_ra_dec is not true in its metadata)"
        )
    if "has_x_y" not in sources.meta:
        raise ValueError(
            "it was not saved as a stellarphot source list (there is no "
            "has_x_y in its metadata)"
        )
    unknown = set(sources["marker name"]) - {
        "TESS Targets",
        "VSX",
        "APASS comparison",
    }
    if unknown:
        raise ValueError(f"it has unknown marker names {', '.join(sorted(unknown))}")

    try:
        # A masked position is NaN, so the star counts as off the image.
        ra, dec = sources["ra"], sources["dec"]
        if isinstance(ra, Masked) or isinstance(dec, Masked):
            ra, dec = Masked(ra).filled(np.nan), Masked(dec).filled(np.nan)
        sources["coord"] = SkyCoord(ra, dec)
    except Exception as err:
        raise ValueError(f"its star positions could not be read ({err})") from err
    return sources


def make_markers(iw, RD, vsx, ent, name_or_coord=None):
    """
    Add markers for APASS, TESS targets, VSX.  Also center on object/coordinate.

    Parameters
    ----------

    iw : `astrowidgets.bqplot.ImageWidget`
        Image widget on which to show the markers.

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
        Markers are added to the image in the image widget.
    """
    # Show the whole image
    iw.set_viewport(fov=max(iw.get_image().data.shape))

    iw.remove_catalog(catalog_label="*")

    if RD:
        iw.load_catalog(
            _coord_catalog_table(RD),
            use_skycoord=True,
            catalog_label="TESS Targets",
            # Larger than the VSX squares so that the circle goes around the
            # square when a target is also a known variable.
            catalog_style={"shape": "circle", "color": "green", "size": 14},
        )

    if name_or_coord is not None:
        if isinstance(name_or_coord, str):
            iw.set_viewport(center=SkyCoord.from_name(name_or_coord))
        else:
            iw.set_viewport(center=name_or_coord)

    if vsx:
        iw.load_catalog(
            _coord_catalog_table(vsx),
            use_skycoord=True,
            catalog_label="VSX",
            catalog_style={"shape": "square", "color": "blue", "size": 10},
        )

    iw.load_catalog(
        _coord_catalog_table(ent),
        use_skycoord=True,
        catalog_label="APASS comparison",
        catalog_style={"shape": "diamond", "color": "red", "size": 10},
    )


def wrap(imagewidget, status_widget):
    """
    Utility function to let you click to select/deselect comparisons.

    Parameters
    ----------

    imagewidget : `astrowidgets.bqplot.ImageWidget`
        Image widget displaying the image and catalogs.

    status_widget : `ipywidgets.HTML`
        Widget in which to display messages for the user.

    """

    def cb(interaction, event_data, buffers):  # noqa: ARG001
        """
        Mouse event callback; bqplot calls it with the three arguments
        above. Click payloads look like
        ``{"event": "click", "domain": {"x": ..., "y": ...}}``.
        """
        if event_data.get("event") != "click":
            return

        x = event_data["domain"]["x"]
        y = event_data["domain"]["y"]
        out_skycoord = imagewidget.get_image().wcs.pixel_to_world(x, y)

        try:
            imagewidget.next_elim += 1
        except AttributeError:
            imagewidget.next_elim = 1

        all_table = _all_catalog_entries(imagewidget)
        if len(all_table) == 0:
            status_widget.value = "Click closer to a star"
            return

        index, d2d, d3d = out_skycoord.match_to_catalog_sky(all_table["coord"])
        if d2d < 10 * u.arcsec:
            # This click hit a star, so any leftover "click closer" message
            # no longer applies.
            status_widget.value = ""
            mouse = all_table["coord"][index].separation(all_table["coord"])
            rat = mouse <= TARGET_MATCH_RADIUS
            elims = [
                name
                for name in all_table["marker name"][rat]
                if name.startswith("elim")
            ]
            if not elims:
                imagewidget.load_catalog(
                    all_table[rat],
                    use_skycoord=True,
                    catalog_label=f"elim{imagewidget.next_elim}",
                    catalog_style={"shape": "plus", "color": "red", "size": 12},
                )
            else:
                for elim in elims:
                    imagewidget.remove_catalog(catalog_label=elim)
        else:
            status_widget.value = "Click closer to a star"

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
        Bright magnitude limit for comparison stars. Defaults to 8.

    dim_mag_limit : float, optional
        Dim magnitude limit for comparison stars.  Defaults to 15.

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

    comparison_catalog : str, optional
        Catalog to draw comparison stars from, "APASS DR9" or "ATLAS refcat2".
        Defaults to "APASS DR9".

    Attributes
    ----------

    photom_apertures_file : str
        File to save photometry aperture information to.

    box : `ipywidgets.Box`
        Box containing the widgets.

    bright_mag_limit : float
        Bright magnitude limit for comparison stars.

    comparison_catalog : str
        Catalog comparison stars are drawn from.

    dim_mag_limit : float
        Dim magnitude limit for comparison stars.

    iw : `astrowidgets.bqplot.ImageWidget`
        Image widget.

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
        dim_mag_limit=15,
        targets_from_file=None,
        object_coordinate=None,
        photom_apertures_file=None,
        overwrite_outputs=True,
        observatory=None,
        comparison_catalog="APASS DR9",
    ):
        self._circle_name = "target circle"
        # Maps a catalog label to the bqplot Label mark holding the text
        # labels for that catalog's stars.
        self._label_marks = {}
        self._file_chooser = FitsOpener()

        # On-screen controls for the comparison star settings, copied to the
        # attributes when the catalogs are queried with them.
        self._comparison_catalog_choice = ipw.Dropdown(
            options=["APASS DR9", "ATLAS refcat2"],
            value=comparison_catalog,
            description="Comparison catalog",
            style=DESC_STYLE,
        )
        self._bright_mag_limit_input = ipw.FloatText(
            value=bright_mag_limit,
            description="Bright mag limit (SR)",
            style=DESC_STYLE,
            layout={"width": "220px"},
        )
        self._dim_mag_limit_input = ipw.FloatText(
            value=dim_mag_limit,
            description="Dim mag limit (SR)",
            style=DESC_STYLE,
            layout={"width": "220px"},
        )
        self._regenerate_button = ipw.Button(
            description="Regenerate",
            tooltip=(
                "Query the catalogs with these settings and show the stars "
                "found. Nothing is saved."
            ),
            disabled=True,
        )
        # The first question: use an existing source list, or generate one.
        self._mode = ipw.ToggleButtons(
            options=[
                ("Use an existing source list", "existing"),
                ("Generate a source list", "generate"),
            ],
            value=None,
            description="Source list:",
            # The default button width cuts both labels short.
            style={**DESC_STYLE, "button_width": "auto"},
        )

        # Saves the source list. The bar's own handlers would mark it saved
        # even while it is waiting for confirmation, so they are replaced.
        self._comparison_save_bar = SaveButtonBar()
        save_bar = self._comparison_save_bar
        save_bar.bn_save.tooltip = _SAVE_TOOLTIP["generate"]
        save_bar.bn_save.on_click(save_bar._save, remove=True)
        save_bar.bn_save.on_click(self._save_clicked)
        save_bar.bn_revert.on_click(save_bar._revert, remove=True)
        save_bar.bn_revert.on_click(self._revert_comparison_settings)
        self._confirm_overwrite = Confirm(widget_to_hide=save_bar)

        self._directory = directory
        self.target_mag = target_mag
        self.bright_mag_limit = bright_mag_limit
        self.dim_mag_limit = dim_mag_limit
        self.comparison_catalog = comparison_catalog
        self.targets_from_file = targets_from_file
        self.target_coord = object_coordinate
        self.observatory = observatory

        # An existing source list chosen to be shown, its path, and its stars
        # on the current image.
        self._loaded_source_list = None
        self._loaded_source_list_path = None
        self._loaded_stars = None
        # The stars the catalog queries found, and the image they were found
        # for, kept so that coming back to generating a list does not query
        # again: (image path, targets, variables, comparison stars).
        self._generated = None
        # Counts the catalog queries, to tell one generated list from the next.
        self._generation = 0
        # What the last save wrote and where: (the path of the existing list
        # copied, or the generation of the generated list; the file written).
        # Saving to that file again replaces it without asking.
        self._saved = (None, None)
        # The file the user was last asked about replacing.
        self._asked_about = None

        # This function defines several attributes in addition to returning the box and
        # image viewer. You should take a look at it to see what it does.
        self.box, self.iw = self._viewer()

        self.photometry_settings = PhotometryWorkingDirSettings()

        if photom_apertures_file is not None:
            original_settings = self.source_locations.value.copy()
            original_settings["source_list_file"] = photom_apertures_file
            self.source_locations.value = original_settings

        self.overwrite_outputs = overwrite_outputs

        if file:
            self._file_chooser.set_file(file, directory=directory)

        # Save the source location settings
        self.source_locations.savebuttonbar.bn_save.click()

        self._update_save_bar()
        self._make_observers()

    def _init(self):
        """
        Handles aspects of initialization that need to be deferred until
        a file is chosen.
        """
        # Read in the ccd data here and load the image into viewer instead of doing
        # it behind the scenes in set_up and make_markers
        self.ccd = self.fits_file.ccd
        self.fits_file.load_in_image_widget(self.iw)

        legend = self._legend
        if self._mode.value == "existing":
            self._set_object()
            # Drop any message from an earlier catalog query.
            self._legend_spinner_box.children = [legend]
            if self._loaded_source_list is not None:
                self._show_loaded_source_list()
            return

        if self._have_generated_stars():
            self._show_generated_stars()
            return

        # A new query uses the settings on screen.
        if self._dim_mag_limit_input.value > self._bright_mag_limit_input.value:
            self.comparison_catalog = self._comparison_catalog_choice.value
            self.bright_mag_limit = self._bright_mag_limit_input.value
            self.dim_mag_limit = self._dim_mag_limit_input.value

        spinner = Spinner(message="Loading variable/comparison stars")
        spinner.start()
        self._legend_spinner_box.children = [spinner]
        self.help_stuff.selected_index = 1

        try:
            # Apply the same dim magnitude limit to the VSX lookup that is used
            # for the comparison stars so faint variables are not included
            # (issue #43).
            self.vsx = set_up(self.ccd, magnitude_limit=self.dim_mag_limit)

            apass, vsx_apass_angle, targets_apass_angle = crossmatch_APASS2VSX(
                self.ccd,
                self.targets_from_file,
                self.vsx,
                catalog=self.comparison_catalog,
                magnitude_limit=self.dim_mag_limit,
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

            self._generated = (
                self.fits_file.path,
                self.targets_from_file,
                self.vsx,
                apass_comps,
            )
            self._show_generated_stars()
        except Exception as err:
            # An exception raised here is invisible to the user -- this runs in
            # a widget callback, whose traceback only goes to the log -- so put
            # the error on screen, then re-raise so the log gets the traceback.
            self._generated = None
            error_message = ipw.HTML(
                value="<p style='color: red'>Loading variable/comparison stars "
                f"failed: {html.escape(str(err))}</p>"
            )
            self._legend_spinner_box.children = [legend, error_message]
            raise
        else:
            self._generation += 1

    def _have_generated_stars(self):
        """
        Whether the catalogs have been queried for the image chosen.
        """
        return self._generated is not None and (
            self._generated[0] == self.fits_file.path
        )

    def _show_generated_stars(self):
        """
        Mark the stars the catalog queries found.
        """
        _, targets, vsx, comparisons = self._generated
        # Set the object here so that the viewer is properly centered
        self._set_object()
        make_markers(
            self.iw, targets, vsx, comparisons, name_or_coord=self.target_coord
        )
        children = [self._legend]
        if len(comparisons) == 0:
            children.append(
                ipw.HTML(
                    value="<p style='color: red'>No comparison stars found "
                    "within the magnitude limits</p>"
                )
            )
        self._legend_spinner_box.children = children

    @property
    def fits_file(self):
        return self._file_chooser

    @property
    def photom_apertures_file(self):
        return self.source_locations.value["source_list_file"]

    @property
    def variables(self):
        """
        An `astropy.table.Table` of the variables in the class.

        The ``star_id`` column is the star's ID in the source list. A variable
        that is also a target has the ``star_id`` of the target.
        """
        comp_table = self.generate_table()
        marked = self._marked_stars()
        vsx_coord = marked["coord"][marked["marker name"] == "VSX"]
        idx, _, _ = vsx_coord.match_to_catalog_sky(self.vsx["coords"])
        our_vsx = self.vsx[idx]
        star_id = np.zeros(len(our_vsx), dtype=int)

        # A variable that generate_table left out for being a target gets the
        # ID of that target; the rest are rows of the source list themselves.
        # Matching to no rows is an error, hence the checks.
        is_target = np.zeros(len(our_vsx), dtype=bool)
        targets = comp_table[comp_table["marker name"] == "TESS Targets"]
        if len(targets) > 0 and len(our_vsx) > 0:
            idx, distance, _ = vsx_coord.match_to_catalog_sky(targets["coord"])
            is_target = distance <= TARGET_MATCH_RADIUS
            star_id[is_target] = targets["star_id"][idx[is_target]]
        if not is_target.all():
            vsx_rows = comp_table[comp_table["marker name"] == "VSX"]
            idx, _, _ = vsx_coord[~is_target].match_to_catalog_sky(vsx_rows["coord"])
            star_id[~is_target] = vsx_rows["star_id"][idx]

        our_vsx["star_id"] = star_id
        our_vsx.sort("star_id")
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
            self._object = ""

        # The first target in an existing source list that is on the image,
        # or else the first object in the input source list aka target file,
        # if there is one, is the target, whatever the image header calls the
        # object.
        if self._mode.value == "existing" and self._loaded_source_list is not None:
            sources = self._loaded_source_list
            targets = sources["coord"][sources["marker name"] == "TESS Targets"]
            if len(targets) > 0:
                targets = targets[self._on_image(targets)[2]]
            if len(targets) > 0:
                self.target_coord = targets[0]
                return

        if self.targets_from_file:
            self.target_coord = self.targets_from_file["coords"][0]
            return

        # Otherwise try to get coordinates from the name, and use the center of
        # the image if that fails.
        try:
            self.target_coord = SkyCoord.from_name(self._object)
        except NameResolveError:
            # The shape is (rows, columns), i.e. (y, x)
            self.target_coord = self.ccd.wcs.pixel_to_world(
                self.ccd.shape[1] / 2, self.ccd.shape[0] / 2
            )

    def _set_file(self, change):  # noqa: ARG002
        """
        Widget callbacks need to accept a change argument, even if not used.
        """
        self._update_comparisons()

    def _set_target_file(self, change):
        """
        Set the target file.
        """
        input_source_list_name = change["new"]
        self.targets_from_file = SourceListData.read(input_source_list_name)
        self._update_comparisons()

    def _mode_changed(self, change):
        """
        Show the controls for the mode chosen and the stars for it.
        """
        mode = change["new"]
        existing = mode == "existing"
        self._existing_list_controls.layout.display = "flex" if existing else "none"
        self._generate_list_controls.layout.display = "none" if existing else "flex"
        self._rest_of_viewer.layout.display = "flex"
        self._help.value = _HELP[mode]
        self._legend.value = _LEGEND[mode]
        self._comparison_save_bar.bn_save.tooltip = _SAVE_TOOLTIP[mode]
        # The variables come from the catalog query, which is not made for an
        # existing source list.
        self._save_var_info.disabled = existing
        self._save_var_info.tooltip = _NO_VARIABLES_TOOLTIP if existing else ""
        self._status_message.value = ""
        self._update_comparisons(requery=False)

    def _source_list_chosen(self, change):
        """
        Load the existing source list chosen to be shown.
        """
        # "." is the value the chooser is reset to, not a choice.
        if change["new"] != ".":
            self._load_source_list(change["new"])

    def _load_source_list(self, path):
        """
        Show the existing source list at ``path``.

        Parameters
        ----------

        path : str
            Path of the source list.

        Notes
        -----
        The list is view only: clicking does not exclude stars. Saving copies
        it to the source list file. A file that is not a usable source list
        leaves the viewer as it was, with a message saying why.
        """
        name = Path(path).name
        try:
            sources = _read_source_list(path)
        except ValueError as err:
            message = f"{name} is not a usable source list: {err}."
            if self._loaded_source_list is not None:
                message += f" Still showing {Path(self._loaded_source_list_path).name}."
            self._source_list_message.value = (
                f"<span style='color: red'>{html.escape(message)}</span>"
            )
            self._reset_source_list_chooser()
            return

        self._loaded_source_list = sources
        self._loaded_source_list_path = str(path)
        self._loaded_stars = None
        self._status_message.value = ""
        self._source_list_message.value = html.escape(
            f"Showing {name}, view only. The save button below the image "
            "chooser copies it to the source list file."
        )
        self._update_comparisons()

    def _reset_source_list_chooser(self):
        # After this, choosing any file, even the last one again, is a change.
        self._choose_source_list.reset()
        self._choose_source_list._value = "."

    def _on_image(self, coords):
        """
        The pixel positions of ``coords`` on the image, and whether each is
        on the image.
        """
        x, y = self.ccd.wcs.world_to_pixel(coords)
        n_y, n_x = self.ccd.shape
        # Stars on the edge count as on the image. NaN positions are off it.
        return x, y, (x >= 0) & (x <= n_x) & (y >= 0) & (y <= n_y)

    def _show_loaded_source_list(self):
        """
        Mark the stars of the loaded source list that are on the image, and
        list the ``star_id`` of those that are not.
        """
        sources = self._loaded_source_list
        x, y, on_image = self._on_image(sources["coord"])
        on = sources[on_image]
        self._loaded_stars = Table(
            {
                "x": x[on_image],
                "y": y[on_image],
                "coord": on["coord"],
                "marker name": on["marker name"],
                "star_id": on["star_id"],
            }
        )

        def stars(marker_name):
            return Table({"coords": on["coord"][on["marker name"] == marker_name]})

        make_markers(
            self.iw,
            stars("TESS Targets"),
            stars("VSX"),
            stars("APASS comparison"),
            name_or_coord=self.target_coord,
        )

        children = [self._legend]
        off_ids = [str(star_id) for star_id in sources["star_id"][~on_image]]
        if off_ids:
            listed = ", ".join(off_ids[:MAX_IDS_LISTED])
            if len(off_ids) > MAX_IDS_LISTED:
                listed += f" and {len(off_ids) - MAX_IDS_LISTED} more"
            if len(on) == 0:
                # Most likely a source list for another field.
                name = Path(self._loaded_source_list_path).name
                style = " style='color: red'"
                message = (
                    f"None of the {len(off_ids)} stars in {name} are on this "
                    f"image. Is it the source list for this field? (star_id "
                    f"{listed})"
                )
            else:
                style = ""
                stars_are = "star is" if len(off_ids) == 1 else "stars are"
                message = (
                    f"{len(off_ids)} {stars_are} off this image and not "
                    f"marked: star_id {listed}"
                )
            children.append(ipw.HTML(value=f"<p{style}>{html.escape(message)}</p>"))
        self._legend_spinner_box.children = children
        self.help_stuff.selected_index = 1

    def _regenerate(self, button=None):  # noqa: ARG002
        """
        Query the catalogs with the settings on screen and show the stars
        found. Nothing is written.
        """
        previous = (
            self.comparison_catalog,
            self.bright_mag_limit,
            self.dim_mag_limit,
            self._generated,
        )
        self._generated = None
        try:
            self._reload_comparisons()
        except Exception:
            # The stars on screen are still the ones for the previous
            # settings, so keep those settings; it can be tried again.
            (
                self.comparison_catalog,
                self.bright_mag_limit,
                self.dim_mag_limit,
                self._generated,
            ) = previous
            raise
        finally:
            self._update_save_bar()

    def _save_clicked(self, button=None):  # noqa: ARG002
        # Clear any message from the last save.
        self._update_save_bar()
        self._save_source_list()

    def _save_source_list(self, confirmed=None):
        """
        Write the source list if it needs saving, first asking before
        replacing a file, unless it is the file this session wrote last.

        Parameters
        ----------

        confirmed : `pathlib.Path`, optional
            Resolved path of a file the user has agreed to replace.
        """
        if self._settings_not_applied() or not self._source_list_needs_saving():
            return
        destination = Path(self.photom_apertures_file).resolve()
        # Without overwrite_outputs the write is refused, with a message, so
        # there is nothing to ask.
        if (
            destination.exists()
            and self.overwrite_outputs
            and destination not in (self._saved[1], confirmed)
        ):
            self._asked_about = destination
            self._confirm_overwrite.message = (
                f"{html.escape(self.photom_apertures_file)} already exists. "
                "Replace it with the source list shown? No keeps the old "
                "file, which photometry will use."
            )
            # Reset so that the answer is a change even if it is the same as
            # the last one.
            self._confirm_overwrite.value = None
            self._confirm_overwrite.show()
            return
        self._write_source_list()

    def _overwrite_answered(self, change):
        # What is on screen may have changed while the question was showing,
        # so go through the checks again.
        if change["new"]:
            self._save_source_list(confirmed=self._asked_about)

    def _write_source_list(self):
        """
        Write the source list to the source list file, or copy the existing
        source list there.
        """
        destination = self.photom_apertures_file
        resolved = Path(destination).resolve()
        loaded = self._loaded_source_list_path
        try:
            if self._mode.value == "existing":
                if resolved.exists() and not self.overwrite_outputs:
                    raise OSError(
                        f"Existing file ({destination}) can not be overwritten. "
                        "Set overwrite_outputs=True to address this."
                    )
                shutil.copyfile(loaded, destination)
                self._saved = (Path(loaded).resolve(), resolved)
                done = f"Copied {Path(loaded).name} to {destination}"
            else:
                self._save_aperture_to_file(None)
                self._saved = (self._generation, resolved)
                done = f"Source list saved to {destination}"
        except Exception as err:
            # Show the error, since the traceback only goes to the log.
            self._comparison_save_bar.message.value = (
                f"<span style='color: red'>{html.escape(str(err))}</span>"
            )
            raise
        if (
            self._mode.value == "generate"
            and loaded is not None
            and Path(loaded).resolve() == resolved
        ):
            # The existing list that was chosen is no longer what is in its
            # file, so it is not shown again as if it were.
            self._loaded_source_list = None
            self._loaded_source_list_path = None
            self._loaded_stars = None
            self._reset_source_list_chooser()
            self._source_list_message.value = html.escape(
                f"{Path(loaded).name} was replaced by the generated source "
                "list, so it is no longer shown. Choose a source list."
            )
        self._update_save_bar()
        self._comparison_save_bar.message.value = f"<i>{html.escape(done)}</i>"

    def _revert_comparison_settings(self, button=None):  # noqa: ARG002
        self._comparison_catalog_choice.value = self.comparison_catalog
        self._bright_mag_limit_input.value = self.bright_mag_limit
        self._dim_mag_limit_input.value = self.dim_mag_limit

    def _settings_not_applied(self):
        """
        Whether the stars shown were found with settings other than those on
        screen.
        """
        # The settings are only used, and only shown, when generating a
        # source list, and with no image there are no stars.
        return (
            self._mode.value == "generate"
            and self._image_chosen()
            and (
                self._comparison_catalog_choice.value,
                self._bright_mag_limit_input.value,
                self._dim_mag_limit_input.value,
            )
            != (self.comparison_catalog, self.bright_mag_limit, self.dim_mag_limit)
        )

    def _source_list_needs_saving(self):
        """
        Whether there is a source list on screen that is not yet in the
        source list file.
        """
        destination = Path(self.photom_apertures_file).resolve()
        if self._mode.value == "existing":
            if self._loaded_source_list_path is None:
                return False
            chosen = Path(self._loaded_source_list_path).resolve()
            return chosen != destination and (chosen, destination) != self._saved
        if self._mode.value == "generate":
            return (
                len(self._marked_stars(include_excluded=True)) > 0
                and (self._generation, destination) != self._saved
            )
        return False

    def _update_save_bar(self, change=None):  # noqa: ARG002
        """
        Enable the save and regenerate buttons if there is anything for them
        to do, and say what there is to do.
        """
        generate = self._mode.value == "generate"
        not_applied = self._settings_not_applied()
        list_unsaved = self._source_list_needs_saving()
        # Any other limits are usable; if no stars fall within them a message
        # says so after the query.
        bad_limits = not_applied and (
            self._dim_mag_limit_input.value <= self._bright_mag_limit_input.value
        )
        destination = self.photom_apertures_file
        if bad_limits:
            message = "Dim limit must be fainter than bright limit"
        elif not_applied:
            message = (
                "Click Regenerate to show the stars for these settings before saving"
            )
        elif list_unsaved and generate:
            message = f"The source list has not been saved to {destination} yet"
        elif list_unsaved:
            name = Path(self._loaded_source_list_path).name
            message = f"{name} has not been copied to {destination} yet"
        elif (
            generate
            and self._image_chosen()
            and len(self._marked_stars(include_excluded=True)) == 0
        ):
            message = "There is no source list to save"
        else:
            message = ""
        save_bar = self._comparison_save_bar
        save_bar.message.value = (
            f"<span style='color: red'>{html.escape(message)}</span>" if message else ""
        )
        save_bar.unsaved_changes = not_applied or list_unsaved
        # The file always holds the stars for the settings on screen.
        save_bar.bn_save.disabled = not_applied or not list_unsaved
        save_bar.bn_revert.disabled = not not_applied
        # Also enabled to try again after a query that failed.
        self._regenerate_button.disabled = (
            not (generate and self._image_chosen())
            or bad_limits
            or not (not_applied or not self._have_generated_stars())
        )

    def _image_chosen(self):
        return self._file_chooser.file_chooser.value != "."

    def _reload_comparisons(self):
        # Labels are for the old stars, so they are removed first and redrawn
        # if they were on.
        self.remove_labels()
        try:
            self._init()
        finally:
            if self._show_labels_button.value:
                self.show_labels()

    def _update_comparisons(self, requery=True):
        """
        Show the stars for the mode chosen, if an image has been chosen.
        Nothing is written until the save bar is clicked.

        Parameters
        ----------
        requery : bool, optional
            Whether to query the catalogs even if the stars for this image
            have already been found.
        """
        try:
            if self._mode.value is not None and self._image_chosen():
                # Take the stars shown before off the image, so that they are
                # not left on it if the catalog query fails. A failed
                # Regenerate keeps them, since they are still the stars for
                # the settings in use.
                self.iw.remove_catalog(catalog_label="*")
                if requery and self._mode.value == "generate":
                    self._generated = None
                self._reload_comparisons()
        finally:
            self._update_save_bar()

    def _make_observers(self):
        for control in (
            self._comparison_catalog_choice,
            self._bright_mag_limit_input,
            self._dim_mag_limit_input,
        ):
            control.observe(self._update_save_bar, names="value")
        self.source_locations.observe(self._update_save_bar, names="_value")
        self._mode.observe(self._mode_changed, names="value")
        self._confirm_overwrite.observe(self._overwrite_answered, names="value")
        self._regenerate_button.on_click(self._regenerate)
        self._show_labels_button.observe(self._show_label_button_handler, names="value")
        self._save_var_info.on_click(self._save_variables_to_file)
        self._file_chooser.file_chooser.observe(self._set_file, names="_value")
        self._choose_input_source_list.observe(self._set_target_file, names="_value")
        self._choose_source_list.observe(self._source_list_chosen, names="_value")

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

        # Convert aperture table into a SourceList objects and output. Stars
        # excluded by clicking are kept, for now (issue #752).
        targets_table = self.generate_table(include_excluded=True)
        if len(targets_table) == 0:
            return
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
                source_location_settings=self.source_locations.value,
            ),
            update=True,
        )

        # For some reason the value of unsaved_changes is not updated until after this
        # function executes, so we force its value here.
        self.source_locations.savebuttonbar.unsaved_changes = False
        # Update the save box title to reflect the save
        self.source_and_title.decorate_title()

    def _viewer(self):
        self._help = ipw.HTML(value=_HELP["generate"])
        self._legend = ipw.HTML(value=_LEGEND["generate"])

        iw = ImageWidget()

        # Messages for the user (e.g. "Click closer to a star") show up here.
        self._status_message = ipw.HTML()
        select_stars = wrap(iw, self._status_message)

        def click_handler(interaction, event_data, buffers):
            # An existing source list is view only, so clicks do not exclude
            # stars.
            if self._mode.value != "existing":
                select_stars(interaction, event_data, buffers)
            elif event_data.get("event") == "click":
                self._status_message.value = (
                    "An existing source list is view only; generate a source "
                    "list to choose stars"
                )

        iw._astro_im.interaction.on_msg(click_handler)

        self.object_name = ipw.HTML(value="<h2>Object: </h2>")
        self._object = None
        controls = self._make_control_bar()

        # Capture the logging message issued about setting the file chooser to a file
        # that does not exist.
        original_logging_level = ipyautoui.custom.filechooser.logger.level
        ipyautoui.custom.filechooser.logger.setLevel(logging.CRITICAL)

        # The logging message will be generated here if it is generated.
        self.source_locations = ui_generator(
            SourceLocationSettings, max_field_width="75px"
        )
        ipyautoui.custom.filechooser.logger.setLevel(original_logging_level)

        self.source_and_title = SettingWithTitle(
            "Source location settings", self.source_locations
        )

        self.source_locations.savebuttonbar.fns_onsave_add_action(self.save)

        # Put the legend in a box whose children can be changed to a
        # spinner while loading.
        self._legend_spinner_box = ipw.VBox()
        self._legend_spinner_box.children = [self._legend]

        self.help_stuff = ipw.Accordion(children=[self._help, self._legend_spinner_box])
        self.help_stuff.titles = ["Help", "Legend"]
        box = ipw.VBox()
        inner_box = ipw.HBox()
        source_legend_box = ipw.VBox()
        source_legend_box.children = [
            self.help_stuff,
            self.object_name,
            self.source_and_title,
        ]

        # Put the status message below the image so messages are visible.
        image_box = ipw.VBox()
        image_box.children = [iw, self._status_message]
        inner_box.children = [image_box, source_legend_box]

        # Add a file chooser for an input source list
        self._choose_input_source_list = FileChooser(
            filter_pattern="*.ecsv",
        )
        self._choose_input_source_list.title = "OPTIONAL: Choose input source list"

        # Add a file chooser for an existing source list to show, view only.
        self._choose_source_list = FileChooser(filter_pattern="*.ecsv")
        self._choose_source_list.title = "Choose an existing source list"
        self._source_list_message = ipw.HTML()
        self._existing_list_controls = ipw.VBox(
            children=[self._choose_source_list, self._source_list_message]
        )

        comparison_settings = ipw.HBox(
            children=[
                self._comparison_catalog_choice,
                self._bright_mag_limit_input,
                self._dim_mag_limit_input,
                self._regenerate_button,
            ]
        )
        self._generate_list_controls = ipw.VBox(
            children=[comparison_settings, self._choose_input_source_list]
        )

        self._rest_of_viewer = ipw.VBox(
            children=[
                self._file_chooser.file_chooser,
                self._comparison_save_bar,
                self._confirm_overwrite,
                inner_box,
                controls,
            ]
        )

        # Only the first question shows until it is answered.
        for widget in (
            self._existing_list_controls,
            self._generate_list_controls,
            self._rest_of_viewer,
        ):
            widget.layout.display = "none"

        box.children = [
            self._mode,
            self._existing_list_controls,
            self._generate_list_controls,
            self._rest_of_viewer,
        ]

        return box, iw

    def _marked_stars(self, include_excluded=False):
        """
        The stars marked in the viewer.

        Parameters
        ----------
        include_excluded : bool, optional
            Whether to include the stars the user has excluded by clicking.

        Returns
        -------
        `astropy.table.Table`
            Table with "x", "y", "coord" and "marker name" columns.
        """
        all_table = _all_catalog_entries(self.iw)
        if len(all_table) == 0:
            return all_table

        elims = np.array([name.startswith("elim") for name in all_table["marker name"]])
        elim_table = all_table[elims]
        marked = all_table[~elims]
        if include_excluded:
            return marked

        index, d2d, d3d = elim_table["coord"].match_to_catalog_sky(marked["coord"])
        marked.remove_rows(index)
        return marked

    def generate_table(self, include_excluded=False):
        """
        Generate the table of stars to use for the aperture file.

        A variable or comparison star within ``TARGET_MATCH_RADIUS`` of a
        target is the target, and is left out so that each star has one row.
        When showing an existing source list, the table holds the stars of
        that list that are on the image, with the ``star_id`` they have in the
        list.

        Parameters
        ----------
        include_excluded : bool, optional
            Whether to include the stars the user has excluded by clicking.

        Returns
        -------
        comp_table : `astropy.table.Table`
            Table of stars to use for the aperture file.
        """
        if self._mode.value == "existing" and self._loaded_stars is not None:
            return self._loaded_stars.copy()

        comp_table = self._marked_stars(include_excluded=include_excluded)
        if len(comp_table) == 0:
            return comp_table

        tess_mark = comp_table["marker name"] == "TESS Targets"
        if tess_mark.any():
            _, target_distance, _ = comp_table["coord"].match_to_catalog_sky(
                comp_table["coord"][tess_mark]
            )
            comp_table = comp_table[tess_mark | (target_distance > TARGET_MATCH_RADIUS)]

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
        tess_mark = comp_table["marker name"] == "TESS Targets"

        # Although the radec input sources lists generated for TESS are supposed to be
        # sorted by distance from the TESS target, they are not always. To ensure that
        # the TESS order is preserved we add a column here that assigns a temporary
        # star ID used for sorting, which we later delete.
        if tess_mark.sum() > 0:
            # If there are multiple TESS targets then assign their sort value so that
            # order is not changed.
            comp_table["sort"][tess_mark] = np.arange(tess_mark.sum())

        # Targets are 0 to n_targets - 1; VSX stars next, then comparisons.
        n_targets = tess_mark.sum()
        comp_table["sort"][apass_mark] = n_targets + 1
        comp_table["sort"][vsx_mark] = n_targets

        # Sort the table
        comp_table.sort(["sort", "separation"])

        # Assign the IDs
        comp_table["star_id"] = range(1, len(comp_table) + 1)

        # Remove the helper columns that were only added to control the sort order
        # so they do not end up in the aperture file (issue #34).
        comp_table.remove_columns(["separation", "sort"])

        return comp_table

    def show_labels(self):
        """
        Show the labels for the stars.

        Returns
        -------

        None
            Labels for the stars are shown.
        """
        label_size = 15
        comp_table = self.generate_table()

        # Each catalog gets one bqplot Label mark holding the labels for all
        # of its stars.
        label_prefix_color = {
            "TESS Targets": ("T", "green"),
            "APASS comparison": ("C", "red"),
            "VSX": ("V", "blue"),
        }

        scales = {
            "x": self.iw._astro_im._scales["x"],
            "y": self.iw._astro_im._scales["y"],
        }

        for marker_name, (prefix, color) in label_prefix_color.items():
            stars = comp_table[comp_table["marker name"] == marker_name]
            if len(stars) == 0:
                continue

            labels = [f"{prefix}{star_id}" for star_id in stars["star_id"]]
            mark = Label(
                x=np.asarray(stars["x"], dtype=float) + 20,
                y=np.asarray(stars["y"], dtype=float) - 20,
                text=labels,
                scales=scales,
                colors=[color],
                default_size=label_size,
            )
            self._label_marks[marker_name] = mark
            # Mirror the mark into the viewer's own mark dictionary so the
            # labels survive the viewer's _update_marks calls (e.g. when a
            # catalog is added or removed).
            self.iw._astro_im._scatter_marks[f"__label__{marker_name}"] = mark

        self.iw._astro_im._update_marks()

    def remove_labels(self):
        """
        Remove the labels for the stars.

        Returns
        -------

        None
            Labels for the stars are removed.
        """
        for name in list(self._label_marks):
            del self._label_marks[name]
            self.iw._astro_im._scatter_marks.pop(f"__label__{name}", None)
        self.iw._astro_im._update_marks()

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
        radius_pixels = (radius / pixel_scale).to(u.pixel).value

        # Draw the circle as a bqplot Lines mark rather than a catalog
        # marker: catalog marker sizes are in screen pixels, so they cannot
        # represent a fixed on-sky radius, and a catalog entry would also
        # show up as a clickable "star" in the click handler and in
        # generate_table.
        x, y = self.iw.get_image().wcs.world_to_pixel(self.target_coord)
        theta = np.linspace(0, 2 * np.pi, 200)
        mark = Lines(
            x=x + radius_pixels * np.cos(theta),
            y=y + radius_pixels * np.sin(theta),
            scales={
                "x": self.iw._astro_im._scales["x"],
                "y": self.iw._astro_im._scales["y"],
            },
            colors=["yellow"],
        )
        # Put the mark in the viewer's own mark dictionary, the same pattern
        # show_labels uses, so the circle survives the viewer's _update_marks
        # calls.
        self.iw._astro_im._scatter_marks[f"__circle__{self._circle_name}"] = mark
        self.iw._astro_im._update_marks()

    def remove_circle(self):
        """
        Remove the circle around the target.

        Returns
        -------

        None
            Circle is removed from the image.
        """
        self.iw._astro_im._scatter_marks.pop(f"__circle__{self._circle_name}", None)
        self.iw._astro_im._update_marks()

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
        self.iw.set_viewport(fov=max(self.ccd.data.shape))

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

        # The viewport accepts an angular field of view directly
        self.iw.set_viewport(fov=width)

        # Show the circle
        self.show_circle()
