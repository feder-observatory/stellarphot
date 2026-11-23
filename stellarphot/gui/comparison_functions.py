import html
import logging
import os
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
from stellarphot.core import coords_in_frame
from stellarphot.gui.custom_widgets import SettingWithTitle, Spinner
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
    Read a source list made by the comparison viewer.

    Parameters
    ----------

    path : str
        Path of the source list.

    Returns
    -------

    `stellarphot.SourceListData`
        The source list, with a "coord" column of the star positions. A
        masked position is NaN.

    Raises
    ------

    ValueError
        If the file cannot be read, or is not a source list the viewer can
        show and photometry can use; the message says why.
    """
    try:
        sources = SourceListData.read(path)
        missing = [
            name
            for name in ("star_id", "marker name", "ra", "dec")
            if name not in sources.colnames
        ]
        if missing:
            raise ValueError(f"it has no {' or '.join(missing)} column")
        if "has_ra_dec" not in sources.meta:
            raise ValueError(
                "it was not saved as a stellarphot source list "
                "(no has_ra_dec in its metadata)"
            )
        if not sources.meta["has_ra_dec"]:
            raise ValueError("it has no sky positions (ra, dec)")
        unknown = set(sources["marker name"]) - {
            "TESS Targets",
            "VSX",
            "APASS comparison",
        }
        if unknown:
            raise ValueError(
                f"it has unknown marker names {', '.join(sorted(unknown))}"
            )
        # A masked position is NaN, so the star counts as off the image.
        ra, dec = sources["ra"], sources["dec"]
        if isinstance(ra, Masked) or isinstance(dec, Masked):
            ra, dec = Masked(ra).filled(np.nan), Masked(dec).filled(np.nan)
        sources["coord"] = SkyCoord(ra, dec)
    except ValueError:
        raise
    except Exception as err:
        raise ValueError(f"it could not be read as a source list ({err})") from err
    return sources


def _same_file(path, other_path):
    """
    Whether two paths name the same existing file.
    """
    try:
        return os.path.samefile(path, other_path)
    except OSError:
        return False


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
        # attributes when they are saved.
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
        self._comparison_save_bar = SaveButtonBar()
        self._comparison_save_bar.bn_save.tooltip = (
            "Re-query the catalog; resets any stars you removed"
        )
        self._comparison_save_bar.fns_onsave_add_action(self._save_comparison_settings)
        self._comparison_save_bar.fns_onrevert_add_action(
            self._revert_comparison_settings
        )

        self._directory = directory
        self.target_mag = target_mag
        self.bright_mag_limit = bright_mag_limit
        self.dim_mag_limit = dim_mag_limit
        self.comparison_catalog = comparison_catalog
        self._comparison_settings_changed()
        self.targets_from_file = targets_from_file
        self.target_coord = object_coordinate
        self.observatory = observatory

        # A source list chosen to be shown instead of one made from the
        # catalogs, its path, its stars on the current image, and the source
        # list file name in use before it was chosen.
        self._loaded_source_list = None
        self._loaded_source_list_path = None
        self._loaded_stars = None
        self._generated_source_list_file = None

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
            self._set_file(None)

        # Save the source location settings
        self.source_locations.savebuttonbar.bn_save.click()

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

        if self._loaded_source_list is not None:
            self._set_object()
            self._show_loaded_source_list()
            return

        spinner = Spinner(message="Loading variable/comparison stars")
        legend = self._legend_spinner_box.children[0]
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

            # Set the object here so that the viewer is properly centered
            self._set_object()

            make_markers(
                self.iw,
                self.targets_from_file,
                self.vsx,
                apass_comps,
                name_or_coord=self.target_coord,
            )
        except Exception as err:
            # An exception raised here is invisible to the user -- this runs in
            # a widget callback, whose traceback only goes to the log -- so put
            # the error on screen, then re-raise so the log gets the traceback.
            error_message = ipw.HTML(
                value="<p style='color: red'>Loading variable/comparison stars "
                f"failed: {html.escape(str(err))}</p>"
            )
            self._legend_spinner_box.children = [legend, error_message]
            raise
        else:
            children = [legend]
            if len(apass_comps) == 0:
                message = "No comparison stars found within the magnitude limits"
                if len(self.generate_table()) == 0:
                    # Nothing at all is marked, so no source list is written
                    # and one from an earlier load would be left as it was.
                    message += "; the source list was not updated"
                children.append(ipw.HTML(value=f"<p style='color: red'>{message}</p>"))
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

        # The first target in a loaded source list, or else the first object
        # in the input source list aka target file, if there is one, is the
        # target, whatever the image header calls the object.
        if self._loaded_source_list is not None:
            sources = self._loaded_source_list
            targets = sources[sources["marker name"] == "TESS Targets"]
            if len(targets) > 0:
                self.target_coord = targets["coord"][0]
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

    def _save_source_location_file(self):
        """
        Save the source location file.
        """
        # Save the initial source list if there is a file name for it and if the target
        # coordinates are known. The target coordinates are needed to generate the
        # source list table. A loaded source list is never written.
        if (
            self._loaded_source_list is None
            and self.source_locations.value["source_list_file"] is not None
            and self.target_coord is not None
        ):
            self._save_aperture_to_file(None)

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

    def _source_list_chosen(self, change):
        """
        Load the source list chosen to be shown.
        """
        # "." is the value the chooser is reset to, not a choice.
        if change["new"] != ".":
            self._load_source_list(change["new"])

    def _reset_source_list_chooser(self):
        # After this, choosing any file, even the one chosen before, is a change.
        self._choose_source_list.reset()
        self._choose_source_list._value = "."

    def _set_source_list_file(self, file_name):
        """
        Set the source list file in the source location settings and save them.
        """
        settings = self.source_locations.value.copy()
        settings["source_list_file"] = file_name
        self.source_locations.value = settings
        self.source_locations.savebuttonbar.bn_save.click()

    def _show_source_list_error(self, message):
        self._source_list_message.value = (
            f"<span style='color: red'>{html.escape(message)}</span>"
        )

    def _load_source_list(self, path):
        """
        Show the source list at ``path`` instead of one made from the catalogs.

        Parameters
        ----------

        path : str
            Path of the source list.

        Notes
        -----
        The list is view only: it is never written, no source list is made
        from the catalogs while it is loaded, and clicking does not exclude
        stars. The source location settings point at it so that photometry
        uses it. A file that is not a usable source list leaves the viewer as
        it was, with a message saying why.
        """
        name = Path(path).name
        try:
            sources = _read_source_list(path)
        except ValueError as err:
            message = f"{name} is not a usable source list: {err}."
            if self._loaded_source_list is not None:
                message += f" Still showing {Path(self._loaded_source_list_path).name}."
            self._show_source_list_error(message)
            self._reset_source_list_chooser()
            return

        if self._loaded_source_list is None:
            self._generated_source_list_file = self.photom_apertures_file
        self._loaded_source_list = sources
        self._loaded_source_list_path = str(path)
        self._loaded_stars = None
        self._set_source_list_file(str(path))
        self._clear_source_list_button.disabled = False
        # The variables come from the catalog query, which is not made.
        self._save_var_info.disabled = True
        self._status_message.value = ""
        self._source_list_message.value = html.escape(
            f"Showing {name}, view only. Clear it to make a new source list "
            "from the catalogs."
        )
        self._update_comparisons()

    def _clear_loaded_source_list(self, button=None):  # noqa: ARG002
        """
        Go back to making the source list from the catalogs.

        Notes
        -----
        The new source list is saved to the source list file in the settings,
        or, if that is still the loaded list, to the file in use before the
        list was loaded. If that is the loaded list itself, it stays loaded so
        that it is not overwritten.
        """
        loaded_path = self._loaded_source_list_path
        new_file = self.photom_apertures_file
        if new_file == loaded_path:
            new_file = self._generated_source_list_file
        if _same_file(new_file, loaded_path):
            self._show_source_list_error(
                f"{Path(loaded_path).name} is also the file new source lists "
                "are saved to, so clearing it would overwrite it. Change the "
                "source list file in the source location settings, or move "
                "the shared list, first."
            )
            return

        self._loaded_source_list = None
        self._loaded_source_list_path = None
        self._loaded_stars = None
        self._reset_source_list_chooser()
        self._set_source_list_file(new_file)
        self._clear_source_list_button.disabled = True
        self._save_var_info.disabled = False
        self._status_message.value = ""
        self._source_list_message.value = ""
        if self._image_chosen():
            # Take the loaded list's stars off the image, so that they are not
            # left looking like a new source list if the catalog query fails.
            self.remove_labels()
            self.iw.remove_catalog(catalog_label="*")
        self._update_comparisons()

    def _show_loaded_source_list(self):
        """
        Mark the stars of the loaded source list that are on the image, and
        list the ``star_id`` of those that are not.
        """
        sources = self._loaded_source_list
        on_image = coords_in_frame(sources["coord"], self.ccd.wcs, self.ccd.shape[::-1])
        on = sources[on_image]
        x, y = self.ccd.wcs.world_to_pixel(on["coord"])
        self._loaded_stars = Table(
            {
                "x": x,
                "y": y,
                "coord": on["coord"],
                "marker name": on["marker name"],
                "ra": on["coord"].ra.degree,
                "dec": on["coord"].dec.degree,
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

        children = [self._legend_spinner_box.children[0]]
        off_ids = [str(star_id) for star_id in sources["star_id"][~on_image]]
        if off_ids:
            stars_are = "star is" if len(off_ids) == 1 else "stars are"
            children.append(
                ipw.HTML(
                    value=f"<p>{len(off_ids)} {stars_are} off this image and "
                    f"not marked: star_id {html.escape(', '.join(off_ids))}</p>"
                )
            )
        self._legend_spinner_box.children = children
        self.help_stuff.selected_index = 1

    def _save_comparison_settings(self):
        previous = (self.comparison_catalog, self.bright_mag_limit, self.dim_mag_limit)
        self.comparison_catalog = self._comparison_catalog_choice.value
        self.bright_mag_limit = self._bright_mag_limit_input.value
        self.dim_mag_limit = self._dim_mag_limit_input.value
        image_chosen = self._image_chosen()
        if image_chosen:
            try:
                self._reload_comparisons()
            except Exception:
                # The stars on screen are still the ones for the previous
                # settings, so keep those settings; the save can be retried.
                self.comparison_catalog, self.bright_mag_limit, self.dim_mag_limit = (
                    previous
                )
                raise
        self._comparison_settings_changed()
        if image_chosen:
            try:
                self._save_source_location_file()
            except Exception as err:
                # The new stars are showing, so the new settings stay. Show the
                # error, since the traceback only goes to the log.
                self._comparison_save_bar.message.value = (
                    f"<span style='color: red'>{html.escape(str(err))}</span>"
                )
                raise

    def _revert_comparison_settings(self):
        self._comparison_catalog_choice.value = self.comparison_catalog
        self._bright_mag_limit_input.value = self.bright_mag_limit
        self._dim_mag_limit_input.value = self.dim_mag_limit

    def _comparison_settings_changed(self, change=None):  # noqa: ARG002
        bright = self._bright_mag_limit_input.value
        dim = self._dim_mag_limit_input.value
        unsaved = (
            self._comparison_catalog_choice.value,
            bright,
            dim,
        ) != (self.comparison_catalog, self.bright_mag_limit, self.dim_mag_limit)
        # Any other limits are usable; if no stars fall within them a message
        # says so after the reload.
        problem = "" if dim > bright else "Dim limit must be fainter than bright limit"
        self._comparison_save_bar.message.value = (
            f"<span style='color: red'>{problem}</span>" if problem else ""
        )
        self._comparison_save_bar.unsaved_changes = unsaved
        self._comparison_save_bar.bn_save.disabled = not unsaved or bool(problem)
        self._comparison_save_bar.bn_revert.disabled = not unsaved

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

    def _update_comparisons(self):
        # Reload the stars and save the source list, if an image has been chosen.
        if self._image_chosen():
            self._reload_comparisons()
            self._save_source_location_file()

    def _make_observers(self):
        for control in (
            self._comparison_catalog_choice,
            self._bright_mag_limit_input,
            self._dim_mag_limit_input,
        ):
            control.observe(self._comparison_settings_changed, names="value")
        self._show_labels_button.observe(self._show_label_button_handler, names="value")
        self._save_var_info.on_click(self._save_variables_to_file)
        self._file_chooser.file_chooser.observe(self._set_file, names="_value")
        self._choose_input_source_list.observe(self._set_target_file, names="_value")
        self._choose_source_list.observe(self._source_list_chosen, names="_value")
        self._clear_source_list_button.on_click(self._clear_loaded_source_list)

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
        header = ipw.HTML(value="""
        <h3>Click and drag to pan, use the scroll wheel to zoom.</h3>
        <h3>Click on a star to exclude it as target or comp.
        Click again to include it.</h3>
        """)

        legend = ipw.HTML(value="""
        <ul>
        <li>Green circles -- Gaia stars within 2.5 arcmin of target</li>
        <li>Red diamonds -- Comparison stars from the chosen catalog</li>
        <li>Blue squares -- VSX variables</li>
        <li>Red + -- Exclude as target or comp</li>
        </ul>
        """)

        iw = ImageWidget()

        # Messages for the user (e.g. "Click closer to a star") show up here.
        self._status_message = ipw.HTML()
        select_stars = wrap(iw, self._status_message)

        def click_handler(interaction, event_data, buffers):
            # A loaded source list is view only, so clicks do not exclude stars.
            if self._loaded_source_list is None:
                select_stars(interaction, event_data, buffers)
            elif event_data.get("event") == "click":
                self._status_message.value = (
                    "The loaded source list is view only; clear it to choose stars"
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
        self._legend_spinner_box.children = [legend]

        self.help_stuff = ipw.Accordion(children=[header, self._legend_spinner_box])
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

        # Add a file chooser for an existing source list to show, view only,
        # instead of making one from the catalogs.
        self._choose_source_list = FileChooser(filter_pattern="*.ecsv")
        self._choose_source_list.title = (
            "OPTIONAL: Use an existing source list (view only)"
        )
        self._clear_source_list_button = ipw.Button(
            description="Clear source list", disabled=True
        )
        self._source_list_message = ipw.HTML()
        load_source_list = ipw.VBox(
            children=[
                ipw.HBox(
                    children=[self._choose_source_list, self._clear_source_list_button]
                ),
                self._source_list_message,
            ]
        )

        comparison_settings = ipw.HBox(
            children=[
                self._comparison_catalog_choice,
                self._bright_mag_limit_input,
                self._dim_mag_limit_input,
                self._comparison_save_bar,
            ]
        )

        box.children = [
            self._file_chooser.file_chooser,
            self._choose_input_source_list,
            load_source_list,
            comparison_settings,
            inner_box,
            controls,
        ]

        return box, iw

    def _marked_stars(self):
        """
        The stars marked in the viewer that the user has not excluded.

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

        index, d2d, d3d = elim_table["coord"].match_to_catalog_sky(marked["coord"])
        marked.remove_rows(index)
        return marked

    def generate_table(self):
        """
        Generate the table of stars to use for the aperture file.

        A variable or comparison star within ``TARGET_MATCH_RADIUS`` of a
        target is the target, and is left out so that each star has one row.
        While a source list is loaded, the table holds the stars of that list
        that are on the image, with the ``star_id`` they have in the list.

        Returns
        -------
        comp_table : `astropy.table.Table`
            Table of stars to use for the aperture file.
        """
        if self._loaded_source_list is not None and self._loaded_stars is not None:
            return self._loaded_stars.copy()

        comp_table = self._marked_stars()
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
