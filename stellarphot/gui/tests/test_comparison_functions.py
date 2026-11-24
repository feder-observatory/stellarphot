import gzip
import os
import warnings
from pathlib import Path

import ipywidgets as ipw
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.table import MaskedColumn, Table
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from astropy.wcs.wcs import FITSFixedWarning
from astrowidgets.bqplot import ImageWidget

from stellarphot import SourceListData
from stellarphot.gui import comparison_functions as cf
from stellarphot.settings import PhotometryWorkingDirSettings

CCD_SHAPE = [2048, 3073]


# Every test in this module runs in a temporary directory because the widgets
# tested here save settings and source lists to the working directory.
pytestmark = pytest.mark.usefixtures("change_to_tmp_dir")


def make_ey_uma_image(with_object=False):
    wcs_file = get_pkg_data_filename("../../tests/data/sample_wcs_ey_uma.fits")
    with fits.open(wcs_file) as hdulist:
        with warnings.catch_warnings():
            # Ignore the warning about the WCS having a different number of
            # axes than the (non-existent) image.
            warnings.filterwarnings(
                "ignore",
                message="The WCS transformation has more",
                category=FITSFixedWarning,
            )
            wcs = WCS(hdulist[0].header)
    wcs.pixel_shape = list(reversed(CCD_SHAPE))
    ccd = CCDData(data=np.zeros(CCD_SHAPE), wcs=wcs, unit="adu")
    if with_object:
        ccd.header["object"] = "EY UMa"
    return ccd


def test_comparison_object_creation():
    # This test simply makes sure we can create the object
    comparison_widget = cf.ComparisonViewer()
    assert isinstance(comparison_widget.box, ipw.Box)


def test_fits_file_property():
    # Make sure that the fits file property has the value we expect and that the
    # property is read-only.
    comparison_widget = cf.ComparisonViewer()
    assert comparison_widget.fits_file == comparison_widget._file_chooser
    with pytest.raises(AttributeError):
        comparison_widget.fits_file = None


def test_coord_catalog_table_accepts_coord_column():
    # A table that already uses the astro-image-display-api column name
    # "coord" should pass through unchanged rather than raising a KeyError
    # when the helper tries to rename "coords".
    ccd = make_ey_uma_image()
    table = Table({"coord": ccd.wcs.pixel_to_world([500.0], [700.0])})

    out = cf._coord_catalog_table(table)

    assert "coord" in out.colnames
    assert len(out) == 1


def test_coord_catalog_table_missing_coordinate_column_raises():
    # A table with neither coordinate column name should produce a clear
    # error rather than a KeyError from rename_column.
    table = Table({"ra": [1.0], "dec": [2.0]})

    with pytest.raises(ValueError, match="coords"):
        cf._coord_catalog_table(table)


def test_wrap_toggles_elim_marker():
    # Clicking on a star should mark it for exclusion with an "elim" marker,
    # and clicking it again should remove the marker.
    ccd = make_ey_uma_image()
    iw = ImageWidget()
    iw.load_image(ccd)

    # Make a one-star catalog at a known pixel position
    star_coord = ccd.wcs.pixel_to_world(1500.0, 1000.0)
    catalog = Table({"coord": [star_coord]})
    iw.load_catalog(
        catalog,
        use_skycoord=True,
        catalog_label="APASS comparison",
        catalog_style={"shape": "cross", "color": "red", "size": 20},
    )

    status = ipw.HTML()
    callback = cf.wrap(iw, status)

    x, y = ccd.wcs.world_to_pixel(star_coord)
    click = {"event": "click", "domain": {"x": float(x), "y": float(y)}}

    # Events other than clicks should be ignored
    callback(
        iw._astro_im.interaction,
        {"event": "mousemove", "domain": {"x": float(x), "y": float(y)}},
        [],
    )
    assert "elim1" not in iw.catalog_labels

    # Click on the star to exclude it...
    callback(iw._astro_im.interaction, click, [])
    assert "elim1" in iw.catalog_labels

    # "plus" is one of the shapes that only the SVG Scatter mark astrowidgets
    # now uses can draw (the ScatterGL shader it used before could not).
    elim_style = iw.get_catalog_style(catalog_label="elim1")
    assert elim_style["shape"] == "plus"
    assert elim_style["color"] == "red"
    # Sizes are linear (bqplot draws size**2 pixels of area), so anything
    # much above 10 dwarfs the stars it is supposed to mark.
    assert elim_style["size"] == 12

    # ...and click again to include it.
    callback(iw._astro_im.interaction, click, [])
    assert "elim1" not in iw.catalog_labels

    # A click far from any star should display a message instead of
    # adding a marker.
    miss = {"event": "click", "domain": {"x": float(x) + 500, "y": float(y) + 500}}
    callback(iw._astro_im.interaction, miss, [])
    assert "Click closer to a star" in status.value
    assert not any(label.startswith("elim") for label in iw.catalog_labels)

    # A successful click afterwards should clear the stale message.
    callback(iw._astro_im.interaction, click, [])
    assert "elim" in "".join(iw.catalog_labels)
    assert status.value == ""


def test_combining_catalogs_with_conflicting_meta_does_not_warn():
    # Catalog tables keep the metadata of the table they were loaded from
    # (e.g. the Vizier catalog name), which differs between catalogs.
    # Combining them for the source table should not warn about the conflict.
    ccd = make_ey_uma_image()
    iw = ImageWidget()
    iw.load_image(ccd)

    style = {"shape": "circle", "color": "green", "size": 20}
    vsx = Table({"coord": [ccd.wcs.pixel_to_world(500.0, 700.0)]})
    vsx.meta["catalog_name"] = "B/vsx/vsx"
    iw.load_catalog(vsx, use_skycoord=True, catalog_label="VSX", catalog_style=style)

    apass = Table({"coord": [ccd.wcs.pixel_to_world(600.0, 800.0)]})
    apass.meta["catalog_name"] = "II/336/apass9"
    iw.load_catalog(
        apass, use_skycoord=True, catalog_label="APASS comparison", catalog_style=style
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        all_entries = cf._all_catalog_entries(iw)

    assert len(all_entries) == 2


def test_all_catalog_entries_minimal_columns_and_mixed_frames():
    # The combined catalog table should contain only the columns the viewer
    # manages, even if a catalog was loaded from a table with extra columns
    # (e.g. an input source list, which already has xcenter/ycenter columns
    # that collide with the aperture-file column names). Coordinates should
    # combine even if the catalogs' SkyCoord frames differ, which happens
    # because coordinates computed from a WCS are typically FK5 while
    # catalog coordinates are ICRS.
    ccd = make_ey_uma_image()
    iw = ImageWidget()
    iw.load_image(ccd)

    style = {"shape": "circle", "color": "green", "size": 20}

    targets = Table(
        {
            "coord": SkyCoord([135.5], [49.8], unit="deg"),
            "xcenter": [10.0],
            "ycenter": [20.0],
            "star_id": [1],
        }
    )
    iw.load_catalog(
        targets, use_skycoord=True, catalog_label="TESS Targets", catalog_style=style
    )

    apass = Table({"coord": SkyCoord([135.6], [49.9], unit="deg", frame="fk5")})
    iw.load_catalog(
        apass,
        use_skycoord=True,
        catalog_label="APASS comparison",
        catalog_style=style,
    )

    entries = cf._all_catalog_entries(iw)

    assert sorted(entries.colnames) == sorted(["x", "y", "coord", "marker name"])
    assert len(entries) == 2


def test_make_markers_with_masked_coordinates():
    # Catalogs fetched through astroquery come back as masked tables, so the
    # SkyCoord coords column built from them holds Masked arrays, which the
    # high-level WCS API refuses to transform. make_markers should cope.
    from astropy.utils.masked import Masked

    ccd = make_ey_uma_image()
    iw = ImageWidget()
    iw.load_image(ccd)

    plain = ccd.wcs.pixel_to_world([500.0, 600.0], [700.0, 800.0])
    masked_coords = SkyCoord(
        ra=Masked(plain.ra.deg, mask=[False, False]),
        dec=Masked(plain.dec.deg, mask=[False, False]),
        unit="deg",
    )
    apass = Table({"coords": masked_coords})

    cf.make_markers(iw, [], [], apass, name_or_coord=None)

    assert len(iw.get_catalog(catalog_label="APASS comparison")) == 2


def test_make_markers_shapes_and_colors():
    # Check that each catalog gets the marker style the legend says it has.
    ccd = make_ey_uma_image()
    iw = ImageWidget()
    iw.load_image(ccd)

    targets = Table({"coords": ccd.wcs.pixel_to_world([300.0], [400.0])})
    vsx = Table({"coords": ccd.wcs.pixel_to_world([500.0, 600.0], [700.0, 800.0])})
    apass = Table(
        {"coords": ccd.wcs.pixel_to_world([1000.0, 1100.0], [1200.0, 1300.0])}
    )

    cf.make_markers(iw, targets, vsx, apass, name_or_coord=None)

    # "diamond" is one of the shapes that only the SVG Scatter mark
    # astrowidgets now uses can draw (the ScatterGL shader it used before
    # could not).
    apass_style = iw.get_catalog_style(catalog_label="APASS comparison")
    assert apass_style["shape"] == "diamond"
    assert apass_style["color"] == "red"

    vsx_style = iw.get_catalog_style(catalog_label="VSX")
    assert vsx_style["shape"] == "square"
    assert vsx_style["color"] == "blue"

    # Sizes are linear (bqplot draws size**2 pixels of area), so anything
    # much above 10 dwarfs the stars it is supposed to mark.
    assert apass_style["size"] == 10
    assert vsx_style["size"] == 10

    # A target that is also a known variable has a circle and a square in the
    # same place. The circle is large enough to go around the square (whose
    # diagonal is sqrt(2) times its size) so that both can be seen.
    target_style = iw.get_catalog_style(catalog_label="TESS Targets")
    assert target_style["shape"] == "circle"
    assert target_style["color"] == "green"
    circle_diameter = 2 * target_style["size"] / np.sqrt(np.pi)
    assert circle_diameter > np.sqrt(2) * vsx_style["size"]

    # Guard against a regression to marks that cannot draw these shapes or
    # that ignore the style's size, as astrowidgets' bqplot backend once did.
    from bqplot import Scatter

    mark = iw._astro_im._scatter_marks["APASS comparison"]
    assert isinstance(mark, Scatter)
    assert mark.default_size == apass_style["size"] ** 2


def _viewer_with_target():
    # A comparison viewer with an image loaded and the target at pixel
    # (1500, 1000). Returns the viewer and the image.
    comparison_widget = cf.ComparisonViewer()
    ccd = make_ey_uma_image()
    comparison_widget.iw.load_image(ccd)
    comparison_widget.target_coord = ccd.wcs.pixel_to_world(1500.0, 1000.0)
    return comparison_widget, ccd


def test_show_circle_draws_bqplot_circle():
    # The target circle should be drawn as a bqplot Lines mark tracing the
    # circle, not as a catalog marker: astrowidgets 0.5.0 ignores the size in
    # catalog_style, and a catalog entry would also show up as a clickable
    # "star" in the click handler and in generate_table.
    from bqplot import Lines

    comparison_widget, _ = _viewer_with_target()

    comparison_widget.show_circle()

    # The circle must not be a catalog...
    assert comparison_widget._circle_name not in comparison_widget.iw.catalog_labels

    # ...it should be a Lines mark tracing a circle of the default radius,
    # 2.5 arcmin at 0.56 arcsec/pixel, centered on the target.
    mark_key = f"__circle__{comparison_widget._circle_name}"
    mark = comparison_widget.iw._astro_im._scatter_marks[mark_key]
    assert isinstance(mark, Lines)
    assert mark in comparison_widget.iw._astro_im._figure.marks

    expected_radius = ((2.5 * u.arcmin) / (0.56 * u.arcsec / u.pixel)).to(u.pixel).value
    radii = np.hypot(np.asarray(mark.x) - 1500.0, np.asarray(mark.y) - 1000.0)
    np.testing.assert_allclose(radii, expected_radius, atol=1)

    comparison_widget.remove_circle()
    assert mark_key not in comparison_widget.iw._astro_im._scatter_marks
    assert mark not in comparison_widget.iw._astro_im._figure.marks

    # Removing when no circle is shown should not raise.
    comparison_widget.remove_circle()


@pytest.mark.parametrize("has_target_list", [True, False])
def test_target_list_sets_target_coord_over_object_name(monkeypatch, has_target_list):
    # The first star in an imported target list is the target, and the object
    # name in the image header is not even looked up (e.g. the image is named
    # for a galaxy and the target is a supernova in it). Without a target
    # list the resolved name is the target.
    comparison_widget, ccd = _viewer_with_target()

    resolved_name = ccd.wcs.pixel_to_world(1000.0, 1000.0)
    first_target = ccd.wcs.pixel_to_world(1100.0, 900.0)

    def from_name(_name):
        if has_target_list:
            pytest.fail("The object name should not be looked up")
        return resolved_name

    monkeypatch.setattr(cf.SkyCoord, "from_name", from_name)

    if has_target_list:
        other_target = ccd.wcs.pixel_to_world(400.0, 300.0)
        comparison_widget.targets_from_file = Table(
            {"coords": SkyCoord([first_target, other_target])}
        )

    comparison_widget._set_object()

    expected = first_target if has_target_list else resolved_name
    assert comparison_widget.target_coord.separation(expected) < 1e-6 * u.arcsec


def test_target_coord_is_image_center_without_list_or_resolvable_name(monkeypatch):
    # With no target list and an object name that cannot be resolved the
    # target is the center of the image. The image is not square, so this
    # also checks that x and y are not swapped.
    comparison_widget, ccd = _viewer_with_target()
    comparison_widget.ccd = ccd

    def from_name(name):
        raise NameResolveError(f"Unable to find coordinates for name '{name}'")

    monkeypatch.setattr(cf.SkyCoord, "from_name", from_name)

    comparison_widget._set_object()

    n_y, n_x = CCD_SHAPE
    center = ccd.wcs.pixel_to_world(n_x / 2, n_y / 2)
    assert comparison_widget.target_coord.separation(center) < 1e-6 * u.arcsec


@pytest.mark.parametrize("n_targets", [0, 2])
def test_generate_table_groups_targets_then_vsx_then_comparisons(n_targets):
    # The source list should have the imported targets first, then the VSX
    # stars, then the comparison stars, with or without imported targets.
    comparison_widget, ccd = _viewer_with_target()

    # Each group has a star nearer the target coordinate than a star of the
    # group that should precede it, so that sorting by distance alone mixes
    # the groups. No star is at the target coordinate itself.
    targets = Table({"coords": ccd.wcs.pixel_to_world([900.0, 1400.0], [900.0, 900.0])})
    targets = targets[:n_targets]
    vsx = Table({"coords": ccd.wcs.pixel_to_world([800.0, 1450.0], [800.0, 950.0])})
    apass = Table({"coords": ccd.wcs.pixel_to_world([1520.0, 700.0], [1020.0, 700.0])})

    cf.make_markers(comparison_widget.iw, targets, vsx, apass, name_or_coord=None)

    table = comparison_widget.generate_table()

    expected_groups = (
        ["TESS Targets"] * n_targets + ["VSX"] * 2 + ["APASS comparison"] * 2
    )
    assert list(table["marker name"]) == expected_groups

    # The imported targets stay in the order they were given.
    np.testing.assert_allclose(
        table["x"][:n_targets], [900.0, 1400.0][:n_targets], atol=1e-3
    )


def _viewer_with_targets_also_in_catalogs():
    # A comparison viewer showing two targets, three variables and three
    # comparison stars. Each target has a variable 0.3 arcsec away (the size
    # of the offset between a Gaia position and a VSX one) and a comparison
    # star 1 arcsec away, all of which are the target itself. The first target
    # also has a variable 3 arcsec away, which is a different star.
    comparison_widget, ccd = _viewer_with_target()
    target_coords = ccd.wcs.pixel_to_world([900.0, 1400.0], [900.0, 900.0])

    def offset_from_targets(arcsec):
        return target_coords.spherical_offsets_by(arcsec * u.arcsec, 0 * u.arcsec)

    targets = Table({"coords": target_coords})
    vsx = Table(
        {
            "coords": SkyCoord([*offset_from_targets(0.3), offset_from_targets(3)[0]]),
            "Name": ["first target", "second target", "neighbor"],
        }
    )
    apass = Table(
        {
            "coords": SkyCoord(
                [*offset_from_targets(-1), ccd.wcs.pixel_to_world(700.0, 700.0)]
            )
        }
    )
    comparison_widget.vsx = vsx
    cf.make_markers(comparison_widget.iw, targets, vsx, apass, name_or_coord=None)
    return comparison_widget


def test_generate_table_has_one_row_for_a_target_also_in_a_catalog():
    # A target that is also a known variable, or also in the comparison
    # catalog, is one star, so it should be in the source list once, as the
    # target.
    comparison_widget = _viewer_with_targets_also_in_catalogs()

    table = comparison_widget.generate_table()

    assert list(table["marker name"]) == [
        "TESS Targets",
        "TESS Targets",
        "VSX",
        "APASS comparison",
    ]
    assert list(table["star_id"]) == [1, 2, 3, 4]


def test_variables_gives_a_target_that_is_a_variable_the_target_star_id():
    # The variable table keeps a variable that is also a target, with the
    # star_id the target has in the source list.
    comparison_widget = _viewer_with_targets_also_in_catalogs()

    variables = comparison_widget.variables

    star_ids = dict(zip(variables["Name"], variables["star_id"], strict=True))
    assert star_ids == {"first target": 1, "second target": 2, "neighbor": 3}


def _viewer_with_variables_near_target(arcsec_from_target):
    # A comparison viewer showing one target, a variable at each of the given
    # distances from it, named for the distance, and a comparison star.
    # Returns the viewer and the pixel position of the target.
    comparison_widget, ccd = _viewer_with_target()
    target_xy = (900.0, 900.0)
    target_coord = ccd.wcs.pixel_to_world(*target_xy)
    targets = Table({"coords": SkyCoord([target_coord])})
    vsx = Table(
        {
            "coords": SkyCoord(
                [
                    target_coord.spherical_offsets_by(d * u.arcsec, 0 * u.arcsec)
                    for d in arcsec_from_target
                ]
            ),
            "Name": arcsec_from_target,
        }
    )
    apass = Table({"coords": SkyCoord([ccd.wcs.pixel_to_world(700.0, 700.0)])})
    comparison_widget.vsx = vsx
    cf.make_markers(comparison_widget.iw, targets, vsx, apass, name_or_coord=None)
    return comparison_widget, target_xy


def test_variable_that_is_a_target_gets_target_star_id_not_a_nearer_variables():
    # The variable 1.9 arcsec from the target is the target. It is nearer to
    # the variable at 2.1 arcsec, a different star, than it is to the target.
    comparison_widget, _ = _viewer_with_variables_near_target([2.1, 1.9])

    variables = comparison_widget.variables

    # The rows are in star_id order, whatever order the variables were in.
    assert list(variables["Name"]) == [1.9, 2.1]
    assert list(variables["star_id"]) == [1, 2]


def test_clicking_a_target_excludes_the_variable_that_is_the_target():
    # Excluding a target excludes the star, so the variable that is the same
    # star must not come back as a row of its own.
    comparison_widget, (x, y) = _viewer_with_variables_near_target([1.5, 3])
    iw = comparison_widget.iw

    click = {"event": "click", "domain": {"x": x, "y": y}}
    cf.wrap(iw, ipw.HTML())(iw._astro_im.interaction, click, [])

    table = comparison_widget.generate_table()
    assert list(table["marker name"]) == ["VSX", "APASS comparison"]
    assert list(comparison_widget.variables["Name"]) == [3]


def test_variables_is_empty_when_every_variable_is_excluded():
    comparison_widget, ccd = _viewer_with_target()
    vsx = Table({"coords": ccd.wcs.pixel_to_world([800.0], [800.0])})
    apass = Table({"coords": ccd.wcs.pixel_to_world([700.0], [700.0])})
    comparison_widget.vsx = vsx
    iw = comparison_widget.iw
    cf.make_markers(iw, [], vsx, apass, name_or_coord=None)
    iw.load_catalog(
        iw.get_catalog(catalog_label="VSX"), use_skycoord=True, catalog_label="elim1"
    )

    variables = comparison_widget.variables

    assert len(variables) == 0
    assert "star_id" in variables.colnames


@pytest.mark.parametrize("source_file_name", [None, "sources.ecsv"])
@pytest.mark.parametrize("has_object", [True, False])
@pytest.mark.remote_data
def test_comparison_properties(tmp_path, has_object, source_file_name):
    # Test that we can load a file...
    ccd = make_ey_uma_image(with_object=has_object)

    file_name = "test.fits"
    ccd.write(tmp_path / file_name, overwrite=True)

    with warnings.catch_warnings():
        # Ignore the warning about the WCS having non-standard keywords (the SIP
        # distortion parameters).
        warnings.filterwarnings(
            "ignore",
            message="Some non-standard WCS keywords were excluded",
            category=AstropyWarning,
        )
        comparison_widget = cf.ComparisonViewer(
            directory=tmp_path,
            file=str(file_name),
            photom_apertures_file=source_file_name,
        )
        # Nothing is loaded until the first question is answered.
        comparison_widget._mode.value = "generate"
    _click_save(comparison_widget)

    # Check that we have some variables in this field of view, which contains the
    # variable star EY UMa
    assert len(comparison_widget.variables) > 0

    # Check that we have APASS stars too
    table = comparison_widget.generate_table()
    assert "APASS comparison" in table["marker name"]

    # The "separation" and "sort" columns are only used to order the table and
    # should not be present in the table returned by generate_table (issue #34).
    assert "separation" not in table.colnames
    assert "sort" not in table.colnames

    # Check that showing labels creates one label mark entry per star.
    comparison_widget.show_labels()

    # There should be a label for each star, so check that the total number of
    # label positions across the bqplot Label marks matches the number of stars.
    table = comparison_widget.generate_table()

    n_labels = sum(len(mark.x) for mark in comparison_widget._label_marks.values())
    assert n_labels == len(table)

    # Make sure the aperture file has been written
    assert os.path.exists(comparison_widget.source_locations.value["source_list_file"])
    # Make sure the aperture file name matches the one we supplied, if
    # we supplied one.
    if source_file_name is not None:
        assert (
            comparison_widget.source_locations.value["source_list_file"]
            == source_file_name
        )

    # Check that a partial photometry settings file exists and that source locations
    # settings have saved correctly.
    partial_settings = PhotometryWorkingDirSettings().load()
    assert (
        partial_settings.source_location_settings.model_dump()
        == comparison_widget.source_locations.value
    )


def _write_ey_uma_file(tmp_path):
    # Write the EY UMa test image to tmp_path. Returns the path of the image
    # and the image.
    ccd = make_ey_uma_image()
    path = tmp_path / "test.fits"
    ccd.write(path, overwrite=True)
    return path, ccd


def _choose_image(comparison_widget, path):
    # Choose an image in the viewer's file chooser, which loads it and the
    # catalogs.
    with warnings.catch_warnings():
        # Ignore the warning about the WCS having non-standard keywords (the SIP
        # distortion parameters).
        warnings.filterwarnings(
            "ignore",
            message="Some non-standard WCS keywords were excluded",
            category=AstropyWarning,
        )
        comparison_widget._file_chooser.file_chooser.value = path


def _viewer_in_mode(mode, **kwargs):
    # A comparison viewer with the first question, use an existing source
    # list or generate one, answered.
    comparison_widget = cf.ComparisonViewer(**kwargs)
    comparison_widget._mode.value = mode
    return comparison_widget


def _click_save(comparison_widget):
    comparison_widget._comparison_save_bar.bn_save.click()


def _regenerate(comparison_widget):
    comparison_widget._regenerate_button.click()


def _shown(widget):
    return widget.layout.display != "none"


def _choose_second_image(comparison_widget, tmp_path):
    # Choose a copy of the image the viewer is showing, as a new image.
    second_image = tmp_path / "second.fits"
    second_image.write_bytes(Path(comparison_widget.fits_file.path).read_bytes())
    _choose_image(comparison_widget, second_image)


def test_spinner_cleared_and_error_shown_when_catalog_load_fails(tmp_path, monkeypatch):
    # If fetching the VSX/APASS catalogs fails, the spinner should not keep
    # spinning forever -- the legend should come back and an error message
    # should be shown to the user.
    path, _ = _write_ey_uma_file(tmp_path)

    comparison_widget = _viewer_in_mode("generate")
    legend = comparison_widget._legend_spinner_box.children[0]

    def failing_set_up(*_args, **_kwargs):
        # The message contains HTML-special characters, as real network errors
        # often do (server responses, URLs with query strings), to check that
        # the on-screen message escapes them instead of injecting markup.
        raise RuntimeError("catalog fetch failed: server said <b>no</b> & quit")

    monkeypatch.setattr(cf, "set_up", failing_set_up)

    # The error should still propagate so it shows up in the log...
    with pytest.raises(RuntimeError, match="catalog fetch failed"):
        _choose_image(comparison_widget, path)

    # ...but the spinner should be gone...
    children = comparison_widget._legend_spinner_box.children
    assert not any(isinstance(child, cf.Spinner) for child in children)

    # ...the legend should be back...
    assert legend in children

    # ...and an error message should be visible.
    error_widgets = [
        child
        for child in children
        if isinstance(child, ipw.HTML) and "catalog fetch failed" in child.value
    ]
    assert len(error_widgets) == 1

    # The exception text must be HTML-escaped, not injected as markup.
    assert "server said &lt;b&gt;no&lt;/b&gt; &amp; quit" in error_widgets[0].value
    assert "<b>no</b>" not in error_widgets[0].value


@pytest.fixture
def image_with_fake_catalogs(tmp_path, mocker):
    # The path of the EY UMa image, with the catalog queries replaced by
    # mocks that return no variables and three comparison candidates, of SR
    # magnitude 9, 12 and 14.
    path, ccd = _write_ey_uma_file(tmp_path)
    candidates = Table(
        {
            "coords": ccd.wcs.pixel_to_world([1000.0, 1200.0, 1400.0], [1000.0] * 3),
            "mag": [9.0, 12.0, 14.0],
            "passband": ["SR"] * 3,
        }
    )
    mocker.patch.object(cf, "set_up", return_value=[])
    mocker.patch.object(
        cf,
        "crossmatch_APASS2VSX",
        side_effect=lambda *_args, **_kwargs: (candidates.copy(), [], []),
    )
    mocker.patch.object(
        cf.SkyCoord, "from_name", side_effect=NameResolveError("no such object")
    )
    return path


@pytest.fixture
def viewer_with_fake_catalogs(image_with_fake_catalogs):
    # A viewer generating a source list for the image with the fake catalogs.
    comparison_widget = _viewer_in_mode("generate")
    _choose_image(comparison_widget, image_with_fake_catalogs)
    return comparison_widget


def test_regenerate_redraws_with_new_settings_and_writes_nothing(
    viewer_with_fake_catalogs,
):
    # Changing the catalog or magnitude limits holds the save until the
    # Regenerate button has re-queried with the new settings and redrawn the
    # comparison markers. Regenerating writes nothing; saving is separate.
    comparison_widget = viewer_with_fake_catalogs
    save_bar = comparison_widget._comparison_save_bar
    regenerate = comparison_widget._regenerate_button
    _click_save(comparison_widget)
    saved = Path("source_locations.ecsv").read_bytes()

    assert cf.crossmatch_APASS2VSX.call_count == 1
    assert cf.crossmatch_APASS2VSX.call_args.kwargs == {
        "catalog": "APASS DR9",
        "magnitude_limit": 15,
    }
    comps = comparison_widget.iw.get_catalog(catalog_label="APASS comparison")
    assert len(comps) == 3
    assert not save_bar.unsaved_changes
    assert save_bar.bn_save.disabled
    assert save_bar.bn_revert.disabled
    assert regenerate.disabled

    comparison_widget._comparison_catalog_choice.value = "ATLAS refcat2"
    # Changing the catalog alone is enough: the stars shown are no longer the
    # ones for the settings, so they can not be saved.
    assert save_bar.unsaved_changes
    assert save_bar.bn_save.disabled
    assert "Click Regenerate" in save_bar.message.value
    assert not regenerate.disabled
    assert not save_bar.bn_revert.disabled

    comparison_widget._dim_mag_limit_input.value = 13

    # Nothing is re-queried until Regenerate is clicked.
    assert cf.crossmatch_APASS2VSX.call_count == 1

    _regenerate(comparison_widget)

    assert cf.crossmatch_APASS2VSX.call_args.kwargs == {
        "catalog": "ATLAS refcat2",
        "magnitude_limit": 13,
    }
    assert cf.set_up.call_args.kwargs == {"magnitude_limit": 13}
    # The marker label stays "APASS comparison" whatever the catalog, since
    # downstream code matches on it.
    comps = comparison_widget.iw.get_catalog(catalog_label="APASS comparison")
    assert len(comps) == 2
    assert Path("source_locations.ecsv").read_bytes() == saved
    assert save_bar.unsaved_changes
    assert not save_bar.bn_save.disabled
    assert "not been saved" in save_bar.message.value
    assert regenerate.disabled
    assert save_bar.bn_revert.disabled

    _click_save(comparison_widget)

    assert len(SourceListData.read("source_locations.ecsv")) == 2
    assert not save_bar.unsaved_changes
    assert save_bar.bn_save.disabled


def test_save_bar_revert_restores_settings(viewer_with_fake_catalogs):
    comparison_widget = viewer_with_fake_catalogs
    save_bar = comparison_widget._comparison_save_bar
    _click_save(comparison_widget)

    comparison_widget._comparison_catalog_choice.value = "ATLAS refcat2"
    comparison_widget._bright_mag_limit_input.value = 10
    comparison_widget._dim_mag_limit_input.value = 13

    save_bar.bn_revert.click()

    assert comparison_widget._comparison_catalog_choice.value == "APASS DR9"
    assert comparison_widget._bright_mag_limit_input.value == 8
    assert comparison_widget._dim_mag_limit_input.value == 15
    assert not save_bar.unsaved_changes
    assert save_bar.bn_save.disabled
    assert save_bar.bn_revert.disabled
    assert cf.crossmatch_APASS2VSX.call_count == 1


def test_constructor_catalog_is_used_for_query(image_with_fake_catalogs):
    comparison_widget = _viewer_in_mode("generate", comparison_catalog="ATLAS refcat2")
    assert not comparison_widget._comparison_save_bar.unsaved_changes

    _choose_image(comparison_widget, image_with_fake_catalogs)

    assert cf.crossmatch_APASS2VSX.call_args.kwargs["catalog"] == "ATLAS refcat2"


def test_settings_changed_before_image_is_chosen_are_used_for_the_query(
    image_with_fake_catalogs,
):
    # With no image there is nothing to regenerate; the query made when the
    # image is chosen uses the settings on screen.
    comparison_widget = _viewer_in_mode("generate")

    comparison_widget._dim_mag_limit_input.value = 13

    assert comparison_widget._regenerate_button.disabled

    _choose_image(comparison_widget, image_with_fake_catalogs)

    assert cf.crossmatch_APASS2VSX.call_args.kwargs["magnitude_limit"] == 13
    assert comparison_widget.dim_mag_limit == 13
    comps = comparison_widget.iw.get_catalog(catalog_label="APASS comparison")
    assert len(comps) == 2
    assert comparison_widget._regenerate_button.disabled


@pytest.mark.parametrize("bright, dim", [(16, 15), (12, 12)])
def test_dim_limit_not_fainter_than_bright_blocks_regenerate(
    viewer_with_fake_catalogs, bright, dim
):
    comparison_widget = viewer_with_fake_catalogs
    save_bar = comparison_widget._comparison_save_bar
    regenerate = comparison_widget._regenerate_button
    _click_save(comparison_widget)

    comparison_widget._bright_mag_limit_input.value = bright
    comparison_widget._dim_mag_limit_input.value = dim

    assert regenerate.disabled
    assert save_bar.bn_save.disabled
    assert not save_bar.bn_revert.disabled
    assert "fainter than bright limit" in save_bar.message.value

    comparison_widget._bright_mag_limit_input.value = 10
    comparison_widget._dim_mag_limit_input.value = 13

    assert not regenerate.disabled
    assert "fainter than bright limit" not in save_bar.message.value


def test_bright_limits_can_be_used(image_with_fake_catalogs):
    # Limits suited to a bright target are as usable on screen as they are in
    # the constructor.
    comparison_widget = _viewer_in_mode(
        "generate", bright_mag_limit=6, dim_mag_limit=8.5
    )
    _choose_image(comparison_widget, image_with_fake_catalogs)
    save_bar = comparison_widget._comparison_save_bar
    assert "fainter than bright limit" not in save_bar.message.value

    comparison_widget._dim_mag_limit_input.value = 8.6

    assert not comparison_widget._regenerate_button.disabled
    assert "fainter than bright limit" not in save_bar.message.value


def test_labels_match_comparisons_after_regenerate(viewer_with_fake_catalogs):
    # Labels shown before a reload are replaced by labels for the reloaded
    # stars, not left on the image.
    comparison_widget = viewer_with_fake_catalogs
    comparison_widget._show_labels_button.value = True
    label_key = "__label__APASS comparison"
    marks = comparison_widget.iw._astro_im._scatter_marks
    assert list(marks[label_key].text) == ["C1", "C2", "C3"]

    comparison_widget._dim_mag_limit_input.value = 13
    _regenerate(comparison_widget)

    assert list(marks[label_key].text) == ["C1", "C2"]

    # With no comparison stars left there are no comparison labels either.
    comparison_widget._bright_mag_limit_input.value = 10
    comparison_widget._dim_mag_limit_input.value = 11
    _regenerate(comparison_widget)

    assert label_key not in marks


def _legend_messages(comparison_widget):
    # The text of the messages shown below the legend.
    return [w.value for w in comparison_widget._legend_spinner_box.children[1:]]


def test_message_shown_when_no_comparison_stars_found(viewer_with_fake_catalogs):
    # Limits that match no candidates are not an error; a message says that no
    # comparison stars were found, and goes away when some are found.
    comparison_widget = viewer_with_fake_catalogs
    save_bar = comparison_widget._comparison_save_bar
    legend = comparison_widget._legend_spinner_box.children[0]

    comparison_widget._bright_mag_limit_input.value = 10
    comparison_widget._dim_mag_limit_input.value = 11
    _regenerate(comparison_widget)

    assert comparison_widget.dim_mag_limit == 11
    messages = _legend_messages(comparison_widget)
    assert len(messages) == 1
    assert "No comparison stars" in messages[0]
    # There are no variables or targets either, so there is nothing to put in
    # a source list. That is said once, by the save bar.
    assert "no source list to save" not in messages[0]
    assert "no source list to save" in save_bar.message.value
    assert not save_bar.unsaved_changes

    comparison_widget._dim_mag_limit_input.value = 13
    _regenerate(comparison_widget)

    assert comparison_widget._legend_spinner_box.children == (legend,)


def test_failed_regenerate_keeps_old_settings_and_markers(viewer_with_fake_catalogs):
    # If the catalog query fails, the settings in use and the stars shown are
    # unchanged and Regenerate stays enabled so it can be tried again.
    # Labels that were showing are still showing.
    comparison_widget = viewer_with_fake_catalogs
    comparison_widget._show_labels_button.value = True
    comparison_widget._comparison_catalog_choice.value = "ATLAS refcat2"
    comparison_widget._bright_mag_limit_input.value = 10
    comparison_widget._dim_mag_limit_input.value = 13
    cf.set_up.side_effect = RuntimeError("catalog is down")

    _regenerate(comparison_widget)

    assert comparison_widget.comparison_catalog == "APASS DR9"
    assert comparison_widget.bright_mag_limit == 8
    assert comparison_widget.dim_mag_limit == 15
    assert not comparison_widget._regenerate_button.disabled
    assert comparison_widget._comparison_save_bar.bn_save.disabled
    comps = comparison_widget.iw.get_catalog(catalog_label="APASS comparison")
    assert len(comps) == 3
    marks = comparison_widget.iw._astro_im._scatter_marks
    assert list(marks["__label__APASS comparison"].text) == ["C1", "C2", "C3"]


def test_settings_kept_when_only_the_source_list_write_fails(image_with_fake_catalogs):
    # The stars were regenerated with the new settings, so those are the
    # settings in use even though the source list could not be written. The
    # error is shown, since an exception in a widget callback only reaches the
    # log, and the source list is still waiting to be saved.
    comparison_widget = _viewer_in_mode("generate", overwrite_outputs=False)
    save_bar = comparison_widget._comparison_save_bar
    _choose_image(comparison_widget, image_with_fake_catalogs)
    # The first save writes the source list, so the next write fails.
    _click_save(comparison_widget)

    comparison_widget._dim_mag_limit_input.value = 13
    _regenerate(comparison_widget)
    _click_save(comparison_widget)

    comps = comparison_widget.iw.get_catalog(catalog_label="APASS comparison")
    assert len(comps) == 2
    assert comparison_widget.dim_mag_limit == 13
    assert save_bar.bn_revert.disabled
    assert "can not be overwritten" in save_bar.message.value
    assert not _shown(comparison_widget._confirm_overwrite)
    assert save_bar.unsaved_changes
    assert not save_bar.bn_save.disabled


def test_no_markers_does_not_raise(tmp_path):
    # With no stars marked there is nothing to list, label or save.
    comparison_widget, _ = _viewer_with_target()

    assert len(comparison_widget.generate_table()) == 0
    comparison_widget.show_labels()
    source_file = tmp_path / "sources.ecsv"
    comparison_widget._save_aperture_to_file(filename=source_file)
    assert not source_file.exists()


# Stars in the shared source list used in the tests of loading a source list:
# star_id, marker name and pixel position on the EY UMa image. The ids are not
# 1..N, to check that the viewer keeps them. The last two are off the image.
SHARED_STARS = [
    (3, "TESS Targets", (900.0, 900.0)),
    (8, "VSX", (800.0, 800.0)),
    (12, "APASS comparison", (1500.0, 1500.0)),
    (15, "APASS comparison", (1700.0, 1200.0)),
    (21, "APASS comparison", (-400.0, 1000.0)),
    (22, "APASS comparison", (3500.0, 1000.0)),
]


def _source_list_table(stars=SHARED_STARS, ccd=None):
    # A table of the given stars, given as in SHARED_STARS on the image ccd
    # (default the EY UMa image), with the columns of a source list made by
    # the comparison viewer on another night.
    if ccd is None:
        ccd = make_ey_uma_image()
    star_ids, marker_names, pixels = zip(*stars, strict=True)
    x, y = (np.array(p) for p in zip(*pixels, strict=True))
    # Source lists are ICRS; the image WCS is not.
    coords = ccd.wcs.pixel_to_world(x, y).icrs
    return Table(
        {
            "star_id": star_ids,
            "marker name": marker_names,
            "ra": coords.ra.deg * u.deg,
            "dec": coords.dec.deg * u.deg,
            # Pixel positions on the other night's image, which are not those
            # on this image.
            "xcenter": (x + 50) * u.pixel,
            "ycenter": (y + 50) * u.pixel,
        }
    )


def _write_list_file(path, stars=SHARED_STARS):
    SourceListData(input_data=_source_list_table(stars)).write(path, overwrite=True)


@pytest.fixture
def shared_source_list(tmp_path):
    # The shared source list, in a directory of its own. Returns its path.
    path = tmp_path / "shared" / "shared-sources.ecsv"
    path.parent.mkdir()
    _write_list_file(path)
    return path


def _write_other_list(path):
    # A source list different from the shared one and from the generated one.
    _write_list_file(path, SHARED_STARS[:2])


@pytest.fixture
def existing_destination(change_to_tmp_dir):  # noqa: ARG001
    # The source list file named in the settings, already there before the
    # viewer is made. Returns its path.
    destination = Path("source_locations.ecsv")
    _write_other_list(destination)
    return destination


def _click(comparison_widget, x, y):
    # Send a click at pixel (x, y) through the viewer's registered callback,
    # as the bqplot front end does.
    interaction = comparison_widget.iw._astro_im.interaction
    interaction._handle_custom_msg({"event": "click", "domain": {"x": x, "y": y}}, [])


def _assert_shows_shared_list(comparison_widget):
    # The viewer marks the stars of the shared source list that are on the
    # image, with the star ids they have in the list, and lists those that
    # are off the image.
    table = comparison_widget.generate_table()
    on_image = SHARED_STARS[:4]
    assert list(table["star_id"]) == [star[0] for star in on_image]
    assert list(table["marker name"]) == [star[1] for star in on_image]
    np.testing.assert_allclose(table["x"], [star[2][0] for star in on_image], atol=1e-3)

    comparison_widget.show_labels()
    marks = comparison_widget.iw._astro_im._scatter_marks
    assert list(marks["__label__TESS Targets"].text) == ["T3"]
    assert list(marks["__label__VSX"].text) == ["V8"]
    assert list(marks["__label__APASS comparison"].text) == ["C12", "C15"]

    messages = _legend_messages(comparison_widget)
    assert len(messages) == 1
    assert "2 stars" in messages[0]
    assert "21, 22" in messages[0]


def _saved_source_list_file():
    return (
        PhotometryWorkingDirSettings().load().source_location_settings.source_list_file
    )


def _existing_list_viewer_for(image, path):
    # A viewer showing the source list at path on the image.
    comparison_widget = _viewer_in_mode("existing")
    comparison_widget._choose_source_list.value = path
    _choose_image(comparison_widget, image)
    return comparison_widget


@pytest.fixture
def existing_list_viewer(image_with_fake_catalogs, shared_source_list):
    # A viewer showing the shared source list on the image with the fake
    # catalogs.
    return _existing_list_viewer_for(image_with_fake_catalogs, shared_source_list)


def test_mode_starts_unanswered_and_rest_of_widget_waits_for_it(
    image_with_fake_catalogs,
):
    # The first question, use an existing source list or generate one, is
    # unanswered even when there is a source list file already, and until it
    # is answered nothing but the question is shown and no catalog is queried.
    _write_list_file(Path("source_locations.ecsv"))
    comparison_widget = cf.ComparisonViewer()
    existing = comparison_widget._existing_list_controls
    generate = comparison_widget._generate_list_controls
    rest = comparison_widget._rest_of_viewer

    assert comparison_widget._mode.value is None
    # The default button width cuts both labels short.
    assert comparison_widget._mode.style.button_width == "auto"
    assert comparison_widget.box.children[0] is comparison_widget._mode
    assert not any(_shown(w) for w in (existing, generate, rest))

    _choose_image(comparison_widget, image_with_fake_catalogs)
    assert cf.crossmatch_APASS2VSX.call_count == 0

    comparison_widget._mode.value = "existing"
    assert _shown(existing) and _shown(rest) and not _shown(generate)
    assert comparison_widget._comparison_save_bar in rest.children
    assert comparison_widget._save_var_info.disabled
    assert cf.crossmatch_APASS2VSX.call_count == 0

    comparison_widget._mode.value = "generate"
    assert _shown(generate) and _shown(rest) and not _shown(existing)
    assert not comparison_widget._save_var_info.disabled
    assert cf.crossmatch_APASS2VSX.call_count == 1


def test_choosing_an_image_does_not_write_the_source_list(viewer_with_fake_catalogs):
    # The source list is written only when the save bar is clicked, and saving
    # it does not query the catalogs again.
    comparison_widget = viewer_with_fake_catalogs
    save_bar = comparison_widget._comparison_save_bar

    assert not os.path.exists("source_locations.ecsv")
    assert save_bar.unsaved_changes
    assert not save_bar.bn_save.disabled
    assert "not been saved" in save_bar.message.value

    _click_save(comparison_widget)

    assert len(SourceListData.read("source_locations.ecsv")) == 3
    assert not save_bar.unsaved_changes
    assert save_bar.bn_save.disabled
    assert "not been saved" not in save_bar.message.value
    assert cf.crossmatch_APASS2VSX.call_count == 1


def test_switching_mode_shows_the_list_for_the_mode_and_writes_nothing(
    viewer_with_fake_catalogs, shared_source_list
):
    comparison_widget = viewer_with_fake_catalogs
    iw = comparison_widget.iw

    comparison_widget._mode.value = "existing"
    # No list is chosen yet, so no stars are shown.
    assert len(iw.catalog_labels) == 0
    assert not comparison_widget._comparison_save_bar.unsaved_changes

    comparison_widget._choose_source_list.value = shared_source_list
    _assert_shows_shared_list(comparison_widget)

    # The stars already found for this image are shown again without another
    # query.
    comparison_widget._mode.value = "generate"
    table = comparison_widget.generate_table()
    assert list(table["marker name"]) == ["APASS comparison"] * 3
    assert cf.crossmatch_APASS2VSX.call_count == 1

    comparison_widget._mode.value = "existing"
    _assert_shows_shared_list(comparison_widget)
    assert cf.crossmatch_APASS2VSX.call_count == 1
    assert not os.path.exists("source_locations.ecsv")


def test_saved_generated_list_stays_saved_after_mode_round_trip(
    viewer_with_fake_catalogs,
):
    comparison_widget = viewer_with_fake_catalogs
    _click_save(comparison_widget)

    comparison_widget._mode.value = "existing"
    comparison_widget._mode.value = "generate"

    assert len(comparison_widget.generate_table()) == 3
    assert cf.crossmatch_APASS2VSX.call_count == 1
    assert not comparison_widget._comparison_save_bar.unsaved_changes


def test_new_image_after_a_generated_save_needs_saving(
    viewer_with_fake_catalogs, tmp_path
):
    # The list saved was the one for the previous image.
    comparison_widget = viewer_with_fake_catalogs
    save_bar = comparison_widget._comparison_save_bar
    _click_save(comparison_widget)

    _choose_second_image(comparison_widget, tmp_path)

    assert cf.crossmatch_APASS2VSX.call_count == 2
    assert save_bar.unsaved_changes
    assert not save_bar.bn_save.disabled
    assert "not been saved" in save_bar.message.value


@pytest.mark.parametrize("image_first", [True, False])
def test_existing_source_list_is_shown_and_not_written(
    image_with_fake_catalogs, shared_source_list, image_first
):
    # An existing source list is shown with its own star ids, without querying
    # the catalogs. Nothing is written and the settings are unchanged until
    # the save bar is clicked.
    comparison_widget = _viewer_in_mode("existing")
    original_contents = shared_source_list.read_bytes()

    if image_first:
        _choose_image(comparison_widget, image_with_fake_catalogs)
    comparison_widget._choose_source_list.value = shared_source_list
    if not image_first:
        _choose_image(comparison_widget, image_with_fake_catalogs)

    _assert_shows_shared_list(comparison_widget)
    assert cf.crossmatch_APASS2VSX.call_count == 0
    assert cf.set_up.call_count == 0
    assert shared_source_list.read_bytes() == original_contents
    assert not os.path.exists("source_locations.ecsv")
    assert comparison_widget.photom_apertures_file == "source_locations.ecsv"
    assert _saved_source_list_file() == "source_locations.ecsv"
    save_bar = comparison_widget._comparison_save_bar
    assert save_bar.unsaved_changes
    # The messages name the button to click and the file it writes.
    assert "save button" in comparison_widget._source_list_message.value
    assert "not been copied to source_locations.ecsv" in save_bar.message.value
    # The first target in the list is the target.
    target = comparison_widget.generate_table()["coord"][0]
    assert comparison_widget.target_coord.separation(target) < 1e-3 * u.arcsec


@pytest.mark.parametrize("image_chosen", [True, False])
def test_saving_copies_existing_source_list(
    image_with_fake_catalogs, shared_source_list, image_chosen
):
    # Saving copies the chosen list to the source list file in the settings,
    # which stays as it was.
    comparison_widget = _viewer_in_mode("existing")
    comparison_widget._choose_source_list.value = shared_source_list
    if image_chosen:
        _choose_image(comparison_widget, image_with_fake_catalogs)

    _click_save(comparison_widget)

    assert Path("source_locations.ecsv").read_bytes() == (
        shared_source_list.read_bytes()
    )
    assert comparison_widget.photom_apertures_file == "source_locations.ecsv"
    assert _saved_source_list_file() == "source_locations.ecsv"
    save_bar = comparison_widget._comparison_save_bar
    assert not save_bar.unsaved_changes
    assert save_bar.bn_save.disabled
    assert (
        "Copied shared-sources.ecsv to source_locations.ecsv" in save_bar.message.value
    )


def test_choosing_the_source_list_file_itself_leaves_nothing_to_save(
    image_with_fake_catalogs,
):
    _write_list_file(Path("source_locations.ecsv"))
    comparison_widget = _viewer_in_mode("existing")
    comparison_widget._choose_source_list.value = Path(
        "source_locations.ecsv"
    ).absolute()
    _choose_image(comparison_widget, image_with_fake_catalogs)

    _assert_shows_shared_list(comparison_widget)
    save_bar = comparison_widget._comparison_save_bar
    assert not save_bar.unsaved_changes
    assert save_bar.bn_save.disabled
    assert not _shown(comparison_widget._confirm_overwrite)


def test_loaded_list_replaced_by_a_generated_one_is_no_longer_shown(
    image_with_fake_catalogs,
):
    # The list shown from memory is no longer what is in the file, so it is
    # dropped rather than shown as saved.
    destination = Path("source_locations.ecsv")
    _write_list_file(destination)
    comparison_widget = _viewer_in_mode("existing")
    comparison_widget._choose_source_list.value = destination.absolute()
    _choose_image(comparison_widget, image_with_fake_catalogs)

    comparison_widget._mode.value = "generate"
    _click_save(comparison_widget)
    comparison_widget._confirm_overwrite._yes.click()
    assert len(SourceListData.read(destination)) == 3

    comparison_widget._mode.value = "existing"

    assert len(comparison_widget.iw.catalog_labels) == 0
    assert len(comparison_widget.generate_table()) == 0
    message = comparison_widget._source_list_message.value
    assert "source_locations.ecsv was replaced" in message

    # Choosing the file again shows what it holds now.
    comparison_widget._choose_source_list.value = destination.absolute()

    table = comparison_widget.generate_table()
    assert list(table["marker name"]) == ["APASS comparison"] * 3


@pytest.mark.parametrize("mode", ["existing", "generate"])
def test_overwriting_a_source_list_needs_confirmation(
    image_with_fake_catalogs, shared_source_list, existing_destination, mode
):
    # A source list file that is already there is replaced only after the
    # user says yes; the file this session wrote last is replaced without
    # asking.
    destination = existing_destination
    original_contents = destination.read_bytes()
    comparison_widget = _viewer_in_mode(mode)
    if mode == "existing":
        comparison_widget._choose_source_list.value = shared_source_list
    _choose_image(comparison_widget, image_with_fake_catalogs)
    save_bar = comparison_widget._comparison_save_bar
    confirm = comparison_widget._confirm_overwrite

    _click_save(comparison_widget)

    assert _shown(confirm)
    assert "source_locations.ecsv" in confirm.message
    # The question says what answering No means.
    assert "No keeps" in confirm.message
    assert destination.read_bytes() == original_contents

    confirm._no.click()

    assert not _shown(confirm)
    assert destination.read_bytes() == original_contents
    assert save_bar.unsaved_changes
    assert not save_bar.bn_save.disabled

    _click_save(comparison_widget)
    confirm._yes.click()

    if mode == "existing":
        assert destination.read_bytes() == shared_source_list.read_bytes()
    else:
        assert len(SourceListData.read(destination)) == 3
    assert not save_bar.unsaved_changes

    # Make something new to save, and save it.
    if mode == "existing":
        other = Path("other.ecsv")
        _write_other_list(other)
        comparison_widget._choose_source_list.value = other.absolute()
    else:
        comparison_widget._dim_mag_limit_input.value = 13
        _regenerate(comparison_widget)
    _click_save(comparison_widget)

    assert not _shown(confirm)
    assert len(SourceListData.read(destination)) == 2
    assert not save_bar.unsaved_changes


def test_no_question_when_outputs_can_not_be_overwritten(
    shared_source_list, existing_destination
):
    # Asking "Replace it?" and then refusing to would be no help.
    original_contents = existing_destination.read_bytes()
    comparison_widget = _viewer_in_mode("existing", overwrite_outputs=False)
    comparison_widget._choose_source_list.value = shared_source_list
    save_bar = comparison_widget._comparison_save_bar

    _click_save(comparison_widget)

    assert not _shown(comparison_widget._confirm_overwrite)
    assert "can not be overwritten" in save_bar.message.value
    assert existing_destination.read_bytes() == original_contents
    assert save_bar.unsaved_changes


def _set_source_list_file(comparison_widget, name):
    # Type a new source list file name into the source location settings.
    settings = comparison_widget.source_locations.value.copy()
    settings["source_list_file"] = name
    comparison_widget.source_locations.value = settings


def _switch_to_existing_with_no_list(comparison_widget):
    comparison_widget._mode.value = "existing"


def _choose_the_destination_itself(comparison_widget):
    comparison_widget._mode.value = "existing"
    comparison_widget._choose_source_list.value = Path(
        "source_locations.ecsv"
    ).absolute()


@pytest.mark.parametrize(
    "change_while_asking",
    [_switch_to_existing_with_no_list, _choose_the_destination_itself],
)
def test_yes_does_nothing_if_there_is_nothing_left_to_save(
    image_with_fake_catalogs, existing_destination, change_while_asking
):
    # Yes applies to what is on screen when it is clicked.
    destination = existing_destination
    original_contents = destination.read_bytes()
    comparison_widget = _viewer_in_mode("generate")
    _choose_image(comparison_widget, image_with_fake_catalogs)
    _click_save(comparison_widget)
    assert _shown(comparison_widget._confirm_overwrite)

    change_while_asking(comparison_widget)
    comparison_widget._confirm_overwrite._yes.click()

    assert destination.read_bytes() == original_contents
    save_bar = comparison_widget._comparison_save_bar
    assert not save_bar.unsaved_changes
    assert save_bar.message.value == ""


@pytest.mark.usefixtures("existing_destination")
def test_yes_does_not_replace_a_file_it_was_not_asked_about(image_with_fake_catalogs):
    other = Path("precious.ecsv")
    _write_other_list(other)
    original_contents = other.read_bytes()
    comparison_widget = _viewer_in_mode("generate")
    _choose_image(comparison_widget, image_with_fake_catalogs)
    _click_save(comparison_widget)

    _set_source_list_file(comparison_widget, "precious.ecsv")
    comparison_widget._confirm_overwrite._yes.click()

    assert other.read_bytes() == original_contents
    assert _shown(comparison_widget._confirm_overwrite)
    assert "precious.ecsv" in comparison_widget._confirm_overwrite.message

    comparison_widget._confirm_overwrite._yes.click()

    assert len(SourceListData.read(other)) == 3


def test_clicks_do_not_change_the_saved_source_list(viewer_with_fake_catalogs):
    # The saved list has every star the catalog query marked, whether or not
    # it was clicked, and a click does not make anything unsaved.
    comparison_widget = viewer_with_fake_catalogs
    save_bar = comparison_widget._comparison_save_bar
    _click(comparison_widget, 1000.0, 1000.0)
    assert len(comparison_widget.generate_table()) == 2

    _click_save(comparison_widget)

    assert len(SourceListData.read("source_locations.ecsv")) == 3

    _click(comparison_widget, 1200.0, 1000.0)

    assert not save_bar.unsaved_changes
    assert save_bar.bn_save.disabled


@pytest.mark.parametrize("file_exists", [True, False])
def test_no_source_list_to_save_when_no_stars_are_marked(
    request, viewer_with_fake_catalogs, file_exists
):
    # Limits that match no stars leave nothing to save: no question, no write.
    destination = Path("source_locations.ecsv")
    if file_exists:
        destination = request.getfixturevalue("existing_destination")
        original_contents = destination.read_bytes()
    comparison_widget = viewer_with_fake_catalogs
    save_bar = comparison_widget._comparison_save_bar
    comparison_widget._bright_mag_limit_input.value = 10
    comparison_widget._dim_mag_limit_input.value = 11
    _regenerate(comparison_widget)

    _click_save(comparison_widget)

    assert not _shown(comparison_widget._confirm_overwrite)
    if file_exists:
        assert destination.read_bytes() == original_contents
    else:
        assert not destination.exists()
    assert not save_bar.unsaved_changes
    assert save_bar.bn_save.disabled
    assert "no source list to save" in save_bar.message.value


def test_new_source_list_file_name_needs_saving(viewer_with_fake_catalogs):
    # The list has been saved, but not to the file now named in the settings.
    comparison_widget = viewer_with_fake_catalogs
    save_bar = comparison_widget._comparison_save_bar
    _click_save(comparison_widget)

    _set_source_list_file(comparison_widget, "new-name.ecsv")

    assert save_bar.unsaved_changes
    _click_save(comparison_widget)
    assert len(SourceListData.read("new-name.ecsv")) == 3
    assert cf.crossmatch_APASS2VSX.call_count == 1


def test_copied_list_stays_saved_after_image_change_and_mode_round_trip(
    existing_list_viewer, tmp_path
):
    comparison_widget = existing_list_viewer
    save_bar = comparison_widget._comparison_save_bar
    _click_save(comparison_widget)

    _choose_second_image(comparison_widget, tmp_path)
    assert not save_bar.unsaved_changes

    comparison_widget._mode.value = "generate"
    comparison_widget._mode.value = "existing"
    assert not save_bar.unsaved_changes


def _switch_to_generate(comparison_widget, tmp_path):  # noqa: ARG001
    comparison_widget._mode.value = "generate"


@pytest.mark.parametrize(
    "viewer, query_again",
    [
        ("viewer_with_fake_catalogs", _choose_second_image),
        ("existing_list_viewer", _switch_to_generate),
    ],
)
def test_failed_query_leaves_no_markers(request, tmp_path, viewer, query_again):
    # The stars that were shown, for the previous image or from the existing
    # list, are not left on the image as if they were the generated list.
    comparison_widget = request.getfixturevalue(viewer)
    comparison_widget._show_labels_button.value = True
    cf.set_up.side_effect = RuntimeError("catalog is down")

    # The error is also raised, so that the log gets the traceback.
    with pytest.raises(RuntimeError, match="catalog is down"):
        query_again(comparison_widget, tmp_path)

    assert len(comparison_widget.iw.catalog_labels) == 0
    assert len(comparison_widget.generate_table()) == 0
    marks = comparison_widget.iw._astro_im._scatter_marks
    assert not any(key.startswith("__label__") for key in marks)
    assert any("catalog is down" in m for m in _legend_messages(comparison_widget))
    assert not comparison_widget._comparison_save_bar.unsaved_changes
    # The query can be tried again.
    assert not comparison_widget._regenerate_button.disabled


def test_catalog_error_cleared_when_switching_to_existing(image_with_fake_catalogs):
    comparison_widget = _viewer_in_mode("generate")
    cf.set_up.side_effect = RuntimeError("catalog is down")
    with pytest.raises(RuntimeError, match="catalog is down"):
        _choose_image(comparison_widget, image_with_fake_catalogs)
    assert _legend_messages(comparison_widget)

    comparison_widget._mode.value = "existing"

    assert _legend_messages(comparison_widget) == []


def test_image_given_to_constructor_loads_when_mode_is_answered(
    image_with_fake_catalogs,
):
    comparison_widget = cf.ComparisonViewer(
        file=image_with_fake_catalogs.name, directory=image_with_fake_catalogs.parent
    )
    assert cf.crossmatch_APASS2VSX.call_count == 0

    comparison_widget._mode.value = "generate"

    assert len(comparison_widget.generate_table()) == 3


def test_existing_list_stars_are_placed_in_the_image_frame(tmp_path):
    # The image WCS is FK4 (B1950), far from the ICRS positions of the list,
    # so the stars are on the image only if their positions are transformed
    # to the image frame.
    ccd = make_ey_uma_image()
    ccd.wcs.wcs.radesys = "FK4"
    ccd.wcs.wcs.equinox = 1950
    image = tmp_path / "fk4.fits"
    ccd.write(image)
    path = tmp_path / "fk4-sources.ecsv"
    SourceListData(input_data=_source_list_table(ccd=ccd)).write(path)

    comparison_widget = _existing_list_viewer_for(image, path)

    assert list(comparison_widget.generate_table()["star_id"]) == [3, 8, 12, 15]


def test_existing_source_list_is_view_only(existing_list_viewer):
    # Clicking a star does not exclude it, and there are no variables to save
    # since the catalogs are not queried.
    comparison_widget = existing_list_viewer
    iw = comparison_widget.iw

    _click(comparison_widget, *SHARED_STARS[2][2])

    assert not any(label.startswith("elim") for label in iw.catalog_labels)
    assert "view only" in comparison_widget._status_message.value
    _assert_shows_shared_list(comparison_widget)
    assert comparison_widget._save_var_info.disabled

    # Clicking excludes stars when generating a source list.
    comparison_widget._mode.value = "generate"
    _click(comparison_widget, 1000.0, 1000.0)
    assert any(label.startswith("elim") for label in iw.catalog_labels)
    assert not comparison_widget._save_var_info.disabled


def test_help_legend_and_tooltips_describe_the_mode():
    comparison_widget = _viewer_in_mode("existing")
    help_text = comparison_widget.help_stuff.children[0]
    legend = comparison_widget._legend_spinner_box.children[0]
    save = comparison_widget._comparison_save_bar.bn_save

    assert "view only" in help_text.value
    assert "to mark it" not in help_text.value
    assert "Red +" not in legend.value
    assert "catalog" not in legend.value
    assert "Copy" in save.tooltip
    assert "existing source list" in comparison_widget._save_var_info.tooltip

    comparison_widget._mode.value = "generate"

    assert "to mark it" in help_text.value
    # After a click the labels no longer match the saved star ids (#752).
    assert "renumbered" in help_text.value
    assert "Red +" in legend.value
    assert "Copy" not in save.tooltip
    assert comparison_widget._save_var_info.tooltip == ""


def test_list_for_another_field_gets_a_warning(image_with_fake_catalogs, tmp_path):
    # None of the stars are on the image. The warning is red, does not list
    # every star, and the view is not centred on the list's target.
    path = tmp_path / "wrong-field.ecsv"
    names = ["TESS Targets"] + ["APASS comparison"] * 14
    _write_list_file(
        path,
        [(i + 1, name, (3500.0 + 10 * i, 1000.0)) for i, name in enumerate(names)],
    )

    comparison_widget = _existing_list_viewer_for(image_with_fake_catalogs, path)

    assert len(comparison_widget.generate_table()) == 0
    (message,) = _legend_messages(comparison_widget)
    assert "color: red" in message
    assert "None of the 15 stars" in message
    assert "9, 10 and 5 more" in message
    n_y, n_x = CCD_SHAPE
    center = make_ey_uma_image().wcs.pixel_to_world(n_x / 2, n_y / 2)
    assert comparison_widget.target_coord.separation(center) < 1e-3 * u.arcsec


def test_loaded_stars_in_the_same_place_keep_their_ids(
    image_with_fake_catalogs, tmp_path
):
    path = tmp_path / "same-place.ecsv"
    _write_list_file(
        path,
        [
            (4, "TESS Targets", (900.0, 900.0)),
            (9, "VSX", (900.0, 900.0)),
            (2, "APASS comparison", (1500.0, 1500.0)),
        ],
    )
    comparison_widget = _existing_list_viewer_for(image_with_fake_catalogs, path)

    table = comparison_widget.generate_table()

    assert list(table["star_id"]) == [4, 9, 2]
    assert list(table["marker name"]) == ["TESS Targets", "VSX", "APASS comparison"]


def test_masked_position_is_reported_off_the_image(image_with_fake_catalogs, tmp_path):
    table = _source_list_table()
    mask = [False, False, True, False, False, False]
    table["ra"] = MaskedColumn(table["ra"].value, mask=mask, unit=u.deg)
    path = tmp_path / "masked.ecsv"
    SourceListData(input_data=table).write(path)
    comparison_widget = _existing_list_viewer_for(image_with_fake_catalogs, path)

    assert list(comparison_widget.generate_table()["star_id"]) == [3, 8, 15]
    messages = _legend_messages(comparison_widget)
    assert "3 stars" in messages[0]
    assert "12, 21, 22" in messages[0]


def _write_text_file(path):
    path.write_text("this is not a table\n")


def _write_table_without_marker_names(path):
    table = _source_list_table()
    table.remove_column("marker name")
    SourceListData(input_data=table).write(path)


def _write_table_without_pixel_positions(path):
    table = _source_list_table()
    table.remove_columns(["xcenter", "ycenter"])
    table.meta.update(has_ra_dec=True, has_x_y=False)
    table.write(path)


def _write_table_without_source_list_meta(path):
    # Written as a plain table, so photometry cannot read it as a source list.
    _source_list_table().write(path)


def _write_table_without_has_x_y(path):
    table = _source_list_table()
    table.meta.update(has_ra_dec=True)
    table.write(path)


def _write_table_with_unknown_marker_name(path):
    table = _source_list_table()
    table["marker name"][0] = "Mystery"
    SourceListData(input_data=table).write(path)


def _write_table_with_text_ra(path):
    table = _source_list_table()
    table["ra"] = [str(ra) for ra in table["ra"].value]
    table.meta.update(has_ra_dec=True, has_x_y=True)
    table.write(path)


@pytest.mark.parametrize(
    "write_bad_file, reason",
    [
        (_write_text_file, "could not be read"),
        (_write_table_without_marker_names, "no marker name column"),
        (_write_table_without_pixel_positions, "no xcenter or ycenter column"),
        (_write_table_without_source_list_meta, "has_ra_dec"),
        (_write_table_without_has_x_y, "has_x_y"),
        (_write_table_with_unknown_marker_name, "unknown marker names Mystery"),
        (_write_table_with_text_ra, "could not be read"),
    ],
)
def test_choosing_an_invalid_source_list_shows_a_message(
    image_with_fake_catalogs, tmp_path, write_bad_file, reason
):
    # A file that is not a usable source list leaves the viewer as it was and
    # says why.
    comparison_widget = _viewer_in_mode("existing")
    _choose_image(comparison_widget, image_with_fake_catalogs)
    settings_file = tmp_path / "partial_photometry_settings.json"
    saved_settings = settings_file.read_bytes()
    bad_file = tmp_path / "bad.ecsv"
    write_bad_file(bad_file)

    comparison_widget._choose_source_list.value = bad_file

    message = comparison_widget._source_list_message.value
    assert "bad.ecsv is not a usable source list" in message
    assert reason in message
    assert settings_file.read_bytes() == saved_settings
    assert len(comparison_widget.generate_table()) == 0
    assert not comparison_widget._comparison_save_bar.unsaved_changes
    assert not os.path.exists("source_locations.ecsv")


def test_fixed_file_can_be_chosen_again(existing_list_viewer, tmp_path):
    comparison_widget = existing_list_viewer
    path = tmp_path / "fixed.ecsv"
    _write_text_file(path)
    comparison_widget._choose_source_list.value = path
    message = comparison_widget._source_list_message.value
    assert "not a usable source list" in message
    assert "Still showing shared-sources.ecsv" in message
    _assert_shows_shared_list(comparison_widget)

    _write_other_list(path)
    comparison_widget._choose_source_list.value = path

    assert list(comparison_widget.generate_table()["star_id"]) == [3, 8]


@pytest.mark.remote_data
def test_loading_second_image_succeeds(tmp_path):
    # Regression test for #384

    # Make a comparison viewer and load TIC 402828941
    comparison_widget = _viewer_in_mode("generate")
    ccd = CCDData.read(
        get_pkg_data_filename(
            "tests/data/TIC-402828941-tiny.fit.bz2", package="stellarphot"
        )
    )
    # The object name in the file, "TIC-402828941", cannot be resolved; this
    # one can.
    ccd.header["object"] = "TIC 402828941"
    file_name = "TIC.fits"
    path = tmp_path / file_name
    ccd.write(path, overwrite=True)

    _choose_image(comparison_widget, path)

    TIC_coord = SkyCoord(305.45862498, 19.43593327, unit="deg")
    # Make sure we have the TIC star as the target coordinates
    assert comparison_widget.target_coord.separation(TIC_coord) < 10 * u.arcsec

    # Load a second image
    ccd2 = CCDData.read(get_pkg_data_filename("../../tests/data/wasp-10-tiny.fit.bz2"))
    # ccd2.data = np.zeros(CCD_SHAPE)
    wasp_file = tmp_path / "wasp-10.fits"
    ccd2.write(wasp_file, overwrite=True)
    wasp_coord = SkyCoord(348.99291924, 31.46286002, unit="deg")

    _choose_image(comparison_widget, wasp_file)
    # Make sure we have WASP-10 as the target coordinates
    assert comparison_widget.target_coord.separation(wasp_coord) < 1 * u.arcsec

    # Also make sure this is where the viewer is actually centered
    viewer_coord = comparison_widget.iw.get_viewport(sky_or_pixel="sky")["center"]

    # It is not clear to me what the viewer position defaults to, but it should be
    # close to the target coordinates, where by close I mean "less than the
    # diagonal width of the frame".
    assert viewer_coord.separation(wasp_coord) < 1 * u.degree


@pytest.mark.remote_data
@pytest.mark.parametrize("tic_id", [402828941, 367710318])
def test_loading_input_source_list(tmp_path, tic_id):
    # Test that we can load a source list from a file
    comparison_widget = _viewer_in_mode("generate")
    compressed_input_source_list = get_pkg_data_filename(
        f"tests/data/TIC-{tic_id}-source-list-input.ecsv.gz", package="stellarphot"
    )
    with gzip.open(compressed_input_source_list, "rb") as f:
        source_list_content = f.read()
    input_source_list = tmp_path / "input-sources.ecsv"
    input_source_list.write_text(source_list_content.decode("utf-8"))
    ccd = CCDData.read(
        get_pkg_data_filename(
            f"tests/data/TIC-{tic_id}-tiny.fit.bz2", package="stellarphot"
        )
    )
    # Name the image for some other object, one whose name can be resolved, to
    # check that the target comes from the input source list anyway.
    ccd.header["object"] = "WASP-10"
    file_name = "TIC.fits"
    path = tmp_path / file_name
    ccd.write(path, overwrite=True)

    _choose_image(comparison_widget, path)

    # Until there is a source list the object name sets the target...
    wasp_coord = SkyCoord(348.99291924, 31.46286002, unit="deg")
    assert comparison_widget.target_coord.separation(wasp_coord) < 1 * u.arcsec

    comparison_widget._choose_input_source_list.value = input_source_list
    _click_save(comparison_widget)

    # Check that the number of tess targets matches the number in the
    # input source list

    comp_viewer_sources = SourceListData.read("source_locations.ecsv")
    input_sources = SourceListData.read(input_source_list)

    # ...and after that the first star in the source list does.
    first_target = SkyCoord(input_sources["ra"][0], input_sources["dec"][0])
    assert comparison_widget.target_coord.separation(first_target) < 0.01 * u.arcsec

    # The "separation" and "sort" columns are only used to order the table and
    # should not be written to the aperture file (issue #34).
    assert "separation" not in comp_viewer_sources.colnames
    assert "sort" not in comp_viewer_sources.colnames

    tess_targets = comp_viewer_sources["marker name"] == "TESS Targets"

    num_tess = np.sum(tess_targets)

    assert num_tess == len(input_sources)

    # Check that the stars are in the correct order that TESS expects
    comp_viewer_coords = SkyCoord(
        comp_viewer_sources["ra"], comp_viewer_sources["dec"], unit="deg"
    )
    input_coords = SkyCoord(input_sources["ra"], input_sources["dec"], unit="deg")
    sep = comp_viewer_coords[tess_targets].separation(input_coords)
    assert np.all(sep.arcsec < 1.0)
