import gzip
import os
import warnings

import ipywidgets as ipw
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.table import Table
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

    vsx = Table({"coords": ccd.wcs.pixel_to_world([500.0, 600.0], [700.0, 800.0])})
    apass = Table(
        {"coords": ccd.wcs.pixel_to_world([1000.0, 1100.0], [1200.0, 1300.0])}
    )

    cf.make_markers(iw, [], vsx, apass, name_or_coord=None)

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


def test_spinner_cleared_and_error_shown_when_catalog_load_fails(tmp_path, monkeypatch):
    # If fetching the VSX/APASS catalogs fails, the spinner should not keep
    # spinning forever -- the legend should come back and an error message
    # should be shown to the user.
    path, _ = _write_ey_uma_file(tmp_path)

    comparison_widget = cf.ComparisonViewer()
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
    # A viewer showing the image with the fake catalogs.
    comparison_widget = cf.ComparisonViewer()
    _choose_image(comparison_widget, image_with_fake_catalogs)
    return comparison_widget


def test_save_bar_refreshes_comparisons_with_new_settings(viewer_with_fake_catalogs):
    # Changing the catalog or magnitude limits marks the settings unsaved, and
    # saving re-queries with the new settings and redraws the comparison
    # markers.
    comparison_widget = viewer_with_fake_catalogs
    save_bar = comparison_widget._comparison_save_bar

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

    comparison_widget._comparison_catalog_choice.value = "ATLAS refcat2"
    # Changing the catalog alone is enough to make the settings unsaved.
    assert save_bar.unsaved_changes
    assert not save_bar.bn_save.disabled

    comparison_widget._dim_mag_limit_input.value = 13

    assert save_bar.unsaved_changes
    assert not save_bar.bn_save.disabled
    assert not save_bar.bn_revert.disabled
    # Nothing is re-queried until the settings are saved.
    assert cf.crossmatch_APASS2VSX.call_count == 1

    save_bar.bn_save.click()

    assert cf.crossmatch_APASS2VSX.call_args.kwargs == {
        "catalog": "ATLAS refcat2",
        "magnitude_limit": 13,
    }
    assert cf.set_up.call_args.kwargs == {"magnitude_limit": 13}
    # The marker label stays "APASS comparison" whatever the catalog, since
    # downstream code matches on it.
    comps = comparison_widget.iw.get_catalog(catalog_label="APASS comparison")
    assert len(comps) == 2
    assert not save_bar.unsaved_changes
    assert save_bar.bn_save.disabled
    assert save_bar.bn_revert.disabled


def test_save_bar_revert_restores_settings(viewer_with_fake_catalogs):
    comparison_widget = viewer_with_fake_catalogs
    save_bar = comparison_widget._comparison_save_bar

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
    comparison_widget = cf.ComparisonViewer(comparison_catalog="ATLAS refcat2")
    assert not comparison_widget._comparison_save_bar.unsaved_changes

    _choose_image(comparison_widget, image_with_fake_catalogs)

    assert cf.crossmatch_APASS2VSX.call_args.kwargs["catalog"] == "ATLAS refcat2"


def test_settings_can_be_saved_before_image_is_chosen():
    # Nothing to reload yet, but the settings are kept for when an image is
    # chosen.
    comparison_widget = cf.ComparisonViewer()
    save_bar = comparison_widget._comparison_save_bar

    comparison_widget._dim_mag_limit_input.value = 13
    save_bar.bn_save.click()

    assert comparison_widget.dim_mag_limit == 13
    assert not save_bar.unsaved_changes
    assert save_bar.bn_save.disabled


@pytest.mark.parametrize("bright, dim", [(16, 15), (12, 12)])
def test_dim_limit_not_fainter_than_bright_blocks_save(
    viewer_with_fake_catalogs, bright, dim
):
    comparison_widget = viewer_with_fake_catalogs
    save_bar = comparison_widget._comparison_save_bar

    comparison_widget._bright_mag_limit_input.value = bright
    comparison_widget._dim_mag_limit_input.value = dim

    assert save_bar.unsaved_changes
    assert save_bar.bn_save.disabled
    assert not save_bar.bn_revert.disabled
    assert "fainter than bright limit" in save_bar.message.value

    comparison_widget._bright_mag_limit_input.value = 10
    comparison_widget._dim_mag_limit_input.value = 13

    assert not save_bar.bn_save.disabled
    assert save_bar.message.value == ""


def test_bright_limits_can_be_saved():
    # Limits suited to a bright target are as usable on screen as they are in
    # the constructor.
    comparison_widget = cf.ComparisonViewer(bright_mag_limit=6, dim_mag_limit=8.5)
    save_bar = comparison_widget._comparison_save_bar
    assert save_bar.message.value == ""

    comparison_widget._dim_mag_limit_input.value = 8.6

    assert not save_bar.bn_save.disabled
    assert save_bar.message.value == ""


def test_labels_match_comparisons_after_save(viewer_with_fake_catalogs):
    # Labels shown before a reload are replaced by labels for the reloaded
    # stars, not left on the image.
    comparison_widget = viewer_with_fake_catalogs
    comparison_widget._show_labels_button.value = True
    label_key = "__label__APASS comparison"
    marks = comparison_widget.iw._astro_im._scatter_marks
    assert list(marks[label_key].text) == ["C1", "C2", "C3"]

    comparison_widget._dim_mag_limit_input.value = 13
    comparison_widget._comparison_save_bar.bn_save.click()

    assert list(marks[label_key].text) == ["C1", "C2"]

    # With no comparison stars left there are no comparison labels either.
    comparison_widget._bright_mag_limit_input.value = 10
    comparison_widget._dim_mag_limit_input.value = 11
    comparison_widget._comparison_save_bar.bn_save.click()

    assert label_key not in marks


def test_message_shown_when_no_comparison_stars_found(viewer_with_fake_catalogs):
    # Limits that match no candidates are not an error; a message says that no
    # comparison stars were found, and goes away when some are found.
    comparison_widget = viewer_with_fake_catalogs
    legend = comparison_widget._legend_spinner_box.children[0]

    def messages():
        return [w.value for w in comparison_widget._legend_spinner_box.children[1:]]

    comparison_widget._bright_mag_limit_input.value = 10
    comparison_widget._dim_mag_limit_input.value = 11
    comparison_widget._comparison_save_bar.bn_save.click()

    assert comparison_widget.dim_mag_limit == 11
    assert len(messages()) == 1
    assert "No comparison stars" in messages()[0]
    # There are no variables or targets either, so there was nothing to put in
    # a source list, and the one from the earlier settings is still on disk.
    assert "source list was not updated" in messages()[0]

    comparison_widget._dim_mag_limit_input.value = 13
    comparison_widget._comparison_save_bar.bn_save.click()

    assert comparison_widget._legend_spinner_box.children == (legend,)


def test_failed_save_keeps_old_settings_and_allows_retry(viewer_with_fake_catalogs):
    # If the catalog query fails, the settings in use are unchanged and the
    # save button stays enabled so the save can be tried again.
    # Labels that were showing are still showing.
    comparison_widget = viewer_with_fake_catalogs
    save_bar = comparison_widget._comparison_save_bar
    comparison_widget._show_labels_button.value = True
    comparison_widget._comparison_catalog_choice.value = "ATLAS refcat2"
    comparison_widget._bright_mag_limit_input.value = 10
    comparison_widget._dim_mag_limit_input.value = 13
    cf.set_up.side_effect = RuntimeError("catalog is down")

    save_bar.bn_save.click()

    assert comparison_widget.comparison_catalog == "APASS DR9"
    assert comparison_widget.bright_mag_limit == 8
    assert comparison_widget.dim_mag_limit == 15
    assert save_bar.unsaved_changes
    assert not save_bar.bn_save.disabled
    marks = comparison_widget.iw._astro_im._scatter_marks
    assert list(marks["__label__APASS comparison"].text) == ["C1", "C2", "C3"]


def test_settings_kept_when_only_the_source_list_write_fails(image_with_fake_catalogs):
    # The stars were reloaded with the new settings, so those are the settings
    # in use even though the source list could not be written. The error is
    # shown, since an exception in a widget callback only reaches the log.
    comparison_widget = cf.ComparisonViewer(overwrite_outputs=False)
    save_bar = comparison_widget._comparison_save_bar
    # Choosing the image writes the source list, so the next write fails.
    _choose_image(comparison_widget, image_with_fake_catalogs)

    comparison_widget._dim_mag_limit_input.value = 13
    save_bar.bn_save.click()

    comps = comparison_widget.iw.get_catalog(catalog_label="APASS comparison")
    assert len(comps) == 2
    assert comparison_widget.dim_mag_limit == 13
    assert not save_bar.unsaved_changes
    assert "can not be overwritten" in save_bar.message.value


def test_no_markers_does_not_raise(tmp_path):
    # With no stars marked there is nothing to list, label or save.
    comparison_widget, _ = _viewer_with_target()

    assert len(comparison_widget.generate_table()) == 0
    comparison_widget.show_labels()
    source_file = tmp_path / "sources.ecsv"
    comparison_widget._save_aperture_to_file(filename=source_file)
    assert not source_file.exists()


@pytest.mark.remote_data
def test_loading_second_image_succeeds(tmp_path):
    # Regression test for #384

    # Make a comparison viewer and load TIC 402828941
    comparison_widget = cf.ComparisonViewer()
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

    with warnings.catch_warnings():
        # Ignore the warning about the WCS having non-standard keywords (the SIP
        # distortion parameters).
        warnings.filterwarnings(
            "ignore",
            message="Some non-standard WCS keywords were excluded",
            category=AstropyWarning,
        )
        comparison_widget._file_chooser.file_chooser.value = path

    TIC_coord = SkyCoord(305.45862498, 19.43593327, unit="deg")
    # Make sure we have the TIC star as the target coordinates
    assert comparison_widget.target_coord.separation(TIC_coord) < 10 * u.arcsec

    # Load a second image
    ccd2 = CCDData.read(get_pkg_data_filename("../../tests/data/wasp-10-tiny.fit.bz2"))
    # ccd2.data = np.zeros(CCD_SHAPE)
    wasp_file = tmp_path / "wasp-10.fits"
    ccd2.write(wasp_file, overwrite=True)
    wasp_coord = SkyCoord(348.99291924, 31.46286002, unit="deg")

    with warnings.catch_warnings():
        # Ignore the warning about the WCS having non-standard keywords (the SIP
        # distortion parameters).
        warnings.filterwarnings(
            "ignore",
            message="Some non-standard WCS keywords were excluded",
            category=AstropyWarning,
        )
        comparison_widget._file_chooser.file_chooser.value = wasp_file
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
    comparison_widget = cf.ComparisonViewer()
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

    with warnings.catch_warnings():
        # Ignore the warning about the WCS having non-standard keywords (the SIP
        # distortion parameters).
        warnings.filterwarnings(
            "ignore",
            message="Some non-standard WCS keywords were excluded",
            category=AstropyWarning,
        )
        comparison_widget._file_chooser.file_chooser.value = path

    # Until there is a source list the object name sets the target...
    wasp_coord = SkyCoord(348.99291924, 31.46286002, unit="deg")
    assert comparison_widget.target_coord.separation(wasp_coord) < 1 * u.arcsec

    comparison_widget._choose_input_source_list.value = input_source_list

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
