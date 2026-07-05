import gzip
import os
import warnings

import ipywidgets as ipw
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
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


def test_click_center_and_marking_neutralized():
    # astrowidgets 0.5.0 has a bug in which the bqplot ImageWidget's built-in
    # _mouse_click handler references attributes (click_center and is_marking)
    # that are never initialized, so any click raises AttributeError and,
    # because ipywidgets runs on_msg callbacks in registration order without
    # exception isolation, blocks our click handler too. ComparisonViewer
    # works around this by setting both attributes to False.
    comparison_widget = cf.ComparisonViewer()
    assert comparison_widget.iw.click_center is False
    assert comparison_widget.iw.is_marking is False


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
        catalog_style={"shape": "diamond", "color": "red", "size": 20},
    )

    status = ipw.HTML()
    callback = cf.wrap(iw, status)

    x, y = ccd.wcs.world_to_pixel(star_coord)
    click = {"event": "click", "domain": {"x": float(x), "y": float(y)}}

    # Events other than clicks should be ignored
    callback(
        iw.viewer.interaction,
        {"event": "mousemove", "domain": {"x": float(x), "y": float(y)}},
        [],
    )
    assert "elim1" not in iw.catalog_labels

    # Click on the star to exclude it...
    callback(iw.viewer.interaction, click, [])
    assert "elim1" in iw.catalog_labels

    # ...and click again to include it.
    callback(iw.viewer.interaction, click, [])
    assert "elim1" not in iw.catalog_labels

    # A click far from any star should display a message instead of
    # adding a marker.
    miss = {"event": "click", "domain": {"x": float(x) + 500, "y": float(y) + 500}}
    callback(iw.viewer.interaction, miss, [])
    assert "Click closer to a star" in status.value
    assert not any(label.startswith("elim") for label in iw.catalog_labels)


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

    apass_style = iw.get_catalog_style(catalog_label="APASS comparison")
    assert apass_style["shape"] == "diamond"
    assert apass_style["color"] == "red"

    vsx_style = iw.get_catalog_style(catalog_label="VSX")
    assert vsx_style["shape"] == "square"
    assert vsx_style["color"] == "blue"


@pytest.mark.parametrize("source_file_name", [None, "sources.ecsv"])
@pytest.mark.parametrize("has_object", [True, False])
@pytest.mark.remote_data
def test_comparison_properties(tmp_path, has_object, source_file_name):
    # Test that we can load a file...
    ccd = make_ey_uma_image(with_object=has_object)

    file_name = "test.fits"
    ccd.write(tmp_path / file_name, overwrite=True)
    # Change working directory for remainder of test so that the save does not pollute
    # the testing directory.
    os.chdir(tmp_path)

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


@pytest.mark.remote_data
def test_loading_second_image_succeeds(tmp_path):
    # Regression test for #384

    # Make a comparison viewer and load EY UMa
    comparison_widget = cf.ComparisonViewer()
    ccd = CCDData.read(
        get_pkg_data_filename(
            "tests/data/TIC-402828941-tiny.fit.bz2", package="stellarphot"
        )
    )
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
    # Make sure we have ey uma as the target coordinates
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

    comparison_widget._choose_input_source_list.value = input_source_list

    # Check that the number of tess targets matches the number in the
    # input source list

    comp_viewer_sources = SourceListData.read("source_locations.ecsv")
    input_sources = SourceListData.read(input_source_list)

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
