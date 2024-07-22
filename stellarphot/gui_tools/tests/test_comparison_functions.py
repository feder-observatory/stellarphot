import os
import re
import warnings

import ipywidgets as ipw
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from astropy.wcs.wcs import FITSFixedWarning

from stellarphot.gui_tools import comparison_functions as cf
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

    # Check that if we show labels then the label names we expect show up in
    # the astrowidgets marker table.
    comparison_widget.show_labels()

    # Suppress a warning about the default marker set containing no stars
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Marker set named.*is empty",
            category=UserWarning,
        )
        label_markers = comparison_widget.iw.get_markers(
            marker_name=comparison_widget._label_name
        )
    # There should be a label for each star, so check that the number of labels matches
    # the length of the table of stars excluding the labels.
    table = comparison_widget.generate_table()

    # Drop table entries that are labels
    table = table[table["marker name"] != comparison_widget._label_name]

    assert len(label_markers) == len(table)

    # Make sure the aperture file has been written
    assert os.path.exists(comparison_widget.source_locations.value["source_list_file"])
    # Make sure the aperture file name matches the one we supplied, if
    # we supplied one.
    if source_file_name is not None:
        assert (
            comparison_widget.source_locations.value["source_list_file"]
            == source_file_name
        )

    # Save the source list settings
    comparison_widget.source_locations.savebuttonbar.bn_save.click()

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
            "tests/data/TIC_402828941-tiny.fit.bz2", package="stellarphot"
        )
    )
    file_name = "TIC.fits"
    path = tmp_path / file_name
    ccd.write(path, overwrite=True)
    print(path)
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
    # Get the value of the HTML widget that shows the coordinates
    view_coord_text = comparison_widget.iw.children[1].value
    print(view_coord_text)
    # Extract the coordinates from the text
    match = re.search(
        r"RA: +(\d+:\d+:[.\d]+), +DEC: +([+\d]+:\d+:[.\d]+)", view_coord_text
    )
    ra, dec = match.groups()
    viewer_coord = SkyCoord(ra, dec, unit=(u.hour, u.deg))

    # It is not clear to me what the viewer position defaults to, but it should be
    # close to the target coordinates, where by close I mean "less than the
    # diagonal width of the frame".
    assert viewer_coord.separation(wasp_coord) < 1 * u.degree
