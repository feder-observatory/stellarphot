import warnings

import ipywidgets as ipw
import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from astropy.wcs.wcs import FITSFixedWarning

from stellarphot.gui_tools import comparison_functions as cf

CCD_SHAPE = [2048, 3073]


def test_comparison_object_creation():
    # This test simply makes sure we can create the object
    comparison_widget = cf.ComparisonViewer()
    assert isinstance(comparison_widget.box, ipw.Box)


@pytest.mark.remote_data
def test_comparison_properties(tmp_path):
    # Test that we can load a file...
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
    # Set the object name to the star in this field, EY UMa
    ccd.header["object"] = "EY UMa"
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
        comparison_widget = cf.ComparisonViewer(directory=tmp_path, file=str(file_name))

    # Check that we have some variables in this field of view, which contains the
    # variable star EY UMa
    assert len(comparison_widget.variables) > 0

    # Check that we have APASS stars too
    table = comparison_widget.generate_table()
    assert "APASS comparison" in table["marker name"]

    # Check that is we show labels then the label names we expect show up in
    # the astrowidgets marker table.
    comparison_widget.show_labels()

    # Supress a warning about the default marker set containing no stars
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


def test_no_tess_buttons_at_init():
    # Make sure TESS controls are not displayed initially
    comparison_widget = cf.ComparisonViewer()
    assert comparison_widget.tess_save_toggle.layout.visibility == "hidden"
    assert comparison_widget._tess_object_info.layout.visibility == "hidden"
