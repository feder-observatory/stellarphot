import os
from pathlib import Path

import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

from stellarphot import SourceListData
from stellarphot.io import TOI
from stellarphot.settings.custom_widgets import TessPhotometrySetup


class TestTessPhotometrySetup:
    # This auto-used fixture changes the working directory to the temporary directory
    # and then changes back to the original directory after the test is done.
    @pytest.fixture(autouse=True)
    def change_to_tmp_dir(self, tmp_path):
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        # Yielding here is important. It means that when the test is done, the remainder
        # of the function will be executed. This is important because the test is run in
        # a temporary directory and we want to change back to the original directory
        # when the test is done.
        yield
        os.chdir(original_dir)

    def test_creation(self):
        widget = TessPhotometrySetup()
        # Initial TIC ID should be 0
        assert widget.tic_id == 0

        # All done display should be hidden initially
        assert widget.all_done.layout.display == "none"

        # Selection method should be set to option 0
        assert widget.drop.value == 0

    def test_choose_tic_id_entry_method(self):
        widget = TessPhotometrySetup()
        # Choose the TIC ID entry method
        widget.drop.value = 0

        # The TIC ID entry should be visible
        assert widget.tic_id_entry.layout.display == "flex"
        # The FITS opener should be hidden
        assert widget.fits_opener.file_chooser.layout.display == "none"

        # now choose the FITS opener method
        widget.drop.value = 1

        # The TIC ID entry should be hidden
        assert widget.tic_id_entry.layout.display == "none"
        # The FITS opener should be visible
        assert widget.fits_opener.file_chooser.layout.display == "flex"

    def test_set_tic_id_via_text_box(self, tess_tic_expected_values):
        widget = TessPhotometrySetup()
        # This needs to be a real TIC ID....
        widget.tic_id_entry.value = tess_tic_expected_values["tic_id"]
        assert widget.tic_id == tess_tic_expected_values["tic_id"]

        # The confirmation dialog should be visible now
        assert widget.confirm.layout.display == "flex"

    def test_get_tic_id_via_file(self, tess_tic_expected_values):
        widget = TessPhotometrySetup()

        # Make a little FITS file with an object keyword
        hdu = fits.PrimaryHDU()
        hdu.header["object"] = f"TIC {tess_tic_expected_values['tic_id']}"

        fits_file = Path("test.fits")

        hdu.writeto(fits_file)

        widget.fits_opener.file_chooser.value = str(fits_file)

        assert widget.tic_id == tess_tic_expected_values["tic_id"]

        # The confirmation dialog should be visible now
        assert widget.confirm.layout.display == "flex"

    def test_click_no_confirm(self, tess_tic_expected_values):
        # Test that clicking "No" on the confirmation dialog hides the dialog
        # and does not display the "All Done" display
        widget = TessPhotometrySetup()
        widget.tic_id_entry.value = tess_tic_expected_values["tic_id"]

        # Confirmation box is displayed, click no
        widget.confirm._no.click()

        # The confirmation dialog should be hidden now
        assert widget.confirm.layout.display == "none"

        # The all done display should still be hidden
        assert widget.all_done.layout.display == "none"

    @pytest.mark.remote_data
    def test_click_yes_confirm(self, tess_tic_expected_values):
        # Test that clicking "Yes" on the confirmation dialog hides the dialog
        # and displays the "All Done" display and creates the files we expect.

        widget = TessPhotometrySetup()
        widget.tic_id_entry.value = tess_tic_expected_values["tic_id"]

        # Confirmation box is displayed, click yes
        widget.confirm._yes.click()

        # The confirmation dialog should be hidden now
        assert widget.confirm.layout.display == "none"

        # The all done display should be visible
        assert widget.all_done.layout.display == "flex"

        # There should be exactly one json file from which we can create a TOI object
        TOI_json_file = list(Path(".").glob("*.json"))
        assert len(TOI_json_file) == 1
        TOI_json_file = TOI_json_file[0]
        assert str(tess_tic_expected_values["tic_id"]) in TOI_json_file.name

        # Make a TOI object from the JSON
        print(TOI_json_file.read_text())
        toi = TOI.model_validate_json(TOI_json_file.read_text())
        # Check that the TIC ID is correct
        assert toi.tic_id == tess_tic_expected_values["tic_id"]

        # There should also be one source list file
        source_list_file = list(Path(".").glob("*.ecsv"))
        assert len(source_list_file) == 1
        source_list_file = source_list_file[0]

        # This should not raise any errors
        sld = SourceListData.read(source_list_file)

        # THe first entry should be the exoplanet candidate
        assert toi.coord.separation(SkyCoord(sld["ra"][0], sld["dec"][0])).arcsecond < 1
