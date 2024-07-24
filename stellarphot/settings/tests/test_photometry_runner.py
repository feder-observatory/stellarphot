import os
from pathlib import Path

import ipywidgets as ipw
import pytest
from astropy.io import fits

from stellarphot.settings import (
    PhotometrySettings,
    PhotometryWorkingDirSettings,
    settings_files,
)
from stellarphot.settings.custom_widgets import PhotometryRunner
from stellarphot.settings.tests.test_models import DEFAULT_PHOTOMETRY_SETTINGS


# See test_settings_file.TestSavedSettings for a detailed description of what the
# following fixture does. In brief, it patches the settings_files.PlatformDirs class
# so that the user_data_dir method returns the temporary directory.
@pytest.fixture(autouse=True)
def fake_settings_dir(mocker, tmp_path):
    mocker.patch.object(
        settings_files.PlatformDirs, "user_data_dir", tmp_path / "stellarphot"
    )


class TestPhotometryRunner:
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

    def test_photometry_runner_creation(self):
        # This test simply makes sure we can create the object
        photometry_runner = PhotometryRunner()
        assert isinstance(photometry_runner, ipw.Box)

    @pytest.mark.parametrize("do_photometry", [True, False])
    def test_photometry_runner_with_valid_settings(self, do_photometry):
        # This test makes sure that the PhotometryRunner widget can be used
        # to *start* a photometry run. It does not check the actual photometry
        # or the results of the photometry run, except to ensure that the expected
        # files are created.
        # Make a settings
        phot_settings = PhotometrySettings.model_validate(DEFAULT_PHOTOMETRY_SETTINGS)

        # Save a copy to the working directory
        wd_settings = PhotometryWorkingDirSettings()
        wd_settings.save(phot_settings)

        # Make sure a source list file exists
        sources = Path(phot_settings.source_location_settings.source_list_file)
        sources.touch()

        # Make sure a fake fits file exists
        fake_fits = Path("fake.fits")
        fits_data = fits.PrimaryHDU(data=[[1, 2], [3, 4]])
        object_name = "Fake Object"
        fits_data.header["object"] = object_name
        fits_data.writeto(fake_fits)

        photometry_runner = PhotometryRunner()
        # Select a fits file
        photometry_runner.fitsopen.file_chooser.reset(".", fake_fits.name)
        photometry_runner.fitsopen.file_chooser._apply_selection()
        photometry_runner.fitsopen.file_chooser.value = fake_fits

        # Check that the message in the information box is as expected
        assert "Photometry will be performed" in photometry_runner.info_box.value
        assert object_name in photometry_runner.info_box.value

        if do_photometry:
            # Run the photometry
            photometry_runner.confirm._yes.click()
            assert "Photometry is running" in photometry_runner.info_box.value

            # Check that the expected file was created, in this case just the notebook.
            # No photometry is actually done because the source is empty and the "image"
            # has no stars.
            assert Path(photometry_runner.photometry_notebook_name).exists()

            # Make sure we do get the error we expect, which is generated when
            # the source list is read. It is an empty file, not actually an ecsv.
            with open(photometry_runner.photometry_notebook_name) as f:
                notebook_text = f.read()
                assert "InconsistentTableError" in notebook_text
        else:
            # Cancel the photometry
            photometry_runner.confirm._no.click()

            assert photometry_runner.info_box.value == ""
