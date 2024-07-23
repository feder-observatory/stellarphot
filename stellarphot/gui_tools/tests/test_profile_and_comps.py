from pathlib import Path

import pytest
from astropy.utils.data import get_pkg_data_filename

from stellarphot.gui_tools import ComparisonAndSeeing


class TestComparisonAndSeeing:
    def copy_fits_to_temp(self, tmp_path):
        # Copy a fits file to a temporary directory
        fits_file = get_pkg_data_filename(
            "tests/data/wasp-10-tiny.fit.bz2", package="stellarphot"
        )
        fits_file = Path(fits_file)
        new_fits_file = tmp_path / fits_file.name
        new_fits_file.write_bytes(fits_file.read_bytes())
        return new_fits_file

    def test_init(self):
        # Just make sure we can create the object
        ComparisonAndSeeing()

    @pytest.mark.remote_data
    def test_setting_fits_file_in_comp_affects_seeing(self, tmp_path):
        # The only tricky-ish thing this class does is make sure that the
        # fits_file in the seeing widget and the fits_file in the comparison
        # widget are always the same. This test checks that setting the fits
        # file in the comparison widget updates the fits file in the seeing
        # widget.
        comp = ComparisonAndSeeing()
        file = self.copy_fits_to_temp(tmp_path)
        # Make sure the file exists -- necessary in some versions of ipyautoui
        file.touch()
        comp.comparison.fits_file.file_chooser._value = str(file)
        assert comp.seeing.fits_file.file_chooser._value == str(file)
        assert comp.comparison.fits_file.file_chooser.selected == str(file)

    @pytest.mark.remote_data
    def test_setting_fits_file_in_seeing_affects_comp(self, tmp_path):
        # This is the same as the previous test, but in the other direction.
        comp = ComparisonAndSeeing()
        file = self.copy_fits_to_temp(tmp_path)
        # Make sure the file exists -- necessary in some versions of ipyautoui
        file.touch()
        comp.seeing.fits_file.file_chooser._value = str(file)
        assert comp.comparison.fits_file.file_chooser._value == str(file)
        assert comp.seeing.fits_file.file_chooser.selected == str(file)
