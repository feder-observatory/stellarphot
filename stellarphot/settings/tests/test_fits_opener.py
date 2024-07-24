from astropy.utils.data import get_pkg_data_filename

from stellarphot.settings.fits_opener import FitsOpener


def test_fits_opener_title():
    opener = FitsOpener(title="Choose an image")
    assert opener.file_chooser.title == "Choose an image"


def test_fits_opener_set_value():
    opener = FitsOpener()
    file = get_pkg_data_filename(
        "tests/data/wasp-10-tiny.fit.bz2", package="stellarphot"
    )
    opener.file_chooser.value = file
    assert opener.path.name == "wasp-10-tiny.fit.bz2"
