from stellarphot.gui_tools import FitsOpener


def test_fits_opener_title():
    opener = FitsOpener(title="Choose an image")
    assert opener.file_chooser.title == "Choose an image"
