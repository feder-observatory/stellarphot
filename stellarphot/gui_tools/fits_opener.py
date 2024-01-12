import warnings
from pathlib import Path

from astropy.io import fits
from astropy.nddata import CCDData
from ipyautoui.custom import FileChooser

__all__ = ["FitsOpener"]


class FitsOpener:
    """
    A class to open FITS files using a file chooser and display them
    in an `astrowidgets.ImageWidget`.

    Parameters
    ----------

    title : str, optional
        The title of the FileChooser widget. The default is "Choose an image".

    filter_pattern : str, optional
        The filter pattern to use for the FileChooser widget. The default is None.

    Attributes
    ----------

    ccd : `astropy.nddata.CCDData`

    file_chooser : `ipyautoui.FileChooser`

    header : `astropy.io.fits.Header`

    object : str
        The object name from the FITS header.

    path : `pathlib.Path`
    """

    def __init__(self, title="Choose an image", filter_pattern=None, **kwargs):
        # ipyautoui.FileChooser doesn't have a title attribute, so we have to
        # add it after creating the widget
        self._fc = FileChooser(**kwargs)
        self._fc.title = title
        if not filter_pattern:
            self._fc.filter_pattern = ["*.fit*", "*.fit*.[bg]z"]
        else:
            self._fc.filter_pattern = filter_pattern

        self._header = {}
        self._selected_cache = self._fc.selected
        self.object = ""
        self._fc.observe(self._set_header, names="_value")

    @property
    def file_chooser(self):
        """
        The actual FileChooser widget.
        """
        return self._fc

    @property
    def header(self):
        """
        The header of the selected FITS file.
        """
        self._set_header()
        return self._header

    @property
    def ccd(self):
        """
        Return image stored in FITS file as CCDData object
        """
        return CCDData.read(self.path)

    @property
    def path(self):
        """
        The path to the selected FITS file.
        """
        return Path(self._fc.selected)

    def _set_header(self, _=None):
        if not self._header or self._fc.selected != self._selected_cache:
            self._selected_cache = self._fc.selected
            try:
                self._header = fits.getheader(self.path)
            except OSError:
                return

        try:
            self.object = self._header["object"]
        except KeyError:
            pass

    def load_in_image_widget(self, image_widget):
        """
        Load the selected image into an `astrowidgets.ImageWidget`, doing the best
        that can be done to suppress warnings.

        Parameters
        ----------

        image_widget : `astrowidgets.ImageWidget`
            The widget into which to load the image.
        """
        with warnings.catch_warnings():
            image_widget.load_fits(str(self.path))

    def set_file(self, file, directory=None):
        """
        Set the selected file of the `ipyfilechooser.FileChooser` to be the input file.

        Parameters
        ----------

        file : Path-like
            The file to be set as selected. Can be a string or a pathlib.Path.
            Cannot contain any directory information.

        directory : Path-like, optional
            Directory the file is in. The default value is the current directory of the
            `ipyfilechooser.FileChooser`.
        """
        if directory is None:
            directory = self._fc.selected_path

        self._fc.reset(directory, file)
        self._fc._apply_selection()
