from pathlib import Path

from ccdproc import ImageFileCollection
import ipywidgets as ipw
from ipyfilechooser import FileChooser

from stellarphot.settings import ApertureSettings, PhotometryFileSettings, ui_generator

__all__ = ['PhotometrySettings']


class PhotometrySettings:
    """
    A class to hold the widgets for photometry settings.

    Attributes
    ----------

    aperture_locations : `pathlib.Path`
        This is the path to the file containing the aperture locations.

    box : `ipywidgets.VBox`
        This is a box containing the widgets.

    image_folder : `pathlib.Path`
        This is the path to the folder containing the images.

    ifc : `ccdproc.ImageFileCollection`
        The ImageFileCollection for the selected folder.

    object_name : str
        The name of the object.
    """
    def __init__(self):
        self._file_loc_widget = ui_generator(PhotometryFileSettings)
        self._object_name = ipw.Dropdown(description='Choose object',
                                         style=dict(description_width='initial'))

        self._file_loc_widget.observe(self._update_locations)
        self.ifc = None
        self._box = ipw.VBox()
        self._box.children = [self._file_loc_widget, self._object_name]

    @property
    def box(self):
        """
        The box containing the widgets.
        """
        return self._box

    @property
    def image_folder(self):
        """
        The path to the folder containing the images.
        """
        return self.file_locations.image_folder

    @property
    def aperture_locations(self):
        """
        The path to the file containing the aperture locations
        """
        return self.file_locations.aperture_locations_file

    @property
    def object_name(self):
        """
        The name of the object.
        """
        return self._object_name.value

    def _update_locations(self, change):
        self.file_locations = PhotometryFileSettings(**self._file_loc_widget.value)
        self._update_ifc(change)
        if Path(self.file_locations.aperture_settings_file).is_file():
            self._update_aperture_settings(change)

    def _update_ifc(self, change):
        self.ifc = ImageFileCollection(self.file_locations.image_folder)
        self._update_object_list(change)

    def _update_object_list(self, change):
        if self.ifc.summary:
            self._object_name.options = sorted(set(self.ifc.summary['object'][~self.ifc.summary['object'].mask]))
        else:
            self._object_name.options = []

    def _update_aperture_settings(self, change):
        self.aperture_settings = ApertureSettings.parse_file(self.file_locations.aperture_settings_file)
