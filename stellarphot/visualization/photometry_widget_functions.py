from pathlib import Path

from ccdproc import ImageFileCollection
import ipywidgets as ipw
from ipyfilechooser import FileChooser


class PhotometrySettings:
    def __init__(self):
        self._image_dir = FileChooser(title="Choose folder with images", show_only_dirs=True)
        self._aperture_file_loc = FileChooser(title='Choose aperture location file')
        self._aperture_settings_loc = FileChooser(title='Choose file with aperture settings')
        self._object_name = ipw.Dropdown(description='Choose object', style=dict(description_width='initial'))

        self._image_dir.register_callback(self._update_ifc)
        self._aperture_settings_loc.register_callback(self._update_aperture_rad)
        self.ifc = None
        self._box = ipw.VBox()
        self._box.children = [self._image_dir, self._aperture_file_loc, self._aperture_settings_loc, self._object_name]

    @property
    def box(self):
        return self._box

    @property
    def image_folder(self):
        return self._image_dir.selected

    @property
    def aperture_locations(self):
        return self._aperture_file_loc.selected

    @property
    def aperture_radius(self):
        return self._aperture_radius

    @property
    def object_name(self):
        return self._object_name.value

    def _update_ifc(self, change):
        self.ifc = ImageFileCollection(change.selected)
        self._update_object_list(change)

    def _update_object_list(self, change):
        if self.ifc.summary:
            self._object_name.options = sorted(set(self.ifc.summary['object'][~self.ifc.summary['object'].mask]))
        else:
            self._object_name.options = []

    def _update_aperture_rad(self, locator):
        with open(locator.selected) as f:
            stuff = f.read()

        self._aperture_radius = int(stuff.split(',')[0])
