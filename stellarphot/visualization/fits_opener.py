from pathlib import Path

from astropy.io import fits

from ipyfilechooser import FileChooser


class FitsOpener:
    def __init__(self, title="Choose an image", filter_pattern=None, **kwargs):
        self._fc = FileChooser(title=title, **kwargs)
        if not filter_pattern:
            self._fc.filter_pattern = ['*.fit*', '*.fit*.[bg]z']

        self._header = {}
        self._selected_cache = self._fc.selected
        self.register_callback(lambda x: None)

    @property
    def file_chooser(self):
        """
        The actual FileChooser widget.
        """
        return self._fc

    @property
    def header(self):
        self._set_header()
        return self._header

    @property
    def path(self):
        return Path(self._fc.selected)

    def _set_header(self):
        if not self._header or self._fc.selected != self._selected_cache:
            self._selected_cache = self._fc.selected
            self._header = fits.getheader(self.path)

    def register_callback(self, callable):
        def wrap_call(change):
            self._set_header()
            callable(change)

        self._fc.register_callback(wrap_call)