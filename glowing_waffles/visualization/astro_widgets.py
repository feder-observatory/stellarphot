from __future__ import print_function

from ipywidgets import Box, HTML
from ginga.web.pgw import ipg

__all__ = ['ImageViewer']


class ImageViewer(Box):
    """
    Basic image viewer Jupyter widget which wraps a ginga widget.
    """
    def __init__(self, image_file=None, width=600, height=600,
                 show_position=False, **kwd):
        super(ImageViewer, self).__init__(**kwd)
        self._html_display = HTML(value="I will have an image soon")
        self.children = [self._html_display]
        self._server = None
        self._viewer = None
        self._width = width
        self._height = height
        self._show_position = show_position
        if self._show_position:
            self._height += 30
        # This call sets both server and viewer
        self._ginga_viewer()

        self._html_display.value = self._viewer_html
        self.image_file = image_file

    def _ginga_viewer(self):
        # Set this to True if you have a non-buggy python OpenCv
        # bindings--it greatly speeds up some operations
        use_opencv = False
        self._server = ipg.make_server(host='localhost',
                                       port=9987,
                                       use_opencv=use_opencv)
        self._server.start(no_ioloop=True)
        if self._show_position:
            viewer_class = ipg.EnhancedCanvasView
        else:
            viewer_class = ipg.BasicCanvasView
        self._viewer = self._server.get_viewer('v1',
                                               width=self._width,
                                               height=self._height,
                                               viewer_class=viewer_class)
        self._viewer.set_zoom_algorithm('rate')
        self._viewer.set_zoomrate(1.01)

    @property
    def _viewer_html(self):
        base_string = \
            '<iframe src={viewer_url} height="{height}" width="{width}"></iframe>'
        width, height = self._viewer.get_window_size()
        return base_string.format(viewer_url=self._viewer.url,
                                  width=width,
                                  height=height)

    @property
    def image_file(self):
        return self._image_file

    @image_file.setter
    def image_file(self, value):
        self._image_file = value
        if value:
            self._viewer.load(value)
