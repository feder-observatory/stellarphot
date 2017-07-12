from __future__ import print_function

from IPython.display import IFrame, display
from ipywidgets import Box, Output
from ginga.web.pgw import ipg
from astropy.io.fits import HDUList

__all__ = ['ImageViewer']


class ImageViewer(Box):
    """
    Basic image viewer Jupyter widget which wraps a ginga widget.

    Set the content of the image viewer via the ``image`` attribute.

    Parameters
    ----------

    image : str, FITS HDU, or numpy array
        Content to display in the widget

    """
    _server = ipg.make_server(host='localhost',
                                       port=9987,
                                       use_opencv=False)
    _server_started = False
    _number_views = 0

    def __init__(self, image=None, width=600, height=600,
                 show_position=False, **kwd):
        super(ImageViewer, self).__init__(**kwd)

        self._image_display = Output(value="I will have an image soon")


        self.children = [self._image_display]
        if not ImageViewer._server_started:
            ImageViewer._server.start(no_ioloop=True)
            ImageViewer._server_started = True
        #self._server = None
        self._viewer = None
        self._width = width
        self._height = height
        self._show_position = show_position
        if self._show_position:
            self._height += 30
        # This call sets both server and viewer
        self._ginga_viewer()

        width, height = self._viewer.get_window_size()

        with self._image_display:
            display(IFrame(self._viewer.url, width, height))

        self.image = image

    def _ginga_viewer(self):
        # Set this to True if you have a non-buggy python OpenCv
        # bindings--it greatly speeds up some operations
        use_opencv = False

        if self._show_position:
            viewer_class = ipg.EnhancedCanvasView
        else:
            viewer_class = ipg.BasicCanvasView

        ImageViewer._number_views += 1
        self._viewer = self._server.get_viewer(str(ImageViewer._number_views),
                                               width=self._width,
                                               height=self._height,
                                               viewer_class=viewer_class)
        self._viewer.set_zoom_algorithm('rate')
        self._viewer.set_zoomrate(1.01)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self._image = value
        if value is not None:
            if isinstance(value, str):
                # Hope this is the name of a FITS file!
                self._viewer.load_fits(value)
            elif isinstance(value, HDUList):
                self._viewer.load_hdu(value)
            else:
                # Sure hope this is really numpy data
                self._viewer.load_data(value)
