from ipywidgets import Box, HTML

__all__ = ['ImageViewer']


class ImageViewer(Box):
    """
    Basic image viewer Jupyter widget which wraps a ginga widget.
    """
    def __init__(self, image_file=None, **kwd):
        super(ImageViewer, self).__init__(**kwd)
        self._html_display = HTML(value="I will have an image soon")
        self.children = [self._html_display]
        self._server = None
        self._viewer = None
        # This call sets both server and viewer
        self._ginga_viewer()

        self._html_display.value = self._viewer_html
        self.image_file = image_file

    def _ginga_viewer(self):
        from ginga.web.pgw import ipg
        # Set this to True if you have a non-buggy python OpenCv
        # bindings--it greatly speeds up some operations
        use_opencv = False
        self._server = ipg.make_server(host='localhost',
                                       port=9987,
                                       use_opencv=use_opencv)
        self._server.start(no_ioloop=True)
        self._viewer = self._server.get_viewer('v1', width=600)

    @property
    def _viewer_html(self):
        base_string = \
            '<iframe src={viewer_url} height="650" width="650"></iframe>'
        return base_string.format(viewer_url=self._viewer.url)

    @property
    def image_file(self):
        return self._image_file

    @image_file.setter
    def image_file(self, value):
        self._image_file = value
        if value:
            self._viewer.load(value)
