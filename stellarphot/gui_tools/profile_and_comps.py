import ipywidgets as ipw

from stellarphot.gui_tools.comparison_functions import ComparisonViewer
from stellarphot.gui_tools.seeing_profile_functions import SeeingProfileWidget

__all__ = ["ComparisonAndSeeing"]


class ComparisonAndSeeing(ipw.VBox):
    """
    Combined viewer for seeing profile and comparison stars.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seeing = SeeingProfileWidget()
        self.comparison = ComparisonViewer()
        self.tabs = ipw.Tab(
            children=[self.seeing.box, self.comparison.box],
            titles=["Seeing Profile", "Comparison Stars"],
        )
        self.children = [self.tabs]
        self.comparison.fits_file.file_chooser.observe(
            self._make_observer(
                self.comparison.fits_file.file_chooser,
                self.seeing.fits_file.file_chooser,
            ),
            "_value",
        )
        self.seeing.fits_file.file_chooser.observe(
            self._make_observer(
                self.seeing.fits_file.file_chooser,
                self.comparison.fits_file.file_chooser,
            ),
            "_value",
        )

    @staticmethod
    def _make_observer(source, target):
        def observer(_):
            target.value = source._value

        return observer
