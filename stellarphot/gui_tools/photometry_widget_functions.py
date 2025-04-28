from pathlib import Path

import ipywidgets as ipw
from ipyautoui.custom import FileChooser

from stellarphot import PhotometryData
from stellarphot.settings.custom_widgets import Spinner

__all__ = ["TessAnalysisInputControls"]


class TessAnalysisInputControls(ipw.VBox):
    """
    A class to hold the widgets for choosing TESS input
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = ipw.Layout(display="none")

        self.phot_chooser = FileChooser(filter_pattern=["*.csv", "*.fits", "*.ecsv"])
        self._fits_openr = ipw.VBox(
            children=[
                ipw.HTML(value="<h3>Select your photometry/flux file</h3>"),
                self.phot_chooser,
            ]
        )
        self.tic_file_chooser = FileChooser(filter_pattern=["*.json"])
        fits_openr2 = ipw.VBox(
            children=[
                ipw.HTML(value="<h3>Select your TESS info file</h3>"),
                self.tic_file_chooser,
            ],
            layout=hidden,
        )
        self._passband = ipw.Dropdown(
            description="Ccoose Filter",
            options=["gp", "ip"],
            disabled=True,
            layout=hidden,
        )

        spinner = Spinner(message="<h4>Loading photometry...</h4>")

        self.phot_data = None

        def update_filter_list(_):
            spinner.start()
            self.phot_data = PhotometryData.read(self.phot_chooser.value)
            passband_data = self.phot_data["passband"]
            fits_openr2.layout.display = "flex"
            self._passband.layout.display = "flex"
            self._passband.options = sorted(set(passband_data))
            self._passband.disabled = False
            self._passband.value = self._passband.options[0]
            spinner.stop()

        self.phot_chooser.observe(update_filter_list, names="_value")
        self.children = [self._fits_openr, spinner, fits_openr2, self._passband]

    @property
    def tic_info_file(self):
        p = Path(self.tic_file_chooser.value)
        selected_file = p.name
        if not selected_file:
            raise ValueError("No TIC info json file selected")
        return p

    @property
    def photometry_data_file(self):
        p = Path(self.phot_chooser.value)
        selected_file = p.name
        if not selected_file:
            raise ValueError("No photometry data file selected")
        return p

    @property
    def passband(self):
        return self._passband.value


def filter_by_dates(
    phot_times=None,
    use_no_data_before=None,
    use_no_data_between=None,
    use_no_data_after=None,
):
    n_dropped = 0

    bad_data = phot_times < use_no_data_before

    n_dropped = bad_data.sum()

    if n_dropped > 0:
        print(
            f"👉👉👉👉 Dropping {n_dropped} data points before "
            f"BJD {use_no_data_before}"
        )

    bad_data = bad_data | (
        (use_no_data_between[0][0] < phot_times)
        & (phot_times < use_no_data_between[0][1])
    )

    new_dropped = bad_data.sum() - n_dropped

    if new_dropped:
        print(
            f"👉👉👉👉 Dropping {new_dropped} data points between "
            f"BJD {use_no_data_between[0][0]} and {use_no_data_between[0][1]}"
        )

    n_dropped += new_dropped

    bad_data = bad_data | (phot_times > use_no_data_after)

    new_dropped = bad_data.sum() - n_dropped

    if new_dropped:
        print(
            f"👉👉👉👉 Dropping {new_dropped} data points after "
            f"BJD {use_no_data_after}"
        )

    n_dropped += new_dropped
    return bad_data
