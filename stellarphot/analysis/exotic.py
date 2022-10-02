import re
from pathlib import Path
import json

import numpy as np
import ipywidgets as ipw
from traitlets import observe, Bool

from astroquery.mast import Catalogs
from astropy.utils.data import get_pkg_data_filename


template_types = ['known', 'candidate']
template_json = {}
to_fill = {}

for template in template_types:
    template_name = get_pkg_data_filename('data/tic-template-for-exotic-'
                                          f'{template}.json')
    with open(template_name) as f:
        template_json[template] = json.load(f)

    template_name = get_pkg_data_filename(f'data/exotic-to-mod-{template}.json')
    with open(Path(template_name)) as f:
        to_fill[template] = json.load(f)

exotic_arguments = dict(
    known=['--nasaexoarch', '--pre'],
    candidate=['--override', '--pre']
)

join_char = "😬"

exotic_tic = {
    "Star Effective Temperature (K)": "Teff",
    "Star Effective Temperature (+) Uncertainty": "epos_Teff",
    "Star Effective Temperature (-) Uncertainty": "eneg_Teff",
    "Star Surface Gravity (log(g))": "logg",
    "Star Surface Gravity (+) Uncertainty": "epos_logg",
    "Star Surface Gravity (-) Uncertainty": "eneg_logg",
    "Host Star Name": "UCAC",
    "Star Metallicity ([FE/H])": "MH",
    "Star Metallicity (+) Uncertainty": "e_MH",
    "Star Metallicity (-) Uncertainty": "e_MH"
}


class MyValid(ipw.Button):
    """
    A more compact indicator of valid entries.
    """
    value = Bool(False, help="Bool value").tag(sync=True)

    def __init__(self, **kwd):
        super().__init__(**kwd)
        self.layout.width = '40px'
        self._set_properties(None)

    @observe('value')
    def _set_properties(self, change):
        if self.value:
            self.style.button_color = 'green'
            self.icon = 'check'
        else:
            self.style.button_color = 'red'
            self.icon = 'times'


def get_tic_info(TIC_ID):
    catalog_data = Catalogs.query_criteria(catalog="Tic", ID=TIC_ID)
    return catalog_data


def make_checker(indicator_widget, value_widget):
    def check_name(change):
        # Valid TIC number is 9 digits
        ticced = re.compile(r'TIC \d{9,10}$')
        owner = change['owner']
        is_tic = ticced.match(change['new'])
        if is_tic:
            if indicator_widget is not None:
                indicator_widget.value = True
            owner.disabled = True
            tic_info = get_tic_info(change['new'][-9:])
            if not tic_info:
                indicator_widget.value = False
                indicator_widget.tooltip = "Not a valid TIC number"
            else:
                populate_boxes(tic_info, value_widget)
            owner.disabled = False
        else:
            owner.disabled = False
            if indicator_widget is not None:
                indicator_widget.value = False
                indicator_widget.tooltip = 'TIC numbers have 9 digits'

    return check_name


def validate_exposure_time(indicator_widget, value_widget):
    def check_exposure(change):
        # Valid Exposure time is greater than zero
        if change['new'] > 0:
            if indicator_widget is not None:
                indicator_widget.value = True
        else:
            if indicator_widget is not None:
                indicator_widget.value = False
    return check_exposure


def populate_boxes(tic_info, value_widget):
    """
    Set the appropriate widget values given information pulled from the
    TIC.
    """
    for k, v in exotic_tic.items():
        exotic_key = join_char.join(["planetary_parameters", k])
        if k == "Host Star Name":
            value_widget['candidate'][exotic_key].value = \
                f'UCAC4 {tic_info[v][0]}'
        elif not np.isnan(tic_info[v][0]):
            value_widget['candidate'][exotic_key].value = tic_info[v][0]


validators = dict(known={}, candidate={})
validators['candidate']['Planet Name'] = make_checker
for k in validators:
    validators[k]['Exposure Time (s)'] = validate_exposure_time


def exotic_settings_widget(init_from_json=None):
    """
    Generate a widget to enter settings from sample.
    """

    # We rely on some global variables:
    global to_fill, template_types
    # This dictionary will contain all of the widgets
    widget_list = {}

    # Each widget has the same layout for its description and its value/input
    layout_description = ipw.Layout(width='70%')
    layout_input = ipw.Layout(width='30%')

    # Maintain a separate dict of just the value widgets
    value_widget = {}

    # For exotic there are two templates, one for known exoplanets and one for
    # candidate exoplanets
    for template in template_types:
        value_widget[template] = {}
        widget_list[template] = []
        for k in to_fill[template]:
            for k2, v in to_fill[template][k].items():
                if isinstance(v, str):
                    input_widget = ipw.Text(value=v)
                elif isinstance(v, float):
                    if v >= 0:
                        input_widget = ipw.FloatText(value=v)
                    else:
                        input_widget = ipw.FloatText(value=v)
                input_widget.layout = layout_input
                hb = ipw.HBox([ipw.HTML(value=k2, layout=layout_description),
                               input_widget])
                try:
                    validator = validators[template][k2]
                except KeyError:
                    pass
                else:
                    kids = list(hb.children)
                    indicator = MyValid(value=False)
                    kids.append(indicator)
                    input_widget.observe(validator(indicator, value_widget),
                                         names='value')

                    hb.children = kids

                val_key = join_char.join([k, k2])
                value_widget[template][val_key] = input_widget
                widget_list[template].append(hb)

    hb2 = {}
    for template in template_types:
        hb2[template] = ipw.HBox([ipw.VBox(widget_list[template][:16],
                                           layout=ipw.Layout(padding='10px')),
                                  ipw.VBox(widget_list[template][16:])])

    select_planet_type = ipw.ToggleButtons(
        description='Known or candidate exoplanet?',
        options=template_types,
        style={'description_width': 'initial'}
    )

    lookup_link_text = dict(known='https://exoplanetarchive.ipac.caltech.edu/',
                            candidate='https://exofop.ipac.caltech.edu/tess/')

    lookup_link_html = {}

    for k, v in lookup_link_text.items():
        lookup_link_html[k] = ipw.HTML(
            f'<h3>For some information about this '
            f'object: <a href="{v}" target="_blank">{v}</a></h3>'
        )

    input_container = ipw.VBox()

    whole_thing = ipw.VBox(children=[select_planet_type, input_container])
    whole_thing.planet_type = select_planet_type
    whole_thing.value_widget = value_widget
    pre_reduced_file = join_char.join(['optional_info', 'Pre-reduced File:'])
    whole_thing.data_file_widget = {
        'candidate': value_widget['candidate'][pre_reduced_file],
        'known': value_widget['known'][pre_reduced_file]
    }

    def observe_select(change):
        input_container.children = [lookup_link_html[select_planet_type.value],
                                    hb2[select_planet_type.value]]

    select_planet_type.observe(observe_select, names='value')
    observe_select(select_planet_type.value)

    return whole_thing


def set_values_from_json_file(widget, json_file):
    with open(json_file) as f:
        input_values = json.load(f)

    planet_type = widget.planet_type.value
    for k, widget in widget.value_widget[planet_type].items():
        k1, k2 = k.split(join_char)
        widget.value = input_values[k1][k2]


def get_values_from_widget(key, whole_thing):
    for k, widget in whole_thing.value_widget[key].items():
        k1, k2 = k.split(join_char)
        template_json[key][k1][k2] = widget.value
    return template_json[key]


def generate_json_file_name(key, whole_thing):
    get_values_from_widget(key, whole_thing)
    user_info = 'user_info'
    planet = 'planetary_parameters'
    filter_key = 'Filter Name (aavso.org/filters)'
    date = template_json[key][user_info]['Observation date']
    planet_name = template_json[key][planet]['Planet Name']
    filter_name = template_json[key][user_info][filter_key]
    name = f'{planet_name}-{date}-{filter_name}'
    return name.replace(' ', '_') + '.json'
