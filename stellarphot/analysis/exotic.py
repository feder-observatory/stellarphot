from pathlib import Path
import json

import ipywidgets as ipw

from astropy.utils.data import get_pkg_data_filename

template_types = ['known', 'candidate']
template_json = {}
to_fill = {}

for template in template_types:
    template_name = get_pkg_data_filename('data/' + f'tic-template-for-exotic-{template}.json')
    with open(template_name) as f:
        template_json[template] = json.load(f)

    template_name = get_pkg_data_filename('data/' + f'exotic-to-mod-{template}.json')
    with open(Path(template_name)) as f:
        to_fill[template] = json.load(f)

exotic_arguments = dict(
    known=['--nasaexoarch', '--pre'],
    candidate=['--override', '--pre']
)

join_char = "😬"


def exotic_settings_widget():
    widget_list = {}

    layout_description = ipw.Layout(width='70%')
    layout_input = ipw.Layout(width='30%')
    value_widget = {}

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
                        input_widget = ipw.FloatText(max=0.0, min=-1e4, value=v)
                input_widget.layout = layout_input
                hb = ipw.HBox([ipw.HTML(value=k2, layout=layout_description), input_widget])
                val_key = join_char.join([k, k2])
                value_widget[template][val_key] = input_widget
                widget_list[template].append(hb)

    hb2 = {}
    for template in template_types:
        hb2[template] = ipw.HBox([ipw.VBox(widget_list[template][:16], layout=ipw.Layout(padding='10px')),
                                  ipw.VBox(widget_list[template][16:])])

    select_planet_type = ipw.ToggleButtons(description='Known or candidate exoplanet?',
                                           options=template_types,
                                           style = {'description_width': 'initial'})

    lookup_link_text = dict(known='https://exoplanetarchive.ipac.caltech.edu/',
                            candidate='https://exofop.ipac.caltech.edu/tess/')

    lookup_link_html = {}

    for k, v in lookup_link_text.items():
        lookup_link_html[k] = ipw.HTML(f'<h3>For some information about this object: <a href="{v}" target="_blank">{v}</a></h3>')

    input_container = ipw.VBox()

    whole_thing = ipw.VBox(children=[select_planet_type, input_container])
    whole_thing.planet_type = select_planet_type
    whole_thing.value_widget = value_widget

    def observe_select(change):
        input_container.children = [lookup_link_html[select_planet_type.value],
                                    hb2[select_planet_type.value]]

    select_planet_type.observe(observe_select, names='value')
    observe_select(select_planet_type.value)

    return whole_thing

whole_thing = exotic_settings_widget()

def get_values_from_widget(key):
    for k, widget in whole_thing.value_widget[key].items():
        k1, k2 = k.split(join_char)
        template_json[key][k1][k2] = widget.value
    return template_json[key]


def generate_json_file_name(key):
    get_values_from_widget(key)
    user_info = 'user_info'
    planet = 'planetary_parameters'
    date = template_json[key][user_info]['Observation date']
    planet_name = template_json[key][planet]['Planet Name']
    filter_name = template_json[key][user_info]['Filter Name (aavso.org/filters)']
    name = f'{planet_name}-{date}-{filter_name}'
    return name.replace(' ', '_') + '.json'
