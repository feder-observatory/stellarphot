from stellarphot.transit_fitting.gui import (
    exotic_settings_widget,
    get_values_from_widget,
)


def test_json_has_stuff():
    widget = exotic_settings_widget()
    stuff = get_values_from_widget(widget, key="known")
    assert "user_info" in stuff.keys()
