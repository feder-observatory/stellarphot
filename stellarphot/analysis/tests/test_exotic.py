from stellarphot.analysis.exotic import exotic_settings_widget, get_values_from_widget


def test_json_has_stuff():
    stuff = get_values_from_widget('known')
    assert 'user_info' in stuff.keys()
