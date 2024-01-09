from astrowidgets import ImageWidget

from stellarphot.gui_tools import seeing_profile_functions as spf


def test_keybindings():
    def simple_bindmap(bindmap):
        bound_keys = {}
        # The keys of the event map are...messy. This converts them to strings
        for key in bindmap.keys():
            modifier = key[1]
            key_name = key[2]
            bound_keys[str(key[0]) + "".join(modifier) + key_name] = key
        return bound_keys

    # This test assumes the ginga widget backend...
    iw = ImageWidget()
    original_bindings = iw._viewer.get_bindmap().eventmap

    bound_keys = simple_bindmap(original_bindings)
    # Spot check a couple of things before we run our function
    assert "Nonekp_D" in bound_keys
    assert "Nonekp_+" in bound_keys
    assert "Nonekp_left" not in bound_keys

    # rebind
    spf.set_keybindings(iw)
    new_bindings = iw._viewer.get_bindmap().eventmap
    bound_keys = simple_bindmap(new_bindings)
    assert "Nonekp_D" not in bound_keys
    assert "Nonekp_+" in bound_keys
    # Yes, the line below is correct...
    assert new_bindings[bound_keys["Nonekp_left"]]["name"] == "pan_right"
