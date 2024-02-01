import json

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from pydantic import ValidationError

from stellarphot.settings import ui_generator
from stellarphot.settings.models import Camera, PhotometryApertures

DEFAULT_APERTURE_SETTINGS = dict(radius=5, gap=10, annulus_width=15, fwhm=3.2)

TEST_CAMERA_VALUES = dict(
    data_unit=u.adu,
    gain=2.0 * u.electron / u.adu,
    name="test camera",
    read_noise=10 * u.electron,
    dark_current=0.01 * u.electron / u.second,
    pixel_scale=0.563 * u.arcsec / u.pix,
    max_data_value=50000 * u.adu,
)


DEFAULT_EXOPLANET_SETTINGS = dict(
    epoch=Time(0, format="jd"),
    period=0 * u.min,
    identifier="a planet",
    coordinate=SkyCoord(
        ra="00:00:00.00", dec="+00:00:00.0", frame="icrs", unit=("hour", "degree")
    ),
    depth=0,
    duration=0 * u.min,
)


def test_camera_attributes():
    # Check that the attributes are set properly
    c = Camera(
        **TEST_CAMERA_VALUES,
    )
    assert c.model_dump() == TEST_CAMERA_VALUES


def test_camera_unitscheck():
    # Check that the units are checked properly

    # Set a clearly incorrect Quantity. Simply removing the units does not lead
    # to an invalid Quantity -- it turns out Quantity(5) is valid, with units of
    # dimensionless_unscaled. So we need to set the units to something that is
    # invalid.
    camera_dict_bad_unit = {
        k: "5 cows" if hasattr(v, "value") else v for k, v in TEST_CAMERA_VALUES.items()
    }
    # All 5 of the attributes after data_unit will be checked for units
    # and noted in the ValidationError message. Rather than checking
    # separately for all 5, we just check for the presence of the
    # right number of errors, which is currently 20 -- 4 for each of the
    # 5 attributes, because of the union schema in _UnitTypePydanticAnnotation
    with pytest.raises(ValidationError, match="20 validation errors"):
        Camera(
            **camera_dict_bad_unit,
        )


def test_camera_negative_max_adu():
    # Check that a negative maximum data value raises an error
    camera_for_test = TEST_CAMERA_VALUES.copy()
    camera_for_test["max_data_value"] = -1 * camera_for_test["max_data_value"]

    # Make sure that a negative max_adu raises an error
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        Camera(
            **camera_for_test,
        )


def test_camera_incompatible_gain_units():
    camera_for_test = TEST_CAMERA_VALUES.copy()
    # Gain unit is incompatible with noise unit (electrons vs. counts)
    camera_for_test["gain"] = 2.0 * u.count / u.adu

    # Make sure that an incompatible gain raises an error
    with pytest.raises(ValidationError, match="Gain units.*not compatible"):
        Camera(
            **camera_for_test,
        )


def test_camera_incompatible_max_val_units():
    camera_for_test = TEST_CAMERA_VALUES.copy()
    # data unit is adu, not count
    camera_for_test["max_data_value"] = 50000 * u.count

    # Make sure that an incompatible gain raises an error
    with pytest.raises(
        ValidationError, match="Maximum data value units.*not consistent"
    ):
        Camera(
            **camera_for_test,
        )


def test_camera_copy():
    # Make sure copy actually copies everything

    c = Camera(
        **TEST_CAMERA_VALUES,
    )
    c2 = c.model_copy()
    assert c2 == c


def test_camera_altunitscheck():
    # Check to see that 'count' is allowed instead of 'electron'
    camera_for_test = dict(
        data_unit=u.adu,
        gain=2.0 * u.count / u.adu,
        name="test camera",
        read_noise=10 * u.count,
        dark_current=0.01 * u.count / u.second,
        pixel_scale=0.563 * u.arcsec / u.pix,
        max_data_value=50000 * u.adu,
    )

    c = Camera(
        **camera_for_test,
    )
    assert c.model_dump() == camera_for_test


def test_camera_schema():
    # Check that we can generate a schema for a Camera and that it
    # has the right number of attributes
    c = Camera(**TEST_CAMERA_VALUES)
    schema = c.model_json_schema()
    assert len(schema["properties"]) == len(TEST_CAMERA_VALUES)


def test_camera_json_round_trip():
    # Check that a camera can be converted to json and back

    c = Camera(**TEST_CAMERA_VALUES)

    c2 = Camera.model_validate_json(c.model_dump_json())
    assert c2 == c


def test_camera_table_round_trip(tmp_path):
    # Check that a camera can be stored as part of an astropy.table.Table
    # metadata and retrieved
    table = Table({"data": [1, 2, 3]})
    c = Camera(**TEST_CAMERA_VALUES)
    table.meta["camera"] = c
    table_path = tmp_path / "test_table.ecsv"
    table.write(table_path)
    new_table = Table.read(table_path)

    assert new_table.meta["camera"] == c


def test_create_aperture_settings_correctly():
    ap_set = PhotometryApertures(**DEFAULT_APERTURE_SETTINGS)
    assert ap_set.radius == DEFAULT_APERTURE_SETTINGS["radius"]
    assert (
        ap_set.inner_annulus
        == DEFAULT_APERTURE_SETTINGS["radius"] + DEFAULT_APERTURE_SETTINGS["gap"]
    )
    assert (
        ap_set.outer_annulus
        == DEFAULT_APERTURE_SETTINGS["radius"]
        + DEFAULT_APERTURE_SETTINGS["gap"]
        + DEFAULT_APERTURE_SETTINGS["annulus_width"]
    )


# Right now Exoplanet doesn't have a schema, so don't test it. Will
# fix after the pydantic 2 transition.
# [Exoplanet, DEFAULT_EXOPLANET_SETTINGS]
@pytest.mark.parametrize(
    "class_, defaults",
    (
        [PhotometryApertures, DEFAULT_APERTURE_SETTINGS],
        [Camera, TEST_CAMERA_VALUES],
    ),
)
def test_aperture_settings_ui_generation(class_, defaults):
    # Check a few things about the UI generation:
    # 1) The UI is generated
    # 2) The UI model matches our input
    # 3) The UI widgets contains the titles we expect
    #
    instance = class_(**defaults)
    instance.model_json_schema()
    # 1) The UI is generated from the class
    ui = ui_generator(class_)

    # 2) The UI model matches our input
    # Set the ui values to the defaults -- the value needs to be whatever would
    # go into a **widget** though, not a **model**. It is easiest to create
    # a model and then use its dict() method to get the widget values.
    values_dict_as_strings = json.loads(class_(**defaults).model_dump_json())
    ui.value = values_dict_as_strings
    assert class_(**ui.value).model_dump() == defaults

    # 3) The UI widgets contains the titles generated from pydantic.
    # Pydantic generically is supposed to generate titles from the field names,
    # replacing "_" with " " and capitalizing the first letter.
    #
    # In fact, ipyautoui pre-pydantic-2 seems to either use the field name,
    # the space-replaced name, or a name with the underscore just removed,
    # not replaced by a space.
    # Hopefully that improves in future versions, but for now we'll just
    # check that the titles are present in the labels.
    # We'll ignore the case but need to replace the underscores
    pydantic_titles = {
        f: [f.replace("_", " "), f.replace("_", "")] for f in defaults.keys()
    }
    title_present = []

    for title in pydantic_titles.keys():
        for box in ui.di_boxes.values():
            label = box.html_title.value
            present = (
                title.lower() in label.lower()
                or pydantic_titles[title][0].lower() in label.lower()
                or pydantic_titles[title][1].lower() in label.lower()
            )
            if present:
                title_present.append(present)
                break
        else:
            title_present.append(False)

    assert all(title_present)


@pytest.mark.parametrize("bad_one", ["radius", "gap", "annulus_width"])
def test_create_invalid_values(bad_one):
    # Check that individual values that are bad raise an error
    bad_settings = DEFAULT_APERTURE_SETTINGS.copy()
    bad_settings[bad_one] = -1
    with pytest.raises(ValidationError, match=bad_one):
        PhotometryApertures(**bad_settings)


# def test_create_exoplanet_correctly():
#     planet = Exoplanet(**DEFAULT_EXOPLANET_SETTINGS)
#     print(planet)
#     assert planet.epoch == DEFAULT_EXOPLANET_SETTINGS["epoch"]
#     assert u.get_physical_type(planet.period) == "time"
#     assert planet.identifier == DEFAULT_EXOPLANET_SETTINGS["identifier"]
#     assert planet.coordinate == DEFAULT_EXOPLANET_SETTINGS["coordinate"]
#     assert planet.depth == DEFAULT_EXOPLANET_SETTINGS["depth"]
#     assert u.get_physical_type(planet.duration) == "time"


# def test_create_invalid_exoplanet():
#     values = DEFAULT_EXOPLANET_SETTINGS.copy()
#     # Make pediod and duration have invalid units for a time
#     values["period"] = values["period"].value * u.m
#     values["duration"] = values["duration"].value * u.m
#     # Check that individual values that are bad raise an error
#     with pytest.raises(ValidationError, match="2 validation errors"):
#         Exoplanet(**values)
