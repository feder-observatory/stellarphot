import json
import random
import re
from copy import deepcopy

import astropy.units as u
import pytest
from astropy.coordinates import EarthLocation, Latitude, Longitude
from astropy.table import Table
from pydantic import ValidationError

from stellarphot.settings import ui_generator
from stellarphot.settings.constants import (
    TEST_APERTURE_SETTINGS,
    TEST_CAMERA_VALUES,
    TEST_EXOPLANET_SETTINGS,
    TEST_LOGGING_SETTINGS,
    TEST_OBSERVATORY_SETTINGS,
    TEST_PASSBAND_MAP,
    TEST_PHOTOMETRY_OPTIONS,
    TEST_PHOTOMETRY_SETTINGS,
    TEST_SOURCE_LOCATION_SETTINGS,
)
from stellarphot.settings.models import (
    Camera,
    Exoplanet,
    FwhmMethods,
    LoggingSettings,
    Observatory,
    PartialPhotometrySettings,
    PassbandMap,
    PassbandMapEntry,
    PhotometryApertures,
    PhotometryOptionalSettings,
    PhotometrySettings,
    SourceLocationSettings,
)

# NOTE: ALWAYS USE DEEPCOPY ON THESE DICTS SO TESTS DO NOT MODIFY THEM
TEST_CAMERA_VALUES = deepcopy(TEST_CAMERA_VALUES)
TEST_APERTURE_SETTINGS = deepcopy(TEST_APERTURE_SETTINGS)
TEST_EXOPLANET_SETTINGS = deepcopy(TEST_EXOPLANET_SETTINGS)
TEST_OBSERVATORY_SETTINGS = deepcopy(TEST_OBSERVATORY_SETTINGS)
TEST_PHOTOMETRY_OPTIONS = deepcopy(TEST_PHOTOMETRY_OPTIONS)
TEST_PASSBAND_MAP = deepcopy(TEST_PASSBAND_MAP)
TEST_PHOTOMETRY_SETTINGS = deepcopy(TEST_PHOTOMETRY_SETTINGS)
TEST_LOGGING_SETTINGS = deepcopy(TEST_LOGGING_SETTINGS)
TEST_SOURCE_LOCATION_SETTINGS = deepcopy(TEST_SOURCE_LOCATION_SETTINGS)


@pytest.mark.parametrize(
    "model,settings",
    [
        [Camera, TEST_CAMERA_VALUES],
        [PhotometryApertures, TEST_APERTURE_SETTINGS],
        [Exoplanet, TEST_EXOPLANET_SETTINGS],
        [Observatory, TEST_OBSERVATORY_SETTINGS],
        [PhotometryOptionalSettings, TEST_PHOTOMETRY_OPTIONS],
        [PassbandMap, TEST_PASSBAND_MAP],
        [PhotometrySettings, TEST_PHOTOMETRY_SETTINGS],
        [LoggingSettings, TEST_LOGGING_SETTINGS],
        [SourceLocationSettings, TEST_SOURCE_LOCATION_SETTINGS],
    ],
)
class TestModelAgnosticActions:
    """
    Collect all of the tests which don't depend on the details of the model
    in one place.
    """

    def test_create_model(self, model, settings):
        # Make sure we can create the model and that the settings are correct.
        mod = model(**settings)
        mod_dict = mod.model_dump()
        for k, v in settings.items():
            assert mod_dict[k] == v
            # if k == "your_filter_names_to_aavso":
            #     # This is the only nested model, so we need to check it separately
            #     assert getattr(mod, k) == [PassbandMapEntry(**x) for x in v]
            # else:
            #     assert getattr(mod, k) == v

    def test_model_copy(self, model, settings):
        # Make sure we can create a copy of the model
        mod = model(**settings)
        mod2 = mod.model_copy()
        assert mod2 == mod

    def tests_model_schema(self, model, settings):
        # Check that we can generate a model schema and that it has the right
        # number of properties -- the schema describes the type but doesn't contain
        # any values.
        mod = model(**settings)
        schema = mod.model_json_schema()
        assert len(schema["properties"]) == len(settings)

    def test_model_json_tround_trip(self, model, settings):
        # Make sure that serializing to json and back gives us the same model
        mod = model(**settings)
        mod2 = model.model_validate_json(mod.model_dump_json())
        assert mod2 == mod

    def test_model_table_round_trip(self, model, settings, tmp_path):
        # Make sure that we can write the model to a table metadata and read it back in
        mod = model(**settings)
        table = Table({"data": [1, 2, 3]})
        table.meta["model"] = mod
        table_path = tmp_path / "test_table.ecsv"
        print(f"{mod=}")
        table.write(table_path)
        new_table = Table.read(table_path)
        assert new_table.meta["model"] == mod

    def test_settings_ui_generation(self, model, settings):
        # Check a few things about the UI generation:
        # 1) The UI is generated
        # 2) The UI model matches our input
        # 3) The UI widgets contains the titles we expect
        #
        instance = model(**settings)
        instance.model_json_schema()
        # 1) The UI is generated from the class
        ui = ui_generator(model)

        # 2) The UI model matches our input
        # Set the ui values to the defaults -- the value needs to be whatever would
        # go into a **widget** though, not a **model**. It is easiest to create
        # a model and then use its dict() method to get the widget values.
        values_dict_as_strings = json.loads(model(**settings).model_dump_json())
        ui.value = values_dict_as_strings
        mod = model(**ui.value)
        mod_dict = mod.model_dump()
        for k, v in settings.items():
            assert mod_dict[k] == v
            # if k == "your_filter_names_to_aavso":
            #     # This is the only nested model, so we need to check it separately
            #     assert getattr(mod, k) == [PassbandMapEntry(**x) for x in v]
            # else:
            #     assert getattr(mod, k) == v

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
            f: [f.replace("_", " "), f.replace("_", "")] for f in settings.keys()
        }
        # Find any title that were explicitly set in the model definition via Field
        explicit_titles = {
            k: v.title for k, v in model.model_fields.items() if v.title is not None
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
                    if title in explicit_titles:
                        present = explicit_titles[title].lower() in label.lower()
                        if present:
                            title_present.append(present)
                            break
            else:
                title_present.append(False)

        assert all(title_present)


@pytest.mark.parametrize(
    "model,settings",
    [
        [Camera, deepcopy(TEST_CAMERA_VALUES)],
        [Observatory, deepcopy(TEST_OBSERVATORY_SETTINGS)],
        [PassbandMap, deepcopy(TEST_PASSBAND_MAP)],
    ],
)
class TestModelsWithName:
    """
    Tests that are specific to models that have a name property.
    """

    @pytest.mark.parametrize(
        "bad_name,error_msg",
        [
            ("", "name must not be empty or contain only whitespace"),
            (" ", "name must not be empty or contain only whitespace"),
            ("  ", "name must not be empty or contain only whitespace"),
            (
                "name with trailing spaces ",
                "name must not have leading or trailing whitespace",
            ),
            (
                " name with leading spaces",
                "name must not have leading or trailing whitespace",
            ),
        ],
    )
    def test_name_cannot_have_awkward_whitespace(
        self, model, settings, bad_name, error_msg
    ):
        settings["name"] = bad_name
        with pytest.raises(ValidationError, match=error_msg):
            model(**settings)

    def test_name_unicode_is_ok(self, model, settings):
        # Test that the name field can be unicode
        settings["name"] = "π"
        assert model(**settings).name == "π"


# Only include models here that have examples that should be tested
@pytest.mark.parametrize(
    "model,settings",
    [
        [Camera, TEST_CAMERA_VALUES],
        [Observatory, TEST_OBSERVATORY_SETTINGS],
    ],
)
class TestModelExamples:
    """ "
    Test that you can make a valid model from the examples. The assumption is that
    all of the first choices in the examples make a valid model, all of the second
    choices make a valid model, etc.

    The purpose for including this test is that users may use the examples as guidance
    so we should make sure the guidance isn't nonsense.
    """

    def test_example(self, model, settings):
        # Get the model's fields so that we can get their examples. fields is dict
        # with the field names as keys and the field objects as values.
        fields = model.model_fields

        examples = {k: f.examples for k, f in fields.items()}
        example_lengths = set(len(e) for e in examples.values() if e is not None)

        # We can't handle more than two different example lengths in an unambiguous way,
        # so we raise an error if we have more than two.
        if len(example_lengths) > 2:
            raise ValueError(f"Too many different example lengths for {model.__name__}")
        elif min(example_lengths) > 1 and len(example_lengths) == 2:
            raise ValueError(
                "Must have the same number of examples for all fields "
                "or one example for some fields and the same number for "
                "the rest."
            )
        max_len = max(example_lengths)
        for k in examples.keys():
            if examples[k] is None:
                examples[k] = [None] * max_len
            elif len(examples[k]) == 1:
                examples[k] = examples[k] * max_len

        for i in range(max_len):
            settings = {k: examples[k][i] for k in examples.keys()}

            mod = model(**settings)

            # Really need to compare some fields as
            # latitude/longitude/quantities/numbers but don't want to hard code that
            # here.
            for k, v in settings.items():
                model_value = getattr(mod, k)

                # For some foolish reason Observatory allows the latitude and longitude
                # to be entered as floats, which we assume are intended to have unit of
                # degrees. Test and handle that case...
                print(k, v)
                if k.lower() in ["latitude", "longitude"] and re.match(
                    r"[+-]?\d+\.\d+$", v
                ):
                    v = v + " degree"

                # Also, latitude and longitude are not Quantity, so handle that too
                if k == "latitude":
                    v = Latitude(v)
                elif k == "longitude":
                    v = Longitude(v)

                if isinstance(model_value, u.Quantity):
                    assert model_value == u.Quantity(v)
                elif isinstance(model_value, u.UnitBase):
                    assert model_value == u.Unit(v)
                else:
                    assert model_value == v
        assert "fwhm_method" in TEST_PHOTOMETRY_OPTIONS


class TestPriorVersionsCompatibility:
    """
    Make sure that all prior versions of settings files are still readable
    by the current version of the code.

    Each method in this test class should contain the release version number
    of the version it is checking for compatibility.
    """

    @pytest.mark.parametrize(
        "old_setting,new_setting",
        (
            [True, FwhmMethods.FIT],
            [False, FwhmMethods.PROFILE],
        ),
    )
    def test_2_0_0a11(self, old_setting, new_setting):
        """
        This version had fwhm_by_fit, which has now changed to fwhm_method.
        """
        old_settings_style = deepcopy(TEST_PHOTOMETRY_SETTINGS)
        # Version 2.0.0a11 had fwhm_by_fit instead of fwhm_method
        # so delete the new entry
        print(old_settings_style["photometry_optional_settings"].keys())
        del old_settings_style["photometry_optional_settings"]["fwhm_method"]

        # Add the old entry, which was a boolean. Later in the test that the
        # old_setting is properly mapped to the expected new_setting
        old_settings_style["photometry_optional_settings"]["fwhm_by_fit"] = old_setting

        # Check that the old settings style can be read in -- no error means all is well
        settings = PhotometrySettings(**old_settings_style)

        # Check that that the new settings style is correct given the old_setting
        assert settings.photometry_optional_settings.fwhm_method == new_setting


def test_partial_photometry_settings():
    """
    Test that we can create a PhotometrySettings object with only a subset of
    the normally required fields.
    """
    # Loop over the individual default photometry settings and make sure we can
    # create a PartialPhotometrySettings object with just that field.

    for k, v in TEST_PHOTOMETRY_SETTINGS.items():
        pps = PartialPhotometrySettings(**{k: v})
        assert pps.model_dump()[k] == v

    choices = list(TEST_PHOTOMETRY_SETTINGS.items())
    for i in range(2, 5):
        # Try a few random subsets of fields
        fields = random.choices(choices, k=i)
        settings = {k: v for k, v in fields}
        pps = PartialPhotometrySettings(**settings)
        for k, v in settings.items():
            assert pps.model_dump()[k] == v


def test_camera_unitscheck():
    # Check that the units are checked properly

    # Set a clearly incorrect Quantity. Simply removing the units does not lead
    # to an invalid Quantity -- it turns out Quantity(5) is valid, with units of
    # dimensionless_unscaled. So we need to set the units to something that is
    # invalid.
    camera_dict_bad_unit = {
        k: "5 cows" if k not in ["name", "data_unit"] else v
        for k, v in TEST_CAMERA_VALUES.items()
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
    camera_for_test = deepcopy(TEST_CAMERA_VALUES)
    camera_for_test["max_data_value"] = -1 * u.Quantity(
        camera_for_test["max_data_value"]
    )

    # Make sure that a negative max_adu raises an error
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        Camera(
            **camera_for_test,
        )


def test_camera_incompatible_gain_units():
    camera_for_test = deepcopy(TEST_CAMERA_VALUES)
    # Gain unit is incompatible with noise unit (electrons vs. counts)
    camera_for_test["gain"] = 2.0 * u.count / u.adu

    # Make sure that an incompatible gain raises an error
    with pytest.raises(ValidationError, match="Gain units.*not compatible"):
        Camera(
            **camera_for_test,
        )


def test_camera_unitsless_gain():
    # Regression test for #299
    c = Camera(
        name="should_work",
        data_unit="electron",
        gain="1",
        dark_current="0.01 electron / second",
        read_noise="1.2 electron",
        pixel_scale="0.55 arcsec / pix",
        max_data_value="80000 electron",
    )

    assert c.gain == u.Quantity(1)


def test_camera_incompatible_max_val_units():
    camera_for_test = TEST_CAMERA_VALUES
    # data unit is adu, not count
    camera_for_test["max_data_value"] = 50000 * u.count

    # Make sure that an incompatible gain raises an error
    with pytest.raises(
        ValidationError, match="Maximum data value units.*not consistent"
    ):
        Camera(
            **camera_for_test,
        )


def test_camera_incompatible_dark_units():
    camera_for_test = TEST_CAMERA_VALUES
    # Dark current unit is incompatible with gain unit (electrons vs. counts)
    camera_for_test["dark_current"] = 0.01 * u.count / u.second

    # Make sure that an incompatible gain raises an error
    with pytest.raises(ValidationError, match="Dark current units.*not compatible"):
        Camera(
            **camera_for_test,
        )


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


def test_create_aperture_settings_correctly():
    ap_set = PhotometryApertures(**TEST_APERTURE_SETTINGS)
    assert ap_set.radius == TEST_APERTURE_SETTINGS["radius"]
    assert (
        ap_set.inner_annulus
        == TEST_APERTURE_SETTINGS["radius"] + TEST_APERTURE_SETTINGS["gap"]
    )
    assert (
        ap_set.outer_annulus
        == TEST_APERTURE_SETTINGS["radius"]
        + TEST_APERTURE_SETTINGS["gap"]
        + TEST_APERTURE_SETTINGS["annulus_width"]
    )


@pytest.mark.parametrize("bad_one", ["radius", "gap", "annulus_width"])
def test_create_invalid_values(bad_one):
    # Check that individual values that are bad raise an error
    bad_settings = TEST_APERTURE_SETTINGS
    bad_settings[bad_one] = -1
    with pytest.raises(ValidationError, match=bad_one):
        PhotometryApertures(**bad_settings)


def test_observatory_earth_location():
    # Check that the earth location is correctly set
    obs = Observatory(**TEST_OBSERVATORY_SETTINGS)
    earth_loc = EarthLocation(
        lat=TEST_OBSERVATORY_SETTINGS["latitude"],
        lon=TEST_OBSERVATORY_SETTINGS["longitude"],
        height=TEST_OBSERVATORY_SETTINGS["elevation"],
    )
    assert obs.earth_location == earth_loc


def test_observatory_lat_long_as_float():
    # To make it easier to enter latitude and longitude in a form (e.g. a GUI),
    # we allow them to be entered as floats with an assumed unit of degrees,
    # not just Quantity objects.
    settings = dict(TEST_OBSERVATORY_SETTINGS)
    settings["latitude"] = u.Quantity(settings["latitude"]).value
    settings["longitude"] = u.Quantity(settings["longitude"]).value
    obs = Observatory(**settings)
    assert obs == Observatory(**TEST_OBSERVATORY_SETTINGS)


def test_source_locations_negative_shift_tolerance():
    # Check that a negative shift tolerance raises an error
    settings = dict(TEST_SOURCE_LOCATION_SETTINGS)
    settings["shift_tolerance"] = -1
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        SourceLocationSettings(**settings)


class TestPassbandMapDictMethods:
    """Test all of the dict methods we implement for the PassbandMap class."""

    def create_passband_map(self):
        return PassbandMap(**TEST_PASSBAND_MAP)

    def default_pb_map_keys(self):
        return [
            v["your_filter_name"]
            for v in TEST_PASSBAND_MAP["your_filter_names_to_aavso"]
        ]

    def default_pb_map_values(self):
        return [
            v["aavso_filter_name"]
            for v in TEST_PASSBAND_MAP["your_filter_names_to_aavso"]
        ]

    def test_keys(self):
        pb_map = self.create_passband_map()
        assert list(pb_map.keys()) == self.default_pb_map_keys()

    def test_values(self):
        pb_map = self.create_passband_map()
        assert list(pb_map.values()) == self.default_pb_map_values()

    def test_item_access(self):
        pb_map = self.create_passband_map()
        assert pb_map["rp"] == "SR"
        assert pb_map.get("rp") == "SR"
        assert pb_map.get("not a key", "foo") == "foo"

    def test_contains(self):
        pb_map = self.create_passband_map()
        assert "rp" in pb_map
        assert "not a key" not in pb_map

    def test_items(self):
        pb_map = self.create_passband_map()
        assert [k for k, v in pb_map.items()] == self.default_pb_map_keys()
        assert [v for k, v in pb_map.items()] == self.default_pb_map_values()

    def test_iteration(self):
        pb_map = self.create_passband_map()
        assert [k for k in pb_map] == self.default_pb_map_keys()

    def test_update_fails(self):
        pb_map = self.create_passband_map()
        with pytest.raises(TypeError, match="does not support item assignment"):
            pb_map["rp"] = "not a band"

    def test_deletion_fails(self):
        pb_map = self.create_passband_map()
        with pytest.raises(TypeError, match="does not support item deletion"):
            del pb_map["rp"]


def test_passband_map_init_with_none():
    with pytest.raises(ValidationError, match="1 validation error for PassbandMap"):
        PassbandMap(name="Test", your_filter_names_to_aavso=None)


def test_passband_map_init_with_passband_map():
    pb_map = PassbandMap(**TEST_PASSBAND_MAP)
    pb_map2 = PassbandMap(name="Example map", your_filter_names_to_aavso=pb_map)
    assert pb_map == pb_map2


def test_passband_map_entry_empty_name_raises_error():
    # Name of your filter cannot be empty
    with pytest.raises(ValidationError, match="name must not be empty"):
        PassbandMapEntry(your_filter_name="", aavso_filter_name="V")


def test_create_invalid_exoplanet():
    # Set some bad values and make sure they raise validation errors
    values = TEST_EXOPLANET_SETTINGS
    # Make pediod and duration have invalid units for a time
    values["period"] = u.Quantity(values["period"]).value * u.m
    values["duration"] = u.Quantity(values["duration"]).value * u.m
    # Check that individual values that are bad raise an error
    with pytest.raises(ValidationError, match="2 validation errors"):
        Exoplanet(**values)
