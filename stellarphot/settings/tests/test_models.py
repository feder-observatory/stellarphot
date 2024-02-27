import json

import astropy.units as u
import pytest
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time
from pydantic import ValidationError

from stellarphot.settings import ui_generator
from stellarphot.settings.models import (
    Camera,
    Exoplanet,
    LoggingSettings,
    Observatory,
    PassbandMap,
    PassbandMapEntry,
    PhotometryApertures,
    PhotometryOptions,
    PhotometrySettings,
    SourceLocationSettings,
)

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

DEFAULT_OBSERVATORY_SETTINGS = dict(
    name="test observatory",
    longitude=43 * u.deg,
    latitude=45 * u.deg,
    elevation=311 * u.m,
    AAVSO_code="test",
    TESS_telescope_code="tess test",
)

# The first setting here is required, the rest are optional. The optional
# settings below are different than the defaults in the model definition.
DEFAULT_PHOTOMETRY_OPTIONS = dict(
    include_dig_noise=False,
    reject_too_close=False,
    reject_background_outliers=False,
    fwhm_by_fit=False,
)

DEFAULT_PASSBAND_MAP = dict(
    your_filter_names_to_aavso=[
        PassbandMapEntry(
            your_filter_name="V",
            aavso_filter_name="V",
        ),
        PassbandMapEntry(
            your_filter_name="B",
            aavso_filter_name="B",
        ),
        PassbandMapEntry(
            your_filter_name="rp",
            aavso_filter_name="SR",
        ),
    ]
)

DEFAULT_LOGGING_SETTINGS = dict(
    logfile="test.log",
    console_log=False,
)

DEFAULT_SOURCE_LOCATION_SETTINGS = dict(
    shift_tolerance=5,
    source_list_file="test.ecsv",
    use_coordinates="pixel",
)

DEFAULT_PHOTOMETRY_SETTINGS = dict(
    camera=Camera(**TEST_CAMERA_VALUES),
    observatory=Observatory(**DEFAULT_OBSERVATORY_SETTINGS),
    photometry_apertures=PhotometryApertures(**DEFAULT_APERTURE_SETTINGS),
    source_locations=SourceLocationSettings(**DEFAULT_SOURCE_LOCATION_SETTINGS),
    photometry_options=PhotometryOptions(**DEFAULT_PHOTOMETRY_OPTIONS),
    passband_map=PassbandMap(**DEFAULT_PASSBAND_MAP),
    logging_settings=LoggingSettings(**DEFAULT_LOGGING_SETTINGS),
)


@pytest.mark.parametrize(
    "model,settings",
    [
        [Camera, TEST_CAMERA_VALUES],
        [PhotometryApertures, DEFAULT_APERTURE_SETTINGS],
        [Exoplanet, DEFAULT_EXOPLANET_SETTINGS],
        [Observatory, DEFAULT_OBSERVATORY_SETTINGS],
        [PhotometryOptions, DEFAULT_PHOTOMETRY_OPTIONS],
        [PassbandMap, DEFAULT_PASSBAND_MAP],
        [PhotometrySettings, DEFAULT_PHOTOMETRY_SETTINGS],
        [LoggingSettings, DEFAULT_LOGGING_SETTINGS],
        [SourceLocationSettings, DEFAULT_SOURCE_LOCATION_SETTINGS],
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
        for k, v in settings.items():
            assert getattr(mod, k) == v

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
        if model == PhotometrySettings or model == PassbandMap:
            pytest.xfail(
                reason="PassbandMap needs a dict widget -- https://github.com/feder-observatory/stellarphot/issues/274"
            )
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
        for k, v in settings.items():
            assert getattr(mod, k) == v

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


def test_camera_incompatible_dark_units():
    camera_for_test = TEST_CAMERA_VALUES.copy()
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


@pytest.mark.parametrize("bad_one", ["radius", "gap", "annulus_width"])
def test_create_invalid_values(bad_one):
    # Check that individual values that are bad raise an error
    bad_settings = DEFAULT_APERTURE_SETTINGS.copy()
    bad_settings[bad_one] = -1
    with pytest.raises(ValidationError, match=bad_one):
        PhotometryApertures(**bad_settings)


def test_observatory_earth_location():
    # Check that the earth location is correctly set
    obs = Observatory(**DEFAULT_OBSERVATORY_SETTINGS)
    earth_loc = EarthLocation(
        lat=DEFAULT_OBSERVATORY_SETTINGS["latitude"],
        lon=DEFAULT_OBSERVATORY_SETTINGS["longitude"],
        height=DEFAULT_OBSERVATORY_SETTINGS["elevation"],
    )
    assert obs.earth_location == earth_loc


def test_observatory_lat_long_as_float():
    # To make it easier to enter latitude and longitude in a form (e.g. a GUI),
    # we allow them to be entered as floats with an assumed unit of degrees,
    # not just Quantity objects.
    settings = dict(DEFAULT_OBSERVATORY_SETTINGS)
    settings["latitude"] = settings["latitude"].value
    settings["longitude"] = settings["longitude"].value
    obs = Observatory(**settings)
    assert obs == Observatory(**DEFAULT_OBSERVATORY_SETTINGS)


def test_source_locations_negative_shift_tolerance():
    # Check that a negative shift tolerance raises an error
    settings = dict(DEFAULT_SOURCE_LOCATION_SETTINGS)
    settings["shift_tolerance"] = -1
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        SourceLocationSettings(**settings)


class TestPassbandMapDictMethods:
    """Test all of the dict methods we implement for the PassbandMap class."""

    def create_passband_map(self):
        return PassbandMap(**DEFAULT_PASSBAND_MAP)

    def default_pb_map_keys(self):
        return [
            v.your_filter_name
            for v in DEFAULT_PASSBAND_MAP["your_filter_names_to_aavso"]
        ]

    def default_pb_map_values(self):
        return [
            v.aavso_filter_name.value
            for v in DEFAULT_PASSBAND_MAP["your_filter_names_to_aavso"]
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
    pb_map = PassbandMap(your_filter_names_to_aavso=None)
    assert pb_map._dict == {}


def test_passband_map_init_with_passband_map():
    pb_map = PassbandMap(**DEFAULT_PASSBAND_MAP)
    pb_map2 = PassbandMap(your_filter_names_to_aavso=pb_map)
    assert pb_map == pb_map2


def test_create_invalid_exoplanet():
    # Set some bad values and make sure they raise validation errors
    values = DEFAULT_EXOPLANET_SETTINGS.copy()
    # Make pediod and duration have invalid units for a time
    values["period"] = values["period"].value * u.m
    values["duration"] = values["duration"].value * u.m
    # Check that individual values that are bad raise an error
    with pytest.raises(ValidationError, match="2 validation errors"):
        Exoplanet(**values)
