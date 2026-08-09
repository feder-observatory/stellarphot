import json
import random
import re
import warnings
from copy import deepcopy
from pathlib import Path

import astropy.units as u
import pytest
from astropy.coordinates import EarthLocation, Latitude, Longitude
from astropy.table import Table, TableAttribute
from astropy.utils.data import get_pkg_data_path
from pydantic import ValidationError

from stellarphot import BaseEnhancedTable
from stellarphot.gui.views import ui_generator
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
    PHOTOMETRY_SETTINGS_FORMAT_VERSION,
    VARIABLE_APERTURE_DEFAULTS,
    Camera,
    Exoplanet,
    FwhmMethods,
    LoggingSettings,
    NewerFormatError,
    Observatory,
    PartialPhotometrySettings,
    PassbandMap,
    PassbandMapEntry,
    PhotometryApertures,
    PhotometryOptionalSettings,
    PhotometrySettings,
    PhotometrySettingsMigrationWarning,
    PhotometrySettingsWarning,
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


# Class below is used in testing roundtripping when a model is a
# table attribute.
class TableWithAttribute(BaseEnhancedTable):
    model = TableAttribute()


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
        # as long as we are using BaseEnhancedTable or a subclass.
        mod = model(**settings)
        table = BaseEnhancedTable({"data": [1, 2, 3]})
        table.meta["model"] = mod
        table_path = tmp_path / "test_table.ecsv"
        table.write(table_path)
        new_table = BaseEnhancedTable.read(table_path)
        assert new_table.meta["model"] == mod

    def test_plain_table_readability(self, model, settings, tmp_path):
        # Make sure that we can write the model to a table metadata and read it back in
        # as long as we are use BaseEnhancedTable or a subclass.
        mod = model(**settings)
        table = BaseEnhancedTable({"data": [1, 2, 3]})
        table.meta["model"] = mod
        table_path = tmp_path / "test_table.ecsv"
        print(f"{mod=}")
        table.write(table_path)
        new_table = Table.read(table_path)
        assert mod.__class__.__name__ == new_table.meta["model"]["_model_name"]

    def test_table_roundtrip_model_as_attribute(self, model, settings, tmp_path):
        # If a model is a table attribute it is saved in the meta in a
        # dictionary whose key is __attributes__. The prior tests check
        # that a model that is directly in the meta can round trip but
        # does not check a model that is an attribute so we do that here.
        the_table = TableWithAttribute()
        the_table.model = model(**settings)
        table_path = tmp_path / "test_table.ecsv"
        the_table.write(table_path)
        new_table = TableWithAttribute.read(table_path)
        # Check that the model is the same
        assert new_table.model == the_table.model

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


def test_camera_json_round_trip_preserves_units():
    # test_model_json_tround_trip above only checks model equality, which
    # would still pass if a value came back with a unit that is merely
    # equivalent (rather than identical) to the original -- e.g. a gain of
    # "2.0 electron / adu" being silently converted to some other, numerically
    # equal, combination of units. Check the units explicitly.
    camera = Camera(**deepcopy(TEST_CAMERA_VALUES))
    camera2 = Camera.model_validate_json(camera.model_dump_json())

    assert camera2.gain.unit == u.electron / u.adu
    assert camera2.gain.value == 2.0
    assert camera2.read_noise.unit == u.electron
    assert camera2.read_noise.value == 10.0
    assert camera2.dark_current.unit == u.electron / u.s
    assert camera2.dark_current.value == 0.01
    assert camera2.pixel_scale.unit == u.arcsec / u.pix
    assert camera2.pixel_scale.value == 0.563
    assert camera2.max_data_value.unit == u.adu
    assert camera2.max_data_value.value == 50000.0
    assert camera2.data_unit == u.adu


def test_observatory_json_round_trip_preserves_units():
    # As above, but for Observatory's latitude/longitude/elevation, which
    # are not plain Quantity objects (Latitude/Longitude/Quantity
    # respectively), so are worth checking on their own.
    observatory = Observatory(**deepcopy(TEST_OBSERVATORY_SETTINGS))
    observatory2 = Observatory.model_validate_json(observatory.model_dump_json())

    assert observatory2.latitude.unit == u.degree
    assert observatory2.latitude.value == 45.0
    assert observatory2.longitude.unit == u.degree
    assert observatory2.longitude.value == 43.0
    assert observatory2.elevation.unit == u.m
    assert observatory2.elevation.value == 311.0


def test_camera_table_round_trip_preserves_units(tmp_path):
    # Companion to test_camera_json_round_trip_preserves_units above, but
    # for the table-metadata round trip (see test_model_table_round_trip).
    camera = Camera(**deepcopy(TEST_CAMERA_VALUES))
    table = BaseEnhancedTable({"data": [1, 2, 3]})
    table.meta["model"] = camera
    table_path = tmp_path / "test_table.ecsv"
    table.write(table_path)
    new_table = BaseEnhancedTable.read(table_path)
    camera2 = new_table.meta["model"]

    assert camera2.gain.unit == u.electron / u.adu
    assert camera2.gain.value == 2.0
    assert camera2.max_data_value.unit == u.adu
    assert camera2.max_data_value.value == 50000.0


def test_observatory_table_round_trip_preserves_units(tmp_path):
    # Companion to test_observatory_json_round_trip_preserves_units above,
    # but for the table-metadata round trip.
    observatory = Observatory(**deepcopy(TEST_OBSERVATORY_SETTINGS))
    table = BaseEnhancedTable({"data": [1, 2, 3]})
    table.meta["model"] = observatory
    table_path = tmp_path / "test_table.ecsv"
    table.write(table_path)
    new_table = BaseEnhancedTable.read(table_path)
    observatory2 = new_table.meta["model"]

    assert observatory2.latitude.unit == u.degree
    assert observatory2.latitude.value == 45.0
    assert observatory2.longitude.unit == u.degree
    assert observatory2.longitude.value == 43.0
    assert observatory2.elevation.unit == u.m
    assert observatory2.elevation.value == 311.0


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

    def test_migration_warning_is_a_photometry_settings_warning(self):
        # The ReviewSettings banner displays warnings of the
        # PhotometrySettingsWarning category; the migration warning must be
        # in that category or the banner will never show it.
        assert issubclass(PhotometrySettingsMigrationWarning, PhotometrySettingsWarning)

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

        # Add the old entry, which was a boolean. Later in the test check that the
        # old_setting is properly mapped to the expected new_setting
        old_settings_style["photometry_optional_settings"]["fwhm_by_fit"] = old_setting

        # Check that the old settings style can be read in -- no error means all is well
        settings = PhotometrySettings(**old_settings_style)

        # Check that that the new settings style is correct given the old_setting
        assert settings.photometry_optional_settings.fwhm_method == new_setting

    def test_reading_2_0_0_alpha_files(self):
        """
        Test that we can read the settings files from
        2.0.0 alpha releases.
        """
        settings_file = Path(
            get_pkg_data_path("data/sample_photometry_settings_2.0.0alpha.json")
        )

        phot_settings = PhotometrySettings.model_validate_json(
            settings_file.read_text()
        )

        assert hasattr(
            phot_settings.photometry_optional_settings, "partial_pixel_method"
        )

        # These files predate the settings_version field; validated directly
        # (i.e. not through PhotometryWorkingDirSettings.load) they get the
        # current version by default.
        assert phot_settings.settings_version == PHOTOMETRY_SETTINGS_FORMAT_VERSION

    @staticmethod
    def _format_1_settings(variable_aperture):
        """
        Settings dict as written by stellarphot before the settings_version
        field existed, i.e. settings format 1.
        """
        old_style = deepcopy(TEST_PHOTOMETRY_SETTINGS)
        old_style.pop("settings_version", None)
        old_style["photometry_apertures"]["variable_aperture"] = variable_aperture
        return old_style

    def test_settings_version_defaults_to_current(self):
        # Settings constructed without an explicit settings_version -- the
        # normal case in code, since only files carry the field -- must get
        # the current format version and must write it out on dump, so that
        # every file the current code saves is marked with the format it was
        # written in.
        settings = PhotometrySettings(**self._format_1_settings(False))
        assert settings.settings_version == PHOTOMETRY_SETTINGS_FORMAT_VERSION
        dumped = json.loads(settings.model_dump_json())
        assert dumped["settings_version"] == PHOTOMETRY_SETTINGS_FORMAT_VERSION

    def test_partial_settings_version_not_none(self):
        # Unlike the other fields, settings_version must NOT default to None
        # in the partial model: a null version in a saved partial file would
        # make it look like a format 1 file.
        pps = PartialPhotometrySettings()
        assert pps.settings_version == PHOTOMETRY_SETTINGS_FORMAT_VERSION
        dumped = json.loads(pps.model_dump_json())
        assert dumped["settings_version"] == PHOTOMETRY_SETTINGS_FORMAT_VERSION

    def test_no_migration_of_unversioned_input_without_context(self):
        # Without the settings-file validation context, unversioned input --
        # e.g. settings constructed in code, or settings embedded in the
        # metadata of an old photometry table -- is not migrated.
        old_style = self._format_1_settings(True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            settings = PhotometrySettings.model_validate_json(json.dumps(old_style))
        apertures = settings.photometry_apertures
        assert apertures.gap == old_style["photometry_apertures"]["gap"]
        assert (
            apertures.annulus_width
            == old_style["photometry_apertures"]["annulus_width"]
        )
        assert settings.settings_version == PHOTOMETRY_SETTINGS_FORMAT_VERSION

    def test_migration_of_unversioned_variable_aperture_with_context(self):
        # The positive migration case at the model-validation level: with the
        # settings-file context, a format 1 file with variable_aperture=True
        # is migrated -- with a warning -- because the old variable-aperture
        # annulus geometry biased the photometry (issue #654).
        old_style = self._format_1_settings(True)
        with pytest.warns(PhotometrySettingsMigrationWarning, match="RESET"):
            settings = PhotometrySettings.model_validate_json(
                json.dumps(old_style), context={"settings_file": True}
            )
        apertures = settings.photometry_apertures
        # The user's aperture choice is kept...
        assert apertures.variable_aperture is True
        assert apertures.radius == old_style["photometry_apertures"]["radius"]
        assert (
            apertures.fwhm_estimate
            == old_style["photometry_apertures"]["fwhm_estimate"]
        )
        # ...but the annulus geometry is reset to the current defaults.
        assert apertures.gap == PhotometryApertures.model_fields["gap"].default
        assert (
            apertures.annulus_width
            == PhotometryApertures.model_fields["annulus_width"].default
        )
        assert settings.settings_version == PHOTOMETRY_SETTINGS_FORMAT_VERSION

    def test_no_migration_of_unversioned_fixed_aperture_with_context(self):
        # Fixed-aperture settings from format 1 have unchanged meaning, so
        # they load as-is with no warning.
        old_style = self._format_1_settings(False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            settings = PhotometrySettings.model_validate_json(
                json.dumps(old_style), context={"settings_file": True}
            )
        apertures = settings.photometry_apertures
        assert apertures.gap == old_style["photometry_apertures"]["gap"]
        assert (
            apertures.annulus_width
            == old_style["photometry_apertures"]["annulus_width"]
        )

    def test_newer_format_version_raises(self):
        # A settings_version newer than the code understands raises an error
        # on every validation path, with no context needed.
        newer = deepcopy(TEST_PHOTOMETRY_SETTINGS)
        newer["settings_version"] = PHOTOMETRY_SETTINGS_FORMAT_VERSION + 1
        with pytest.raises(NewerFormatError, match="newer version of stellarphot"):
            PhotometrySettings.model_validate_json(json.dumps(newer))
        # NewerFormatError must not be a ValueError, or pydantic would fold it
        # into a ValidationError inside the validator that raises it.
        assert not issubclass(NewerFormatError, ValueError)
        assert not issubclass(NewerFormatError, ValidationError)


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


class TestPhotometryApertureSettings:
    """
    Put tests specific to, and isolated to, `PhotometryAperture` settings here.
    """

    @pytest.mark.parametrize(
        "variable_aperture,radius",
        [
            (True, 1.5),
            (False, 5.0),
        ],
    )
    def test_create_aperture_settings_correctly(self, variable_aperture, radius):
        # Check that the inner and outer annulus are set correctly.
        ap_set = PhotometryApertures(**TEST_APERTURE_SETTINGS)
        assert ap_set.radius == TEST_APERTURE_SETTINGS["radius"]

        ap_set.variable_aperture = variable_aperture
        ap_set.radius = radius
        if variable_aperture:
            # In variable mode radius, gap and annulus_width are all
            # multiples of the FWHM. See #654.
            expected_inner = (
                radius + TEST_APERTURE_SETTINGS["gap"]
            ) * TEST_APERTURE_SETTINGS["fwhm_estimate"]
            expected_outer = (
                radius
                + TEST_APERTURE_SETTINGS["gap"]
                + TEST_APERTURE_SETTINGS["annulus_width"]
            ) * TEST_APERTURE_SETTINGS["fwhm_estimate"]
        else:
            expected_inner = radius + TEST_APERTURE_SETTINGS["gap"]
            expected_outer = (
                radius
                + TEST_APERTURE_SETTINGS["gap"]
                + TEST_APERTURE_SETTINGS["annulus_width"]
            )
        assert ap_set.inner_annulus == pytest.approx(expected_inner)
        assert ap_set.outer_annulus == pytest.approx(expected_outer)

    def test_create_aperture_settings_variable_aperture(self):
        # Check that the variable aperture flag is set correctly
        # and that radius_pixels is calculated correctly.
        settings = deepcopy(TEST_APERTURE_SETTINGS)
        settings["variable_aperture"] = True
        # The radius below is intended as a multiple of the FWHM
        settings["radius"] = 1.5
        ap_set = PhotometryApertures(**settings)
        assert ap_set.variable_aperture is True

        # Check that the radius in pixels is correct
        fwhm = 5.0
        radius_pix = ap_set.radius_pixels(fwhm)
        assert radius_pix == pytest.approx(7.5, rel=1e-6)

    @pytest.mark.parametrize(
        "variable_aperture,radius",
        [
            (True, 1.5),
            (False, 5.0),
        ],
    )
    def test_annulus_pixels_methods(self, variable_aperture, radius):
        # The annulus radii should follow the FWHM passed in, just like
        # radius_pixels does, so that in variable-aperture mode the annulus
        # can track the per-image FWHM. See #654.
        settings = deepcopy(TEST_APERTURE_SETTINGS)
        settings["variable_aperture"] = variable_aperture
        settings["radius"] = radius
        ap_set = PhotometryApertures(**settings)

        # Deliberately different from the settings' fwhm_estimate
        fwhm = 2 * settings["fwhm_estimate"]
        if variable_aperture:
            # radius, gap and annulus_width are all multiples of the FWHM
            # in variable mode. See #654.
            expected_inner = (radius + settings["gap"]) * fwhm
            expected_outer = expected_inner + settings["annulus_width"] * fwhm
        else:
            expected_inner = radius + settings["gap"]
            expected_outer = expected_inner + settings["annulus_width"]

        assert ap_set.inner_annulus_pixels(fwhm) == pytest.approx(expected_inner)
        assert ap_set.outer_annulus_pixels(fwhm) == pytest.approx(expected_outer)

        # The properties should still be based on the settings' fwhm_estimate
        assert ap_set.inner_annulus == pytest.approx(
            ap_set.inner_annulus_pixels(settings["fwhm_estimate"])
        )
        assert ap_set.outer_annulus == pytest.approx(
            ap_set.outer_annulus_pixels(settings["fwhm_estimate"])
        )

    def test_variable_aperture_defaults_injected(self):
        # With variable_aperture=True and no geometry values given, the
        # variable-mode defaults (multiples of FWHM) should be injected
        # instead of the fixed-mode (pixel) field defaults. See #654.
        ap_set = PhotometryApertures(variable_aperture=True)
        for key, value in VARIABLE_APERTURE_DEFAULTS.items():
            assert getattr(ap_set, key) == value

    def test_variable_aperture_defaults_not_injected_when_present(self):
        # Values the caller supplies are never overridden by the
        # variable-mode defaults.
        ap_set = PhotometryApertures(
            variable_aperture=True, radius=2.5, gap=3.0, annulus_width=2.0
        )
        assert ap_set.radius == 2.5
        assert ap_set.gap == 3.0
        assert ap_set.annulus_width == 2.0

    def test_variable_aperture_defaults_partial_injection(self):
        # Only the missing keys get the variable-mode defaults.
        ap_set = PhotometryApertures(variable_aperture=True, gap=3.0)
        assert ap_set.gap == 3.0
        assert ap_set.radius == VARIABLE_APERTURE_DEFAULTS["radius"]
        assert ap_set.annulus_width == VARIABLE_APERTURE_DEFAULTS["annulus_width"]

    def test_variable_aperture_defaults_explicit_none_still_errors(self):
        # An explicit None is present in the input, so it is not replaced
        # by a default and still fails validation.
        with pytest.raises(ValidationError, match="radius"):
            PhotometryApertures(variable_aperture=True, radius=None)

    def test_variable_aperture_defaults_with_fwhm_alias(self):
        # The old "fwhm" alias for fwhm_estimate passes through the
        # default-injection validator untouched.
        ap_set = PhotometryApertures(variable_aperture=True, fwhm=4.0)
        assert ap_set.fwhm_estimate == 4.0
        assert ap_set.radius == VARIABLE_APERTURE_DEFAULTS["radius"]

    def test_variable_aperture_assignment_does_not_reinject(self):
        # pydantic v2 does not run before-validators on assignment, so
        # flipping variable_aperture on an existing instance reinterprets
        # the existing numbers rather than replacing them. This is the
        # intended semantics (a unit re-declaration must not silently
        # rewrite user values); this test pins it.
        ap_set = PhotometryApertures()
        ap_set.variable_aperture = True
        assert ap_set.radius == PhotometryApertures.model_fields["radius"].default
        assert ap_set.gap == PhotometryApertures.model_fields["gap"].default

    def test_variable_aperture_defaults_do_not_mutate_caller_dict(self):
        # The GUI passes live dicts into the model; the validator must not
        # write the injected defaults back into the caller's dict. Use
        # model_validate so the dict object itself reaches the validator.
        settings = {"variable_aperture": True}
        PhotometryApertures.model_validate(settings)
        assert settings == {"variable_aperture": True}

    def test_variable_aperture_geometry_identities(self):
        # In variable mode every geometry number is an exact multiple of the
        # FWHM: inner = (radius + gap) * fwhm and
        # outer = (radius + gap + annulus_width) * fwhm. These identities are
        # the core of the #654 semantics change, so pin them exactly.
        settings = deepcopy(TEST_APERTURE_SETTINGS)
        settings["variable_aperture"] = True
        settings["radius"] = 1.5
        settings["gap"] = 2.0
        settings["annulus_width"] = 1.5
        ap_set = PhotometryApertures(**settings)

        for fwhm in [1.0, 3.7, 10.0]:
            assert ap_set.radius_pixels(fwhm) == pytest.approx(1.5 * fwhm, rel=1e-12)
            assert ap_set.inner_annulus_pixels(fwhm) == pytest.approx(
                (1.5 + 2.0) * fwhm, rel=1e-12
            )
            assert ap_set.outer_annulus_pixels(fwhm) == pytest.approx(
                (1.5 + 2.0 + 1.5) * fwhm, rel=1e-12
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
