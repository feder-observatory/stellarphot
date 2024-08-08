import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from stellarphot.settings import (
    SETTINGS_FILE_VERSION,
    Camera,
    Observatory,
    PartialPhotometrySettings,
    PassbandMap,
    PhotometrySettings,
    PhotometryWorkingDirSettings,
    SavedSettings,
    settings_files,  # This import is needed for mocking -- see TestSavedSettings
)
from stellarphot.settings.tests.test_models import TEST_PHOTOMETRY_SETTINGS

CAMERA = """
{
    "name": "Aspen CG 16m",
    "data_unit": "adu",
    "gain": "1.5 electron / adu",
    "read_noise": "10.0 electron",
    "dark_current": "0.01 electron / s",
    "pixel_scale": "0.6 arcsec / pix",
    "max_data_value": "50000.0 adu"
}
"""

OBSERVATORY = """
{
    "name": "Feder",
    "latitude": "46d52m25.68s",
    "longitude": "263d13m55.92s",
    "elevation": "311.0 m",
    "AAVSO_code": null,
    "TESS_telescope_code": null
}
"""

PASSBAND_MAP = """
{
    "name": "Filter wheel 1",
    "your_filter_names_to_aavso": [
        {
            "your_filter_name": "rp",
            "aavso_filter_name": "SR"
        },
        {
            "your_filter_name": "gp",
            "aavso_filter_name": "SG"
        }
    ]
}
"""


# Keep this test out of the class so that it uses the real settings path.
def test_settings_path_contains_package_and_version():
    # Make sure that the path to the settings file contains the package name and
    # version.
    saved_settings = SavedSettings(_create_path=False)
    assert "stellarphot" in str(saved_settings.settings_path)
    assert SETTINGS_FILE_VERSION in str(saved_settings.settings_path)


class TestSavedSettings:
    # This pytest fixture is used to create a fake settings directory for the tests.

    # Being a fixture means it can be passed into tests or other functions, just like
    # the fixture tmp_path.
    # The autouse=True parameter means that this fixture will be provided to every test
    # in this class without needing to be explicitly passed in.
    @pytest.fixture(autouse=True)
    def fake_settings_dir(self, mocker, tmp_path):
        # mocker is a pytest fixture provided by the pytest-mock package. It is used to
        # mock objects and functions.
        # Mocking means providing a fake version of an object, function, attribute, or
        # method that can be used in place of the real thing.

        # One of the confusing things is figuring out what to mock. In this case, we are
        # mocking the user_data_dir attribute of the PlatformDirs class in the
        # settings_files module. To make sure that is the PlatformDirs class we are
        # mocking, we need to specifically mock settings_files.PlatformDirs. A few
        # things that wouldn't work, for example, are importing PlatformsDirs directly
        # from platformdirs in this module and then trying to mock that, or importing
        # PlatformDirs from settings_files and then trying to mock that. Actually,
        # that last thing might work, but there is some values in being explicit here.
        # doing it that way does mean importing the settings_files module.
        #
        # This attribute is used to determine the path to the
        # settings directory. By mocking it, we can control where the settings directory
        # is created and use a temporary directory for the tests.

        # stellarphot is added to the name of the directory to make sure we start
        # without a stellarphot directory for each test.
        mocker.patch.object(
            settings_files.PlatformDirs, "user_data_dir", tmp_path / "stellarphot"
        )

    def test_settings_path_is_created_if_not_exists(self):
        # Check that the settings path is created if it doesn't exist.
        # It is important to use settings_files.PlatformDirs instead
        # of, say, importing PlatformDirs directly because we want to use the mocked
        # version of the attribute.
        assert not Path(settings_files.PlatformDirs.user_data_dir).exists()
        saved_settings = SavedSettings()
        assert saved_settings.settings_path.exists()

    @pytest.mark.parametrize(
        "klass,item_json",
        [(Camera, CAMERA), (Observatory, OBSERVATORY), (PassbandMap, PASSBAND_MAP)],
    )
    def test_add_saved_item(self, klass, item_json):
        # Test that items are properly saved and loaded.
        saved_settings = SavedSettings()
        assert saved_settings.settings_path.exists()
        # Add a camera.
        item = klass.model_validate_json(item_json)
        saved_settings.add_item(item)
        # Load the cameras
        saved_items = saved_settings.get_items(item)
        assert len(saved_items.as_dict) == 1
        assert saved_items.as_dict[item.name] == item

    @pytest.mark.parametrize(
        "klass,item_json",
        [(Camera, CAMERA), (Observatory, OBSERVATORY), (PassbandMap, PASSBAND_MAP)],
    )
    def test_adding_multiple_items_of_same_type(self, klass, item_json):
        # Test that items are properly saved and loaded.
        saved_settings = SavedSettings()
        assert saved_settings.settings_path.exists()
        # Add an item
        item1 = klass.model_validate_json(item_json)
        item2 = klass.model_validate_json(
            item_json.replace(item1.name, item1.name + "2")
        )
        saved_settings.add_item(item1)
        saved_settings.add_item(item2)
        # Load the items -- any instance of the class (e.g. Camera) or a
        # string (e.g. "camera") should work for getting the items.
        saved_items = saved_settings.get_items(item1)
        assert len(saved_items.as_dict) == 2
        assert saved_items.as_dict[item2.name] == item2
        assert saved_items.as_dict[item1.name] == item1

    def test_add_existing_saved_item_raises_error(self):
        # Test that adding an existing camera raises an error. Other items follow the
        # same pattern, so only cameras are tested.
        saved_settings = SavedSettings()
        # Add a camera.
        item = Camera.model_validate_json(CAMERA)
        saved_settings.add_item(item)
        # Add the same camera again.
        with pytest.raises(
            ValueError, match="Aspen CG 16m already exists in cameras.json"
        ):
            saved_settings.add_item(item)

    def test_adding_multiple_types_of_items(self):
        # Test that adding multiple types of items works.
        saved_settings = SavedSettings()
        # Add a camera.
        camera = Camera.model_validate_json(CAMERA)
        saved_settings.add_item(camera)
        # Add an observatory.
        observatory = Observatory.model_validate_json(OBSERVATORY)
        saved_settings.add_item(observatory)
        # Add a passband map.
        passband_map = PassbandMap.model_validate_json(PASSBAND_MAP)
        saved_settings.add_item(passband_map)
        # Load the items
        cameras = saved_settings.get_items("camera")
        assert len(cameras.as_dict) == 1
        assert cameras.as_dict[camera.name] == camera
        observatories = saved_settings.get_items("observatory")
        assert len(observatories.as_dict) == 1
        assert observatories.as_dict[observatory.name] == observatory
        passband_maps = saved_settings.get_items("passband_map")
        assert len(passband_maps.as_dict) == 1
        assert passband_maps.as_dict[passband_map.name] == passband_map

    def test_delete_without_confirm_raises_error(self):
        # Trying to delete settings without confirming should raise an error.
        saved_settings = SavedSettings()
        with pytest.raises(ValueError, match="You must confirm deletion by passing"):
            saved_settings.cameras.delete()

    def test_delete_with_confirm_deletes_file(self):
        # Test that deleting a settings file works.
        saved_settings = SavedSettings()
        # Add a camera.
        camera = Camera.model_validate_json(CAMERA)
        saved_settings.add_item(camera)
        # Delete the camera.
        saved_settings.cameras.delete(confirm=True)
        assert not (
            saved_settings.settings_path / saved_settings.cameras._file_name
        ).exists()

    def test_deleting_all_settings_without_confirm_raises_error(self):
        # Trying to delete all settings without confirming should raise an error.
        saved_settings = SavedSettings(_create_path=False)
        with pytest.raises(ValueError, match="You must confirm deletion by passing"):
            saved_settings.delete()

    def test_delete_all_settings_with_confirm_deletes_files(self):
        # Test that deleting all settings files works.
        saved_settings = SavedSettings()
        # Add a camera.
        camera = Camera.model_validate_json(CAMERA)
        saved_settings.add_item(camera)
        # Add an observatory.
        observatory = Observatory.model_validate_json(OBSERVATORY)
        saved_settings.add_item(observatory)
        # Add a passband map.
        passband_map = PassbandMap.model_validate_json(PASSBAND_MAP)
        saved_settings.add_item(passband_map)

        # Delete all settings.
        saved_settings.delete(confirm=True, delete_settings_folder=True)
        assert not (
            saved_settings.settings_path / saved_settings.cameras._file_name
        ).exists()
        assert not (
            saved_settings.settings_path / saved_settings.observatories._file_name
        ).exists()
        assert not (
            saved_settings.settings_path / saved_settings.passband_maps._file_name
        ).exists()
        assert not saved_settings.settings_path.exists()

    def test_delete_all_with_no_settings_works(self):
        # Test that deleting all settings files works when no settings are present.
        saved_settings = SavedSettings()
        # Delete all settings but not the settings folder
        saved_settings.delete(confirm=True)
        assert len(list(saved_settings.settings_path.glob("*"))) == 0

        # Delete all settings and the settings folder
        saved_settings.delete(confirm=True, delete_settings_folder=True)
        assert not saved_settings.settings_path.exists()

    def test_delete_item_from_collection_works(self):
        # Test that deleting an item from a collection works.
        saved_settings = SavedSettings()
        # Add a camera.
        camera = Camera.model_validate_json(CAMERA)
        saved_settings.add_item(camera)
        camera2 = Camera.model_validate_json(CAMERA.replace("Aspen CG 16m", "foo"))
        saved_settings.add_item(camera2)

        # Make sure both cameras are in the collection.
        assert len(saved_settings.cameras.as_dict) == 2
        # Delete the second camera.
        saved_settings.cameras.delete(name=camera2.name, confirm=True)
        assert len(saved_settings.cameras.as_dict) == 1

    def test_delete_item_from_collection_with_unknown_item_fails(self):
        # Test that trying to delete an unknown item from a collection fails.
        saved_settings = SavedSettings()
        camera = Camera.model_validate_json(CAMERA)
        saved_settings.add_item(camera)
        with pytest.raises(ValueError, match="not found in"):
            saved_settings.cameras.delete(name=camera.name + "foo", confirm=True)

    def test_revtrieving_item_by_name_works(self):
        # Test that retrieving an item by name works.
        saved_settings = SavedSettings()
        # Add a camera.
        camera = Camera.model_validate_json(CAMERA)
        saved_settings.add_item(camera)
        # Retrieve the camera by name.
        retrieved_camera = saved_settings.cameras.get(camera.name)
        assert retrieved_camera == camera

    def test_get_item_with_unknown_item_fails(self):
        # Test that trying to get an unknown item fails.
        saved_settings = SavedSettings()
        with pytest.raises(ValueError, match="Unknown item foo of type"):
            saved_settings.get_items("foo")

    def test_add_item_with_unknown_item_fails(self):
        # Test that trying to add an unknown item fails.
        saved_settings = SavedSettings()
        with pytest.raises(ValueError, match="Unknown item foo of type"):
            saved_settings.add_item("foo")

    @pytest.mark.parametrize(
        "klass,item_json",
        [(Camera, CAMERA), (Observatory, OBSERVATORY), (PassbandMap, PASSBAND_MAP)],
    )
    def test_saved_settings_delete_item(self, klass, item_json):
        # Test that items can be deleted.
        saved_settings = SavedSettings()
        # Add item.
        item = klass.model_validate_json(item_json)
        saved_settings.add_item(item)
        # Verify that the item is there
        assert saved_settings.get_items(klass.__name__).as_dict[item.name] == item

        saved_settings.delete_item(item, confirm=True)
        # Verify that the item was deleted.
        assert len(saved_settings.get_items(klass.__name__).as_dict) == 0

    def test_saved_settings_delete_item_with_unknown_item_fails(self):
        # Test that trying to delete an unknown item fails.
        saved_settings = SavedSettings()
        with pytest.raises(ValueError, match="Unknown item foo of type"):
            saved_settings.delete_item("foo", confirm=True)

    def test_saved_settings_delete_item_with_confirm_false_fails(self):
        # Test that trying to delete an item without confirming fails.
        saved_settings = SavedSettings()
        # Make a camera and save it
        camera = Camera.model_validate_json(CAMERA)
        saved_settings.add_item(camera)
        with pytest.raises(ValueError, match="You must confirm deletion by passing"):
            saved_settings.delete_item(camera, confirm=False)

    def test_saved_settings_delete_item_valid_item_not_in_collection_fails(self):
        # Test that trying to delete an item that is not in the collection fails.
        saved_settings = SavedSettings()
        # Make a camera but don't save it
        camera = Camera.model_validate_json(CAMERA)
        with pytest.raises(ValueError, match="not found in"):
            saved_settings.delete_item(camera, confirm=True)

    def test_saved_settings_round_trip_with_unicode_name(self):
        # Test that items with unicode names can be saved and loaded.
        saved_settings = SavedSettings()
        # Add a camera. This particular name causes a failure on Windows because the
        # default encoding doesn't include Korean characters.
        camera_name = "크레이그"
        camera = Camera(
            name=camera_name,
            data_unit="adu",
            gain="1.5 electron / adu",
            read_noise="10.0 electron",
            dark_current="0.01 electron / s",
            pixel_scale="0.6 arcsec / pix",
            max_data_value="50000.0 adu",
        )
        saved_settings.add_item(camera)
        # Load the camera.
        loaded_camera = saved_settings.get_items("camera").as_dict[camera_name]
        assert loaded_camera == camera


class TestPhotometryWorkingDirSettings:
    def setup_class(cls):
        cls.temp_dir = TemporaryDirectory()

    def teardown_class(cls):
        cls.temp_dir.cleanup()

    def setup_method(self, _):
        # Need to accept the second argument, but don't use it.
        self.original_wdir = Path.cwd()
        os.chdir(self.temp_dir.name)
        for file in Path.cwd().glob("*.json"):
            file.unlink()

    def teardown_method(self, _):
        os.chdir(self.original_wdir)

    def test_sneaky_name_not_accepted(self):
        sneaky_names = [
            "../myfile.json",
            "/some/absolute/path",
            "file_with_no_json_extension.txt",
            "file_with_no_extension",
            " started with a space",
        ]
        for name in sneaky_names:
            with pytest.raises(ValueError, match="not a valid name. The name can"):
                PhotometryWorkingDirSettings(settings_file_name=name)

    def test_bad_settings_value_raises_error(self):
        settings_file = PhotometryWorkingDirSettings()
        error_message = (
            "Settings must be PhotometrySettings or PartialPhotometrySettings"
        )
        with pytest.raises(ValueError, match=error_message):
            settings_file.save("foo")

    def test_save_partial_settings(self):
        # Test that saving partial settings works.
        settings_file = PhotometryWorkingDirSettings()
        settings = PartialPhotometrySettings()
        settings_file.save(settings)
        assert settings_file.partial_settings_file.exists()
        assert not settings_file.settings_file.exists()
        assert settings_file.partial_settings == settings

    def test_save_complete_settings(self):
        # Test that saving complete settings works.
        settings_file = PhotometryWorkingDirSettings()
        settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        settings_file.save(settings)
        assert settings_file.settings_file.exists()
        assert not settings_file.partial_settings_file.exists()
        assert settings_file.settings == settings

    def test_save_partial_settings_that_are_full_settings(self):
        # Test that saving partial settings that are actually full settings works.
        settings_file = PhotometryWorkingDirSettings()
        settings = PartialPhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        settings_file.save(settings)
        assert settings_file.settings_file.exists()
        assert not settings_file.partial_settings_file.exists()
        assert settings_file.settings == settings
        assert settings_file.partial_settings is None

    def test_save_partial_then_full_settings(self):
        # Test that saving partial settings and then full settings works.
        settings_file = PhotometryWorkingDirSettings()
        partial_settings = PartialPhotometrySettings()
        settings_file.save(partial_settings)
        assert settings_file.partial_settings_file.exists()
        assert not settings_file.settings_file.exists()
        assert settings_file.partial_settings == partial_settings

        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        settings_file.save(full_settings)
        assert settings_file.settings_file.exists()
        assert not settings_file.partial_settings_file.exists()
        assert settings_file.settings == full_settings
        assert settings_file.partial_settings is None

    @pytest.mark.parametrize("update", [True, False])
    def test_save_full_then_partial_settings(self, update):
        # Test that saving full settings and then partial settings generates
        # the expected error.
        settings_file = PhotometryWorkingDirSettings()
        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        settings_file.save(full_settings)
        assert settings_file.settings_file.exists()
        assert not settings_file.partial_settings_file.exists()
        assert settings_file.settings == full_settings

        camera = Camera.model_validate_json(CAMERA)
        # Change the camera name we we can detect whether the setting saved
        # to the working directory has been updated.
        camera.name = "new camera"
        partial_settings = PartialPhotometrySettings(camera=camera)

        if update:
            settings_file.save(partial_settings, update=update)
        else:
            error_message = (
                "Cannot save partial settings when full settings already exist"
            )
            with pytest.raises(ValueError, match=error_message):
                settings_file.save(partial_settings, update=update)

        assert not settings_file.partial_settings_file.exists()
        assert settings_file.settings_file.exists()
        assert settings_file.partial_settings is None
        if update:
            assert settings_file.settings.camera.name == camera.name
        else:
            assert settings_file.settings.camera.name == full_settings.camera.name

    def test_load_conflicting_partial_and_full_settings(self):
        # Make a valid partial settings file and a valid full settings file
        # that conflict with each other.
        settings_file = PhotometryWorkingDirSettings()
        camera = Camera.model_validate_json(CAMERA)
        partial_settings = PartialPhotometrySettings(camera=camera)
        # write these settings directly to the directory to avoid any conflict
        # resolution in the save method.
        with settings_file.partial_settings_file.open("w") as f:
            f.write(partial_settings.model_dump_json())

        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        with settings_file.settings_file.open("w") as f:
            f.write(full_settings.model_dump_json())

        # Try to load the settings. This should raise an error because the
        # settings conflict.
        error_message = "Partial settings and full settings do not match"
        with pytest.raises(ValueError, match=error_message):
            settings_file.load()

    def test_load_partial_and_full_both_valid(self):
        # Make a valid partial settings file that is actually a full file and
        # a valid full settings file that do not conflict with each other.
        settings_file = PhotometryWorkingDirSettings()

        partial_settings = PartialPhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        # write these settings directly to the directory to avoid any conflict
        # resolution in the save method.
        with settings_file.partial_settings_file.open("w") as f:
            f.write(partial_settings.model_dump_json())

        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        with settings_file.settings_file.open("w") as f:
            f.write(full_settings.model_dump_json())

        # Load the settings. This should work because the settings don't conflict.
        settings = settings_file.load()
        assert settings == full_settings

        # Should have no partial settings
        assert not settings_file.partial_settings_file.exists()
        assert settings_file.partial_settings is None

    def test_load_no_settings(self):
        # Test that loading settings when no settings exist raise error.
        settings_file = PhotometryWorkingDirSettings()
        error_message = f"Settings file {settings_file.settings_file} does not exist"
        with pytest.raises(ValueError, match=error_message):
            settings_file.load()

    @pytest.mark.parametrize("full_settings", [True, False])
    def test_load_bad_settings(self, full_settings):
        # Test that loading bad settings raises an error.
        settings_file = PhotometryWorkingDirSettings()
        if full_settings:
            file = settings_file.settings_file
        else:
            file = settings_file.partial_settings_file
        with file.open("w") as f:
            f.write("{bad: settings}")
        error_message = "Error loading "
        with pytest.raises(ValueError, match=error_message):
            settings_file.load()

    def test_load_one_setting_present(self):
        # Test that loading settings when only one setting is present raises an error.
        settings_file = PhotometryWorkingDirSettings()
        partial_settings = PartialPhotometrySettings()
        with settings_file.partial_settings_file.open("w") as f:
            f.write(partial_settings.model_dump_json())

        settings = settings_file.load()
        assert settings == partial_settings

    def test_save_updates_instead_of_replacing(self):
        # Test that saving settings adds to whatever partial settings have
        # already been saved instead of dumping anything that used to be
        # there.
        settings_file = PhotometryWorkingDirSettings()

        # Save a Camera first
        partial_settings_cam = PartialPhotometrySettings(
            camera=TEST_PHOTOMETRY_SETTINGS["camera"]
        )
        settings_file.save(partial_settings_cam, update=True)
        from_file = settings_file.load()
        # Make sure the camera is there
        assert from_file.camera == TEST_PHOTOMETRY_SETTINGS["camera"]

        # Save a different item, like an observatory
        partial_settings_obs = PartialPhotometrySettings(
            observatory=TEST_PHOTOMETRY_SETTINGS["observatory"]
        )
        settings_file.save(partial_settings_obs, update=True)
        from_file2 = settings_file.load()
        # Make sure the camera is still there
        assert from_file2.camera == TEST_PHOTOMETRY_SETTINGS["camera"]
        # Make sure the observatory is there
        assert from_file2.observatory == TEST_PHOTOMETRY_SETTINGS["observatory"]

    def test_save_update_completing_partial_makes_full(self):
        # Test that saving a partial settings file and then updating it to a
        # full settings file works.
        settings_file = PhotometryWorkingDirSettings()
        almost_complete_settings = TEST_PHOTOMETRY_SETTINGS.copy()
        the_observatory = almost_complete_settings.pop("observatory")

        # Make and save an object that has all settings except observatory
        partial_settings = PartialPhotometrySettings(**almost_complete_settings)
        settings_file.save(partial_settings, update=True)
        from_file = settings_file.load()
        # Make sure we have the partial settings
        assert from_file == partial_settings

        # Save the observatory
        the_last_setting = PartialPhotometrySettings(observatory=the_observatory)
        settings_file.save(the_last_setting, update=True)
        from_file2 = settings_file.load()
        # Make sure we have the full settings
        assert from_file2 == PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        # Make sure we have no partial settings
        assert settings_file.partial_settings is None
