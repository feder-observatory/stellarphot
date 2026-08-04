import os
import warnings
from copy import deepcopy
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
    PhotometrySettingsWarning,
    PhotometryWorkingDirSettings,
    SavedSettings,
    settings_files,  # This import is needed for mocking -- see TestSavedSettings
)
from stellarphot.settings.constants import TEST_PHOTOMETRY_SETTINGS

TEST_PHOTOMETRY_SETTINGS = deepcopy(TEST_PHOTOMETRY_SETTINGS)

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
        for file in Path.cwd().glob("*.json*"):
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

    def test_save_partial_update_with_unreadable_full_settings(self):
        # An existing full settings file that cannot be read should not make
        # a partial save crash. The partial settings are saved on their own
        # and the unreadable file is left in place.
        settings_file = PhotometryWorkingDirSettings()
        bad_content = '{"pasta": "carbonara"}'
        settings_file.settings_file.write_text(bad_content)

        camera = Camera.model_validate_json(CAMERA)
        partial_settings = PartialPhotometrySettings(camera=camera)
        settings_file.save(partial_settings, update=True)

        assert settings_file.partial_settings_file.exists()
        assert settings_file.partial_settings == partial_settings
        assert settings_file.settings_file.read_text() == bad_content

    def test_save_partial_with_unreadable_partial_settings_makes_backup(self):
        # An existing partial settings file that cannot be read should not be
        # overwritten by a partial save. The new partial settings are written
        # and the unreadable original is preserved with a .bak suffix.
        settings_file = PhotometryWorkingDirSettings()
        bad_content = '{"pasta": "carbonara"}'
        settings_file.partial_settings_file.write_text(bad_content)

        camera = Camera.model_validate_json(CAMERA)
        partial_settings = PartialPhotometrySettings(camera=camera)
        settings_file.save(partial_settings, update=True)

        assert settings_file.partial_settings_file.exists()
        assert settings_file.partial_settings == partial_settings
        backup_file = settings_file.partial_settings_file.with_name(
            settings_file.partial_settings_file.name + ".bak"
        )
        assert backup_file.read_text() == bad_content

    def test_save_full_with_unreadable_full_settings_makes_backup(self):
        # An existing full settings file that cannot be read should not be
        # overwritten when partial settings that are actually complete are
        # saved to the full settings file. The new full settings are written
        # and the unreadable original is preserved with a .bak suffix.
        settings_file = PhotometryWorkingDirSettings()
        bad_content = '{"pasta": "amatriciana"}'
        settings_file.settings_file.write_text(bad_content)

        settings = PartialPhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        settings_file.save(settings, update=True)

        assert settings_file.settings_file.exists()
        assert settings_file.settings == settings
        backup_file = settings_file.settings_file.with_name(
            settings_file.settings_file.name + ".bak"
        )
        assert backup_file.read_text() == bad_content

    def test_save_full_with_unreadable_partial_settings_makes_backup(self):
        # Saving full settings normally deletes any partial settings file. If
        # the partial settings file cannot be read it should instead be
        # preserved with a .bak suffix.
        settings_file = PhotometryWorkingDirSettings()
        bad_content = '{"pasta": "puttanesca"}'
        settings_file.partial_settings_file.write_text(bad_content)

        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        settings_file.save(full_settings)

        assert settings_file.settings_file.exists()
        assert settings_file.settings == full_settings
        assert not settings_file.partial_settings_file.exists()
        backup_file = settings_file.partial_settings_file.with_name(
            settings_file.partial_settings_file.name + ".bak"
        )
        assert backup_file.read_text() == bad_content

    def test_save_partial_update_with_corrupt_partial_and_valid_full(self):
        # Test that when both a valid full settings file and a corrupt partial
        # settings file exist, a partial save with update=True will:
        # 1. Rename the corrupt partial file to .bak
        # 2. Merge the new partial into the valid full settings
        # 3. Save the result as a full settings file (since the partial is
        #    compatible with full)
        settings_file = PhotometryWorkingDirSettings()

        # Write a valid full settings file
        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        settings_file.settings_file.write_text(full_settings.model_dump_json(indent=4))

        # Write a corrupt partial settings file
        corrupt_partial_content = '{"pasta": "carbonara"}'
        settings_file.partial_settings_file.write_text(corrupt_partial_content)

        # Create a fresh instance to trigger load()
        settings_file = PhotometryWorkingDirSettings()

        # Save a new partial settings with a different camera
        camera = Camera.model_validate_json(CAMERA)
        new_partial = PartialPhotometrySettings(camera=camera)
        settings_file.save(new_partial, update=True)

        # Assert: corrupt partial file is renamed to .bak
        assert not settings_file.partial_settings_file.exists()
        backup_file = settings_file.partial_settings_file.with_name(
            settings_file.partial_settings_file.name + ".bak"
        )
        assert backup_file.exists()
        assert backup_file.read_text() == corrupt_partial_content

        # Assert: subsequent load succeeds and returns merged settings
        fresh_instance = PhotometryWorkingDirSettings()
        loaded = fresh_instance.load()
        # The loaded settings should have the new camera and all other fields
        # from the original full settings
        assert loaded.camera == camera
        assert (
            loaded.observatory
            == PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS).observatory
        )

    def test_save_partial_update_with_corrupt_full_and_valid_partial(self):
        # Test that when both a corrupt full settings file and a valid partial
        # settings file exist, a partial save with update=True will:
        # 1. Leave the corrupt full file in place (un-renamed)
        # 2. Merge into the valid partial and save the result
        # 3. Not create a .bak of the corrupt full file
        settings_file = PhotometryWorkingDirSettings()

        # Write a valid partial settings file with a camera
        camera = Camera.model_validate_json(CAMERA)
        partial_settings = PartialPhotometrySettings(camera=camera)
        settings_file.partial_settings_file.write_text(
            partial_settings.model_dump_json(indent=4)
        )

        # Write a corrupt full settings file
        corrupt_full_content = '{"pasta": "carbonara"}'
        settings_file.settings_file.write_text(corrupt_full_content)

        # Create a fresh instance
        settings_file = PhotometryWorkingDirSettings()

        # Save a new partial settings with an observatory
        observatory = Observatory(**TEST_PHOTOMETRY_SETTINGS["observatory"])
        new_partial = PartialPhotometrySettings(observatory=observatory)
        settings_file.save(new_partial, update=True)

        # Assert: corrupt full file is left in place un-renamed
        assert settings_file.settings_file.exists()
        assert settings_file.settings_file.read_text() == corrupt_full_content
        backup_file = settings_file.settings_file.with_name(
            settings_file.settings_file.name + ".bak"
        )
        assert not backup_file.exists()

        # Assert: partial file on disk contains merged settings
        # (old camera + new observatory). Verify by parsing the file directly
        # since the corrupt full file prevents load() from succeeding.
        partial_from_disk = PartialPhotometrySettings.model_validate_json(
            settings_file.partial_settings_file.read_text()
        )
        assert partial_from_disk.camera == camera
        assert partial_from_disk.observatory == observatory

    @pytest.mark.parametrize("save_what", ["partial_update", "full"])
    def test_save_with_conflicting_partial_settings_makes_backup(self, save_what):
        # A readable partial settings file that conflicts with the full
        # settings loses the conflict when a save writes full settings; it
        # should be preserved with a .bak extension rather than deleted,
        # because its differing values cannot be reconstructed from the
        # saved settings.
        settings_file = PhotometryWorkingDirSettings()

        # Write conflicting valid files directly to the directory to avoid
        # any conflict resolution in the save method.
        camera = Camera.model_validate_json(CAMERA)
        partial_content = PartialPhotometrySettings(camera=camera).model_dump_json()
        settings_file.partial_settings_file.write_text(partial_content)

        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        settings_file.settings_file.write_text(full_settings.model_dump_json())

        # Create a fresh instance and save full settings, either directly or
        # by way of a partial update that merges into the full settings.
        settings_file = PhotometryWorkingDirSettings()
        if save_what == "partial_update":
            to_save = PartialPhotometrySettings(
                observatory=Observatory(**TEST_PHOTOMETRY_SETTINGS["observatory"])
            )
        else:
            to_save = full_settings
        settings_file.save(to_save, update=True)

        # Assert: the conflicting partial file is set aside, not deleted
        assert not settings_file.partial_settings_file.exists()
        backup_file = settings_file.partial_settings_file.with_name(
            settings_file.partial_settings_file.name + ".bak"
        )
        assert backup_file.read_text() == partial_content

        # Assert: the directory loads cleanly, with the full settings winning
        # the conflict
        assert PhotometryWorkingDirSettings().load() == full_settings

    def test_unreadable_properties_reflect_last_load(self):
        # After a load(), the *_unreadable properties report exactly which
        # existing files could not be read: here the full settings file is
        # corrupt and there is no partial settings file at all.
        settings_file = PhotometryWorkingDirSettings()
        settings_file.settings_file.write_text('{"pasta": "carbonara"}')

        with pytest.raises(ValueError, match="Error loading settings"):
            settings_file.load()

        assert settings_file.full_settings_unreadable
        assert not settings_file.partial_settings_unreadable

    def test_save_does_not_overwrite_existing_backup(self):
        # A .bak file may hold the only copy of settings from an earlier
        # corruption, so a save that needs to set aside another unreadable
        # file must use a numbered backup name (.bak1, .bak2, ...) rather
        # than overwriting the existing .bak.
        settings_file = PhotometryWorkingDirSettings()
        old_backup_content = '{"pasta": "amatriciana"}'
        backup_file = settings_file.settings_file.with_name(
            settings_file.settings_file.name + ".bak"
        )
        backup_file.write_text(old_backup_content)

        bad_content = '{"pasta": "carbonara"}'
        settings_file.settings_file.write_text(bad_content)

        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        settings_file.save(full_settings)

        # The earlier backup is untouched and the unreadable file went to
        # the next available numbered name.
        assert backup_file.read_text() == old_backup_content
        numbered_backup = settings_file.settings_file.with_name(
            settings_file.settings_file.name + ".bak1"
        )
        assert numbered_backup.read_text() == bad_content
        assert settings_file.settings == full_settings

    def test_failed_write_preserves_partial_settings_file(self, mocker):
        # Saving full settings disposes of the partial settings file, whose
        # non-None values may exist nowhere else if the write of the full
        # settings then fails. The disposal must therefore happen only after
        # the full settings are safely on disk.
        settings_file = PhotometryWorkingDirSettings()
        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        # The partial camera matches the full settings being saved, which is
        # the case where save() would simply delete (not back up) the
        # partial settings file.
        partial_content = PartialPhotometrySettings(
            camera=full_settings.camera
        ).model_dump_json()
        settings_file.partial_settings_file.write_text(partial_content)

        # Serialization happens before the settings file is opened for
        # writing, so this failure stands in for any failure to get the new
        # settings onto disk.
        mocker.patch.object(
            PhotometrySettings, "model_dump_json", side_effect=RuntimeError("disk full")
        )
        with pytest.raises(RuntimeError, match="disk full"):
            settings_file.save(full_settings)

        assert settings_file.partial_settings_file.read_text() == partial_content

    def test_failed_write_leaves_existing_settings_intact(self, mocker):
        # A failure partway through writing the new settings must not
        # truncate or destroy the existing readable settings file; the
        # write goes to a temporary file that atomically replaces the
        # target only on success.
        settings_file = PhotometryWorkingDirSettings()
        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        settings_file.save(full_settings)
        original = settings_file.settings_file.read_text()

        mocker.patch.object(Path, "write_text", side_effect=OSError("disk full"))
        with pytest.raises(OSError, match="disk full"):
            PhotometryWorkingDirSettings().save(full_settings)

        assert settings_file.settings_file.read_text() == original
        assert not settings_file.settings_file.with_name(
            settings_file.settings_file.name + ".tmp"
        ).exists()

    def test_save_completing_partial_no_backup_for_replaced_value(self):
        # A partial update that both replaces a previously saved partial
        # value and completes the full settings deletes the on-disk partial
        # file without a backup when its only divergence is the value the
        # caller explicitly replaced -- replacing it was the point of the
        # save, so nothing is lost.
        settings_file = PhotometryWorkingDirSettings()
        old_camera = Camera.model_validate_json(CAMERA)
        settings_file.partial_settings_file.write_text(
            PartialPhotometrySettings(camera=old_camera).model_dump_json()
        )

        new_settings = PartialPhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        assert new_settings.camera != old_camera
        settings_file.save(new_settings, update=True)

        assert settings_file.settings.camera == new_settings.camera
        assert not settings_file.partial_settings_file.exists()
        backup_file = settings_file.partial_settings_file.with_name(
            settings_file.partial_settings_file.name + ".bak"
        )
        assert not backup_file.exists()

    def test_save_suppresses_non_settings_warnings_from_load(self, mocker):
        # save() calls load() internally for bookkeeping. Warnings from that
        # load (e.g. library warnings) were already shown when the settings
        # were first loaded, so save() should not repeat them -- but see
        # test_save_lets_settings_warnings_through for the one category that
        # must still get through.
        real_load = PhotometryWorkingDirSettings.load

        def warning_load(_self):
            warnings.warn("Settings were migrated", UserWarning, stacklevel=2)
            return real_load(_self)

        mocker.patch.object(
            settings_files.PhotometryWorkingDirSettings, "load", warning_load
        )

        settings_file = PhotometryWorkingDirSettings()
        camera = Camera.model_validate_json(CAMERA)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            settings_file.save(PartialPhotometrySettings(camera=camera), update=True)

    def test_save_lets_settings_warnings_through(self, mocker):
        # save()'s internal load suppresses library warnings, but a
        # PhotometrySettingsWarning (e.g. a settings-format migration
        # message) must reach a caller whose first interaction with the
        # settings is a save().
        real_load = PhotometryWorkingDirSettings.load

        def warning_load(_self):
            warnings.warn(
                "Settings were migrated", PhotometrySettingsWarning, stacklevel=2
            )
            return real_load(_self)

        mocker.patch.object(
            settings_files.PhotometryWorkingDirSettings, "load", warning_load
        )

        settings_file = PhotometryWorkingDirSettings()
        camera = Camera.model_validate_json(CAMERA)
        with pytest.warns(PhotometrySettingsWarning, match="migrated"):
            settings_file.save(PartialPhotometrySettings(camera=camera), update=True)

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
        assert from_file.camera == Camera(**TEST_PHOTOMETRY_SETTINGS["camera"])

        # Save a different item, like an observatory
        partial_settings_obs = PartialPhotometrySettings(
            observatory=Observatory(**TEST_PHOTOMETRY_SETTINGS["observatory"])
        )
        settings_file.save(partial_settings_obs, update=True)
        from_file2 = settings_file.load()
        # Make sure the camera is still there
        assert from_file2.camera == Camera(**TEST_PHOTOMETRY_SETTINGS["camera"])
        # Make sure the observatory is there
        assert from_file2.observatory == Observatory(
            **TEST_PHOTOMETRY_SETTINGS["observatory"]
        )

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

    def test_save_full_direct_with_differing_partial_settings_makes_backup(self):
        # A direct save of full settings (not a partial update) should still
        # preserve a readable, on-disk partial settings file as .bak when one
        # of its non-None values is not carried into the full settings being
        # written -- that value would otherwise be silently lost, since a
        # direct full save never merges in the on-disk partial settings.
        settings_file = PhotometryWorkingDirSettings()
        camera = Camera.model_validate_json(CAMERA)
        partial_content = PartialPhotometrySettings(camera=camera).model_dump_json()
        settings_file.partial_settings_file.write_text(partial_content)

        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        settings_file.save(full_settings)

        assert not settings_file.partial_settings_file.exists()
        backup_file = settings_file.partial_settings_file.with_name(
            settings_file.partial_settings_file.name + ".bak"
        )
        assert backup_file.read_text() == partial_content
        assert settings_file.settings_file.exists()
        assert settings_file.settings == full_settings

    def test_save_full_direct_with_matching_partial_settings_no_backup(self):
        # If the only non-None value in the on-disk partial settings matches
        # the corresponding value in the full settings being written, nothing
        # would be lost by discarding the partial file, so no .bak should be
        # created.
        settings_file = PhotometryWorkingDirSettings()
        matching_camera = Camera(**TEST_PHOTOMETRY_SETTINGS["camera"])
        partial_content = PartialPhotometrySettings(
            camera=matching_camera
        ).model_dump_json()
        settings_file.partial_settings_file.write_text(partial_content)

        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        settings_file.save(full_settings)

        assert not settings_file.partial_settings_file.exists()
        backup_file = settings_file.partial_settings_file.with_name(
            settings_file.partial_settings_file.name + ".bak"
        )
        assert not backup_file.exists()
        assert settings_file.settings_file.exists()
        assert settings_file.settings == full_settings

    def test_load_settings_file_invalid_utf8(self):
        # A settings file that is not valid UTF-8 should be treated the same
        # as a ValidationError: load() reports it as a readable-but-bad file,
        # and a subsequent save preserves the original bytes as a .bak rather
        # than silently overwriting them.
        settings_file = PhotometryWorkingDirSettings()
        bad_bytes = b"\xff\xfe not utf8"
        settings_file.settings_file.write_bytes(bad_bytes)

        with pytest.raises(ValueError, match="Error loading settings"):
            settings_file.load()

        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        settings_file.save(full_settings)

        assert settings_file.settings_file.exists()
        assert settings_file.settings == full_settings
        backup_file = settings_file.settings_file.with_name(
            settings_file.settings_file.name + ".bak"
        )
        assert backup_file.read_bytes() == bad_bytes

    def test_load_error_chains_exception(self):
        # The ValueError raised by a failed load should chain the underlying
        # exception via `from`, so the original cause is not lost.
        settings_file = PhotometryWorkingDirSettings()
        with settings_file.settings_file.open("w") as f:
            f.write("{bad: settings}")

        with pytest.raises(ValueError) as exc_info:
            settings_file.load()

        assert exc_info.value.__cause__ is not None

    def test_load_conflict_error_message_puts_paths_on_later_lines(self):
        # The GUI shows only the first line of an error prominently and folds
        # the rest into collapsed details, so the file paths must not appear
        # on the first line of the conflict message.
        settings_file = PhotometryWorkingDirSettings()
        camera = Camera.model_validate_json(CAMERA)
        partial_settings = PartialPhotometrySettings(camera=camera)
        with settings_file.partial_settings_file.open("w") as f:
            f.write(partial_settings.model_dump_json())

        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        with settings_file.settings_file.open("w") as f:
            f.write(full_settings.model_dump_json())

        with pytest.raises(ValueError) as exc_info:
            settings_file.load()

        message = str(exc_info.value)
        first_line = message.splitlines()[0]
        assert "photometry_settings.json" not in first_line
        assert "\nFull settings:" in message
        assert str(settings_file.settings_file) in message
        assert str(settings_file.partial_settings_file) in message

    def test_unreadable_properties_false_before_load(self):
        # Before any load() call, the *_unreadable properties should be False
        # even if a valid settings file exists on disk.
        settings_file = PhotometryWorkingDirSettings()

        # Write a valid full settings file
        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        settings_file.settings_file.write_text(full_settings.model_dump_json(indent=4))

        # Create a fresh instance
        fresh_instance = PhotometryWorkingDirSettings()

        # Before load(), both properties should be False
        assert not fresh_instance.full_settings_unreadable
        assert not fresh_instance.partial_settings_unreadable

        # After load(), they should still be False because the file is readable
        fresh_instance.load()
        assert not fresh_instance.full_settings_unreadable
        assert not fresh_instance.partial_settings_unreadable

    def test_unreadable_properties_corrupt_file_before_and_after_load(self):
        # Before any load() call, the *_unreadable properties should be False
        # even if a corrupt settings file exists. After load() fails on a
        # corrupt file, the properties should report True.
        settings_file = PhotometryWorkingDirSettings()

        # Write a corrupt full settings file
        bad_content = '{"pasta": "carbonara"}'
        settings_file.settings_file.write_text(bad_content)

        # Create a fresh instance
        fresh_instance = PhotometryWorkingDirSettings()

        # Before load(), full_settings_unreadable should be False
        assert not fresh_instance.full_settings_unreadable

        # Attempt to load - this should raise
        with pytest.raises(ValueError, match="Error loading settings"):
            fresh_instance.load()

        # After load() fails, full_settings_unreadable should be True
        assert fresh_instance.full_settings_unreadable
        assert not fresh_instance.partial_settings_unreadable

    def test_save_partial_no_update_with_unreadable_full_settings_message(self):
        # When trying to save partial settings with update=False and the full
        # settings file exists but is unreadable, the error message should be
        # specific about the unreadable file, not the generic message about
        # full settings already existing.
        settings_file = PhotometryWorkingDirSettings()

        # Write a corrupt full settings file
        bad_content = '{"pasta": "carbonara"}'
        settings_file.settings_file.write_text(bad_content)

        # Create a fresh instance and try to save partial with update=False
        fresh_instance = PhotometryWorkingDirSettings()
        partial_settings = PartialPhotometrySettings(
            camera=Camera.model_validate_json(CAMERA)
        )

        # Should raise ValueError with "could not be read" in the message
        with pytest.raises(ValueError, match="could not be read"):
            fresh_instance.save(partial_settings, update=False)

        # Verify the unreadable file's content is untouched
        assert fresh_instance.settings_file.read_text() == bad_content
