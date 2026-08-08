import json
import os
import warnings
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from astropy.utils.data import get_pkg_data_path

from stellarphot.settings import (
    SETTINGS_FILE_VERSION,
    Camera,
    NewerFormatError,
    Observatory,
    PartialPhotometrySettings,
    PassbandMap,
    PhotometryApertures,
    PhotometrySettings,
    PhotometrySettingsMigrationWarning,
    PhotometryWorkingDirSettings,
    SavedSettings,
    SettingsFileReadError,
    settings_files,  # This import is needed for mocking -- see TestSavedSettings
)
from stellarphot.settings.constants import TEST_PHOTOMETRY_SETTINGS
from stellarphot.settings.models import PHOTOMETRY_SETTINGS_FORMAT_VERSION

TEST_PHOTOMETRY_SETTINGS = deepcopy(TEST_PHOTOMETRY_SETTINGS)


def bak_path(path, suffix=".bak"):
    """Return the backup name save() uses when setting aside ``path``."""
    return path.with_name(path.name + suffix)


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

    def test_failed_write_leaves_saved_item_file_intact(self, mocker):
        # An interrupted write of cameras.json (or the other saved-item
        # files) must not truncate the existing file; the write goes to a
        # temporary file that atomically replaces the target only on
        # success.
        saved_settings = SavedSettings()
        camera = Camera.model_validate_json(CAMERA)
        saved_settings.add_item(camera)
        file_path = saved_settings.settings_path / "cameras.json"
        original = file_path.read_text()

        camera2 = Camera.model_validate_json(
            CAMERA.replace("Aspen CG 16m", "Aspen CG 16m 2")
        )
        mocker.patch.object(Path, "write_text", side_effect=OSError("disk full"))
        with pytest.raises(OSError, match="disk full"):
            saved_settings.add_item(camera2)

        assert file_path.read_text() == original
        assert not list(file_path.parent.glob("*.tmp"))

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


class TestCopyAside:
    # _copy_aside must never leave a partial backup behind: an empty or
    # truncated .bak would make later set-asides skip to .bak1 while a
    # user recovering by hand finds the useless .bak first.

    def test_read_failure_creates_no_backup(self, tmp_path, mocker):
        # The source is read before the backup is created, so a source
        # that cannot be read leaves nothing behind.
        source = tmp_path / "settings.json"
        source.write_text("important stuff")

        mocker.patch.object(Path, "read_bytes", side_effect=PermissionError("denied"))
        with pytest.raises(PermissionError, match="denied"):
            settings_files._copy_aside(source)

        assert not bak_path(source).exists()

    def test_write_failure_removes_partial_backup(self, tmp_path, mocker):
        source = tmp_path / "settings.json"
        source.write_text("important stuff")

        real_open = Path.open

        class FailingWriter:
            def __init__(self, handle):
                self._handle = handle

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self._handle.close()
                return False

            def write(self, _data):
                raise OSError("no space left on device")

        def failing_open(self, mode="r", *args, **kwargs):
            handle = real_open(self, mode, *args, **kwargs)
            return FailingWriter(handle) if "x" in mode else handle

        mocker.patch.object(Path, "open", failing_open)
        with pytest.raises(OSError, match="no space"):
            settings_files._copy_aside(source)

        assert not bak_path(source).exists()
        assert source.read_text() == "important stuff"

    def test_lost_backup_name_race_does_not_delete_existing_backup(
        self, tmp_path, mocker
    ):
        # If another writer claims the chosen backup name between
        # _backup_path and the exclusive open, the FileExistsError that
        # results must not delete the other writer's file -- only a
        # backup this call actually created may be cleaned up.
        source = tmp_path / "settings.json"
        source.write_text("new corruption")
        existing = bak_path(source)
        existing.write_text("previous backup")

        mocker.patch.object(settings_files, "_backup_path", return_value=existing)
        with pytest.raises(FileExistsError):
            settings_files._copy_aside(source)

        assert existing.read_text() == "previous backup"


class TestPhotometryWorkingDirSettings:
    def setup_class(cls):
        cls.temp_dir = TemporaryDirectory()

    def teardown_class(cls):
        cls.temp_dir.cleanup()

    def setup_method(self, _):
        # Need to accept the second argument, but don't use it.
        self.original_wdir = Path.cwd()
        os.chdir(self.temp_dir.name)
        # The glob pattern also matches the .json.bak and .json.tmp files
        # some tests leave behind.
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

    # Each case corrupts one settings file, saves new settings, and checks
    # that the save lands in the right file while the corrupt content is
    # preserved as a .bak backup rather than overwritten or deleted. The
    # first case is also a regression test: an unreadable full settings
    # file used to make save(partial, update=True) crash with an
    # AttributeError because the failed load left self._settings as None.
    @pytest.mark.parametrize(
        "corrupt, saved, expect_full",
        [
            # unreadable full + incomplete partial save: partial written,
            # full set aside
            ("full", "partial", False),
            # unreadable partial + incomplete partial save: new partial
            # written, old partial set aside
            ("partial", "partial", False),
            # unreadable full + complete partial save: full written
            ("full", "complete", True),
            # unreadable partial + full save: full written, partial set
            # aside instead of the usual deletion
            ("partial", "full", True),
        ],
    )
    def test_save_with_unreadable_settings_file_makes_backup(
        self, corrupt, saved, expect_full
    ):
        settings_file = PhotometryWorkingDirSettings()
        bad_content = '{"pasta": "carbonara"}'
        corrupt_path = (
            settings_file.settings_file
            if corrupt == "full"
            else settings_file.partial_settings_file
        )
        corrupt_path.write_text(bad_content)

        if saved == "full":
            to_save = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
            settings_file.save(to_save)
        else:
            to_save = (
                PartialPhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
                if saved == "complete"
                else PartialPhotometrySettings(
                    camera=Camera.model_validate_json(CAMERA)
                )
            )
            settings_file.save(to_save, update=True)

        assert settings_file.settings_file.exists() == expect_full
        assert settings_file.partial_settings_file.exists() != expect_full
        if expect_full:
            assert settings_file.settings == to_save
        else:
            assert settings_file.partial_settings == to_save
        assert bak_path(corrupt_path).read_text() == bad_content

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
        backup_file = bak_path(settings_file.partial_settings_file)
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
        # 1. Set the corrupt full file aside as .bak, content preserved
        # 2. Merge into the valid partial and save the result
        # 3. Leave the directory loadable again
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

        # Assert: corrupt full file was set aside as .bak, content preserved
        assert not settings_file.settings_file.exists()
        backup_file = bak_path(settings_file.settings_file)
        assert backup_file.read_text() == corrupt_full_content

        # Assert: with the corrupt file out of the way, load() succeeds and
        # returns the merged settings (old camera + new observatory).
        loaded = PhotometryWorkingDirSettings().load()
        assert loaded.camera == camera
        assert loaded.observatory == observatory

    def test_save_does_not_overwrite_existing_backup(self):
        # A .bak file may hold the only copy of settings from an earlier
        # corruption, so a save that needs to set aside another unreadable
        # file must use a numbered backup name (.bak1, .bak2, ...) rather
        # than overwriting the existing .bak.
        settings_file = PhotometryWorkingDirSettings()
        old_backup_content = '{"pasta": "amatriciana"}'
        backup_file = bak_path(settings_file.settings_file)
        backup_file.write_text(old_backup_content)

        bad_content = '{"pasta": "carbonara"}'
        settings_file.settings_file.write_text(bad_content)

        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        settings_file.save(full_settings)

        # The earlier backup is untouched and the unreadable file went to
        # the next available numbered name.
        assert backup_file.read_text() == old_backup_content
        numbered_backup = bak_path(settings_file.settings_file, ".bak1")
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
        # The temporary file has a randomized name, so check that nothing
        # matching its pattern was left behind.
        assert not list(settings_file.settings_file.parent.glob("*.tmp"))

    def test_failed_write_leaves_unreadable_file_in_place(self, mocker):
        # If writing the new settings fails, an unreadable file at the
        # target name must remain in place under its original name -- not
        # be renamed to .bak with nothing left in its place.
        settings_file = PhotometryWorkingDirSettings()
        bad_content = '{"pasta": "carbonara"}'
        settings_file.settings_file.write_text(bad_content)

        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        mocker.patch.object(Path, "write_text", side_effect=OSError("disk full"))
        with pytest.raises(OSError, match="disk full"):
            settings_file.save(full_settings)

        assert settings_file.settings_file.read_text() == bad_content
        assert not bak_path(settings_file.settings_file).exists()

    def test_failed_replace_leaves_unreadable_file_in_place(self, mocker):
        # If the atomic replace of the target file itself fails, the
        # unreadable file being set aside must still be present under its
        # original name -- the backup is a copy, not a rename, until the
        # new settings are actually in place.
        settings_file = PhotometryWorkingDirSettings()
        bad_content = '{"pasta": "carbonara"}'
        settings_file.settings_file.write_text(bad_content)

        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)

        # Fail only the replace whose target is the settings file, so any
        # rename to a backup name still succeeds.
        real_replace = Path.replace

        def flaky_replace(self, target):
            if target == settings_file.settings_file:
                raise OSError("sharing violation")
            return real_replace(self, target)

        mocker.patch.object(Path, "replace", flaky_replace)
        with pytest.raises(OSError, match="sharing violation"):
            settings_file.save(full_settings)

        assert settings_file.settings_file.read_text() == bad_content
        # The backup copy made before the replace is still around, and no
        # temporary file is left behind.
        assert bak_path(settings_file.settings_file).read_text() == bad_content
        assert not list(settings_file.settings_file.parent.glob("*.tmp"))

    def test_load_oserror_raises_settings_file_read_error(self, mocker):
        # An OS-level failure to read a settings file is a different
        # situation than a file with bad contents: the settings may be
        # perfectly valid. load() signals that with SettingsFileReadError,
        # which subclasses ValueError so existing callers are unaffected.
        settings_file = PhotometryWorkingDirSettings()
        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        settings_file.save(full_settings)

        mocker.patch.object(Path, "open", side_effect=PermissionError("denied"))
        with pytest.raises(SettingsFileReadError, match="denied"):
            settings_file.load()

    def test_save_partial_update_with_conflicting_partial_and_full(self):
        # When a valid full settings file and a conflicting valid partial
        # file both exist, a partial save with update=True merges into the
        # full settings. The conflicting partial's values must not leak into
        # the result -- a field the full settings legitimately set to None
        # must stay None -- and the conflicting partial file, whose
        # differing values are being discarded, is set aside as .bak rather
        # than deleted.
        settings_file = PhotometryWorkingDirSettings()
        full_dict = deepcopy(TEST_PHOTOMETRY_SETTINGS)
        full_dict["passband_map"] = None
        full_settings = PhotometrySettings(**full_dict)
        settings_file.settings_file.write_text(full_settings.model_dump_json(indent=4))

        stale_partial = PartialPhotometrySettings(
            passband_map=PassbandMap.model_validate_json(PASSBAND_MAP)
        )
        stale_content = stale_partial.model_dump_json(indent=4)
        settings_file.partial_settings_file.write_text(stale_content)

        settings_file = PhotometryWorkingDirSettings()
        camera = Camera.model_validate_json(CAMERA)
        settings_file.save(PartialPhotometrySettings(camera=camera), update=True)

        loaded = PhotometryWorkingDirSettings().load()
        assert loaded.camera == camera
        # The stale partial's passband_map must not be resurrected.
        assert loaded.passband_map is None
        # The conflicting partial file is preserved, not deleted.
        assert not settings_file.partial_settings_file.exists()
        assert (
            bak_path(settings_file.partial_settings_file).read_text() == stale_content
        )

    def test_load_corrupt_json_raises_plain_value_error(self):
        # A file that can be read but contains invalid settings is not a
        # read error; it raises plain ValueError, not SettingsFileReadError.
        settings_file = PhotometryWorkingDirSettings()
        settings_file.settings_file.write_text('{"pasta": "carbonara"}')

        with pytest.raises(ValueError, match="Error loading settings") as exc_info:
            settings_file.load()

        assert not isinstance(exc_info.value, SettingsFileReadError)

    def test_load_settings_file_invalid_utf8(self):
        # A settings file that is not valid UTF-8 cannot be read at all, so
        # load() raises SettingsFileReadError (a ValueError subclass), and a
        # subsequent save preserves the original bytes as a .bak rather
        # than silently overwriting them.
        settings_file = PhotometryWorkingDirSettings()
        bad_bytes = b"\xff\xfe not utf8"
        settings_file.settings_file.write_bytes(bad_bytes)

        with pytest.raises(SettingsFileReadError, match="Error loading settings"):
            settings_file.load()

        full_settings = PhotometrySettings(**TEST_PHOTOMETRY_SETTINGS)
        settings_file.save(full_settings)

        assert settings_file.settings_file.exists()
        assert settings_file.settings == full_settings
        backup_file = bak_path(settings_file.settings_file)
        assert backup_file.read_bytes() == bad_bytes

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

    # The tests below check the handling of settings format versions. Files
    # written before the settings_version field existed are "format 1" files.
    @staticmethod
    def _write_format_1_file(file_name, variable_aperture):
        """
        Write a settings file the way stellarphot did before the
        settings_version field existed, returning the dict that was written.
        """
        old_style = deepcopy(TEST_PHOTOMETRY_SETTINGS)
        old_style.pop("settings_version", None)
        old_style["photometry_apertures"]["variable_aperture"] = variable_aperture
        Path(file_name).write_text(json.dumps(old_style))
        return old_style

    def test_load_format1_fixed_aperture_loads_unchanged_no_warning(self):
        # Fixed-aperture settings have unchanged meaning, so a format 1 file
        # loads as-is, silently.
        settings_file = PhotometryWorkingDirSettings()
        original = self._write_format_1_file(
            settings_file.settings_file, variable_aperture=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = settings_file.load()
        apertures = loaded.photometry_apertures.model_dump()
        for key, value in original["photometry_apertures"].items():
            assert apertures[key] == value
        assert loaded.settings_version == PHOTOMETRY_SETTINGS_FORMAT_VERSION

    def test_load_format1_variable_aperture_migrates_and_warns(self):
        settings_file = PhotometryWorkingDirSettings()
        original = self._write_format_1_file(
            settings_file.settings_file, variable_aperture=True
        )
        with pytest.warns(PhotometrySettingsMigrationWarning, match="RESET"):
            loaded = settings_file.load()
        apertures = loaded.photometry_apertures
        # The user's aperture choice survives...
        assert apertures.variable_aperture is True
        assert apertures.radius == original["photometry_apertures"]["radius"]
        assert (
            apertures.fwhm_estimate == original["photometry_apertures"]["fwhm_estimate"]
        )
        # ...but the annulus geometry is reset to the current defaults.
        assert apertures.gap == PhotometryApertures.model_fields["gap"].default
        assert (
            apertures.annulus_width
            == PhotometryApertures.model_fields["annulus_width"].default
        )
        assert loaded.settings_version == PHOTOMETRY_SETTINGS_FORMAT_VERSION

    def test_format1_file_upgraded_on_save(self):
        settings_file = PhotometryWorkingDirSettings()
        self._write_format_1_file(settings_file.settings_file, variable_aperture=True)
        with pytest.warns(PhotometrySettingsMigrationWarning):
            loaded = settings_file.load()
        # save() re-loads the file internally, so it warns once more.
        with pytest.warns(PhotometrySettingsMigrationWarning):
            settings_file.save(loaded)

        raw = json.loads(settings_file.settings_file.read_text())
        assert raw["settings_version"] == PHOTOMETRY_SETTINGS_FORMAT_VERSION

        # The saved file is format 2 now, so loading it again is silent.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PhotometryWorkingDirSettings().load()

    def test_load_partial_format1_variable_aperture_warns(self):
        # Old partial files had every key present, with null for unset ones.
        settings_file = PhotometryWorkingDirSettings()
        original = deepcopy(TEST_PHOTOMETRY_SETTINGS)
        original.pop("settings_version", None)
        partial = {key: None for key in original}
        partial["photometry_apertures"] = original["photometry_apertures"]
        partial["photometry_apertures"]["variable_aperture"] = True
        settings_file.partial_settings_file.write_text(json.dumps(partial))

        with pytest.warns(PhotometrySettingsMigrationWarning, match="RESET"):
            loaded = settings_file.load()
        apertures = loaded.photometry_apertures
        assert apertures.variable_aperture is True
        assert apertures.radius == partial["photometry_apertures"]["radius"]
        assert apertures.gap == PhotometryApertures.model_fields["gap"].default
        assert (
            apertures.annulus_width
            == PhotometryApertures.model_fields["annulus_width"].default
        )
        assert loaded.settings_version == PHOTOMETRY_SETTINGS_FORMAT_VERSION

    def test_load_partial_format1_no_apertures_no_warning(self):
        # The negative case: the migration must not fire when there is
        # nothing to migrate. A warning that fired on every old file
        # regardless of content would train users to ignore it; it must be
        # reserved for files where gap/annulus_width were actually reset.
        settings_file = PhotometryWorkingDirSettings()
        # A format-1 partial file in the old on-disk style: every key
        # present, null for unset values, no settings_version, and only
        # camera actually filled in -- so there is no aperture section for
        # the migration to touch.
        partial = {key: None for key in TEST_PHOTOMETRY_SETTINGS}
        partial.pop("settings_version", None)
        partial["camera"] = TEST_PHOTOMETRY_SETTINGS["camera"]
        settings_file.partial_settings_file.write_text(json.dumps(partial))

        # simplefilter("error") promotes any warning on load, including
        # PhotometrySettingsMigrationWarning, to a test failure.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = settings_file.load()
        assert loaded.camera == Camera(**TEST_PHOTOMETRY_SETTINGS["camera"])
        # The missing settings_version picks up the current default, so the
        # next save writes an up-to-date file.
        assert loaded.settings_version == PHOTOMETRY_SETTINGS_FORMAT_VERSION

    def test_partial_saved_by_current_code_carries_version(self):
        # Partial files written by the current code must carry the version so
        # that they are not mistaken for format 1 files when read back.
        settings_file = PhotometryWorkingDirSettings()
        settings_file.save(
            PartialPhotometrySettings(camera=Camera.model_validate_json(CAMERA))
        )
        raw = json.loads(settings_file.partial_settings_file.read_text())
        assert raw["settings_version"] == PHOTOMETRY_SETTINGS_FORMAT_VERSION

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PhotometryWorkingDirSettings().load()

    # Parametrized over full/partial because load() validates the two files
    # through separate model_validate_json calls, and both paths must refuse
    # a too-new file. The partial path is the one that would be easy to
    # break accidentally: PartialPhotometrySettings is generated by
    # _make_partial_model and could plausibly lose the field validator in a
    # refactor.
    @pytest.mark.parametrize("full_settings", [True, False])
    def test_load_newer_format_version_raises(self, full_settings):
        # A file claiming a settings_version greater than this code
        # understands -- i.e. written by a future stellarphot -- must raise
        # NewerFormatError rather than be mis-read under current-format
        # assumptions.
        settings_file = PhotometryWorkingDirSettings()
        newer = deepcopy(TEST_PHOTOMETRY_SETTINGS)
        newer["settings_version"] = PHOTOMETRY_SETTINGS_FORMAT_VERSION + 1
        if full_settings:
            file = settings_file.settings_file
        else:
            file = settings_file.partial_settings_file
        file.write_text(json.dumps(newer))

        with pytest.raises(NewerFormatError, match="newer version of stellarphot"):
            settings_file.load()

    def test_save_refuses_to_clobber_newer_format_file(self):
        # The data-loss guard. It pins the design decision that
        # NewerFormatError subclasses Exception rather than ValueError:
        # save() starts by calling load() and treats a ValueError as
        # "nothing usable on disk, safe to write". If NewerFormatError were
        # a ValueError -- or were raised inside a validator as one, which
        # pydantic folds into ValidationError and load() re-raises as
        # ValueError -- save() would swallow it and overwrite a file
        # written by a newer stellarphot with a downgraded one.
        settings_file = PhotometryWorkingDirSettings()
        newer = deepcopy(TEST_PHOTOMETRY_SETTINGS)
        newer["settings_version"] = PHOTOMETRY_SETTINGS_FORMAT_VERSION + 1
        content = json.dumps(newer)
        settings_file.settings_file.write_text(content)

        # The two asserts check both halves of the guarantee: the error
        # propagates out of save(), and the bytes on disk are unchanged.
        partial = PartialPhotometrySettings(camera=Camera.model_validate_json(CAMERA))
        with pytest.raises(NewerFormatError, match="newer version of stellarphot"):
            settings_file.save(partial, update=True)
        assert settings_file.settings_file.read_text() == content

    def test_load_golden_2_0_0alpha_file_migrates(self):
        # An actual settings file from a 2.0.0 alpha release, which is a
        # format 1 file with variable_aperture=True.
        golden = Path(
            get_pkg_data_path("data/sample_photometry_settings_2.0.0alpha.json")
        )
        original = json.loads(golden.read_text())

        settings_file = PhotometryWorkingDirSettings()
        settings_file.settings_file.write_text(golden.read_text())
        with pytest.warns(PhotometrySettingsMigrationWarning, match="RESET"):
            loaded = settings_file.load()
        apertures = loaded.photometry_apertures
        assert apertures.variable_aperture is True
        assert apertures.radius == original["photometry_apertures"]["radius"]
        assert (
            apertures.fwhm_estimate == original["photometry_apertures"]["fwhm_estimate"]
        )
        assert apertures.gap == PhotometryApertures.model_fields["gap"].default
        assert (
            apertures.annulus_width
            == PhotometryApertures.model_fields["annulus_width"].default
        )
