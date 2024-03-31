import pytest

from stellarphot.settings import (
    SETTINGS_FILE_VERSION,
    Camera,
    Observatory,
    PassbandMap,
    SavedSettings,
)

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


class TestSavedSettings:
    def test_settings_path_contains_package_and_version(self):
        saved_settings = SavedSettings(_create_path=False)
        assert "stellarphot" in str(saved_settings.settings_path)
        assert SETTINGS_FILE_VERSION in str(saved_settings.settings_path)

    @pytest.mark.parametrize(
        "klass,item_json",
        [(Camera, CAMERA), (Observatory, OBSERVATORY), (PassbandMap, PASSBAND_MAP)],
    )
    def test_add_saved_item(self, klass, item_json, tmp_path):
        # Test that items are properly saved and loaded.
        saved_settings = SavedSettings(_testing_path=tmp_path)
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
    def test_adding_multiple_items_of_same_type(self, klass, item_json, tmp_path):
        # Test that items are properly saved and loaded.
        saved_settings = SavedSettings(_testing_path=tmp_path)
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

    def test_add_existing_saved_item_raises_error(self, tmp_path):
        # Test that adding an existing camera raises an error. Other items follow the
        # same pattern, so only cameras are tested.
        saved_settings = SavedSettings(_testing_path=tmp_path)
        # Add a camera.
        item = Camera.model_validate_json(CAMERA)
        saved_settings.add_item(item)
        # Add the same camera again.
        with pytest.raises(
            ValueError, match="Aspen CG 16m already exists in cameras.json"
        ):
            saved_settings.add_item(item)

    def test_adding_multiple_types_of_items(self, tmp_path):
        # Test that adding multiple types of items works.
        saved_settings = SavedSettings(_testing_path=tmp_path)
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

    def test_delete_without_confirm_raises_error(self, tmp_path):
        # Trying to delete settings without confirming should raise an error.
        saved_settings = SavedSettings(_testing_path=tmp_path)
        with pytest.raises(ValueError, match="You must confirm deletion by passing"):
            saved_settings.cameras.delete()

    def test_delete_with_confirm_deletes_file(self, tmp_path):
        # Test that deleting a settings file works.
        saved_settings = SavedSettings(_testing_path=tmp_path)
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

    def test_delete_all_settings_with_confirm_deletes_files(self, tmp_path):
        # Test that deleting all settings files works.
        saved_settings = SavedSettings(_testing_path=tmp_path)
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

    def test_delete_all_with_no_settings_works(self, tmp_path):
        # Test that deleting all settings files works when no settings are present.
        saved_settings = SavedSettings(_testing_path=tmp_path)
        # Delete all settings but not the settings folder
        saved_settings.delete(confirm=True)
        assert len(list(saved_settings.settings_path.glob("*"))) == 0

        # Delete all settings and the settings folder
        saved_settings.delete(confirm=True, delete_settings_folder=True)
        assert not saved_settings.settings_path.exists()

    def test_delete_item_from_collection_works(self, tmp_path):
        # Test that deleting an item from a collection works.
        saved_settings = SavedSettings(_testing_path=tmp_path)
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

    def test_revtrieving_item_by_name_works(self, tmp_path):
        # Test that retrieving an item by name works.
        saved_settings = SavedSettings(_testing_path=tmp_path)
        # Add a camera.
        camera = Camera.model_validate_json(CAMERA)
        saved_settings.add_item(camera)
        # Retrieve the camera by name.
        retrieved_camera = saved_settings.cameras.get(camera.name)
        assert retrieved_camera == camera
