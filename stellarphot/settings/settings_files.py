from pathlib import Path
from typing import ClassVar

from platformdirs import PlatformDirs
from pydantic import BaseModel

from .models import Camera, Observatory, PassbandMap

__all__ = ["SavedSettings", "SETTINGS_FILE_VERSION"]

# We will have to version settings formats, I think. Hopefully this changes rarely
# or never.
SETTINGS_FILE_VERSION = "2"  # value chosen to match amjor version of stellarphot


class SavedFileOperations:
    # Provide a place to store the path to the settings file. Annotate as a ClassVar
    # so that pydantic doesn't think it is a field. Also mark it as private to
    # discourage direct access.
    _settings_path: ClassVar = None

    def save(self):
        file_path = self._settings_path / self._file_name
        json_data = self.model_dump_json(indent=4)
        with file_path.open("w") as f:
            f.write(json_data)

    def get(self, name):
        """
        Get the item with the given name.

        Parameters
        ----------
        name : str
            Name of the item to get.
        """
        return self.as_dict[name]

    @classmethod
    def load_model(cls):
        file_path = cls._settings_path / cls._file_name
        if not file_path.exists():
            instance = cls(as_dict={})
        else:
            with file_path.open() as f:
                instance = cls.model_validate_json(f.read())

        return instance

    @classmethod
    def delete(cls, confirm=False, name=None):
        """
        Delete the settings file for this class.

        Parameters
        ----------
        confirm : bool, optional
            If True, the file is deleted. If False, a ValueError is raised.

        name : str, optional
            Name of the item to delete. If provided, only the item with this name is
            deleted. If not provided, the entire file is deleted.
        """
        if not confirm:
            raise ValueError("You must confirm deletion by passing confirm=True")

        file_path = cls._settings_path / cls._file_name
        if name is not None:
            # Only delete the named item
            instance = cls.load_model()
            if name not in instance.as_dict:
                raise ValueError(f"{name} not found in {cls._file_name}")
            del instance.as_dict[name]
            instance.save()
        else:
            # Delete the entire file
            file_path.unlink(missing_ok=True)


class Cameras(SavedFileOperations, BaseModel):
    # Using the ClassVar annotation means this is treated as a class variable rather
    # than a pydantic field. We don't want pydantic storing the name of the settings
    # file in the settings file itself.
    _file_name: ClassVar[str] = "cameras.json"
    "Name of the file where the cameras are saved."

    as_dict: dict[str, Camera]
    "Dictionary of cameras, keyed by camera name."


class Observatories(SavedFileOperations, BaseModel):
    _file_name: ClassVar[str] = "observatories.json"
    "Name of the file where the observatories are saved."
    as_dict: dict[str, Observatory]
    "Dictionary of observatories, keyed by observatory name."


class PassbandMaps(SavedFileOperations, BaseModel):
    _file_name: ClassVar[str] = "passband_maps.json"
    "Name of the file where the passband maps are saved."
    as_dict: dict[str, PassbandMap]
    "Dictionary of passband maps, keyed by passband map name."


class SavedSettings:
    """
    Handle loading and saving of settings files from disk.
    """

    def __init__(self, _testing_path=None, _create_path=True):
        """
        Parameters
        ----------

        _testing_path : Path, optional
            Path to use for testing purposes. If not provided, the default path is used.

        _create_path : bool, optional
            If True, the directory where settings files are stored is created if it does
            not exist. If False, the directory is not created, which is useful for
            testing.
        """
        if _testing_path is not None:
            data_dir = _testing_path
        else:
            data_dir = PlatformDirs(
                "stellarphot", version=SETTINGS_FILE_VERSION
            ).user_data_dir
        self._settings_path = Path(data_dir)
        if _create_path:
            if not self.settings_path.exists():
                self.settings_path.mkdir(parents=True)

        # Make the path available to the SavedFileOperations classes.
        SavedFileOperations._settings_path = self.settings_path

    @property
    def settings_path(self):
        """
        Path to the directory where settings files are stored.
        """
        return self._settings_path

    @property
    def cameras(self) -> Cameras:
        """
        Cameras stored in the settings.
        """
        # Note that we always reload in case the file has changed.
        return Cameras.load_model()

    @property
    def observatories(self) -> Observatories:
        """
        Observatories stored in the settings.
        """
        return Observatories.load_model()

    @property
    def passband_maps(self) -> PassbandMaps:
        """
        Passband maps stored in the settings.
        """
        return PassbandMaps.load_model()

    def get_items(self, item_type):
        """
        Get the items of a given type.

        Parameters
        ----------
        item_type : str | Camera | Observatory | PassbandMap
            The type of item to get.
        """
        match item_type:
            case Camera() | "camera" | Camera.__name__:
                return self.cameras
            case Observatory() | "observatory" | Observatory.__name__:
                return self.observatories
            case PassbandMap() | "passband_map" | PassbandMap.__name__:
                return self.passband_maps
            case _:
                raise ValueError(
                    f"Unknown item {item_type} of type {type(item_type)}. Must be "
                    "Camera, Observatory, or PassbandMap, or "
                    "'camera', 'observatory', or 'passband_map'"
                )

    def add_item(self, item):
        """
        Add an item to the settings.

        Parameters
        ----------
        item : Camera | Observatory | PassbandMap
            The item to add.
        """
        match item:
            case Camera() as to_add:
                container = self.cameras
            case Observatory() as to_add:
                container = self.observatories
            case PassbandMap() as to_add:
                container = self.passband_maps
            case _:
                raise ValueError(
                    f"Unknown item {item} of type {type(item)}. Must be Camera, "
                    "Observatory, or PassbandMap"
                )

        if to_add.name in container.as_dict:
            raise ValueError(f"{to_add.name} already exists in {container._file_name}")

        container.as_dict[to_add.name] = to_add
        container.save()

    def delete(self, confirm=False, delete_settings_folder=False):
        """
        Delete all settings files.

        Parameters
        ----------
        confirm : bool, optional
            If True, the files are deleted. If False, a ValueError is raised.

        delete_settings_folder : bool, optional
            If True, the directory where settings files are stored is deleted. If False,
            only the settings files are deleted.
        """
        if not confirm:
            raise ValueError("You must confirm deletion by passing confirm=True")
        Cameras.delete(confirm=confirm)
        Observatories.delete(confirm=confirm)
        PassbandMaps.delete(confirm=confirm)
        if delete_settings_folder:
            self.settings_path.rmdir()

    def delete_item(self, item, confirm=False):
        """
        Delete an item from the settings.

        Parameters
        ----------
        item : Camera | Observatory | PassbandMap
            The item to delete.

        confirm : bool, optional
            If True, the item is deleted. If False, a ValueError is raised.
        """
        match item:
            case Camera() as to_delete:
                klass = Cameras
            case Observatory() as to_delete:
                klass = Observatories
            case PassbandMap() as to_delete:
                klass = PassbandMaps
            case _:
                raise ValueError(
                    f"Unknown item {item} of type {type(item)}. Must be Camera, "
                    "Observatory, or PassbandMap"
                )

        klass.delete(confirm=confirm, name=to_delete.name)
