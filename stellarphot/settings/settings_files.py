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
    def save(self, path: Path):
        file_path = path / self._file_name
        json_data = self.model_dump_json(indent=4)
        with file_path.open("w") as f:
            f.write(json_data)

    @classmethod
    def load_model(cls, path):
        file_path = path / cls._file_name
        if not file_path.exists():
            return cls(items={})
        with file_path.open() as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def delete(cls, settings_path, confirm=False):
        """
        Delete the settings file for this class.

        Parameters
        ----------
        confirm : bool, optional
            If True, the file is deleted. If False, a ValueError is raised.
        """
        if confirm:
            file_path = settings_path / cls._file_name
            file_path.unlink(missing_ok=True)
        else:
            raise ValueError("You must confirm deletion by passing confirm=True")


class Cameras(SavedFileOperations, BaseModel):
    # Using the ClassVar annotation means this is treated as a class variable rather
    # than a pydantic field. We don't pydantic storing the name of the settings file in
    # the settings file itself.
    _file_name: ClassVar[str] = "cameras.json"
    "Name of the file where the cameras are saved."

    items: dict[str, Camera]
    "Dictionary of cameras, keyed by camera name."


class Observatories(SavedFileOperations, BaseModel):
    _file_name: ClassVar[str] = "observatories.json"
    "Name of the file where the observatories are saved."
    items: dict[str, Observatory]
    "Dictionary of observatories, keyed by observatory name."


class PassbandMaps(SavedFileOperations, BaseModel):
    _file_name: ClassVar[str] = "passband_maps.json"
    "Name of the file where the passband maps are saved."
    items: dict[str, PassbandMap]
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
        return Cameras.load_model(self.settings_path)

    @property
    def observatories(self) -> Observatories:
        """
        Observatories stored in the settings.
        """
        return Observatories.load_model(self.settings_path)

    @property
    def passband_maps(self) -> PassbandMaps:
        """
        Passband maps stored in the settings.
        """
        return PassbandMaps.load_model(self.settings_path)

    def get_items(self, item_type):
        """
        Get the items of a given type.

        Parameters
        ----------
        item_type : str | Camera | Observatory | PassbandMap
            The type of item to get.
        """
        match item_type:
            case Camera() | "camera":
                return self.cameras
            case Observatory() | "observatory":
                return self.observatories
            case PassbandMap() | "passband_map":
                return self.passband_maps
            case _:
                raise ValueError(f"Unknown item type {item_type}")

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
                raise ValueError("Unknown item type")

        if to_add.name in container.items:
            raise ValueError(f"{to_add.name} already exists in {container._file_name}")

        container.items[to_add.name] = to_add
        container.save(self.settings_path)

    def delete(self, confirm=False):
        """
        Delete all settings files.

        Parameters
        ----------
        confirm : bool, optional
            If True, the files are deleted. If False, a ValueError is raised.
        """
        if not confirm:
            raise ValueError("You must confirm deletion by passing confirm=True")
        self.cameras.delete(self.settings_path, confirm=confirm)
        self.observatories.delete(self.settings_path, confirm=confirm)
        self.passband_maps.delete(self.settings_path, confirm=confirm)
        self.settings_path.rmdir()
