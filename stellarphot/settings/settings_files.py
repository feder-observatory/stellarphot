from pathlib import Path
from typing import ClassVar

from platformdirs import PlatformDirs
from pydantic import BaseModel

from .models import Camera, Observatory, PassbandMap

# We will have to version settings formats, I think. Hopefully this changes rarely
# or never.
SETTINGS_FILE_VERSION = "2"  # value chosen to match amjor version of stellarphot


class SavedFileOperations:
    @classmethod
    def save(cls, path, model):
        file_path = path / cls.file_name
        json_data = model.model_dump_json(indent=4)
        with file_path.open("w") as f:
            f.write(json_data)

    @classmethod
    def load_model(cls, path):
        file_path = path / cls.file_name
        if not file_path.exists():
            return cls(items={})
        with file_path.open() as f:
            return cls.model_validate_json(f.read())


class Cameras(SavedFileOperations, BaseModel):
    # Using the ClassVar annotation means this is treated as a class variable rather
    # than a pydantic field. We don't pydantic storing the name of the settings file in
    # the settings file itself.
    file_name: ClassVar[str] = "cameras.json"
    items: dict[str, Camera]


class Observatories(BaseModel):
    file_name: ClassVar[str] = "observatories.json"
    items: dict[str, Observatory]


class PassbandMaps(BaseModel):
    file_name: ClassVar[str] = "passband_maps.json"
    items: dict[str, PassbandMap]


class SavedSettings:
    def __init__(self):
        data_dir = PlatformDirs(
            "stellarphot", version=SETTINGS_FILE_VERSION
        ).user_data_dir
        self.settings_path = Path(data_dir)
        if not self.settings_path.exists():
            self.settings_path.mkdir(parents=True)

    @property
    def cameras(self) -> Cameras:
        self._cameras = Cameras.load_model(self.settings_path)
        return self._cameras

    def add_camera(self, camera: Camera):
        self.cameras.items[camera.name] = camera
        Cameras.save(self.settings_path, self._cameras)
