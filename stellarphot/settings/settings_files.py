import os
import re
import tempfile
from pathlib import Path
from typing import ClassVar

from platformdirs import PlatformDirs
from pydantic import BaseModel, ValidationError

from .models import (
    Camera,
    Observatory,
    PartialPhotometrySettings,
    PassbandMap,
    PhotometrySettings,
)

__all__ = [
    "SavedSettings",
    "SETTINGS_FILE_VERSION",
    "PhotometryWorkingDirSettings",
    "SettingsFileReadError",
]

# We will have to version settings formats, I think. Hopefully this changes rarely
# or never.
SETTINGS_FILE_VERSION = "2"  # value chosen to match major version of stellarphot

ENCODING = "utf-8"


class SettingsFileReadError(ValueError):
    """
    Raised when a settings file exists but cannot be read because of an
    operating-system or encoding error, as opposed to a readable file that
    contains invalid settings.

    Subclasses `ValueError` so callers that catch `ValueError` are
    unaffected, while letting a caller distinguish "your settings exist but
    could not be read" from "no or invalid settings".
    """


def _backup_path(path):
    """
    Choose a backup name for a file that does not already exist.

    Parameters
    ----------
    path : `pathlib.Path`
        The file a backup name is needed for.

    Returns
    -------
    `pathlib.Path`
        The first available of ``<name>.bak``, ``<name>.bak1``,
        ``<name>.bak2``, ... so an earlier backup -- which may hold
        the only copy of settings from a previous corruption -- is
        never overwritten.
    """
    backup = path.with_name(path.name + ".bak")
    counter = 0
    while backup.exists():
        counter += 1
        backup = path.with_name(f"{path.name}.bak{counter}")
    return backup


def _move_aside(path):
    """
    Rename a file to a backup name that does not already exist.

    Parameters
    ----------
    path : `pathlib.Path`
        The file to rename.

    Returns
    -------
    `pathlib.Path`
        The path the file was renamed to.
    """
    backup = _backup_path(path)
    # replace() gives deterministic cross-platform behavior. In the
    # (unlikely) race where the chosen backup name is created between
    # the exists() check in _backup_path and this call, the
    # just-created file is silently overwritten -- a narrow window
    # this single-user, widget-driven code accepts.
    path.replace(backup)
    return backup


def _copy_aside(path):
    """
    Copy a file to a backup name that does not already exist, leaving
    the original in place.

    Used instead of `_move_aside` when the original must survive at its
    own name until a subsequent step succeeds -- a failure after the
    copy leaves the file where it was.

    Parameters
    ----------
    path : `pathlib.Path`
        The file to copy.

    Returns
    -------
    `pathlib.Path`
        The path of the backup copy.
    """
    backup = _backup_path(path)
    # Copy bytes, not text -- the file being set aside may not be
    # decodable. Reading before the backup is created means a source
    # that cannot be read leaves nothing behind. Exclusive creation
    # ("x") preserves the guarantee that an existing backup is never
    # overwritten even if the chosen name appears between _backup_path
    # and this write.
    data = path.read_bytes()
    try:
        with backup.open("xb") as f:
            f.write(data)
    except FileExistsError:
        # The exclusive open created nothing -- the file at the backup
        # name belongs to another writer, so it must not be removed.
        raise
    except OSError:
        # A failed write leaves an empty or truncated backup that later
        # set-asides would skip past; remove it before propagating.
        backup.unlink(missing_ok=True)
        raise
    return backup


def _atomic_write_json(file, json_data, set_aside_target=False):
    """
    Write JSON content to a file so that a failure at any point leaves
    any existing file at the target untouched.

    The content goes to a temporary file in the same directory that
    atomically replaces the target on success. mkstemp creates the
    temporary file exclusively with a unique name, so overlapping writes
    cannot collide on it.

    Parameters
    ----------
    file : `pathlib.Path`
        The destination file.

    json_data : str
        The JSON content to write.

    set_aside_target : bool, optional
        If True, preserve the existing (presumably unreadable) file at
        the target as a ``.bak`` copy before it is replaced. The backup
        is a copy made between writing the temporary file and the
        replace, so a failure at either point leaves the target in place
        under its original name.
    """
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=file.parent, prefix=file.name + ".", suffix=".tmp"
    )
    os.close(tmp_fd)
    tmp_file = Path(tmp_name)
    try:
        tmp_file.write_text(json_data, encoding=ENCODING)
        if set_aside_target:
            _copy_aside(file)
        tmp_file.replace(file)
    finally:
        # After a successful replace the temporary file no longer exists,
        # so this only cleans up after a failed write.
        tmp_file.unlink(missing_ok=True)


class SavedFileOperations:
    # Provide a place to store the path to the settings file. Annotate as a ClassVar
    # so that pydantic doesn't think it is a field. Also mark it as private to
    # discourage direct access.
    _settings_path: ClassVar = None

    def save(self):
        file_path = self._settings_path / self._file_name
        json_data = self.model_dump_json(indent=4)
        _atomic_write_json(file_path, json_data)

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
            with file_path.open(encoding=ENCODING) as f:
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


class PhotometryWorkingDirSettings:
    """
    Class to save in-progress and complete photometry settings in the working directory.
    """

    def __init__(self, settings_file_name="photometry_settings.json"):
        """
        Parameters
        ----------
        settings_file_name : str, optional
            Name of the settings file. Must end with '.json', contain only
            alphanumeric characters, hyphens, underscores, and spaces and begin
            with an alphanumeric character.
        """
        self._working_dir = Path(".")
        self._check_bad_file_name(settings_file_name)
        self._settings_file = self._working_dir / Path(settings_file_name)
        self._partial_settings_file = self._working_dir / Path(
            "partial_" + settings_file_name
        )
        self._partial_settings = None
        self._settings = None
        self._full_settings_unreadable = False
        self._partial_settings_unreadable = False

    @property
    def settings(self):
        """
        The full settings, or None
        """
        return self._settings

    @property
    def partial_settings(self):
        """
        The partial settings, or None
        """
        return self._partial_settings

    # Properties for settings file and partial settings file
    @property
    def settings_file(self):
        return self._settings_file

    @property
    def partial_settings_file(self):
        return self._partial_settings_file

    def _check_bad_file_name(self, file_name):
        good_name = re.compile(r"^\w+[\w\d\-_ ]*\.json$")
        if not good_name.match(file_name):
            raise ValueError(
                f"Settings file name {file_name} is not a valid name. The name can "
                "only contain alphanumeric characters, hyphens, underscores, and "
                "spaces, and must end with '.json'"
            )

    def _are_partial_actually_full(self, settings):
        """
        Check if the partial settings are actually full settings.

        Parameters
        ----------
        settings : PartialPhotometrySettings
            The settings to check.
        """
        try:
            PhotometrySettings.model_validate(settings.model_dump())
        except ValidationError:
            return False
        else:
            return True

    def _update_settings_from_partial(self, disk_settings, partial_settings):
        """
        Update the settings on disk, which may be full or partial, with the partial
        settings.

        Parameters
        ----------
        disk_settings : PhotometrySettings or PartialPhotometrySettings
            The settings on disk.

        partial_settings : PartialPhotometrySettings
            The partial settings to update with.

        Returns
        -------
        PhotometrySettings pr PartialPhotometrySettings
            The updated settings. The return type is the same as the type
            of disk_settings.
        """
        # Grab a dict of the settings, only keeping values that are not
        # None. Making it dict so that the update method can be used to
        # merge the two sets of settings.

        passed_partial_settings = {
            k: v for k, v in partial_settings.model_dump().items() if v is not None
        }

        # The order matters here. Keys in the argument passed_partial_settings
        # will overwrite the keys in disk settings.
        # Note that update works in-place.
        disk_settings.update(passed_partial_settings)

        return disk_settings

    def save(self, settings, update=False):
        """
        Save the partial or full settings to the working directory. Note well that
        this removes any partial settings file if called with valid full settings.

        Parameters
        ----------
        settings : PhotometrySettings
            The settings to save.

        update : bool, optional
            If True, the settings are updated -- see Note below for more description.
            If False, the settings are overwritten.

        Notes
        -----

        If ``update`` is True, then the settings are updated. This means that if the
        settings passed in are partial and there is already a partial setting saved on
        disk, then the settings from disk that are not in the new settings are added to
        the new settings.

        This also means that in the event that the settings in the argument have, say,
        a `~stellarphot.settings.Camera`, and the file on disk also has one, the one
        in the argument is the one that will be kept.

        Finally, if we are passed a partial setting and there is a full setting on disk,
        then the partial settings are merged with the full settings, and the full
        settings are saved.

        An existing settings file that cannot be read is never overwritten in
        place; it is preserved under a ``.bak`` name before the new settings
        replace it, and a save also sets aside, with the same ``.bak``
        naming, any settings file its pre-save load found unreadable even
        when the save does not rewrite that particular file. Existing backups
        are never overwritten; if a ``.bak`` file already exists, numbered
        suffixes (``.bak1``, ``.bak2``, ...) are used instead. The settings
        are written to a temporary file that atomically replaces the target,
        and the partial settings file is disposed of only after new full
        settings are safely on disk, so a failed write cannot truncate or
        destroy existing settings.
        """
        full_settings = False

        try:
            _ = self.load()
        except ValueError:
            # load() raises ValueError when there are no settings files at
            # all and when a file exists but cannot be read. The save
            # proceeds in both cases; the unreadable-file case relies on the
            # flags below so the problem file is set aside as .bak rather
            # than destroyed.
            pass

        # Whether each existing settings file failed to parse during the
        # bookkeeping load above. load() records these flags on the instance
        # before raising, and it parses both files before raising, so the
        # flags are accurate even when only one of the two files is
        # unreadable. They decide below whether a file about to be replaced
        # is set aside as a backup instead of destroyed.
        unreadable_full = self._full_settings_unreadable
        unreadable_partial = self._partial_settings_unreadable

        # When both files were readable, load() either removed a partial
        # file that matched the full settings or raised because the two
        # conflict. Both still being loaded therefore means the partial
        # settings conflict with the full settings: their values must not
        # leak into what is saved, and the partial file -- which may hold
        # the only copy of the conflicting values -- is preserved as a
        # backup below instead of deleted.
        conflicting_partial = (
            self._settings is not None and self._partial_settings is not None
        )

        match settings:
            case PartialPhotometrySettings():
                # This case MUST come first, because PartialPhotometrySettings is a
                # subclass of PhotometrySettings.

                # Are there already full settings?
                if self._settings_file.exists():
                    if not update:
                        # If so, we can't save partial settings if the update flag
                        # is False.
                        raise ValueError(
                            "Cannot save partial settings when full "
                            "settings already exist."
                        )
                    elif self._settings is not None:
                        # Load the full settings and update them with the
                        # partial settings
                        disk_settings = self._settings.model_dump()

                        disk_settings = self._update_settings_from_partial(
                            disk_settings, settings
                        )

                        disk_settings = PhotometrySettings.model_validate(disk_settings)

                        settings = disk_settings
                    # If the settings file exists but could not be read then
                    # self._settings is None and there is nothing to merge the
                    # partial settings into. The partial settings are saved on
                    # their own and the unreadable file is renamed with a .bak
                    # suffix once the new settings are safely on disk.

                # Are we updating or replacing partial settings? A
                # conflicting partial is skipped: the settings here have
                # already been merged into the full settings above, and
                # merging them into the stale partial would resurrect any
                # of its values the full settings legitimately set to None.
                if (
                    update
                    and not conflicting_partial
                    and (self._partial_settings is not None)
                ):
                    # Get the partial settings that were loaded from disk
                    existing_partial_settings = self._partial_settings.model_dump()

                    # Update the partial settings with the new settings
                    existing_partial_settings = self._update_settings_from_partial(
                        existing_partial_settings, settings
                    )

                    # Validate the updated partial settings
                    settings = PartialPhotometrySettings.model_validate(
                        existing_partial_settings
                    )
                # set variable file to point to appropriate (partial or full)
                # settings file location.
                if self._are_partial_actually_full(settings):
                    self._settings = settings
                    file = self._settings_file
                    full_settings = True
                else:
                    # Update the partial settings with the new settings
                    self._partial_settings = settings
                    file = self._partial_settings_file
            case PhotometrySettings():
                self._settings = settings
                file = self._settings_file
                full_settings = True

            case _:
                raise ValueError(
                    "Settings must be PhotometrySettings or PartialPhotometrySettings, "
                    f"not {type(settings)}"
                )

        # If the file we are about to write exists but could not be read, its
        # contents must be set aside rather than overwritten. The backup is a
        # copy made between writing the temporary file and the replace below,
        # so a failure at either point leaves the unreadable file in place
        # under its original name instead of leaving no settings file at all.
        set_aside_target = (file == self._settings_file and unreadable_full) or (
            file == self._partial_settings_file and unreadable_partial
        )

        # Write the settings to a file. The settings themselves are models, so we
        # are guaranteed to write the correct model type (partial or full settings)
        # to the file.
        json_data = settings.model_dump_json(indent=4)
        _atomic_write_json(file, json_data, set_aside_target=set_aside_target)

        if not full_settings and unreadable_full and self._settings_file.exists():
            # This save wrote only the partial file, but the bookkeeping load
            # found the full settings file unreadable. Leaving that file
            # behind would keep every future load failing, so it is set
            # aside now that the new partial settings are safely on disk.
            _move_aside(self._settings_file)
            self._settings = None

        if full_settings:
            # Now that the full settings that supersede the partial settings
            # are safely on disk, the partial settings file can be disposed
            # of. Doing this only after a successful write means a failed
            # write cannot destroy the partial settings. An unreadable or
            # conflicting partial settings file is preserved as .bak instead
            # of deleted -- it may hold the only copy of some values.
            if self._partial_settings_file.exists():
                if unreadable_partial or conflicting_partial:
                    _move_aside(self._partial_settings_file)
                else:
                    self._partial_settings_file.unlink()
            self._partial_settings = None

    def load(self):
        """
        Load full or partial settings.

        Returns
        -------
        PhotometrySettings | PartialPhotometrySettings
            The settings loaded from disk.

        Raises
        ------
        ValueError
            If no settings file exists, if a settings file can be read but
            contains invalid settings, or if the partial and full settings
            files are both readable but conflict with each other.
        SettingsFileReadError
            If a settings file exists but cannot be read because of an
            operating-system or encoding error. This is a subclass of
            `ValueError`.
        """
        # Assume we have nothing to begin....
        self._partial_settings = None
        self._settings = None
        self._full_settings_unreadable = False
        self._partial_settings_unreadable = False

        if not (self._settings_file.exists() or self._partial_settings_file.exists()):
            raise ValueError(f"Settings file {self._settings_file} does not exist")

        # Load PartialPhotometrySettings first, if it exists. A failure
        # leaves self._partial_settings as None, exactly like the
        # missing-file case.
        self._partial_settings, partial_exc = self._try_load(
            self._partial_settings_file, PartialPhotometrySettings
        )
        self._partial_settings_unreadable = partial_exc is not None

        # Now load full settings if they exist
        self._settings, full_exc = self._try_load(
            self._settings_file, PhotometrySettings
        )
        self._full_settings_unreadable = full_exc is not None

        # Both files are parsed before any error is raised so that the
        # in-memory settings reflect every file that could be read; save()
        # relies on that to merge into readable settings and to set aside
        # only genuinely unreadable files as .bak.
        errors = []
        if partial_exc is not None:
            errors.append(f"Error loading partial settings: {partial_exc}")
        if full_exc is not None:
            errors.append(f"Error loading settings: {full_exc}")
        if errors:
            # An OS-level or encoding failure means the settings themselves
            # may be perfectly valid, so it is reported with a distinct
            # (ValueError-subclass) exception type.
            exceptions = [e for e in (partial_exc, full_exc) if e is not None]
            error_class = (
                SettingsFileReadError
                if any(isinstance(e, (OSError, UnicodeDecodeError)) for e in exceptions)
                else ValueError
            )
            raise error_class("\n".join(errors)) from exceptions[-1]

        # Handle case where we have valid partial and valid full settings
        self._resolve_full_partial_conflict()
        return self._settings or self._partial_settings

    @staticmethod
    def _try_load(path, model_cls):
        """
        Parse one settings file.

        Parameters
        ----------
        path : `pathlib.Path`
            The settings file to parse.

        model_cls : type
            The pydantic model class to validate the file contents against.

        Returns
        -------
        model, exception : (``model_cls`` or None, Exception or None)
            The parsed settings and ``None`` on success; ``None`` and the
            exception when the file is missing or cannot be read (the
            exception is ``None`` for a missing file).
        """
        if not path.exists():
            return None, None
        try:
            with path.open(encoding=ENCODING) as f:
                return model_cls.model_validate_json(f.read()), None
        except (ValidationError, OSError, UnicodeDecodeError) as e:
            return None, e

    def _resolve_full_partial_conflict(self):
        """
        Resolve the conflict between full and partial settings, if any.

        Five cases:
        1. No partial settings, no full settings: Nothing to do.
        2. Partial settings, no full settings: Load partial settings.
        3. No partial settings, full settings: Nothing to do.
        4. Partial settings, full settings, and they match: delete partial settings.
        5. Partial settings, full settings, and they don't match: raise ValueError.
        """
        # Handle cases 1 through 3 -- no conflicts in these cases
        if self._partial_settings is None or self._settings is None:
            # Nothing to do, return
            return

        # Both are not None, so try construction full from partial, since partial
        # settings can be full.
        try:
            full_from_partial = PhotometrySettings.model_validate(
                self._partial_settings.model_dump()
            )
        except ValidationError:
            full_from_partial = None

        if full_from_partial != self._settings:
            raise ValueError(
                "Partial settings and full settings do not match. "
                "Please resolve the discrepancy by deleting one of the "
                "settings files."
                f"Folder with settings: {self._working_dir}"
                f"Partial settings: {self._partial_settings_file} "
                f"Full settings: {self._settings_file}"
            )

        # If we reach here, then the partial settings and full settings match, so we
        # can delete the partial settings.
        self._partial_settings_file.unlink()

        # and set the partial settings to None
        self._partial_settings = None
