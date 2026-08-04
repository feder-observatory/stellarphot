import re
import warnings
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
    "PhotometrySettingsWarning",
    "PhotometryWorkingDirSettings",
]

# We will have to version settings formats, I think. Hopefully this changes rarely
# or never.
SETTINGS_FILE_VERSION = "2"  # value chosen to match major version of stellarphot

ENCODING = "utf-8"


class PhotometrySettingsWarning(UserWarning):
    """
    Warning category for user-actionable problems with saved photometry
    settings files (e.g. a settings-format migration). Warnings of this
    category are shown to the user in the `~stellarphot.gui.ReviewSettings`
    banner; plain `UserWarning`\\s from libraries on the load path are not.
    """


class SavedFileOperations:
    # Provide a place to store the path to the settings file. Annotate as a ClassVar
    # so that pydantic doesn't think it is a field. Also mark it as private to
    # discourage direct access.
    _settings_path: ClassVar = None

    def save(self):
        file_path = self._settings_path / self._file_name
        json_data = self.model_dump_json(indent=4)
        with file_path.open("w", encoding=ENCODING) as f:
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

    @property
    def full_settings_unreadable(self):
        """
        True when the full settings file existed but could not be read by the
        most recent `load` on this instance. Always False before `load` has
        been called. The flag reflects that load, not the live state of the
        disk: it is not cleared by the file being fixed or removed, and not
        set by a bad file appearing, until the next `load`.
        """
        return self._full_settings_unreadable

    @property
    def partial_settings_unreadable(self):
        """
        True when the partial settings file existed but could not be read by
        the most recent `load` on this instance. Always False before `load`
        has been called. The flag reflects that load, not the live state of
        the disk: it is not cleared by the file being fixed or removed, and
        not set by a bad file appearing, until the next `load`.
        """
        return self._partial_settings_unreadable

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

    def _move_aside(self, path):
        """
        Rename ``path`` to a backup name that does not already exist and
        return the backup path.

        The first available of ``path.bak``, ``path.bak1``, ``path.bak2``,
        ... is used, so an earlier backup -- which may hold the only copy of
        settings from a previous corruption -- is never overwritten.
        """
        backup = path.with_name(path.name + ".bak")
        counter = 0
        while backup.exists():
            counter += 1
            backup = path.with_name(f"{path.name}.bak{counter}")
        # replace() gives deterministic cross-platform behavior. In the
        # (unlikely) race where the chosen backup name is created between
        # the exists() check above and this call, the just-created file is
        # silently overwritten -- a narrow window this single-user,
        # widget-driven code accepts. rename() would not close it either:
        # os.rename also silently replaces an existing file on POSIX.
        path.replace(backup)
        return backup

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
        place; it is renamed with a ``.bak`` suffix before the new settings are
        written. A readable partial settings file whose non-None values are
        not all carried into the full settings being written is likewise
        renamed with a ``.bak`` suffix, rather than deleted, when a save
        writes full settings -- this includes, but is not limited to, the
        case where the partial settings conflict with the full settings.
        Partial values that differ from the written settings only because
        the ``settings`` argument explicitly replaced them do not trigger
        the backup: the caller supplied the new value deliberately.
        Existing backups are never overwritten; if a ``.bak`` file already
        exists, numbered suffixes (``.bak1``, ``.bak2``, ...) are used
        instead.
        """
        full_settings = False

        try:
            # This load is internal bookkeeping. Library warnings it
            # generates were already surfaced when the settings were first
            # loaded, so don't repeat them on every save -- but let
            # PhotometrySettingsWarning through, so a caller whose first
            # interaction with the settings is a save() still sees
            # user-actionable problems such as a settings-format migration.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                warnings.simplefilter("default", PhotometrySettingsWarning)
                _ = self.load()
        except ValueError:
            # load() raises ValueError when there are no settings files at
            # all, when a file exists but cannot be read, and when readable
            # partial and full settings conflict. The save proceeds in every
            # one of those cases; the unreadable-file cases rely on the
            # snapshot below so the problem file is set aside as .bak
            # rather than destroyed.
            pass

        # Whether each existing settings file failed to parse during the
        # bookkeeping load above. load() latches these flags and parses both
        # files before raising, so they are accurate even when only one of
        # the two files is unreadable. They decide below whether a file
        # about to be replaced is renamed aside instead of destroyed.
        unreadable_full = self.full_settings_unreadable
        unreadable_partial = self.partial_settings_unreadable

        match settings:
            case PartialPhotometrySettings():
                # This case MUST come first, because PartialPhotometrySettings is a
                # subclass of PhotometrySettings.

                # Values the caller explicitly provided. Used at disposal
                # time below: a partial value on disk that differs from the
                # written settings only because the caller replaced it is
                # deleted rather than backed up -- replacing it was the
                # point of the save.
                explicitly_replaced = {
                    k for k, v in settings.model_dump().items() if v is not None
                }

                # Are there already full settings?
                if self._settings_file.exists():
                    if not update:
                        # If so, we can't save partial settings if the update flag
                        # is False.
                        if unreadable_full:
                            raise ValueError(
                                "Cannot save partial settings: a full settings file "
                                f"exists at {self._settings_file} but could not be "
                                "read. Fix or remove that file, or save with "
                                "update=True, which saves the partial settings and "
                                "leaves the unreadable file in place."
                            )
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
                    # suffix before anything overwrites it.

                # Are we updating or replacing partial settings?
                if update and self._partial_settings is not None:
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
                # A full-settings save replaces everything wholesale; it
                # carries no signal about which partial values the caller
                # looked at, so none are treated as deliberate replacements
                # and any differing partial value still gets a backup.
                explicitly_replaced = set()
                self._settings = settings
                file = self._settings_file
                full_settings = True

            case _:
                raise ValueError(
                    "Settings must be PhotometrySettings or PartialPhotometrySettings, "
                    f"not {type(settings)}"
                )

        # If the file we are about to write exists but could not be read, rename
        # it aside so its contents are preserved rather than overwritten. This
        # must happen before the write below, which would clobber the file.
        if file == self._settings_file and unreadable_full:
            self._move_aside(self._settings_file)
        elif file == self._partial_settings_file and unreadable_partial:
            self._move_aside(self._partial_settings_file)

        # Write the settings to a file. The settings themselves are models, so we
        # are guaranteed to write the correct model type (partial or full settings)
        # to the file. Write to a temporary file in the same directory and then
        # atomically replace the target, so that a failure at any point during
        # serialization or writing leaves any existing settings file untouched.
        json_data = settings.model_dump_json(indent=4)
        tmp_file = file.with_name(file.name + ".tmp")
        try:
            tmp_file.write_text(json_data, encoding=ENCODING)
            tmp_file.replace(file)
        finally:
            # After a successful replace the temporary file no longer exists,
            # so this only cleans up after a failed write.
            tmp_file.unlink(missing_ok=True)

        if full_settings:
            # Now that the full settings that supersede the partial settings
            # are safely on disk, the partial settings file can be disposed
            # of. Doing this only after a successful write means a failed
            # write cannot destroy the partial settings, which may be the
            # only copy of some values.
            #
            # Preserve the on-disk partial settings as .bak, instead of
            # deleting them, whenever some of their information would
            # otherwise be lost: either the file could not be read at all, or
            # it is readable but has a non-None value that differs from what
            # was just written as the full settings. self._partial_settings
            # still reflects whatever was loaded from disk at the top of this
            # method -- none of the full-settings code paths above reassign it
            # -- so this comparison is against the pre-save partial settings.
            written = settings.model_dump()
            partial_values_lost = self._partial_settings is not None and any(
                v is not None and written.get(k) != v and k not in explicitly_replaced
                for k, v in self._partial_settings.model_dump().items()
            )
            if unreadable_partial or partial_values_lost:
                self._move_aside(self._partial_settings_file)
            else:
                self._partial_settings_file.unlink(missing_ok=True)
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
            If no settings file exists, if a settings file exists but cannot
            be read, or if the partial and full settings files are both
            readable but conflict with each other.
        """
        # Assume we have nothing to begin....
        self._partial_settings = None
        self._settings = None
        self._full_settings_unreadable = False
        self._partial_settings_unreadable = False

        if not (self._settings_file.exists() or self._partial_settings_file.exists()):
            raise ValueError(f"Settings file {self._settings_file} does not exist")

        errors = []
        last_exc = None

        # Load PartialPhotometrySettings first, if it exists. A read failure
        # (ValidationError, or an OSError/UnicodeDecodeError while opening or
        # decoding the file) leaves self._partial_settings as None, exactly
        # like the missing/unparseable-content case below.
        if self._partial_settings_file.exists():
            try:
                with self._partial_settings_file.open(encoding=ENCODING) as f:
                    content = f.read()
                self._partial_settings = PartialPhotometrySettings.model_validate_json(
                    content
                )
            except (ValidationError, OSError, UnicodeDecodeError) as e:
                self._partial_settings_unreadable = True
                errors.append(f"Error loading partial settings: {e}")
                last_exc = e

        # Now load full settings if they exist
        if self._settings_file.exists():
            try:
                with self._settings_file.open(encoding=ENCODING) as f:
                    content = f.read()
                self._settings = PhotometrySettings.model_validate_json(content)
            except (ValidationError, OSError, UnicodeDecodeError) as e:
                self._full_settings_unreadable = True
                errors.append(f"Error loading settings: {e}")
                last_exc = e

        # Both files are parsed before any error is raised so that the
        # in-memory settings reflect every file that could be read; save()
        # relies on that to merge into readable settings and to rename only
        # genuinely unreadable files to .bak.
        if errors:
            raise ValueError("\n".join(errors)) from last_exc

        # Handle case where we have valid partial and valid full settings
        self._resolve_full_partial_conflict()
        return self._settings or self._partial_settings

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
                "Partial settings and full settings do not match.\n"
                "Saving full settings will resolve the conflict; the partial "
                "settings file will be preserved with a .bak suffix.\n"
                f"Folder with settings: {self._working_dir}\n"
                f"Partial settings: {self._partial_settings_file}\n"
                f"Full settings: {self._settings_file}"
            )

        # If we reach here, then the partial settings and full settings match, so we
        # can delete the partial settings.
        self._partial_settings_file.unlink()

        # and set the partial settings to None
        self._partial_settings = None
