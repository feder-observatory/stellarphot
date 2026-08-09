Adding new settings
===================

We use combination of `pydantic`_ and `ipyautoui`_ to handle settings in
``stellarphot``. The settings are defined in the ``settings/models.py`` file. Try to
group settings in a logical way. The settings related to photometry are grouped
together in a single `stellarphot.settings.PhotometrySettings`  class.

Typically to add new settings you do not need to do much beyond adding a
new class in ``settings/models.py`` and modifying any code that uses those settings to
take the new settings object as an argument. The graphical notebook interface
is generated on the fly from the settings object, so you do not need to write much
new code for that.

For example, the `stellarphot.settings.ApertureSettings` class defines the settings
related to the aperture photometry. It is used in
`stellarphot.gui.SeeingProfileWidget`. The relevant lines are:

.. code-block:: python

    self.aperture_settings = ui_generator(ApertureSettings)
    self.aperture_settings.show_savebuttonbar = True
    self.aperture_settings.path = Path(self.aperture_settings_file_name.value)
    self.save_aps = ipw.Button(description="Save settings")
    vb.children = [self.aperture_settings_file_name, self.aperture_settings] #, self.save_aps] #, self.in_t, self.out_t]


Settings format versioning
==========================

Saved photometry settings files carry a ``settings_version`` field, an integer
that identifies the version of the on-disk settings *format* (it is not the
stellarphot version). The field has *minimum-reader* semantics: it records the
format at which the schema of the saved model last changed, meaning "you must
understand format ≥ N to read this file" — not "the format that was current
when the file was written". Files written before the field existed are format
``1``; the current format version is ``PHOTOMETRY_SETTINGS_FORMAT_VERSION`` in
``settings/models.py``.

When ``PhotometryWorkingDirSettings.load`` reads a settings file it passes the
validation context ``{"settings_file": True}``, which enables the migration
logic in ``PhotometrySettings._migrate_settings_file``. The context gate
matters: settings constructed in code (for example from GUI widget values)
also lack a ``settings_version`` key and must *not* be treated as old files.
A file with a ``settings_version`` newer than the code understands raises
``NewerFormatError`` on any validation path, so it is neither misread nor
overwritten on save.

This in-file field is the *only* settings versioning mechanism.
``SETTINGS_FILE_VERSION`` in ``settings/settings_files.py``, a component of
the path to the global camera/observatory/passband-map store, is frozen at
``"2"`` permanently: bumping it would point stellarphot at an empty directory,
silently orphaning every user's saved cameras, observatories and passband
maps. The files in the global store (``cameras.json``, ``observatories.json``,
``passband_maps.json``) are format ``1`` by definition until their schemas
first change; they gain a ``settings_version`` field only at that time, since
adding it earlier would itself break older readers via ``extra="forbid"``.

If you change the *meaning* of any saved setting:

#. Increment ``PHOTOMETRY_SETTINGS_FORMAT_VERSION``.
#. Set the stamped ``settings_version`` default *only* on the models whose
   schema actually changed, so files saved by unchanged models remain readable
   by older stellarphot versions (minimum-reader semantics).
#. Extend ``PhotometrySettings._migrate_settings_file`` to migrate files
   written in the older formats, warning with
   ``PhotometrySettingsMigrationWarning`` if a saved value is modified.
   That warning is a ``PhotometrySettingsWarning``, the category the
   ``ReviewSettings`` widget displays in a banner to the user.
#. Add tests to ``TestPriorVersionsCompatibility`` in
   ``settings/tests/test_models.py`` and to the format-version tests in
   ``settings/tests/test_settings_file.py``.
#. Never change ``SETTINGS_FILE_VERSION`` (the directory path component).

Format ``2`` covers both the introduction of ``settings_version`` and the
change to gap/annulus-width semantics for variable apertures (issue ``#654``);
those ship together in stellarphot 2.2.0 with no separate bump.

.. _pydantic: https://docs.pydantic.dev/latest/
.. _ipyautoui: https://maxfordham.github.io/ipyautoui/
