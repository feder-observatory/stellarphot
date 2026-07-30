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
stellarphot version). Files written before the field existed are format ``1``;
the current format version is ``PHOTOMETRY_SETTINGS_FORMAT_VERSION`` in
``settings/models.py``.

When ``PhotometryWorkingDirSettings.load`` reads a settings file it passes the
validation context ``{"settings_file": True}``, which enables the migration
logic in ``PhotometrySettings._migrate_settings_file``. The context gate
matters: settings constructed in code (for example from GUI widget values)
also lack a ``settings_version`` key and must *not* be treated as old files.
A file with a ``settings_version`` newer than the code understands raises
``NewerFormatError`` on any validation path, so it is neither misread nor
overwritten on save.

If you change the *meaning* of any saved setting:

#. Increment ``PHOTOMETRY_SETTINGS_FORMAT_VERSION``.
#. Extend ``PhotometrySettings._migrate_settings_file`` to migrate files
   written in the older formats, warning with
   ``PhotometrySettingsMigrationWarning`` if a saved value is modified.
#. Add tests to ``TestPriorVersionsCompatibility`` in
   ``settings/tests/test_models.py`` and to the format-version tests in
   ``settings/tests/test_settings_file.py``.

.. _pydantic: https://docs.pydantic.dev/latest/
.. _ipyautoui: https://maxfordham.github.io/ipyautoui/
