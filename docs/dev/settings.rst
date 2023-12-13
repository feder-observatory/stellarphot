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
`stellarphot.gui_tools.SeeingProfileWidget`. The relevant lines are:

.. code-block:: python

    self.aperture_settings = ui_generator(ApertureSettings)
    self.aperture_settings.show_savebuttonbar = True
    self.aperture_settings.path = Path(self.aperture_settings_file_name.value)
    self.save_aps = ipw.Button(description="Save settings")
    vb.children = [self.aperture_settings_file_name, self.aperture_settings] #, self.save_aps] #, self.in_t, self.out_t]


.. _pydantic: https://docs.pydantic.dev/latest/
.. _ipyautoui: https://maxfordham.github.io/ipyautoui/
