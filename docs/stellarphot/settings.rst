Settings
========

Overview
--------

Using ``stellarphot`` starts with defining a bunch of configuration settings. Some of these,
like details about your observing location and camera, will change infrequently.
Others, like what size apertures to use for photometry and where in each image those
apertures should be placed, may change from night to night.

All the settings can be made through a graphical interface, via the command line, or by editing
an existing settings file.

The settings are grouped into these categories:

- `~stellarphot.settings.Observatory`
- `~stellarphot.settings.Camera`
- `~stellarphot.settings.PassbandMap`
- `~stellarphot.settings.PhotometryApertures`
- `~stellarphot.settings.SourceLocationSettings`
- `~stellarphot.settings.PhotometryOptionalSettings`
- `~stellarphot.settings.LoggingSettings`


A copy of the settings are stored in a file called `stellarphot_settings.json` in the working directory
where you are using stellarphot. It is these settings that are used when you run the photometry.

Settings can be generated using JupyterLab with a graphical interface, by using the command line,
or by editing a settings file directly.

Settings groups
---------------


Observatory information
^^^^^^^^^^^^^^^^^^^^^^^

TBD

Camera information
^^^^^^^^^^^^^^^^^^^

TBD

Provide a source list
^^^^^^^^^^^^^^^^^^^^^^

TBD

Some optional settings
^^^^^^^^^^^^^^^^^^^^^^

TBD

Entering settings programmatically
----------------------------------

Saved settings
--------------

In addition to the settings file that is created in the working directory, you can save some settings
on your system to make it easier to reuse the settings in the future. The settings that can be saved
this way are the ones most likely to be reused: `~stellarphot.settings.Camera`, `~stellarphot.settings.Observatory`,
and `~stellarphot.settings.PassbandMap`. The ``name`` property of each of these objects is used to
identify the settings.

For example, if you have a camera that you use frequently, you can save the camera settings to a file
and then load those settings in future sessions. If you are using the JupyterLab graphical interface,
every new camera you create will be saved when you click the "Save" button. If you are working programmatically,
you can save the camera settings to a file by using the `add_item` method of a
`~stellarphot.settings.SavedSettings` object.

Suppose you have saved a camera named "My Fancy Camera". To reuse that camera in a future session, you can select
it in a dropdown in the JupyterLab graphical interface, or you can load it programmatically by using the
`get_item` method of the `~stellarphot.settings.SavedSettings` class.

An example of creating such a camera, saving it, and then loading is below::

        from stellarphot.settings import Camera, SavedSettings

        # Create a camera
        camera = Camera(
            name="My Fancy Camera",
            data_unit="adu",
            gain="1.0 electron/adu",
            read_noise="10.0 electron",
            dark_current="0.0 electron/s",
            pixel_scale="1.0 arcsec/pixel",
            max_data_value="50000 adu",
        )

        # Get access to the saved settings
        saved_settings = SavedSettings()

        # Save the camera
        saved_settings.add_item(camera)

        # Load the camera
        new_camera = saved_settings.cameras.get("My Fancy Camera")

        # Compare the two cameras to see if they are the same
        print(camera == new_camera)

More about saved settings
-------------------------

The `~stellarphot.settings.SavedSettings` class is a container for the saved settings. The location on
disk depends on the operating system. You can find the location by running the following code::

        from stellarphot.settings import SavedSettings

        saved_settings = SavedSettings()
        print(saved_settings.settings_path)

All settings can be deleted using the `delete` method of `~stellarphot.settings.SavedSettings`. You can delete
all settings, or just the camera, observatory, or passband map settings. In each case you must pass in the
argument ``confirm=True`` or an error will be raised. For example, to delete all settings::

        from stellarphot.settings import SavedSettings

        saved_settings = SavedSettings()
        saved_settings.delete(confirm=True)

Deleting just the camera settings would be done like this::

        from stellarphot.settings import SavedSettings

        saved_settings = SavedSettings()
        saved_settings.cameras.delete(confirm=True)

Finally, you can delete a single camera from the saved settings like this::

        from stellarphot.settings import SavedSettings

        saved_settings = SavedSettings()
        saved_settings.cameras.delete(name="My Fancy Camera", confirm=True)

Reference/API
=============

.. automodapi:: stellarphot.settings
