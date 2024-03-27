Documentation
=============

Stellarphot is a package for performing photometry on calibrated (reduced) astronomical images. It
provides a simple interface for performing aperture photometry of either a single
image or a directory with multiple images. It is designed to be easy to use for both
non-programmers and programmers.


Getting Started
===============

Installation
------------

Testing
+++++++

If you are testing a pre-release version of stellarphot we recommend setting up
a virtual environment and installing stellarphot in this environment.

Creating an environment with `conda` or `mamba`:

```bash
mamba create -n stellarphot-test python=3.11
mamba activate stellarphot-test
pip install --pre --upgrade stellarphot
```

Creating an environment with `virtualenv`:

```bash
python3 -m venv stellarphot-test
source stellarphot-test/bin/activate
pip install --pre --upgrade stellarphot
```

Overview
--------

Using ``stellarphot`` starts with defining a bunch of configuration settings. Some of these,
like details about your observing location and camera, will change infrequently.
Others, like what size apertures to use for photometry and where in each image those
apertures should be placed, may change from night to night.

All the settings can be made through a graphical interface, via the command line, or by editing
an existing settings file.

The settings are grouped into these categories:

- Observatory
- Camera
- Passband Map
- Photometry Apertures
- Source Location Settings
- Optional Settings
- Logging Settings

The first three categories each include a ``name`` property, which is used to identify the
settings and provides a shortcut to re-using those settings in the future.

A copy of the settings are stored in a file called `stellarphot_settings.json` in the working directory
where you are using stellarphot. It is these settings that are used when you run the photometry.

Settings can be generated using a jupyter notebook with a graphical interface, by using the command line,
or by editing a settings file directly.

Graphical interface for generating settings
-------------------------------------------

To generate settings using a graphical interface, start Jupyter lab. In the launcher will be a section called
"Stellarphot" with a link to "Generate Settings". Clicking on this link will open a notebook where you can enter settings.

Command line interface for generating settings
----------------------------------------------

To generate settings using the command line, run the following command:

```bash
stellarphot-settings
```

This will generate a settings file in the directory in which you run the command
called `stellarphot_settings.json`. Edit that file in the editor of your choice.

Editing a settings file directly
--------------------------------

The settings file is a JSON file that can be edited in any text editor.


Provide your observatory information
-------------------------------------

.. autopydantic_field:: stellarphot.settings.Observatory.latitude
    :field-show-constraints: False
..     :model-show-config-summary: False
..     :model-show-field-summary: False
.. .. autopydantic_model:: stellarphot.settings.Observatory
..     :model-show-config-summary: False
..     :model-show-field-summary: False

Provide your camera information
-------------------------------

TBD

Provide a source list
---------------------

TBD

Some optional settings
-----------------------

TBD

Performing photometry
---------------------

Once you have made your settings doing photometry is a two line process. First, you
create a photometry object:

```python
from stellarphot.photometry import AperurePhotometry
phot = AperurePhotometry(photometry_settings)
```

Then you can perform photometry on a single image:

```python
phot(image)
```

If you have a directory of images you can perform photometry on all of them at once like this:

```python
phot(directory, object_of_interest="M13")
```


.. toctree::
  :maxdepth: 3

  stellarphot/index.rst


Developer Documentation
=======================

.. toctree::
   :maxdepth: 1

   dev/index.rst
