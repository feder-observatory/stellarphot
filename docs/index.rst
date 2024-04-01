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



Graphical interface for generating settings
-------------------------------------------

To generate settings using a graphical interface, start JupyterLab. In the launcher will be a section called
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
  stellarphot/settings.rst


Developer Documentation
=======================

.. toctree::
   :maxdepth: 1

   dev/index.rst
