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

There is a graphical interface for making all of the settings below. They can also be set
programmatically in Python or by editing a file in your favorite text editor.

Provide your observatory information
-------------------------------------

TBD

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
