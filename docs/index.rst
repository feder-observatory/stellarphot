Documentation
=============

Stellarphot is a package for performing photometry on astronomical images. It
provides a simple interface for performing aperture photometry of either a single
image or a directory with multiple images.


Getting Started
===============

Installation
------------

Install it with pip or conda.

Overview
--------

Using ``stellarphot`` starts with defining a bunch of configuration settings. Some of these,
like details about your observing location and camera, will change infrequently.
Others, like what size apertures to use for photometry and where in each image those
apertures should be placed, may change from night to night.

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
