*************************
stellarphot Documentation
*************************

This is the documentation for stellarphot.

Photometry objects always have these attributes, also available dictionary-style:

+ ``mag`` -- the calibrated magnitude; may be missing (e.g. your data prior to calibration)
+ ``mag_err`` -- error in ``mag``
+ ``inst_mag`` -- the instrumental magnitude; may be missing (e.g. catalog data)
+ ``inst_mag_err`` -- error int the ``inst_mag``
+ ``band`` -- the filter of the magnitude; required
+ ``BJD`` -- Barycentric Julian Date of the midpoint of the observation; may be missing/masked
+ ``RA2000`` -- right ascension in degrees, in the ICRS frame at epoch 2000
+ ``DEC2000`` -- declination in degrees in the ICRS frame at epoch 2000

There may be additional fields.

Reference/API
=============

.. automodapi:: stellarphot
.. automodapi:: stellarphot.photometry
.. automodapi:: stellarphot.differential_photometry
.. automodapi:: stellarphot.visualization
.. automodapi:: stellarphot.io
