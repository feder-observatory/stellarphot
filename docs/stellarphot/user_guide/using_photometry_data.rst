Working with  Photometry Data
#############################


Photometry data attributes
--------------------------

`~stellarphot.PhotometryData` objects always have these attributes, also available dictionary-style:

+ ``coord`` -- an astropy `~astropy.coordinates.SkyCoord` object with the coordinates for each star
+ ``mag_inst`` -- the instrumental magnitude
+ ``mag_inst_err`` -- error in the instrument magnitude
+ ``passband`` -- the filter of the magnitude; required
+ ``bjd`` -- Barycentric Julian Date of the midpoint of the observation; may be missing/masked

These objects are also astropy tables, and there is much more information
available in the table beyond these attributes.

Generating a light curve from photometry data
---------------------------------------------

One of the more common tasks when performing photometry is generating a light curve
for one or more of the objects in the field, i.e. magnitude or flux vs time for that
object. The `~stellarphot.PhotometryData.lightcurve_for` method lets you generate a
`lightkurve.LightCurve` object for any of the stars in your field. You can specify the
name of the star, the coordinates of the star or the ``star_id`` assigned by stellarphot.
See the lightkurve documentation for examples of how to plot a light curve,
perform a periodogram on it, fold it, and more.
