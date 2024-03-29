2.0.0 (unreleased)
------------------

New Features
^^^^^^^^^^^^
+ Development of new data classes for handling source list, photometry, and catalog data which include data format validation. [#125]
+ Aperture photometry streamlined into ``single_image_photometry`` and ``multi_image_photometry`` functions that use the new data classes. [#141]
+ ``multi_image_photometry`` now is a wrapper for single_image_photometry instead of a completely separate function.
+ Photometry related notebooks updated to use new data classes and new functions. [#151]
+ Logging has been implemented for photometry, so all the output can now be logged to a file. [#150]
+ Add class to hold the file locations needed for the photometry notebook. [#168]

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
+ Major reorganizaiton of code including moving functions to new modules. [#130, #133]
+ Now requires python 3.10 or later. [#147]
+ Use pydantic for aperture settings. [#154]
+ Stomped bug in handling of ``NaN``s in ``single_image_photometry``. [#157]

Bug Fixes
^^^^^^^^^
+ Fixed dependence on non-release version of astrowidgets for overwrite capability on output images. [#108]
+ Fixed computation of FWHM when fitting to data that includes NaNs. [#164]

1.3.9 (2023-06-16)
------------------

New Features
^^^^^^^^^^^^

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
+ Old and redundant notebooks have been purged and bad references to `glowing-waffles` instead of `stellarphot` [#94]
+ Most functions are now linked to the documentation. [#90]
+ Many functions and classes that had missing documentation have now had docstrings added. [#100]

Bug Fixes
^^^^^^^^^

+ Runs without errors on release version of astrowidgets (0.3.0) [#93]
+ Runs without errors on current numpy (1.24.3) and astropy (5.3). [#92]


1.1.2 (2022-10-18)
------------------

New Features
^^^^^^^^^^^^

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes
^^^^^^^^^
+ Simplify comparison notebook.


1.1.1 (2022-10-18)
------------------

New Features
^^^^^^^^^^^^


Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes
^^^^^^^^^

+ Include photometry notebooks in wheel.

1.1.0 (2022-10-18)
------------------

New Features
^^^^^^^^^^^^

+ Add two photometry notebooks and refactor underlying functions. [#73]

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes
^^^^^^^^^

1.0.4 (2022-10-13)
------------------

New Features
^^^^^^^^^^^^

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes
^^^^^^^^^

+ Ignore ``NaN``s in the calculation of AAVSO magnitudes. [#72]

1.0.3 (2022-10-08)
------------------

New Features
^^^^^^^^^^^^

+ Add equality method for AstroImageJ aperture objects. [#71]

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes
^^^^^^^^^

+ Fix error in generation of AstroImageJ data tables and aperture files. [#71]
+ Allow TIC ID numbers to have 9 or 10 digits. [#71]


1.0.2 (2022-06-01)
------------------

New Features
^^^^^^^^^^^^

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes
^^^^^^^^^

+ Pin astropy version until changes to modeling can be incorporated. [#69]

1.0.1 (2022-06-01)
------------------

New Features
^^^^^^^^^^^^

+ GUI for making EXOTIC settings. [#59]

Bug Fixes
^^^^^^^^^

+ Handle the case when no VSX variables are present in the field. [#62]

+ Exclude comparison stars from relative flux calculation if counts are ``NaN``. [#57]

+ Fix handling of comparison stars near the edge of the field of view. [#55]
