1.4.13 (2024-02-14)
------------------

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
+ Allow exposure keyword in FITS header to be set.

1.4.12 (2024-02-12)
------------------

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
+ Astropy 6 compatibility.

1.4.11 (2023-11-29)
------------------

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
+ Fix minor errors notebook 06 and 07.

1.4.10 (2023-11-17)
------------------

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
+ Make it easier to exclude data form transit analysis.

1.4.9 (2023-11-08)
------------------

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

+ Add more file chooser dialogs and other simplifications.

1.4.8 (2023-10-27)
------------------

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

+ Use `astropy.timeseries.LombScargle` instead of `gatspy` for periodogram.

1.4.7 (2023-10-20)
------------------

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

+ Add option to display predetermined label names in `ComparisonViewer.show_labels`.

1.4.6 (2023-09-29)
------------------

Bug Fixes
^^^^^^^^^

+ Fix field length in AAVSO writer. [#172]
+ Fix issue in ``TessTargetFile`` on Windows. [#171]

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

+ Add a number to the name of the photometry transform notebook. [#173]

1.4.5 (2023-09-27)
------------------

Bug Fixes
^^^^^^^^^

+ Re-update imports in ``transform_pared_back.ipynb``.

1.4.4 (2023-09-27)
------------------

Bug Fixes
^^^^^^^^^

+ Update imports in ``transform_pared_back.ipynb``.

1.4.3 (2023-09-27)
------------------

Bug Fixes
^^^^^^^^^

+ Do not use ``Quantity`` in boolean comparisons. [#170]


1.4.2 (2023-08-14)
------------------

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

+ Do release from proper branch.

1.4.1 (2023-08-14)
------------------

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

+ Include AAVSO file format description. [#155]

Bug Fixes
^^^^^^^^^

1.4.0 (2023-08-03)
------------------

New Features
^^^^^^^^^^^^

+ Add class for writing AAVSO files. [#146]


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
