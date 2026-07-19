2.1.2 (2026-07-19)
-------------------

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
+ Removed the deprecated ``exo_fitting`` optional-dependency alias now that
  ``exoplanet`` has been the documented name for a full release cycle;
  nothing in the codebase, docs, or notebooks referenced the old name. [#636]
+ Fixed the PyPI homepage link (it pointed at ``stellarphot/stellarphot``
  instead of ``feder-observatory/stellarphot``) and added Documentation,
  Repository, Issues, and Changelog project URLs. [#636]
+ Bumped the minimum supported Python to 3.11 and aligned ``tool.black``'s
  target-version and ``tox.ini``'s envlist with it; CI already tested
  3.11-3.13, but black and tox still referenced 3.10. [#636]
+ Updated the stale project description and author metadata, and switched
  ``license`` to the PEP 639 SPDX string form instead of the legacy
  ``{ text = ... }`` table. [#636]
+ Added `pixi` support for developers who choose to clone this repo. [#639]
+ Updated the installation instructions in the README to reflect installation
  with `uv` and `pixi` as options. Also added badges to show what versions are
  currently available via PyPI and conda-forge. [#643]

2.1.1 (2026-07-14)
-------------------

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
+ ``refcat2`` IDs are now locally generated IAU-style coordinate designations
  (``REFCAT2SP JRRR.RRRRR±DD.DDDDD``, coordinates truncated to five decimal
  places) instead of Gaia DR2 source_ids fetched from CDS XMatch. The id
  format changes, and stars are no longer silently dropped when the Gaia
  crossmatch misses (nothing is dropped anymore). The XMatch service has no
  mirror — unlike the Vizier queries, which fall back across several servers
  — and an XMatch outage previously broke ``refcat2()`` entirely. This
  supersedes the XMatch row-index join fix from #586, since there is no
  XMatch query left to join back. [#628]
+ Coordinate-based designations now follow the IAU designation specification:
  coordinates are truncated instead of rounded, so a designation names the
  field that actually contains the star, and the RA no longer carries a
  leading ``+`` sign (the spec puts a sign only on the Dec). As a result,
  ``apass_dr9`` ids all change; ids are regenerated on every fetch, so
  nothing persisted is affected. [#628]

2.1.0 (2026-07-14)
-------------------

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
+ ``pip install stellarphot`` no longer installs the Jupyter/ipywidgets/
  ipyautoui GUI and notebook stack by default; install
  ``stellarphot[gui]`` for the notebooks, ``stellarphot[exoplanet]`` for
  exoplanet light-curve fitting, or ``stellarphot[all]`` for both. GUI and
  widget code has moved into a new ``stellarphot.gui`` subpackage (with
  back-compat shims that raise ``AstropyDeprecationWarning`` for the old
  import paths), and the core data structures can now be imported without
  pulling in the GUI stack at all. As part of this, ``stellarphot.io`` no
  longer re-exports the contents of its submodules: import directly from
  ``stellarphot.io.aavso``, ``stellarphot.io.aij``, or ``stellarphot.io.tess``.
  ``ui_generator`` is no longer importable from ``stellarphot.settings``; import
  it from ``stellarphot.settings.views``. [#566, #567, #569, #570, #571, #572]
+ The catalog-fetcher functions ``apass_dr9``, ``vsx_vizier`` and ``refcat2``
  have moved out of ``stellarphot.core`` into the new ``stellarphot.catalogs``
  module; ``stellarphot.core`` now holds the table data-structure classes only.
  They remain available as top-level names (``from stellarphot import
  apass_dr9``) and, for one or two releases, from ``stellarphot.core`` via a
  back-compat shim that raises an ``AstropyDeprecationWarning``. [#194]
+ The ``change_to_tmp_dir`` test fixture, previously duplicated in four test
  modules, is now defined once in the top-level ``conftest.py``. [#426]
+ The duplicated logging setup in ``single_image_photometry`` and
  ``multi_image_photometry`` has been factored into a shared helper. [#152]
+ The USNO u'g'r'i'z' to SDSS DR7 ugriz transform matrix has been verified
  entry-by-entry against its SDSS reference page, and tests now pin every
  coefficient of the transform. [#611]
+ Transit fitting has been rebuilt on `lmfit
  <https://lmfit.github.io/lmfit-py/>`_ (the forward model is still computed
  with pytransit). The model parameters are exposed as an ``lmfit.Parameters``
  at ``TransitModelFit.params``; ``fit()`` returns (and stores as
  ``fit_result``) an lmfit result with real per-parameter uncertainties and a
  correctly-counted BIC; and the new ``compare_detrend_options()`` compares
  the BIC of every combination of detrending parameters without corrupting
  the fit state. ``VariableArgsFitter`` and the ``model``, ``BIC`` and
  ``n_fit_parameters`` attributes are gone, ``setup_model()`` is now
  keyword-only, the no-op ``eccentricity`` parameter has been removed (the
  orbit has always been circular, and the guards added for it in [#614] are
  no longer needed), the default inclination bounds are now (50, 90) degrees,
  ``a`` is bounded below by 1, and the airmass/width/spp trend parameters
  now default to not varying in fits. The RoadRunner model is built with a
  widened radius-ratio table (``klims=(0.005, 0.6)``) so that a fitted ``rp``
  at its upper bound of 0.5 stays strictly inside the table, avoiding an
  out-of-bounds read in pytransit's native evaluator that crashed the fit on
  Windows and some macOS builds. [#625]

Bug Fixes
^^^^^^^^^
+ The photometry logging no longer clears the root logger's handlers or removes
  handlers it did not add. Stellarphot now tags its own handlers, removes only
  those, and disables propagation so its messages are not duplicated by the
  root logger. [#153]
+ The comparison-star viewer now applies its dim magnitude limit to the VSX
  variable-star lookup, so variables fainter than the limit are no longer
  marked. [#43]
+ Fixed ``TypeError`` in ``multi_image_photometry`` when exactly one source was
  missing from at least one image and ``reject_unmatched`` was enabled. [#474]
+ ``CatalogData.from_vizier`` now retries the query against a VizieR mirror
  when a server returns an empty result (as happens when a server is up but
  its database is unreachable), and raises an informative ``RuntimeError``
  instead of an ``IndexError`` if every server comes back empty. [#585]
+ ``PhotometryData.add_bjd_col`` no longer sets the BJD to NaN for the whole
  table when a single row is missing an RA or Dec value. The BJD is now
  computed for every row that has coordinates and only the rows without
  coordinates are masked in the resulting time column. A warning is issued
  when rows are skipped. [#622]
+ ``comparison_utils.in_field`` no longer swaps the x/y image bounds, which
  excluded valid comparison-star candidates near the long edge of non-square
  images (and could include off-frame stars). [#612]
+ The v1 to v2 photometry migration now writes ``NaN`` and emits a warning for
  observations whose passband has no matching ``mag_inst_<band>`` column,
  instead of silently writing ``mag_inst = 0``. [#613]
+ ``TransitModelFit.fit`` now raises an error if the ``eccentricity`` parameter
  has been un-fixed, and warns if a fixed nonzero value is set, instead of
  silently ignoring the parameter. [#614] The parameter has since been removed
  entirely as part of the lmfit rewrite. [#625]
+ ``TransitModelFit.setup_model`` no longer sets the orbital radius to
  infinity when called with the default ``duration`` of zero (the estimate is
  skipped unless a positive duration is given, and likewise for the planet
  radius when ``depth`` is zero), and a duration at least as long as the
  period now raises a clear ``ValueError``. [#625]
+ Comparing the BIC of detrending options no longer requires destructively
  refitting the model in a notebook loop; ``compare_detrend_options()`` runs
  each candidate fit on a copy of the parameters. [#625]
+ ``generate_aij_table`` no longer classifies a star as a comparison star based
  on an arbitrarily distant sky match; a match within a couple of arcseconds is
  now required. [#615]
+ ``TOI.transit_time_for_observation`` now returns the transit nearest the
  observation when the observation is before the tabulated epoch; previously
  the result could be off by up to a full period. [#616]
+ Fixed an operator-precedence bug in ``transform_to_catalog`` that disabled
  the cross-match distance cut, letting badly matched stars bias the fitted
  transform coefficients. [#617]
+ ``calc_aij_relative_flux`` now raises an error when none of the comparison
  stars match the photometry data, or when one or more times have no valid
  comparison stars, instead of silently returning relative fluxes equal to
  the raw net counts. [#618]
+ ``add_relative_flux_column`` no longer raises a ``NameError`` when the input
  photometry data already contains a ``bjd`` column. [#618]
+ The relative flux error of a comparison star is now computed against the
  same ensemble as its relative flux, with the star itself excluded from
  both the comparison counts and the comparison error, making the reported
  error and SNR consistent with the flux. Previously the error used the
  full ensemble including the star itself. [#618]
+ The per-image comparison-star consistency check in
  ``calc_aij_relative_flux`` now counts the comparison stars present in
  each image instead of counting nonzero fluxes, so a comparison star with
  exactly zero net counts (a legitimate measured value) no longer triggers
  a misleading "Different number of stars in comparison sets" error, and a
  star with zero counts now gets a finite relative flux error. [#618]
+ Saturated pixels (those above the camera's maximum data value) no longer
  silently poison the aperture sums in ``single_image_photometry``. They are
  now masked in the photometry, and sources with saturated pixels in their
  aperture are flagged in a new boolean ``saturated`` column and have their
  ``aperture_net_cnts`` set to NaN. [#591]
+ A source whose centroid cannot be computed when ``use_coordinates="sky"``
  (for example, because the source is completely saturated) no longer crashes
  ``single_image_photometry`` -- and with it an entire multi-image run -- with
  a ``ValueError``. The source now falls back to its WCS-derived position,
  like sources whose centroid shifts by more than ``shift_tolerance``. [#592]
+ With ``reject_background_outliers=False``, the reported ``sky_per_pix_med``
  and ``sky_per_pix_std`` are now computed from the pixels in the annulus
  instead of from the rectangular bounding box of the annulus, which included
  the core of the star and the corners outside the annulus. [#602]
+ ``SourceListData.drop_x_y`` now writes the NaN placeholder ``xcenter`` and
  ``ycenter`` columns with the documented ``pix`` unit instead of ``deg``,
  which had been copied from ``drop_ra_dec``. [#598]
+ ``calc_vmag`` now checks that the variable star itself has a match within
  1 arcsecond in the photometry data, warning and returning ``NaN`` when it
  does not. Previously the magnitude of the nearest unrelated star was
  silently reported as the variable's when the target was absent from the
  data. [#599]
+ ``TOI.from_tic_id`` now retrieves the information for a single TIC ID,
  including its coordinates, from ExoFOP's single-target endpoint instead of
  downloading the entire multi-megabyte TOI table (and querying MAST for the
  coordinates), which frequently timed out on CI. The parsing of the ExoFOP
  response is now covered by offline tests using a captured server response,
  and the remote-data TESS tests treat server outages and timeouts as expected
  failures instead of errors. [#623]

2.0.4 (2026-06-04)
-------------------

New Features
^^^^^^^^^^^^
+ Added AAVSO Extended File Format export for ensemble photometry:
  ``PhotometryData.write_aavso_extended`` (backed by a new
  ``stellarphot.io.write_aavso_extended``) writes a target/check-star pair
  as an AAVSO Extended File Format CSV ready for upload to WebObs, validated
  against a new ``AAVSOSubmissionHeader`` pydantic model. [#560]

Bug Fixes
^^^^^^^^^
+ Fixed TESS-target comparison-star handling: the target coordinates are no
  longer reset to the center of the frame when a name cannot be resolved,
  and the input TESS source-list order (which is not guaranteed to be
  sorted by distance from the target) is now preserved instead of being
  overwritten by a distance sort. [#562]

2.0.3 (2025-11-21)
-------------------

Bug Fixes
^^^^^^^^^
+ ``calc_aij_relative_flux`` now also excludes comparison stars that are
  missing from one or more images (previously only stars flagged bad or
  containing NaNs were excluded), so the same comparison set is used
  consistently across every image. [#549]

2.0.2 (2025-10-22)
-------------------

Bug Fixes
^^^^^^^^^
+ Updated the Gaia DR2 CDS XMatch catalog identifier used by ``refcat2``
  (``gaia_dr2_j2015p5`` to ``vizier:gaia_dr2_j2015p5``) after CDS changed the
  identifier, which had been causing the crossmatch to fail. [#545]

2.0.1 (2025-09-12)
-------------------

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
+ Pinned ``astrowidgets`` to below 0.4.0; the astrowidgets 0.4 API is not yet
  supported. [#543]

2.0.0 (2025-06-23)
-------------------

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
+ Now requires Python 3.10 or later. [#147]
+ Use pydantic for aperture settings. [#154]
+ Stomped bug in handling of ``NaN``s in ``single_image_photometry``. [#157]

Bug Fixes
^^^^^^^^^
+ Fixed dependence on non-release version of astrowidgets for overwrite capability on output images. [#108]
+ Fixed computation of FWHM when fitting to data that includes NaNs. [#164]

1.4.15 (2024-08-16)
-------------------

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
+ Increase minimum angular separation between comparison stars and variable stars.

1.4.14 (2024-03-27)
-------------------

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
+ Use ccd data unit instead of adu.

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
