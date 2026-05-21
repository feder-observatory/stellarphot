Calibrating Photometry
######################

Exporting to AAVSO
==================

Once you have calibrated magnitudes for your target and a check star, you can
write a file in the `AAVSO Extended File Format
<https://www.aavso.org/aavso-extended-file-format>`_ that the AAVSO WebObs
loader accepts. Stellarphot's writer produces an *ensemble* submission
(``CNAME=ENSEMBLE``, ``CMAG=na``) with one target star and one check star,
paired observation-by-observation by ``(date-obs, passband)``.

The writer expects the ``passband`` column to already contain valid AAVSO
filter names. If you started from instrumental filter names, use a
:class:`~stellarphot.settings.PassbandMap` when constructing
:class:`~stellarphot.PhotometryData` and the column will be remapped in place.

Example
-------

.. code-block:: python

    from stellarphot.settings import AAVSOSubmissionHeader

    header = AAVSOSubmissionHeader(
        type="EXTENDED",
        obscode="ABC",
        software="stellarphot 1.4",
        delim="comma",
        date_format="JD",
    )

    phot_data.write_aavso_extended(
        "submission.csv",
        header=header,
        target_star_id=1,
        target_name="V0533 Her",
        check_star_id=6,
        check_name="000-BLS-123",
        chart="X12345",
        mag_column="mag_inst_cal",
        mag_error_column="mag_inst_cal_error",
        trans=False,
    )

The ``mag_column`` and ``mag_error_column`` arguments name the calibrated
magnitude column and its uncertainty column to read for the target and the
check star.

.. note::

   This release supports ``DATE=JD`` only. The AAVSO spec also allows ``HJD``
   and ``EXCEL`` dates; using either currently raises ``NotImplementedError``.
   The writer always emits ``#OBSTYPE=CCD`` and ``MTYPE=STD`` (consistent with
   ensemble photometry using standardized magnitudes).
