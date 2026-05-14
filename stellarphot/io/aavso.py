"""Writer for the AAVSO Extended File Format used by WebObs.

Implements ensemble photometry submissions (CNAME=ENSEMBLE, CMAG=na) with one
target star and one check star. The data layout follows the spec mirrored in
``stellarphot/io/aavso_submission_schema.yml``.

v1 limitations:
- ``DATE=JD`` only. ``HJD`` and ``EXCEL`` are valid in the header model but
  raise ``NotImplementedError`` from the writer.
- ``MTYPE`` is hardcoded to ``STD`` (calibrated/standardized magnitudes), which
  is the correct value when CNAME=ENSEMBLE.
- ``OBSTYPE`` is hardcoded to ``CCD``.
"""

import math
from pathlib import Path

from astropy.time import Time

from stellarphot.settings.aavso_models import AAVSOFilters
from stellarphot.settings.aavso_submission import AAVSOSubmissionHeader

__all__ = ["write_aavso_extended"]


ALLOWED_EXTENSIONS = frozenset({".txt", ".csv", ".tsv"})

# AAVSO data column order, in the sequence the spec requires.
DATA_COLUMNS = (
    "STARID",
    "DATE",
    "MAGNITUDE",
    "MAGERR",
    "FILTER",
    "TRANS",
    "MTYPE",
    "CNAME",
    "CMAG",
    "KNAME",
    "KMAG",
    "AIRMASS",
    "GROUP",
    "CHART",
    "NOTES",
)

# Per-field maximum character counts pulled from the spec. Fields not listed
# (FILTER, TRANS, MTYPE, NOTES) have no length limit. AIRMASS is special: the
# spec says it should be truncated rather than rejected.
FIELD_LIMITS = {
    "STARID": 30,
    "DATE": 16,
    "MAGNITUDE": 8,
    "MAGERR": 6,
    "CNAME": 20,
    "CMAG": 8,
    "KNAME": 20,
    "KMAG": 8,
    "AIRMASS": 7,
    "GROUP": 5,
    "CHART": 20,
}


def _is_valid_filter(value):
    try:
        AAVSOFilters(value)
    except ValueError:
        return False
    return True


def _enforce_limit(name, value):
    """Validate that the stringified field does not exceed its limit.

    AIRMASS truncates; every other limited field raises.
    """
    limit = FIELD_LIMITS.get(name)
    if limit is None or len(value) <= limit:
        return value
    if name == "AIRMASS":
        return value[:limit]
    raise ValueError(
        f"AAVSO field {name}={value!r} exceeds the {limit}-character limit "
        f"(got {len(value)} characters)."
    )


def _format_jd(date_obs, exposure):
    """JD at mid-exposure in UT. ``exposure`` carries seconds units."""
    midpoint = Time(date_obs) + exposure / 2
    return f"{midpoint.jd:.5f}"


def _to_float(value):
    """Coerce a table cell (possibly an astropy ``Quantity``) to a plain float.

    Columns in ``PhotometryData`` may carry units (e.g. ``mag_error`` has
    ``1/adu`` in the test fixture, ``exposure`` has seconds). For string
    formatting we only need the numeric magnitude, so strip the unit.
    """
    try:
        return float(value.value)
    except AttributeError:
        return float(value)


def _format_float(value, decimals):
    return f"{_to_float(value):.{decimals}f}"


def write_aavso_extended(
    phot_data,
    path,
    *,
    header,
    target_star_id,
    target_name,
    check_star_id,
    check_name,
    chart,
    mag_column,
    mag_error_column,
    trans=False,
    group=None,
    notes="na",
):
    """Write an AAVSO Extended File Format submission for ensemble photometry.

    Parameters
    ----------
    phot_data : `stellarphot.PhotometryData`
        Table of photometry results. Must contain at least the target star
        and the check star, paired by ``(file, passband)``.

    path : str or `pathlib.Path`
        Destination file. Must have a ``.txt``, ``.csv`` or ``.tsv`` suffix.

    header : `stellarphot.settings.AAVSOSubmissionHeader`
        Header parameters. Only ``date_format="JD"`` is supported in v1.

    target_star_id : str or int
        The ``star_id`` value identifying the target rows in ``phot_data``.

    target_name : str
        The string written into the ``STARID`` column for every target row.

    check_star_id : str or int
        The ``star_id`` value identifying the check-star rows.

    check_name : str
        The string written into the ``KNAME`` column.

    chart : str
        The AAVSO chart sequence ID written into the ``CHART`` column.

    mag_column : str
        Name of the column in ``phot_data`` containing the calibrated magnitude
        for the target. The same column is read for the check-star rows.

    mag_error_column : str
        Name of the column in ``phot_data`` containing the magnitude error.

    trans : bool, optional
        ``True`` to emit ``TRANS=YES``, ``False`` (default) for ``TRANS=NO``.

    group : int or None, optional
        Optional grouping identifier. ``None`` (default) emits ``GROUP=na``.

    notes : str, optional
        Text written into the ``NOTES`` column. Defaults to ``"na"``.
    """
    if not isinstance(header, AAVSOSubmissionHeader):
        raise TypeError(
            "header must be an AAVSOSubmissionHeader instance; "
            f"got {type(header).__name__}."
        )

    if header.date_format != "JD":
        raise NotImplementedError(
            f"AAVSO writer only supports DATE=JD in this release; "
            f"got date_format={header.date_format!r}."
        )

    path = Path(path)
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"AAVSO submission file must have one of {sorted(ALLOWED_EXTENSIONS)} "
            f"extensions; got {path.suffix!r}."
        )

    delimiter = header.data_delimiter()

    # Split the table into target and check-star rows.
    target_mask = phot_data["star_id"] == target_star_id
    check_mask = phot_data["star_id"] == check_star_id

    if not target_mask.any():
        raise ValueError(f"No rows in phot_data have star_id={target_star_id!r}.")
    if not check_mask.any():
        raise ValueError(f"No rows in phot_data have star_id={check_star_id!r}.")

    target_rows = phot_data[target_mask]
    check_rows = phot_data[check_mask]

    # Index check star rows by (file, passband) for fast lookup.
    check_index = {}
    for row in check_rows:
        key = (row["file"], row["passband"])
        check_index[key] = row

    group_field = "na" if group is None else str(group)
    trans_field = "YES" if trans else "NO"

    data_lines = []
    for row in target_rows:
        passband = row["passband"]
        if not _is_valid_filter(passband):
            raise ValueError(
                f"Row passband {passband!r} is not a valid AAVSO filter. "
                "Apply a PassbandMap so the column uses AAVSO filter names "
                "before exporting."
            )

        key = (row["file"], passband)
        if key not in check_index:
            raise ValueError(
                "No check-star row found for "
                f"(file={row['file']!r}, passband={passband!r}); "
                f"check_star_id={check_star_id!r} must have a matching "
                "observation for every target observation."
            )
        check_row = check_index[key]

        starid = _enforce_limit("STARID", str(target_name))
        date = _enforce_limit("DATE", _format_jd(row["date-obs"], row["exposure"]))
        magnitude = _enforce_limit("MAGNITUDE", _format_float(row[mag_column], 4))

        err_value = row[mag_error_column]
        if err_value is None:
            magerr = "na"
        else:
            err_float = _to_float(err_value)
            if math.isnan(err_float):
                magerr = "na"
            else:
                magerr = _enforce_limit("MAGERR", _format_float(err_float, 3))

        filter_field = _enforce_limit("FILTER", str(passband))
        cname = _enforce_limit("CNAME", "ENSEMBLE")
        cmag = _enforce_limit("CMAG", "na")
        kname = _enforce_limit("KNAME", str(check_name))
        kmag = _enforce_limit("KMAG", _format_float(check_row[mag_column], 4))
        airmass = _enforce_limit("AIRMASS", f"{_to_float(row['airmass']):.4f}")
        group_value = _enforce_limit("GROUP", group_field)
        chart_field = _enforce_limit("CHART", str(chart))
        notes_field = str(notes) if notes is not None else "na"
        if not notes_field:
            notes_field = "na"

        fields = [
            starid,
            date,
            magnitude,
            magerr,
            filter_field,
            trans_field,
            "STD",
            cname,
            cmag,
            kname,
            kmag,
            airmass,
            group_value,
            chart_field,
            notes_field,
        ]
        data_lines.append(delimiter.join(fields))

    with open(path, "w") as f:
        for line in header.header_lines():
            f.write(line + "\n")
        for line in data_lines:
            f.write(line + "\n")

    return path
