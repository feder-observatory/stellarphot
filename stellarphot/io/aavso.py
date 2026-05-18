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

import io
import math
from pathlib import Path

from astropy.table import Column, QTable, Table, join
from astropy.time import Time

from stellarphot.settings.aavso_models import AAVSOFilters
from stellarphot.settings.aavso_submission import AAVSOSubmissionHeader

__all__ = ["write_aavso_extended"]


ALLOWED_EXTENSIONS = frozenset({".txt", ".csv", ".tsv"})

# AAVSO data column order, in the sequence the spec requires. The AAVSO
# sample files prepend a row of these names with "#" before the data.
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


def _to_float(value):
    """Coerce a value (possibly an astropy ``Quantity``) to a plain float."""
    try:
        return float(value.value)
    except AttributeError:
        return float(value)


def _format_mag(value):
    return f"{_to_float(value):.4f}"


def _format_magerr(value):
    if value is None:
        return "na"
    f = _to_float(value)
    if math.isnan(f):
        return "na"
    return f"{f:.3f}"


def _format_airmass(value):
    return f"{_to_float(value):.4f}"


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

    target_mask = phot_data["star_id"] == target_star_id
    check_mask = phot_data["star_id"] == check_star_id

    if not target_mask.any():
        raise ValueError(f"No rows in phot_data have star_id={target_star_id!r}.")
    if not check_mask.any():
        raise ValueError(f"No rows in phot_data have star_id={check_star_id!r}.")

    # Reject invalid filters before doing any heavier work.
    for passband in phot_data["passband"][target_mask]:
        if not _is_valid_filter(passband):
            raise ValueError(
                f"Row passband {passband!r} is not a valid AAVSO filter. "
                "Apply a PassbandMap so the column uses AAVSO filter names "
                "before exporting."
            )

    # Pull just the columns we need from each side so the join result is small
    # and the renamed columns are unambiguous.
    target_cols = [
        "file",
        "passband",
        "date-obs",
        "exposure",
        "airmass",
        mag_column,
        mag_error_column,
    ]
    check_cols = ["file", "passband", mag_column]

    target_subset = QTable(phot_data[target_mask][target_cols], copy=True)
    check_subset = QTable(phot_data[check_mask][check_cols], copy=True)

    # Join on (file, passband) — this replaces the manual lookup dictionary
    # and naturally drops target rows that have no matching check observation.
    paired = join(
        target_subset,
        check_subset,
        keys=["file", "passband"],
        table_names=["target", "check"],
        join_type="left",
    )

    # Detect target rows without a matching check observation. After a left
    # join those rows have the check magnitude masked.
    check_mag_col = f"{mag_column}_check"
    if hasattr(paired[check_mag_col], "mask") and paired[check_mag_col].mask.any():
        missing = paired[paired[check_mag_col].mask][["file", "passband"]]
        first = missing[0]
        raise ValueError(
            "No check-star row found for "
            f"(file={first['file']!r}, passband={first['passband']!r}); "
            f"check_star_id={check_star_id!r} must have a matching "
            "observation for every target observation."
        )

    # Preserve the original target row order so the output is stable and easy
    # to compare against the input table.
    paired.sort(["file", "passband"])

    group_field = "na" if group is None else str(group)
    trans_field = "YES" if trans else "NO"
    notes_field = str(notes) if notes else "na"

    n = len(paired)
    target_mag_col = f"{mag_column}_target"

    # Build per-row string columns in AAVSO order.
    date_values = [
        f"{(Time(row['date-obs']) + row['exposure'] / 2).jd:.5f}" for row in paired
    ]
    mag_values = [_format_mag(v) for v in paired[target_mag_col]]
    err_values = [_format_magerr(v) for v in paired[mag_error_column]]
    kmag_values = [_format_mag(v) for v in paired[check_mag_col]]
    airmass_values = [_format_airmass(v) for v in paired["airmass"]]
    filter_values = [str(p) for p in paired["passband"]]

    columns = {
        "STARID": [str(target_name)] * n,
        "DATE": date_values,
        "MAGNITUDE": mag_values,
        "MAGERR": err_values,
        "FILTER": filter_values,
        "TRANS": [trans_field] * n,
        "MTYPE": ["STD"] * n,
        "CNAME": ["ENSEMBLE"] * n,
        "CMAG": ["na"] * n,
        "KNAME": [str(check_name)] * n,
        "KMAG": kmag_values,
        "AIRMASS": [_enforce_limit("AIRMASS", v) for v in airmass_values],
        "GROUP": [group_field] * n,
        "CHART": [str(chart)] * n,
        "NOTES": [notes_field] * n,
    }

    # Enforce length limits on every column that has one (AIRMASS already
    # truncated above). Validation fires before any I/O.
    out_table = Table()
    for name in DATA_COLUMNS:
        values = columns[name]
        if name in FIELD_LIMITS and name != "AIRMASS":
            values = [_enforce_limit(name, v) for v in values]
        out_table[name] = Column(values, dtype=str)

    # Write the data rows to a string buffer via astropy's ascii writer, then
    # assemble the final file with the parameter header and the
    # column-name row prefixed with "#".
    buf = io.StringIO()
    out_table.write(buf, format="ascii.no_header", delimiter=delimiter)
    data_text = buf.getvalue()

    column_header = "#" + delimiter.join(DATA_COLUMNS)

    with open(path, "w") as f:
        for line in header.header_lines():
            f.write(line + "\n")
        f.write(column_header + "\n")
        f.write(data_text)
        if not data_text.endswith("\n"):
            f.write("\n")

    return path
