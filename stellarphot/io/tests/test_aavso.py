import numpy as np
import pytest
from astropy.io import ascii as ap_ascii
from astropy.time import Time
from astropy.utils.data import get_pkg_data_filename

from stellarphot import PhotometryData
from stellarphot.io.aavso import (
    DATA_COLUMNS,
    write_aavso_extended,
)
from stellarphot.settings.aavso_submission import AAVSOSubmissionHeader

# ---- fixtures ---------------------------------------------------------------


@pytest.fixture
def header():
    return AAVSOSubmissionHeader(
        type="EXTENDED",
        obscode="ABC",
        software="stellarphot test",
        delim=",",
        date_format="JD",
    )


@pytest.fixture
def phot_table():
    """The test photometry fixture, with no further modification.

    The fixture has 8 stars, two files, passband ``SR`` (a valid AAVSO filter).
    """
    data_file = get_pkg_data_filename(
        "data/test_photometry_data.ecsv", package="stellarphot.tests"
    )
    return PhotometryData.read(data_file)


@pytest.fixture
def writer_kwargs(header):
    """Default kwargs for write_aavso_extended built against ``phot_table``.

    Star 1 is the target, star 6 is the check star.
    """
    return dict(
        header=header,
        target_star_id=1,
        target_name="V0533 Her",
        check_star_id=6,
        check_name="check1",
        chart="X12345",
        mag_column="mag_inst",
        mag_error_column="mag_error",
    )


# ---- Step 2: skeleton + header --------------------------------------------


class TestHeaderAndFileFormat:
    def test_rejects_unknown_extensions(self, tmp_path, phot_table, writer_kwargs):
        bad = tmp_path / "bad.foo"
        with pytest.raises(ValueError, match="must have one of"):
            write_aavso_extended(phot_table, bad, **writer_kwargs)

    @pytest.mark.parametrize("ext", [".txt", ".csv", ".tsv"])
    def test_accepts_allowed_extensions(self, tmp_path, phot_table, writer_kwargs, ext):
        out = tmp_path / f"sub{ext}"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        assert out.exists()

    def test_first_six_lines_are_header(self, tmp_path, phot_table, writer_kwargs):
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        lines = out.read_text().splitlines()
        assert len(lines) >= 6
        for line in lines[:6]:
            assert line.startswith("#")
        # Hardcoded OBSTYPE
        assert lines[5] == "#OBSTYPE=CCD"

    def test_column_header_row_follows_parameter_header(
        self, tmp_path, phot_table, writer_kwargs
    ):
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        lines = out.read_text().splitlines()
        assert lines[6] == "#" + ",".join(DATA_COLUMNS)

    def test_data_uses_configured_delimiter(self, tmp_path, phot_table, writer_kwargs):
        # Default delim=","
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        lines = out.read_text().splitlines()
        # First data row (line 7, after 6 parameter lines + column header) has
        # 14 commas (15 fields).
        assert lines[7].count(",") == len(DATA_COLUMNS) - 1

    def test_delim_comma_writes_comma_in_header_but_separates_with_commas(
        self, tmp_path, phot_table, writer_kwargs
    ):
        writer_kwargs["header"] = AAVSOSubmissionHeader(
            type="EXTENDED",
            obscode="ABC",
            software="stellarphot test",
            delim="comma",
            date_format="JD",
        )
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        lines = out.read_text().splitlines()
        assert lines[3] == "#DELIM=comma"
        # Data row still uses literal commas as separators
        assert lines[7].count(",") == len(DATA_COLUMNS) - 1

    def test_delim_tab_writes_tab_in_data(self, tmp_path, phot_table, writer_kwargs):
        writer_kwargs["header"] = AAVSOSubmissionHeader(
            type="EXTENDED",
            obscode="ABC",
            software="stellarphot test",
            delim="tab",
            date_format="JD",
        )
        out = tmp_path / "sub.tsv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        lines = out.read_text().splitlines()
        assert lines[3] == "#DELIM=tab"
        assert lines[7].count("\t") == len(DATA_COLUMNS) - 1

    def test_non_jd_date_format_raises(self, tmp_path, phot_table, writer_kwargs):
        writer_kwargs["header"] = AAVSOSubmissionHeader(
            type="EXTENDED",
            obscode="ABC",
            software="stellarphot test",
            delim=",",
            date_format="HJD",
        )
        out = tmp_path / "sub.csv"
        with pytest.raises(NotImplementedError, match="JD"):
            write_aavso_extended(phot_table, out, **writer_kwargs)


# ---- Step 3: target-star row construction ----------------------------------


def _read_data_rows(path):
    """Read the file and return the data rows as an astropy Table."""
    return ap_ascii.read(
        str(path),
        format="no_header",
        delimiter=",",
        comment="#",
        names=list(DATA_COLUMNS),
    )


class TestTargetRows:
    def test_one_row_per_target_observation(self, tmp_path, phot_table, writer_kwargs):
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        rows = _read_data_rows(out)
        expected = (phot_table["star_id"] == writer_kwargs["target_star_id"]).sum()
        assert len(rows) == expected

    def test_starid_is_target_name(self, tmp_path, phot_table, writer_kwargs):
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        rows = _read_data_rows(out)
        assert set(rows["STARID"]) == {writer_kwargs["target_name"]}

    def test_cname_and_cmag_are_ensemble_constants(
        self, tmp_path, phot_table, writer_kwargs
    ):
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        rows = _read_data_rows(out)
        assert set(rows["CNAME"]) == {"ENSEMBLE"}
        assert set(rows["CMAG"]) == {"na"}

    def test_chart_and_mtype(self, tmp_path, phot_table, writer_kwargs):
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        rows = _read_data_rows(out)
        assert set(rows["CHART"]) == {writer_kwargs["chart"]}
        assert set(rows["MTYPE"]) == {"STD"}

    def test_date_is_jd_at_mid_exposure(self, tmp_path, phot_table, writer_kwargs):
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        rows = _read_data_rows(out)

        target_rows = phot_table[
            phot_table["star_id"] == writer_kwargs["target_star_id"]
        ]
        for written_jd, src in zip(rows["DATE"], target_rows, strict=True):
            expected = (Time(src["date-obs"]) + src["exposure"] / 2).jd
            assert abs(float(written_jd) - expected) < 1e-5

    def test_magnitude_and_magerr_columns_used(
        self, tmp_path, phot_table, writer_kwargs
    ):
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        rows = _read_data_rows(out)

        target_rows = phot_table[
            phot_table["star_id"] == writer_kwargs["target_star_id"]
        ]
        for mag_str, err_str, src in zip(
            rows["MAGNITUDE"], rows["MAGERR"], target_rows, strict=True
        ):
            assert abs(float(mag_str) - float(src["mag_inst"])) < 1e-4
            # mag_error in the fixture carries 1/adu units; .value strips them.
            assert abs(float(err_str) - float(src["mag_error"].value)) < 1e-3

    def test_filter_is_row_passband(self, tmp_path, phot_table, writer_kwargs):
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        rows = _read_data_rows(out)
        # All rows in fixture are SR
        assert set(rows["FILTER"]) == {"SR"}

    def test_invalid_filter_raises(self, tmp_path, phot_table, writer_kwargs):
        bad = phot_table.copy()
        # "XX" is not in AAVSOFilters; same width as the existing column so the
        # assignment does not need a dtype change.
        bad["passband"][:] = "XX"
        out = tmp_path / "sub.csv"
        with pytest.raises(ValueError, match="not a valid AAVSO filter"):
            write_aavso_extended(bad, out, **writer_kwargs)

    @pytest.mark.parametrize("trans,expected", [(False, "NO"), (True, "YES")])
    def test_trans_reflects_kwarg(
        self, tmp_path, phot_table, writer_kwargs, trans, expected
    ):
        writer_kwargs["trans"] = trans
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        rows = _read_data_rows(out)
        assert set(rows["TRANS"]) == {expected}

    def test_group_default_is_na(self, tmp_path, phot_table, writer_kwargs):
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        rows = _read_data_rows(out)
        assert set(rows["GROUP"]) == {"na"}

    def test_group_integer(self, tmp_path, phot_table, writer_kwargs):
        writer_kwargs["group"] = 7
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        rows = _read_data_rows(out)
        assert set(rows["GROUP"]) == {7}

    def test_airmass_truncates(self, tmp_path, phot_table, writer_kwargs):
        # Force a very long airmass value
        long_data = phot_table.copy()
        long_data["airmass"] = 1.123456789
        out = tmp_path / "sub.csv"
        write_aavso_extended(long_data, out, **writer_kwargs)
        rows = _read_data_rows(out)
        for value in rows["AIRMASS"]:
            assert len(str(value)) <= 7

    def test_field_too_long_raises(self, tmp_path, phot_table, writer_kwargs):
        writer_kwargs["target_name"] = "x" * 31  # exceeds STARID limit of 30
        out = tmp_path / "sub.csv"
        with pytest.raises(ValueError, match="STARID"):
            write_aavso_extended(phot_table, out, **writer_kwargs)


# ---- Step 4: check-star pairing -------------------------------------------


class TestCheckStarPairing:
    def test_kmag_comes_from_check_star_row(self, tmp_path, phot_table, writer_kwargs):
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        rows = _read_data_rows(out)

        # Build lookup of (date-obs, passband) -> check mag, matching the
        # writer's join key. Using (file, passband) would silently pass even
        # if the writer regressed for fixtures with reused filenames.
        check_rows = phot_table[phot_table["star_id"] == writer_kwargs["check_star_id"]]
        check_lookup = {
            (str(r["date-obs"]), r["passband"]): float(r["mag_inst"])
            for r in check_rows
        }

        target_rows = phot_table[
            phot_table["star_id"] == writer_kwargs["target_star_id"]
        ]
        for written_kmag, src in zip(rows["KMAG"], target_rows, strict=True):
            expected = check_lookup[(str(src["date-obs"]), src["passband"])]
            assert abs(float(written_kmag) - expected) < 1e-4

    def test_kname_is_check_name_kwarg(self, tmp_path, phot_table, writer_kwargs):
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        rows = _read_data_rows(out)
        assert set(rows["KNAME"]) == {writer_kwargs["check_name"]}

    def test_missing_check_row_raises(self, tmp_path, phot_table, writer_kwargs):
        # Drop one of the check-star observations to break pairing.
        check_id = writer_kwargs["check_star_id"]
        mask_check = phot_table["star_id"] == check_id
        bad = phot_table[
            ~mask_check | (np.arange(len(phot_table)) != np.where(mask_check)[0][0])
        ]
        # Sanity: target still has 2 observations, but check star is now
        # missing one (date-obs, passband) pair.
        out = tmp_path / "sub.csv"
        with pytest.raises(ValueError, match="No check-star row"):
            write_aavso_extended(bad, out, **writer_kwargs)

    def test_missing_target_raises(self, tmp_path, phot_table, writer_kwargs):
        writer_kwargs["target_star_id"] = 9999
        out = tmp_path / "sub.csv"
        with pytest.raises(ValueError, match="No rows in phot_data"):
            write_aavso_extended(phot_table, out, **writer_kwargs)


# ---- Step 5: PhotometryData convenience method -----------------------------


class TestPhotometryDataMethod:
    def test_method_matches_function(self, tmp_path, phot_table, writer_kwargs):
        func_out = tmp_path / "func.csv"
        meth_out = tmp_path / "meth.csv"
        write_aavso_extended(phot_table, func_out, **writer_kwargs)
        phot_table.write_aavso_extended(meth_out, **writer_kwargs)
        assert func_out.read_bytes() == meth_out.read_bytes()


# ---- Input validation ------------------------------------------------------


class TestInputValidation:
    def test_target_and_check_must_differ(self, tmp_path, phot_table, writer_kwargs):
        writer_kwargs["check_star_id"] = writer_kwargs["target_star_id"]
        out = tmp_path / "sub.csv"
        with pytest.raises(ValueError, match="must be different"):
            write_aavso_extended(phot_table, out, **writer_kwargs)

    @pytest.mark.parametrize("field", ["target_name", "check_name", "chart"])
    def test_blank_required_identifier_rejected(
        self, tmp_path, phot_table, writer_kwargs, field
    ):
        writer_kwargs[field] = "   "
        out = tmp_path / "sub.csv"
        with pytest.raises(ValueError, match=field):
            write_aavso_extended(phot_table, out, **writer_kwargs)

    def test_identifiers_are_stripped(self, tmp_path, phot_table, writer_kwargs):
        writer_kwargs["target_name"] = "  V0533 Her  "
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        rows = _read_data_rows(out)
        assert set(rows["STARID"]) == {"V0533 Her"}

    def test_blank_notes_becomes_na(self, tmp_path, phot_table, writer_kwargs):
        writer_kwargs["notes"] = "   "
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        rows = _read_data_rows(out)
        assert set(rows["NOTES"]) == {"na"}

    @pytest.mark.parametrize("field", ["target_name", "check_name", "chart", "notes"])
    def test_delimiter_in_field_rejected(
        self, tmp_path, phot_table, writer_kwargs, field
    ):
        # Default delim is comma; injecting one would create an extra column.
        writer_kwargs[field] = "bad,value"
        out = tmp_path / "sub.csv"
        with pytest.raises(ValueError, match="delimiter"):
            write_aavso_extended(phot_table, out, **writer_kwargs)

    def test_newline_in_field_rejected(self, tmp_path, phot_table, writer_kwargs):
        writer_kwargs["notes"] = "line1\nline2"
        out = tmp_path / "sub.csv"
        with pytest.raises(ValueError, match="newline"):
            write_aavso_extended(phot_table, out, **writer_kwargs)

    @pytest.mark.parametrize("delim", [".", "n", "a", "A"])
    def test_delimiter_collides_with_rendered_field(
        self, tmp_path, phot_table, writer_kwargs, delim
    ):
        # "." appears in every formatted numeric field (DATE, MAGNITUDE,
        # MAGERR, KMAG, AIRMASS); "n"/"a" appear in the literal missing value
        # "na"; "A" appears in AAVSO column names (STARID, DATE, ...). All
        # pass the header model's character check but produce mis-parseable
        # output and must be rejected before any I/O.
        writer_kwargs["header"] = AAVSOSubmissionHeader(
            type="EXTENDED",
            obscode="ABC",
            software="stellarphot test",
            delim=delim,
            date_format="JD",
        )
        out = tmp_path / "sub.txt"
        with pytest.raises(ValueError, match="delimiter"):
            write_aavso_extended(phot_table, out, **writer_kwargs)


# ---- Non-finite numeric handling -------------------------------------------


class TestNonFiniteValues:
    def test_nan_target_magnitude_raises(self, tmp_path, phot_table, writer_kwargs):
        bad = phot_table.copy()
        target_mask = bad["star_id"] == writer_kwargs["target_star_id"]
        idx = np.where(target_mask)[0][0]
        bad["mag_inst"][idx] = float("nan")
        out = tmp_path / "sub.csv"
        with pytest.raises(ValueError, match="MAGNITUDE"):
            write_aavso_extended(bad, out, **writer_kwargs)

    def test_nan_check_magnitude_raises(self, tmp_path, phot_table, writer_kwargs):
        bad = phot_table.copy()
        check_mask = bad["star_id"] == writer_kwargs["check_star_id"]
        idx = np.where(check_mask)[0][0]
        bad["mag_inst"][idx] = float("nan")
        out = tmp_path / "sub.csv"
        with pytest.raises(ValueError, match="KMAG"):
            write_aavso_extended(bad, out, **writer_kwargs)

    def test_nan_magerr_becomes_na(self, tmp_path, phot_table, writer_kwargs):
        bad = phot_table.copy()
        target_mask = bad["star_id"] == writer_kwargs["target_star_id"]
        idx = np.where(target_mask)[0][0]
        bad["mag_error"][idx] = float("nan")
        out = tmp_path / "sub.csv"
        write_aavso_extended(bad, out, **writer_kwargs)
        rows = _read_data_rows(out)
        # At least one MAGERR cell should read "na"; the rest are numeric.
        assert "na" in {str(v) for v in rows["MAGERR"]}

    def test_inf_magerr_becomes_na(self, tmp_path, phot_table, writer_kwargs):
        bad = phot_table.copy()
        target_mask = bad["star_id"] == writer_kwargs["target_star_id"]
        idx = np.where(target_mask)[0][0]
        bad["mag_error"][idx] = float("inf")
        out = tmp_path / "sub.csv"
        write_aavso_extended(bad, out, **writer_kwargs)
        rows = _read_data_rows(out)
        assert "na" in {str(v) for v in rows["MAGERR"]}

    def test_nan_airmass_becomes_na(self, tmp_path, phot_table, writer_kwargs):
        bad = phot_table.copy()
        target_mask = bad["star_id"] == writer_kwargs["target_star_id"]
        idx = np.where(target_mask)[0][0]
        bad["airmass"][idx] = float("nan")
        out = tmp_path / "sub.csv"
        write_aavso_extended(bad, out, **writer_kwargs)
        rows = _read_data_rows(out)
        assert "na" in {str(v) for v in rows["AIRMASS"]}


# ---- Kwarg type validation -------------------------------------------------


class TestKwargTypeValidation:
    @pytest.mark.parametrize("value", ["False", "NO", "YES", "yes", "", 0, 1])
    def test_trans_rejects_non_bool(self, tmp_path, phot_table, writer_kwargs, value):
        writer_kwargs["trans"] = value
        out = tmp_path / "sub.csv"
        with pytest.raises(TypeError, match="trans"):
            write_aavso_extended(phot_table, out, **writer_kwargs)

    @pytest.mark.parametrize("value", [True, False])
    def test_trans_accepts_bool(self, tmp_path, phot_table, writer_kwargs, value):
        writer_kwargs["trans"] = value
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        rows = _read_data_rows(out)
        assert set(rows["TRANS"]) == {"YES" if value else "NO"}

    @pytest.mark.parametrize(
        "value,expected",
        [
            (7, 7),
            ("5", 5),
            (5.0, 5),
        ],
    )
    def test_group_accepts_int_like(
        self, tmp_path, phot_table, writer_kwargs, value, expected
    ):
        writer_kwargs["group"] = value
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        rows = _read_data_rows(out)
        assert set(rows["GROUP"]) == {expected}

    def test_group_accepts_numpy_int(self, tmp_path, phot_table, writer_kwargs):
        writer_kwargs["group"] = np.int64(42)
        out = tmp_path / "sub.csv"
        write_aavso_extended(phot_table, out, **writer_kwargs)
        rows = _read_data_rows(out)
        assert set(rows["GROUP"]) == {42}

    @pytest.mark.parametrize("value", ["abc", "5.5", [], {}, object()])
    def test_group_rejects_non_int_like(
        self, tmp_path, phot_table, writer_kwargs, value
    ):
        writer_kwargs["group"] = value
        out = tmp_path / "sub.csv"
        with pytest.raises((TypeError, ValueError), match="group"):
            write_aavso_extended(phot_table, out, **writer_kwargs)

    @pytest.mark.parametrize("value", [5.5, 0.1, float("inf"), float("nan")])
    def test_group_rejects_non_integer_float(
        self, tmp_path, phot_table, writer_kwargs, value
    ):
        writer_kwargs["group"] = value
        out = tmp_path / "sub.csv"
        with pytest.raises(ValueError, match="group"):
            write_aavso_extended(phot_table, out, **writer_kwargs)

    @pytest.mark.parametrize("value", [True, False])
    def test_group_rejects_bool(self, tmp_path, phot_table, writer_kwargs, value):
        writer_kwargs["group"] = value
        out = tmp_path / "sub.csv"
        with pytest.raises(TypeError, match="group"):
            write_aavso_extended(phot_table, out, **writer_kwargs)
