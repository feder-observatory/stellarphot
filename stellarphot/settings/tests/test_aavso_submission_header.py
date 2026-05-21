import json

import pytest
from pydantic import ValidationError

from stellarphot.settings.aavso_submission import AAVSOSubmissionHeader


def _good_kwargs(**overrides):
    base = dict(
        type="EXTENDED",
        obscode="ABC",
        software="stellarphot test",
        delim=",",
        date_format="JD",
    )
    base.update(overrides)
    return base


class TestAAVSOSubmissionHeader:
    def test_create_and_round_trip(self):
        h = AAVSOSubmissionHeader(**_good_kwargs())
        as_json = h.model_dump_json()
        # Round-trip through JSON should give an equal model
        again = AAVSOSubmissionHeader(**json.loads(as_json))
        assert again == h

    def test_type_must_be_extended(self):
        with pytest.raises(ValidationError):
            AAVSOSubmissionHeader(**_good_kwargs(type="VISUAL"))
        # Lowercase is also wrong
        with pytest.raises(ValidationError):
            AAVSOSubmissionHeader(**_good_kwargs(type="extended"))

    @pytest.mark.parametrize("good", ["JD", "HJD", "EXCEL"])
    def test_date_format_accepts_spec_values(self, good):
        h = AAVSOSubmissionHeader(**_good_kwargs(date_format=good))
        assert h.date_format == good

    @pytest.mark.parametrize("bad", ["jd", "hjd", "MJD", ""])
    def test_date_format_rejects_others(self, bad):
        with pytest.raises(ValidationError):
            AAVSOSubmissionHeader(**_good_kwargs(date_format=bad))

    @pytest.mark.parametrize("delim", [",", ";", "!", "comma", "tab"])
    def test_delim_accepts_spec_values(self, delim):
        h = AAVSOSubmissionHeader(**_good_kwargs(delim=delim))
        assert h.delim == delim

    @pytest.mark.parametrize(
        "bad", ["|", "#", " ", "", ",,", "\x00", "\x1f", "\x7f", "\xff"]
    )
    def test_delim_rejects_forbidden_values(self, bad):
        with pytest.raises(ValidationError):
            AAVSOSubmissionHeader(**_good_kwargs(delim=bad))

    def test_software_max_length(self):
        # 255 is the limit per the schema; 255 is OK, 256 is not
        AAVSOSubmissionHeader(**_good_kwargs(software="x" * 255))
        with pytest.raises(ValidationError):
            AAVSOSubmissionHeader(**_good_kwargs(software="x" * 256))

    def test_header_lines_returns_six_lines_in_spec_order(self):
        h = AAVSOSubmissionHeader(
            **_good_kwargs(
                obscode="ABC",
                software="stellarphot 1.0",
                delim="comma",
                date_format="JD",
            )
        )
        lines = h.header_lines()
        assert lines == [
            "#TYPE=EXTENDED",
            "#OBSCODE=ABC",
            "#SOFTWARE=stellarphot 1.0",
            "#DELIM=comma",
            "#DATE=JD",
            "#OBSTYPE=CCD",
        ]
        # Should not contain trailing newlines
        for line in lines:
            assert not line.endswith("\n")

    def test_obscode_required(self):
        kw = _good_kwargs()
        del kw["obscode"]
        with pytest.raises(ValidationError):
            AAVSOSubmissionHeader(**kw)

    def test_obscode_rejects_empty(self):
        with pytest.raises(ValidationError):
            AAVSOSubmissionHeader(**_good_kwargs(obscode=""))

    def test_software_required(self):
        kw = _good_kwargs()
        del kw["software"]
        with pytest.raises(ValidationError):
            AAVSOSubmissionHeader(**kw)
