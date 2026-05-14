"""Pydantic model for the AAVSO Extended File Format submission header.

The structure is driven by ``stellarphot/io/aavso_submission_schema.yml``,
which mirrors the AAVSO specification. The yaml is loaded once at import
time and used to derive the allowed values, length limits and forbidden
characters.
"""

from pathlib import Path

import yaml
from pydantic import Field, field_validator

from .models import MODEL_DEFAULT_CONFIGURATION, BaseModelWithTableRep, NonEmptyStr

__all__ = ["AAVSOSubmissionHeader"]


_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "io" / "aavso_submission_schema.yml"
)
_SCHEMA = yaml.safe_load(_SCHEMA_PATH.read_text())
_COMMENTS = _SCHEMA["comments"]

# Allowed values pulled from the schema so they stay in sync with the spec.
TYPE_OPTIONS = tuple(_COMMENTS["TYPE"]["options"])
DATE_OPTIONS = tuple(_COMMENTS["DATE"]["options"])
SOFTWARE_LIMIT = int(_COMMENTS["SOFTWARE"]["limit"])

# DELIM rules from the schema: cannot use pipe, hash, or space; the literal
# words "comma" and "tab" are allowed as escapes for Excel users and tab
# delimiters respectively.
DELIM_FORBIDDEN_CHARS = frozenset({"|", "#", " "})
DELIM_KEYWORDS = frozenset({"comma", "tab"})

# The writer always emits OBSTYPE=CCD; it is not user-settable.
OBSTYPE = "CCD"


class AAVSOSubmissionHeader(BaseModelWithTableRep):
    """Header parameters for an AAVSO Extended File Format submission.

    Five fields map 1:1 to the ``#``-prefixed parameter lines required by the
    AAVSO loader (TYPE, OBSCODE, SOFTWARE, DELIM, DATE). The sixth header
    line, ``#OBSTYPE=CCD``, is always emitted by the writer and is not a
    field on this model.
    """

    model_config = MODEL_DEFAULT_CONFIGURATION

    type: str = Field(
        default="EXTENDED", description="Always EXTENDED for this format."
    )
    obscode: NonEmptyStr = Field(description="Official AAVSO observer code.")
    software: NonEmptyStr = Field(description="Name and version of the software used.")
    delim: str = Field(
        description="Field delimiter character or the word 'comma'/'tab'."
    )
    date_format: str = Field(
        default="JD",
        description="Date format: JD, HJD, or EXCEL.",
    )

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v):
        if v not in TYPE_OPTIONS:
            raise ValueError(f"type must be one of {TYPE_OPTIONS}; got {v!r}")
        return v

    @field_validator("date_format")
    @classmethod
    def _validate_date_format(cls, v):
        if v not in DATE_OPTIONS:
            raise ValueError(f"date_format must be one of {DATE_OPTIONS}; got {v!r}")
        return v

    @field_validator("software")
    @classmethod
    def _validate_software(cls, v):
        if len(v) > SOFTWARE_LIMIT:
            raise ValueError(
                f"software must be at most {SOFTWARE_LIMIT} characters; got {len(v)}"
            )
        return v

    @field_validator("delim")
    @classmethod
    def _validate_delim(cls, v):
        if v in DELIM_KEYWORDS:
            return v
        if len(v) != 1:
            raise ValueError(
                "delim must be a single character or one of 'comma'/'tab'; "
                f"got {v!r}"
            )
        if v in DELIM_FORBIDDEN_CHARS:
            raise ValueError(
                f"delim cannot be one of {sorted(DELIM_FORBIDDEN_CHARS)}; got {v!r}"
            )
        if not (32 <= ord(v) <= 126):
            raise ValueError(
                f"delim must be an ASCII character with code 32-126; got {v!r}"
            )
        return v

    def header_lines(self) -> list[str]:
        """Return the six AAVSO ``#`` header lines in spec order.

        OBSTYPE is hardcoded to ``CCD``; the other five values come from the
        model fields. No trailing newlines.
        """
        return [
            f"#TYPE={self.type}",
            f"#OBSCODE={self.obscode}",
            f"#SOFTWARE={self.software}",
            f"#DELIM={self.delim}",
            f"#DATE={self.date_format}",
            f"#OBSTYPE={OBSTYPE}",
        ]

    def data_delimiter(self) -> str:
        """Return the actual character used to separate data fields.

        The header writes ``comma`` and ``tab`` literally, but the data rows
        use ``,`` and ``\\t`` respectively.
        """
        if self.delim == "comma":
            return ","
        if self.delim == "tab":
            return "\t"
        return self.delim
