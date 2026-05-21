"""Pydantic model for the AAVSO Extended File Format submission header.

The structure mirrors ``stellarphot/io/aavso_submission_schema.yml``, which
is a snapshot of the AAVSO specification. The yaml is loaded once at
import time and used to check that the constants below stay in sync with
the spec — the type annotations themselves use ``Literal`` so the
allowed values are visible to static analysis and to the pydantic-
generated JSON schema.
"""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import AfterValidator, Field

from .models import MODEL_DEFAULT_CONFIGURATION, BaseModelWithTableRep, NonEmptyStr

__all__ = ["AAVSOSubmissionHeader"]


_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "io" / "aavso_submission_schema.yml"
)
_SCHEMA = yaml.safe_load(_SCHEMA_PATH.read_text())
_COMMENTS = _SCHEMA["comments"]

# Values pulled from the schema and checked against the Literal annotations
# below so the two cannot silently drift apart.
_TYPE_OPTIONS = tuple(_COMMENTS["TYPE"]["options"])
_DATE_OPTIONS = tuple(_COMMENTS["DATE"]["options"])
SOFTWARE_LIMIT = int(_COMMENTS["SOFTWARE"]["limit"])

# These are drift guards between the schema YAML and the Literal annotations
# below. They must run even under ``python -O`` (where ``assert`` is stripped),
# so use explicit ``raise`` instead.
if _TYPE_OPTIONS != ("EXTENDED",):
    raise RuntimeError(
        f"TYPE options drift: schema={_TYPE_OPTIONS!r} model=('EXTENDED',)"
    )
if _DATE_OPTIONS != ("JD", "HJD", "EXCEL"):
    raise RuntimeError(
        f"DATE options drift: schema={_DATE_OPTIONS!r} model=('JD','HJD','EXCEL')"
    )

# DELIM rules from the schema: cannot use pipe, hash, or space; the literal
# words "comma" and "tab" are allowed as escapes for Excel users and tab
# delimiters respectively.
DELIM_FORBIDDEN_CHARS = frozenset({"|", "#", " "})
DELIM_KEYWORDS = frozenset({"comma", "tab"})

# The writer always emits OBSTYPE=CCD; it is not user-settable.
OBSTYPE = "CCD"


def _validate_delim(value: str) -> str:
    if value in DELIM_KEYWORDS:
        return value
    if len(value) != 1:
        raise ValueError(
            "delim must be a single character or one of 'comma'/'tab'; "
            f"got {value!r}"
        )
    if value in DELIM_FORBIDDEN_CHARS:
        raise ValueError(
            f"delim cannot be one of {sorted(DELIM_FORBIDDEN_CHARS)}; got {value!r}"
        )
    if not (32 <= ord(value) <= 126):
        raise ValueError(
            f"delim must be an ASCII character with code 32-126; got {value!r}"
        )
    return value


class AAVSOSubmissionHeader(BaseModelWithTableRep):
    """Header parameters for an AAVSO Extended File Format submission.

    Five fields map 1:1 to the ``#``-prefixed parameter lines required by the
    AAVSO loader (TYPE, OBSCODE, SOFTWARE, DELIM, DATE). The sixth header
    line, ``#OBSTYPE=CCD``, is always emitted by the writer and is not a
    field on this model.
    """

    model_config = MODEL_DEFAULT_CONFIGURATION

    type: Annotated[
        Literal["EXTENDED"],
        Field(description="Always EXTENDED for this format."),
    ] = "EXTENDED"
    obscode: Annotated[
        NonEmptyStr,
        Field(description="Official AAVSO observer code."),
    ]
    software: Annotated[
        NonEmptyStr,
        Field(
            description="Name and version of the software used.",
            max_length=SOFTWARE_LIMIT,
        ),
    ]
    delim: Annotated[
        str,
        AfterValidator(_validate_delim),
        Field(description="Field delimiter character or the word 'comma'/'tab'."),
    ]
    date_format: Annotated[
        Literal["JD", "HJD", "EXCEL"],
        Field(description="Date format: JD, HJD, or EXCEL."),
    ] = "JD"

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
