"""Pydantic model for the AAVSO Extended File Format submission header.

The structure mirrors ``stellarphot/io/aavso_submission_schema.yml``, which
is a snapshot of the AAVSO specification. The allowed values for ``TYPE``
and ``DATE`` are encoded directly as ``Literal`` annotations so they are
visible to static analysis and to the pydantic-generated JSON schema, and
``SOFTWARE_LIMIT`` is hardcoded below. The YAML is not read at runtime;
consistency with the spec snapshot is verified by
``test_schema_matches_hardcoded_constants`` in the tests for this module.
"""

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, BeforeValidator, Field

try:
    from stellarphot.version import version as __version__
except ImportError:
    __version__ = "unknown"

from .models import MODEL_DEFAULT_CONFIGURATION, NonEmptyStr

__all__ = ["AAVSOSubmissionHeader"]


# Mirrors ``comments.SOFTWARE.limit`` in aavso_submission_schema.yml;
# drift-checked in the tests for this module.
SOFTWARE_LIMIT = 255

# DELIM rules from the schema: cannot use pipe, hash, or space; the literal
# words "comma" and "tab" are allowed as escapes for Excel users and tab
# delimiters respectively.
DELIM_FORBIDDEN_CHARS = frozenset({"|", "#", " "})
DELIM_KEYWORDS = frozenset({"comma", "tab"})

# The writer always emits OBSTYPE=CCD; it is not user-settable.
OBSTYPE = "CCD"

# Map header DELIM keywords to the literal character used between data fields.
_DELIM_CHAR_MAP = {"comma": ",", "tab": "\t"}


def _upper_if_str(value):
    return value.upper() if isinstance(value, str) else value


def _validate_delim(value: str) -> str:
    # The "comma"/"tab" keywords are case-insensitive per the schema; normalize
    # to lowercase so the header line always emits the canonical form.
    if value.lower() in DELIM_KEYWORDS:
        return value.lower()
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


class AAVSOSubmissionHeader(BaseModel):
    """Header parameters for an AAVSO Extended File Format submission.

    Five fields map 1:1 to the ``#``-prefixed parameter lines required by the
    AAVSO loader (TYPE, OBSCODE, SOFTWARE, DELIM, DATE). The sixth header
    line, ``#OBSTYPE=CCD``, is always emitted by the writer and is not a
    field on this model.
    """

    model_config = MODEL_DEFAULT_CONFIGURATION

    type: Annotated[
        Literal["EXTENDED"],
        BeforeValidator(_upper_if_str),
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
    ] = f"stellarphot {__version__}"
    delim: Annotated[
        str,
        AfterValidator(_validate_delim),
        Field(description="Field delimiter character or the word 'comma'/'tab'."),
    ] = ","
    date_format: Annotated[
        Literal["JD", "HJD", "EXCEL"],
        BeforeValidator(_upper_if_str),
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

    @property
    def data_delimiter(self) -> str:
        """Return the actual character used to separate data fields.

        The header writes ``comma`` and ``tab`` literally, but the data rows
        use ``,`` and ``\\t`` respectively.
        """
        return _DELIM_CHAR_MAP.get(self.delim, self.delim)
