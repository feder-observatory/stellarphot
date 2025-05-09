from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field


class MagnitudeSystemNames(StrEnum):
    """Enum for different magnitude systems."""

    AB = "AB"
    Vega = "Vega"
    JC = "JC"
    SDSSDR7 = "SDSSDR7"
    GAIA = "GAIA"
    TESS = "TESS"
    PANSTARRS1 = "PANSTARRS1"
    UGRIZ = "ugriz"


class MagnitudeSystemPassbands(BaseModel):
    system_name: MagnitudeSystemNames
    passbands_names: list[str]
    passband_range: tuple[float, float]
    # Down the road add something like symphot.SpectralElement


class MagnitudeSystemTransform(BaseModel):
    """
    Class for magnitude system transformation from one single
    system to another.
    """

    name: str
    reference: Annotated[
        str,
        Field(
            description=(
                "Reference for the transformation, e.g., a " "paper or documentation."
            )
        ),
    ]
    from_system: MagnitudeSystemNames = Field(
        description="The magnitude system to transform from."
    )
    to_system: MagnitudeSystemNames = Field(
        description="The magnitude system to transform to."
    )
    transform_coeofficients: Annotated[
        list[float],
        Field(
            description=(
                "Coefficients for the transformation. The "
                "number of coefficients depends on the transformation."
            )
        ),
    ]
