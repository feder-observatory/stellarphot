# Objects that contains the user settings for the program.

from enum import Enum

from astropy import units as un

from pydantic import BaseModel, Field


class ApertureUnit(Enum):
    PIXEL = un.pix
    ARCSEC = un.arcsec


class ApertureSettings(BaseModel):
    radius : float
    inner_annulus : float
    outer_annulus : float
    unit : ApertureUnit = Field(default=ApertureUnit.PIXEL, enum=[au.value for au in ApertureUnit])

    class Config:
        use_enum_values = True


    # @validator('unit')
    # @classmethod
    # def check_unit(cls, v):
    #     if not isinstance(v, un.Unit):
    #         raise TypeError('unit must be an Astropy unit')
    #     return v
