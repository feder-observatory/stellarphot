# Objects that contains the user settings for the program.

from enum import Enum

from astropy import units as un

from pydantic import BaseModel, Field

from .autowidgets import CustomBoundedIntTex

class ApertureUnit(Enum):
    PIXEL = un.pix
    ARCSEC = un.arcsec


class ApertureSettings(BaseModel):
    radius : conint(ge=1) = Field(autoui=CustomBoundedIntTex)
    inner_annulus : conint(ge=1) = Field(autoui=CustomBoundedIntTex)
    outer_annulus : conint(ge=1) = Field(autoui=CustomBoundedIntTex)

    class Config:
        use_enum_values = True


    # @validator('unit')
    # @classmethod
    # def check_unit(cls, v):
    #     if not isinstance(v, un.Unit):
    #         raise TypeError('unit must be an Astropy unit')
    #     return v
