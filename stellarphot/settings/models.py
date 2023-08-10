# Objects that contains the user settings for the program.

from enum import Enum

from astropy import units as un

from pydantic import BaseModel, Field, conint, root_validator

from .autowidgets import CustomBoundedIntTex

class ApertureUnit(Enum):
    PIXEL = un.pix
    ARCSEC = un.arcsec


class ApertureSettings(BaseModel):
    radius : conint(ge=1) = Field(autoui=CustomBoundedIntTex)
    inner_annulus : conint(ge=1) = Field(autoui=CustomBoundedIntTex)
    outer_annulus : conint(ge=1) = Field(autoui=CustomBoundedIntTex)

    class Config:
        validate_assignment = True
        validate_all = True

    @root_validator(skip_on_failure=True)
    def check_annuli(cls, values):
        if values['inner_annulus'] >= values['outer_annulus']:
            raise ValueError('inner_annulus must be smaller than outer_annulus')
        if values['radius'] >= values['inner_annulus']:
            raise ValueError('radius must be smaller than inner_annulus')
        return values
