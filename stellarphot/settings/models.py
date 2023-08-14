# Objects that contains the user settings for the program.

from enum import Enum

from astropy import units as un

from pydantic import BaseModel, Field, conint, root_validator

from .autowidgets import CustomBoundedIntTex


__all__ = ['ApertureSettings']

class ApertureUnit(Enum):
    PIXEL = un.pix
    ARCSEC = un.arcsec


class ApertureSettings(BaseModel):
    """
    Settings for aperture photometry.

    Parameters
    ----------

    radius : int
        Radius of the aperture in pixels.

    inner_annulus : int
        Inner radius of the annulus in pixels.

    outer_annulus : int
        Outer radius of the annulus in pixels.
    """
    radius : conint(ge=1) = Field(autoui=CustomBoundedIntTex, default=1)
    gap : conint(ge=1) = Field(autoui=CustomBoundedIntTex, default=1)
    annulus_width : conint(ge=1) = Field(autoui=CustomBoundedIntTex, default=1)

    class Config:
        validate_assignment = True
        validate_all = True

    @property
    def inner_annulus(self):
        return self.radius + self.gap

    @property
    def outer_annulus(self):
        return self.inner_annulus + self.annulus_width
