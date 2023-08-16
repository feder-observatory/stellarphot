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

    gap : int
        Distance between the radius and the inner annulus in pixels.

    annulus_width : int
        Width of the annulus in pixels.

    Atributes
    ---------

    inner_annulus : int
        Radius of the inner annulus in pixels.

    outer_annulus : int
        Radius of the outer annulus in pixels.

    Examples
    --------

    To create an `ApertureSettings` object, you can pass in the radius, gap,
    and annulus_width as keyword arguments:

    >>> aperture_settings = ApertureSettings(radius=4, gap=10, annulus_width=15)
    """
    radius : conint(ge=1) = Field(autoui=CustomBoundedIntTex, default=1)
    gap : conint(ge=1) = Field(autoui=CustomBoundedIntTex, default=1)
    annulus_width : conint(ge=1) = Field(autoui=CustomBoundedIntTex, default=1)

    class Config:
        validate_assignment = True
        validate_all = True

    @property
    def inner_annulus(self):
        """
        Radius of the inner annulus in pixels.
        """
        return self.radius + self.gap

    @property
    def outer_annulus(self):
        """
        Radius of the outer annulus in pixels.
        """
        return self.inner_annulus + self.annulus_width
