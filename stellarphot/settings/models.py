# Objects that contains the user settings for the program.

from pathlib import Path

from pydantic import BaseModel, Field, conint

from .autowidgets import CustomBoundedIntTex

__all__ = [
    'ApertureSettings',
    'PhotometryFileSettings'
]


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

    Attributes
    ----------

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


class PhotometryFileSettings(BaseModel):
    """
    An evolutionary step on the way to having a monolithic set of photometry settings.
    """
    image_folder : Path = Field(show_only_dirs=True, default='',
                                description="Folder containing the calibrated images")
    aperture_settings_file : Path = Field(filter_pattern='*.json', default='')
    aperture_locations_file : Path = Field(filter_pattern=['*.ecsv', '*.csv'], default='')
