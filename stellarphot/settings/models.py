# Objects that contains the user settings for the program.

from pathlib import Path

from pydantic import BaseModel, Field, conint

from .autowidgets import CustomBoundedIntTex

from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
import astropy.units as u

from ..core import QuantityType


__all__ = ["ApertureSettings", "PhotometryFileSettings", "Exoplanet"]


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

    radius: conint(ge=1) = Field(autoui=CustomBoundedIntTex, default=1)
    gap: conint(ge=1) = Field(autoui=CustomBoundedIntTex, default=1)
    annulus_width: conint(ge=1) = Field(autoui=CustomBoundedIntTex, default=1)

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

    image_folder: Path = Field(
        show_only_dirs=True,
        default="",
        description="Folder containing the calibrated images",
    )
    aperture_settings_file: Path = Field(filter_pattern="*.json", default="")
    aperture_locations_file: Path = Field(
        filter_pattern=["*.ecsv", "*.csv"], default=""
    )


class TimeType(Time):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return Time(v)


class SkyCoordType(SkyCoord):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return SkyCoord(v)


class Exoplanet(BaseModel):
    """
    Create an object representing an Exoplanet.

    Parameters
    ----------

    epoch : `astropy.time.Time`, optional
        Epoch of the exoplanet.

    period : `astropy.units.Quantity`, optional
        Period of the exoplanet.

    Identifier : str
        Identifier of the exoplanet.

    coordinate : `astropy.coordinates.SkyCoord`
        Coordinates of the exoplanet.

    depth : float
        Depth of the exoplanet.

    duration : `astropy.units.Quantity`, optional
        Duration of the exoplanet transit as a Quantity with units of time,
        not required.

    Examples
    --------

    To create an `Exoplanet` object, you can pass in the epoch,
     period, identifier, coordinate, depth, and duration as keyword arguments:

    >>> planet  = Exoplanet(epoch=Time( 2455909.29280, format="jd"),
    ...                     period=1.21749 * u.day,
    ...                     identifier="KELT-1b",
    ...                     coordinate=SkyCoord(ra="00:01:26.9169",
    ...                                         dec="+39:23:01.7821",
    ...                                         frame="icrs",
    ...                                         unit=("hour", "degree")),
    ...                     depth=0.006,
    ...                     duration=120 * u.min)
    """

    epoch: TimeType | None = None
    period: QuantityType | None = None
    identifier: str
    coordinate: SkyCoordType
    depth: float | None = None
    duration: QuantityType | None = None

    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        json_encoders = {
            Quantity: lambda v: f"{v.value} {v.unit}",
            QuantityType: lambda v: f"{v.value} {v.unit}",
            Time: lambda v: f"{v.value}",
        }

    @classmethod
    def validate_period(cls, values):
        """
        Checks that the period has physical units of time and raises an error if that is not true. 
        """
        if u.get_physical_type(values["period"]) != "time":
            raise ValueError(
                f"Period does not have time units,"
                f"currently has {values['period'].unit} units."
            )

    @classmethod
    def validate_duration(cls, values):
        """
        Checks that the duration has physical units of time and raises an error if that is not true. 
        """
        if u.get_physical_type(values["duration"]) != "time":
            raise ValueError(
                f"Duration does not have time units,"
                f"currently has {values['duration'].unit} units."
            )
