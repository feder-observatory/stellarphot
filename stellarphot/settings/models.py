# Objects that contains the user settings for the program.

from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io.misc.yaml import AstropyDumper, AstropyLoader
from astropy.time import Time
from astropy.units import IrreducibleUnit, Quantity, Unit
from pydantic import BaseModel, Field, confloat, conint, root_validator, validator

from .astropy_pydantic import PixelScaleType, QuantityType, UnitType
from .autowidgets import CustomBoundedIntTex

__all__ = ["Camera", "ApertureSettings", "PhotometryFileSettings", "Exoplanet"]


class Camera(BaseModel):
    """
    A class to represent a CCD-based camera.

    Parameters
    ----------

    data_unit : `astropy.units.Unit`
        The unit of the data.

    gain : `astropy.units.Quantity`
        The gain of the camera in units such the product of `gain`
        times the image data has units equal to that of the `read_noise`.

    name : str
        The name of the camera; can be anything that helps the user identify
        the camera.

    read_noise : `astropy.units.Quantity`
        The read noise of the camera with units.

    dark_current : `astropy.units.Quantity`
        The dark current of the camera in units such that, when multiplied by
        exposure time, the unit matches the units of the `read_noise`.

    pixel_scale : `astropy.units.Quantity`
        The pixel scale of the camera in units of arcseconds per pixel.

    max_data_value : `astropy.units.Quantity`
        The maximum pixel value to allow while performing photometry. Pixel values
        above this will be set to ``NaN``. The unit must be ``data_unit``.

    Attributes
    ----------

    data_unit : `astropy.units.Unit`
        The unit of the data.

    gain : `astropy.units.Quantity`
        The gain of the camera in units such the product of `gain`
        times the image data has units equal to that of the `read_noise`.

    name : str
        The name of the camera; can be anything that helps the user identify
        the camera.

    read_noise : `astropy.units.Quantity`
        The read noise of the camera with units.

    dark_current : `astropy.units.Quantity`
        The dark current of the camera in units such that, when multiplied by
        exposure time, the unit matches the units of the `read_noise`.

    pixel_scale : `astropy.units.Quantity`
        The pixel scale of the camera in units of arcseconds per pixel.

    max_data_value : `astropy.units.Quantity`
        The maximum pixel value to allow while performing photometry. Pixel values
        above this will be set to ``NaN``. The unit must be ``data_unit``.

    Notes
    -----
    The gain, read noise, and dark current are all assumed to be constant
    across the entire CCD.

    Examples
    --------
    >>> from astropy import units as u
    >>> from stellarphot.settings import Camera
    >>> camera = Camera(data_unit="adu",
    ...                 gain=1.0 * u.electron / u.adu,
    ...                 name="test camera",
    ...                 read_noise=1.0 * u.electron,
    ...                 dark_current=0.01 * u.electron / u.second,
    ...                 pixel_scale=0.563 * u.arcsec / u.pixel,
    ...                 max_data_value=50000 * u.adu)
    >>> camera.data_unit
    Unit("adu")
    >>> camera.gain
    <Quantity 1. electron / adu>
    >>> camera.name
    'test camera'
    >>> camera.read_noise
    <Quantity 1. electron>
    >>> camera.dark_current
    <Quantity 0.01 electron / s>
    >>> camera.pixel_scale
    <Quantity 0.563 arcsec / pix>
    >>> camera.max_data_value
    <Quantity 50000. adu>
    """

    data_unit: UnitType = Field(
        description="units of the data", examples=["adu", "counts", "DN", "electrons"]
    )
    gain: QuantityType = Field(
        description="unit should be consistent with data and read noise",
        examples=["1.0 electron / adu"],
    )
    name: str
    read_noise: QuantityType = Field(
        description="unit should be consistent with dark current",
        examples=["10.0 electron"],
    )
    dark_current: QuantityType = Field(
        description="unit consistent with read noise, per unit time",
        examples=["0.01 electron / second"],
    )
    pixel_scale: PixelScaleType = Field(
        description="units of angle per pixel", examples=["0.6 arcsec / pix"]
    )
    max_data_value: QuantityType = Field(
        description="maximum data value while performing photometry",
        examples=["50000 adu"],
    )

    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        json_encoders = {
            Quantity: lambda v: f"{v.value} {v.unit}",
            QuantityType: lambda v: f"{v.value} {v.unit}",
            Unit: lambda v: f"{v}",
            IrreducibleUnit: lambda v: f"{v}",
            PixelScaleType: lambda v: f"{v.value} {v.unit}",
        }

    # When the switch to pydantic v2 happens, this root_validator will need
    # to be replaced by a model_validator decorator.
    @root_validator(skip_on_failure=True)
    @classmethod
    def validate_gain(cls, values):
        # Get read noise units
        rn_unit = Quantity(values["read_noise"]).unit

        # Check that gain and read noise have compatible units, that is that
        # gain is read noise per data unit.
        gain = values["gain"]
        if (
            len(gain.unit.bases) != 2
            or gain.unit.bases[0] != rn_unit
            or gain.unit.bases[1] != values["data_unit"]
        ):
            raise ValueError(
                f"Gain units {gain.unit} are not compatible with "
                f"read noise units {rn_unit}."
            )

        # Check that dark current and read noise have compatible units, that is
        # that dark current is read noise per second.
        dark_current = values["dark_current"]
        if (
            len(dark_current.unit.bases) != 2
            or dark_current.unit.bases[0] != rn_unit
            or dark_current.unit.bases[1] != u.s
        ):
            raise ValueError(
                f"Dark current units {dark_current.unit} are not "
                f"compatible with read noise units {rn_unit}."
            )

        # Check that maximum data value is consistent with data units
        if values["max_data_value"].unit != values["data_unit"]:
            raise ValueError(
                f"Maximum data value units {values['max_data_value'].unit} "
                f"are not consistent with data units {values['data_unit']}."
            )
        return values

    @validator("max_data_value")
    @classmethod
    def validate_max_data_value(cls, v):
        if v.value <= 0:
            raise ValueError("max_data_value must be positive")
        return v


# Add YAML round-tripping for Camera
def camera_representer(dumper, cam):
    return dumper.represent_mapping("!Camera", cam.dict())


def camera_constructor(loader, node):
    return Camera(**loader.construct_mapping(node))


AstropyDumper.add_representer(Camera, camera_representer)
AstropyLoader.add_constructor("!Camera", camera_constructor)


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

    fwhm : float
        Full width at half maximum of the typical star in pixels.

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

    >>> aperture_settings = ApertureSettings(
    ...     radius=4,
    ...     gap=10,
    ...     annulus_width=15,
    ...     fwhm=3.0
    ... )
    """

    radius: conint(ge=1) = Field(autoui=CustomBoundedIntTex, default=1)
    gap: conint(ge=1) = Field(autoui=CustomBoundedIntTex, default=1)
    annulus_width: conint(ge=1) = Field(autoui=CustomBoundedIntTex, default=1)
    fwhm: confloat(gt=0)

    class Config:
        validate_assignment = True
        validate_all = True
        extra = "forbid"

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
        Duration of the exoplanet transit.

    Examples
    --------

    To create an `Exoplanet` object, you can pass in the epoch,
     period, identifier, coordinate, depth, and duration as keyword arguments:

    >>> from astropy.time import Time
    >>> from astropy.coordinates import SkyCoord
    >>> from astropy import units as u
    >>> planet  = Exoplanet(epoch=Time(2455909.29280, format="jd"),
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

    @validator("period")
    @classmethod
    def validate_period(cls, value):
        """
        Checks that the period has physical units of time and raises an error
        if that is not true.
        """
        if u.get_physical_type(value) != "time":
            raise ValueError(
                f"Period does not have time units," f"currently has {value.unit} units."
            )
        return value

    @validator("duration")
    @classmethod
    def validate_duration(cls, value):
        """
        Checks that the duration has physical units of time and raises an error
        if that is not true.
        """
        if u.get_physical_type(value) != "time":
            raise ValueError(
                f"Duration does not have time units,"
                f"currently has {value.unit} units."
            )
        return value
