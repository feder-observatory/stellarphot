# Objects that contains the user settings for the program.

from pathlib import Path
from typing import Annotated, Literal

from astropy.coordinates import EarthLocation, Latitude, Longitude, SkyCoord
from astropy.io.misc.yaml import AstropyDumper, AstropyLoader
from astropy.time import Time
from astropy.units import Quantity, Unit
from astropy.utils import lazyproperty
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from .astropy_pydantic import (
    AstropyValidator,
    EquivalentTo,
    QuantityType,
    UnitType,
    WithPhysicalType,
    _UnitQuantTypePydanticAnnotation,
)

__all__ = [
    "Camera",
    "PhotometryApertures",
    "PhotometryFileSettings",
    "PhotometryOptions",
    "Exoplanet",
    "Observatory",
]

# Most models should use the default configuration, but it can be customized if needed.
MODEL_DEFAULT_CONFIGURATION = ConfigDict(
    # Make sure default values are valid
    validate_default=True,
    # Make sure changes to values made after initialization are valid
    validate_assignment=True,
    # Make sure there are no extra fields
    extra="forbid",
)


def add_degree_to_float(value, _handler):
    """
    Translate a value that can be a number to a string with "degree" appended.
    """
    try:
        as_number = float(value)
    except (ValueError, TypeError):
        # A value error will happen if the value is not a number.
        # A type error will happen at least in the case where the value is
        # an astropy Quantity.
        return value
    else:
        return f"{as_number} degree"


class BaseModelWithTableRep(BaseModel):
    """
    Class to add to a pydantic model YAML serialization to an Astropy table.
    """

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.generate_table_representers()

    def generate_table_representers(self):
        """
        Call this method during initialization of a class to add the YAML
        Table representation for the class.
        """
        class_string = f"!{self.__class__.__name__}"

        # Add YAML round-tripping for the model
        def _representer(dumper, model):
            # THIS SHOULD TAP INTO ASTROPY'S YAML DUMPER SOMEHOW
            # This is a little hacky at the moment. It seems like YAML
            # has trouble reading in a dictionary, so though model.model_dump()
            # works fine for writing, we can't construct from the dump.
            #
            # Instead of figuring out the right way to do that, this just dumps
            # the json representation as a string.
            return dumper.represent_mapping(
                class_string, {"model_json_string": model.model_dump_json()}
            )

        def _constructor(loader, node):
            # This loads the simple dictionary we dumped in _representer,
            # then initializes the model with the json string.
            mapping = loader.construct_mapping(node)
            return self.__class__.model_validate_json(mapping["model_json_string"])

        AstropyDumper.add_representer(self.__class__, _representer)
        AstropyLoader.add_constructor(class_string, _constructor)


class Camera(BaseModelWithTableRep):
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

    model_config = MODEL_DEFAULT_CONFIGURATION
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
    pixel_scale: Annotated[
        QuantityType,
        EquivalentTo(Unit("arcsec / pix")),
        Field(description="units of angle per pixel", examples=["0.6 arcsec / pix"]),
    ]
    max_data_value: Annotated[
        QuantityType,
        Field(
            description="maximum data value while performing photometry",
            examples=["50000 adu"],
            gt=0,
        ),
    ]

    # Run the model validator after the default validator. Unlike in pydantic 1,
    # mode="after" passes in an instance as an argument not a value.
    @model_validator(mode="after")
    def validate_gain(self):
        # Get read noise units
        rn_unit = Quantity(self.read_noise).unit

        # Check that gain and read noise have compatible units, that is that
        # gain is read noise per data unit.
        gain = self.gain
        if (
            len(gain.unit.bases) != 2
            or gain.unit.bases[0] != rn_unit
            or gain.unit.bases[1] != self.data_unit
        ):
            raise ValueError(
                f"Gain units {gain.unit} are not compatible with "
                f"read noise units {rn_unit}."
            )

        # Check that dark current and read noise have compatible units, that is
        # that dark current is read noise per second.
        dark_current = self.dark_current
        if (
            len(dark_current.unit.bases) != 2
            or dark_current.unit.bases[0] != rn_unit
            or dark_current.unit.bases[1] != Unit("s")
        ):
            raise ValueError(
                f"Dark current units {dark_current.unit} are not "
                f"compatible with read noise units {rn_unit}."
            )

        # Check that maximum data value is consistent with data units
        if self.max_data_value.unit != self.data_unit:
            raise ValueError(
                f"Maximum data value units {self.max_data_value.unit} "
                f"are not consistent with data units {self.data_unit}."
            )
        return self


class PhotometryApertures(BaseModelWithTableRep):
    """
    Settings for aperture photometry.

    Parameters
    ----------

    radius : int
        Radius of the aperture in pixels, must be greater than or equal to 1.

    gap : int
        Distance between the radius and the inner annulus in pixels, must be greater
        than or equal to 1.

    annulus_width : int
        Width of the annulus in pixels, must be greater than or equal to 1.

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

    To create an `PhotometryApertures` object, you can pass in the radius, gap,
    and annulus_width as keyword arguments:

    >>> aperture_settings = PhotometryApertures(
    ...     radius=4,
    ...     gap=10,
    ...     annulus_width=15,
    ...     fwhm=3.0
    ... )
    """

    model_config = MODEL_DEFAULT_CONFIGURATION

    radius: Annotated[
        PositiveInt,
        Field(default=1, json_schema_extra=dict(autoui="ipywidgets.BoundedIntText")),
    ]
    gap: Annotated[
        PositiveInt,
        Field(default=1, json_schema_extra=dict(autoui="ipywidgets.BoundedIntText")),
    ]
    annulus_width: Annotated[
        PositiveInt,
        Field(default=1, json_schema_extra=dict(autoui="ipywidgets.BoundedIntText")),
    ]
    # Disable the UI element by default because it is often calculate from an image
    fwhm: Annotated[PositiveFloat, Field(disabled=True, default=1.0)]

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


class PhotometryFileSettings(BaseModelWithTableRep):
    """
    An evolutionary step on the way to having a monolithic set of photometry settings.
    """

    model_config = MODEL_DEFAULT_CONFIGURATION

    image_folder: Path = Field(
        show_only_dirs=True,
        default="",
        description="Folder containing the calibrated images",
    )
    aperture_settings_file: Path = Field(filter_pattern="*.json", default="")
    aperture_locations_file: Path = Field(
        filter_pattern=["*.ecsv", "*.csv"], default=""
    )


class Observatory(BaseModelWithTableRep):
    """
    Class to represent an observatory.

    Parameters
    ----------
    name : str
        Name of the observatory.

    latitude : `astropy.coordinates.Latitude` or other valid latitude representation
        Latitude of the observatory. Use a positive number for north and negative
        for south.

    longitude : `astropy.coordinates.Longitude` or other valid longitude representation
        Longitude of the observatory. Use a positive number for east and negative
        for west.

    elevation : `astropy.units.Quantity`
        Elevation of the observatory.

    AAVSO_code : str, optional
        AAVSO observer code.

    TESS_telescope_code : str, optional
        TESS telescope code.
    """

    model_config = MODEL_DEFAULT_CONFIGURATION

    name: str
    latitude: Annotated[
        Latitude, _UnitQuantTypePydanticAnnotation, BeforeValidator(add_degree_to_float)
    ]
    longitude: Annotated[
        Longitude,
        _UnitQuantTypePydanticAnnotation,
        BeforeValidator(add_degree_to_float),
    ]
    elevation: Annotated[QuantityType, WithPhysicalType("length")]
    AAVSO_code: str | None = None
    TESS_telescope_code: str | None = None

    @lazyproperty
    def earth_location(self):
        """
        Return an `astropy.coordinates.EarthLocation` object for the observatory.
        """
        return EarthLocation(
            lat=self.latitude, lon=self.longitude, height=self.elevation
        )


class PhotometryOptions(BaseModelWithTableRep):
    """
    Options for performing photometry.

    Parameters
    ----------
    shift_tolerance : `pydantic.NonNegativeFloat`
        Since source positions need to be computed on each image using
        the sky position and WCS, the computed x/y positions are refined
        afterward by centroiding the sources.  This setting constrols
        the tolerance in pixels for the shift between the the computed
        positions and the refined positions, in pixels.  The expected
        shift shift should not be more than the FWHM, so a measured FWHM
        might be a good value to provide here.

    use_coordinates : `typing.Literal["sky", "pixel"]`
        If ``'pixel'``, use the x/y positions in the sourcelist for
        performing aperture photometry.  If ``'sky'``, use the ra/dec
        positions in the sourcelist and the WCS of the `ccd_image` to
        compute the x/y positions on the image.

    include_dig_noise : bool, optional (Default: True)
        If ``True``, include the digitization noise in the calculation of the
        noise for each observation.  If ``False``, only the Poisson noise from
        the source and the sky will be included.

    reject_too_close : bool, optional (Default: True)
        If ``True``, any sources that are closer than twice the aperture radius
        are rejected.  If ``False``, all sources in field are used.

    reject_background_outliers : bool, optional (Default: True)
        If ``True``, sigma clip the pixels in the annulus to reject outlying
        pixels (e.g. like stars in the annulus)

    fwhm_by_fit : bool, optional (default: True)
        If ``True``, the FWHM will be calculated by fitting a Gaussian to
        the star. If ``False``, the FWHM will be calculated by finding the
        second order moments of the light distribution. Default is ``True``.

    logfile : str, optional (Default: None)
        Name of the file to which log messages should be written.  It will
        be created in the `directory_with_images` directory.  If None,
        all messages are logged to stdout.

    console_log: bool, optional (Default: True)
        If ``True`` and `logfile` is set, log messages will also be written to
        stdout.  If ``False``, log messages will not be written to stdout
        if `logfile` is set.

    Examples
    --------

    The only option that must be set explicitly is the `shift_tolerance`:

    >>> from stellarphot.settings import PhotometryOptions
    >>> photometry_options = PhotometryOptions(shift_tolerance=5.2)
    >>> photometry_options
    PhotometryOptions(shift_tolerance=5.2, use_coordinates='sky',
    include_dig_noise=True, reject_too_close=True, reject_background_outliers=True,
    fwhm_by_fit=True, logfile=None, console_log=True)


    You can also set the other options explicitly when you create the options:

    >>> photometry_options = PhotometryOptions(
    ...     shift_tolerance=5.2,
    ...     use_coordinates="pixel",
    ...     include_dig_noise=True,
    ...     reject_too_close=False,
    ...     reject_background_outliers=True,
    ...     fwhm_by_fit=True,
    ...     logfile=None,
    ...     console_log=True
    ... )
    >>> photometry_options
    PhotometryOptions(shift_tolerance=5.2, use_coordinates='pixel',
    include_dig_noise=True, reject_too_close=False, reject_background_outliers=True,
    fwhm_by_fit=True, logfile=None, console_log=True)

    You can also change individual options after the object is created:

    >>> photometry_options.use_coordinates = "sky"
    >>> photometry_options.use_coordinates
    'sky'
    """

    model_config = MODEL_DEFAULT_CONFIGURATION

    shift_tolerance: NonNegativeFloat
    use_coordinates: Literal["sky", "pixel"] = "sky"
    include_dig_noise: bool = True
    reject_too_close: bool = True
    reject_background_outliers: bool = True
    fwhm_by_fit: bool = True
    logfile: str | None = None
    console_log: bool = True


class Exoplanet(BaseModelWithTableRep):
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

    model_config = MODEL_DEFAULT_CONFIGURATION

    epoch: Annotated[Time, AstropyValidator] | None = None
    period: Annotated[QuantityType, WithPhysicalType("time")] | None = None
    identifier: str
    coordinate: Annotated[SkyCoord, AstropyValidator]
    depth: float | None = None
    duration: Annotated[QuantityType, WithPhysicalType("time")] | None = None
