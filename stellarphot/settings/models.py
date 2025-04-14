# Objects that contains the user settings for the program.

import re
from copy import deepcopy
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypeVar

from astropy.coordinates import EarthLocation, Latitude, Longitude, SkyCoord
from astropy.time import Time
from astropy.units import Quantity, Unit, UnitConversionError
from astropy.utils import lazyproperty
from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    create_model,
    field_validator,
    model_validator,
)

from .aavso_models import AAVSOFilters
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
    "FwhmMethods",
    "LoggingSettings",
    "PartialPhotometrySettings",
    "PassbandMap",
    "PhotometryApertures",
    "PhotometryFileSettings",
    "PhotometryRunSettings",
    "PhotometrySettings",
    "PhotometryOptionalSettings",
    "Exoplanet",
    "Observatory",
    "SourceLocationSettings",
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


# Make a type for a non-empty string for use in name fields
# Names may not have any leading or trailing spaces, and cannot simply
# be spaces. Though this could be implemented as a regular expression
# pattern, that leads to a validation message likely to confuse users,
# along the lines of (for pattern r"^\S$|^\S.*\S$"):
#
# String should match pattern "^\\S$|^\\S.*\\S$"
#
# Instead, the custom validation function below checks for errors and
# raises a ValueError with a more user-friendly message.
def _non_empty_string_validator(value):
    if not value.strip():
        raise ValueError("name must not be empty or contain only whitespace.")

    if not re.search(r"^\S$|^\S.*\S$", value):
        # Name must have exactly one non-whitespace character or
        # at least two non-whitespace characters with any amount of
        # other characters between them.
        raise ValueError("name must not have leading or trailing whitespace.")

    return value


NonEmptyStr = Annotated[str, AfterValidator(_non_empty_string_validator)]


def _extract_short_description(docstring: str) -> str:
    """
    Extract the first line of the docstring as a short description.
    """
    # Everything up the the first blank line is the short description
    short_desc = docstring.split("\n\n")[0]

    # This may have some extra whitespace at the beginning and end, remove that
    short_desc = short_desc.strip()

    # split the string and rejoin with spaces in case there are multiple spaces
    # between words, e.g. caused by a multi-line docstring
    return " ".join(short_desc.split())


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

    # NOTE WELL that this will set the configuration for all subclasses of this
    model_config = MODEL_DEFAULT_CONFIGURATION

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)


class Camera(BaseModelWithTableRep):
    """
    A class to represent a CCD-based camera.

    Parameters
    ----------

    name : str
        The name of the camera; can be anything that helps the user identify
        the camera.

    data_unit : `astropy.units.Unit`
        The unit of the data.

    gain : `astropy.units.Quantity`
        The gain of the camera in units such the product of `gain`
        times the image data has units equal to that of the `read_noise`.

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

    # This ensures that just the first line of the docstring is used as the
    # model description. The json schema is really hard to read if the
    # description is too long.

    # WHen a subclass has a ConfigDict it is merged with the parent class's
    # ConfigDict.
    model_config = ConfigDict(
        json_schema_extra=dict(
            description=_extract_short_description(__doc__),
        )
    )

    name: Annotated[
        NonEmptyStr,
        Field(
            description="Name of the camera",
            examples=["SBIG FakeCam", "ZWO NadaCam", "CG16m"],
        ),
    ]
    data_unit: UnitType = Field(
        description="units of the data", examples=["adu", "DN", "count"]
    )
    gain: QuantityType = Field(
        description="unit should be consistent with data and read noise",
        examples=["1.5 electron / adu", "1.0 electron / DN", "1.0 photon / count"],
    )
    read_noise: QuantityType = Field(
        description="unit should be consistent with dark current",
        examples=["10.0 electron", "10.0 electron", "10.0 photon"],
    )
    dark_current: QuantityType = Field(
        description="unit consistent with read noise, per unit time",
        examples=[
            "0.01 electron / second",
            "0.01 electron / second",
            "0.01 photon / second",
        ],
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
            examples=["50000 adu", "50000 DN", "50000 count"],
            gt=0,
        ),
    ]

    # Run the model validator after the default validator. Unlike in pydantic 1,
    # mode="after" passes in an instance as an argument not a value.
    @model_validator(mode="after")
    def validate_gain(self):
        # Get read noise units
        rn_unit = Quantity(self.read_noise).unit
        # Get dark current units
        dark_unit = Quantity(self.dark_current).unit

        # Check that gain and read noise have compatible units, that is that
        # gain is read noise per data unit.
        gain = self.gain

        try:
            gain.to(self.read_noise.unit / self.data_unit)
        except UnitConversionError as e:
            raise ValueError(
                f"Gain units {gain.unit} are not compatible with "
                f"read noise units {rn_unit}."
            ) from e

        # Check that dark current and read noise have compatible units, that is
        # that dark current is read noise per second.
        dark_current = self.dark_current
        try:
            dark_current.to(self.read_noise.unit / Unit("s"))
        except UnitConversionError as e:
            raise ValueError(
                f"Dark current units {dark_unit} are not "
                f"compatible with read noise units {rn_unit}."
            ) from e

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

    # This ensures that just the first line of the docstring is used as the
    # model description. The json schema is really hard to read if the
    # description is too long.

    # WHen a subclass has a ConfigDict it is merged with the parent class's
    # ConfigDict.
    model_config = ConfigDict(
        json_schema_extra=dict(
            description=_extract_short_description(__doc__),
        )
    )

    radius: Annotated[
        PositiveInt,
        Field(
            default=1,
            description="radius of circular aperture, in pixels",
            json_schema_extra=dict(autoui="ipywidgets.BoundedIntText"),
        ),
    ]
    gap: Annotated[
        PositiveInt,
        Field(
            default=1,
            description="Size of gap between aperture and annulus, in pixels",
            json_schema_extra=dict(autoui="ipywidgets.BoundedIntText"),
        ),
    ]
    annulus_width: Annotated[
        PositiveInt,
        Field(
            default=1,
            description=("distance between inner and outer radii of annulus in pixels"),
            json_schema_extra=dict(autoui="ipywidgets.BoundedIntText"),
        ),
    ]
    # Disable the UI element by default because it is often calculate from an image
    fwhm: Annotated[PositiveFloat, Field(disabled=True, default=1.0, title="FWHM")]

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

    Examples
    --------

    The first example shows how to set up an observatory whose latitude and longitude
    are 30° North and 100° West, respectively, and is at an elevation of 1000 meters.

    >>> from astropy.coordinates import Latitude, Longitude
    >>> from astropy import units as u
    >>> from stellarphot.settings import Observatory
    >>> observatory = Observatory(
    ...     name="test observatory",
    ...     latitude=Latitude(30.0 * u.deg),
    ...     longitude=Longitude(-100.0 * u.deg),
    ...     elevation=1000 * u.m,
    ... )
    >>> observatory
    Observatory(name='test observatory', latitude=<Latitude 30. deg>,
    longitude=<Longitude 260. deg>, elevation=<Quantity 1000. m>, AAVSO_code=None,
    TESS_telescope_code=None)

    Note that units can be omitted from the latitude and longitude if and only if the
    units are degrees and the values are in decimal degrees.

    >>> # You can also just provide numbers for the latitude and longitude
    >>> observatory = Observatory(
    ...     name="test observatory",
    ...     latitude=30.0,
    ...     longitude=-100.0,
    ...     elevation=1000 * u.m,
    ... )
    >>> observatory
    Observatory(name='test observatory', latitude=<Latitude 30. deg>,
    longitude=<Longitude 260. deg>, elevation=<Quantity 1000. m>, AAVSO_code=None,
    TESS_telescope_code=None)

    An observatory located at 46° 00' 00.00" South and 96° 00' 00.00" East, and at an
    elevation of 2300 meters would be created like this:

    >>> observatory = Observatory(
    ...     name="test observatory",
    ...     latitude="-46d00m00.00s",
    ...     longitude="96d00m00.00s",
    ...     elevation=2.3 * u.km,
    ... )
    >>> observatory
    Observatory(name='test observatory', latitude=<Latitude -46. deg>,...


    It would be fine to use decmial degrees for the latitude and longitude in the
    above example, but the example is given in part to show how to use sexagesimal
    notation.

    """

    # This ensures that just the first line of the docstring is used as the
    # model description. The json schema is really hard to read if the
    # description is too long.

    # WHen a subclass has a ConfigDict it is merged with the parent class's
    # ConfigDict.
    model_config = ConfigDict(
        json_schema_extra=dict(
            description=_extract_short_description(__doc__),
        )
    )

    name: Annotated[
        NonEmptyStr,
        Field(description="Name of the observatory", examples=["My Observatory"]),
    ]
    latitude: Annotated[
        Latitude,
        _UnitQuantTypePydanticAnnotation,
        BeforeValidator(add_degree_to_float),
        Field(
            description="Latitude of the observatory",
            examples=[
                "46.8738",
                "46.8738 degree",
                "46:52:25.68 degree",
                "46d52m25.68s",
            ],
        ),
    ]
    longitude: Annotated[
        Longitude,
        _UnitQuantTypePydanticAnnotation,
        BeforeValidator(add_degree_to_float),
        Field(
            description="Longitude of the observatory",
            examples=[
                "-96.7678",
                "-96d46m04.08s",
                "263.2322 degree",
                "263d13m55.92s",
            ],
        ),
    ]
    elevation: Annotated[
        QuantityType,
        WithPhysicalType("length"),
        Field(
            description="Elevation of the observatory",
            examples=["1000 m", "1 km", "3.241e-14 pc", "1e12 nm"],
        ),
    ]
    AAVSO_code: Annotated[str | None, Field(description="AAVSO code for observer")] = (
        None
    )
    TESS_telescope_code: Annotated[
        str | None, Field(description="Code for observatory used by TFOP SG01")
    ] = None

    @lazyproperty
    def earth_location(self):
        """
        Return an `astropy.coordinates.EarthLocation` object for the observatory.
        """
        return EarthLocation(
            lat=self.latitude, lon=self.longitude, height=self.elevation
        )


class SourceLocationSettings(BaseModelWithTableRep):
    """
    Settings for the location of the source list and the image files.

    Parameters
    ----------
    source_list_file : str
        Name of a file with a table of extracted sources with positions in terms of
        pixel coordinates OR RA/Dec coordinates. If both positions provided,
        the one that will be used is determined by `use_coordinates`. For RA/Dec
        coordinates to be used, `ccd_image` must have a valid WCS.

    use_coordinates : `typing.Literal["sky", "pixel"]`, optional
        If ``'pixel'``, use the x/y positions in the sourcelist for
        performing aperture photometry.  If ``'sky'``, use the ra/dec
        positions in the sourcelist and the WCS of the `ccd_image` to
        compute the x/y positions on the image.

    shift_tolerance : `pydantic.NonNegativeFloat`, optional
        Since source positions need to be computed on each image using
        the sky position and WCS, the computed x/y positions are refined
        afterward by centroiding the sources.  This setting controls
        the tolerance in pixels for the shift between the the computed
        positions and the refined positions, in pixels.  The expected
        shift shift should not be more than the FWHM, so a measured FWHM
        might be a good value to provide here.

    Examples
    --------
    >>> from stellarphot.settings import SourceLocationSettings
    >>> source_location_settings = SourceLocationSettings(
    ...     source_list_file="source_list.ecsv",
    ...     use_coordinates="sky",
    ...     shift_tolerance=5.0
    ... )
    >>> source_location_settings
    SourceLocationSettings(source_list_file='source_list.ecsv', use_coordinates='sky',
    shift_tolerance=5.0)
    """

    # This ensures that just the first line of the docstring is used as the
    # model description. The json schema is really hard to read if the
    # description is too long.

    # WHen a subclass has a ConfigDict it is merged with the parent class's
    # ConfigDict.
    model_config = ConfigDict(
        json_schema_extra=dict(
            description=_extract_short_description(__doc__),
        )
    )

    source_list_file: Annotated[
        str,
        Field(
            json_schema_extra=dict(
                autoui="ipyautoui.custom.FileChooser",
                filter_pattern=["*.ecsv", "*.csv"],
            )
        ),
    ] = "source_locations.ecsv"
    use_coordinates: Annotated[
        Literal["sky", "pixel"],
        Field(description="coordinates to use for locating sources in image"),
    ] = "sky"

    shift_tolerance: Annotated[
        NonNegativeFloat,
        Field(
            description=(
                "Maximum shift between source position in list and "
                "in image, in pixels"
            )
        ),
    ] = 5.0


class FwhmMethods(StrEnum):
    """
    Available methods for finding FWHM of stars.
    """

    FIT = "fit"
    MOMENTS = "moments"
    PROFILE = "profile"


class PhotometryOptionalSettings(BaseModelWithTableRep):
    """
    Options for performing photometry.

    Parameters
    ----------
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

    fwhm_method : `FwhmMethods` (default: ``FwhmMethods.FIT``)
        Method for finding the FWHM of the star. 'fit'fits a 1D Gaussian to the
        star, 'profiile' fits a 1D Gaussian to the radial profile, and
        'moments' uses second order moments of the image, which is terrible.

    method : `typing.Literal["exact", "center", "subpixel"]`, optional
        How to handle partial pixels in the aperture.  If ``'exact'``, the fraction of
        the flux included is the fraction of the pixel within the aperture. If
        ``'center'``, whether a pixel's flux is included is determined by whether the
        center of the pixel is within the aperture. If ``'subpixel'``, the flux included
        is determineid by breaking  region into subpixels. The default is ``exact``.
        For more information, see the
        `photutils documentation <https://photutils.readthedocs.io/en/stable/aperture.html#aperture-and-pixel-overlap>`_.

    Examples
    --------

    In many cases the default options are fine:

    >>> from stellarphot.settings import PhotometryOptionalSettings
    >>> photometry_options = PhotometryOptionalSettings()
    >>> photometry_options
    PhotometryOptionalSettings(include_dig_noise=True, reject_too_close=True,...

    You can also set options explicitly when you create the options:

    >>> photometry_options = PhotometryOptionalSettings(
    ...     include_dig_noise=True,
    ...     reject_too_close=False,
    ...     reject_background_outliers=True,
    ...     fwhm_by_fit=True,
    ...     method="center"
    ... )
    >>> photometry_options
    PhotometryOptionalSettings(include_dig_noise=True, reject_too_close=False,...
    reject_background_outliers=True, fwhm_method=<FwhmMethods.FIT: 'fit'>,...

    You can also change individual options after the object is created:

    >>> photometry_options.reject_background_outliers = False
    >>> photometry_options.reject_background_outliers
    False
    """

    # This ensures that just the first line of the docstring is used as the
    # model description. The json schema is really hard to read if the
    # description is too long.

    # When a subclass has a ConfigDict it is merged with the parent class's
    # ConfigDict.
    model_config = ConfigDict(
        json_schema_extra=dict(
            description=_extract_short_description(__doc__),
        )
    )

    # Made this a static method to define inside this class, since this is the only
    # class that needs to handle this.
    @staticmethod
    def _validate_fwhm_method(value: Any) -> Any:
        if isinstance(value, bool):
            # This is likely an old fwhm_by_fit option, so we convert to one
            # of the new options.
            # Note that originally False meant MOMENTS but it turns out that
            # method is completely incorrect
            return FwhmMethods.FIT if value else FwhmMethods.PROFILE
        else:
            # Pass the value along to pydantic to validate
            return value

    include_dig_noise: Annotated[
        bool,
        Field(
            title="Include Digitization Noise",
            description=(
                "Should the digitization noise be included in the calculation of "
                "the noise for each observation?"
            ),
        ),
    ] = True

    reject_too_close: Annotated[
        bool,
        Field(
            description=(
                "Should sources that are too close each other be "
                "excluded from photometry?"
            )
        ),
    ] = True

    reject_background_outliers: Annotated[
        bool,
        Field(
            description=(
                "Should extreme pixels in the annulus be rejected "
                "(e.g. from a star in the annulus)?"
            )
        ),
    ] = True

    fwhm_method: Annotated[
        FwhmMethods,
        BeforeValidator(_validate_fwhm_method),
        Field(
            description=(
                "Method for finding the FWHM of the star. 'fit'fits a 1D Gaussian to "
                "the star, 'profiile' fits a 1D Gaussian to the radial profile, and "
                "'moments' uses second order moments of the image, which is terrible."
            ),
            validation_alias=AliasChoices(
                "fwhm_by_fit",  # for backwards compatibility,
                "fwhm_method",  # yes, pydantic does make you do this
            ),
        ),
    ] = FwhmMethods.PROFILE

    method: Annotated[
        Literal["exact", "center", "subpixel"],
        Field(
            description="How to handle partial pixels in the aperture.",
        ),
    ] = "exact"


class PassbandMapEntry(BaseModel):
    """
    A mapping from a single filter name to its corresponding AAVSO filter name.

    Parameters
    ----------
    your_filter_name : str
        Instrumental filter name.

    aavso_filter_name : `stellarphot.settings.AAVSOFilters`
        AAVSO filter name.

    """

    # This ensures that just the first line of the docstring is used as the
    # model description. The json schema is really hard to read if the
    # description is too long.
    model_config = ConfigDict(
        json_schema_extra=dict(
            description=_extract_short_description(__doc__),
        )
    )

    your_filter_name: Annotated[
        NonEmptyStr, Field(description="Instrumental Filter Name")
    ]
    aavso_filter_name: Annotated[AAVSOFilters, Field(title="AAVSO Filter Name")]


class PassbandMap(BaseModelWithTableRep):
    """
    Class to represent a mapping from one set of filter names to another that behaves
    like a `dict`.

    Parameters
    ----------

    name : str
        Name of the passband map.

    your_filter_names_to_aavso : list[`stellarphot.settings.PassbandMapEntry`]
        A list of pairs of your filter name and the corresponding AAVSO filter name.
        This is used to rename the passband entries in the output photometry table.
        Note that, as shown in the example below, you can initialize this with a
        dictionary, and it will be converted to a list of `PassbandMapEntry` objects.

    Notes
    -----

    This class behaves like a dictionary in terms of accessing individual entries but
    you _cannot_ use the `dict` methods to modify the object. This means, for example,
    that if ``my_map`` is a `PassbandMap` object, you can access the AAVSO passband
    that corresponds to your ``B`` passband with ``my_map["B"]`` but you _cannot_
    set entries like this ``my_map["B"] = "B"`` and you _cannot_ delete entries like
    this ``del my_map["B"]``.

    Examples
    --------
    >>> from stellarphot.settings import PassbandMap
    >>> passband_map = PassbandMap(
    ...     name="Sample map",
    ...     your_filter_names_to_aavso={"B": "B", "rp": "SR"}
    ... )
    >>> passband_map
    PassbandMap(name='Sample map', your_filter_names_to_aavso=[PassbandMapEntry(your_...
    >>> # You can access the AAVSO filter name for a given filter name using dict syntax
    >>> passband_map["B"]
    'B'
    >>> passband_map["rp"]
    'SR'
    >>> # If you prefer you can access the individual entries in
    >>> # the list of PassbandMapEntry
    >>> passband_map.your_filter_names_to_aavso[1]
    PassbandMapEntry(your_filter_name='rp', aavso_filter_name=<AAVSOFilters.SR: 'SR'>)
    >>> # Getting the AAVSO filter name this way is a little cumbersome though:
    >>> passband_map.your_filter_names_to_aavso[1].aavso_filter_name.value
    'SR'

    """

    # This ensures that just the first line of the docstring is used as the
    # model description. The json schema is really hard to read if the
    # description is too long.

    # WHen a subclass has a ConfigDict it is merged with the parent class's
    # ConfigDict.
    model_config = ConfigDict(
        json_schema_extra=dict(
            description=_extract_short_description(__doc__),
        )
    )

    name: Annotated[
        NonEmptyStr,
        Field(
            description="Name of the passband map",
            examples=["Filter wheel 1"],
        ),
    ]
    your_filter_names_to_aavso: list[PassbandMapEntry]

    def model_post_init(self, __context: Any) -> None:
        # Create a dictionary from the list of entries so that the object
        # can behave like a dictionary.
        self._dict = {
            entry.your_filter_name: entry.aavso_filter_name.value
            for entry in self.your_filter_names_to_aavso
        }

    @field_validator("your_filter_names_to_aavso", mode="before")
    @classmethod
    def validate_your_filter_names_to_aavso(cls, v):
        if isinstance(v, PassbandMap):
            return v.your_filter_names_to_aavso
        elif isinstance(v, dict):
            return [
                PassbandMapEntry(your_filter_name=k, aavso_filter_name=v)
                for k, v in v.items()
            ]
        else:
            return v

    # All of the remaining methods are to make the object behave like a dictionary.
    # It would have been preferable to subclass `collections.UserDict` but that
    # doesn't work with pydantic models because UserDict objects have a .data attribute
    # but we don't want a pydantic field named "data".
    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def get(self, key, default=None):
        return self._dict.get(key, default)

    def __contains__(self, key):
        return key in self._dict

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)


class LoggingSettings(BaseModelWithTableRep):
    """
    Settings for logging.

    Parameters
    ----------
    logfile : str, optional (Default: None)
        Name of the file to which log messages should be written.  It will
        be created in the `directory_with_images` directory.  If None,
        no log file will be created.

    console_log: bool, optional (Default: True)
        If ``True``, log messages will be written to stdout.  If ``False``, log
        messages will not be written to stdout.

    Examples
    --------
    >>> from stellarphot.settings import LoggingSettings
    >>> logging_settings = LoggingSettings()
    >>> logging_settings
    LoggingSettings(logfile=None, console_log=True)
    """

    # This ensures that just the first line of the docstring is used as the
    # model description. The json schema is really hard to read if the
    # description is too long.

    # WHen a subclass has a ConfigDict it is merged with the parent class's
    # ConfigDict.
    model_config = ConfigDict(
        json_schema_extra=dict(
            description=_extract_short_description(__doc__),
        )
    )

    logfile: Annotated[str | None, Field(description="File to save log to")] = None
    console_log: Annotated[bool, Field(description="Show log on console?")] = True


SCHEMA_EXTRAS = dict(show_null=True)


class PhotometrySettings(BaseModelWithTableRep):
    """
    Settings for performing aperture photometry.

    Parameters
    ----------

    camera : `stellarphot.settings.Camera`
        Camera object which has gain, read noise and dark current set.

    observatory : `stellarphot.settings.Observatory`
        Observatory information.  Used for calculating the BJD.

    photometry_apertures : `stellarphot.settings.PhotometryApertures`
        Radius, inner and outer annulus radii settings and FWHM.

    source_locations : `stellarphot.settings.SourceLocationSettings`
        Settings for the location of the sources for which photometry
        will be performed. See the documentation for
        `~stellarphot.settings.SourceLocationSettings` for details.

    photometry_options : `stellarphot.settings.PhotometryOptions`
        Several options for the details of performing the photometry. See the
        documentation for `~stellarphot.settings.PhotometryOptions` for details.

    passband_map: `stellarphot.settings.PassbandMap`, optional
        A dictionary containing instrumental passband names as keys and
        AAVSO passband names as values. This is used to rename the passband
        entries in the output photometry table from what is in the source list
        to be AAVSO standard names, if available for that filter.

    logging_settings : `stellarphot.settings.LoggingSettings`
        Settings for logging. See the documentation for
        `~stellarphot.settings.LoggingSettings` for details.

    """

    # This ensures that just the first line of the docstring is used as the
    # model description. The json schema is really hard to read if the
    # description is too long.

    # WHen a subclass has a ConfigDict it is merged with the parent class's
    # ConfigDict.
    model_config = ConfigDict(
        json_schema_extra=dict(
            description=_extract_short_description(__doc__),
        )
    )

    camera: Annotated[
        Camera,
        Field(
            description=_extract_short_description(Camera.__doc__),
            json_schema_extra=SCHEMA_EXTRAS,
        ),
    ]
    observatory: Annotated[
        Observatory,
        Field(
            description=_extract_short_description(Observatory.__doc__),
            json_schema_extra=SCHEMA_EXTRAS,
        ),
    ]
    passband_map: Annotated[
        PassbandMap | None,
        Field(
            description=_extract_short_description(PassbandMap.__doc__),
            json_schema_extra=SCHEMA_EXTRAS,
        ),
    ]
    photometry_apertures: Annotated[
        PhotometryApertures,
        Field(
            description=_extract_short_description(PhotometryApertures.__doc__),
            json_schema_extra=SCHEMA_EXTRAS,
        ),
    ]
    source_location_settings: Annotated[
        SourceLocationSettings,
        Field(
            description=_extract_short_description(SourceLocationSettings.__doc__),
            json_schema_extra=SCHEMA_EXTRAS,
        ),
    ]
    photometry_optional_settings: Annotated[
        PhotometryOptionalSettings,
        Field(
            description=_extract_short_description(PhotometryOptionalSettings.__doc__),
            json_schema_extra=SCHEMA_EXTRAS,
        ),
    ]
    logging_settings: Annotated[
        LoggingSettings,
        Field(
            description=_extract_short_description(LoggingSettings.__doc__),
            json_schema_extra=SCHEMA_EXTRAS,
        ),
    ]


# The code for _make_partial_model is adapted from
# https://github.com/pydantic/pydantic/issues/3120#issuecomment-1528030416

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


def _make_partial_model(model: type[BaseModelT], default=None) -> type[BaseModelT]:
    new_fields = {}

    for field_name, field_info in model.model_fields.items():
        new = deepcopy(field_info)
        new.default = default
        new.annotation = Optional[field_info.annotation]  # type: ignore  # noqa: UP007
        new_fields[field_name] = (new.annotation, new)
    return create_model(  # type: ignore
        f"Partial{model.__name__}",
        __base__=model,
        __doc__=model.__doc__,
        __module__=model.__module__,
        **new_fields,
    )


PartialPhotometrySettings = _make_partial_model(PhotometrySettings)


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


class PhotometryRunSettings(BaseModel):
    """
    Settings for a photometry run.

    Parameters
    ----------
    photometry_settings : `stellarphot.settings.PhotometrySettings`
        Settings for performing aperture photometry.

    exoplanet : `stellarphot.settings.Exoplanet`, optional
        Information about an exoplanet transit.

    """

    directory_with_images: Path = "."
    photometry_settings_file: Path = "photometry_settings.json"
    reject_unmatched: Annotated[
        bool,
        Field(
            description="Drop any stars that do not appear in all images",
            tooltip="Reject unmatched stars",
        ),
    ] = False
    object_of_interest: str | None = None
