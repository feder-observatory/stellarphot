import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pydantic import ValidationError

from stellarphot.settings.models import ApertureSettings, Camera, Exoplanet

DEFAULT_APERTURE_SETTINGS = dict(radius=5, gap=10, annulus_width=15)

TEST_CAMERA_VALUES = dict(
    data_unit=u.adu,
    gain=2.0 * u.electron / u.adu,
    read_noise=10 * u.electron,
    dark_current=0.01 * u.electron / u.second,
    pixel_scale=0.563 * u.arcsec / u.pix,
    max_data_value=50000 * u.adu,
)


def test_camera_attributes():
    # Check that the attributes are set properly
    data_unit = u.adu
    gain = 2.0 * u.electron / u.adu
    read_noise = 10 * u.electron
    dark_current = 0.01 * u.electron / u.second
    pixel_scale = 0.563 * u.arcsec / u.pix
    max_val = 50000 * u.adu

    c = Camera(
        data_unit=data_unit,
        gain=gain,
        read_noise=read_noise,
        dark_current=dark_current,
        pixel_scale=pixel_scale,
        max_data_value=max_val,
    )
    assert c.data_unit == data_unit
    assert c.gain == gain
    assert c.dark_current == dark_current
    assert c.read_noise == read_noise
    assert c.pixel_scale == pixel_scale
    assert c.max_data_value == max_val


def test_camera_unitscheck():
    # Check that the units are checked properly
    gain = 2.0 * u.electron / u.adu
    read_noise = 10 * u.electron
    dark_current = 0.01 * u.electron / u.second
    pixel_scale = 0.563 * u.arcsec / u.pix
    max_adu = 50000 * u.adu

    # All 5 of the attributes after data_unit will be checked for units
    # and noted in the ValidationError message. Rather than checking
    # separately for all 5, we just check for the presence of the
    # right number of errors
    with pytest.raises(ValidationError, match="5 validation errors"):
        Camera(
            data_unit=u.adu,
            gain=gain.value,
            read_noise=read_noise.value,
            dark_current=dark_current.value,
            pixel_scale=pixel_scale.value,
            max_data_value=max_adu.value,
        )


def test_camera_negative_max_adu():
    # Check that the units are checked properly
    data_unit = u.adu
    gain = 2.0 * u.electron / u.adu
    read_noise = 10 * u.electron
    dark_current = 0.01 * u.electron / u.second
    pixel_scale = 0.563 * u.arcsec / u.pix
    max_val = -50000 * u.adu

    # Make sure that a negative max_adu raises an error
    with pytest.raises(ValidationError, match="must be positive"):
        Camera(
            data_unit=data_unit,
            gain=gain,
            read_noise=read_noise,
            dark_current=dark_current,
            pixel_scale=pixel_scale,
            max_data_value=max_val,
        )


def test_camera_incompatible_gain_units():
    data_unit = u.adu
    gain = 2.0 * u.count / u.adu
    read_noise = 10 * u.electron
    dark_current = 0.01 * u.electron / u.second
    pixel_scale = 0.563 * u.arcsec / u.pix
    max_val = 50000 * u.adu

    # Make sure that an incompatible gain raises an error
    with pytest.raises(ValidationError, match="Gain units.*not compatible"):
        Camera(
            data_unit=data_unit,
            gain=gain,
            read_noise=read_noise,
            dark_current=dark_current,
            pixel_scale=pixel_scale,
            max_data_value=max_val,
        )


def test_camera_incompatible_max_val_units():
    data_unit = u.adu
    gain = 2.0 * u.electron / u.adu
    read_noise = 10 * u.electron
    dark_current = 0.01 * u.electron / u.second
    pixel_scale = 0.563 * u.arcsec / u.pix
    max_val = 50000 * u.count

    # Make sure that an incompatible gain raises an error
    with pytest.raises(
        ValidationError, match="Maximum data value units.*not consistent"
    ):
        Camera(
            data_unit=data_unit,
            gain=gain,
            read_noise=read_noise,
            dark_current=dark_current,
            pixel_scale=pixel_scale,
            max_data_value=max_val,
        )


def test_camera_copy():
    # Make sure copy actually copies everything
    gain = 2.0 * u.electron / u.adu
    read_noise = 10 * u.electron
    dark_current = 0.01 * u.electron / u.second
    pixel_scale = 0.563 * u.arcsec / u.pix
    c = Camera(
        data_unit=u.adu,
        gain=gain,
        read_noise=read_noise,
        dark_current=dark_current,
        pixel_scale=pixel_scale,
        max_data_value=65535 * u.adu,
    )
    c2 = c.copy()
    assert c2 == c


def test_camera_altunitscheck():
    # Check to see that 'count' is allowed instead of 'electron'
    data_unit = u.adu
    gain = 2.0 * u.count / u.adu
    read_noise = 10 * u.count
    dark_current = 0.01 * u.count / u.second
    pixel_scale = 0.563 * u.arcsec / u.pix
    max_val = 50000 * u.adu

    c = Camera(
        data_unit=data_unit,
        gain=gain,
        read_noise=read_noise,
        dark_current=dark_current,
        pixel_scale=pixel_scale,
        max_data_value=max_val,
    )
    assert c.data_unit == data_unit
    assert c.gain == gain
    assert c.dark_current == dark_current
    assert c.read_noise == read_noise
    assert c.pixel_scale == pixel_scale
    assert c.max_data_value == max_val


def test_camera_schema():
    # Check that we can generate a schema for a Camera and that it
    # has the right number of attributes
    c = Camera(
        data_unit=u.adu,
        gain=5 * u.electron / u.adu,
        read_noise=1 * u.electron,
        dark_current=0.1 * u.electron / u.second,
        pixel_scale=0.6 * u.arcsec / u.pix,
        max_data_value=65535 * u.adu,
    )
    schema = c.schema()
    assert len(schema["properties"]) == 6


def test_camera_json_round_trip():
    # Check that a camera can be converted to json and back
    data_unit = u.adu
    gain = 2.0 * u.electron / u.adu
    read_noise = 10 * u.electron
    dark_current = 0.01 * u.electron / u.second
    pixel_scale = 0.563 * u.arcsec / u.pix
    max_val = 50000 * u.adu

    c = Camera(
        data_unit=data_unit,
        gain=gain,
        read_noise=read_noise,
        dark_current=dark_current,
        pixel_scale=pixel_scale,
        max_data_value=max_val,
    )

    c2 = Camera.parse_raw(c.json())
    assert c2 == c


def test_create_aperture_settings_correctly():
    ap_set = ApertureSettings(**DEFAULT_APERTURE_SETTINGS)
    assert ap_set.radius == DEFAULT_APERTURE_SETTINGS["radius"]
    assert (
        ap_set.inner_annulus
        == DEFAULT_APERTURE_SETTINGS["radius"] + DEFAULT_APERTURE_SETTINGS["gap"]
    )
    assert (
        ap_set.outer_annulus
        == DEFAULT_APERTURE_SETTINGS["radius"]
        + DEFAULT_APERTURE_SETTINGS["gap"]
        + DEFAULT_APERTURE_SETTINGS["annulus_width"]
    )


@pytest.mark.parametrize("bad_one", ["radius", "gap", "annulus_width"])
def test_create_invalid_values(bad_one):
    # Check that individual values that are bad raise an error
    bad_settings = DEFAULT_APERTURE_SETTINGS.copy()
    bad_settings[bad_one] = -1
    with pytest.raises(ValidationError, match=bad_one):
        ApertureSettings(**bad_settings)


DEFAULT_EXOPLANET_SETTINGS = dict(
    epoch=Time(0, format="jd"),
    period=0 * u.min,
    identifier="a planet",
    coordinate=SkyCoord(
        ra="00:00:00.00", dec="+00:00:00.0", frame="icrs", unit=("hour", "degree")
    ),
    depth=0,
    duration=0 * u.min,
)


def test_create_exoplanet_correctly():
    planet = Exoplanet(**DEFAULT_EXOPLANET_SETTINGS)
    print(planet)
    assert planet.epoch == DEFAULT_EXOPLANET_SETTINGS["epoch"]
    assert u.get_physical_type(planet.period) == "time"
    assert planet.identifier == DEFAULT_EXOPLANET_SETTINGS["identifier"]
    assert planet.coordinate == DEFAULT_EXOPLANET_SETTINGS["coordinate"]
    assert planet.depth == DEFAULT_EXOPLANET_SETTINGS["depth"]
    assert u.get_physical_type(planet.duration) == "time"


def test_create_invalid_exoplanet():
    values = DEFAULT_EXOPLANET_SETTINGS.copy()
    # Make pediod and duration have invalid units for a time
    values["period"] = values["period"].value * u.m
    values["duration"] = values["duration"].value * u.m
    # Check that individual values that are bad raise an error
    with pytest.raises(ValidationError, match="2 validation errors"):
        Exoplanet(**values)
