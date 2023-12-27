from astropy.time import Time
from pydantic import ValidationError
import pytest

from stellarphot.settings.models import ApertureSettings,Exoplanet

from astropy.coordinates import SkyCoord
import astropy.units as u

DEFAULT_APERTURE_SETTINGS = dict(radius=5, gap=10, annulus_width=15)


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
    identifier="",
    coordinate=SkyCoord(
        ra="00:00:00.00", dec="+00:00:00.0", frame="icrs", unit=("hour", "degree")
    ),
    depth=0,
    duration=0 * u.min,
)


def test_create_exoplanet_correctly():
    planet = Exoplanet(**DEFAULT_EXOPLANET_SETTINGS)
    assert planet.epoch == DEFAULT_EXOPLANET_SETTINGS["epoch"]
    assert u.get_physical_type(planet.period) == "time"
    assert planet.identifier == DEFAULT_EXOPLANET_SETTINGS["identifier"]
    assert planet.coordinate == DEFAULT_EXOPLANET_SETTINGS["coordinate"]
    assert planet.depth == DEFAULT_EXOPLANET_SETTINGS["depth"]
    assert u.get_physical_type(planet.duration) == "time"
