from astropy import units as un
from pydantic import ValidationError
import pytest

from stellarphot.settings.models import ApertureSettings, ApertureUnit


DEFAULT_APERTURE_SETTINGS = dict(radius=5,
                                 inner_annulus=10,
                                 outer_annulus=15)


def test_create_aperture_settings_correctly():
    ap_set = ApertureSettings(
        **DEFAULT_APERTURE_SETTINGS,
        unit=ApertureUnit.PIXEL)
    assert ap_set.radius == 5
    assert ap_set.unit == un.pix


def test_create_with_astropy_unit():
    ap_set = ApertureSettings(
        **DEFAULT_APERTURE_SETTINGS,
        unit=un.pix)
    assert ap_set.unit == un.pix


@pytest.mark.parametrize('unit', [1, 'pix', un.meter])
def test_create_with_wrong_unit(unit):
    with pytest.raises(ValidationError):
        ApertureSettings(
            **DEFAULT_APERTURE_SETTINGS,
            unit=unit)
