from astropy import units as un
from pydantic import ValidationError
import pytest

from stellarphot.settings.models import ApertureSettings, ApertureUnit


DEFAULT_APERTURE_SETTINGS = dict(radius=5,
                                 inner_annulus=10,
                                 outer_annulus=15)


def test_create_aperture_settings_correctly():
    ap_set = ApertureSettings(**DEFAULT_APERTURE_SETTINGS)
    assert ap_set.radius == DEFAULT_APERTURE_SETTINGS['radius']
    assert ap_set.inner_annulus == DEFAULT_APERTURE_SETTINGS['inner_annulus']
    assert ap_set.outer_annulus == DEFAULT_APERTURE_SETTINGS['outer_annulus']


@pytest.mark.parametrize('bad_one', ['radius', 'inner_annulus', 'outer_annulus'])
def test_create_invalid_values(bad_one):
    # Check that individual values that are bad raise an error
    bad_settings = DEFAULT_APERTURE_SETTINGS.copy()
    bad_settings[bad_one] = -1
    with pytest.raises(ValidationError, match=bad_one):
        ApertureSettings(**bad_settings)


def test_set_invalid_values():
    ap = ApertureSettings(**DEFAULT_APERTURE_SETTINGS)
    bad_settings = DEFAULT_APERTURE_SETTINGS.copy()

    with pytest.raises(ValidationError, match="radius must be smaller"):
        ap.radius = 2 * ap.outer_annulus

    with pytest.raises(ValidationError, match="inner_annulus must be smaller"):
        ap.inner_annulus = ap.outer_annulus + 1


# @pytest.mark.parametrize('unit', [1, 'pix', un.meter])
# def test_create_with_wrong_unit(unit):
#     with pytest.raises(ValidationError):
#         ApertureSettings(
#             **DEFAULT_APERTURE_SETTINGS,
#             unit=unit)
