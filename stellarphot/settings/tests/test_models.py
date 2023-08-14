from astropy import units as un
from pydantic import ValidationError
import pytest

from stellarphot.settings.models import ApertureSettings, ApertureUnit


DEFAULT_APERTURE_SETTINGS = dict(radius=5,
                                 gap=10,
                                 annulus_width=15)


def test_create_aperture_settings_correctly():
    ap_set = ApertureSettings(**DEFAULT_APERTURE_SETTINGS)
    assert ap_set.radius == DEFAULT_APERTURE_SETTINGS['radius']
    assert (ap_set.inner_annulus ==
            DEFAULT_APERTURE_SETTINGS['radius'] + DEFAULT_APERTURE_SETTINGS['gap'])
    assert (ap_set.outer_annulus ==
            DEFAULT_APERTURE_SETTINGS['radius'] +
            DEFAULT_APERTURE_SETTINGS['gap'] +
            DEFAULT_APERTURE_SETTINGS['annulus_width'])


@pytest.mark.parametrize('bad_one', ['radius', 'gap', 'annulus_width'])
def test_create_invalid_values(bad_one):
    # Check that individual values that are bad raise an error
    bad_settings = DEFAULT_APERTURE_SETTINGS.copy()
    bad_settings[bad_one] = -1
    with pytest.raises(ValidationError, match=bad_one):
        ApertureSettings(**bad_settings)
