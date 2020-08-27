import pytest

import numpy as np

from stellarphot.analysis.transit_fitting import TransitModelFit


def test_transit_fit_value_length_check():
    # Check that setting inconsistent lengths raises an error
    tmod = TransitModelFit()

    # Should work, all others are None
    tmod.times = list(range(5))

    # Should work, new length is 5
    tmod.airmass = list(range(5))

    # Each of these should raise an error, has different length
    with pytest.raises(ValueError) as e:
        tmod.spp = list(range(3))
    assert 'Length of spp not consistent' in str(e.value)

    with pytest.raises(ValueError) as e:
        tmod.width = list(range(3))
    assert 'Length of width not consistent' in str(e.value)

    with pytest.raises(ValueError) as e:
        tmod.times = list(range(3))
    assert 'Length of times not consistent' in str(e.value)

    with pytest.raises(ValueError) as e:
        tmod.airmass = list(range(3))
    assert 'Length of airmass not consistent' in str(e.value)


def test_transit_fit_setting_independent_vars():
    tmod = TransitModelFit()

    values = list(range(5))

    for attr in ['times', 'airmass', 'width', 'spp']:
        setattr(tmod, attr, values)
        assert getattr(tmod, attr) == values


def test_transit_fit_setting_independent_vars_to_none():
    """
    Setting to None should always work.
    """
    tmod = TransitModelFit()

    values = list(range(5))

    # First set values to something that is not None
    for attr in ['times', 'airmass', 'width', 'spp']:
        setattr(tmod, attr, values)

    # Now set to None and check value
    for attr in ['times', 'airmass', 'width', 'spp']:
        setattr(tmod, attr, None)
        assert getattr(tmod, attr) is None


def test_transit_create_model():
    # Creating a model should not give us an error and should lead
    # to expected values of parameters.
    tmod = TransitModelFit()

    times = list(range(5))

    tmod.times = times
    tmod.setup_model(t0=2, period=5, duration=0.1,
                     depth=10, inclination=90)

    expected_a = 1 / np.sin(0.1 * np.pi / 5)
    assert tmod._model.t0 == 2
    assert tmod._model.rp == 0.1
    assert np.abs(tmod._model.a - expected_a) < 1e-7
    assert tmod._model.period == 5
