import sys

import pytest
import numpy as np

batman = pytest.importorskip('batman')

from stellarphot.analysis.transit_fitting import TransitModelFit


DEFAULT_TESTING_PARAMS = dict(
    t0=2, period=5, duration=0.1,
    depth=10, inclination=90,
    airmass_trend=0.005,
    width_trend=0.01,
    spp_trend=0.002
)

pytestmark = pytest.mark.skipif(sys.platform.startswith('win'),
                    reason="Intermittent hard crash on windows")


def _create_data_from_model_with_trends(transit_model, noise_dev=0.01):
    # Given a fully set up transit model, create some fake
    # data with a touch of noise.

    # Make our own batman model and data from it
    model = batman.TransitModel(transit_model._batman_params,
                                transit_model.times
                                )
    data = model.light_curve(transit_model._batman_params)

    if transit_model.airmass is not None:
        data += transit_model.model.airmass_trend * (transit_model.airmass)

    if transit_model.width is not None:
        data += transit_model.model.width_trend * transit_model.width

    if transit_model.spp is not None:
        data += transit_model.model.spp_trend * transit_model.spp

    if noise_dev > 0:
        # Make some noise
        generator = np.random.default_rng(432132)
        noise = generator.normal(scale=noise_dev, size=len(data))

        data += noise

    return data


def test_transit_fit_value_length_check():
    # Check that setting inconsistent lengths raises an error
    tmod = TransitModelFit()

    # Should work, all others are None
    tmod.times = np.array(list(range(5)))

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

    with pytest.raises(ValueError) as e:
        tmod.data = list(range(3))
    assert 'Length of data not consistent' in str(e.value)


@pytest.mark.parametrize('detrend_by', ['airmass', 'spp', 'width'])
def test_setting_unsetting_parameters_updates_detrend(detrend_by):
    tmod = TransitModelFit()

    dummy_values = np.array(list(range(5)))
    # Should work, all others are None
    tmod.times = dummy_values

    setattr(tmod, detrend_by, dummy_values)
    assert detrend_by in tmod._detrend_parameters

    setattr(tmod, detrend_by, None)
    assert detrend_by not in tmod._detrend_parameters


def test_transit_fit_setting_independent_vars():
    tmod = TransitModelFit()

    values = np.array(list(range(5)))

    for attr in ['times', 'airmass', 'width', 'spp']:
        setattr(tmod, attr, values)
        assert (getattr(tmod, attr) == values).all()


def test_transit_fit_setting_independent_vars_to_none():
    """
    Setting to None should always work.
    """
    tmod = TransitModelFit()

    values = np.array(list(range(5)))

    # First set values to something that is not None
    for attr in ['times', 'airmass', 'width', 'spp', 'data']:
        setattr(tmod, attr, values)

    # Now set to None and check value
    for attr in ['times', 'airmass', 'width', 'spp', 'data']:
        setattr(tmod, attr, None)
        assert getattr(tmod, attr) is None


def test_transit_create_model():
    # Creating a model should not give us an error and should lead
    # to expected values of parameters.
    tmod = TransitModelFit()

    tmod.setup_model(**DEFAULT_TESTING_PARAMS)

    duration = DEFAULT_TESTING_PARAMS['duration']
    period = DEFAULT_TESTING_PARAMS['period']

    expected_a = 1 / np.sin(duration * np.pi / period)
    assert tmod.model.t0 == DEFAULT_TESTING_PARAMS['t0']
    assert tmod.model.rp == np.sqrt(DEFAULT_TESTING_PARAMS['depth'] / 1000)
    assert np.abs(tmod.model.a - expected_a) < 1e-7
    assert tmod.model.period == DEFAULT_TESTING_PARAMS['period']
    assert tmod.model.airmass_trend == DEFAULT_TESTING_PARAMS['airmass_trend']
    assert tmod.model.width_trend == DEFAULT_TESTING_PARAMS['width_trend']
    assert tmod.model.spp_trend == DEFAULT_TESTING_PARAMS['spp_trend']


def _make_transit_model_with_data(noise_dev=1e-1,
                                  with_airmass=False,
                                  with_width=False,
                                  with_spp=False):
    tmod = TransitModelFit()

    tmod.setup_model(**DEFAULT_TESTING_PARAMS)

    t0 = DEFAULT_TESTING_PARAMS['t0']
    duration = DEFAULT_TESTING_PARAMS['duration']

    # Last factor ensures some out of transit data
    start = t0 - duration / 2 * 1.3
    end = t0 + duration / 2 * 1.3

    times = np.linspace(start, end, num=100)
    tmod.times = times

    # Make an airmass that starts at 1.4 and decreases to 1.0
    # over the duration and is parabolic centered on the final time.
    airmass = 0.4 / (times[0] - times[-1])**2 * (times - times[-1]) ** 2 + 1.0

    tmod.airmass = airmass if with_airmass else None

    # Make a width that increases linearly with time. Never happen IRL, but
    # that is ok. Let's start at 5 pixels and increase to 9.
    width = 4 * (times - times[0]) / (times[-1] - times[0]) + 5

    tmod.width = width if with_width else None

    # sky_per_pixel...let's make that sinusoidal. Not realistic but should be
    # easy for the fitter to pick out this trend.
    spp = 30 + 5 * np.sin(4 * np.pi * times / (times[-1] - times[0]))

    tmod.spp = spp if with_spp else None

    data = _create_data_from_model_with_trends(tmod, noise_dev=noise_dev)

    tmod.data = data

    return tmod


def test_transit_fit_all_parameters():
    tmod = _make_transit_model_with_data(noise_dev=1e-5,
                                         with_airmass=True,
                                         with_width=True,
                                         with_spp=True)

    duration = DEFAULT_TESTING_PARAMS['duration']
    period = DEFAULT_TESTING_PARAMS['period']

    expected_a = 1 / np.sin(duration * np.pi / period)

    assert np.abs(tmod.model.a - expected_a) < 1e-6

    tmod.fit()

    # Check the non-exoplanet trends
    for fit_trend in ['airmass_trend', 'width_trend', 'spp_trend']:
        assert (np.abs(getattr(tmod.model, fit_trend)
                       - DEFAULT_TESTING_PARAMS[fit_trend]) < 1e-5)

    # Check a few of the exoplanet parameters
    assert np.abs(tmod.model.t0 - DEFAULT_TESTING_PARAMS['t0']) < 1e-3

    assert np.abs(tmod.model.a - expected_a) < 1e-2

    expected_rp = np.sqrt(DEFAULT_TESTING_PARAMS['depth'] / 1000)
    assert np.abs(tmod.model.rp - expected_rp) < 1e-4

    assert 'airmass' in tmod._detrend_parameters
    assert 'width' in tmod._detrend_parameters
    assert 'spp' in tmod._detrend_parameters


def test_transit_model_detrend():
    tmod = _make_transit_model_with_data(noise_dev=0,
                                         with_airmass=True,
                                         with_width=True,
                                         with_spp=True)

    no_trends = _make_transit_model_with_data(noise_dev=0,
                                              with_airmass=False,
                                              with_width=False,
                                              with_spp=False)
    full_model = tmod.model_light_curve()

    np.testing.assert_allclose(full_model - no_trends.model_light_curve(),
                               tmod.model.airmass_trend * tmod.airmass +
                               tmod.model.width_trend * tmod.width +
                               tmod.model.spp_trend * tmod.spp)

    for trend in tmod._all_detrend_params:
        detrended_model = tmod.model_light_curve(detrend_by=trend)
        trend_param = getattr(tmod.model, f'{trend}_trend')
        trend_data = getattr(tmod, trend)
        assert trend_param != 0
        assert trend_data is not None
        np.testing.assert_allclose(full_model - detrended_model,
                                   trend_param * trend_data)

    detrended_model = tmod.model_light_curve(detrend_by='all')
    np.testing.assert_allclose(detrended_model, no_trends.model_light_curve())


def test_transit_data_detrend():
    tmod = _make_transit_model_with_data(noise_dev=0,
                                         with_airmass=True,
                                         with_width=True,
                                         with_spp=True)

    no_trends = _make_transit_model_with_data(noise_dev=0,
                                              with_airmass=False,
                                              with_width=False,
                                              with_spp=False)

    np.testing.assert_allclose(tmod.data_light_curve(detrend_by='all'),
                               no_trends.data)
    np.testing.assert_allclose(tmod.data - no_trends.data,
                               tmod.model.airmass_trend * tmod.airmass +
                               tmod.model.width_trend * tmod.width +
                               tmod.model.spp_trend * tmod.spp)


def test_transit_fit_parameters_unfreeze_as_expected():
    tmod = _make_transit_model_with_data(noise_dev=1e-5,
                                         with_airmass=False,
                                         with_width=False,
                                         with_spp=False)

    # None of these are fixed by default
    assert not tmod.model.airmass_trend.fixed
    assert not tmod.model.width_trend.fixed
    assert not tmod.model.spp_trend.fixed

    tmod.fit()

    # They should return to their original state after the
    # fit even though they are temporarily fixed during the
    # fit.
    assert not tmod.model.airmass_trend.fixed
    assert not tmod.model.width_trend.fixed
    assert not tmod.model.spp_trend.fixed
