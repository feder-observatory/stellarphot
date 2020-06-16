import numpy as np

from ..magnitude_transforms import (filter_transform,
                                    calculate_transform_coefficients)

import pytest

from astropy.modeling import models
from astropy.table import Table, Column
from astropy.utils.data import get_pkg_data_filename


def generate_catalog_mags(instrument_mags, color, model):
    """
    Generate catalog magnitudes from instrumental magnitudes
    given a model that relates the two.
    """
    return instrument_mags + model(color)


@pytest.mark.parametrize('bad_system', [None, 'monkeys'])
def test_filter_transform_bad_system(bad_system):
    fake_data = Table()
    with pytest.raises(ValueError) as e:
        filter_transform(fake_data, 'B', transform=bad_system)
    assert 'Must be one of' in str(e.value)
    assert str(bad_system) in str(e.value)


@pytest.mark.parametrize('system', ['ivezic', 'jester'])
def test_filter_transform(system):
    data_file = get_pkg_data_filename('data/mag_transform.csv')
    data = Table.read(data_file)
    in_system = data['system'] == system
    data = data[in_system]
    for output_filter in ['B', 'V', 'R', 'I']:
        f = filter_transform(data, output_filter, g='g', r='r', i='i',
                             transform=system)
        np.testing.assert_allclose(f, data[output_filter])


def test_filter_transform_bad_filter():
    with pytest.raises(ValueError) as e:
        filter_transform([], 'not a filter', transform='jester')
    assert 'the desired filter must be a string R B V or I' in str(e)


@pytest.mark.parametrize('order', [1, 2, 5])
def test_catalog_same_as_input(order):
    # Check that we get the correct transform when catalog magnitudes
    # are identical to instrument magnitudes.
    instr_mags = Column(name='instrumental', data=[10, 12.5, 11])
    zero = models.Const1D(0.0)
    color = Column(name='color', data=[1.0] * len(instr_mags))
    catalog = generate_catalog_mags(instr_mags, color, zero)

    _, fit_model = calculate_transform_coefficients(instr_mags,
                                                    catalog,
                                                    color,
                                                    order=order)
    assert len(fit_model.parameters) == order + 1
    assert all(fit_model.parameters == 0)


@pytest.mark.parametrize('order', [1, 2, 5])
def test_catalog_linear_to_input(order):
    # Check that we recover the correct relationship between
    # the catalog and instrumental magnitudes when the relationship
    # between the two is linear.
    n_stars = 100
    input_mags = np.random.random_integers(0, 50, n_stars) / 10 + 10
    instr_mags = Column(name='instrumental',
                        data=input_mags)
    true_relationship = models.Polynomial1D(1, c0=0.5, c1=0.75)
    color = Column(name='color',
                   data=np.linspace(0.0, 1.0, num=len(instr_mags)))
    catalog = generate_catalog_mags(instr_mags, color, true_relationship)

    _, fit_model = calculate_transform_coefficients(instr_mags,
                                                    catalog,
                                                    color,
                                                    order=order)
    assert len(fit_model.parameters) == order + 1
    assert np.abs(fit_model.c0 - true_relationship.c0) < 1e-7
    assert np.abs(fit_model.c1 - true_relationship.c1) < 1e-7

    if order >= 2:
        # Spot check some higher order terms -- they should be zero
        assert all(np.abs(fit_model.parameters[2:]) < 1e-7)


@pytest.mark.parametrize('order', [1, 2, 5])
def test_catalog_quadratic_to_input(order):
    # Check that we recover the correct relationship between
    # the catalog and instrumental magnitudes when the relationship
    # between the two is linear.
    n_stars = 100
    input_mags = np.random.random_integers(0, 50, n_stars) / 10 + 10
    instr_mags = Column(name='instrumental',
                        data=input_mags)
    true_relationship = models.Polynomial1D(2, c0=0.5, c1=0.75, c2=0.25)
    color = Column(name='color',
                   data=np.linspace(0.0, 1.0, num=len(instr_mags)))
    catalog = generate_catalog_mags(instr_mags, color, true_relationship)
    _, fit_model = calculate_transform_coefficients(instr_mags,
                                                    catalog,
                                                    color,
                                                    order=order)
    assert len(fit_model.parameters) == order + 1
    if order >= 2:
        # We expect a good fit in this case
        assert np.abs(fit_model.c0 - true_relationship.c0) < 1e-7
        assert np.abs(fit_model.c1 - true_relationship.c1) < 1e-7
        assert np.abs(fit_model.c2 - true_relationship.c2) < 1e-7
    else:
        # But a line just can't fit a quadratic that well
        assert np.abs(fit_model.c0 - true_relationship.c0) > 1e-7
        assert np.abs(fit_model.c1 - true_relationship.c1) > 1e-7
    if order >= 2:
        # Spot check some higher order terms -- they should be zero
        assert all(np.abs(fit_model.parameters[3:]) < 1e-7)


@pytest.mark.parametrize('faintest_magnitude', [None, 14])
def test_faintest_magnitude_has_effect(faintest_magnitude):
    # Check that the limit on magnitude when doing fits is respected.
    # We'll do this by setting up a linear relationship then
    # setting the catalog data fainter than the limit to nonsense.
    # Two outcomes we expect:
    #
    # 1. Without a limiting magnitude the fit should not be very good.
    # 2. With a limit the fit should be as good as it was before.
    n_stars = 100

    # Magnitudes are 10 to 15
    rg = np.random.default_rng(1024)
    input_mags = rg.integers(0, high=50, size=n_stars) / 10 + 10
    instr_mags = Column(name='instrumental',
                        data=input_mags)
    true_relationship = models.Polynomial1D(1, c0=0.5, c1=0.75)
    color = Column(name='color',
                   data=np.linspace(0.0, 1.0, num=len(instr_mags)))
    catalog = generate_catalog_mags(instr_mags, color, true_relationship)

    faint_ones = catalog >= 14
    assert faint_ones.sum() > 0
    assert faint_ones.sum() < n_stars / 2

    # Scramble the faint ones
    catalog[faint_ones] = (catalog[faint_ones]
                           + 5 * rg.random(faint_ones.sum()))

    _, fit_model = calculate_transform_coefficients(
                        instr_mags,
                        catalog,
                        color,
                        order=1,
                        faintest_mag=faintest_magnitude,
                        sigma=5000 # So that nothing is clipped
                    )

    if faintest_magnitude:
        assert np.abs(fit_model.c0 - true_relationship.c0) < 1e-7
        assert np.abs(fit_model.c1 - true_relationship.c1) < 1e-7
    else:
        assert np.abs(fit_model.c0 - true_relationship.c0) > 1e-2
        assert np.abs(fit_model.c1 - true_relationship.c1) > 1e-2

# def test_no_matches():
#     RAs = Column(name='RA',
#                  data=[1, 2, 3],
#                  unit='degree')
#     i_Dec = Column(name='Dec',
#                    data=[20, 20, 20],
#                    unit='degree')

#     c_Dec = Column(name='Dec',
#                    data=[2, 2, 2],
#                    unit='degree')

#     R_mags = Column(name='R',
#                     data=[15, 15, 15])
#     e_R = Column(name='e_R',
#                  data=[0.05, 0.1, 0.02])

#     B_cat = Column(name='B',
#                    data=[16, 16, 16])

#     e_B_cat = Column(name='e_B',
#                      data=[0.01, 0.04, 0.05])

#     V_cat = Column(name='V',
#                    data=[14, 15, 16])

#     e_V_cat = Column(name='e_V',
#                      data=[0.01, 0.04, 0.05])

#     instrumental = Table([RAs, i_Dec, R_mags, e_R])
#     catalog = Table([RAs, c_Dec, R_mags, B_cat, V_cat, e_R, e_B_cat, e_V_cat])

#     # Test for fail if for_filter is omitted.
#     with pytest.raises(ValueError) as e:
#         standard_magnitude_transform(instrumental, catalog)
#     assert 'Must provide a value for for_filter.' in str(e)

#     # Test for appropriate error if filter is present in catalog but
#     # not instrumental.
#     with pytest.raises(ValueError) as e:
#         standard_magnitude_transform(instrumental, catalog, for_filter='V')
#     assert 'Filter V not found in instrumental table' in str(e)

#     # Test that an error is raised when catalog has no matches to sources.
#     with pytest.raises(ValueError) as e:
#         standard_magnitude_transform(instrumental, catalog, for_filter='R')
#     assert ('No matches found between instrumental and catalog '
#             'tables.' in str(e))

#     catalog['Dec'] = instrumental['Dec']
#     # Add a 20 maagnitude offset...
#     instrumental['R'] -= 20
#     result = standard_magnitude_transform(instrumental, catalog, 'R')
#     assert result
#     print('=========>>>>>>', result['R'])
#     np.testing.assert_allclose(result['R'][0], [0], atol=3e-5)
#     np.testing.assert_allclose(result['R'][1], 20, rtol=1e-5)
