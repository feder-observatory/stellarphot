import numpy as np

from ..magnitude_transforms import (filter_transform,
                                    calculate_transform_coefficients,
                                    transform_magnitudes)

import pytest

from astropy.modeling import models
from astropy.table import Table, Column
from astropy.utils.data import get_pkg_data_filename
from astropy import units as u


def generate_input_mags(n_stars):
    # Generate n_stars with magnitude in range 10 to 15
    rg = np.random.default_rng(1024)
    input_mags = rg.integers(0, high=50, size=n_stars) / 10 + 10
    instr_mags = Column(name='instrumental',
                        data=input_mags)
    return instr_mags


def generate_catalog_mags(instrument_mags, color, model):
    """
    Generate catalog magnitudes from instrumental magnitudes
    given a model that relates the two.
    """
    return instrument_mags + model(color)


def generate_star_coordinates(n_stars,
                              ra_start=180 * u.degree,
                              dec_start=45 * u.degree,
                              separation=10 * u.arcsec):
    """
    Generate RA/Dec coordinates for a set of stars.
    """
    # The plus one guarantees we'll have enough positions for
    # all of our stars.
    max_index = np.int(np.sqrt(n_stars)) + 1

    grids = np.mgrid[:max_index, :max_index]
    dec_grid, ra_grid = grids
    dec_offsets = separation * dec_grid
    ra_offsets = separation * ra_grid
    ra = (ra_start + ra_offsets).flatten()
    dec = (dec_start + dec_offsets).flatten()

    # Slice to return the correct number of positions.
    return ra[:n_stars], dec[:n_stars]


def generate_tables(n_stars, mag_model):
    """
    Generate both tables needed for transforming magnitudes.
    """
    instr_mags = generate_input_mags(n_stars)

    # Set name to match default value in function.
    instr_mags.name = 'mag_inst_r'

    # Set name to be default name for color.
    color = Column(name='B-V',
                   data=np.linspace(0.0, 1.0, num=len(instr_mags)))
    catalog = generate_catalog_mags(instr_mags, color, mag_model)

    # Again, set default name.
    catalog.name = 'r_mag'

    # We'll use the same RA/Dec for the catalog and and the instrumental
    # magnitudes.
    ra, dec = generate_star_coordinates(n_stars)

    # Instrumental magnitudes
    ra_col = Column(name='RA', data=ra)
    dec_col = Column(name='Dec', data=dec)

    instrumental = Table([instr_mags, ra_col, dec_col])

    # Yes, these really do need to be renamed for the catalog table
    ra_col.name = 'RAJ2000'
    dec_col.name = 'DEJ2000'

    catalog_table = Table([catalog, ra_col, dec_col, color])
    return instrumental, catalog_table


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
    instr_mags = generate_input_mags(n_stars)
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
    instr_mags = generate_input_mags(n_stars)
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

    instr_mags = generate_input_mags(n_stars)
    true_relationship = models.Polynomial1D(1, c0=0.5, c1=0.75)
    color = Column(name='color',
                   data=np.linspace(0.0, 1.0, num=len(instr_mags)))
    catalog = generate_catalog_mags(instr_mags, color, true_relationship)

    faint_ones = catalog >= 14
    assert faint_ones.sum() > 0
    assert faint_ones.sum() < n_stars / 2

    # Scramble the faint ones
    rg = np.random.default_rng(40482)
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


@pytest.mark.parametrize('order', [1, 2, 5])
def test_transform_magnitudes_identical_input(order):
    # Analogous to the test case for calculate_transform_coefficients
    # above where the input magnitudes are identical, except the input
    # objects have coordinates.
    n_stars = 100

    zero = models.Const1D(0.0)

    instrumental, catalog_table = generate_tables(n_stars, zero)

    calib_mags, stars_with_match, transform = \
        transform_magnitudes(instrumental, catalog_table, catalog_table,
                             order=order)

    print(calib_mags)
    assert all(calib_mags == catalog_table['r_mag'])
    assert all(stars_with_match)
    assert len(transform.parameters) == order + 1
    assert all(transform.parameters == 0)


@pytest.mark.parametrize('order', [1, 2, 5])
def test_transform_magnitudes_identical_coord_quad_mags(order):
    # Analogous to the test case for calculate_transform_coefficients
    # above where the input magnitudes are identical, except the input
    # objects have coordinates.
    n_stars = 100

    true_relationship = models.Polynomial1D(2, c0=0.5, c1=0.75, c2=0.25)

    instrumental, catalog_table = generate_tables(n_stars, true_relationship)

    calib_mags, stars_with_match, transform = \
        transform_magnitudes(instrumental, catalog_table, catalog_table,
                             order=order)

    assert all(stars_with_match)
    assert len(transform.parameters) == order + 1
    if order >= 2:
        # We expect a good fit in this case
        np.testing.assert_allclose(calib_mags, catalog_table['r_mag'],
                                   rtol=1e-7, atol=1e-7)
        assert np.abs(transform.c0 - true_relationship.c0) < 1e-7
        assert np.abs(transform.c1 - true_relationship.c1) < 1e-7
        assert np.abs(transform.c2 - true_relationship.c2) < 1e-7
    else:
        # But a line just can't fit a quadratic that well
        assert (np.abs(calib_mags - catalog_table['r_mag']) > 1e-5).all()
        assert np.abs(transform.c0 - true_relationship.c0) > 1e-7
        assert np.abs(transform.c1 - true_relationship.c1) > 1e-7
    if order >= 2:
        # Spot check some higher order terms -- they should be zero
        assert all(np.abs(transform.parameters[3:]) < 1e-7)


def test_coordinate_mismatches():
    # Test that stars without close coordinate matches end up
    # marked appropriately.
    n_stars = 100

    true_relationship = models.Polynomial1D(1, c0=0.5, c1=0.75)

    instrumental, catalog_table = generate_tables(n_stars, true_relationship)

    # Mess up the coordinates of half of the stars so that they don't match.
    catalog_table['RAJ2000'][50:] = (catalog_table['RAJ2000'][50:] +
                                     0.5 * u.degree)

    calib_mags, stars_with_match, transform = \
        transform_magnitudes(instrumental, catalog_table, catalog_table[:50],
                             order=2)

    assert all(stars_with_match[:50])
    assert all(~stars_with_match[50:])


def test_coordinate_all_mismatches():
    # Test that when no stars match stuff goes badly.
    n_stars = 100

    true_relationship = models.Polynomial1D(1, c0=0.5, c1=0.75)

    instrumental, catalog_table = generate_tables(n_stars, true_relationship)

    # Mess up the coordinates of half of the stars so that they don't match.
    catalog_table['RAJ2000'] = catalog_table['RAJ2000'] + 0.5 * u.degree

    calib_mags, stars_with_match, transform = \
        transform_magnitudes(instrumental, catalog_table, catalog_table[:50],
                             order=2)

    assert not any(stars_with_match)
