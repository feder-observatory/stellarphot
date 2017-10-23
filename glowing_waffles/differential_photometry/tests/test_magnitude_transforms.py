from __future__ import print_function, division

import numpy as np

from ..magnitude_transforms import standard_magnitude_transform, filter_transform

import pytest

from astropy.table import Table, Column
from astropy.utils.data import get_pkg_data_filename


@pytest.mark.parametrize('bad_system', [None, 'monkeys'])
def test_filter_transform_bad_system(bad_system):
    fake_data = Table()
    with pytest.raises(ValueError) as e:
        filter_transform(fake_data, 'B', transform=bad_system)
    assert 'Must be one of' in str(e)
    assert str(bad_system) in str(e)


@pytest.mark.parametrize('system', ['ivezic', 'jester'])
def test_filter_transform(system):
    data_file = get_pkg_data_filename('data/mag_transform.csv')
    data = Table.read(data_file)
    in_system = data['system'] == system
    data = data[in_system]
    print(data)
    for output_filter in ['B', 'V', 'R', 'I']:
        f = filter_transform(data, output_filter, g='g', r='r', i='i',
                             transform=system)
        print(f)
        np.testing.assert_allclose(f, data[output_filter])


def test_no_matches():
    RAs = Column(name='RA',
                 data=[1, 2, 3],
                 unit='degree')
    i_Dec = Column(name='Dec',
                   data=[20, 20, 20],
                   unit='degree')

    c_Dec = Column(name='Dec',
                   data=[2, 2, 2],
                   unit='degree')

    R_mags = Column(name='R',
                    data=[15, 15, 15])
    e_R = Column(name='e_R',
                 data=[0.05, 0.1, 0.02])

    B_cat = Column(name='B',
                   data=[16, 16, 16])

    e_B_cat = Column(name='e_B',
                     data=[0.01, 0.04, 0.05])

    V_cat = Column(name='V',
                   data=[14, 15, 16])

    e_V_cat = Column(name='e_V',
                     data=[0.01, 0.04, 0.05])

    instrumental = Table([RAs, i_Dec, R_mags, e_R])
    catalog = Table([RAs, c_Dec, R_mags, B_cat, V_cat, e_R, e_B_cat, e_V_cat])

    # Test for fail if for_filter is omitted.
    with pytest.raises(ValueError) as e:
        standard_magnitude_transform(instrumental, catalog)
    assert 'Must provide a value for for_filter.' in str(e)

    # Test for appropriate error if filter is present in catalog but
    # not instrumental.
    with pytest.raises(ValueError) as e:
        standard_magnitude_transform(instrumental, catalog, for_filter='V')
    assert 'Filter V not found in instrumental table' in str(e)

    # Test that an error is raised when catalog has no matches to sources.
    with pytest.raises(ValueError) as e:
        standard_magnitude_transform(instrumental, catalog, for_filter='R')
    assert ('No matches found between instrumental and catalog '
            'tables.' in str(e))

    catalog['Dec'] = instrumental['Dec']
    # Add a 20 maagnitude offset...
    instrumental['R'] -= 20
    result = standard_magnitude_transform(instrumental, catalog, 'R')
    assert result
    print('=========>>>>>>', result['R'])
    np.testing.assert_allclose(result['R'][0], [0], atol=3e-5)
    np.testing.assert_allclose(result['R'][1], 20, rtol=1e-5)
