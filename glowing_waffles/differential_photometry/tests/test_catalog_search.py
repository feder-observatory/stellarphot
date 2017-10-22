from __future__ import print_function, division

import numpy as np
import pytest

from ..catalog_search import catalog_clean

from astropy.table import Table


def a_table(masked=False):
    test_table = Table([(1, 2, 3), (1, -1, -1)], names=('a', 'b'),
                       masked=masked)
    return test_table


def test_clean_criteria_none_removed():
    """
    If all rows satisfy the criteria, none should be removed.
    """
    inp = a_table()
    criteria = {'a': '>0'}
    out = catalog_clean(inp, **criteria)
    assert len(out) == len(inp)
    assert (out == inp).all()


@pytest.mark.parametrize("condition",
                         ['>0', '=1', '!=-1', '>=1'])
def test_clean_criteria_some_removed(condition):
    """
    Try a few filters which remove the second row and check that it is
    removed.
    """
    inp = a_table()
    criteria = {'b': condition}
    out = catalog_clean(inp, **criteria)
    assert len(out) == 1
    assert (out[0] == inp[0]).all()


@pytest.mark.parametrize("clean_masked",
                         [False, True])
def test_clean_masked_handled_correctly(clean_masked):
    inp = a_table(masked=True)
    # Mask negative values
    inp['b'].mask = inp['b'] < 0
    out = catalog_clean(inp, remove_rows_with_mask=clean_masked)
    if clean_masked:
        assert len(out) == 1
        assert (np.array(out[0]) == np.array(inp[0])).all()
    else:
        assert len(out) == len(inp)
        assert (out == inp).all()


def test_clean_masked_and_criteria():
    """
    Check whether removing masked rows and using a criteria work
    together.
    """
    inp = a_table(masked=True)
    # Mask the first row.
    inp['b'].mask = inp['b'] > 0

    inp_copy = inp.copy()
    # This should remove the third row.
    criteria = {'a': '<=2'}

    out = catalog_clean(inp, remove_rows_with_mask=True, **criteria)

    # Is only one row left?
    assert len(out) == 1

    # Is the row that is left the same as the second row of the input?
    assert (np.array(out[0]) == np.array(inp[1])).all()

    # Is the input table unchanged?
    assert (inp == inp_copy).all()


@pytest.mark.parametrize("criteria,error_msg", [
                         ({'a': '5'}, "not understood"),
                         ({'a': '<foo'}, "could not convert string")])
def test_clean_bad_criteria(criteria, error_msg):
    """
    Make sure the appropriate error is raised when bad criteria are used.
    """
    inp = a_table(masked=False)

    with pytest.raises(ValueError) as e:
        catalog_clean(inp, **criteria)
    assert error_msg in str(e)
