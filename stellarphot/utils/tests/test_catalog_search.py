import numpy as np
import pytest

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import CCDData

from ..catalog_search import catalog_clean, in_frame, \
                             catalog_search, find_known_variables, \
                             find_apass_stars, filter_catalog
from ...tests.make_wcs import make_wcs

CCD_SHAPE = [2048, 3073]


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
    assert error_msg in str(e.value)


def test_in_frame():
    # This wcs has the identity matrix as the coordinate transform
    wcs = make_wcs()
    # Make the image 10 x 10 pixels
    wcs.pixel_shape = (10, 10)
    coordinates_all_out = SkyCoord(ra=[50, 50], dec=[0, 0], unit='degree')
    should_all_be_out = in_frame(wcs, coordinates_all_out)
    assert not all(should_all_be_out)

    ra_cen, dec_cen = wcs.wcs.crval
    coordinates_all_in = SkyCoord(ra=[ra_cen, ra_cen + 1],
                                  dec=[dec_cen, dec_cen + 1],
                                  unit='degree')
    should_be_all_in = in_frame(wcs, coordinates_all_in)
    assert all(should_be_all_in)

    some_in_some_out = SkyCoord(ra=[50, ra_cen + 1],
                                dec=[0, dec_cen + 1],
                                unit='degree')

    in_out = in_frame(wcs, some_in_some_out)
    assert not in_out[0]
    assert in_out[1]


@pytest.mark.parametrize('clip, data_file',
                         [(True, 'data/clipped_ey_uma_vsx.fits'),
                          (False, 'data/unclipped_ey_uma_vsx.fits')])
def test_catalog_search(clip, data_file):
    # Check that a catalog search of VSX gives us what we expect.
    # I suppose this really isn't future-proof, since more variables
    # could be discovered in the future....
    data = get_pkg_data_filename(data_file)
    expected = Table.read(data)
    wcs_file = get_pkg_data_filename('data/sample_wcs_ey_uma.fits')
    wcs = WCS(fits.open(wcs_file)[0].header)
    wcs.pixel_shape = list(reversed(CCD_SHAPE))
    actual = catalog_search(wcs, CCD_SHAPE, 'B/vsx/vsx',
                            clip_by_frame=clip)
    assert all(actual['OID'] == expected['OID'])


def test_find_known_variables():
    # Under the hood this calls catalog search on the VSX
    # catalog and clips to the frame.
    data_file = 'data/clipped_ey_uma_vsx.fits'
    data = get_pkg_data_filename(data_file)
    expected = Table.read(data)
    wcs_file = get_pkg_data_filename('data/sample_wcs_ey_uma.fits')
    wcs = WCS(fits.open(wcs_file)[0].header)
    wcs.pixel_shape = list(reversed(CCD_SHAPE))
    ccd = CCDData(data=np.zeros(CCD_SHAPE), wcs=wcs, unit='adu')
    vsx = find_known_variables(ccd)
    assert expected['OID'] == vsx['OID']
    assert expected['Name'] == vsx['Name']


def test_catalog_search_from_wcs_or_coord():
    data_file = 'data/sample_wcs_ey_uma.fits'
    data = get_pkg_data_filename(data_file)
    wcs = WCS(fits.open(data)[0].header)
    wcs.pixel_shape = (4096, 4096)
    # Try the search using the WCS alone
    vsx_vars = catalog_search(wcs, [4096, 4096], 'B/vsx/vsx',
                              clip_by_frame=False)
    # And try it using a coordinate instead
    cen_coord = wcs.pixel_to_world(4096 / 2, 4096 / 2)
    # But right now clipping by frame doesn't work without the WCS
    # so don't do that
    vsx_vars2 = catalog_search(cen_coord, [4096, 4096], 'B/vsx/vsx',
                               clip_by_frame=False)
    assert len(vsx_vars) > 0
    assert len(vsx_vars) == len(vsx_vars2)


def test_catalog_search_with_coord_and_frame_clip_fails():
    # Check that calling catalog_search with a coordinate instead
    # of WCS and with clip_by_frame = True generates an appropriate
    # error.
    data_file = 'data/sample_wcs_ey_uma.fits'
    data = get_pkg_data_filename(data_file)
    wcs = WCS(fits.open(data)[0].header)
    cen_coord = wcs.pixel_to_world(4096 / 2, 4096 / 2)
    with pytest.raises(ValueError) as e:
        _ = catalog_search(cen_coord, [4096, 4096], 'B/vsx/vsx',
                           clip_by_frame=True)
    assert 'To clip entries by frame' in str(e.value)


def test_find_apass():
    # This is really checking from APASS DR9 on Vizier, or at least that
    # is where the "expected" data is drawn from.
    expected_all = Table.read(get_pkg_data_filename('data/all_apass_ey_uma_sorted_ra_first_20.fits'))
    expected_low_error = Table.read(get_pkg_data_filename('data/low_error_apass_ey_uma_sorted_ra_first_20.fits'))
    wcs_file = get_pkg_data_filename('data/sample_wcs_ey_uma.fits')
    wcs = WCS(fits.open(wcs_file)[0].header)
    wcs.pixel_shape = list(reversed(CCD_SHAPE))
    ccd = CCDData(data=np.zeros(CCD_SHAPE), wcs=wcs, unit='adu')
    all_apass, apass_low_error = find_apass_stars(ccd)
    # print(all_apass)
    # REference data was sorted by RA, first 20 entries kept
    all_apass.sort('RAJ2000')
    all_apass = all_apass[:20]
    apass_low_error.sort('RAJ2000')
    apass_low_error = apass_low_error[:20]
    # It is hard to imagine the RAs matching and other entries not matching,
    # so just check the RAs.
    assert all(all_apass['RAJ2000'] == expected_all['RAJ2000'])
    assert all(apass_low_error['RAJ2000'] == expected_low_error['RAJ2000'])


def test_filter_catalog():
    # Check basic functionality of table filtering.
    table = a_table()
    print(table)
    output = filter_catalog(table, a=1.5)
    assert output[0]
    assert not output[1]
