import pytest

from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table
from stellarphot.io import AAVSOExtendedFileFormat

DEFAULT_OBSCODE = "ABCDE"


# def test_no_obscode_raises_error():
#     with pytest.raises(TypeError, match="observer_code"):
#         aef = AAVSOExtendedFileFormat()


# def test_default_values():
#     aef = AAVSOExtendedFileFormat(DEFAULT_OBSCODE)
#     assert aef.delim == ","
#     assert len(aef.variable_data) == 0


def test_setting_type_raises_error():
    aef = AAVSOExtendedFileFormat(DEFAULT_OBSCODE)
    with pytest.raises(AttributeError, match="can't set attribute"):
        aef.type = 'STD'


def test_making_table():
    aef = AAVSOExtendedFileFormat(DEFAULT_OBSCODE)
    test_file = get_pkg_data_filename('data/dy-her-aavso-test-data.csv')
    input_data = Table.read(test_file)
    var_star_name = "dy her"
    var_star_id = 1
    # Data here should be the calibrated magnitude and error
    aef.set_data_columns(input_data,
                         {'mag': 'magnitude',
                          'mag_error': 'magerr',
                          'airmass': 'airmass',
                          "jd_mid_utc": "date"},
                         star_id=var_star_id)
    aef.starid = var_star_name
    aef.filter = "rp"

    # If the method is NOT ensemble, then report check star instrumental magnitude.
    check_star_auid = "000-BBX-643"
    check_star_id = 63
    aef.set_data_columns(input_data,
                         {'mag_inst': 'kmag'},
                         star_id=check_star_id)
    aef.kname = check_star_auid

    # Instrumental mag of the comparison star unless ensemble.
    comp_star_auid = "000-BJR-400"
    comp_star_id = 26
    aef.set_data_columns(input_data,
                         {'mag_inst': 'cmag'},
                         star_id=comp_star_id)
    aef.cname = comp_star_auid
    aef_table = aef.to_table()
    assert set(aef_table["KNAME"]) == set([check_star_auid])
    assert set(aef_table["CNAME"]) == set([comp_star_auid])
    assert set(aef_table["STARID"]) == set([var_star_name])
    assert set(aef_table["FILTER"]) == set(["rp"])
    assert len(aef_table) == len(set(input_data['BJD']))
    print(aef_table["DATE"])
    assert 0

# def test_writing():
#     aef = AAVSOExtendedFileFormat(DEFAULT_OBSCODE)
#     aef.write('foo.csv')
#     assert 0