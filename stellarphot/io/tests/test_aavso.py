import pytest

from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table
from stellarphot.io import AAVSOExtendedFileFormat, AAVSOExtendedFileFormatColumns

DEFAULT_OBSCODE = "ABCDE"


def test_no_obscode_raises_error():
    with pytest.raises(TypeError, match="obscode"):
        aef = AAVSOExtendedFileFormat()


def test_default_values():
    aef = AAVSOExtendedFileFormat(DEFAULT_OBSCODE)
    assert aef.delim == ","
    assert len(aef.magnitude) == 0


def test_setting_type_raises_error():
    aef = AAVSOExtendedFileFormat(DEFAULT_OBSCODE)
    with pytest.raises(AttributeError, match="can't set attribute"):
        aef.type = 'STD'


def set_up_aef():
    aef = AAVSOExtendedFileFormat(DEFAULT_OBSCODE)
    aef_cols = AAVSOExtendedFileFormatColumns
    test_file = get_pkg_data_filename('data/dy-her-aavso-test-data.csv')
    input_data = Table.read(test_file)
    var_star_name = "dy her"
    var_star_id = 1
    # Data here should be the calibrated magnitude and error
    aef.set_data_columns(input_data,
                         {'mag': aef_cols.VARIABLE_MAG,
                          'mag_error': aef_cols.VARIABLE_MAG_ERROR,
                          'airmass': aef_cols.AIRMASS,
                          "jd_mid_utc": aef_cols.JD,},
                         star_id=var_star_id)
    aef.starid = var_star_name
    aef.filter = "rp"

    # If the method is NOT ensemble, then report check star instrumental magnitude.
    check_star_auid = "000-BBX-643"
    check_star_id = 63
    aef.set_data_columns(input_data,
                         {'mag_inst': aef_cols.CHECK_STAR_MAG},
                         star_id=check_star_id)
    aef.kname = check_star_auid

    # Instrumental mag of the comparison star unless ensemble.
    comp_star_auid = "000-BJR-400"
    comp_star_id = 26
    aef.set_data_columns(input_data,
                         {'mag_inst': aef_cols.COMP_STAR_MAG},
                         star_id=comp_star_id)
    aef.cname = comp_star_auid
    return aef, input_data


def test_making_table():
    aef, input_data = set_up_aef()
    aef_table = aef.to_table()
    check_star_auid = "000-BBX-643"
    comp_star_auid = "000-BJR-400"
    var_star_name = "dy her"
    assert set(aef_table["KNAME"]) == set([check_star_auid])
    assert set(aef_table["CNAME"]) == set([comp_star_auid])
    assert set(aef_table["STARID"]) == set([var_star_name])
    assert set(aef_table["FILTER"]) == set(["rp"])
    assert len(aef_table) == len(set(input_data['BJD']))


def test_making_table_ensemble():
    aef, input_data = set_up_aef()
    aef.ensemble = True
    assert set(aef.to_table()["CMAG"]) == set(["na"])

    with pytest.raises(ValueError, match="Cannot set comparison star"):
        aef.set_data_columns(input_data,
                             {'mag_inst': AAVSOExtendedFileFormatColumns.COMP_STAR_MAG},
                              star_id=26)
