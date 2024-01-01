import warnings

import pytest
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from astropy.io import ascii, fits
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.nddata import CCDData
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from astropy.wcs.wcs import FITSFixedWarning

from pydantic import ValidationError
from stellarphot.core import (
    Camera,
    BaseEnhancedTable,
    PhotometryData,
    CatalogData,
    apass_dr9,
    vsx_vizier,
    SourceListData,
)


def test_camera_attributes():
    # Check that the attributes are set properly
    data_unit = u.adu
    gain = 2.0 * u.electron / u.adu
    read_noise = 10 * u.electron
    dark_current = 0.01 * u.electron / u.second
    pixel_scale = 0.563 * u.arcsec / u.pix
    max_val = 50000 * u.adu

    c = Camera(
        data_unit=data_unit,
        gain=gain,
        read_noise=read_noise,
        dark_current=dark_current,
        pixel_scale=pixel_scale,
        max_data_value=max_val,
    )
    assert c.data_unit == data_unit
    assert c.gain == gain
    assert c.dark_current == dark_current
    assert c.read_noise == read_noise
    assert c.pixel_scale == pixel_scale
    assert c.max_data_value == max_val


def test_camera_unitscheck():
    # Check that the units are checked properly
    gain = 2.0 * u.electron / u.adu
    read_noise = 10 * u.electron
    dark_current = 0.01 * u.electron / u.second
    pixel_scale = 0.563 * u.arcsec / u.pix
    max_adu = 50000 * u.adu

    # All 5 of the attributes after data_unit will be checked for units
    # and noted in the ValidationError message. Rather than checking
    # separately for all 5, we just check for the presence of the
    # right number of errors
    with pytest.raises(ValidationError, match="5 validation errors"):
        Camera(
            data_unit=u.adu,
            gain=gain.value,
            read_noise=read_noise.value,
            dark_current=dark_current.value,
            pixel_scale=pixel_scale.value,
            max_data_value=max_adu.value,
        )


def test_camera_negative_max_adu():
    # Check that the units are checked properly
    data_unit = u.adu
    gain = 2.0 * u.electron / u.adu
    read_noise = 10 * u.electron
    dark_current = 0.01 * u.electron / u.second
    pixel_scale = 0.563 * u.arcsec / u.pix
    max_val = -50000 * u.adu

    # Make sure that a negative max_adu raises an error
    with pytest.raises(ValidationError, match="must be positive"):
        Camera(
            data_unit=u.adu,
            gain=gain,
            read_noise=read_noise,
            dark_current=dark_current,
            pixel_scale=pixel_scale,
            max_data_value=max_val,
        )


def test_camera_incompatible_gain_units():
    data_unit = u.adu
    gain = 2.0 * u.count / u.adu
    read_noise = 10 * u.electron
    dark_current = 0.01 * u.electron / u.second
    pixel_scale = 0.563 * u.arcsec / u.pix
    max_val = 50000 * u.adu

    # Make sure that an incompatible gain raises an error
    with pytest.raises(ValidationError, match="Gain units.*not compatible"):
        Camera(
            data_unit=u.adu,
            gain=gain,
            read_noise=read_noise,
            dark_current=dark_current,
            pixel_scale=pixel_scale,
            max_data_value=max_val,
        )


def test_camera_incompatible_max_val_units():
    data_unit = u.adu
    gain = 2.0 * u.electron / u.adu
    read_noise = 10 * u.electron
    dark_current = 0.01 * u.electron / u.second
    pixel_scale = 0.563 * u.arcsec / u.pix
    max_val = 50000 * u.count

    # Make sure that an incompatible gain raises an error
    with pytest.raises(
        ValidationError, match="Maximum data value units.*not consistent"
    ):
        Camera(
            data_unit=u.adu,
            gain=gain,
            read_noise=read_noise,
            dark_current=dark_current,
            pixel_scale=pixel_scale,
            max_data_value=max_val,
        )


def test_camera_copy():
    # Make sure copy actually copies everything
    gain = 2.0 * u.electron / u.adu
    read_noise = 10 * u.electron
    dark_current = 0.01 * u.electron / u.second
    pixel_scale = 0.563 * u.arcsec / u.pix
    c = Camera(
        data_unit=u.adu,
        gain=gain,
        read_noise=read_noise,
        dark_current=dark_current,
        pixel_scale=pixel_scale,
        max_data_value=65535 * u.adu,
    )
    c2 = c.copy()
    assert c2 == c


def test_camera_altunitscheck():
    # Check to see that 'count' is allowed instead of 'electron'
    data_unit = u.adu
    gain = 2.0 * u.count / u.adu
    read_noise = 10 * u.count
    dark_current = 0.01 * u.count / u.second
    pixel_scale = 0.563 * u.arcsec / u.pix
    max_val = 50000 * u.adu

    c = Camera(
        data_unit=data_unit,
        gain=gain,
        read_noise=read_noise,
        dark_current=dark_current,
        pixel_scale=pixel_scale,
        max_data_value=max_val,
    )
    assert c.data_unit == data_unit
    assert c.gain == gain
    assert c.dark_current == dark_current
    assert c.read_noise == read_noise
    assert c.pixel_scale == pixel_scale
    assert c.max_data_value == max_val


def test_camera_schema():
    # Check that we can generate a schema for a Camera and that it
    # has the right number of attributes
    c = Camera(
        data_unit=u.adu,
        gain=5 * u.electron / u.adu,
        read_noise=1 * u.electron,
        dark_current=0.1 * u.electron / u.second,
        pixel_scale=0.6 * u.arcsec / u.pix,
        max_data_value=65535 * u.adu,
    )
    schema = c.schema()
    assert len(schema["properties"]) == 6


# Create several test descriptions for use in base_enhanced_table tests.
test_descript = {
    "id": None,
    "ra": u.deg,
    "dec": u.deg,
    "sky_per_pix_avg": u.adu,
    "sky_per_pix_med": u.adu,
    "sky_per_pix_std": u.adu,
    "fwhm_x": u.pix,
    "fwhm_y": u.pix,
    "width": u.pix,
}

# Define a realistic table of astronomical data contianing one row
data = np.array(
    [
        [
            1,
            78.17278712191920,
            22.505771480719400,
            31.798216414544900,
            31.658750534057600,
            9.294325523269860,
            13.02511260943810,
            13.02511260943810,
            13.02511260943810,
        ]
    ]
)
colnames = [
    "id",
    "ra",
    "dec",
    "sky_per_pix_avg",
    "sky_per_pix_med",
    "sky_per_pix_std",
    "fwhm_x",
    "fwhm_y",
    "width",
]
coltypes = [
    "int",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
]
colunits = [None, u.deg, u.deg, u.adu, u.adu, u.adu, u.pix, u.pix, u.pix]
testdata = Table(data, names=colnames, dtype=coltypes, units=colunits)

# Define some configuration information assuming Feder telescope
feder_cg_16m = Camera(
    data_unit=u.adu,
    gain=1.5 * u.electron / u.adu,
    read_noise=10.0 * u.electron,
    dark_current=0.01 * u.electron / u.second,
    pixel_scale=0.563 * u.arcsec / u.pix,
    max_data_value=50000 * u.adu,
)
feder_passbands = {"up": "SU", "gp": "SG", "rp": "SR", "zp": "SZ", "ip": "SI"}
feder_obs = EarthLocation(lat=46.86678, lon=-96.45328, height=311)


def test_base_enhanced_table_blank():
    # This should just return a blank BaseEnhancedTable
    test_base = BaseEnhancedTable()
    assert isinstance(test_base, BaseEnhancedTable)
    assert len(test_base) == 0


def test_base_enhanced_table_from_existing_table():
    # Should create a populated dataset properly and display the astropy data
    test_base2 = BaseEnhancedTable(table_description=test_descript, input_data=testdata)
    assert len(test_base2["ra"]) == 1
    assert len(test_base2["dec"]) == 1


def test_base_enhanced_table_clean():
    # Check that the clean method exists
    test_base = BaseEnhancedTable(table_description=test_descript, input_data=testdata)
    # Add a row so that we can clean something
    test_base_two = test_base.copy()
    test_base_two.add_row(test_base[0])
    test_base_two["ra"][1] = -test_base_two["ra"][1]
    test_cleaned = test_base_two.clean(ra=">0.0")
    assert len(test_cleaned) == 1
    assert test_cleaned == test_base


def a_table(masked=False):
    test_table = Table([(1, 2, 3), (1, -1, -2)], names=("a", "b"), masked=masked)
    test_table = BaseEnhancedTable(
        table_description={"a": None, "b": None}, input_data=test_table
    )
    return test_table


def test_bet_clean_criteria_none_removed():
    """
    If all rows satisfy the criteria, none should be removed.
    """
    inp = a_table()
    criteria = {"a": ">0"}
    out = inp.clean(**criteria)
    assert len(out) == len(inp)
    assert (out == inp).all()


@pytest.mark.parametrize(
    "condition,input_row", [(">0", 0), ("=1", 0), (">=1", 0), ("<-1", 2), ("=-1", 1)]
)
def test_bet_clean_criteria_some_removed(condition, input_row):
    """
    Try a few filters which leave only one row and make sure that row is
    returned.
    """
    inp = a_table()
    criteria = {"b": condition}
    out = inp.clean(**criteria)
    assert len(out) == 1
    assert (out[0] == inp[input_row]).all()


@pytest.mark.parametrize(
    "criteria,error_msg",
    [({"a": "5"}, "not understood"), ({"a": "<foo"}, "could not convert string")],
)
def test_clean_bad_criteria(criteria, error_msg):
    """
    Make sure the appropriate error is raised when bad criteria are used.
    """
    inp = a_table(masked=False)

    with pytest.raises(ValueError, match=error_msg):
        inp.clean(**criteria)


@pytest.mark.parametrize("clean_masked", [False, True])
def test_clean_masked_handled_correctly(clean_masked):
    inp = a_table(masked=True)
    # Mask negative values
    inp["b"].mask = inp["b"] < 0
    out = inp.clean(remove_rows_with_mask=clean_masked)
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
    inp["b"].mask = inp["b"] > 0

    inp_copy = inp.copy()
    # This should remove the third row.
    criteria = {"a": "<=2"}

    out = inp.clean(remove_rows_with_mask=True, **criteria)

    # Is only one row left?
    assert len(out) == 1

    # Is the row that is left the same as the second row of the input?
    assert (np.array(out[0]) == np.array(inp[1])).all()

    # Is the input table unchanged?
    assert (inp == inp_copy).all()


def test_clean_criteria_none_removed():
    """
    If all rows satisfy the criteria, none should be removed.
    """
    inp = a_table()
    criteria = {"a": ">0"}
    out = inp.clean(**criteria)
    assert len(out) == len(inp)
    assert (out == inp).all()


def test_base_enhanced_table_missing_column():
    # Should raise exception because the RA data is missing from input data
    testdata_nora = testdata.copy()
    testdata_nora.remove_column("ra")
    with pytest.raises(ValueError):
        test_base = BaseEnhancedTable(
            table_description=test_descript, input_data=testdata_nora
        )


def test_base_enhanced_table_missing_badunits():
    # This will fail due to RA being in units of hours
    bad_ra_descript = test_descript.copy()
    bad_ra_descript[1, 2] = u.hr

    with pytest.raises(ValueError):
        test_base = BaseEnhancedTable(
            table_description=bad_ra_descript, input_data=testdata
        )


def test_base_enhanced_table_recursive():
    # Should create a populated dataset properly and display the astropy data
    test_base2 = BaseEnhancedTable(table_description=test_descript, input_data=testdata)
    assert len(test_base2["ra"]) == 1
    assert len(test_base2["dec"]) == 1

    # Attempt recursive call
    with pytest.raises(TypeError):
        test_base3 = BaseEnhancedTable(
            table_description=test_descript, input_data=test_base2
        )


# Define a realistic table of photometry data (a bit corrupted)
photdata = np.array(
    [
        [
            1,
            2049.145245206124,
            2054.0849947477964,
            109070.60831212997,
            154443.9371254444,  # has wrong units
            78.17278712191924,
            22.505771480719375,
            31.798216414544864,  # has wrong units
            31.658750534057617,  # has wrong units
            9.294325523269857,  # has wrong units
            13.02511260943813,  # has wrong units
            13.02511260943813,  # has wrong units
            13.02511260943813,  # has wrong units
            29.0,
            2642.079421669016,
            44.0,
            59.0,
            4853.760649796231,
            120.0,
            "2022-11-27T06:26:29.620",  # created as a string, not a Time object
            59909,
            25057.195077483062,
            2459910.7754060575,
            -6.239606167785804,
            1.115,
            "ip",
            "TIC_467615239.01-S001-R001-C001-ip.fit",
            1,
            0.02320185643388203,
            803.1970935659333,
            535.4647290439556,  # has wrong units
            46.795229859903905,  # has wrong units -- maybe??
        ]
    ]
)
photcolnames = [
    "id",
    "xcenter",
    "ycenter",
    "aperture_sum",
    "annulus_sum",
    "ra",
    "dec",
    "sky_per_pix_avg",
    "sky_per_pix_med",
    "sky_per_pix_std",
    "fwhm_x",
    "fwhm_y",
    "width",
    "aperture",
    "aperture_area",
    "annulus_inner",
    "annulus_outer",
    "annulus_area",
    "exposure",
    "date-obs",
    "night",
    "aperture_net_cnts",
    "bjd",
    "mag_inst",
    "airmass",
    "passband",
    "file",
    "star_id",
    "mag_error",
    "noise_electrons",
    "noise_cnts",
    "snr",
]
photcoltypes = [
    "int",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "str",
    "int",
    "float",
    "float",
    "float",
    "float",
    "str",
    "str",
    "int",
    "float",
    "float",
    "float",
    "float",
]
photcolunits = [
    None,
    u.pix,
    u.pix,
    u.adu,
    None,
    u.deg,
    u.deg,
    u.adu,
    u.adu,
    u.adu,
    None,
    None,
    None,
    u.pix,
    u.pix,
    u.pix,
    u.pix,
    u.pix,
    u.s,
    None,
    None,
    u.adu,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    u.electron,
    None,
    u.adu,
]

# Define initial bad table
testphot_data = Table(
    photdata, names=photcolnames, dtype=photcoltypes, units=photcolunits
)

# Convert times to correct time format but leave bad units
testphot_goodTime = testphot_data.copy()
# The way this was originally written used Column(Time, ...) which
# led to a column of generic objects rather that Time objects. This
# prevented Table from being able to write the table to a file because it
# couldn't figure out how to write the column.
testphot_goodTime["date-obs"] = Time(
    testphot_goodTime["date-obs"], format="isot", scale="utc"
)

# Fix all the units for PhotometryData
phot_descript = {
    "star_id": None,
    "ra": u.deg,
    "dec": u.deg,
    "xcenter": u.pix,
    "ycenter": u.pix,
    "fwhm_x": u.pix,
    "fwhm_y": u.pix,
    "width": u.pix,
    "aperture": u.pix,
    "annulus_inner": u.pix,
    "annulus_outer": u.pix,
    "aperture_sum": None,
    "annulus_sum": None,
    "sky_per_pix_avg": None,
    "sky_per_pix_med": None,
    "sky_per_pix_std": None,
    "aperture_net_cnts": None,
    "noise_cnts": None,
    "noise_electrons": u.electron,
    "exposure": u.second,
    "date-obs": None,
    "airmass": None,
    "passband": None,
    "file": None,
}
testphot_goodUnits = testphot_goodTime.copy()
for this_col, this_unit in phot_descript.items():
    testphot_goodUnits[this_col].unit = this_unit

# Fix the units for the counts-related columns
counts_columns = ["aperture_sum", "annulus_sum", "aperture_net_cnts", "noise_cnts"]
counts_per_pixel_sqr_columns = ["sky_per_pix_avg", "sky_per_pix_med", "sky_per_pix_std"]
for this_col in counts_columns:
    testphot_goodUnits[this_col].unit = u.adu
for this_col in counts_per_pixel_sqr_columns:
    testphot_goodUnits[this_col].unit = u.adu * u.pixel**-1

# Remove calculated columns from the test data to produce clean data
computed_columns = ["bjd", "night"]
testphot_clean = testphot_goodUnits.copy()
for this_col in computed_columns:
    del testphot_clean[this_col]


def test_photometry_blank():
    # This should just return a blank PhotometryData
    test_base = PhotometryData()
    assert isinstance(test_base, PhotometryData)
    assert len(test_base) == 0


def test_photometry_data():
    # Create photometry data instance
    phot_data = PhotometryData(
        observatory=feder_obs,
        camera=feder_cg_16m,
        passband_map=feder_passbands,
        input_data=testphot_clean,
    )

    # Check some aspects of that data are sound
    assert phot_data.camera.gain == 1.5 * u.electron / u.adu
    assert phot_data.camera.read_noise == 10.0 * u.electron
    assert phot_data.camera.dark_current == 0.01 * u.electron / u.second
    assert phot_data.camera.pixel_scale == 0.563 * u.arcsec / u.pix
    np.testing.assert_almost_equal(phot_data.observatory.lat.value, 46.86678)
    assert phot_data.observatory.lat.unit == u.deg
    np.testing.assert_almost_equal(phot_data.observatory.lon.value, -96.45328)
    assert phot_data.observatory.lon.unit == u.deg
    assert round(phot_data.observatory.height.value) == 311
    assert phot_data.observatory.height.unit == u.m
    assert phot_data["night"][0] == 59909

    # Checking the BJD computation against Ohio State online calculator for
    # UTC 2022 11 27 06 27 29.620
    # Latitude 46.86678
    # Longitude -96.45328
    # Elevation 311
    # RA 78.17278712191924
    # Dec 22 30 20.77733059
    # which returned 2459910.775405664 (Uses custom IDL, astropy is SOFA checked).
    # Demand a difference of less than 1/20 of a second.
    assert (phot_data["bjd"][0].value - 2459910.775405664) * 86400 < 0.05


def test_photometry_roundtrip_ecsv(tmp_path):
    # Check that we can save the test data to ECSV and restore it
    file_path = tmp_path / "test_photometry.ecsv"
    phot_data = PhotometryData(
        observatory=feder_obs,
        camera=feder_cg_16m,
        passband_map=feder_passbands,
        input_data=testphot_clean,
    )
    phot_data.write(file_path)
    phot_data2 = PhotometryData.read(file_path)
    # Check a couple of the columns that are not standard types
    assert phot_data["date-obs"] == phot_data2["date-obs"]
    assert phot_data["ra"] == phot_data2["ra"]


def test_photometry_slicing():
    # Create photometry data instance
    phot_data = PhotometryData(
        observatory=feder_obs,
        camera=feder_cg_16m,
        passband_map=feder_passbands,
        input_data=testphot_clean,
    )

    # Test slicing works as expected, leaving attributes intact
    two_cols = phot_data[["ra", "dec"]]
    assert two_cols.camera.gain == 1.5 * u.electron / u.adu
    assert two_cols.camera.read_noise == 10.0 * u.electron
    assert two_cols.camera.dark_current == 0.01 * u.electron / u.second
    assert two_cols.camera.pixel_scale == 0.563 * u.arcsec / u.pix
    np.testing.assert_almost_equal(two_cols.observatory.lat.value, 46.86678)
    assert two_cols.observatory.lat.unit == u.deg
    np.testing.assert_almost_equal(two_cols.observatory.lon.value, -96.45328)
    assert two_cols.observatory.lon.unit == u.deg
    assert round(two_cols.observatory.height.value) == 311
    assert two_cols.observatory.height.unit == u.m


def test_photometry_recursive():
    # Create photometry data instance
    phot_data = PhotometryData(
        observatory=feder_obs,
        camera=feder_cg_16m,
        passband_map=feder_passbands,
        input_data=testphot_clean,
    )

    # Attempt recursive call
    with pytest.raises(TypeError):
        phot_data = PhotometryData(
            observatory=feder_obs,
            camera=feder_cg_16m,
            passband_map=feder_passbands,
            input_data=phot_data,
        )


def test_photometry_badtime():
    with pytest.raises(ValueError):
        phot_data = PhotometryData(
            observatory=feder_obs,
            camera=feder_cg_16m,
            passband_map=feder_passbands,
            input_data=testphot_data,
        )


def test_photometry_inconsistent_count_units():
    with pytest.raises(ValueError):
        phot_data = PhotometryData(
            observatory=feder_obs,
            camera=feder_cg_16m,
            passband_map=feder_passbands,
            input_data=testphot_goodTime,
        )


def test_photometry_inconsistent_computed_col_exists():
    with pytest.raises(ValueError):
        phot_data = PhotometryData(
            observatory=feder_obs,
            camera=feder_cg_16m,
            passband_map=feder_passbands,
            input_data=testphot_goodUnits,
        )

    phot_data = PhotometryData(
        observatory=feder_obs,
        camera=feder_cg_16m,
        passband_map=feder_passbands,
        input_data=testphot_goodUnits,
        retain_user_computed=True,
    )
    # This keeps a bad user column for 'snr' which has bogus units, so check the units
    # cause a crash in the math.
    with pytest.raises(u.core.UnitConversionError):
        assert np.abs(phot_data["snr"][0] - 46.795229859903905) < 1e-6
    assert np.abs(phot_data["snr"][0].value - 46.795229859903905) < 1e-6


# Load test catalog
test_cat = ascii.read(
    get_pkg_data_filename("data/test_vsx_table.ecsv"), format="ecsv", fast_reader=False
)


def test_catalog_missing_col():
    # Fails with ValueError due to not having 'ra' column
    with pytest.raises(ValueError):
        catalog_dat = CatalogData(
            input_data=test_cat, catalog_name="VSX", catalog_source="Vizier"
        )


def test_catalog_colname_map():
    # Map column names
    vsx_colname_map = {
        "Name": "id",
        "RAJ2000": "ra",
        "DEJ2000": "dec",
        "max": "mag",
        "n_max": "passband",
    }
    catalog_dat = CatalogData(
        input_data=test_cat,
        catalog_name="VSX",
        catalog_source="Vizier",
        colname_map=vsx_colname_map,
    )

    assert catalog_dat["id"][0] == "ASASSN-V J000052.03+002216.6"
    assert np.abs(catalog_dat["mag"][0].value - 12.660)
    assert catalog_dat["passband"][0] == "g"
    assert catalog_dat.catalog_name == "VSX"
    assert catalog_dat.catalog_source == "Vizier"


def test_catalog_bandpassmap():
    # Map column and bandpass names
    vsx_colname_map = {
        "Name": "id",
        "RAJ2000": "ra",
        "DEJ2000": "dec",
        "max": "mag",
        "n_max": "passband",
    }
    passband_map = {"g": "SG", "r": "SR"}
    catalog_dat = CatalogData(
        input_data=test_cat,
        catalog_name="VSX",
        catalog_source="Vizier",
        colname_map=vsx_colname_map,
        passband_map=passband_map,
    )

    assert catalog_dat["passband"][0] == "SG"
    assert catalog_dat.catalog_name == "VSX"
    assert catalog_dat.catalog_source == "Vizier"


def test_catalog_recursive():
    # Construct good objects
    vsx_colname_map = {
        "Name": "id",
        "RAJ2000": "ra",
        "DEJ2000": "dec",
        "max": "mag",
        "n_max": "passband",
    }
    catalog_dat = CatalogData(
        input_data=test_cat,
        catalog_name="VSX",
        catalog_source="Vizier",
        colname_map=vsx_colname_map,
    )

    # Attempt recursive call
    with pytest.raises(TypeError):
        catalog_dat2 = CatalogData(
            input_data=catalog_dat,
            catalog_name="VSX",
            catalog_source="Vizier",
            colname_map=vsx_colname_map,
        )


def test_tidy_vizier_catalog():
    # Test just the part of the code that converts the table returned by Vizier
    # into a table that can be used by CatalogData.
    apass_input = Table.read(get_pkg_data_filename("data/test_apass_subset.ecsv"))

    result = CatalogData._tidy_vizier_catalog(
        apass_input,
        r"^([a-zA-Z]+|[a-zA-Z]+-[a-zA-Z]+)_?mag$",
        r"^([a-zA-Z]+-[a-zA-Z]+)$",
    )
    assert len(result) == 6

    # Check some column names
    assert "passband" in result.colnames
    assert "mag" in result.colnames
    assert "mag_error" in result.colnames

    # Spot check a couple of values
    one_star = 16572870
    one_Vmag = 13.399
    one_Vmag_error = 0.075

    just_one = result[(result["recno"] == one_star) & (result["passband"] == "V")]
    assert np.abs(just_one["mag"][0] - one_Vmag) < 1e-6
    assert np.abs(just_one["mag_error"][0] - one_Vmag_error) < 1e-6


def test_tidy_vizier_catalog_several_mags():
    # Test table conversion when there are several magnitude columns.
    apass_input = Table.read(get_pkg_data_filename("data/test_apass_subset.ecsv"))

    # Make sure the columns we exxpect in the teset data are there before proceeding
    assert "Vmag" in apass_input.colnames
    assert "i_mag" in apass_input.colnames
    assert "B-V" in apass_input.colnames

    # Add a B magnitude column, and an r-i color. The values are nonsense.
    apass_input["Bmag"] = apass_input["Vmag"]
    apass_input["r-i"] = apass_input["B-V"]

    result = CatalogData._tidy_vizier_catalog(
        apass_input,
        r"^([a-zA-Z]+|[a-zA-Z]+-[a-zA-Z]+)_?mag$",
        r"^([a-zA-Z]+-[a-zA-Z]+)$",
    )

    assert set(result["passband"]) == {"V", "B", "i", "r-i", "B-V"}


@pytest.mark.remote_data
def test_catalog_from_vizier_search_apass():
    # Nothing special about this point...
    sc = SkyCoord(ra=0, dec=0, unit="deg")

    # Small enough radius to get only one star
    radius = 0.03 * u.deg

    apass_colnames = {
        "recno": "id",  # There is no APASS ID, this is the one generated by Vizier
        "RAJ2000": "ra",
        "DEJ2000": "dec",
    }

    apass = CatalogData.from_vizier(
        sc,
        "II/336/apass9",
        radius=radius,
        colname_map=apass_colnames,
        clip_by_frame=False,
    )
    assert len(apass) == 6

    # Got the values below from Vizier on 2023-11-28
    assert apass["id"][0] == 17672748

    just_V = apass[apass["passband"] == "V"]
    assert np.abs(just_V["mag"][0] - 15.559) < 1e-6


@pytest.mark.remote_data
def test_catalog_from_vizier_search_vsx():
    # Do a cone search with a small enough radius to return exaclty one star,
    # DQ Psc, which happens to already be in the test data.
    coordinate = SkyCoord(ra=359.94371 * u.deg, dec=-0.2801 * u.deg)
    vsx_map = dict(
        Name="id",
        RAJ2000="ra",
        DEJ2000="dec",
    )

    # This one is easier -- it already has the passband in a column name.
    # We'll use the maximum magnitude as the magnitude column.
    def prepare_cat(cat):
        cat.rename_column("max", "mag")
        cat.rename_column("n_max", "passband")
        return cat

    my_cat = CatalogData.from_vizier(
        coordinate,
        "B/vsx/vsx",
        radius=0.1 * u.arcmin,
        clip_by_frame=False,
        colname_map=vsx_map,
        prepare_catalog=prepare_cat,
    )

    assert my_cat["id"][0] == "DQ Psc"
    assert my_cat["passband"][0] == "Hp"
    assert my_cat["Type"][0] == "SRB"


def test_from_vizier_with_coord_and_frame_clip_fails():
    # Check that calling from_vizier with a coordinate instead
    # of WCS and with clip_by_frame = True generates an appropriate
    # error.
    data_file = "data/sample_wcs_ey_uma.fits"
    data = get_pkg_data_filename(data_file)
    with fits.open(data) as hdulist:
        with warnings.catch_warnings():
            # Ignore the warning about the WCS having a different number of
            # axes than the (non-existent) image.
            warnings.filterwarnings(
                "ignore",
                message="The WCS transformation has more",
                category=FITSFixedWarning,
            )
            wcs = WCS(hdulist[0].header)
    cen_coord = wcs.pixel_to_world(4096 / 2, 4096 / 2)
    with pytest.raises(ValueError, match="To clip entries by frame"):
        _ = CatalogData.from_vizier(cen_coord, "B/vsx/vsx", clip_by_frame=True)


@pytest.mark.remote_data
@pytest.mark.parametrize(
    "clip, data_file",
    [(True, "data/clipped_ey_uma_vsx.fits"), (False, "data/unclipped_ey_uma_vsx.fits")],
)
def test_vsx_results(clip, data_file):
    # Check that a catalog search of VSX gives us what we expect.
    # I suppose this really isn't future-proof, since more variables
    # could be discovered in the future....
    data = get_pkg_data_filename(data_file)
    expected = Table.read(data)
    wcs_file = get_pkg_data_filename("data/sample_wcs_ey_uma.fits")
    with fits.open(wcs_file) as hdulist:
        with warnings.catch_warnings():
            # Ignore the warning about the WCS having a different number of
            # axes than the (non-existent) image.
            warnings.filterwarnings(
                "ignore",
                message="The WCS transformation has more",
                category=FITSFixedWarning,
            )
            wcs = WCS(hdulist[0].header)
    CCD_SHAPE = [2048, 3073]
    wcs.pixel_shape = list(reversed(CCD_SHAPE))
    ccd = CCDData(data=np.zeros(CCD_SHAPE), wcs=wcs, unit="adu")

    # Turn this into an HDU to get the standard FITS image keywords
    ccd_im = ccd.to_hdu()

    actual = vsx_vizier(ccd_im[0].header, radius=0.5 * u.degree, clip_by_frame=clip)
    assert set(actual["OID"]) == set(expected["OID"])


@pytest.mark.remote_data
def test_find_apass():
    CCD_SHAPE = [2048, 3073]
    # This is really checking from APASS DR9 on Vizier, or at least that
    # is where the "expected" data is drawn from.
    expected_all = Table.read(
        get_pkg_data_filename("data/all_apass_ey_uma_sorted_ra_first_20.fits")
    )

    wcs_file = get_pkg_data_filename("data/sample_wcs_ey_uma.fits")
    with fits.open(wcs_file) as hdulist:
        with warnings.catch_warnings():
            # Ignore the warning about the WCS having a different number of
            # axes than the (non-existent) image.
            warnings.filterwarnings(
                "ignore",
                message="The WCS transformation has more",
                category=FITSFixedWarning,
            )
            wcs = WCS(hdulist[0].header)
    wcs.pixel_shape = list(reversed(CCD_SHAPE))
    ccd = CCDData(data=np.zeros(CCD_SHAPE), wcs=wcs, unit="adu")

    # Turn this into an HDU to get the standard FITS image keywords
    ccd_im = ccd.to_hdu()
    all_apass = apass_dr9(ccd_im[0].header, radius=1 * u.deg)

    # Reference data was sorted by RA, first 20 entries kept
    # There are 6 magnitude or color columns, so 6 * 20 = 120 rows
    # in the resulting table.
    all_apass.sort("ra")
    all_apass = all_apass[:120]

    # It is hard to imagine the RAs matching and other entries not matching,
    # so just check the RAs.
    assert set(ra.value for ra in all_apass["ra"]) == set(expected_all["RAJ2000"])


# Load test apertures
test_sl_data = ascii.read(
    get_pkg_data_filename("data/test_sourcelist.ecsv"), format="ecsv", fast_reader=False
)


def test_sourcelist():
    sl_test = SourceListData(input_data=test_sl_data, colname_map=None)
    assert sl_test["star_id"][0] == 0


def test_sourcelist_no_skypos():
    test_sl_data2 = test_sl_data.copy()
    del test_sl_data2["ra"]
    del test_sl_data2["dec"]
    sl_test = SourceListData(input_data=test_sl_data2, colname_map=None)
    assert sl_test["star_id"][0] == 0
    assert np.isnan(sl_test["ra"][4])
    assert np.isnan(sl_test["dec"][2])


def test_sourcelist_no_imgpos():
    test_sl_data3 = test_sl_data.copy()
    del test_sl_data3["xcenter"]
    del test_sl_data3["ycenter"]
    sl_test = SourceListData(input_data=test_sl_data3, colname_map=None)
    assert sl_test["star_id"][0] == 0
    assert np.isnan(sl_test["xcenter"][4])
    assert np.isnan(sl_test["ycenter"][2])


def test_sourcelist_missing_cols():
    test_sl_data4 = test_sl_data.copy()
    del test_sl_data4["ra"]
    del test_sl_data4["dec"]
    del test_sl_data4["xcenter"]
    del test_sl_data4["ycenter"]
    with pytest.raises(ValueError):
        sl_test = SourceListData(input_data=test_sl_data4, colname_map=None)

    test_sl_data5 = test_sl_data.copy()
    del test_sl_data5["star_id"]
    with pytest.raises(ValueError):
        sl_test = SourceListData(input_data=test_sl_data5, colname_map=None)


def test_sourcelist_recursive():
    # Create good sourcelist data instance
    sl_test = SourceListData(input_data=test_sl_data, colname_map=None)
    assert sl_test["star_id"][0] == 0

    # Attempt recursive call
    with pytest.raises(TypeError):
        sl_test2 = SourceListData(input_data=sl_test, colname_map=None)


def test_sourcelist_dropping_skycoords():
    # Create good sourcelist data instance
    sl_test = SourceListData(input_data=test_sl_data, colname_map=None)

    # Drop sky coordinates
    sl_test.drop_ra_dec()
    assert not sl_test.has_ra_dec
    assert sl_test.has_x_y


def test_sourcelist_dropping_imagecoords():
    # Create good sourcelist data instance
    sl_test = SourceListData(input_data=test_sl_data, colname_map=None)

    # Drop sky coordinates
    sl_test.drop_x_y()
    assert sl_test.has_ra_dec
    assert not sl_test.has_x_y


def test_sourcelist_slicing():
    # Create good sourcelist data instance
    sl_test = SourceListData(input_data=test_sl_data, colname_map=None)

    # Test slicing works as expected
    slicing_test = sl_test[:][1:3]

    # compare this slice to the original data table passed in
    assert slicing_test["star_id"][0] == 1
    assert slicing_test["star_id"][1] == 2
    assert slicing_test["xcenter"][0] == sl_test["xcenter"][1]
    assert slicing_test["xcenter"][1] == sl_test["xcenter"][2]
    # Checking attributes survive slicing
    assert slicing_test.has_ra_dec
    assert slicing_test.has_x_y
