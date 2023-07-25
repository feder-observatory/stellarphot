import pytest
import numpy as np
from astropy import units as u
from astropy.table import Table, Column
from astropy.time import Time
from astropy.io import ascii
from astropy.coordinates import EarthLocation
from astropy.utils.data import get_pkg_data_filename

from stellarphot.core import (Camera, BaseEnhancedTable, PhotometryData,
                              CatalogData, SourceListData)


def test_camera_attributes():
    gain = 2.0 * u.electron / u.adu
    read_noise = 10 * u.electron
    dark_current = 0.01 * u.electron / u.second
    c = Camera(gain=gain, read_noise=read_noise, dark_current=dark_current)
    assert c.gain == gain
    assert c.dark_current == dark_current
    assert c.read_noise == read_noise


def test_camera_unitscheck():
    gain = 2.0
    read_noise = 10
    dark_current = 0.01
    with pytest.raises(TypeError):
        c = Camera(gain=gain, read_noise=read_noise, dark_current=dark_current)


# Create several test descriptions for use in base_enhanced_table tests.
test_descript = {'id': None,
                 'ra': u.deg,
                 'dec' : u.deg,
                 'sky_per_pix_avg' : u.adu,
                 'sky_per_pix_med' : u.adu,
                 'sky_per_pix_std' : u.adu,
                 'fwhm_x' : u.pix,
                 'fwhm_y' : u.pix,
                 'width' : u.pix}

# Define a realistic table of astronomical data contianing one row
data = np.array([[1, 78.17278712191920, 22.505771480719400, 31.798216414544900,
                    31.658750534057600, 9.294325523269860, 13.02511260943810,
                    13.02511260943810, 13.02511260943810]])
colnames = ['id', 'ra', 'dec', 'sky_per_pix_avg', 'sky_per_pix_med', 'sky_per_pix_std',
            'fwhm_x', 'fwhm_y', 'width']
coltypes = ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
            'float']
colunits = [None,  u.deg,  u.deg,  u.adu,  u.adu,  u.adu,  u.pix,  u.pix,  u.pix]
testdata = Table(data, names=colnames, dtype=coltypes, units=colunits)

# Define some configuration information assuming Feder telescope
feder_cg_16m = Camera(gain = 1.5 * u.electron / u.adu,
                      read_noise = 10.0 * u.electron,
                      dark_current=0.01 * u.electron / u.second)
feder_passbands = {'up':'SU', 'gp':'SG', 'rp':'SR', 'zp':'SZ', 'ip':'SI'}
feder_obs = EarthLocation(lat = 46.86678,lon=-96.45328, height=311)

# Define a realistic table of photometry data (a bit corrupted)
photdata = np.array([[1, 2049.145245206124, 2054.0849947477964, 109070.60831212997,
                      154443.9371254444, 78.17278712191924, 22.505771480719375,
                      31.798216414544864, 31.658750534057617, 9.294325523269857,
                      13.02511260943813, 13.02511260943813, 13.02511260943813, 29.0,
                      2642.079421669016, 44.0, 59.0, 4853.760649796231, 120.0,
                      '2022-11-27T06:26:29.620', 59909, 25057.195077483062,
                      2459910.7754060575, -6.239606167785804, 1.115, 'ip',
                      'TIC_467615239.01-S001-R001-C001-ip.fit', 1, 0.02320185643388203,
                      803.1970935659333, 535.4647290439556, 46.795229859903905]])
photcolnames = ['id', 'xcenter', 'ycenter', 'aperture_sum', 'annulus_sum', 'ra', 'dec',
                'sky_per_pix_avg', 'sky_per_pix_med', 'sky_per_pix_std', 'fwhm_x',
                'fwhm_y', 'width', 'aperture', 'aperture_area', 'annulus_inner',
                'annulus_outer', 'annulus_area', 'exposure', 'date-obs', 'night',
                'aperture_net_cnts', 'bjd', 'mag_inst', 'airmass', 'passband', 'file',
                'star_id', 'mag_error', 'noise_electrons', 'noise_cnts', 'snr']
photcoltypes = ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                'float', 'float', 'float', 'str', 'int', 'float', 'float', 'float',
                'float', 'str', 'str', 'int', 'float', 'float', 'float', 'float']
photcolunits = [None, u.pix, u.pix, u.adu, None, u.deg, u.deg, u.adu, u.adu, u.adu,
                None, None, None, u.pix, u.pix*u.pix, u.pix, u.pix, u.pix*u.pix, u.s,
                None, None, u.adu, None, None, None, None, None, None, None,
                u.electron, None, u.adu]

# Define initial bad table
testphot_data = Table(photdata, names=photcolnames, dtype=photcoltypes,
                      units=photcolunits)

# Convert times to correct time format but leave bad units
testphot_goodTime = testphot_data.copy()
testphot_goodTime['date-obs'] = Column(data=Time(testphot_goodTime['date-obs'],
                                                 format='isot', scale='utc'),
                                       name='date-obs')

# Fix the units for the counts-related columns
counts_columns = ['aperture_sum', 'annulus_sum', 'sky_per_pix_avg', 'sky_per_pix_med',
                          'sky_per_pix_std', 'aperture_net_cnts', 'noise_cnts']
testphot_goodCounts = testphot_goodTime.copy()
for this_col in counts_columns:
    testphot_goodCounts[this_col].unit = u.adu

# Fix all the units for PhotometryData
phot_descript = {
    'star_id' : None,
    'ra' : u.deg,
    'dec' : u.deg,
    'xcenter' : u.pix,
    'ycenter' : u.pix,
    'fwhm_x' : u.pix,
    'fwhm_y' : u.pix,
    'width' : u.pix,
    'aperture' : u.pix,
    'annulus_inner' : u.pix,
    'annulus_outer' : u.pix,
    'aperture_sum' : None,
    'annulus_sum' : None,
    'sky_per_pix_avg' : None,
    'sky_per_pix_med' : None,
    'sky_per_pix_std' : None,
    'aperture_net_cnts' : None,
    'noise_cnts' : None,
    'noise_electrons' : u.electron,
    'exposure' : u.second,
    'date-obs' : None,
    'airmass' : None,
    'passband' : None,
    'file' : None
}
testphot_goodUnits = testphot_goodTime.copy()
for this_col, this_unit in phot_descript.items():
    testphot_goodUnits[this_col].unit = this_unit

# Remove calculated columns from the test data to produce clean data
computed_columns = ['bjd', 'night']
testphot_clean = testphot_goodUnits.copy()
for this_col in computed_columns:
    del testphot_clean[this_col]

# Load test catalog
test_cat = ascii.read(get_pkg_data_filename('data/test_vsx_table.ecsv'), format='ecsv',
                      fast_reader=False)

# Load test apertures
test_sl_data = ascii.read(get_pkg_data_filename('data/test_sourcelist.ecsv'),
                             format='ecsv',
                             fast_reader=False)


def test_base_enhanced_table_blank():
    # This should just return a blank BaseEnhancedTable
    test_base = BaseEnhancedTable()
    assert type(test_base) == BaseEnhancedTable
    assert len(test_base) == 0


def test_base_enhanced_table_from_existing_table():
    # Should create a populated dataset properly and display the astropy data
    test_base2 = BaseEnhancedTable(table_description=test_descript, input_data=testdata)
    assert len(test_base2['ra']) == 1
    assert len(test_base2['dec']) == 1


def test_base_enhanced_table_missing_column():
    # Should raise exception because the RA data is missing from input data
    testdata_nora = testdata.copy()
    testdata_nora.remove_column('ra')
    with pytest.raises(ValueError):
        test_base = BaseEnhancedTable(table_description=test_descript,
                                      input_data=testdata_nora)


def test_base_enhanced_table_missing_badunits():
    # This will fail due to RA being in units of hours
    bad_ra_descript = test_descript.copy()
    bad_ra_descript[1,2] = u.hr

    with pytest.raises(ValueError):
        test_base = BaseEnhancedTable(table_description=bad_ra_descript,
                                      input_data=testdata)


def test_base_enhanced_table_recursive():
    # Should create a populated dataset properly and display the astropy data
    test_base2 = BaseEnhancedTable(table_description=test_descript, input_data=testdata)
    assert len(test_base2['ra']) == 1
    assert len(test_base2['dec']) == 1

    # Attempt recursive call
    with pytest.raises(TypeError):
        test_base3 = BaseEnhancedTable(table_description=test_descript,
                                       input_data=test_base2)


def test_photometry_blank():
    # This should just return a blank PhotometryData
    test_base = PhotometryData()
    assert type(test_base) == PhotometryData
    assert len(test_base) == 0


def test_photometry_data():
    # Create photometry data instance
    phot_data = PhotometryData(observatory=feder_obs, camera=feder_cg_16m,
                               passband_map=feder_passbands, input_data=testphot_clean)

    # Check some aspects of that data are sound
    assert phot_data.camera.gain == 1.5 *  u.electron / u.adu
    assert phot_data.camera.read_noise == 10.0 * u.electron
    assert phot_data.camera.dark_current == 0.01 * u.electron / u.second
    assert phot_data.observatory.lat.value == 46.86678
    assert phot_data.observatory.lat.unit == u.deg
    assert phot_data.observatory.lon.value == -96.45328
    assert phot_data.observatory.lon.unit == u.deg
    assert round(phot_data.observatory.height.value) == 311
    assert phot_data.observatory.height.unit == u.m
    assert phot_data['night'][0] == 59909

    # Checking the BJD computation against Ohio State online calculator for
    # UTC 2022 11 27 06 27 29.620
    # Latitude 46.86678
    # Longitude -96.45328
    # Elevation 311
    # RA 78.17278712191924
    # Dec 22 30 20.77733059
    # which returned 2459910.775405664 (Uses custom IDL, astropy is SOFA checked).
    # Demand a difference of less than 1/20 of a second.
    assert (phot_data['bjd'][0].value - 2459910.775405664)*86400 < 0.05


def test_photometry_recursive():
    # Create photometry data instance
    phot_data = PhotometryData(observatory=feder_obs, camera=feder_cg_16m,
                               passband_map=feder_passbands, input_data=testphot_clean)

    # Attempt recursive call
    with pytest.raises(TypeError):
        phot_data = PhotometryData(observatory=feder_obs, camera=feder_cg_16m,
                               passband_map=feder_passbands, input_data=phot_data)


def test_photometry_badtime():
    with pytest.raises(ValueError):
        phot_data = PhotometryData(observatory=feder_obs, camera=feder_cg_16m,
                                   passband_map=feder_passbands,
                                   input_data=testphot_data)


def test_photometry_inconsistent_count_units():
    with pytest.raises(ValueError):
        phot_data = PhotometryData(observatory=feder_obs, camera=feder_cg_16m,
                                   passband_map=feder_passbands,
                                   input_data=testphot_goodTime)


def test_photometry_inconsistent_badunits():
    with pytest.raises(ValueError):
        phot_data = PhotometryData(observatory=feder_obs, camera=feder_cg_16m,
                                   passband_map=feder_passbands,
                                   input_data=testphot_goodCounts)


def test_photometry_inconsistent_computed_col_exists():
    with pytest.raises(ValueError):
        phot_data = PhotometryData(observatory=feder_obs, camera=feder_cg_16m,
                                   passband_map=feder_passbands,
                                   input_data=testphot_goodUnits)

    phot_data = PhotometryData(observatory=feder_obs, camera=feder_cg_16m,
                               passband_map=feder_passbands,
                               input_data=testphot_goodUnits,
                               retain_user_computed=True)
    # This keeps a bad user column for 'snr' which has bogus units, so check the units
    # cause a crash in the math.
    with pytest.raises(u.core.UnitConversionError):
        assert np.abs(phot_data['snr'][0] - 46.795229859903905) < 1e-6
    assert np.abs(phot_data['snr'][0].value - 46.795229859903905) < 1e-6


def test_catalog_missing_col():
    # Fails with ValueError due to not having 'ra' column
    with pytest.raises(ValueError):
        catalog_dat = CatalogData(input_data=test_cat, catalog_name="VSX",
                                  catalog_source="Vizier")


def test_catalog_colname_map():
    # Map column names
    vsx_colname_map = {'Name':'id', 'RAJ2000':'ra', 'DEJ2000':'dec', 'max':'mag',
                       'n_max':'passband'}
    catalog_dat = CatalogData(input_data=test_cat, catalog_name="VSX",
                              catalog_source="Vizier",
                              colname_map=vsx_colname_map)

    assert catalog_dat['id'][0] == 'ASASSN-V J000052.03+002216.6'
    assert np.abs(catalog_dat['mag'][0].value - 12.660)
    assert catalog_dat['passband'][0] == 'g'
    assert catalog_dat.catalog_name == 'VSX'
    assert catalog_dat.catalog_source == 'Vizier'


def test_catalog_bandpassmap():
    # Map column and bandpass names
    vsx_colname_map = {'Name':'id', 'RAJ2000':'ra', 'DEJ2000':'dec', 'max':'mag',
                       'n_max':'passband'}
    passband_map = {'g' :'SG', 'r':'SR'}
    catalog_dat = CatalogData(input_data=test_cat, catalog_name="VSX",
                              catalog_source="Vizier", colname_map=vsx_colname_map,
                              passband_map=passband_map)

    assert catalog_dat['passband'][0] == 'SG'
    assert catalog_dat.catalog_name == 'VSX'
    assert catalog_dat.catalog_source == 'Vizier'


def test_catalog_recursive():
    # Construct good objects
    vsx_colname_map = {'Name':'id', 'RAJ2000':'ra', 'DEJ2000':'dec', 'max':'mag',
                       'n_max':'passband'}
    catalog_dat = CatalogData(input_data=test_cat, catalog_name="VSX",
                              catalog_source="Vizier", colname_map=vsx_colname_map)

    # Attempt recursive call
    with pytest.raises(TypeError):
        catalog_dat2 = CatalogData(input_data=catalog_dat, catalog_name="VSX",
                                   catalog_source="Vizier", colname_map=vsx_colname_map)


def test_sourcelist():
    sl_test = SourceListData(input_data=test_sl_data, colname_map=None)
    assert sl_test['star_id'][0] == 0


def test_sourcelist_no_skypos():
    test_sl_data2 = test_sl_data.copy()
    del test_sl_data2['ra']
    del test_sl_data2['dec']
    sl_test = SourceListData(input_data=test_sl_data2, colname_map=None)
    assert sl_test['star_id'][0] == 0
    assert np.isnan(sl_test['ra'][4])
    assert np.isnan(sl_test['dec'][2])


def test_sourcelist_no_imgpos():
    test_sl_data3 = test_sl_data.copy()
    del test_sl_data3['xcenter']
    del test_sl_data3['ycenter']
    sl_test = SourceListData(input_data=test_sl_data3, colname_map=None)
    assert sl_test['star_id'][0] == 0
    assert np.isnan(sl_test['xcenter'][4])
    assert np.isnan(sl_test['ycenter'][2])


def test_sourcelist_missing_cols():
    test_sl_data4 = test_sl_data.copy()
    del test_sl_data4['ra']
    del test_sl_data4['dec']
    del test_sl_data4['xcenter']
    del test_sl_data4['ycenter']
    with pytest.raises(ValueError):
        sl_test = SourceListData(input_data=test_sl_data4, colname_map=None)

    test_sl_data5 = test_sl_data.copy()
    del test_sl_data5['star_id']
    with pytest.raises(ValueError):
        sl_test = SourceListData(input_data=test_sl_data5, colname_map=None)


def test_sourcelist_recursive():
    # Create good sourcelist data instance
    sl_test = SourceListData(input_data=test_sl_data, colname_map=None)
    assert sl_test['star_id'][0] == 0

    # Attempt recursive call
    with pytest.raises(TypeError):
        sl_test2 = SourceListData(input_data=sl_test, colname_map=None)


def test_sourcelist_slicing():
    # Create good sourcelist data instance
    sl_test = SourceListData(input_data=test_sl_data, colname_map=None)

    # Test slicing works as expected
    slicing_test = sl_test[:][1:3]

    # compare this slice to the original data table passed in
    assert slicing_test['star_id'][0] == 1
    assert slicing_test['star_id'][1] == 2
    assert slicing_test['xcenter'][0] == sl_test['xcenter'][1]
    assert slicing_test['xcenter'][1] == sl_test['xcenter'][2]