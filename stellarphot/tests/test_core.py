import pytest
import numpy as np
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import EarthLocation

from stellarphot.core import Camera, BaseEnhancedTable, PhotometryData


def test_camera_attributes():
    gain = 2.0
    read_noise = 10
    dark_current = 0.01
    c = Camera(gain=gain, read_noise=read_noise, dark_current=dark_current)
    assert c.gain == gain
    assert c.dark_current == dark_current
    assert c.read_noise == read_noise


def test_base_enhanced_table():
    # Define a realistic table of photometry data
    data = np.array([[1, 78.17278712191920, 22.505771480719400, 31.798216414544900,
                      31.658750534057600, 9.294325523269860, 13.02511260943810,
                      13.02511260943810, 13.02511260943810]])
    colnames = ['id', 'RA', 'Dec', 'sky_per_pix_avg', 'sky_per_pix_med', 'sky_per_pix_std',
                'fwhm_x', 'fwhm_y', 'width']
    coltypes = ['<i8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8']
    colunits = [None,  u.deg,  u.deg,  u.adu,  u.adu,  u.adu,  None,  None,  None]
    testdata = Table(data, names=colnames, dtype=coltypes, units=colunits)

    # Create several test descriptions
    test_descript = np.array([['id', '<i8', None, 'id'],
                                ['RA', '<f8', u.deg, 'ra'],
                                ['Dec', '<f8', u.deg, 'dec'],
                                ['sky_per_pix_avg', '<f8', u.adu, 'spp_avg'],
                                ['sky_per_pix_med', '<f8', u.adu, 'spp_med'],
                                ['sky_per_pix_std', '<f8', u.adu, 'spp_std'],
                                ['fwhm_x', '<f8', None, None],
                                ['fwhm_y', '<f8', None, None],
                                ['width', '<f8', None, 'fwhm']])

    # This should create a blank data set
    test_base = BaseEnhancedTable(test_descript)

    assert isinstance(test_base.data, Table)
    assert len(test_base.ra) == 0
    assert len(test_base.dec) == 0

    # Should create a populated dataset properly and display the astropy data
    test_base2 = BaseEnhancedTable(test_descript, testdata)
    assert len(test_base2.ra) == 1
    assert len(test_base2.dec) == 1

    # Should raise exception because no inputs are passed
    with pytest.raises(Exception):
        test_base = BaseEnhancedTable()

    # this should raise exception because one of the required attributes is missing
    broke_descript = np.copy(test_descript)
    broke_descript[0,3] = 'unique_id'
    with pytest.raises(Exception):
        test_base = BaseEnhancedTable(broke_descript, testdata)

    # Should raise exception because the RA data is missing from input data
    testdata2 = testdata.copy()
    testdata2.remove_column('RA')
    with pytest.raises(Exception):
        test_base = BaseEnhancedTable(test_descript, testdata2)

    # This will fail due to RA being in units of hours
    bad_ra_descript = test_descript.copy()
    bad_ra_descript[1,2] = u.hr

    with pytest.raises(Exception):
        test_base = BaseEnhancedTable(bad_ra_descript)


def test_photometry_data():
    # Define some configuration information assuming Feder telescope
    feder_cg_16m = Camera(gain=1.5, read_noise=10.0, dark_current=0.01)
    feder_filters = {'up':'SU', 'gp':'SG', 'rp':'SR', 'zp':'SZ', 'ip':'SI'}
    feder_obs = EarthLocation(lat = 46.86678,lon=-96.45328, height=311)

    # Define a realistic table of photometry data
    data = np.array([[1, 2049.145245206124, 2054.0849947477964, 109070.60831212997, 154443.9371254444, 78.17278712191924,
            22.505771480719375, 31.798216414544864, 31.658750534057617, 9.294325523269857, 13.02511260943813,
            13.02511260943813, 13.02511260943813, 29.0, 2642.079421669016, 44.0, 59.0, 4853.760649796231, 120.0,
            '2022-11-27T06:26:29.620', 59909, 25057.195077483062, 2459910.7754060575, -6.239606167785804, 1.115,
            'ip', 'TIC_467615239.01-S001-R001-C001-ip.fit', 1, 0.02320185643388203, 803.1970935659333,
            535.4647290439556, 46.795229859903905]])
    colnames = ['id','xcenter','ycenter','aperture_sum','annulus_sum','RA','Dec','sky_per_pix_avg',
            'sky_per_pix_med','sky_per_pix_std','fwhm_x','fwhm_y','width','aperture','aperture_area',
            'annulus_inner','annulus_outer','annulus_area','exposure','date-obs','night','aperture_net_flux',
            'BJD','mag_inst','airmass','filter','file','star_id','mag_error','noise','noise-aij','snr']
    coltypes = ['<i8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8',
            '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<U23', '<i8', '<f8', '<f8', '<f8',
            '<f8', '<U2', '<U38', '<i8', '<f8', '<f8', '<f8', '<f8']
    colunits = [None, u.pix, u.pix, u.adu, None, u.deg, u.deg, u.adu, u.adu, u.adu, None, None, None,
            u.pix, None, u.pix, u.pix, None, u.s, None, None, u.adu, None, None, None, None, None,
            None, u.adu**-1, None, None, u.adu]
    testdata = Table(data, names=colnames, dtype=coltypes, units=colunits)

    # Create photometry data instance
    phot_data = PhotometryData(observatory=feder_obs, camera=feder_cg_16m, filter_map=feder_filters, data=testdata)

    # Check some aspects of that data are sound
    assert phot_data.camera.gain == 1.5
    assert phot_data.camera.read_noise == 10.0
    assert phot_data.camera.dark_current == 0.01
    assert phot_data.observatory.lat.value == 46.86678
    assert phot_data.observatory.lat.unit == u.deg
    assert phot_data.observatory.lon.value == -96.45328
    assert phot_data.observatory.lon.unit == u.deg
    assert round(phot_data.observatory.height.value) == 311
    assert phot_data.observatory.height.unit == u.m
    assert phot_data.filter_map['up'] == 'SU'
    assert type(phot_data.data) == Table
    assert len(phot_data.ra) == 1
    assert len(phot_data.dec) == 1