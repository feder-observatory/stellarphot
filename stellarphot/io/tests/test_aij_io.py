from pathlib import Path

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename

from stellarphot.io import ApertureFileAIJ
from stellarphot.io.aij import ApertureAIJ


def test_aperture_eq():
    ap1 = ApertureAIJ()
    ap2 = ApertureAIJ()
    ap3 = ApertureAIJ()
    ap3.rback1 = 27.0
    assert ap1 == ap2
    assert ap2 == ap3


def test_aperture_file_content():
    # Test that the format of the generated aperture file
    # matches the expected format
    ap = ApertureFileAIJ()
    ref_data = get_pkg_data_filename('data/apertures_as_table.csv')
    ref_table = Table.read(ref_data)

    ap.multiaperture.xapertures = ref_table['x']

    # AIJ has origin in different place than the reference table.
    ap.multiaperture.yapertures = np.around((4096 - ref_table['y']),
                                            decimals=4)

    ap.multiaperture.raapertures = ref_table['ra']
    ap.multiaperture.decapertures = ref_table['dec']

    ap.multiaperture.isrefstar = ref_table['isrefstar']
    ap.multiaperture.centroidstar = ref_table['centroidstar']
    ap.multiaperture.isalignstar = ref_table['isalignstar']

    ap.multiaperture.absmagapertures = ref_table['absmag']

    ref_aperture_file = \
        get_pkg_data_filename('data/aij-sample-apertures.aperture')

    ref_apertures = ApertureFileAIJ.read(ref_aperture_file)

    assert ref_apertures == ap


def test_aperture_creation_from_table():
    # Check that generating an aperture object from an
    # aperture table gives the right result.

    ref_data = get_pkg_data_filename('data/apertures_as_table.csv')
    ref_table = Table.read(ref_data)

    # Need to create a coord column
    coordinates = SkyCoord(ra=ref_table['ra'], dec=ref_table['dec'],
                           unit='degree')
    ref_table['coord'] = coordinates

    # Delete some columns that are not in the usual aperture table
    del ref_table['ra'], ref_table['dec'], \
        ref_table['isalignstar'], ref_table['centroidstar']

    # Generate marker names that match those in aperture file
    # Note that the csv reads in the True/False column as text,
    # not as bool.
    ref_table['marker name'] = ['APASS comparison'
                                if v == 'True' else "TESS target"
                                for v in ref_table['isrefstar']]

    del ref_table['isrefstar']

    ap_info = ApertureAIJ()
    ap_aij = ApertureFileAIJ.from_table(ref_table, aperture_rad=ap_info.radius,
                                        inner_annulus=ap_info.rback1,
                                        outer_annulus=ap_info.rback2,
                                        )
    ref_aperture_file = \
        get_pkg_data_filename('data/aij-sample-apertures.aperture')

    ref_apertures = ApertureFileAIJ.read(ref_aperture_file)

    assert ref_apertures == ap_aij
