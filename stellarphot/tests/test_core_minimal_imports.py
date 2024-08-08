from astropy.utils.data import get_pkg_data_filename

from stellarphot.core import PhotometryData


# Why is this in a separate file?
#
# The bug we are trying to reproduce happens only no objects from
# stellarphot.settings.models have been created. In test_core we create several of
# the objects, which adds to the table registry methods for reading our custom objects.
# Here we do a bare minimum of imports to avoid that.
def test_photometry_file_read():
    # Regression test for #408
    file_name = get_pkg_data_filename("data/test_photometry_data.ecsv")
    phot_data = PhotometryData.read(file_name)
    assert phot_data.camera is not None
