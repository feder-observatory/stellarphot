import re

import astropy.units as u
from astropy.table import Table

import numpy as np

__all__ = ['parse_aij_table']


def parse_aij_table(table_name):
    """
    Return a list of objects, one for each source, from an AstroImageJ
    photometry table.

    Parameters
    ----------

    table_name : str
        Name of the table.
    """

    # Read in the raw table.
    if table_name.endswith('.csv'):
        # The table may have been edited and changed to csv.
        raw = Table.read(table_name)
    else:
        # The default, though, is tab-separated text with a file extension xls.
        raw = Table.read(table_name, format='ascii.tab')

    # Extract the names of all columns which are not specific to a source.
    # Source columns end with _TX or _CX where X is one or more digits.
    source_column = r'.*_[CT]\d+'
    common_columns = []
    for name in raw.colnames:
        if not re.search(source_column, name):
            if name.strip():
                common_columns.append(name)

    # Get all of the source designations from the names of the net counts
    # columns.
    flux_columns = [name for name in raw.colnames if
                    name.startswith('Source-Sky')]
    source_column_ids = [c.split('_')[1] for c in flux_columns]

    # For the first source, grab all of the column names that are specific to
    # a source. These will be the same for all sources.
    first_source_names = [name for name in raw.colnames if
                          name.endswith(source_column_ids[0])]
    generic_source_columns = [name.rsplit('_', 1)[0] for name in
                              first_source_names]

    # Make a list of star objects. Not sure this is actually better than
    # a list of tables or something simple like that.
    stars = []
    for idx, source in enumerate(source_column_ids):
        specifc_column_names = ['_'.join([g, source]) for g in
                                generic_source_columns]
        all_names = common_columns + specifc_column_names
        my_table = raw[all_names]
        for spec, gen in zip(specifc_column_names, generic_source_columns):
            my_table.rename_column(spec, gen)
        stars.append(Star(my_table, idx + 1))

    return stars


class Star(object):
    def __init__(self, table, id_num):
        self._table = table
        self._table['DEC'].unit = u.degree
        self.id = id_num

    @property
    def airmass(self):
        return self._table['AIRMASS']

    @property
    def counts(self):
        return self._table['Source-Sky']

    @property
    def ra(self):
        return self._table['RA'] / 24 * 360 * u.degree

    @property
    def dec(self):
        return self._table['DEC']

    @property
    def error(self):
        return self._table['Source_Error']

    @property
    def sky_per_pixel(self):
        return self._table['Sky/Pixel']

    @property
    def peak(self):
        return self._table['Peak']

    @property
    def jd_utc_start(self):
        return self._table['JD_UTC']

    @property
    def mjd_start(self):
        return self._table['J.D.-2400000'] - 0.5

    @property
    def exposure(self):
        try:
            return self._table['EXPOSURE']
        except KeyError:
            return self._table['EXPTIME']

    @property
    def magnitude(self):
        return -2.5 * np.log10(self.counts) + 2.5 * np.log10(self.exposure)

    @property
    def snr(self):
        return self.counts / self.error

    @property
    def magnitude_error(self):
        return 1.0857 / self.snr

    @property
    def bjd_tdb(self):
        return self._table['BJD_TDB']
