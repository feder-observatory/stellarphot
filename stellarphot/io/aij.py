from pathlib import Path
import re

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack

import numpy as np

__all__ = ['parse_aij_table', 'ApertureFileAIJ']


class ApertureAIJ:
    """
    Represent the aperture information AstroImageJ saves.
    """
    def __init__(self):
        # Outer annulus radius
        self.rback2 = 41.0

        # Inner annulus radius
        self.rback1 = 27.0

        # Aperture radius
        self.radius = 15.0

        # Defaults for this match the defaults for stellarphot.
        self.removebackstars = True

        # Not sure what this does but stellarphot doesn't use it.
        self.backplane = False


class MultiApertureAIJ:
    """
    Represent the multi-aperture information that AstroImageJ saves
    """
    def __init__(self):
        # Default values for these chosen to match AIJ defaults
        # They are not used by stellarphot
        self.naperturesmax = 1000
        self.apfwhmfactor = 1.4
        self.usevarsizeap = False

        # Each attribute below should be list-like, all of the same length
        self.isrefstar = []
        self.centroidstar = []
        self.isalignstar = []

        self.xapertures = []
        self.yapertures = []

        self.absmagapertures = []

        self.raapertures = []
        self.decapertures = []


class ApertureFileAIJ:
    """
    Represent AstroImageJ aperture file.
    """
    def __init__(self):
        self.aperture = ApertureAIJ()
        self.multiaperture = MultiApertureAIJ()

    def __str__(self):
        lines = []

        top_level_attrib = vars(self)
        for name, attrib in top_level_attrib.items():
            base_attrib = vars(attrib)
            for bname, battrib in base_attrib.items():
                try:
                    value = ','.join([str(v).lower() for v in battrib])
                except TypeError:
                    value = battrib

                if value is True:
                    value = 'true'
                elif value is False:
                    value = 'false'

                lines.append(f'.{name}.{bname}={value}')

        # Add a trailing blank line
        return '\n'.join(lines) + '\n'

    def write(self, file):
        p = Path(file)
        p.write_text(str(self))

    @classmethod
    def from_table(cls, aperture_table,
                   aperture_rad=None, inner_annulus=None, outer_annulus=None,
                   default_absmag=99.999, default_isalign=True,
                   default_centroidstar=True):
        """
        Create an `stellarphot.io.ApertureFileAIJ` from a stellarphot aperture
        table and info about the aperture sizes.

        Parameters
        ----------

        aperture_table : `astropy.table.Table`
            Table of aperture information.

        aperture_rad : number
            Radius of aperture.

        inner_annulus : number
            Inner radius of annulus.

        outer_annulus : number
            Outer radius of annulus.
        """
        # Create the instance
        apAIJ = cls()

        # Populate aperture properties
        apAIJ.aperture.rback2 = outer_annulus
        apAIJ.aperture.rback1 = inner_annulus
        apAIJ.aperture.radius = aperture_rad

        # Remaining properties of apAIJ.aperture default to the
        # correct values for stellarphot

        # Populate multiaperture properties
        n_apertures = len(aperture_table)
        columns = aperture_table.colnames

        # A boolean column for this would be better, but this will do
        # for now.
        apAIJ.multiaperture.isrefstar = [('comparison' in name.lower())
                           for name in aperture_table['marker name']]

        # These are not currently in the table but that could change...
        if 'centroidstar' not in columns:
            apAIJ.multiaperture.centroidstar = [default_centroidstar] * n_apertures
        else:
            apAIJ.multiaperture.centroidstar = aperture_table['centroidstar']

        if 'isalign' not in columns:
            apAIJ.multiaperture.isalignstar = [default_isalign] * n_apertures
        else:
            apAIJ.multiaperture.isalignstar = aperture_table['isalign']

        apAIJ.multiaperture.xapertures = aperture_table['x']
        apAIJ.multiaperture.yapertures = aperture_table['y']

        if 'absmag' not in columns:
            apAIJ.multiaperture.absmagapertures = [default_absmag] * n_apertures
        else:
            apAIJ.multiaperture.absmagapertures = aperture_table['absmag']

        apAIJ.multiaperture.raapertures = aperture_table['coord'].ra.degree
        apAIJ.multiaperture.decapertures = aperture_table['coord'].dec.degree

        return apAIJ


def _is_comp(star_coord, comp_table):
    idx, d2d, _ = star_coord.match_to_catalog_sky(comp_table['coord'])
    return 'comparison' in comp_table['marker name'][idx]


def generate_aij_table(table_name, comparison_table):
    info_columns = {
        'date-obs': 'DATE_OBS',
        'airmass': 'AIRMASS',
        'BJD': 'BJD_MOBS',
        'exposure': 'EXPOSURE',
        'filter': 'FILTER',
        'aperture': 'Source_Radius',
        'annulus_inner': 'Sky_Rad(min)',
        'annulus_outer': 'Sky_Rad(max)'
    }
    by_source_columns = {
        'xcenter': 'X(IJ)',
        'ycenter': 'Y(IJ)',
        'aperture_net_flux': 'Source-Sky',
        'aperture_area': 'N_Src_Pixels',
        'noise-aij': 'Source_Error',
        'snr': 'Source_SNR',
        'sky_per_pix_avg': 'Sky/Pixel',
        'annulus_area': 'N_Sky_Pixels',
        'fwhm_x': 'X-Width',
        'fwhm_y': 'Y-Width',
        'width': 'Width',
        'relative_flux': 'rel_flux',
        'relative_flux_error': 'rel_flux_err',
        'relative_flux_snr': 'rel_flux_SNR',
        'comparison counts': 'tot_C_cnts',
        'comparison error': 'tot_C_err'
    }
    by_star = table_name.group_by('id')
    base_table = by_star.groups[0][list(info_columns.keys())]
    for star_id, sub_table in zip(by_star.groups.keys, by_star.groups):
        star_co = SkyCoord(ra=sub_table['RA'][0], dec=sub_table['Dec'][0],
                           unit='degree')

        if _is_comp(star_co, comparison_table):
            char = 'C'
        else:
            char = 'T'

        new_table = sub_table[list(by_source_columns.keys())] #  + ['BJD']

        for old_col, new_col in by_source_columns.items():
            new_column_name = new_col + f'_{char}{star_id[0]}'
            new_table.rename_column(old_col, new_column_name)
        base_table = hstack([base_table, new_table])
    for old_col, new_col in info_columns.items():
        base_table.rename_column(old_col, new_col)

    return base_table

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
