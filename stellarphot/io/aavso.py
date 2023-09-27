# Class to represent a file in AAVSO extended format
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from importlib import resources
import yaml

from astropy.table import Table, Column

__all__ = [
    "AAVSOExtendedFileFormatColumns",
    "AAVSOExtendedFileFormat"
]


class AAVSOExtendedFileFormatColumns(Enum):
    VARIABLE_MAG = "magnitude"
    VARIABLE_MAG_ERROR = "magerr"
    VARIABLE_STAR_ID_COLUMN = "starid"
    AIRMASS = "airmass"
    JD = "date"
    PASSBAND = "filter"
    CHECK_STAR_MAG = "kmag"
    CHECK_STAR_NAME = "kname"
    COMP_STAR_MAG = "cmag"
    COMP_STAR_NAME = "cname"


@dataclass
class AAVSOExtendedFileFormat:
    """
    A class to represent a file in AAVSO extended format.

    Parameters
    ----------
    obscode : str
        The observer's code.

    Attributes
    ----------
    obscode : str
        The observer's code.

    type : str
        The type of observation. Defaults to 'EXTENDED'.

    software : str
        The software used to generate the file. Defaults to 'stellarphot'.

    delim : str
        The delimiter used in the file. Defaults to ','.

    date_format : str
        The format of the date. Defaults to 'JD'.

    date : `astropy.table.Column`
        The date/time of the observation in the format `date_format`.

    obstype : str
        The type of observation. Defaults to 'CCD'.

    magnitude : `astropy.table.Column`
        The magnitude of the observation.

    magerr : `astropy.table.Column`
        The error in the magnitude of the observation.

    filter : `astropy.table.Column`
        The filter/band used for the observation. Must be one of the AAVSO passbands,
        which are listed at https://www.aavso.org/filters.

    kmag : `astropy.table.Column`
        The magnitude of the check star.

    cmag : `astropy.table.Column`
        The magnitude of the comparison star.

    ensemble : bool
        Whether the observation is an ensemble observation. Defaults to False.

    group : int
        The group number of the ensemble observation. Defaults to 0.

    trans : str
        Whether the observation is a transformed observation. Defaults to 'NO'.

    chart : str
        The chart used for the observation. Defaults to ''.

    notes : str
        Notes about the observation. Defaults to ''.

    starid : str
        The AUID of the star. Defaults to ''.

    cname : str
        The AUID of the comparison star. Defaults to ''.

    kname : str
        The AUID of the check star. Defaults to ''.

    airmass : `astropy.table.Column`
        The airmass of the observation.

    mtype : str
        The type of magnitude. Defaults to 'STD'.
    """
    obscode : str
    _type: str = 'EXTENDED'
    _software: str = 'stellarphot'
    delim: str = ","
    date_format: str = 'JD'
    date: Column = field(default_factory=Column)
    obstype: str = 'CCD'
    magnitude: Column = field(default_factory=Column)
    magerr: Column = field(default_factory=Column)
    filter: Column = field(default_factory=Column)
    kmag: Column = field(default_factory=Column)
    cmag: Column = field(default_factory=Column)
    group : int  = ""
    trans : str = "NO"
    chart : str = ""
    notes : str = ""
    starid : str = ""
    cname : str = ""
    kname : str = ""
    airmass : Column = field(default_factory=Column)
    mtype : str = "STD"

    def __post_init__(self):
        self._ensemble = False

    @property
    def ensemble(self):
        return self._ensemble

    @ensemble.setter
    def ensemble(self, value):
        if value:
            self.cmag = Column(["na"] * len(self.magnitude), name="cmag")
            self.cname = Column(["ENSEMBLE"] * len(self.magnitude), name="cname")

        self._ensemble = value

    @property
    def type(self):
        return self._type

    @property
    def software(self):
        return self._software

    def set_data_columns(self, data, column_map, star_id=None):
        """
        Set the data columns from an astropy table.

        Parameters
        ----------

        data : `astropy.table.Table`
            Table containing the data to be written to the file.
        column_map : dict
            Dictionary mapping the column names in the table to the column names
            in the file.The values in the dictionary should be column names from
            `AAVSOExtendedFileFormatColumns`.
        star_id : str, optional
            ID of the star to be written to the file. If not provided, all stars
            in the table will be written to the file.
        """
        if star_id is not None:
            use_data = data[data['id'] == star_id]
        else:
            use_data = data
        for source_column, destination_column in column_map.items():
            if (self.ensemble and
                (destination_column == AAVSOExtendedFileFormatColumns.COMP_STAR_MAG or
                 destination_column == AAVSOExtendedFileFormatColumns.COMP_STAR_NAME)):
                raise ValueError("Cannot set comparison star magnitude or name for ensemble observation")

            try:
                setattr(self, destination_column.value, use_data[source_column])
            except AttributeError:
                raise AttributeError(f"Column {destination_column.value} not allowed in AAVSO extended format")
            except KeyError:
                raise KeyError(f"Column {source_column} not found in data")

    def set_column_from_single_value(self, value, column):
        setattr(self, column, Column([value] * len(self.variable_mag), name=column))

    def to_table(self):
        """
        Convert the file to an astropy table.

        Returns
        -------
        table : `astropy.table.Table`
            The table representation of the file. Note that there is important information
            in the table's meta attribute.
        """
        # Make a table
        table_description = resources.read_text('stellarphot.io', 'aavso_submission_schema.yml')
        table_structure = yaml.safe_load(table_description)

        table_dict = {}
        length = 0

        # Begin setting up table structure, with column names as keys
        # and the column data as values. If the "column" is really
        # a single item, then make a single entry as the value.
        # It will be turned into a column eventually.
        for key in table_structure['data'].keys():
            # woot! my first walrus operator!
            if isinstance((item := getattr(self, key.lower())), Column):
                # We have a column, but it might not have been set.
                if len(item) == 0:
                    item = "na"
                table_dict[key] = item
                # Keep track of the longest column
                length = len(item) if len(item) > length else length
            else:
                table_dict[key] = item if len(item) > 0 else "na"

        # Fix up anything that is supposed to be a column and isn't yet.
        for k, v in table_dict.items():
            if not isinstance(v, Column) or len(v) != length:
                # If the length is not correct assume we have a single value
                # and make a column out of it.
                table_dict[k] = Column([v] * length, name=k)

        table = Table(table_dict)
        # All of the AAVSO "comment" type entries go in meta.
        table.meta['comments'] = []
        for key in table_structure['comments'].keys():
            if key == "DATE":
                value = self.date_format
            else:
                value = getattr(self, key.lower())
            table.meta['comments'].append(f"{key}={value}")

        # Specify formats for the columns that have formats
        for k, v in table_structure['data'].items():
            length_limit = v.get('limit', None)
            if length_limit:
                if v['type'] == 'float':
                    table[k].info.format = f"{length_limit}f"
                elif v['type'] == 'str':
                    if k == 'STARID':
                        print(table[k])
                    table[k].info.format = f".{length_limit}s"
                else:
                    raise ValueError(f"Unknown type {v['type']}")
        return table

    def write(self, filename, overwrite=False):
        """
        Write the file to disk.

        Parameters
        ----------
        filename : str, pathlib.Path
            The file to write to.
        """
        table = self.to_table()
        p = Path(filename)
        table.write(p,
                    delimiter=self.delim,
                    comment="#",
                    overwrite=overwrite)
