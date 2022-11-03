# Class to representa file in AAVSO extended format
from dataclasses import dataclass, field
from typing import Union
from pathlib import Path
from importlib import resources
import yaml

from astropy.table import Table

__all__ = [
    "AAVSOExtendedFileFormat"
]


@dataclass
class AAVSOExtendedFileFormat:
    observer_code: str
    _type: str = 'EXTENDED'
    _software: str = 'stellarphot'
    delim: str = ","
    date: str = 'HJD'
    obstype: str = 'CCD'
    variable_data: Table = field(default_factory=Table)
    check_data: Table = field(default_factory=Table)
    comparison_data: Union[Table, None] = None
    ensemble: bool = False

    @property
    def type(self):
        return self._type

    @property
    def software(self):
        return self._software

    def set_data(self, destination, data,
                 time='time', mag='mag', error='err', airmass='airmass'):
        """
        Set data for each thingy
        """
        # Standardize column order

        standard_data = {
            time: data[time],
            airmass: data[airmass],
            mag: data[mag],
            error: data[error],
        }
        standard_data = Table(standard_data)
        if destination == "variable":
            self.variable_data = standard_data
        elif destination == "check":
            self.check_data = standard_data
        elif destination == "comparison":
            if self.ensemble:
                raise ValueError("Cannot set comparison data for ensemble magnitudeas")
            self.comparison_data = standard_data

    def write(self, file):
        p = Path(file)

        # Make a table
        table_description = resources.read_text('stellarphot.io', 'aavso_submission_schema.yml')
        table_structure = yaml.safe_load(table_description)
        print(table_structure['data'].keys())