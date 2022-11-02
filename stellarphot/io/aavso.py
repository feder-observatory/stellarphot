# Class to representa file in AAVSO extended format
from dataclasses import dataclass

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
    data: Table = Table()

    @property
    def type(self):
        return self._type