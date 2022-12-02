from dataclasses import dataclass, field
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy import units as u
from astropy.utils.data import download_file

from astroquery.mast import Catalogs

import requests

from stellarphot.analysis.exotic import get_tic_info

__all__ = ["TessSubmission", "TOI", "TessTargetFile"]

# Makes me want to vomit, but....
DEFAULT_TABLE_LOCATION = "who.the.fuck.knows"
TOI_TABLE_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?output=csv"
TIC_regex = re.compile(r"[tT][iI][cC][^\d]?(?P<star>\d+)(?P<planet>\.\d\d)?")


@dataclass
class TessSubmission:
    telescope_code: str
    filter: str
    utc_start: int
    tic_id: int
    planet_number: int

    def __post_init__(self, *args, **kwargs):
        self._tic_info = None

    @classmethod
    def from_header(cls, header, telescope_code="", planet=0):
        # Set some default dummy values

        tic_id = 0
        filter = ""
        fails = {}
        try:
            dateobs = header['date-obs']
        except KeyError:
            fails["utc_start"] = "UTC date of first image"
        else:
            dateobs = dateobs.split("T")[0].replace("-", "")

        try:
            filter = header['filter']
        except KeyError:
            fails["filter"] = ("filter/passband")

        try:
            obj = header['object']
        except KeyError:
            fails['tic_id'] = "TIC ID number"
        else:
            result = TIC_regex.match(obj)
            if result:
                tic_id = int(result.group("star"))
                # Explicit argument overrules the header
                if result.group("planet") and not planet:
                    # Drop the leading period from the match
                    planet = int(result.group("planet")[1:])
            else:
                # No star from the object after all
                fails['tic_id'] = "TIC ID number"

        fail_msg = ""
        fail = []
        for k, v in fails.items():
            fail.append(f"Unable to determine {k}, {v}, from header.")

        fail = "\n".join(fail)

        if fail:
            raise ValueError(fail)

        return cls(utc_start=dateobs,
                   filter=filter,
                   telescope_code=telescope_code,
                   tic_id=tic_id,
                   planet_number=planet)

    def _valid_tele_code(self):
        return len(self.telescope_code) > 0

    def _valid_planet(self):
        return self.planet_number > 0

    def _valid_tic_num(self):
        return self.tic_id < 10_000_000_000

    def _valid(self):
        """
        Check whether the information so far is valid, meaning:
         + Telescope code is not the empty string
         + Planet number is not zero
         + TIC ID is not more than 10 digits
        """
        valid = (
            self._valid_tele_code() and
            self._valid_planet() and
            self._valid_tic_num()
        )
        return valid

    @property
    def base_name(self):
        if self._valid():
            pieces = [
                f"TIC{self.tic_id}-{self.planet_number:02d}",
                self.utc_start,
                self.telescope_code,
                self.filter
            ]
            return "_".join(pieces)

    @property
    def seeing_profile(self):
        return self.base_name + "_seeing-profile.png"

    @property
    def field_image(self):
        return self.base_name + "_field.png"

    @property
    def field_image_zoom(self):
        return self.base_name + "_field-zoom.png"

    @property
    def apertures(self):
        return self.base_name + "_measurements.apertures"

    @property
    def tic_coord(self):
        if not self._tic_info:
            self._tic_info = get_tic_info(self.tic_id)
        return SkyCoord(ra=self._tic_info['ra'][0], dec=self._tic_info['dec'][0], unit='degree')

    def invalid_parts(self):
        if self._valid():
            return

        if not self._valid_tele_code():
            print(f"Telescope code {self.telescope_code} is not valid")

        if not self._valid_planet():
            print(f"Planet number {self.planet_number} is not valid")

        if not self._valid_tic_num():
            print(f"TIC ID {self.tic_id} is not valid.")


class TOI:
    def __init__(self, tic_id, toi_table=DEFAULT_TABLE_LOCATION, allow_download=True):
        path = Path(toi_table)
        if not path.is_file():
            if not allow_download:
                raise ValueError(f"File {toi_table} not found.")
            toi_table = download_file(TOI_TABLE_URL, cache=True, show_progress=True, timeout=60)

        self._toi_table = Table.read(toi_table, format="ascii.csv")
        self._toi_table = self._toi_table[self._toi_table['TIC ID'] == tic_id]
        if len(self._toi_table) != 1:
            raise RuntimeError(f"Found {len(self._toi_table)} rows in table, expected one.")
        self._tic_info = get_tic_info(tic_id)

    @property
    def tess_mag(self):
        return self._toi_table['TESS Mag'][0]

    @property
    def tess_mag_error(self):
        return self._toi_table['TESS Mag err'][0]

    @property
    def depth(self):
        """
        Depth, parts per thousand.
        """
        return self._toi_table['Depth (ppm)'][0] / 1000

    @property
    def depth_error(self):
        """
        Error in depth, parts per thousand.
        """
        return self._toi_table['Depth (ppm) err'][0] / 1000

    @property
    def epoch(self):
        return Time(self._toi_table['Epoch (BJD)'][0], scale='tdb', format='jd')

    @property
    def epoch_error(self):
        return self._toi_table['Epoch (BJD) err'][0] * u.day

    @property
    def period(self):
        return self._toi_table['Period (days)'][0] * u.day

    @property
    def period_error(self):
        return self._toi_table['Period (days) err'][0] * u.day

    @property
    def duration(self):
        return self._toi_table['Duration (hours)'][0] * u.hour

    @property
    def duration_error(self):
        return self._toi_table['Duration (hours) err'][0] * u.hour

    @property
    def coord(self):
        return SkyCoord(ra=self._tic_info['ra'][0], dec=self._tic_info['dec'][0], unit='degree')

    @property
    def tic_id(self):
        return self._tic_info['ID'][0]

@dataclass
class TessTargetFile:
    coord : SkyCoord
    magnitude : float
    depth : float
    file : str = ""
    aperture_server : str = "https://www.astro.louisville.edu/"

    def __post_init__(self):
        if not self.file:
            self.file = NamedTemporaryFile()
        self._path = Path(self.file.name)
        self.target_file = self._retrieve_target_file()
        self.target_table = self._build_table()

    def _retrieve_target_file(self):
        params = dict(
            ra = self.coord.ra.to_string(unit='hour', decimal=False, sep=":"),
            dec = self.coord.dec.to_string(unit='degree', decimal=False, sep=":"),
            mag=self.magnitude,
            depth=self.depth
        )
        result = requests.get(self.aperture_server + "cgi-bin/gaia_to_aij/upload_request.cgi", params=params)
        links = re.search('href="(.+)"', result.text.replace('\n', ''), )
        download_link = self.aperture_server + links[1]
        target_file_contents = requests.get(download_link)
        with open(self._path, "w") as f:
            f.write(target_file_contents.text)


    def _build_table(self):
        from stellarphot.visualization.comparison_functions import read_file

        self.table = read_file(self._path)
