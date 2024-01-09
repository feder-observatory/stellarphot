import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy.utils.data import download_file

from stellarphot.transit_fitting.io import get_tic_info

__all__ = ["TessSubmission", "TOI", "TessTargetFile"]

# Makes me want to vomit, but....
DEFAULT_TABLE_LOCATION = "who.the.heck.knows"
TOI_TABLE_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?output=csv"
GAIA_APERTURE_SERVER = "https://www.astro.louisville.edu/"
TIC_regex = re.compile(r"[tT][iI][cC][^\d]?(?P<star>\d+)(?P<planet>\.\d\d)?")


@dataclass
class TessSubmission:
    """
    A data class to represent TESS submissions.

    Parameters
    ----------

    telescope_code : str
        The telescope code, e.g. "SRO" or "TJO"

    filter : str
        The filter used for the observations, e.g. "Ic" or "Rc"

    utc_start : str
        The UTC date of the first observation, in YYYYMMDD format

    tic_id : int
        The TIC ID of the target

    planet_number : int
        The planet number, if applicable

    Attributes
    ----------

    apertures : str

    base_name : str

    field_image : str

    field_image_zoom : str

    filter: str
        The filter used for the observations, e.g. "Ic" or "Rc"

    planet_number : int
        The planet number, if applicable

    seeing_profile : str

    tic_coord : `astropy.coordinates.SkyCoord`

    tic_id : int
        The TIC ID of the target

    telescope_code : str
        The telescope code, e.g. "SRO" or "TJO"

    utc_start : str
        The UTC date of the first observation, in YYYYMMDD format

    """

    telescope_code: str
    filter: str
    utc_start: int
    tic_id: int
    planet_number: int

    def __post_init__(self, *args, **kwargs):
        self._tic_info = None

    @classmethod
    def from_header(cls, header, telescope_code="", planet=0):
        """
        Create a TessSubmission from a FITS header

        Parameters
        ----------
        header : `astropy.io.fits.Header`
            The FITS header to parse

        telescope_code : str
            The telescope code, e.g. "SRO" or "TJO"

        planet : int
            The planet number, if applicable
        """

        tic_id = 0
        filter = ""
        fails = {}
        try:
            dateobs = header["date-obs"]
        except KeyError:
            fails["utc_start"] = "UTC date of first image"
        else:
            dateobs = dateobs.split("T")[0].replace("-", "")

        try:
            filter = header["filter"]
        except KeyError:
            fails["filter"] = "filter/passband"

        try:
            obj = header["object"]
        except KeyError:
            fails["tic_id"] = "TIC ID number"
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
                fails["tic_id"] = "TIC ID number"

        fail = []
        for k, v in fails.items():
            fail.append(f"Unable to determine {k}, {v}, from header.")

        fail = "\n".join(fail)

        if fail:
            raise ValueError(fail)

        return cls(
            utc_start=dateobs,
            filter=filter,
            telescope_code=telescope_code,
            tic_id=tic_id,
            planet_number=planet,
        )

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
            self._valid_tele_code() and self._valid_planet() and self._valid_tic_num()
        )
        return valid

    @property
    def base_name(self):
        """
        The base name of the submission, e.g. "TIC123456789-01_20200101_SRO_Ic"
        """
        if self._valid():
            pieces = [
                f"TIC{self.tic_id}-{self.planet_number:02d}",
                self.utc_start,
                self.telescope_code,
                self.filter,
            ]
            return "_".join(pieces)

    @property
    def seeing_profile(self):
        """
        The name of the seeing profile file,
        e.g. "TIC123456789-01_20200101_SRO_Ic_seeing-profile.png"
        """
        return self.base_name + "_seeing-profile.png"

    @property
    def field_image(self):
        """
        The name of the field image file,
        e.g. "TIC123456789-01_20200101_SRO_Ic_field.png"
        """
        return self.base_name + "_field.png"

    @property
    def field_image_zoom(self):
        """
        The name of the zoomed-in field image file,
        e.g. "TIC123456789-01_20200101_SRO_Ic_field-zoom.png"
        """
        return self.base_name + "_field-zoom.png"

    @property
    def apertures(self):
        """
        The name of the apertures file,
        e.g. "TIC123456789-01_20200101_SRO_Ic_measurements.apertures"
        """
        return self.base_name + "_measurements.apertures"

    @property
    def tic_coord(self):
        """
        The SkyCoord of the target, from the TIC catalog.
        """
        if not self._tic_info:
            self._tic_info = get_tic_info(self.tic_id)
        return SkyCoord(
            ra=self._tic_info["ra"][0], dec=self._tic_info["dec"][0], unit="degree"
        )

    def invalid_parts(self):
        """
        Prints a string identifying parts of the submission that are invalid.
        If submission valid, returns nothing.
        """
        if self._valid():
            return

        if not self._valid_tele_code():
            print(f"Telescope code {self.telescope_code} is not valid")

        if not self._valid_planet():
            print(f"Planet number {self.planet_number} is not valid")

        if not self._valid_tic_num():
            print(f"TIC ID {self.tic_id} is not valid.")


class TOI:
    """
    A class to hold information about a TOI (TESS Object of Interest).

    Parameters
    ----------

    tic_id : int
        The TIC ID of the target.

    toi_table : str, optional
        The path to the TOI table. If not provided, the default table will
        be downloaded.

    allow_download : bool, optional
        Whether to allow the default table to be downloaded if it is not found.

    Attributes
    ----------

    coord : `astropy.coordinates.SkyCoord`

    depth : float

    depth_error : float

    duration : float

    duration_error : float

    epoch : float

    epoch_error : float

    period : float

    period_error : float

    tess_mag : float

    tess_mag_error : float

    tic_id : int
    """

    def __init__(self, tic_id, toi_table=DEFAULT_TABLE_LOCATION, allow_download=True):
        path = Path(toi_table)
        if not path.is_file():
            if not allow_download:
                raise ValueError(f"File {toi_table} not found.")
            toi_table = download_file(
                TOI_TABLE_URL, cache=True, show_progress=True, timeout=60
            )

        self._toi_table = Table.read(toi_table, format="ascii.csv")
        self._toi_table = self._toi_table[self._toi_table["TIC ID"] == tic_id]
        if len(self._toi_table) != 1:
            raise RuntimeError(
                f"Found {len(self._toi_table)} rows in table, expected one."
            )
        self._tic_info = get_tic_info(tic_id)

    @property
    def tess_mag(self):
        """
        The TESS magnitude of the target.
        """
        return self._toi_table["TESS Mag"][0]

    @property
    def tess_mag_error(self):
        """
        The uncertainty in the TESS magnitude.
        """
        return self._toi_table["TESS Mag err"][0]

    @property
    def depth(self):
        """
        The transit depth of the target in parts per thousand.
        """
        return self._toi_table["Depth (ppm)"][0] / 1000

    @property
    def depth_error(self):
        """
        The uncertainty in the transit depth in parts per thousand.
        """
        return self._toi_table["Depth (ppm) err"][0] / 1000

    @property
    def epoch(self):
        """
        The epoch of the transit.
        """
        return Time(self._toi_table["Epoch (BJD)"][0], scale="tdb", format="jd")

    @property
    def epoch_error(self):
        """
        The uncertainty in the epoch of the transit.
        """
        return self._toi_table["Epoch (BJD) err"][0] * u.day

    @property
    def period(self):
        """
        The period of the transit.
        """
        return self._toi_table["Period (days)"][0] * u.day

    @property
    def period_error(self):
        """
        The uncertainty in the period of the transit.
        """
        return self._toi_table["Period (days) err"][0] * u.day

    @property
    def duration(self):
        """
        The duration of the transit.
        """
        return self._toi_table["Duration (hours)"][0] * u.hour

    @property
    def duration_error(self):
        """
        The uncertainty in the duration of the transit.
        """
        return self._toi_table["Duration (hours) err"][0] * u.hour

    @property
    def coord(self):
        """
        The coordinates of the target.
        """
        return SkyCoord(
            ra=self._tic_info["ra"][0], dec=self._tic_info["dec"][0], unit="degree"
        )

    @property
    def tic_id(self):
        """
        The TIC ID of the target.
        """
        return self._tic_info["ID"][0]


@dataclass
class TessTargetFile:
    """
    A class to hold information about a TESS target file.  It will retrieve all
    GAIA EDR3 sources within 2.5 arcminutes of the target using the online service at:
    https://www.astro.louisville.edu/gaia_to_aij/index.html

    Parameters
    ----------

    coord : `astropy.coordinates.SkyCoord`
        The coordinates of the target.

    magnitude : float
        The magnitude of the target.

    depth : float
        The depth of the transit.

    file : str, optional
        The path to a file that will be written containing the GAIA sources
        within 2.5 arcminutes the TESS target. If not provided, a temporary
        file will be created.

    Attributes
    ----------

    aperture_server : str
        The URL of the aperture server.

    coord : `astropy.coordinates.SkyCoord`
        The coordinates of the target.

    depth : float
        The depth of the transit.

    file : str
        The path to the file to create with the downloaded GAIA
        data. If not provided, a temporary file will be created.

    magnitude : float
        The magnitude of the target.

    table : `astropy.table.Table`
        A table of targets read in from target_file.

    target_file : str
        The path to the target file.

    target_table : `astropy.table.Table`
        The target table.

    """

    coord: SkyCoord
    magnitude: float
    depth: float
    file: str = ""

    def __post_init__(self):
        self.aperture_server = GAIA_APERTURE_SERVER
        if not self.file:
            # Be sure not to delete the file -- otherwise, one windows,
            # the file is deleted immediately as far as I can tell.
            self.file = NamedTemporaryFile(delete=False)
        self._path = Path(self.file.name)
        self.target_file = self._retrieve_target_file()
        self.target_table = self._build_table()

    def _retrieve_target_file(self):
        params = dict(
            ra=self.coord.ra.to_string(unit="hour", decimal=False, sep=":"),
            dec=self.coord.dec.to_string(unit="degree", decimal=False, sep=":"),
            mag=self.magnitude,
            depth=self.depth,
        )
        result = requests.get(
            self.aperture_server + "cgi-bin/gaia_to_aij/upload_request.cgi",
            params=params,
        )
        links = re.search(
            'href="(.+)"',
            result.text.replace("\n", ""),
        )
        download_link = self.aperture_server + links[1]
        target_file_contents = requests.get(download_link)
        # Write GAIA data to local file
        with open(self._path, "w") as f:
            f.write(target_file_contents.text)

    def _build_table(self):
        from stellarphot.utils.comparison_utils import read_file

        self.table = read_file(self._path)
