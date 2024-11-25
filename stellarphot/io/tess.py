import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy.utils.data import download_file
from pydantic import BaseModel, ConfigDict

from stellarphot import SourceListData
from stellarphot.settings.astropy_pydantic import (
    AstropyValidator,
    QuantityType,
    WithPhysicalType,
)
from stellarphot.transit_fitting.io import get_tic_info

__all__ = ["tess_photometry_setup", "TessSubmission", "TOI", "TessTargetFile"]

# Makes me want to vomit, but....
DEFAULT_TABLE_LOCATION = "who.the.heck.knows"
TOI_TABLE_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?output=csv"
GAIA_APERTURE_SERVER = "https://www.astro.louisville.edu/"
TIC_regex = re.compile(r"[tT][iI][cC][^\d]?(?P<star>\d+)(?P<planet>\.\d\d)?")

MODEL_DEFAULT_CONFIGURATION = ConfigDict(
    # Make sure default values are valid
    validate_default=True,
    # Make sure changes to values made after initialization are valid
    validate_assignment=True,
    # Make sure there are no extra fields
    extra="forbid",
)


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


class TOI(BaseModel):
    """
    Represent a TOI from the ExoFOP database.

    Parameters
    ----------

    tic_id : int
        The TIC ID of the target.

    coord : `astropy.coordinates.SkyCoord`
        The coordinates of the target.

    depth_ppt : float
        The depth of the transit in parts per thousand.

    depth_error_ppt : float
        The error in the depth of the transit in parts per thousand.

    duration : `astropy.units.Quantity`
        The duration of the transit; must have units of time.

    duration_error : `astropy.units.Quantity`
        The error in the duration of the transit; must have units of time.

    epoch : `astropy.time.Time`
        The epoch of the transit.

    epoch_error : `astropy.units.Quantity`
        The error in the epoch of the transit; must have units of time.

    period : `astropy.units.Quantity`
        The period of the transit; must have units of time.

    period_error : `astropy.units.Quantity`
        The error in the period of the transit; must have units of time.

    tess_mag : float
        The TESS magnitude of the target.

    tess_mag_error : float
        The error in the TESS magnitude of the target.
    """

    model_config = MODEL_DEFAULT_CONFIGURATION

    tic_id: int
    coord: Annotated[SkyCoord, AstropyValidator]
    depth_ppt: float
    depth_error_ppt: float
    duration: Annotated[QuantityType, WithPhysicalType("time")]
    duration_error: Annotated[QuantityType, WithPhysicalType("time")]
    epoch: Annotated[Time, AstropyValidator]
    epoch_error: Annotated[QuantityType, WithPhysicalType("time")]
    period: Annotated[QuantityType, WithPhysicalType("time")]
    period_error: Annotated[QuantityType, WithPhysicalType("time")]
    tess_mag: float
    tess_mag_error: float

    @classmethod
    def from_tic_id(cls, tic_id):
        """
        Create a TOI object from a numerical TIC ID number. This will be obtained
        from ExoFOP-TESS and the TESS Input Catalog (TIC) at MAST.
        """
        toi_table = download_file(
            TOI_TABLE_URL, cache=True, show_progress=True, timeout=60
        )
        toi_table = Table.read(toi_table, format="ascii.csv")
        toi_table = toi_table[toi_table["TIC ID"] == tic_id]
        if len(toi_table) != 1:  # pragma: no cover
            raise RuntimeError(f"Found {len(toi_table)} rows in table, expected one.")
        toi_table = toi_table[0]

        # Retrieve some additional information from the TIC catalog at MAST, and grab
        # the first row of the table.
        tic_info = get_tic_info(tic_id)[0]

        return cls(
            tic_id=tic_id,
            coord=SkyCoord(ra=tic_info["ra"], dec=tic_info["dec"], unit="degree"),
            depth_ppt=toi_table["Depth (ppm)"] / 1000,
            depth_error_ppt=toi_table["Depth (ppm) err"] / 1000,
            duration=toi_table["Duration (hours)"] * u.hour,
            duration_error=toi_table["Duration (hours) err"] * u.hour,
            epoch=Time(toi_table["Epoch (BJD)"], scale="tdb", format="jd"),
            epoch_error=toi_table["Epoch (BJD) err"] * u.day,
            period=toi_table["Period (days)"] * u.day,
            period_error=toi_table["Period (days) err"] * u.day,
            tess_mag=toi_table["TESS Mag"],
            tess_mag_error=toi_table["TESS Mag err"],
        )

    def transit_time_for_observation(self, obs_times):
        """
        Calculate the transit time for a set of observation times.

        Parameters
        ----------

        obs_times : `astropy.time.Time`
            The times of the observations.

        Returns
        -------

        `astropy.time.Time`
            The transit times for the observations.
        """
        first_obs = obs_times[0]
        # Three possible cases here. Either the first time is close to, but before, a
        # transit, or it is close to, but just after a transit, or it is nowhere close
        # to a transit.
        # Assume that the first time is just before a transit
        cycle_number = int((first_obs - self.epoch) / self.period + 1)
        that_transit = cycle_number * self.period + self.epoch

        # Check -- is the first time closer to this transit or the one before it?
        previous_transit = that_transit - self.period
        if abs(first_obs - previous_transit) < abs(first_obs - that_transit):
            that_transit = previous_transit

        # Check -- are we way, way, way off from a transit?
        if abs(first_obs - that_transit) > 3 * self.duration:
            warnings.warn("Observation times are far from a transit.", stacklevel=2)

        return that_transit


def tess_photometry_setup(tic_id=None, TOI_object=None, overwrite=False):
    """
    Set up the photometry for a TESS target.

    Parameters
    ----------

    TIC_ID : int, optional
        The TIC ID of the target. Must provide either this or TOI_object.

    TOI_object : `stellarphot.io.TOI`, optional
        The TOI object for the target. Must provide either this or TIC_ID.

    Returns
    -------

    Nothing. Writes files with the TESS target information.
    """
    if tic_id:
        toi = TOI.from_tic_id(tic_id)
    elif TOI_object:
        toi = TOI_object
    else:
        raise ValueError("Must provide either TIC ID or TOI object.")

    ttf = TessTargetFile(toi.coord, toi.tess_mag, toi.depth_ppt)
    new_table = Table(
        {
            "star_id": list(range(len(ttf.table))),
            "ra": ttf.table["coords"].ra,
            "dec": ttf.table["coords"].dec,
            "marker name": ["TESS Targets"] * len(ttf.table),
            "coords": ttf.table["coords"],
        }
    )
    sld = SourceListData(input_data=new_table)
    sl_name = f"TIC-{toi.tic_id}-source-list-input.ecsv"
    try:
        sld.write(sl_name, overwrite=overwrite)
    except OSError as e:
        raise FileExistsError(
            f"Source list {sl_name} already exists: Use overwrite=True to replace"
        ) from e

    info_path = Path(f"TIC-{toi.tic_id}-info.json")
    if info_path.exists() and not overwrite:
        raise FileExistsError(
            f"{info_path} already exists. Use overwrite=True to replace."
        )
    with open(info_path, "w") as f:
        f.write(toi.model_dump_json(indent=2))


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
            # Be sure not to delete the file -- otherwise, on windows,
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
            timeout=15,  # If no response in 15 seconds we won't ever get one...
        )
        if result.status_code != 200:
            raise requests.ConnectionError(
                f"Failed to retrieve target file: {result.text}"
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
