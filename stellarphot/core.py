from astropy import units as u
from astropy.table import Table, Column

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.timeseries.core import BaseTimeSeries, autocheck_required_columns
from astropy.time import Time

import numpy as np

__all__ = ['Camera', 'BaseEnhancedTable', 'PhotometryData']


class Camera:
    """
    A class to represent a CCD-based camera.

    Parameters
    ----------

    gain : `astropy.quantity.Quantity`
        The gain of the camera in units such that the unit of the product `gain`
        times the image data matches the unit of the `read_noise`.

    read_noise : `astropy.quantity.Quantity`
        The read noise of the camera in units of electrons.

    dark_current : `astropy.quantity.Quantity`
        The dark current of the camera in units such that, when multiplied by
        exposure time, the unit matches the unit of the `read_noise`.

    Attributes
    ----------

    gain : `astropy.quantity.Quantity`
        The gain of the camera in units such that the unit of the product
        `gain` times the image data matches the unit of the `read_noise`.

    read_noise : `astropy.quantity.Quantity`
        The read noise of the camera in units of electrons.

    dark_current : `astropy.quantity.Quantity`
        The dark current of the camera in units such that, when multiplied
        by exposure time, the unit matches the unit of the `read_noise`.

    Notes
    -----
    The gain, read noise, and dark current are all assumed to be constant
    across the entire CCD.

    Examples
    --------
    >>> from astropy import units as u
    >>> from stellarphot import Camera
    >>> camera = Camera(gain=1.0 * u.electron / u.adu,
    ...                 read_noise=1.0 * u.electron,
    ...                 dark_current=0.01 * u.electron / u.second)
    >>> camera.gain
    <Quantity 1. electron / adu>
    >>> camera.read_noise
    <Quantity 1. electron>
    >>> camera.dark_current
    <Quantity 0.01 electron / s>

    """
    def __init__(self, gain=1.0 * u.electron / u.adu,
                 read_noise=1.0 * u.electron,
                 dark_current=0.01 * u.electron / u.second):
        super().__init__()
        self.gain = gain
        self.read_noise = read_noise
        self.dark_current = dark_current


@autocheck_required_columns
class BaseEnhancedTable(BaseTimeSeries):
    """
    A class to validate an `astropy.table.Table` table of astronomical data during
    creation.

    This is based on the `astropy.timeseries.BaseTimeSeries` class to
    allow inheriting its ability to have required columns (but not requiring
    the time column).  We extend this to allow for checking of the units for
    the columns to match the table_description.

    Parameters
    ----------

    table_description: dict
        This is a dictionary where each key is a required table column name
        and the value corresponding to each key is the required dtype
        (can be None).  This is used to check the format of the input data
        table.

    data: `astropy.table.Table`
        A table containing astronomical data of interest.  This table
        will be checked to make sure all columns listed in table_description
        exist and have the right runits. However, additional columns that
        may be present in data but are not listed in table_description will
        NOT be removed.

    Attributes
    ----------
    data: `astropy.table.Table`
        A table formatted to match the table_description formatting information
        but containing no data.
    """

    def __init__(self, table_description, data):
        # Handle parameters
        self._table_description = table_description
        self.data = data.copy()

        # Build the data table
        if not isinstance(self.data, Table):
            raise TypeError(f"You must provide an astropy Table as data (it is type {type(self.data)}).")
        else:
            # Using the BaseTimeSeries class to handle the data table means required columns
            # are checked.
            super().__init__(data=self.data)

            # Check that self.data exists now
            if self.data is None:
                raise ValueError(f"self.data should exist at this point, but doesn't, crap.")

            # Confirm a proper table description is passed
            if not isinstance(self._table_description, dict):
                raise TypeError(f"You must provide a dict as table_description (it is type {type(self._table_description)}).")

            # Check the format of the data table matches the table_description by checking
            # each column listed in table_description exists and is the correct units.
            # NOTE: This ignores any columns not in the table_description, it does not remove them.
            for this_col, this_unit in self._table_description.items():
                if this_unit is not None:
                    # Check type
                    if self.data[this_col].unit != this_unit:
                        raise ValueError(f"data[{this_col}] is of wrong unit (should be {this_unit} but reported as {self.data[this_col].unit}).")


class PhotometryData(BaseEnhancedTable):
    """
    A base class to hold reduced photometry data that provides the convience
    of validating the data table is in the proper format including units.
    This is based on the `astropy.timeseries.BaseTimeSeries` class to allow
    inheriting its ability to have required columns (but not requiring the
    time column).

    Parameters
    ----------

    observatory: `astropy.coordinates.EarthLocation`
        The location of the observatory.

    camera: `stellarphot.Camera`
        A description of the CCD used to perform the photometry.

    data: `astropy.table.Table`
        A table containing all the instrumental aperture photometry results.

    passband_map: dict, optional (Default: None)
        A dictionary containing instrumental passband names as keys and
        AAVSO passband names as values. This is used to automatically
        update the passband column to AAVSO standard names if desired.

    retain_user_computed: bool, optional (Default: False)
        If True, any computed columns (see USAGE NOTES below) that already
        exist in `data` will be retained.  If False, will throw an error
        if any computed columns already exist in `data`.

    USAGE NOTES: If you input a data file, it MUST contain the following columns
    in the following column names with the following units (if applicable).  The
    'consistent count units' simply means it can be any unit for counts, but it
    must be the same for all the columns listed.

    name                  unit
    -----------------     -------
    star_id               None
    RA                    u.deg
    Dec                   u.deg
    xcenter               u.pix
    ycenter               u.pix
    fwhm_x                u.pix
    fwhm_y                u.pix
    width                 u.pix
    aperture              u.pix
    annulus_inner         u.pix
    annulus_outer         u.pix
    aperture_sum          consistent count units
    annulus_sum           consistent count units
    sky_per_pix_avg       consistent count units
    sky_per_pix_med       consistent count units
    sky_per_pix_std       consistent count units
    aperture_net_cnts     consistent count units
    noise                 consistent count units
    exposure              u.s
    date-obs              astropy.time.Time with scale='utc'
    airmass               None
    passband              None
    file                  None

    In addition to these required columns, the following columns are computed based
    on the input data during creation.

    aperture_area
    annulus_area
    snr
    bjd
    night
    mag_inst
    mag_error

    If these computed columns already exist in `data` class the class
    will throw an error a ValueError UNLESS`ignore_computed=True`
    is passed to the initializer, in which case the columns will be
    retained and not replaced with the computed values.

    Attributes
    ----------
    camera: `stellarphot.Camera`
        A description of the CCD used to perform the photometry.

    data: `astropy.table.Table`
        A table containing all the instrumental aperture photometry results.

    observatory: `astropy.coordinates.EarthLocation`
        The location of the observatory.
    """

    # Define columns in the photo_table and provide information about their type, and units.
    phot_descript = {
        'star_id' : None,
        'ra' : u.deg,
        'dec' : u.deg,
        'xcenter' : u.pix,
        'ycenter' : u.pix,
        'fwhm_x' : u.pix,
        'fwhm_y' : u.pix,
        'width' : u.pix,
        'aperture' : u.pix,
        'annulus_inner' : u.pix,
        'annulus_outer' : u.pix,
        'aperture_sum' : None,
        'annulus_sum' : None,
        'sky_per_pix_avg' : None,
        'sky_per_pix_med' : None,
        'sky_per_pix_std' : None,
        'aperture_net_cnts' : None,
        'noise' : None,
        'exposure' : u.s,
        'date-obs' : None,
        'airmass' : None,
        'passband' : None,
        'file' : None
    }

    def __init__(self, observatory, camera, data,  passband_map=None, retain_user_computed=False):
        # Set metavariables
        self.observatory = observatory
        self.camera = camera
        self.passband_map = passband_map

        # Check the time column is correct format.
        # Correct dtype and class
        if (data['date-obs'].dtype != object) or (data['date-obs'][0].__class__ != Time):
            raise ValueError(f"data['date-obs'] is not column of astropy.time.Time objects.")
        # Correct scale is set for first element
        if (data['date-obs'][0].scale != 'utc'):
            raise ValueError(f"data['date-obs'] astropy.time.Time must have scale='utc', not \'{data['date-obs'][0].scale}\'.")

        # Check for consistency of counts-related columns
        counts_columns = ['aperture_sum', 'annulus_sum', 'sky_per_pix_avg', 'sky_per_pix_med',
                          'sky_per_pix_std', 'aperture_net_cnts', 'noise']
        cnts_unit = data[counts_columns[0]].unit
        for this_col in counts_columns[1:]:
            if data[this_col].unit != cnts_unit:
                raise ValueError(f"data[{this_col}] has inconsistent units with data[{counts_columns[0]}] (should be {cnts_unit} but reported as {data[this_col].unit}).")

        # Convert input data to self.data object (which will also automatically check
        # for required columns)
        super().__init__(self.phot_descript, data=data.copy())

        # Check that self.data exists now
        if self.data is None:
            raise ValueError(f"self.data should exist at this point but doesn't, crap.")

        # Compute additional columns (not done yet)
        computed_columns = ['aperture_area', 'annulus_area', 'snr', 'bjd', 'night',
                            'mag_inst', 'mag_error']

        # Check if columns exist already, if they do and ignore_computed is False, throw an error
        for this_col in computed_columns:
            if this_col in self.data.colnames:
                if not retain_user_computed:
                    raise ValueError(f"Computed column '{this_col}' already exist in data.  If you want to keep them, pass retain_user_computed=True to the initializer.")
            else:
                # Compute the columns that need to be computed (switch requries python 3.10)
                match this_col:
                    case 'aperture_area':
                        self.data['aperture_area'] = np.pi * self.data['aperture']**2

                    case 'annulus_area':
                        self.data['annulus_area'] = np.pi * (self.data['annulus_outer']**2 -self.data['annulus_inner']**2)

                    case 'snr':
                        # Since noise in counts, the proper way to compute SNR is sqrt(gain)*counts/noise
                        self.data['snr'] = np.sqrt(self.camera.gain) * self.data['aperture_net_cnts'].value / self.data['noise'].value

                    case 'bjd':
                        self.data['bjd'] = self.add_bjd_col()

                    case 'night':
                        # Generate integer counter for nights. This should be approximately the MJD at noon local
                        # time just before the evening of the observation.
                        hr_offset = int(self.observatory.lon.value/15)
                        # Compute offset to 12pm Local Time before evening
                        LocalTime = Time(self.data['date-obs']) + hr_offset
                        hr = LocalTime.ymdhms.hour
                        # Compute number of hours to shift to arrive at 12 noon local time
                        shift_hr = hr.copy()
                        shift_hr[hr < 12] = shift_hr[hr < 12] + 12
                        shift_hr[hr >= 12] = shift_hr[hr >= 12] - 12
                        shift = Column(data = -shift_hr * u.hr - LocalTime.ymdhms.minute * u.min - LocalTime.ymdhms.second*u.s, name='shift')
                        # Compute MJD at local noon before the evening of this observation
                        self.data['night'] = Column(data=np.array((Time(self.data['date-obs']) + shift).to_value('mjd'), dtype=int), name='night')

                    case 'mag_inst':
                        self.data['mag_inst'] = -2.5 * np.log10(self.camera.gain * self.data['aperture_net_cnts'].value /
                                                    self.data['exposure'].value)

                    case 'mag_error':
                        # data['snr'] must be computed first
                        self.data['mag_inst'] = 1.085736205 / self.data['snr']

                    case _:
                        raise ValueError(f"Trying to compute a column ({this_col}), something that should never happen.")


        # Apply the filter/passband name update
        if passband_map is not None:
            self.update_passbands()


    def update_passbands(self):
        # Converts filter names in filter column to AAVSO standard names
        for orig_pb, aavso_pb in self.passband_map.items():
            mask = self.data['passband'] == orig_pb
            self.data['passband'][mask] = aavso_pb

    def add_bjd_col(self):
        """
        Returns a astropy column of barycentric Julian date times corresponding to
        the input observations.  It modifies that table in place.

        Parameters
        ----------

        phot_table: `astropy.table.Table`
            A stellarphot photometry table.

        """
        place = EarthLocation(lat=self.observatory.lat, lon=self.observatory.lon)

        # Convert times at start of each observation to TDB (Barycentric Dynamical Time)
        times = Time(self.data['date-obs'])
        times_tdb = times.tdb
        times_tdb.format='jd' # Switch to JD format

        # Compute light travel time corrections
        ip_peg = SkyCoord(ra=self.data['ra'], dec=self.data['dec'], unit='degree')
        ltt_bary = times.light_travel_time(ip_peg, location=place)
        time_barycenter = times_tdb + ltt_bary

        # Return BJD at midpoint of exposure at each location
        return Time(time_barycenter + self.data['exposure'] / 2, scale='tdb')
