from astropy import units as u
from astropy.table import QTable, Table, Column
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time


import numpy as np

__all__ = ['Camera', 'BaseEnhancedTable', 'PhotometryData', 'CatalogData',
           'AperturesData']


class Camera:
    """
    A class to represent a CCD-based camera.

    Parameters
    ----------

    gain : `astropy.quantity.Quantity`
        The gain of the camera in units such that the unit of the product `gain`
        times the image data matches the unit of the `read_noise`.

    read_noise : `astropy.quantity.Quantity`
        The read noise of the camera with units

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

        # Check that input has units and if not crap out.
        if isinstance(gain, u.Quantity):
            self.gain = gain
        else:
            raise TypeError("gain must have units.")

        if isinstance(read_noise, u.Quantity):
            self.read_noise = read_noise
        else:
            raise TypeError("read_noise must have units.")

        if isinstance(dark_current, u.Quantity):
            self.dark_current = dark_current
        else:
            raise TypeError("dark_current must have units.")

    def copy(self):
        return Camera(gain=self.gain,
                      read_noise=self.read_noise,
                      dark_current=self.dark_current)

    def __copy__(self):
        return self.copy()


class BaseEnhancedTable(QTable):
    """
    A class to validate an `astropy.table.QTable` table of astronomical data during
    creation and store metadata as attributes.

    This is based on the `astropy.timeseries.QTable` class. We extend this to
    allow for checking of the units for the columns to match the table_description,
    but what this returns is just an `astropy.table.QTable`.

    Parameters
    ----------

    table_description: dict or dict-like
        This is a dictionary where each key is a required table column name
        and the value corresponding to each key is the required dtype
        (can be None).  This is used to check the format of the input data
        table.  Columns will be output in the order of the keys in the dictionary
        with any additional columns tacked on the end. This requires python >=3.7
        otherwise the key order is not guaranteed.

    data: `astropy.table.QTable`
        A table containing astronomical data of interest.  This table
        will be checked to make sure all columns listed in table_description
        exist and have the right units. Additional columns that
        may be present in data but are not listed in table_description will
        NOT be removed.  This data is copied, so any changes made during
        validation will not only affect the data attribute of the instance,
        the original input data is left unchanged.

    colname_map: dict, optional (Default: None)
        A dictionary containing old column names as keys and new column
        names as values.  This is used to automatically update the column
        names to the desired names BEFORE the validation is performed.
    """

    def __init__(self, table_description, data, colname_map=None):
        # Make copy of input data
        orig_data = data.copy()

        # Confirm a proper table description is passed (that is dict-like with keys and
        # values)
        try:
            self._table_description = {k: v for k, v in table_description.items()}
        except AttributeError:
            raise TypeError("You must provide a dict as table_description (input "
                f"table_description is type {type(self._table_description)}).")

        # Build the data table
        if not isinstance(data, Table):
            raise TypeError("You must provide an astropy Table as data (input data is "
                            f"type {type(data)}).")
        else:
            # Rename columns before validation (if needed)
            if colname_map is not None:
                # Confirm a proper colname_map is passed
                try:
                    self._colname_map = {k: v for k, v in colname_map.items()}
                except AttributeError:
                    raise TypeError("You must provide a dict as table_description "
                                    "(input table_description is type "
                                    f"{type(self._table_description)}).")

                self._update_colnames(self._colname_map,orig_data)

            # Validate the columns
            self._validate_columns(orig_data)

            # Revise column order to be in the order listed in table_description
            # with unlisted columns tacked on the end
            order_col_list = list(self._table_description.keys())
            for col in orig_data.colnames:
                if col not in order_col_list:
                    order_col_list.append(col)
            orig_data = orig_data[order_col_list]

            # Using the QTable class to handle the data table means required
            # columns are checked.
            super().__init__(data=orig_data)


    def _validate_columns(self, data):
        # Check the format of the data table matches the table_description by
        # checking each column listed in table_description exists and is the
        # correct units.
        # NOTE: This ignores any columns not in the table_description, it
        # does not remove them.
        for this_col, this_unit in self._table_description.items():
            if this_unit is not None:
                # Check type
                try:
                    if data[this_col].unit != this_unit:
                        raise ValueError(f"data['{this_col}'] is of wrong unit "
                                            f"(should be {this_unit} but reported "
                                            f"as {data[this_col].unit}).")
                except KeyError:
                    raise ValueError(f"data['{this_col}'] is missing from input "
                                        "data.")
            else: # Check that columns with no units but are required exist!
                try:
                    tempvar = data[this_col]
                except KeyError:
                    raise ValueError(f"data['{this_col}'] is missing from input "
                                     "data.")


    def _update_colnames(self, colname_map, data):
        # Change column names as desired, done before validating the columns,
        # which is why we work on _orig_data
        for orig_name, new_name in colname_map.items():
            try:
                data.rename_column(orig_name, new_name)
            except KeyError:
                raise ValueError(f"data['{orig_name}'] is missing from input "
                                         "data but listed in colname_map!")

    def _update_passbands(self):
        # Converts filter names in filter column to AAVSO standard names
        # Assumes _passband_map is in namespace.
        for orig_pb, aavso_pb in self._passband_map.items():
            mask = self['passband'] == orig_pb
            self['passband'][mask] = aavso_pb


class PhotometryData(BaseEnhancedTable):
    """
    A modified `astropy.table.QTable` to hold reduced photometry data that
    provides the convience of validating the data table is in the proper
    format including units.  It returns an `astropy.table.QTable` with
    additional attributes describing the observatory and camera.

    Parameters
    ----------

    observatory: `astropy.coordinates.EarthLocation`
        The location of the observatory.

    camera: `stellarphot.Camera`
        A description of the CCD used to perform the photometry.

    data: `astropy.table.QTable`
        A table containing all the instrumental aperture photometry results
        to be validated.

    colname_map: dict, optional (Default: None)
        A dictionary containing old column names as keys and new column
        names as values.  This is used to automatically update the column
        names to the desired names before the validation is performed.

    passband_map: dict, optional (Default: None)
        A dictionary containing instrumental passband names as keys and
        AAVSO passband names as values. This is used to automatically
        update the passband column to AAVSO standard names if desired.

    retain_user_computed: bool, optional (Default: False)
        If True, any computed columns (see USAGE NOTES below) that already
        exist in `data` will be retained.  If False, will throw an error
        if any computed columns already exist in `data`.

    Attributes
    ----------
    camera: `stellarphot.Camera`
        A description of the CCD used to perform the photometry.

    observatory: `astropy.coordinates.EarthLocation`
        The location of the observatory.


    Notes
    -----

    The input `astropy.table.QTable` `data` must MUST contain the following columns
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
    aperture_area         u.pix * u.pix
    annulus_inner         u.pix
    annulus_outer         u.pix
    annulus_area          u.pix * u.pix
    aperture_sum          consistent count units
    annulus_sum           consistent count units
    sky_per_pix_avg       consistent count units
    sky_per_pix_med       consistent count units
    sky_per_pix_std       consistent count units
    aperture_net_cnts     consistent count units
    noise                 consistent count units
    snr                   None
    mag_inst              None
    mag_error             None
    exposure              u.s
    date-obs              astropy.time.Time with scale='utc'
    airmass               None
    passband              None
    file                  None

    In addition to these required columns, the following columns are computed based
    on the input data during creation.

    bjd
    night

    If these computed columns already exist in `data` class the class
    will throw an error a ValueError UNLESS`ignore_computed=True`
    is passed to the initializer, in which case the columns will be
    retained and not replaced with the computed values.
    """

    # Define columns that must be in table and provide information about their type, and
    # units.
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
        'aperture_area' : u.pix * u.pix,
        'annulus_inner' : u.pix,
        'annulus_outer' : u.pix,
        'annulus_area' : u.pix * u.pix,
        'aperture_sum' : None,
        'annulus_sum' : None,
        'sky_per_pix_avg' : None,
        'sky_per_pix_med' : None,
        'sky_per_pix_std' : None,
        'aperture_net_cnts' : None,
        'noise' : None,
        'snr' : None,
        'mag_inst' : None,
        'mag_error' : None,
        'exposure' : u.s,
        'date-obs' : None,
        'airmass' : None,
        'passband' : None,
        'file' : None
    }

    def __init__(self, observatory, camera, data,
                 colname_map=None, passband_map=None, retain_user_computed=False):
        # Set attributes describing the observatory and camera as well as corrections to
        # passband names.
        self.observatory = observatory.copy()
        self.camera = camera.copy()
        self._passband_map = passband_map.copy()

        # Check the time column is correct format and scale
        try:
            if (data['date-obs'][0].scale != 'utc'):
                raise ValueError("data['date-obs'] astropy.time.Time must have scale="
                                 f"'utc', not \'{data['date-obs'][0].scale}\'.")
        except AttributeError:
            # Happens if first item dosn't have a "scale"
            raise ValueError("data['date-obs'] isn't column of astropy.time.Time "
                             "entries.")

        # Check for consistency of counts-related columns
        counts_columns = ['aperture_sum', 'annulus_sum', 'sky_per_pix_avg',
                          'sky_per_pix_med', 'sky_per_pix_std', 'aperture_net_cnts',
                          'noise']
        cnts_unit = data[counts_columns[0]].unit
        for this_col in counts_columns[1:]:
            if data[this_col].unit != cnts_unit:
                raise ValueError(f"data['{this_col}'] has inconsistent units with "
                                 f"data['{counts_columns[0]}'] (should be {cnts_unit}"
                                 f" but it's {data[this_col].unit}).")

        # Convert input data to QTable (while also checking for required columns)
        super().__init__(self.phot_descript, data=data, colname_map=colname_map)

        # Compute additional columns (not done yet)
        computed_columns = ['bjd', 'night']

        # Check if columns exist already, if they do and retain_user_computed is False,
        # throw an error.
        for this_col in computed_columns:
            if this_col in self.colnames:
                if not retain_user_computed:
                    raise ValueError(f"Computed column '{this_col}' already exist in "
                                     "data.  If you want to keep them, pass "
                                     "retain_user_computed=True to the initializer.")
            else:
                # Compute the columns that need to be computed (match requries python
                # >=3.10)
                match this_col:
                    case 'bjd':
                        self['bjd'] = self.add_bjd_col()

                    case 'night':
                        # Generate integer counter for nights. This should be
                        # approximately the MJD at noon local before the evening of
                        # the observation.
                        hr_offset = int(self.observatory.lon.value/15)
                        # Compute offset to 12pm Local Time before evening
                        LocalTime = Time(self['date-obs']) + hr_offset*u.hr
                        hr = LocalTime.ymdhms.hour
                        # Compute number of hours to shift to arrive at 12 noon local
                        # time
                        shift_hr = hr.copy()
                        shift_hr[hr < 12] = shift_hr[hr < 12] + 12
                        shift_hr[hr >= 12] = shift_hr[hr >= 12] - 12
                        delta = -shift_hr * u.hr - LocalTime.ymdhms.minute * u.min \
                            - LocalTime.ymdhms.second*u.s
                        shift = Column(data = delta, name='shift')
                        # Compute MJD at local noon before the evening of this
                        # observation
                        self['night'] = Column(data=np.array((Time(self['date-obs'])
                                                              + shift).to_value('mjd'),
                                                             dtype=int), name='night')

                    case _:
                        raise ValueError(f"Trying to compute column ({this_col}). This "
                                         "should never happen.")

        # Apply the filter/passband name update
        if passband_map is not None:
            self._update_passbands()

    def add_bjd_col(self):
        """
        Returns a astropy column of barycentric Julian date times corresponding to
        the input observations.  It modifies that table in place.
        """
        place = EarthLocation(lat=self.observatory.lat, lon=self.observatory.lon)

        # Convert times at start of each observation to TDB (Barycentric Dynamical Time)
        times = Time(self['date-obs'])
        times_tdb = times.tdb
        times_tdb.format='jd' # Switch to JD format

        # Compute light travel time corrections
        ip_peg = SkyCoord(ra=self['ra'], dec=self['dec'], unit='degree')
        ltt_bary = times.light_travel_time(ip_peg, location=place)
        time_barycenter = times_tdb + ltt_bary

        # Return BJD at midpoint of exposure at each location
        return Time(time_barycenter + self['exposure'] / 2, scale='tdb')


class CatalogData(BaseEnhancedTable):
    """
    A class to hold astronomical catalog data while performing validation
    to confirm the minumum required columns ('id', 'ra', and 'dec') are present
    and have the correct units.

    As a convience function, when the user passes in an astropy table to validate,
    the user can also pass in a col_rename dict which can be used to rename columns
    in the data table BEFORE the check that the required columns are present.

    Parameters
    ----------
    data: `astropy.table.Table`
        A table containing all the astronomical catalog data to be validated.
        This data is copied, so any changes made during validation will not
        affect the input data, only the data in the class.

    name: str
        User readable name for the catalog.

    data_source: str
        User readable designation for the source of the catalog (could be a
        URL or a journal reference).

    colname_map: dict, optional (Default: None)
        A dictionary containing old column names as keys and new column
        names as values.  This is used to automatically update the column
        names to the desired names BEFORE the validation is performed.

    passband_map: dict, optional (Default: None)
        A dictionary containing instrumental passband names as keys and
        AAVSO passband names as values. This is used to automatically
        update the passband column to AAVSO standard names if desired.

    USAGE NOTES: If you input a data file, it MUST contain the following columns
    in the following column names with the following units (if applicable).  The
    'consistent count units' simply means it can be any unit for counts, but it
    must be the same for all the columns listed.

    name                  unit
    -----------------     -------
    id                    None
    ra                    u.deg
    dec                   u.deg
    mag                   None
    passband              None

    Attributes
    ----------
    name: str
        User readable name for the catalog.

    data_source: str
        User readable designation for the source of the catalog (could be a
        URL or a journal reference).
    """

    # Define columns that must be in table and provide information about their type, and
    # units.
    catalog_descript = {
        'id' : None,
        'ra' : u.deg,
        'dec' : u.deg,
        'mag' : None,
        'passband' : None
    }

    def __init__(self, data, name, data_source, colname_map=None, passband_map=None):
        # Set attributes
        self.name = str(name)
        self.data_source = str(data_source)
        self._passband_map = passband_map

        # Convert input data to QTable (while also checking for required columns)
        super().__init__(self.catalog_descript, data=data, colname_map=colname_map)

        # Apply the filter/passband name update
        if passband_map is not None:
            self._update_passbands()


class AperturesData(BaseEnhancedTable):
    """
    A class to hold information on the selected apertures to pass to
    aperture photometry routines.

    Parameters
    ----------
    data: `astropy.table.Table`
        A table containing all the apertures data to be validated.
        This data is copied, so any changes made during validation will not
        affect the input data, only the data in the class.

    aperture : `astroy.Quantity`, optional (Default: None)
        The radius of the aperture in pixels (assuming it is fixed).

    annulus_inner : `astroy.Quantity`, optional (Default: None)
        The inner radius of the annulus in pixels (assuming it is fixed).

    annulus_outer : `astroy.Quantity`, optional (Default: None)
        The outer radius of the annulus in pixels (assuming it is fixed).

    colname_map: dict, optional (Default: None)
        A dictionary containing old column names as keys and new column
        names as values.  This is used to automatically update the column
        names to the desired names BEFORE the validation is performed.

    NOTES
    -----

    If you input a data file, it MUST contain the following columns
    in the following column names with the following units (if applicable).  The
    'consistent count units' simply means it can be any unit for counts, but it
    must be the same for all the columns listed.

    name                  unit
    -----------------     -------
    star_id               None

    In addition to these columns you must have EITHER

    name                  unit
    -----------------     -------
    ra                    u.deg
    dec                   u.deg

    and/or

    name                  unit
    -----------------     -------
    xcenter               u.pix
    ycenter               u.pix

    to define the locations of the apertures.  If one locaton pair
    is provided but not the other, the missing columns will be added but
    assigned NaN values.  It is ok to provide both sky and image location, but
    no validation is done to ensure they are consistent.

    Attributes
    ----------

    aperture : `astroy.Quantity`, optional (Default: None)
        The radius of the aperture in pixels (assuming it is fixed).

    annulus_inner : `astroy.Quantity`, optional (Default: None)
        The inner radius of the annulus in pixels (assuming it is fixed).

    annulus_outer : `astroy.Quantity`, optional (Default: None)
        The outer radius of the annulus in pixels (assuming it is fixed).
    """

    # Define columns that must be in table and provide information about their type, and
    # units.
    aperture_descript = {
        'star_id' : None,
        'ra' : u.deg,
        'dec' : u.deg,
        'xcenter' : u.pix,
        'ycenter' : u.pix
    }

    def __init__(self, data, aperture=None, annulus_inner=None, annulus_outer=None,
                 colname_map=None):
        # Process inputs and save as needed
        data = data.copy()
        if aperture is not None:
            if isinstance(aperture, u.Quantity):
                self.aperture = aperture.copy()
            else:
                raise TypeError("aperture must have units.")
        else:
            self.aperture = None

        if annulus_inner is not None:
            if isinstance(annulus_inner, u.Quantity):
                self.annulus_inner = annulus_inner.copy()
            else:
                raise TypeError("annulus_inner must have units.")
        else:
            self.annulus_inner = None

        if annulus_outer is not None:
            if isinstance(annulus_outer, u.Quantity):
                self.annulus_outer = annulus_outer.copy()
            else:
                raise TypeError("annulus_outer must have units.")
        else:
            self.annulus_outer = None

        # Rename columns before checking for ra/dec or xcenter/ycenter
        # columns being missing.
        if colname_map is not None:
            # Confirm a proper colname_map is passed
            try:
                self._colname_map = {k: v for k, v in colname_map.items()}
            except AttributeError:
                raise TypeError("You must provide a dict as table_description (it "
                                f"is type {type(self._colname_map)}).")
            self._update_colnames(self._colname_map, data)

            # No need to repeat this
            self._colname_map = None
        else:
            self._colname_map = None

        # Check if RA/Dec or xcenter/ycenter are missing
        nosky_pos = ('ra' not in data.colnames or
                     'dec' not in data.colnames)
        noimg_pos = ('xcenter' not in data.colnames or
                     'ycenter' not in data.colnames)
        if (nosky_pos and noimg_pos):
            raise ValueError("data must have either sky (ra, dec) or "+
                             "image (xcenter, ycenter) position.")

        # Create empty versions of any missing columns
        for this_col in ['ra', 'dec', 'xcenter', 'ycenter']:
            # Create blank ra/dec columns
            if (this_col not in data.colnames):
                data[this_col] = Column(data=np.full(len(data), np.nan),
                                              name=this_col,
                                              unit=self.aperture_descript[this_col])

        # Convert input data to QTable (while also checking for required columns)
        super().__init__(self.aperture_descript, data=data, colname_map=None)
