from astropy import units as u
from astropy.table import QTable, Table, TableAttribute, Column
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time


import numpy as np

__all__ = ['Camera', 'BaseEnhancedTable', 'PhotometryData', 'CatalogData',
           'SourceListData']


class Camera:
    """
    A class to represent a CCD-based camera.

    Parameters
    ----------

    gain : `astropy.quantity.Quantity`
        The gain of the camera in units such that the unit of the product `gain`
        times the image data matches the unit of the `read_noise`.

    read_noise : `astropy.quantity.Quantity`
        The read noise of the camera with units.

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

    table_description: dict or dict-like (Default: None)
        This is a dictionary where each key is a required table column name
        and the value corresponding to each key is the required dtype
        (can be None).  This is used to check the format of the input data
        table.  Columns will be output in the order of the keys in the dictionary
        with any additional columns tacked on the end.

    input_data: `astropy.table.QTable` (Default: None)
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

    Notes
    -----

    This class is based on the `astropy.timeseries.QTable` class.  If no
    table_description and no input_data is provided, then an empty QTable
    is returned.  If table_description and input_data are provided, then
    validation of the inputs is performed and the resulting table is returned.
    """

    def __init__(self, table_description=None, input_data=None, colname_map=None,
                 **kwargs):
        if (table_description is None) and (input_data is None):
            # Assume user is trying to create an empty table and let QTable
            # handle it
            super().__init__(**kwargs)
        else:
            # Confirm a proper table description is passed (that is dict-like with keys
            # and values)
            try:
                self._table_description = {k: v for k, v in table_description.items()}
            except AttributeError:
                raise TypeError("You must provide a dict as table_description (input "
                    f"table_description is type {type(self._table_description)}).")

            # Check data before copying to avoid recusive loop and non-QTable
            # data input.
            if not isinstance(input_data, Table) or isinstance(input_data,
                                                               BaseEnhancedTable):
                raise TypeError("You must provide an astropy Table and NOT a "
                                "BaseEnhancedTable as input_data (currently of "
                                f"type {type(input_data)}).")

            # Copy data before potential modification
            data = input_data.copy()

            # Rename columns before validation (if needed)
            if colname_map is not None:
                # Confirm a proper colname_map is passed
                try:
                    self._colname_map = {k: v for k, v in colname_map.items()}
                except AttributeError:
                    raise TypeError("You must provide a dict as table_description "
                                    "(input table_description is type "
                                    f"{type(self._table_description)}).")

                self._update_colnames(self._colname_map, data)

            # Validate the columns
            self._validate_columns(data)

            # Revise column order to be in the order listed in table_description
            # with unlisted columns tacked on the end
            order_col_list = list(self._table_description.keys())
            for col in data.colnames:
                if col not in order_col_list:
                    order_col_list.append(col)
            data = data[order_col_list]

            # Call QTable initializer to finish up
            super().__init__(data=data, **kwargs)


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
    format including units.  It returns an `PhotometryData` which is
    a `astropy.table.QTable` with additional attributes describing
    the observatory and camera.

    Parameters
    ----------

    observatory: `astropy.coordinates.EarthLocation`, optional (Default: None)
        The location of the observatory.

    camera: `stellarphot.Camera`, optional (Default: None)
        A description of the CCD used to perform the photometry.

    input_data: `astropy.table.QTable`, optional (Default: None)
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
    gain : float
        The gain of the camera in units such that the unit of the product `gain`
        times the image data matches the unit of the `read_noise`.

    read_noise : float
        The read noise of the camera with units.

    dark_current : float
        The dark current of the camera in units such that, when multiplied by
        exposure time, the unit matches the unit of the `read_noise`.

    lat: float
        The location of the observatory in degrees North.

    lon: float
        The location of the observatory in degrees East.

    height: float
        Altitute of the observatory in meters.

    Notes
    -----
    For validation of inputs, you must provide camera, observatory, AND input_data,
    if you do not, an empty table will be returned.

    To be accepted as valid, the  `input_data` must MUST contain the following columns
    with the following units.  The 'consistent count units' simply means it can be any
    unit, but it must be the same for all the 'consistent count units' columns.

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
    noise_cnts            consistent count units
    noise_electrons       u.electron
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
        'noise_cnts' : None,
        'noise_electrons' : u.electron,
        'snr' : None,
        'mag_inst' : None,
        'mag_error' : None,
        'exposure' : u.second,
        'date-obs' : None,
        'airmass' : None,
        'passband' : None,
        'file' : None
    }
    observatory = None
    camera = None

    def __init__(self, observatory=None, camera=None, input_data=None,
                 colname_map=None, passband_map=None, retain_user_computed=False,
                 **kwargs):
        if (observatory is None) and (camera is None) and (input_data is None):
            super().__init__(**kwargs)
        else:
            # Save passband filter name mapping
            self._passband_map = passband_map.copy()

            # Perform input validation
            if not isinstance(observatory, EarthLocation):
                raise TypeError("observatory must be an "
                                "astropy.coordinates.EarthLocation object.")
            if not isinstance(camera, Camera):
                raise TypeError("camera must be a stellarphot.Camera object.")

            # Check the time column is correct format and scale
            try:
                if (input_data['date-obs'][0].scale != 'utc'):
                    raise ValueError("input_data['date-obs'] astropy.time.Time must "
                                     "have scale='utc', "
                                    f"not \'{input_data['date-obs'][0].scale}\'.")
            except AttributeError:
                # Happens if first item dosn't have a "scale"
                raise ValueError("input_data['date-obs'] isn't column of "
                                 "astropy.time.Time entries.")

            # Convert input data to QTable (while also checking for required columns)
            super().__init__(table_description=self.phot_descript,
                             input_data=input_data, colname_map=colname_map,
                            **kwargs)

            # Add the TableAttributes directly to meta (and adding attribute
            # functions below) since using TableAttributes results in a
            # inability to access the values to due a
            # AttributeError: 'TableAttribute' object has no attribute 'name'
            self.meta['lat'] = observatory.lat
            self.meta['lon'] = observatory.lon
            self.meta['height'] = observatory.height
            self.meta['gain'] = camera.gain
            self.meta['read_noise'] = camera.read_noise
            self.meta['dark_current'] = camera.dark_current

            # Check for consistency of counts-related columns
            counts_columns = ['aperture_sum', 'annulus_sum', 'sky_per_pix_avg',
                            'sky_per_pix_med', 'sky_per_pix_std', 'aperture_net_cnts',
                            'noise_cnts']
            cnts_unit = self[counts_columns[0]].unit
            for this_col in counts_columns[1:]:
                if input_data[this_col].unit != cnts_unit:
                    raise ValueError(f"input_data['{this_col}'] has inconsistent units "
                                    f"with input_data['{counts_columns[0]}'] (should "
                                    f"be {cnts_unit} but it's "
                                    f"{input_data[this_col].unit}).")

            # Compute additional columns (not done yet)
            computed_columns = ['bjd', 'night']

            # Check if columns exist already, if they do and retain_user_computed is
            # False,  throw an error.
            for this_col in computed_columns:
                if this_col in self.colnames:
                    if not retain_user_computed:
                        raise ValueError(f"Computed column '{this_col}' already exist "
                                        "in data. If you want to keep them, set "
                                        "retain_user_computed=True.")
                else:
                    # Compute the columns that need to be computed (match requries
                    # python>=3.10)
                    match this_col:
                        case 'bjd':
                            self['bjd'] = self.add_bjd_col(observatory)

                        case 'night':
                            # Generate integer counter for nights. This should be
                            # approximately the MJD at noon local before the evening of
                            # the observation.
                            hr_offset = int(observatory.lon.value/15)
                            # Compute offset to 12pm Local Time before evening
                            LocalTime = Time(self['date-obs']) + hr_offset*u.hr
                            hr = LocalTime.ymdhms.hour
                            # Compute number of hours to shift to arrive at 12 noon
                            # local time
                            shift_hr = hr.copy()
                            shift_hr[hr < 12] = shift_hr[hr < 12] + 12
                            shift_hr[hr >= 12] = shift_hr[hr >= 12] - 12
                            delta = -shift_hr * u.hr - LocalTime.ymdhms.minute * u.min \
                                - LocalTime.ymdhms.second*u.s
                            shift = Column(data = delta, name='shift')
                            # Compute MJD at local noon before the evening of this
                            # observation
                            self['night'] = Column(data=
                                                   np.array((Time(self['date-obs'])
                                                             + shift).to_value('mjd'),
                                                            dtype=int),
                                                   name='night')

                        case _:
                            raise ValueError(f"Trying to compute column ({this_col}). "
                                            "This should never happen.")

            # Apply the filter/passband name update
            if passband_map is not None:
                self._update_passbands()

    def add_bjd_col(self, observatory):
        """
        Returns a astropy column of barycentric Julian date times corresponding to
        the input observations.  It modifies that table in place.
        """

        # Convert times at start of each observation to TDB (Barycentric Dynamical Time)
        times = Time(self['date-obs'])
        times_tdb = times.tdb
        times_tdb.format='jd' # Switch to JD format

        # Compute light travel time corrections
        ip_peg = SkyCoord(ra=self['ra'], dec=self['dec'], unit='degree')
        ltt_bary = times.light_travel_time(ip_peg, location=observatory)
        time_barycenter = times_tdb + ltt_bary

        # Return BJD at midpoint of exposure at each location
        return Time(time_barycenter + self['exposure'] / 2, scale='tdb')

    @property
    def lat(self):
        return self.meta['lat']

    @property
    def lon(self):
        return self.meta['lon']

    @property
    def height(self):
        return self.meta['height']

    @property
    def gain(self):
        return self.meta['gain']

    @property
    def read_noise(self):
        return self.meta['read_noise']

    @property
    def dark_current(self):
        return self.meta['dark_current']

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
    input_data: `astropy.table.Table`, optional (Default: None)
        A table containing all the astronomical catalog data to be validated.
        This data is copied, so any changes made during validation will not
        affect the input data, only the data in the class.

    catalog_name: str, optional (Default: None)
        User readable name for the catalog.

    catalog_source: str, optional (Default: None)
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

    Notes
    -----
    For validation of inputs, you must provide input_data, catalog_name, and
    catalog_source.  If you do not, an empty table will be returned.

    input_data MUST contain the following columns with the following units:

    name                  unit
    -----------------     -------
    id                    None
    ra                    u.deg
    dec                   u.deg
    mag                   None
    passband              None

    Attributes
    ----------
    catalog_name: str
        User readable name for the catalog.

    catalog_source: str
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
    catalog_name = None
    catalog_source = None

    def __init__(self, input_data=None, catalog_name=None, catalog_source=None,
                 colname_map=None, passband_map=None,
                 **kwargs):
        if (input_data is None) and (catalog_name is None) and (catalog_source is None):
            super().__init__(**kwargs)
        else:
            self._passband_map = passband_map

            if (input_data is not None):
                # Convert input data to QTable (while also checking for required
                # columns)
                super().__init__(table_description=self.catalog_descript,
                                input_data=input_data,
                                colname_map=colname_map, **kwargs)
                # Add the TableAttributes directly to meta (and adding attribute
                # functions below) since using TableAttributes results in a
                # inability to access the values to due a
                # AttributeError: 'TableAttribute' object has no attribute 'name'
                self.meta['catalog_name'] = str(catalog_name)
                self.meta['catalog_source'] = str(catalog_source)

            else:
                raise ValueError("You must provide input_data to CatalogData.")

            # Apply the filter/passband name update
            if passband_map is not None:
                self._update_passbands()

    @property
    def catalog_name(self):
        return self.meta['catalog_name']

    @property
    def catalog_source(self):
        return self.meta['catalog_source']


class SourceListData(BaseEnhancedTable):
    """
    A class to hold information on the source lists to pass to
    aperture photometry routines.  It verifies either image-based
    locations (x/y) or sky-based locations (ra/dec) exist.

    Parameters
    ----------
    input_data: `astropy.table.Table`, optional (Default: None)
        A table containing all the source list data to be validated.
        This data is copied, so any changes made during validation will not
        affect the input data, only the data in the class.

    colname_map: dict, optional (Default: None)
        A dictionary containing old column names as keys and new column
        names as values.  This is used to automatically update the column
        names to the desired names BEFORE the validation is performed.

    Attributes
    ----------
    has_ra_dec: bool
        True if the table has sky-based locations (ra/dec), False otherwise.

    has_x_y: bool
        True if the table has image-based locations (x/y), False otherwise.

    Notes
    -----
    For validation of inputs, you must provide input_data, if you do not,
    an empty table will be returned.

    input_data MUST contain the following columns in the following column:

    name                  unit
    -----------------     -------
    star_id               None

    In addition to the star_id columns you must have EITHER

    name                  unit
    -----------------     -------
    ra                    u.deg
    dec                   u.deg

    and/or

    name                  unit
    -----------------     -------
    xcenter               u.pix
    ycenter               u.pix

    to define the locations of the sources.  If one locaton pair is provided but not
    the other, the missing columns will be added but assigned NaN values.  It is ok
    to provide both sky and image location, but no validation is done to ensure they
    are consistent.
    """

    # Define columns that must be in table and provide information about their type, and
    # units.
    sourcelist_descript = {
        'star_id' : None,
        'ra' : u.deg,
        'dec' : u.deg,
        'xcenter' : u.pix,
        'ycenter' : u.pix
    }

    def __init__(self, input_data=None, colname_map=None, **kwargs):
        if (input_data is None):
            super().__init__(**kwargs)
        else:
            # Check data before copying to avoid recusive loop and non-QTable
            # data input.
            if not isinstance(input_data, Table) or isinstance(input_data,
                                                               BaseEnhancedTable):
                raise TypeError("input_data must be an astropy Table (and not a "
                                "BaseEnhancedTable) as data.")

            # Process inputs and save as needed
            data = input_data.copy()

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
            self.meta['has_ra_dec'] = True
            self.meta['has_x_y'] = True
            nosky_pos = ('ra' not in data.colnames or
                        'dec' not in data.colnames)
            noimg_pos = ('xcenter' not in data.colnames or
                        'ycenter' not in data.colnames)
            if nosky_pos:
                self.meta['has_ra_dec'] = False
            if noimg_pos:
                self.meta['has_x_y'] = False

            if (nosky_pos and noimg_pos):
                raise ValueError("data must have either sky (ra, dec) or "+
                                "image (xcenter, ycenter) position.")

            # Create empty versions of any missing columns
            for this_col in ['ra', 'dec', 'xcenter', 'ycenter']:
                # Create blank ra/dec columns
                if (this_col not in data.colnames):
                    data[this_col] = Column(data=np.full(len(data), np.nan),
                                                name=this_col,
                                                unit=self.sourcelist_descript[this_col])

            # Convert input data to QTable (while also checking for required columns)
            super().__init__(table_description=self.sourcelist_descript,
                             input_data=data, colname_map=None, **kwargs)

    @property
    def has_ra_dec(self):
        return self.meta['has_ra_dec']

    @property
    def has_x_y(self):
        return self.meta['has_x_y']
