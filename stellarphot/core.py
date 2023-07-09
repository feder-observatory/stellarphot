from astropy import units as u
from astropy.table import Table
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


class BaseEnhancedTable:
    """
    A base class to hold an `astropy.table.Table` data table and extend it
    so that chosen columns can be accessed as attributes.  This base class
    imposes the requirement that the table have columns corresponding to
    attributes 'id', 'ra', and 'dec', as well as confirming
    the 'ra' and 'dec' are in degrees.

    Parameters
    ----------

    table_description: `numpy.ndarray`
        This is a 2-D numpy array where each row of the array is the
        table column name, dtype, astropy unit (can be None), and
        the associated attribute name (can also be None, in which case
        tha column of the table has no associated instance attribute).
        The description must include 'ra', 'dec', and 'id' attributes
        among the listed attributes.

    data: `astropy.table.Table`, optional
        A table containing astronomical data of interest, with at least object
        ids, right ascensions, and declinations. If no data is passed,
        an empty data table with the proper columns but no data is created.
        If a table is provided, its format is checked against the
        table_description.

        If data is passed in, it will be checked to make sure all columns
        listed in table_description exist, HOWEVER, additional columns in data
        not listed in table_description will NOT be removed.

    Attributes
    ----------
    data: `astropy.table.Table`
        A table formatted to match the table_description formatting information
        but containing no data.

    dec: `astropy.table.Column`
        A column of declination values with units of degrees

    id: `astropy.table.Column`
        A column of object id values (no specific format assumed)

    ra: `astropy.table.Column`
        A column of right ascension values with units of degrees

    Other attributes may be created as defined by table_description.
    """

    def __init__(self, table_description, data=None):
        # Handle parameters
        self._table_description = table_description
        self.data = data

        # Confirm a proper table description is passed
        if not isinstance(self._table_description, np.ndarray):
            raise TypeError(f"You must provide a 4-column numpy array as table_description (it is type {type(self._table_description)}).")
        elif self._table_description.shape[1] != 4:
            raise ValueError(f"table_description must be a 4-column numpy array (it has {self._table_description.shape[1]} columns).")

        # Extract appropriate information from input `table_description`
        colnames = self._table_description[:,0].tolist()
        coltypes = self._table_description[:,1].tolist()
        colunits = self._table_description[:,2].tolist()
        attrnames = self._table_description[:,3].tolist()

        # Check that required attributes are in the input attributes
        required_attr = ['ra', 'dec', 'id']
        required_units = [u.deg, u.deg, None]
        for i, this_attr in enumerate(required_attr):
            if this_attr not in attrnames:
                raise ValueError(f"Required attribute '{this_attr}' is not defined in your table_description.")
            if required_units[i] is not None:
                if colunits[attrnames.index(this_attr)] != required_units[i]:
                    raise ValueError(f"Required attribute '{this_attr}' must have units {required_units[i]} (table_description has '{colunits[attrnames.index(this_attr)]}').")

        # Perform processing once initialized
        if self.data is None:
            # Create empty astropy table with right format and data
            self.data = Table(names=colnames, masked=False, dtype=coltypes, units=colunits)
        else:
            # Check the format of the data table matches the table_description by checking
            # each column listed in table_description exists and is he write type and unit.
            # NOTE: This ignores any columns not in the table_description, it does not remove them.
            for i, this_col in enumerate(colnames):
                # Failure is assumed to be due to bad column value
                try:
                    data[this_col]
                except KeyError:
                    raise ValueError(f"column {this_col} is not present in input data.")
                # Check type
                if data[this_col].dtype != coltypes[i]:
                    raise ValueError(f"data[{this_col}] is of wrong type (declared {coltypes[i]} but reported as {data[this_col].dtype}).")
                if data[this_col].unit != colunits[i]:
                    raise ValueError(f"data[{this_col}] is of wrong unit (declared {colunits[i]} but reported as {data[this_col].unit}).")

        # Create attibutes corresponding to table column
        self.cols_to_attr(colnames, attrnames)


    def cols_to_attr(self, colnames, attrnames):
        # Connects table columns to attributes
        for i, attr in enumerate(attrnames):
            if attr is not None:
                self.__setattr__(attr, self.data[colnames[i]])


class PhotometryData(BaseEnhancedTable):
    """
    A base class to hold an `astropy.table.Table` table of reduce photometry
    data, extending it with a mapping of various table columns to attributes.

    Parameters
    ----------

    observatory: `astropy.coordinates.EarthLocation`
        The location of the observatory.

    camera: `stellarphot.Camera`
        A description of the CCD used to perform the photometry.

    filter_map: dict
        A dictionary containing instrumental filter names as keys and
        AAVSO filter names as values.

    data: `astropy.table.Table`, optional
        A table containing all the instrumental aperture photometry results.
        If no data is passed, an empty data table with the proper columns but
        no data is created.

    USAGE NOTES: If you input a data file, it MUST contain the following columns
    in the following column names, types, and units:

    name                 dtype      unit
    -----------------   -------     -------
    id                  int
    xcenter             float       pix
    ycenter             float       pix
    aperture_sum        float       adu
    annulus_sum         float
    RA                  float       deg
    Dec                 float       deg
    sky_per_pix_avg     float       adu
    sky_per_pix_med     float       adu
    sky_per_pix_std     float       adu
    fwhm_x              float
    fwhm_y              float
    width               float
    aperture            float       pix
    aperture_area       float
    annulus_inner       float       pix
    annulus_outer       float       pix
    annulus_area        float
    exposure            float       s
    date-obs            str23
    night               int
    aperture_net_flux   float       adu
    BJD                 float
    mag_inst            float
    airmass             float
    filter              str2
    file                str38
    star_id             int
    mag_error           float       1 / adu
    noise               float
    noise-aij           float
    snr                 float       adu

    Attributes
    ----------
    airmass: `astropy.table.Column`
        A column of airmass of observations.

    ann_area: `astropy.table.Column`
        A column of annulus area.

    ann_in: `astropy.table.Column`
        A column of inner annulus radii (in pixels).

    ann_out: `astropy.table.Column`
        A column of outer annulus radii (in pixels).

    ann_sum: `astropy.table.Column`
        A column of sum of counts in the annuli.

    ap_netflux: `astropy.table.Column`
        A column of net fluxes in the apertures.

    ap_rad: `astropy.table.Column`
        A column of aperture radii (in pixels).

    ap_sum: `astropy.table.Column`
        A column of sum of counts in the apertures.

    bjd: `astropy.table.Column`
        A column of barycentric Julian Dates.

    data: `astropy.table.Table`
        A table containing all the instrumental aperture photometry results.

    date: `astropy.table.Column`
        A column of dates.

    dec: `astropy.table.Column`
        A column of declination values with units of degrees

    exposure: `astropy.table.Column`
        A column of exposure (in sec).

    id: `astropy.table.Column`
        A column of unique object identifers.

    mag_err: `astropy.table.Column`
        A column of magnitude uncertainties.

    mag_inst: `astropy.table.Column`
        A column of instrumental magnitudes

    night: `astropy.table.Column`
        A column of nights.

    noise: `astropy.table.Column`
        A column of computed noise.

    noise_aij: `astropy.table.Column`
        A column of noise in the AstroImageJ format.

    ra: `astropy.table.Column`
        A column of right ascension values with units of degrees

    snr: `astropy.table.Column`
        A column of signal to noise ratios.

    spp_avg: `astropy.table.Column`
        A column of average sky counts per pixel.

    spp_med: `astropy.table.Column`
        A column of median sky counts per pixel.

    spp_std: `astropy.table.Column`
        A column of the standar deviation of sky counts per pixel.

    star_id: `astropy.table.Column`
        A column of unique object ids.

    xcenter: `astropy.table.Column`
        A column of the central x position of this object.

    ycenter: `astropy.table.Column`
        A column of the central y position of this object.
    """

    # Define columns in the photo_table and provide information about their type, units,
    # and what attribute name to map that column into (each row of array is
    # name, dtype, unit, attr_name.
    phot_descript = np.array([['id', 'int', None, 'id'],
                            ['xcenter', 'float', u.pix, 'xcenter'],
                            ['ycenter', 'float', u.pix, 'ycenter'],
                            ['aperture_sum', 'float', u.adu, 'ap_sum'],
                            ['annulus_sum', 'float', None, 'ann_sum'],
                            ['RA', 'float', u.deg, 'ra'],
                            ['Dec', 'float', u.deg, 'dec'],
                            ['sky_per_pix_avg', 'float', u.adu, 'spp_avg'],
                            ['sky_per_pix_med', 'float', u.adu, 'spp_med'],
                            ['sky_per_pix_std', 'float', u.adu, 'spp_std'],
                            ['fwhm_x', 'float', None, None],
                            ['fwhm_y', 'float', None, None],
                            ['width', 'float', None, 'fwhm'],
                            ['aperture', 'float', u.pix, 'ap_rad'],
                            ['aperture_area', 'float', None, 'ap_area'],
                            ['annulus_inner', 'float', u.pix, 'ann_in'],
                            ['annulus_outer', 'float', u.pix, 'ann_out'],
                            ['annulus_area', 'float', None, 'ann_area'],
                            ['exposure', 'float', u.s, 'exposure'],
                            ['date-obs', '<U23', None, 'date'],
                            ['night', 'int', None, 'night'],
                            ['aperture_net_flux', 'float', u.adu, 'ap_netflux'],
                            ['BJD', 'float', None, 'bjd'],
                            ['mag_inst', 'float', None, 'mag_inst'],
                            ['airmass', 'float', None, 'airmass'],
                            ['filter', '<U2', None, 'filter'],
                            ['file', '<U38', None, None],
                            ['star_id', 'int', None, 'star_id'],
                            ['mag_error', 'float', u.adu**-1, 'mag_err'],
                            ['noise', 'float', None, 'noise'],
                            ['noise-aij', 'float', None, 'noise_aij'],
                            ['snr', 'float', u.adu, 'snr']])

    def __init__(self, observatory, camera, filter_map, data = None):
        # Set metavariables
        self.observatory = observatory
        self.camera = camera
        self.filter_map = filter_map

        # Handle data table
        super().__init__(self.phot_descript, data)


    def update_filters(self):
        # Converts filter names in filter column to AAVSO standard names
        for this_filter in self.filter_map:
            mask = self.filter == this_filter
            self.filter[mask] = self.filter_map[this_filter]