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

    table_descript: `numpy.ndarray`
        This is a 2-D numpy array where each row of the array is the
        table column name, dtype, astropy unit (can be None), and
        the associated attribute name (can also be None, in which case
        tha column of the table has no associated instance attribute).
        The description must include 'ra', 'dec', and 'id' attributes
        among the listed attributes.

    data: `astropy.table.Table`, optional
        A table containing all the instrumental aperture photometry results
        for a given field on a given night.  If no data is passed,
        an empty data table with the proper columns but no data is created.
        If a table is provided, its format is checked against the
        table_descript.

        If data is passed in, it will be checked to make sure all columns
        listed in table_descript exist, HOWEVER, additional columns in data
        not listed in table_descript will NOT be removed.

    Attributes
    ----------
    data: `astropy.table.Table`
        A table formatted to match the table_descript formatting information
        but containing no data.

    dec: `astropy.table.Column`
        A column of declination values with units of degrees

    id: `astropy.table.Column`
        A column of object id values (no specific format assumed)

    ra: `astropy.table.Column`
        A column of right ascension values with units of degrees

    Other attributes may be created as defined by table_descript.
    """

    def __init__(self, table_descript=None, data=None):
        # Handle parameters
        self._table_descript = table_descript
        self.data = data

        # Confirm a proper table description is passed
        if (type(self._table_descript) is not np.ndarray):
            raise TypeError(f"You must provide a 4-column numpy array as table_descript (it is type {type(self._table_descript)}).")
        elif self._table_descript.shape[1] != 4:
            raise ValueError(f"table_descript must be a 4-column numpy array (it has {self._table_descript.shape[1]} columns).")

        # Extract appropriate information from input `table_descript`
        colnames = self._table_descript[:,0].tolist()
        coltypes = self._table_descript[:,1].tolist()
        colunits = self._table_descript[:,2].tolist()
        attrnames = self._table_descript[:,3].tolist()

        # Check that required attributes are in the input attributes
        required_attr = ['ra', 'dec', 'id']
        required_units = [u.deg, u.deg, None]
        for i, this_attr in enumerate(required_attr):
            if this_attr not in attrnames:
                raise ValueError(f"Required attribute '{this_attr}' is not defined in your table_descript.")
            if required_units[i] is not None:
                if colunits[attrnames.index(this_attr)] != required_units[i]:
                    raise ValueError(f"Required attribute '{this_attr}' must have units {required_units[i]} (table_descript has '{colunits[attrnames.index(this_attr)]}').")

        # Perform processing once initialized
        if self.data is None:
            # Create empty astropy table with right format and data
            self.data = Table(names=colnames, masked=False, dtype=coltypes, units=colunits)
        else:
            # Check the format of the data table matches the table_descript by checking
            # each column listed in table_descript exists and is he write type and unit.
            # NOTE: This ignores any columns not in the table_descript, it does not remove them.
            for i,this_col in enumerate(colnames):
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
        for i,attr in enumerate(attrnames):
            if attr is not None:
                self.__setattr__(attr, self.data[colnames[i]])


class PhotometryData(BaseEnhancedTable):
    """
    A base class to hold an `astropy.table.Table` table of reduce photometry
    data, extending it with a mapping of various table columns to attributes.

    USAGE NOTES: If you input a data file, it MUST contain the following columns
    in the following column names, types, and units:

    name                 dtype      unit
    -----------------   -------     -------
    id                  int64
    xcenter             float64     pix
    ycenter             float64     pix
    aperture_sum        float64     adu
    annulus_sum         float64
    RA                  float64     deg
    Dec                 float64     deg
    sky_per_pix_avg     float64     adu
    sky_per_pix_med     float64     adu
    sky_per_pix_std     float64     adu
    fwhm_x              float64
    fwhm_y              float64
    width               float64
    aperture float64    pix
    aperture_area       float64
    annulus_inner       float64     pix
    annulus_outer       float64     pix
    annulus_area        float64
    exposure            float64       s
    date-obs            str23
    night               int64
    aperture_net_flux   float64     adu
    BJD                 float64
    mag_inst            float64
    airmass             float64
    filter              str2
    file                 str38
    star_id             int64
    mag_error           float64     1 / adu
    noise               float64
    noise-aij           float64
    snr                 float64     adu

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
    phot_descript = np.array([['id', '<i8', None, 'id'],
                            ['xcenter', '<f8', u.pix, 'xcenter'],
                            ['ycenter', '<f8', u.pix, 'ycenter'],
                            ['aperture_sum', '<f8', u.adu, 'ap_sum'],
                            ['annulus_sum', '<f8', None, 'ann_sum'],
                            ['RA', '<f8', u.deg, 'ra'],
                            ['Dec', '<f8', u.deg, 'dec'],
                            ['sky_per_pix_avg', '<f8', u.adu, 'spp_avg'],
                            ['sky_per_pix_med', '<f8', u.adu, 'spp_med'],
                            ['sky_per_pix_std', '<f8', u.adu, 'spp_std'],
                            ['fwhm_x', '<f8', None, None],
                            ['fwhm_y', '<f8', None, None],
                            ['width', '<f8', None, 'fwhm'],
                            ['aperture', '<f8', u.pix, 'ap_rad'],
                            ['aperture_area', '<f8', None, 'ap_area'],
                            ['annulus_inner', '<f8', u.pix, 'ann_in'],
                            ['annulus_outer', '<f8', u.pix, 'ann_out'],
                            ['annulus_area', '<f8', None, 'ann_area'],
                            ['exposure', '<f8', u.s, 'exposure'],
                            ['date-obs', '<U23', None, 'date'],
                            ['night', '<i8', None, 'night'],
                            ['aperture_net_flux', '<f8', u.adu, 'ap_netflux'],
                            ['BJD', '<f8', None, 'bjd'],
                            ['mag_inst', '<f8', None, 'mag_inst'],
                            ['airmass', '<f8', None, 'airmass'],
                            ['filter', '<U2', None, 'filter'],
                            ['file', '<U38', None, None],
                            ['star_id', '<i8', None, 'star_id'],
                            ['mag_error', '<f8', u.adu**-1, 'mag_err'],
                            ['noise', '<f8', None, 'noise'],
                            ['noise-aij', '<f8', None, 'noise_aij'],
                            ['snr', '<f8', u.adu, 'snr']])

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