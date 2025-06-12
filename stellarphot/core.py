import re
from copy import deepcopy

import lightkurve as lk
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io.ascii import InconsistentTableError
from astropy.table import Column, QTable, Table, TableAttribute
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch

from .settings import Camera, Observatory, PassbandMap
from .table_representations import (
    _generate_old_table_representers,
    deserialize_models_in_table_meta,
    serialize_models_in_table_meta,
)

__all__ = [
    "BaseEnhancedTable",
    "PhotometryData",
    "CatalogData",
    "apass_dr9",
    "refcat2",
    "vsx_vizier",
    "SourceListData",
]


class BaseEnhancedTable(QTable):
    """
    A class to validate an `astropy.table.QTable` table of astronomical data during
    creation and store metadata as attributes.

    This is based on the `astropy.timeseries.QTable` class. We extend this to
    allow for checking of the units for the columns to match the table_description,
    but what this returns is just an `astropy.table.QTable`.

    Parameters
    ----------

    input_data: `astropy.table.QTable` (Default: None)
        A table containing astronomical data of interest.  This table
        will be checked to make sure all columns listed in table_description
        exist and have the right units. Additional columns that
        may be present in data but are not listed in table_description will
        NOT be removed.  This data is copied, so any changes made during
        validation will not only affect the data attribute of the instance,
        the original input data is left unchanged.

    table_description: dict or dict-like (Default: None)
        This is a dictionary where each key is a required table column name
        and the value corresponding to each key is the required dtype
        (can be None).  This is used to check the format of the input data
        table.  Columns will be output in the order of the keys in the dictionary
        with any additional columns tacked on the end.

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

    def __init__(
        self, *args, input_data=None, table_description=None, colname_map=None, **kwargs
    ):
        if (table_description is None) and (input_data is None):
            # Assume user is trying to create an empty table and let QTable
            # handle it
            super().__init__(*args, **kwargs)
        else:
            # Confirm a proper table description is passed (that is dict-like with keys
            # and values)
            try:
                self._table_description = {k: v for k, v in table_description.items()}
            except AttributeError as err:
                raise TypeError(
                    "You must provide a dict as table_description (input "
                    f"table_description is type {type(table_description)})."
                ) from err

            # Check data before copying to avoid recursive loop and non-QTable
            # data input.
            if not isinstance(input_data, Table) or isinstance(
                input_data, BaseEnhancedTable
            ):
                raise TypeError(
                    "You must provide an astropy Table and NOT a "
                    "BaseEnhancedTable as input_data (currently of "
                    f"type {type(input_data)})."
                )

            # Copy data before potential modification
            data = input_data.copy()

            # Rename columns before validation (if needed)
            if colname_map is not None:
                # Confirm a proper colname_map is passed
                try:
                    self._colname_map = {k: v for k, v in colname_map.items()}
                except AttributeError as err:
                    raise TypeError(
                        "You must provide a dict as table_description "
                        "(input table_description is type "
                        f"{type(self._table_description)})."
                    ) from err

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
                        raise ValueError(
                            f"data['{this_col}'] is of wrong unit "
                            f"(should be {this_unit} but reported "
                            f"as {data[this_col].unit})."
                        )
                except KeyError as err:
                    raise ValueError(
                        f"data['{this_col}'] is missing from input " "data."
                    ) from err
            else:  # Check that columns with no units but are required exist!
                try:
                    _ = data[this_col]
                except KeyError as err:
                    raise ValueError(
                        f"data['{this_col}'] is missing from input " "data."
                    ) from err

    def _update_colnames(self, colname_map, data):
        # Change column names as desired, done before validating the columns,
        # which is why we work on _orig_data
        for orig_name, new_name in colname_map.items():
            try:
                data.rename_column(orig_name, new_name)
            except KeyError as err:
                raise ValueError(
                    f"data['{orig_name}'] is missing from input "
                    "data but listed in colname_map!"
                ) from err

    def _update_passbands(self):
        # Converts filter names in filter column to AAVSO standard names
        # Assumes _passband_map is in namespace.

        # Create a list of new passband names instead of trying to change names in place
        # in case any of the new names are longer than the longest of the old names.
        # If that happens, astropy by default just truncates the names.
        new_filter_name = [
            (self._passband_map[orig_pb] if orig_pb in self._passband_map else orig_pb)
            for orig_pb in self["passband"]
        ]

        self["passband"] = new_filter_name

    def clean(self, remove_rows_with_mask=False, **other_restrictions):
        """
        Return a catalog with only the rows that meet the criteria specified.

        Parameters
        ----------

        catalog : `astropy.table.Table`
            Table of catalog information. There are no restrictions on the columns.

        remove_rows_with_mask : bool, optional
            If ``True``, remove rows in which one or more of the values is masked.

        other_restrictions: dict, optional
            Key/value pairs in which the key is the name of a column in the
            catalog and the value is the criteria that values in that column
            must satisfy to be kept in the cleaned catalog. The criteria must be
            simple, beginning with a comparison operator and including a value.
            See Examples below.

        Returns
        -------

        same type as object whose method was called
            Table with filtered data

        Examples
        --------

        >>> from astropy.table import Table
        >>> from stellarphot import BaseEnhancedTable  # Any subclasses will work too
        >>> t = Table([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], names=('a', 'b'), masked=True)
        >>> bet = BaseEnhancedTable(t)
        >>> bet['a'].mask = [True, False, False]
        >>> bet['b'].mask = [False, False, True]
        >>> bet.clean(remove_rows_with_mask=True)
        <BaseEnhancedTable length=1>
           a       b
        float64 float64
        ------- -------
          2.0     2.0

        >>> bet.clean(a='>2')
        <BaseEnhancedTable length=1>
           a       b
        float64 float64
        ------- -------
            3.0    --

        """
        comparisons = {
            "<": np.less,
            "=": np.equal,
            ">": np.greater,
            "<=": np.less_equal,
            ">=": np.greater_equal,
            "!=": np.not_equal,
        }

        recognized_comparison_ops = "|".join(comparisons.keys())
        criteria_re = re.compile(rf"({recognized_comparison_ops})([-+a-zA-Z0-9]+)")

        keepers = np.ones([len(self)], dtype=bool)

        if remove_rows_with_mask and self.has_masked_values:
            for c in self.columns:
                keepers &= ~self[c].mask

        for column, restriction in other_restrictions.items():
            results = criteria_re.match(restriction)
            if not results:
                raise ValueError(f"Criteria {column}{restriction} not " "understood.")
            comparison_func = comparisons[results.group(1)]
            comparison_value = results.group(2)
            new_keepers = comparison_func(self[column], float(comparison_value))
            keepers = keepers & new_keepers

        return self[keepers]

    @classmethod
    def read(cls, *args, **kwargs):
        """
        Read a table from a file and return it as an instance of this class.

        Parameters
        ----------
        filename : str
            The name of the file to read.
        **kwargs : dict
            Additional keyword arguments to pass to the `astropy.table.Table.read`
            method.
        """
        # Try reading the table using the QTable.read method
        try:
            table = QTable.read(*args, **kwargs)
        except InconsistentTableError:
            # Likely reading an old Table that has models in the metadata.
            # Keep this around for a while to support old tables.
            _generate_old_table_representers()
            table = QTable.read(*args, **kwargs)
        else:
            # If we got here, we can assume the table is a new one and has
            # models as dictionaries in the metadata.
            deserialize_models_in_table_meta(table)
        return cls(table)

    def write(self, *args, **kwargs):
        """
        Write the table to a file.

        Parameters
        ----------
        filename : str
            The name of the file to write.
        **kwargs : dict
            Additional keyword arguments to pass to the `astropy.table.Table.write`
            method.
        """
        original_meta = deepcopy(self.meta)
        serialize_models_in_table_meta(self.meta)
        super().write(*args, **kwargs)
        self.meta = original_meta


class PhotometryData(BaseEnhancedTable):
    """
    A modified `astropy.table.QTable` to hold reduced photometry data that
    provides the convenience of validating the data table is in the proper
    format including units.  It returns an `PhotometryData` which is
    a `astropy.table.QTable` with additional attributes describing
    the observatory and camera.

    Parameters
    ----------

    input_data: `astropy.table.QTable`, optional (Default: None)
        A table containing all the instrumental aperture photometry results
        to be validated.  Note: It is allowed for the 'ra' and 'dec' columns
        to have np.nan values, but if they do, the 'bjd' column will not be
        computed and will also be left with 'np.nan'.

    observatory: `stellarphot.settings.Observatory`, optional (Default: None)
        Information about the observatory.

    camera: `stellarphot.Camera`, optional (Default: None)
        A description of the CCD used to perform the photometry.

    colname_map: dict, optional (Default: None)
        A dictionary containing old column names as keys and new column
        names as values.  This is used to automatically update the column
        names to the desired names before the validation is performed.

    passband_map: `stellarphot.settings.PassbandMap`, optional (Default: None)
        An object containing a mapping from instrumental passband names to
        AAVSO passband names. This is used to automatically
        update the passband column to AAVSO standard names if desired. See
        the documentation for `stellarphot.settings.PassbandMap` for more
        information. The object behaves like a dictionary when accessing it.

    retain_user_computed: bool, optional (Default: False)
        If True, any computed columns (see USAGE NOTES below) that already
        exist in `data` will be retained.  If False, will throw an error
        if any computed columns already exist in `data`.

    Attributes
    ----------
    camera: `stellarphot.Camera`
        A description of the CCD used to perform the photometry.

    observatory: `stellarphot.settings.Observatory`, optional (Default: None)
        Information about the observatory.

    Notes
    -----
    For validation of inputs, you must provide camera, observatory, AND input_data,
    if you do not, an empty table will be returned.

    To be accepted as valid, the  `input_data` must MUST contain the following columns
    with the following units.  The data in those columns is NOT validated, the values in
    those columns could be invalid.  Furthermore, the 'consistent count units' below
    simply means it can be any unit, but it must be the same for all the columns with
    'consistent count units'.

    =================     =======
    Column name           Unit
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
    aperture_area         u.pix
    annulus_inner         u.pix
    annulus_outer         u.pix
    annulus_area          u.pix
    aperture_sum          consistent count units
    annulus_sum           consistent count units
    sky_per_pix_avg       consistent count units (per pixel)
    sky_per_pix_med       consistent count units (per pixel)
    sky_per_pix_std       consistent count units (per pixel)
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
    =================     =======

    In addition to these required columns, the following columns are created based
    on the input data during creation.

    night

    If these computed columns already exist in `data` class the class
    will throw an error a `ValueError` UNLESS ``ignore_computed=True``
    is passed to the initializer, in which case the columns will be
    retained and not replaced with the computed values.
    """

    # Define columns that must be in table and provide information about their type, and
    # units.
    phot_descript = {
        "star_id": None,
        "ra": u.deg,
        "dec": u.deg,
        "xcenter": u.pix,
        "ycenter": u.pix,
        "fwhm_x": u.pix,
        "fwhm_y": u.pix,
        "width": u.pix,
        "aperture": u.pix,
        "aperture_area": u.pix,
        "annulus_inner": u.pix,
        "annulus_outer": u.pix,
        "annulus_area": u.pix,
        "aperture_sum": None,
        "annulus_sum": None,
        "sky_per_pix_avg": None,
        "sky_per_pix_med": None,
        "sky_per_pix_std": None,
        "aperture_net_cnts": None,
        "noise_cnts": None,
        "noise_electrons": u.electron,
        "snr": None,
        "mag_inst": None,
        "mag_error": None,
        "exposure": u.second,
        "date-obs": None,
        "airmass": None,
        "passband": None,
        "file": None,
    }

    observatory = TableAttribute(default=None)
    camera = TableAttribute(default=None)

    def __init__(
        self,
        *args,
        input_data=None,
        colname_map=None,
        passband_map=None,
        retain_user_computed=False,
        **kwargs,
    ):
        if (
            (self.observatory is None)
            and (self.camera is None)
            and (input_data is None)
        ):
            super().__init__(*args, **kwargs)
        else:
            # Check the time column is correct format and scale
            try:
                if input_data["date-obs"][0].scale != "utc":
                    raise ValueError(
                        "input_data['date-obs'] astropy.time.Time must "
                        "have scale='utc', "
                        f"not '{input_data['date-obs'][0].scale}'."
                    )
            except AttributeError as err:
                # Happens if first item doesn't have a "scale"
                raise ValueError(
                    "input_data['date-obs'] isn't column of "
                    "astropy.time.Time entries."
                ) from err

            # Convert input data to QTable (while also checking for required columns)
            super().__init__(
                input_data=input_data,
                table_description=self.phot_descript,
                colname_map=colname_map,
                **kwargs,
            )

            # From this point forward we should be using self to get at any data
            # columns, because that is where BaseEnhancedTable has put the data.

            # Perform input validation
            if not isinstance(self.observatory, Observatory):
                raise TypeError(
                    "observatory must be an "
                    "stellarphot.settings.Observatory object instead "
                    f"of type {type(self.observatory)}."
                )
            if not isinstance(self.camera, Camera):
                raise TypeError(
                    "camera must be a stellarphot.Camera object instead "
                    f"of type {type(self.camera)}."
                )

            # Check for consistency of counts-related columns
            counts_columns = [
                "aperture_sum",
                "annulus_sum",
                "aperture_net_cnts",
                "noise_cnts",
            ]
            counts_per_pixel_columns = [
                "sky_per_pix_avg",
                "sky_per_pix_med",
                "sky_per_pix_std",
            ]
            cnts_unit = self[counts_columns[0]].unit
            for this_col in counts_columns[1:]:
                if self[this_col].unit != cnts_unit:
                    raise ValueError(
                        f"input_data['{this_col}'] has inconsistent units "
                        f"with input_data['{counts_columns[0]}'] (should "
                        f"be {cnts_unit} but it's "
                        f"{self[this_col].unit})."
                    )
            for this_col in counts_per_pixel_columns:
                if cnts_unit is None:
                    perpixel = u.pixel**-1
                else:
                    perpixel = cnts_unit * u.pixel**-1
                if self[this_col].unit != perpixel:
                    raise ValueError(
                        f"input_data['{this_col}'] has inconsistent units "
                        f"with input_data['{counts_columns[0]}'] (should "
                        f"be {perpixel} but it's "
                        f"{self[this_col].unit})."
                    )

            # Compute additional columns
            computed_columns = ["night"]

            # Check if columns exist already, if they do and retain_user_computed is
            # False,  throw an error.
            for this_col in computed_columns:
                if this_col in self.colnames:
                    if not retain_user_computed:
                        raise ValueError(
                            f"Computed column '{this_col}' already exist "
                            "in data. If you want to keep them, set "
                            "retain_user_computed=True."
                        )
                else:
                    # Compute the columns that need to be computed (match requires
                    # python>=3.10)
                    match this_col:

                        case "night":
                            # Generate integer counter for nights. This should be
                            # approximately the MJD at noon local before the evening of
                            # the observation.
                            hr_offset = int(
                                self.observatory.earth_location.lon.value / 15
                            )
                            # Compute offset to 12pm Local Time before evening
                            LocalTime = Time(self["date-obs"]) + hr_offset * u.hr
                            hr = LocalTime.ymdhms.hour
                            # Compute number of hours to shift to arrive at 12 noon
                            # local time
                            shift_hr = hr.copy()
                            shift_hr[hr < 12] = shift_hr[hr < 12] + 12
                            shift_hr[hr >= 12] = shift_hr[hr >= 12] - 12
                            delta = (
                                -shift_hr * u.hr
                                - LocalTime.ymdhms.minute * u.min
                                - LocalTime.ymdhms.second * u.s
                            )
                            shift = Column(data=delta, name="shift")
                            # Compute MJD at local noon before the evening of this
                            # observation.
                            self["night"] = Column(
                                data=np.array(
                                    (Time(self["date-obs"]) + shift).to_value("mjd"),
                                    dtype=int,
                                ),
                                name="night",
                            )

                        case _:  # pragma: no cover
                            raise ValueError(
                                f"Trying to compute column ({this_col}). "
                                "This should never happen."
                            )

            # Apply the filter/passband name update
            if passband_map is not None:
                self._passband_map = passband_map.model_copy()
                self._update_passbands()

    def add_bjd_col(self, observatory=None, bjd_coordinates=None):
        """
        Returns a astropy column of barycentric Julian date times corresponding to
        the input observations.  It modifies that table in place.

        Parameters
        ----------
        observatory: `stellarphot.settings.Observatory`
            Information about the observatory. Defaults to the observatory in
            the table metadata.

        bjd_coordinates: `astropy.coordinates.SkyCoord`, optional (Default: None)
            The coordinates to use for computing the BJD. If None, the RA and Dec
            columns in the table will be used.
        """
        if observatory is None:
            if self.observatory is None:
                raise ValueError(
                    "You must provide an observatory object to compute BJD."
                )
            observatory = self.observatory

        if bjd_coordinates is None and (
            np.isnan(self["ra"]).any() or np.isnan(self["dec"]).any()
        ):
            print(
                "WARNING: BJD could not be computed "
                "because some RA or Dec values are missing."
            )
            self["bjd"] = np.full(len(self), np.nan)
        else:
            # Convert times at start of each observation to TDB (Barycentric Dynamical
            # Time)
            times = Time(self["date-obs"])
            times_tdb = times.tdb
            times_tdb.format = "jd"  # Switch to JD format

            # Compute light travel time corrections
            if bjd_coordinates is None:
                sky_coords = SkyCoord(ra=self["ra"], dec=self["dec"], unit="degree")
            else:
                sky_coords = bjd_coordinates

            ltt_bary = times.light_travel_time(
                sky_coords, location=observatory.earth_location
            )
            time_barycenter = times_tdb + ltt_bary

            # Return BJD at midpoint of exposure at each location
            self["bjd"] = Time(time_barycenter + self["exposure"] / 2, scale="tdb")

    def lightcurve_for(self, target, flux_column="mag_inst", passband=None):
        """
        Return the light curve for a single star as a `lightkurve.LightCurve` object.
        One of the parameters `star_id`, `coordinates` or `name` must be specified.

        Parameters
        ----------
        target : str, int, or `astropy.coordinates.SkyCoord`
            The target star. This can be a star_id, a SkyCoord object, or a name that
            can be resolved by `astropy.coordinates.SkyCoord.from_name`.

        flux_column : str, optional
            The name of the column to use as the flux. Default is 'mag_inst'. This need
            not actually be a flux.

        passband : str, optional
            The passband to use to generate the lightcurve for. This is only
            needed if there is more that one passband in the data.

        Returns
        -------
        `lightkurve.LightCurve`
            The light curve for the star. This includes all of the columns in the
            `stellarphot.`PhotometryData` object and columns ``time``, ``flux``, and
            ``flux_err``.
        """

        # This will get set if we need to find the star_id from the coordinates
        coordinates = None

        if isinstance(target, str):
            # If the target is a string, it could be a star_id or a name that can be
            # resolved to coordinates.

            # Try star_id first, since that doesn't require a network call
            if target in self["star_id"]:
                star_id = target
            else:
                coordinates = SkyCoord.from_name(target)
        elif isinstance(target, SkyCoord):
            coordinates = target
        else:
            star_id = target

        if coordinates is not None:
            # Find the star_id for the closest coordinate match
            my_coordinates = SkyCoord(self["ra"], self["dec"])
            idx, d2d, _ = coordinates.match_to_catalog_sky(my_coordinates)
            star_id = self["star_id"][idx]
            if d2d > 1 * u.arcsec:
                raise ValueError(
                    f"No matching star in the photometry data found at {coordinates}."
                )

        if star_id not in self["star_id"]:
            raise ValueError(f"No star found that matched {target}.")

        star_data = self[self["star_id"] == star_id]

        passbands = set(star_data["passband"])

        passband_strings = ", ".join(sorted(passbands))
        if len(passbands) > 1 and passband is None:
            raise ValueError(
                f"Multiple passbands found for this star: {passband_strings}. "
                f"You must specify a passband."
            )

        if passband is not None:
            if passband not in passbands:
                raise ValueError(
                    f"Passband {passband} not found for this star. "
                    f"Passbands in the data are {passband_strings}."
                )
            star_data = star_data[star_data["passband"] == passband]

        # Create the columns that light curve needs, adding metadata about where each
        # column came from.
        star_data["time"] = star_data["bjd"]
        star_data.meta["time"] = "BJD at midpoint of exposure, column bjd"

        star_data["flux"] = star_data[flux_column]
        star_data.meta["flux"] = "Instrumental magnitude, column mag_inst"

        # Why value? Because the instrumental magnitude error is fubar,
        # see #463
        flux_error_col = (
            "mag_error" if flux_column == "mag_inst" else flux_column + "_error"
        )
        star_data["flux_err"] = star_data[flux_error_col].value
        star_data.meta["flux_err"] = "Error in instrumental magnitude, column mag_error"

        return lk.LightCurve(star_data)


class CatalogData(BaseEnhancedTable):
    """
    A class to hold astronomical catalog data while performing validation
    to confirm the minimum required columns ('id', 'ra', 'dec', 'mag', and
    'passband') are present and have the correct units.

    As a convenience function, when the user passes in an astropy table to validate,
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

    passband_map: `stellarphot.settings.PassbandMap`, optional (Default: None)
        An object containing a mapping from instrumental passband names to
        AAVSO passband names. This is used to automatically
        update the passband column to AAVSO standard names if desired. See
        the documentation for `stellarphot.settings.PassbandMap` for more
        information. The object behaves like a dictionary when accessing it.

    no_catalog_error: bool, optional (Default: False)
        If True, the catalog data does not contain error information, so a
        column of NaNs will be added for the error values.

    Attributes
    ----------
    catalog_name: str
        User readable name for the catalog.

    catalog_source: str
        User readable designation for the source of the catalog (could be a
        URL or a journal reference).

    passband_map: `stellarphot.settings.PassbandMap`
        An object containing a mapping from instrumental passband names to
        AAVSO passband names. This is used to automatically
        update the passband column to AAVSO standard names if desired. See
        the documentation for `stellarphot.settings.PassbandMap` for more
        information. The object behaves like a dictionary when accessing it.

    Notes
    -----
    For validation of inputs, you must provide input_data, catalog_name, and
    catalog_source.  If you do not, an empty table will be returned.

    input_data MUST contain the following columns with the following units:

    =================     =======
    Column Name           Unit
    -----------------     -------
    id                    None
    ra                    u.deg
    dec                   u.deg
    mag                   None
    mag_error             None   (see note below)
    passband              None
    =================     =======

    If the catalog data does not contain error information, then set the option
    ``no_catalog_error=True`` when creating the object.  This will add a column
    of NaNs for the error values.
    """

    # Define columns that must be in table and provide information about their type, and
    # units.
    catalog_descript = {
        "id": None,
        "ra": u.deg,
        "dec": u.deg,
        "mag": None,
        "mag_error": None,
        "passband": None,
    }

    def __init__(
        self,
        *args,
        input_data=None,
        catalog_name=None,
        catalog_source=None,
        colname_map=None,
        passband_map=None,
        no_catalog_error=False,
        **kwargs,
    ):
        if (input_data is None) and (catalog_name is None) and (catalog_source is None):
            super().__init__(*args, **kwargs)
        else:
            self._passband_map = passband_map

            if input_data is not None:
                # Check whether the colname_map has an entry for mag_error
                # that is None. If it does, and there is no mag_error column
                # in the input data, then add a column of NaNs to the input
                # data for mag_error.

                if (
                    # colname_map must be provided
                    colname_map is not None
                    and
                    # mag_error must not be in the colname map as a value
                    "mag_error" not in colname_map.values()
                    and
                    # mag_error must not be in the input data
                    "mag_error" not in input_data.colnames
                    and
                    # user has opted in to making a column of NaNs
                    no_catalog_error
                ):
                    input_data_copy = input_data.copy()
                    # Make the error column of NaNs
                    input_data_copy["mag_error"] = np.full(len(input_data), np.nan)
                else:
                    input_data_copy = input_data

                # Convert input data to QTable (while also checking for required
                # columns)
                super().__init__(
                    table_description=self.catalog_descript,
                    input_data=input_data_copy,
                    colname_map=colname_map,
                    **kwargs,
                )
                # Add the TableAttributes directly to meta (and adding attribute
                # functions below) since using TableAttributes results in a
                # inability to access the values to due a
                # AttributeError: 'TableAttribute' object has no attribute 'name'
                self.meta["catalog_name"] = str(catalog_name)
                self.meta["catalog_source"] = str(catalog_source)

                # Apply the filter/passband name update
                if passband_map is not None:
                    self._passband_map = passband_map.copy()
                    self._update_passbands()

            else:
                raise ValueError("You must provide input_data to CatalogData.")

    @property
    def catalog_name(self):
        return self.meta["catalog_name"]

    @property
    def catalog_source(self):
        return self.meta["catalog_source"]

    @property
    def passband_map(self):
        return self._passband_map

    @passband_map.setter
    def passband_map(self, value):
        self._passband_map = value

    @staticmethod
    def _tidy_vizier_catalog(data, mag_column_regex, color_column_regex):
        """
        Transform a Vizier catalog with magnitudes into tidy structure. Or
        at least tidier -- this only handles changing magnitude and color
        columns into tidy format. In that format each row is a single
        observation of a single object in a single passband.

        Parameters
        ----------

        data : `astropy.table.Table`
            Table of catalog information. There are no restrictions on the columns.

        mag_column_regex : str
            Regular expression to match magnitude columns.

        color_column_regex : str
            Regular expression to match color columns.

        Returns
        -------

        `astropy.table.Table`
            Table with magnitude and color columns in tidy format. All other
            columns are preserved as they were in the input data.
        """

        mag_re = re.compile(mag_column_regex)
        color_re = re.compile(color_column_regex)

        # Find all the magnitude and color columns
        mag_match = [mag_re.match(col) for col in data.colnames]
        color_match = [color_re.match(col) for col in data.colnames]

        # create a single list of all the matches
        matches = [
            m_match if m_match else c_match
            for m_match, c_match in zip(mag_match, color_match, strict=True)
        ]

        # The passband should be the first group match.
        passbands = [match[1] for match in matches if match]

        # The original column names for those that match
        orig_cols = [match.string for match in matches if match]

        # Each magnitude column should have an error column whose name
        # is the magnitude column name with 'e_' prepended. While prepending
        # is what pandas will need to transform the data, many non-magnitude
        # columns will also start ``e_`` and we don't want to change those,
        # so we will rename the error columns too.
        mag_err_cols = [f"e_{col}" for col in orig_cols]

        # Dictionary to update the magnitude column names. The prepended
        # part could be anything, but the choice below is unlikely to be
        # used in a column name in a real catalog.
        mag_col_prepend = "magstphot"
        mag_col_map = {
            orig_col: f"{mag_col_prepend}_{passband}"
            for orig_col, passband in zip(orig_cols, passbands, strict=True)
        }

        # Dictionary to update the magnitude error column names. The
        # prepended part could be anything, but the choice below is
        # unlikely to be used in a column name in a real catalog.
        mag_err_col_prepend = "errorstphot"
        mag_err_col_map = {
            orig_col: f"{mag_err_col_prepend}_{passband}"
            for orig_col, passband in zip(mag_err_cols, passbands, strict=True)
        }

        # All columns except those we have renamed should be preserved, so make
        # a list of them for use in wide_to_long.
        id_columns = set(data.colnames) - set(orig_cols) - set(mag_err_cols)

        # Make the input data into a Pandas DataFrame
        df = data.to_pandas()

        # Rename the magnitude and magnitude error columns
        df.rename(columns=mag_col_map, inplace=True)
        df.rename(columns=mag_err_col_map, inplace=True)

        # Make the DataFrame tidy
        df = pd.wide_to_long(
            df,
            stubnames=[mag_col_prepend, mag_err_col_prepend],
            i=id_columns,
            j="passband",
            sep="_",
            suffix=".*",
        )

        # Make the magnitude and error column names more sensible
        df.rename(
            columns={mag_col_prepend: "mag", mag_err_col_prepend: "mag_error"},
            inplace=True,
        )
        # Reset the index, which is otherwise a multi-index of the id columns.
        df = df.reset_index()

        # Convert back to an astropy table
        return Table.from_pandas(df)

    @classmethod
    def from_vizier(
        cls,
        field_center,
        desired_catalog,
        radius=0.5 * u.degree,
        clip_by_frame=False,
        padding=100,
        magnitude_limit=None,
        magnitude_limit_passband=None,
        colname_map=None,
        mag_column_regex=r"^([a-zA-Z]+'?|[a-zA-Z]+'?-[a-zA-Z]+'?)_?mag$",
        color_column_regex=r"^([a-zA-Z]+-[a-zA-Z]+)$",
        prepare_catalog=None,
        no_catalog_error=False,
        tidy_catalog=True,
    ):
        """
        Return the items from catalog that are within the search radius and
        (optionally) within the field of view of a frame.

        Parameters
        ----------

        field_center : `astropy.coordinates.SkyCoord`, `astropy.wcs.WCS`, or FITS header
            Either a `~astropy.coordinates.SkyCoord` object, a `~astropy.wcs.WCS` object
            or a FITS header with WCS information. The input coordinate should be the
            center of the frame; if a header or WCS is the input then the center of the
            frame will be determined from the WCS.

        desired_catalog : str
            Vizier name of the catalog to be searched.

        radius : float, optional
            Radius, in degrees, around which to search. Default is 0.5.

        clip_by_frame : bool, optional
            If ``True``, only return items that are within the field of view
            of the frame. Default is ``True``.

        padding : int, optional
            Coordinates need to be at least this many pixels in from the edge
            of the frame to be considered in the field of view. Default value
            is 100.

        magnitude_limit : float, optional
            If provided, only return items with magnitudes less than or equal
            to this value.

        magnitude_limit_passband : str, optional
            If provided, the passband to use for the magnitude limit. The name of
            the passband must be the name native to the catalog. If not
            provided, the first passband in the catalog will be used. If this
            is provided then the `magnitude_limit` must also be provided.

        colname_map : dict, optional
            Dictionary mapping column names in the catalog to column names in
            a `stellarphot.CatalogData` object. Default is ``None``.

        mag_column_regex : str, optional
            Regular expression to match magnitude columns. See notes below for
            more information about the default value.

        color_column_regex : str, optional
            Regular expression to match color columns. See notes below for
            more information about the default value.

        prepare_catalog : callable, optional
            Function to call on the catalog after it is retrieved from Vizier.


        no_catalog_error: bool, optional (Default: False)
            If True, the catalog data does not contain error information, so a
            column of NaNs will be added for the error values.

        tidy_catalog : bool, optional
            If ``True``, the catalog will be tidied into a long format with one
            row per passband *after* running the catalog through `prepare_catlog` if
            that is not ``None``. If ``False``, no tidying will be done. See Notes
            below for more information about tidy data format.

        Returns
        -------

        `stellarphot.CatalogData`
            Table of catalog information.

        Notes
        -----

        In many Vizier catalogs, the magnitude columns are named with a passband
        name followed by ``mag``, sometimes with an underscore ``_`` in between.
        For example, the Johnson V magnitude column is
        ``Vmag`` or ``V_mag``. The default value for ``mag_column_regex`` will match any
        column name that starts with a letter or letters, followed by ``mag`` or
        ``_mag``.

        In many Vizier catalogs, the color columns are named with the passbands
        separated by a hyphen. For example, the Johnson V-I color column is
        ``V-I``. The default value for ``color_column_regex`` will match any
        column name that starts with a letter or letters, followed by a hyphen,
        followed by a letter or letters.

        Tidy data formats are those where each row is a single observation of a
        single object in a single passband.
        """

        if isinstance(field_center, SkyCoord):
            # Center was passed in, just use it.
            center = field_center
            if clip_by_frame:
                raise ValueError(
                    "To clip entries by frame you must use "
                    "a WCS as the first argument."
                )
        elif isinstance(field_center, WCS):
            center = SkyCoord(*field_center.wcs.crval, unit="deg")
        else:
            wcs = WCS(field_center)
            # The header may not have contained WCS information. In that case
            # the WCS CTYPE will be empty strings and we need to raise an
            # error.
            if wcs.wcs.ctype[0] == "" and wcs.wcs.ctype[1] == "":
                raise ValueError(
                    f"Invalid coordinates in input {field_center}. Make sure the "
                    "header contains valid WCS information or pass in a WCS or "
                    "coordinate."
                )
            center = SkyCoord(*wcs.wcs.crval, unit="deg")

        # Get catalog via cone search -- all columns, no limit on rows
        column_filter = (
            {f"{magnitude_limit_passband}": f"<={magnitude_limit}"}
            if magnitude_limit is not None
            else {}
        )
        vizier = Vizier(
            columns=["all"],
            row_limit=-1,
            column_filters=column_filter,
            timeout=180,
        )
        cat = vizier.query_region(center, radius=radius, catalog=desired_catalog)

        # Vizier always returns list even if there is only one element. Grab that
        # element.
        cat = cat[0]

        final_cat = prepare_catalog(cat) if prepare_catalog is not None else cat

        if tidy_catalog:
            final_cat = CatalogData._tidy_vizier_catalog(
                final_cat, mag_column_regex, color_column_regex
            )

        # Since we go through pandas, we lose the units, so we need to add them back
        # in.
        #
        # We need to swap the key/values on the input map to get the old column names
        # as values.
        invert_map = {v: k for k, v in colname_map.items()}
        final_cat[invert_map["ra"]].unit = u.deg
        final_cat[invert_map["dec"]].unit = u.deg

        # Make the CatalogData object....
        cat = cls(
            input_data=final_cat,
            colname_map=colname_map,
            catalog_name=desired_catalog,
            catalog_source="Vizier",
            no_catalog_error=no_catalog_error,
        )

        # ...and now that the column names are standardized, clip by frame if
        # desired.
        if clip_by_frame:
            cat_coords = SkyCoord(ra=cat["ra"], dec=cat["dec"])
            wcs = WCS(field_center)
            x, y = wcs.all_world2pix(cat_coords.ra, cat_coords.dec, 0)
            in_x = (x >= padding) & (x <= wcs.pixel_shape[0] - padding)
            in_y = (y >= padding) & (y <= wcs.pixel_shape[1] - padding)
            in_fov = in_x & in_y
            cat = cat[in_fov]

        return cat

    def passband_columns(self, passbands=None, transformer=None):
        """
        Return an `astropy.table.Table` with passbands as column names instead
        of the default format, which has a single column for passbands.

        Parameters
        ----------
        passbands : list, optional
            List of passbands to include in the output. If not provided, all
            passbands in the catalog will be included.

        transformer : callable, optional
            Function to transform the data in the passband columns. The function
            should take a single argument, which is the data in the passband
            column, and return the transformed data. If not provided, no
            transformation will be applied.

        Returns
        -------
        `astropy.table.Table`
            Table of catalog information with passbands as column names. See Notes below
            for important details about column names.

        Notes
        -----

        The column names in the output will be the passband names with ``mag_`` as a
        prefix. An error column for each passband will be generated, with the prefix
        ``mag_error_``. If the catalog already has columns with these names, they will
        be overwritten. The input catalog will not be changed.
        """
        catalog_passbands = set(self["passband"])
        if passbands is None:
            passbands = catalog_passbands
        input_passbands = set(passbands)
        missing_passbands = input_passbands - catalog_passbands
        if missing_passbands == input_passbands:
            # The user only request transformed passbands, give them all of the
            # native passbands instead. The catalog native passbands will be necessary
            # to do any transforms of the data.
            input_passbands = catalog_passbands
        if missing_passbands and transformer is None:
            # If there are missing passbands and no transformer, raise an error
            raise ValueError(
                f"Passbands \"{', '.join(missing_passbands)}\" not found in catalog."
            )
        passband_mask = np.zeros(len(self), dtype=bool)
        for passband in input_passbands:
            passband_mask |= self["passband"] == passband

        reduced_input = self[passband_mask]

        # Grab a copy of the metadata to make sure it is preserved. to_pandas will
        # strip the metadata from the table.
        metadata = reduced_input.meta.copy()

        # Switch to pandas for making the new table.
        df = reduced_input.to_pandas()

        # This makes a MultiIndex for the columns -- "mag" and "mag_error" are the
        # top level, and the passbands are the second level.
        df = df.pivot(
            columns="passband", index=["id", "ra", "dec"], values=["mag", "mag_error"]
        )

        # The column names are a MultiIndex, so we flatten them to either "mag_band"
        # or "mag_error_band", where "band" is the passband name.
        df.columns = df.columns.to_series().str.join("_")

        # We also reset the index which was set to the id, ra, and dec columns above.
        df = df.reset_index()

        # Convert back to an astropy table
        return_table = Table.from_pandas(df)

        # Add the metadata back to the table
        return_table.meta.update(metadata)

        # If we have missing columns try feeding the table into the transformer
        if missing_passbands:
            return_table = transformer(return_table)
            still_missing = [
                band
                for band in missing_passbands
                if f"mag_{band}" not in return_table.colnames
            ]
            if still_missing:
                raise ValueError(
                    f"Transformer did not add columns for passbands "
                    f"{', '.join(still_missing)}."
                )
        return return_table


def apass_dr9(
    field_center,
    radius=1 * u.degree,
    clip_by_frame=False,
    padding=100,
    magnitude_limit=None,
    magnitude_limit_passband="V",
):
    """
    Return the items from APASS DR9 that are within the search radius and
    (optionally) within the field of view of a frame.

    Parameters
    ----------
    field_center : `astropy.coordinates.SkyCoord`, `astropy.wcs.WCS`, or FITS header
        Either a `~astropy.coordinates.SkyCoord` object, a `~astropy.wcs.WCS` object
        or a FITS header with WCS information. The input coordinate should be the
        center of the frame; if a header or WCS is the input then the center of the
        frame will be determined from the WCS.

    radius : `astropy.units.Quantity`, optional
        Radius around which to search.

    clip_by_frame : bool, optional
        If ``True``, only return items that are within the field of view
        of the frame.

    padding : int, optional
        Coordinates need to be at least this many pixels in from the edge
        of the frame to be considered in the field of view. Default value
        is 100.

    magnitude_limit : float, optional
        If provided, only return items with magnitudes less than or equal
        to this value.

    magnitude_limit_passband : str, optional
        If provided, the passband to use for the magnitude limit. The name of
        the passband must be one of the AAVSO standard passband names.

    Returns
    -------

    `stellarphot.CatalogData`
        Table of catalog information.

    Notes
    -----
    APASS DR9 does not include an identifier column. Thought Vizier does provide
    a ``recno`` column, it is does not stay the same over time. This function generates
    an ID based on the coordinates of the APASS star, following the guidelines in
    `IAU designation specification <https://cds.unistra.fr/Dic/iau-spec.html>`_.

    """
    apass_colnames = {
        # There is no APASS ID, and this isn't a real ID either...but we need something
        # for ID, and every APASS line is guaranteed to have a field number, so we'll
        # use it. We replace the id column below anyway.
        "Field": "id",
        "RAJ2000": "ra",
        "DEJ2000": "dec",
    }

    aavso_passband_to_aavso_colnames = dict(
        B="Bmag",
        V="Vmag",
        SG="g'mag",
        SR="r'mag",
        SI="i'mag",
    )
    # Make sure the magnitude limit passband is one of the AAVSO standard passband names
    if magnitude_limit_passband:
        if magnitude_limit_passband not in aavso_passband_to_aavso_colnames:
            raise ValueError(
                "magnitude_limit_passband must be one of "
                f"{', '.join(aavso_passband_to_aavso_colnames.keys())}."
            )
        else:
            # If it is valid, then use the refcat2 column name for the passband
            magnitude_limit_passband = aavso_passband_to_aavso_colnames[
                magnitude_limit_passband
            ]

    raw_catalog = CatalogData.from_vizier(
        field_center,
        "II/336/apass9",
        radius=radius,
        clip_by_frame=clip_by_frame,
        padding=padding,
        colname_map=apass_colnames,
        magnitude_limit=magnitude_limit,
        magnitude_limit_passband=magnitude_limit_passband,
    )

    # IAU requires an acronym to star, so make it APASS plus SP for stellarphot
    designation_acronym = "APASSSP"

    # The formats below include 4 digits after the decimal point (accuracy of about
    # 0.5 arcsec), a leading sign (+ or -) and leading zeros so that the RA is always
    # three digits before the decimal and the DEC is always two digits before the
    # decimal.
    coord_string = [
        f"J{ra.to('degree').value:0=+9.4f}{dec.to('degree').value:0=+8.4f}"
        for ra, dec in zip(raw_catalog["ra"], raw_catalog["dec"], strict=True)
    ]

    # IAU says there is a space between the acronym and the coordinates.
    raw_catalog["id"] = [f"{designation_acronym} {coord}" for coord in coord_string]

    # Translate the passbands to AAVSO standard names.
    # No need to change B and V since those are already correct.
    # Do this *after* initialization so that the original APASS band names
    # are used for the tidy-ification operation.
    raw_catalog.passband_map = PassbandMap(
        name="APASS",
        your_filter_names_to_aavso={
            "g": "SG",
            "r": "SR",
            "i": "SI",
            "g'": "SG",
            "r'": "SR",
            "i'": "SI",
        },
    )
    raw_catalog._update_passbands()

    return raw_catalog


def vsx_vizier(
    field_center,
    radius=1 * u.degree,
    clip_by_frame=False,
    padding=100,
    magnitude_limit=None,
    magnitude_limit_passband=None,
):
    """
    Return the items from the copy of VSX on Vizier that are within the search
    radius and (optionally) within the field of view of a frame.

    Parameters
    ----------
    field_center : `astropy.coordinates.SkyCoord`, `astropy.wcs.WCS`, or FITS header
        Either a `~astropy.coordinates.SkyCoord` object, a `~astropy.wcs.WCS` object
        or a FITS header with WCS information. The input coordinate should be the
        center of the frame; if a header or WCS is the input then the center of the
        frame will be determined from the WCS.

    radius : `astropy.units.Quantity`, optional
        Radius around which to search.

    clip_by_frame : bool, optional
        If ``True``, only return items that are within the field of view
        of the frame.

    padding : int, optional
        Coordinates need to be at least this many pixels in from the edge
        of the frame to be considered in the field of view. Default value
        is 100.

    magnitude_limit : float, optional
        If provided, only return items with a brightest magnitudes less than or equal
        to this value.

    magnitude_limit_passband : str, optional
        There is no straightforward way to limit the VSX catalog by passband. The
        magnitude limit will be applied to the variable star's magnitude at maximum.

    Returns
    -------
    `stellarphot.CatalogData`
        Table of catalog information.
    """
    vsx_map = dict(
        Name="id",
        RAJ2000="ra",
        DEJ2000="dec",
    )

    if magnitude_limit_passband is not None:
        raise ValueError(
            "There is no straightforward way to limit the VSX catalog by passband. "
            "The magnitude limit will be applied to the variable star's magnitude "
            "at maximum."
        )

    if magnitude_limit is not None:
        magnitude_limit_passband = "max"

    # This one is easier -- it already has the passband in a column name.
    # We'll use the maximum magnitude as the magnitude column.
    def prepare_cat(cat):
        cat.rename_column("max", "mag")
        cat.rename_column("n_max", "passband")
        return cat

    return CatalogData.from_vizier(
        field_center,
        "B/vsx/vsx",
        radius=radius,
        clip_by_frame=clip_by_frame,
        padding=padding,
        colname_map=vsx_map,
        prepare_catalog=prepare_cat,
        no_catalog_error=True,
        tidy_catalog=False,
        magnitude_limit=magnitude_limit,
        magnitude_limit_passband=magnitude_limit_passband,
    )


def refcat2(
    field_center,
    radius=1 * u.degree,
    clip_by_frame=False,
    padding=100,
    magnitude_limit=None,
    magnitude_limit_passband="SR",
):
    """
    Return the items from Refcat2 that are within the search radius and
    (optionally) within the field of view of a frame.

    Parameters
    ----------
    field_center : `astropy.coordinates.SkyCoord`, `astropy.wcs.WCS`, or FITS header
        Either a `~astropy.coordinates.SkyCoord` object, a `~astropy.wcs.WCS` object
        or a FITS header with WCS information. The input coordinate should be the
        center of the frame; if a header or WCS is the input then the center of the
        frame will be determined from the WCS.

    radius : `astropy.units.Quantity`, optional
        Radius around which to search.

    clip_by_frame : bool, optional
        If ``True``, only return items that are within the field of view
        of the frame.

    padding : int, optional
        Coordinates need to be at least this many pixels in from the edge
        of the frame to be considered in the field of view. Default value
        is 100.

    magnitude_limit : float, optional
        If provided, only return items with magnitudes less than or equal
        to this value.

    magnitude_limit_passband : str, optional
        If provided, the passband to use for the magnitude limit. The name of
        the passband must be one of the AAVSO standard passband names.

    Returns
    -------

    `stellarphot.CatalogData`
        Table of catalog information.

    Notes
    -----
    Refcat2 includes Gaia DR2 RA/Dec and magnitudes but does **not** include
    the Gaia DR2 ID number. This function looks up the Gaia DR2 ID number and uses
    it as the ID column.

    The reference for the refcat2 paper is:

    Tonry, J. L., Denneau, L., Flewelling, H., et al. 2018, ApJ, 867,
    https://iopscience.iop.org/article/10.3847/1538-4357/aae386
    """
    refcat2_colnames = {
        # There is no refcat2 ID number, but below we will match the Gaia DR2
        # ID number to the RA/Dec and use that as the ID.
        "RA_ICRS": "ra",
        "DE_ICRS": "dec",
    }

    aavso_passband_to_refcat_colnames = dict(
        SG="gmag",
        SR="rmag",
        SI="imag",
        SZ="zmag",
    )
    # Make sure the magnitude limit passband is one of the AAVSO standard passband names
    if magnitude_limit_passband:
        if magnitude_limit_passband not in aavso_passband_to_refcat_colnames:
            raise ValueError(
                "magnitude_limit_passband must be one of "
                f"{', '.join(aavso_passband_to_refcat_colnames.keys())}."
            )
        else:
            # If it is valid, then use the refcat2 column name for the passband
            magnitude_limit_passband = aavso_passband_to_refcat_colnames[
                magnitude_limit_passband
            ]

    def process_refcat2(catalog):
        """
        This function does a few things:

        1. Filter out galaxies from the catalog.
        2. Only keep stars that are in the Gaia DR2 catalog.
        3. Add the Gaia DR2 ID number to the catalog as the ID column.
        """
        # 1.
        # The refcat2 paper says that "Virtually all galaxies can be rejected by
        # selecting objects for which Gaia provides a nonzero proper-motion
        # uncertainty," which in the Vizer download are called e_pmRA and e_pmDE,
        # "at the cost of about 0.7% of all real stars." Seems like a reasonable
        # trade-off. Vizier omits the zero entries and astroquery returns a mask for the
        # zero entries, so galaxies are the masked ones.
        galaxies = catalog["e_pmRA"].mask & catalog["e_pmDE"].mask
        catalog = catalog[~galaxies]

        # 2.
        # Also from the paper, "A non-Gaia star may be identified in Refcat2 because it
        # will always have dGaia = 0." In the Vizier version of refcat2, this column is
        # called e_Gmag and instead of being zero, the value is masked.
        catalog = catalog[~catalog["e_Gmag"].mask]

        # 3.
        # Everything left should be a Gaia star, so match to that.
        # This adds some not-insignificant time to getting the catalog, but
        # the result is automatically cached by astroquery, which helps.
        result = XMatch.query(
            cat1=catalog,
            cat2="gaia_dr2_j2015p5",  # "vizier:I/345/gaia2",
            max_distance=0.01 * u.arcsec,
            colRA1="RA_ICRS",
            colDec1="DE_ICRS",
        )
        catalog["id"] = result["source_id"]
        return catalog

    raw_catalog = CatalogData.from_vizier(
        field_center,
        "J/ApJ/867/105/refcat2",
        radius=radius,
        clip_by_frame=clip_by_frame,
        padding=padding,
        colname_map=refcat2_colnames,
        prepare_catalog=process_refcat2,
        magnitude_limit=magnitude_limit,
        magnitude_limit_passband=magnitude_limit_passband,
    )

    # Translate the passbands to AAVSO standard names.
    # No need to change B and V since those are already correct.
    # Do this *after* initialization so that the original passband names
    # are used for the tidy-ification operation.
    raw_catalog.passband_map = PassbandMap(
        name="refcat2",
        your_filter_names_to_aavso={
            "G": "GG",
            "BP": "GBP",
            "RP": "GRP",
            "g": "SG",
            "r": "SR",
            "i": "SI",
            "z": "SZ",
        },
    )
    raw_catalog._update_passbands()

    return raw_catalog


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

    =================     =======
    Column Name           Unit
    -----------------     -------
    star_id               None
    =================     =======

    In addition to the star_id columns you must have EITHER

    =================     =======
    Column Name           Unit
    -----------------     -------
    ra                    u.deg
    dec                   u.deg
    =================     =======

    and/or

    =================     =======
    Column Name           Unit
    -----------------     -------
    xcenter               u.pix
    ycenter               u.pix
    =================     =======

    to define the locations of the sources.  If one locaton pair is provided but not
    the other, the missing columns will be added but assigned NaN values.  It is ok
    to provide both sky and image location, but no validation is done to ensure they
    are consistent.
    """

    # Define columns that must be in table and provide information about their type, and
    # units.
    sourcelist_descript = {
        "star_id": None,
        "ra": u.deg,
        "dec": u.deg,
        "xcenter": u.pix,
        "ycenter": u.pix,
    }

    def __init__(self, *args, input_data=None, colname_map=None, **kwargs):
        if input_data is None:
            super().__init__(*args, **kwargs)
        else:
            # Check data before copying to avoid recursive loop and non-QTable
            # data input.
            if not isinstance(input_data, Table) or isinstance(
                input_data, BaseEnhancedTable
            ):
                raise TypeError(
                    "input_data must be an astropy Table (and not a "
                    "BaseEnhancedTable) as data."
                )

            # Process inputs and save as needed
            data = input_data.copy()

            # Rename columns before checking for ra/dec or xcenter/ycenter
            # columns being missing.
            if colname_map is not None:
                # Confirm a proper colname_map is passed
                try:
                    self._colname_map = {k: v for k, v in colname_map.items()}
                except AttributeError as err:
                    raise TypeError(
                        "You must provide a dict as table_description (it "
                        f"is type {type(self._colname_map)})."
                    ) from err
                self._update_colnames(self._colname_map, data)

                # No need to repeat this
                self._colname_map = None
            else:
                self._colname_map = None

            # Check if RA/Dec or xcenter/ycenter are missing
            ra_dec_present = True
            x_y_present = True
            nosky_pos = (
                "ra" not in data.colnames
                or "dec" not in data.colnames
                or np.isnan(data["ra"].value).all()
                or np.isnan(data["dec"].value).all()
            )
            noimg_pos = (
                "xcenter" not in data.colnames
                or "ycenter" not in data.colnames
                or np.isnan(data["xcenter"].value).all()
                or np.isnan(data["ycenter"].value).all()
            )

            if nosky_pos:
                ra_dec_present = False
            if noimg_pos:
                x_y_present = False

            if nosky_pos and noimg_pos:
                raise ValueError(
                    "data must have either sky (ra, dec) or "
                    + "image (xcenter, ycenter) position."
                )

            # Create empty versions of any missing columns
            for this_col in ["ra", "dec", "xcenter", "ycenter"]:
                # Create blank ra/dec columns
                if this_col not in data.colnames:
                    data[this_col] = Column(
                        data=np.full(len(data), np.nan),
                        name=this_col,
                        unit=self.sourcelist_descript[this_col],
                    )

            # Convert input data to QTable (while also checking for required columns)
            super().__init__(
                table_description=self.sourcelist_descript,
                input_data=data,
                colname_map=None,
                **kwargs,
            )
            self.meta["has_ra_dec"] = ra_dec_present
            self.meta["has_x_y"] = x_y_present

    @property
    def has_ra_dec(self):
        return self.meta["has_ra_dec"]

    @property
    def has_x_y(self):
        return self.meta["has_x_y"]

    def drop_ra_dec(self):
        # drop sky-based positions from existing SourceListData structure
        self.meta["has_ra_dec"] = False
        self["ra"] = Column(data=np.full(len(self), np.nan), name="ra", unit=u.deg)
        self["dec"] = Column(data=np.full(len(self), np.nan), name="dec", unit=u.deg)

    def drop_x_y(self):
        # drop image-based positionsfrom existing SourceListData structure
        self.meta["has_x_y"] = False
        self["xcenter"] = Column(data=np.full(len(self), np.nan), name="ra", unit=u.deg)
        self["ycenter"] = Column(
            data=np.full(len(self), np.nan), name="dec", unit=u.deg
        )
