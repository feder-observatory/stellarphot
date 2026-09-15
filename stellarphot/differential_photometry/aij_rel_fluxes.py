import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import QTable, Table
from astropy.time import Time
from astropy.utils.exceptions import AstropyDeprecationWarning

from stellarphot import PhotometryData, SourceListData

__all__ = ["add_in_quadrature", "calc_aij_relative_flux", "add_relative_flux_column"]

# Largest separation between a comparison star's position and the input
# position of the same star in the photometry data before the comparison
# star table is judged not to be from the source list the photometry was
# made from.
_POSITION_CHECK_LIMIT = 1 * u.arcsec


def add_in_quadrature(array):
    """
    Add an array of numbers in quadrature.
    """
    return np.sqrt((array**2).sum())


def _sky_coords(table, ra_column="ra", dec_column="dec"):
    """
    Make a `~astropy.coordinates.SkyCoord` from two table columns, which are
    taken to be in degrees if they have no unit.
    """
    if table[ra_column].unit is None:
        unit = "degree"
    else:
        # Pulled this from the source code -- None is ok but need
        # to match the number of coordinates.
        unit = [None, None]
    return SkyCoord(ra=table[ra_column], dec=table[dec_column], unit=unit)


def _comp_rows_by_id(star_data, comp_stars, comp_coords, star_id_column):
    """
    Return a boolean mask of the rows of ``star_data`` that are comparison
    stars, matched by ``star_id_column``, after checking that the comparison
    star positions agree with the input positions in ``star_data``.
    """
    comp_ids = np.asarray(comp_stars[star_id_column])
    star_ids = np.asarray(star_data[star_id_column])
    is_comp = np.isin(star_ids, comp_ids)

    if not np.any(is_comp):
        return is_comp

    # Compare every comparison star row in star_data with its entry in
    # comp_stars. The input positions are the ones the photometry was made
    # from, so a mismatch means comp_stars is not from that source list.
    if "ra_input" in star_data.colnames:
        star_coords = _sky_coords(star_data[is_comp], "ra_input", "dec_input")
    else:
        star_coords = _sky_coords(star_data[is_comp])

    comp_index = {star_id: index for index, star_id in enumerate(comp_ids)}
    comp_row = [comp_index[star_id] for star_id in star_ids[is_comp]]
    separation = star_coords.separation(comp_coords[comp_row]).arcsec

    worst = Table(
        data=[star_ids[is_comp], separation], names=["star_id", "separation"]
    ).group_by("star_id")
    worst = worst.groups.aggregate(np.max)
    mismatched = worst[worst["separation"] > _POSITION_CHECK_LIMIT.to_value(u.arcsec)]

    if len(mismatched) > 0:
        details = ", ".join(
            f"{star_id_column} {star_id} ({separation:.2f} arcsec)"
            for star_id, separation in mismatched.iterrows()
        )
        raise ValueError(
            f"The positions of comparison star(s) {details} in comp_stars are "
            f"more than {_POSITION_CHECK_LIMIT} from the input positions in "
            "the photometry data for the same star, so the source list does "
            "not appear to be the one the photometry was made from."
        )

    return is_comp


def _comp_rows_by_position(star_data, comp_coords, star_id_column):
    """
    Return a boolean mask of the rows of ``star_data`` that are comparison
    stars, matched by position. A star whose position is mismatched at any
    one time is excluded at every time. Deprecated; kept for comparison star
    tables that have no ``star_id_column``.
    """
    warnings.warn(
        "Matching comparison stars to the photometry by position is deprecated; "
        f"comp_stars should have a '{star_id_column}' column with the same "
        "star ids as the photometry data, as a stellarphot source list does. "
        "Deprecated since stellarphot 2.2.0; matching by position will be "
        "removed in stellarphot 3.0.0.",
        AstropyDeprecationWarning,
        stacklevel=3,
    )
    star_data_coords = _sky_coords(star_data)

    # Check for matches of stars in star data to the stars in comp_stars
    # and eliminate as comps any stars for which the separation is bigger
    # than 1.2 arcsec in any of the frames.
    _, d2d, _ = star_data_coords.match_to_catalog_sky(comp_coords)

    # Not sure this is really close enough for a good match...
    good = d2d < 1.2 * u.arcsec

    check_for_bad = Table(
        data=[star_data[star_id_column].data, good], names=["star_id", "good"]
    )
    check_for_bad = check_for_bad.group_by("star_id")
    is_all_good = check_for_bad.groups.aggregate(np.all)

    for comp in is_all_good["star_id"][~is_all_good["good"]]:
        good[star_data[star_id_column] == comp] = False

    return good


def calc_aij_relative_flux(
    star_data,
    comp_stars,
    in_place=True,
    coord_column=None,
    star_id_column="star_id",
    counts_column_name="aperture_net_cnts",
):
    """
    Calculate AstroImageJ-style flux ratios.

    Parameters
    ----------

    star_data : 'stellarphot.PhotometryData'
        Photometry data from one or more images.

    comp_stars : '~astropy.table.Table'
        Table of comparison stars in the field. Must contain the column
        named by ``star_id_column``, with the same star ids as ``star_data``,
        and either the column named by ``coord_column`` or columns called
        ``ra`` and ``dec``. A table without ``star_id_column`` is matched
        by position instead, which is deprecated. Not all of the comparison
        stars will necessarily be used; see Notes.

    in_place : bool,  optional
        If ``True``, add new columns to input table. Otherwise, return
        new table with those columns added.

    coord_column : str,  optional
        If provided, use this column for the comparison star coordinates.
        If not provided, the coordinates are generated with SkyCoord from
        the ``ra`` and ``dec`` columns.

    counts_column_name : str,  optional
        If provided, use this column to find counts.

    star_id_column : str,  optional
        Name of the column that provides a unique identifier for each
        comparison star.

    Returns
    -------

    `stellarphot.PhotometryData` or None
        The return type depends on the value of ``in_place``. If it is
        ``False``, then the new columns are returned as a separate table,
        otherwise the columns are simply added to the input table.

    Raises
    ------

    ValueError
        If the position of a comparison star in ``comp_stars`` is more than
        1 arcsec from the input position of the star with the same id in
        ``star_data``.

    RuntimeError
        If no comparison star is in ``star_data``, or if there is a time at
        which no comparison star has valid data.

    Notes
    -----

    Comparison stars are matched to ``star_data`` by star id because the
    photometry code copies the ids from the source list, so the ids agree by
    construction. Positions are only a sanity check: the comparison star
    position is compared with the input position in ``star_data``
    (``ra_input``/``dec_input`` if present, otherwise ``ra``/``dec``), and
    a separation of more than 1 arcsec raises an error rather than silently
    dropping the star, since it means ``comp_stars`` is not from the source
    list the photometry was made from. Matching by position would instead
    start dropping comparison stars once ``ra``/``dec`` in the photometry
    are the measured centroid positions (see issue #710).

    A comparison star is excluded from the comparison set at every time if,
    at any one time, it is missing from ``star_data`` or its net counts are
    ``NaN``.
    """

    if coord_column is not None:
        comp_coords = comp_stars[coord_column]
    else:
        comp_coords = _sky_coords(comp_stars)

    if star_id_column in comp_stars.colnames:
        good = _comp_rows_by_id(star_data, comp_stars, comp_coords, star_id_column)
    else:
        good = _comp_rows_by_position(star_data, comp_coords, star_id_column)

    if not np.any(good):
        raise RuntimeError(
            "No comparison stars matched the stars in the photometry "
            "data, so relative flux cannot be calculated. Check that the "
            "comparison star table is from the source list the photometry "
            "was made from."
        )

    # Check for comps that are only in some of the images
    # Make a small table with just star IDs and date-obs
    check_for_missing = Table(
        data=[star_data[star_id_column], star_data["date-obs"]],
        names=["star_id", "date-obs"],
    ).group_by("date-obs")
    star_id_sets = check_for_missing.groups.aggregate(set)["star_id"]
    good_ids = set.intersection(*star_id_sets)
    bad_comps = set(star_data[star_id_column]) - good_ids

    # Check whether any of the comp stars have NaN values and,
    # if they do, exclude them from the comp set.
    check_for_nan = Table(
        data=[star_data[star_id_column].data, star_data[counts_column_name].data],
        names=["star_id", "net_counts"],
    )
    check_for_nan = check_for_nan.group_by("star_id")
    check_for_nan["good"] = ~np.isnan(check_for_nan["net_counts"])
    is_all_good = check_for_nan.groups.aggregate(np.all)

    bad_comps = bad_comps | set(is_all_good["star_id"][~is_all_good["good"]])

    for comp in bad_comps:
        this_comp = star_data[star_id_column] == comp
        good[this_comp] = False

    # Every time in the input data must have at least one comparison star;
    # otherwise the comparison counts at the times with no comparison stars
    # would silently be set to 1. Note that a comparison star that is bad at
    # any one time is excluded as a comparison star at every time, so this
    # also catches the case in which every comparison star has been
    # excluded.
    star_times = np.asarray(star_data["date-obs"].value)
    times_with_comps = set(star_times[good])
    if set(star_times) - times_with_comps:
        raise RuntimeError(
            "There are one or more times in the photometry data at which "
            "none of the comparison stars has valid data, so relative flux "
            "cannot be calculated. A comparison star is excluded from every "
            "time if it is missing or has NaN counts at even one time."
        )

    error_column_name = "noise_electrons"
    # Calculate comp star counts for each time

    # Make a small table with just counts, errors and time for all of the comparison
    # stars.

    comp_fluxes = star_data["date-obs", counts_column_name, error_column_name][good]
    # Convert comp_fluxes to a regular Table, not a QTable, to work around
    # https://github.com/astropy/astropy/issues/10944
    # in which it was reported that QTable columns with units cannot be aggregated.

    comp_fluxes = Table(comp_fluxes)

    # Check whether any of the columns are masked, but with no masked values,
    # and convert to regular column...eventually

    comp_fluxes = comp_fluxes.group_by("date-obs")
    comp_totals = comp_fluxes.groups.aggregate(np.sum)[counts_column_name]
    # Count the comparison stars in each image by counting the rows in each
    # group. Counting nonzero fluxes would be wrong here -- exactly zero net
    # counts is a legitimate measured value (net counts can even be negative
    # after sky subtraction) and must not be mistaken for a missing star.
    comp_num_stars = np.diff(comp_fluxes.groups.indices)
    comp_errors = comp_fluxes.groups.aggregate(add_in_quadrature)[error_column_name]

    comp_total_vector = np.ones_like(star_data[counts_column_name])
    comp_error_vector = np.ones_like(star_data[error_column_name])

    if len(set(comp_num_stars)) > 1:
        raise RuntimeError("Different number of stars in comparison sets")

    # Calculate relative flux for every star

    # Have to remove the flux of the star if the star is a comparison
    # star.
    # Use the .value below so that we can set the array to 1 and multiply
    # by it without affecting units of the result.
    is_comp = np.zeros_like(star_data[counts_column_name]).value
    is_comp[good] = 1
    flux_offset = -star_data[counts_column_name] * is_comp

    # Convert comp_fluxes back to a QTable and redo groups
    comp_fluxes = QTable(comp_fluxes)
    comp_fluxes = comp_fluxes.group_by("date-obs")
    # This seems a little hacky; there must be a better way
    for date_obs, comp_total, comp_error in zip(
        comp_fluxes.groups.keys, comp_totals, comp_errors, strict=True
    ):
        this_time = star_data["date-obs"] == date_obs[0]
        comp_total_vector[this_time] *= comp_total
        comp_error_vector[this_time] = comp_error * comp_fluxes[error_column_name].unit

    # A comparison star is excluded from its own comparison ensemble: its
    # flux is removed from the comparison total (via flux_offset) and its
    # error is removed from the comparison error, so that the relative flux
    # and its error are computed against the same ensemble.
    comp_total_used = comp_total_vector + flux_offset
    # Clip protects against tiny negative values from floating point
    # roundoff when a single comparison star dominates the ensemble error.
    comp_error_used = np.sqrt(
        np.clip(
            comp_error_vector**2 - (star_data[error_column_name] * is_comp) ** 2,
            0,
            None,
        )
    )

    relative_flux = star_data[counts_column_name] / comp_total_used
    relative_flux = relative_flux.flatten()

    # This is the usual error propagation for a ratio, written to avoid
    # dividing by the star's counts so that a star with exactly zero counts
    # gets a finite error instead of NaN.
    rel_flux_error = np.sqrt(
        (star_data[error_column_name] / comp_total_used) ** 2
        + (star_data[counts_column_name] * comp_error_used / comp_total_used**2) ** 2
    )

    # Add these columns to table
    if not in_place:
        star_data = star_data.copy()

    star_data["relative_flux"] = relative_flux
    star_data["relative_flux_error"] = rel_flux_error
    star_data["relative_flux_snr"] = relative_flux / rel_flux_error

    # AIJ records the total comparison counts even though that total is used
    # only for the targets, not the comparison.
    star_data["comparison counts"] = comp_total_vector  # + flux_offset
    star_data["comparison error"] = comp_error_vector

    return star_data


def add_relative_flux_column(
    photometry_data_file,
    source_list_file,
    add_name="-relative-flux",
    verbose=False,
):
    """
    Add AIJ-style relative flux columns to a photometry data file.

    Parameters
    ----------

    photometry_data_file : str
        Path to the photometry data file.

    source_list_file : str
        Path to the source list file.

    add_name : str,  optional
        String to add to the end of the new file name before the suffix.

    Returns
    -------

    None; writes a file with the relative flux columns added.
    """
    if verbose:
        print("Reading photometry data and source list")
    photometry_data = PhotometryData.read(photometry_data_file)
    source_list = SourceListData.read(source_list_file)
    output_file = photometry_data_file.stem + add_name + photometry_data_file.suffix
    source_list["coord"] = SkyCoord(
        ra=source_list["ra"], dec=source_list["dec"], frame="icrs"
    )
    comp_bool = source_list["marker name"] == ["APASS comparison"]
    only_comp_stars = source_list[comp_bool]
    if verbose:
        print("Adding AIJ-style relative flux columns to photometry data")
    flux_table = calc_aij_relative_flux(photometry_data, only_comp_stars)

    flux_group = flux_table.group_by("file")

    # Add bjd if needed
    if "bjd" not in flux_group.colnames:
        if verbose:
            print("Adding BJD column to photometry data")
        # Accumulate the BJD here
        bjds = []

        for group in flux_group.groups:
            mean_ra = group["ra"].mean()
            mean_dec = group["dec"].mean()
            group.add_bjd_col(bjd_coordinates=SkyCoord(mean_ra, mean_dec))
            bjds.extend(group["bjd"].jd)

        # Each (ephemeral) group had a BJD, this adds the column to the
        # original table.
        flux_group["bjd"] = Time(bjds, scale="tdb", format="jd")

    if verbose:
        print("Writing photometry data with relative flux columns")
    flux_group.write(output_file, overwrite=True)
