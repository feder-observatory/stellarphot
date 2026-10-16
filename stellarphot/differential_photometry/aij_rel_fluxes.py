import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import QTable, Table
from astropy.time import Time
from astropy.utils.exceptions import AstropyDeprecationWarning, AstropyUserWarning

from stellarphot import PhotometryData, SourceListData
from stellarphot.utils.magnitude_transforms import _validate_angle

__all__ = ["add_in_quadrature", "calc_aij_relative_flux", "add_relative_flux_column"]

# Default of the ``position_check_limit`` parameter of
# `calc_aij_relative_flux`: the largest separation between a comparison
# star's position and the input position of the same star in the photometry
# data before the comparison star table is judged not to be from the source
# list the photometry was made from. The same as the default ``match_radius``
# of `~stellarphot.utils.magnitude_transforms.transform_to_catalog`.
_POSITION_CHECK_LIMIT = 2 * u.arcsec


def add_in_quadrature(array):
    """
    Add an array of numbers in quadrature.
    """
    return np.sqrt((array**2).sum())


def _sky_coords(table, ra_column="ra", dec_column="dec"):
    """
    Make sky coordinates from two columns of a table.

    Parameters
    ----------

    table : `astropy.table.Table`
        Table containing the coordinate columns.

    ra_column : str, optional
        Name of the column containing the right ascension.

    dec_column : str, optional
        Name of the column containing the declination.

    Returns
    -------

    `astropy.coordinates.SkyCoord`
        Coordinates from the two columns.

    Notes
    -----

    Columns with no unit are taken to be in degrees. When the columns do
    have units, ``unit`` is passed as ``[None, None]`` because `SkyCoord`
    wants one entry per coordinate.
    """
    if table[ra_column].unit is None:
        unit = "degree"
    else:
        unit = [None, None]
    return SkyCoord(ra=table[ra_column], dec=table[dec_column], unit=unit)


def _comp_rows_by_id(
    star_data, comp_stars, star_id_column, coord_column, position_check_limit
):
    """
    Find the rows of the photometry that are comparison stars, by star id.

    Parameters
    ----------

    star_data : `stellarphot.PhotometryData`
        Photometry data from one or more images.

    comp_stars : `astropy.table.Table`
        Table of comparison stars, which must have the column named by
        ``star_id_column``.

    star_id_column : str
        Name of the column with the star id.

    coord_column : str or None
        Name of the column in ``comp_stars`` with the coordinates. If
        ``None``, the ``ra`` and ``dec`` columns are used.

    position_check_limit : `astropy.units.Quantity` or None
        Largest separation allowed between a comparison star's position and
        the input position of the same star in ``star_data``. If ``None``,
        the positions are not checked.

    Returns
    -------

    `numpy.ndarray`
        Boolean mask, ``True`` for the rows of ``star_data`` that are
        comparison stars.

    Raises
    ------

    ValueError
        If a star id appears more than once in ``comp_stars``, or if a
        comparison star's position disagrees with the input position of the
        star with the same id in ``star_data``.

    Notes
    -----

    The ids agree by construction because the photometry code copies them
    from the source list, so the position comparison is only a check that
    ``comp_stars`` came from the source list the photometry was made from.
    It is made once per comparison star, since a star's input position is
    the same in every image. The check is skipped, with a warning, for a
    star whose position is NaN in either table and when ``comp_stars`` has
    no coordinates at all.
    """
    comp_ids = np.asarray(comp_stars[star_id_column])
    unique_ids, counts = np.unique(comp_ids, return_counts=True)
    if np.any(counts > 1):
        raise ValueError(
            f"comp_stars has more than one row for {star_id_column} "
            f"{unique_ids[counts > 1].tolist()}; each comparison star must "
            "appear exactly once."
        )

    star_ids = np.asarray(star_data[star_id_column])
    is_comp = np.isin(star_ids, comp_ids)

    if not np.any(is_comp):
        # The caller raises for this.
        return is_comp

    missing = np.setdiff1d(comp_ids, star_ids)
    if len(missing):
        warnings.warn(
            f"These comparison stars have no photometry: {missing.tolist()}. "
            "They are not part of the comparison ensemble.",
            AstropyUserWarning,
            stacklevel=3,
        )

    if position_check_limit is None:
        return is_comp

    if coord_column is not None:
        comp_coords = comp_stars[coord_column]
    elif {"ra", "dec"} <= set(comp_stars.colnames):
        comp_coords = _sky_coords(comp_stars)
    else:
        warnings.warn(
            "comp_stars has no coordinates, so the comparison star positions "
            "were not checked against the input positions in the photometry "
            "data.",
            AstropyUserWarning,
            stacklevel=3,
        )
        return is_comp

    # One photometry row per comparison star id. np.unique returns the ids
    # sorted, so both sides end up in the same order.
    ids_checked, first = np.unique(star_ids[is_comp], return_index=True)
    star_rows = np.flatnonzero(is_comp)[first]

    order = np.argsort(comp_ids)
    comp_rows = order[np.searchsorted(comp_ids[order], ids_checked)]

    checked = star_data[star_rows]
    if {"ra_input", "dec_input"} <= set(star_data.colnames):
        star_coords = _sky_coords(checked, "ra_input", "dec_input")
    else:
        star_coords = _sky_coords(checked)

    separation = star_coords.separation(comp_coords[comp_rows])

    unchecked = np.isnan(separation)
    if np.any(unchecked):
        warnings.warn(
            f"The positions of comparison star(s) {star_id_column} "
            f"{ids_checked[unchecked].tolist()} could not be checked against "
            "the photometry data because the position in comp_stars or the "
            "input position in the photometry data is NaN.",
            AstropyUserWarning,
            stacklevel=3,
        )

    mismatched = separation > position_check_limit
    if np.any(mismatched):
        details = ", ".join(
            f"{star_id_column} {star_id} ({sep.to_value(u.arcsec):.3g} arcsec)"
            for star_id, sep in zip(
                ids_checked[mismatched], separation[mismatched], strict=True
            )
        )
        raise ValueError(
            f"The positions of comparison star(s) {details} in comp_stars are "
            f"more than {position_check_limit.to_value(u.arcsec):.3g} arcsec "
            "from the input positions in the photometry data for the same "
            "star, so the source list does not appear to be the one the "
            "photometry was made from."
        )

    return is_comp


def _comp_rows_by_position(star_data, comp_stars, star_id_column, coord_column):
    """
    Find the rows of the photometry that are comparison stars, by position.

    Deprecated; kept for comparison star tables that have no star id column.

    Parameters
    ----------

    star_data : `stellarphot.PhotometryData`
        Photometry data from one or more images.

    comp_stars : `astropy.table.Table`
        Table of comparison stars.

    star_id_column : str
        Name of the column in ``star_data`` with the star id.

    coord_column : str or None
        Name of the column in ``comp_stars`` with the coordinates. If
        ``None``, the ``ra`` and ``dec`` columns are used.

    Returns
    -------

    good : `numpy.ndarray`
        Boolean mask, ``True`` for the rows of ``star_data`` whose position
        matches a comparison star.

    position_bad : set
        Ids of the stars whose position does not match a comparison star at
        one or more times. The caller excludes these at every time.
    """
    warnings.warn(
        "Matching comparison stars by position is deprecated; comp_stars "
        f"should have a '{star_id_column}' column, as a stellarphot source "
        "list does. Deprecated since stellarphot 2.2.0; position matching "
        "will be removed in 3.0.0.",
        AstropyDeprecationWarning,
        stacklevel=3,
    )
    if coord_column is not None:
        comp_coords = comp_stars[coord_column]
    else:
        comp_coords = _sky_coords(comp_stars)
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
    position_bad = set(is_all_good["star_id"][~is_all_good["good"]])

    return good, position_bad


def calc_aij_relative_flux(
    star_data,
    comp_stars,
    in_place=True,
    coord_column=None,
    star_id_column="star_id",
    counts_column_name="aperture_net_cnts",
    position_check_limit=_POSITION_CHECK_LIMIT,
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
        and should contain either the column named by ``coord_column`` or
        columns called ``ra`` and ``dec``, which are used only to check
        that the table is from the right source list. A table without
        ``star_id_column`` is matched by position instead, which is
        deprecated. Not all of the comparison stars will necessarily be
        used; see Notes.

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

    position_check_limit : `astropy.units.Quantity` or None, optional
        Largest separation allowed between a comparison star's position and
        the input position of the star with the same id in ``star_data``;
        must be a single positive, finite angle. ``None`` skips the check.
        Defaults to 2 arcsec.

    Returns
    -------

    `stellarphot.PhotometryData` or None
        The return type depends on the value of ``in_place``. If it is
        ``False``, then the new columns are returned as a separate table,
        otherwise the columns are simply added to the input table.

    Raises
    ------

    ValueError
        If ``position_check_limit`` is not a positive, finite angle, if a
        star id appears more than once in ``comp_stars``, or if the position
        of a comparison star in ``comp_stars`` is more than
        ``position_check_limit`` from the input position of the star with
        the same id in ``star_data``.

    RuntimeError
        If no comparison star is in ``star_data``, if there is a time at
        which no comparison star has valid data, or if the number of
        comparison stars is not the same at every time.

    Notes
    -----

    Comparison stars are matched to ``star_data`` by star id; the photometry
    code copies the ids from the source list, so they agree by construction.
    The positions are only checked, not matched on: if a comparison star's
    position differs from the input position (``ra_input``/``dec_input`` if
    present, otherwise ``ra``/``dec``) of the star with the same id by more
    than ``position_check_limit``, ``comp_stars`` is not from the source list
    the photometry was made from and a `ValueError` is raised. A comparison
    star with no photometry at all is left out of the comparison set with a
    warning.

    A comparison star is excluded from the comparison set at every time if, at
    any one time, it is missing from ``star_data``, its net counts are ``NaN``,
    or -- in the deprecated matching-by-position path only -- its position does
    not match.
    """
    if position_check_limit is not None:
        position_check_limit = _validate_angle(
            "position_check_limit",
            position_check_limit,
            "Pass None to skip the position check.",
        )

    if star_id_column in comp_stars.colnames:
        good = _comp_rows_by_id(
            star_data, comp_stars, star_id_column, coord_column, position_check_limit
        )
        position_bad = set()
    else:
        good, position_bad = _comp_rows_by_position(
            star_data, comp_stars, star_id_column, coord_column
        )

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
    bad_comps = set.union(*star_id_sets) - good_ids

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
    bad_comps = bad_comps | position_bad

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
            "time if it is missing or has NaN counts at even one time, or -- "
            "when comparison stars are matched by position -- if its position "
            "does not match at even one time."
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
