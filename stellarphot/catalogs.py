# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Catalog-fetcher factory functions.

These build a :class:`stellarphot.CatalogData` table from an online catalog
(Vizier/astroquery). They live here, separate from :mod:`stellarphot.core`,
because they are data *retrieval* helpers rather than data-structure
definitions. ``core.py`` holds the table classes only.
"""

import logging
import time
import warnings

import numpy as np
from astropy import units as u
from astropy.table import Table, unique
from astroquery.xmatch import XMatch

from .core import CatalogData
from .settings import PassbandMap

__all__ = [
    "apass_dr9",
    "vsx_vizier",
    "refcat2",
]

logger = logging.getLogger(__name__)


def _attach_gaia_ids(catalog, xmatch_result, index_col="_sp_index"):
    """
    Attach Gaia source_ids from an XMatch result to a catalog, joining on an
    index column rather than on row order, which XMatch does not preserve.

    Parameters
    ----------
    catalog : `astropy.table.Table`
        The table the IDs are being attached to. The values of ``index_col``
        in ``xmatch_result`` are row numbers into this table.

    xmatch_result : `astropy.table.Table`
        Result of an XMatch query whose uploaded table included ``index_col``.
        Must contain ``index_col``, ``angDist`` and ``source_id`` columns.
        This table is sorted in place.

    index_col : str, optional
        Name of the column in ``xmatch_result`` holding the row number of the
        matching ``catalog`` row.

    Returns
    -------
    `astropy.table.Table`
        The matched rows of ``catalog``, in their original order, with a new
        ``id`` column holding the Gaia source_id. Rows with no match are
        dropped, with a warning saying how many.
    """
    # When one input row matches several Gaia sources, keep only the nearest.
    xmatch_result.sort([index_col, "angDist"])
    nearest = unique(xmatch_result, keys=index_col, keep="first")

    n_unmatched = len(catalog) - len(nearest)
    if n_unmatched > 0:
        warnings.warn(
            f"{n_unmatched} of {len(catalog)} catalog entries had no Gaia "
            "match and have been dropped.",
            stacklevel=2,
        )

    # nearest[index_col] holds the row numbers of the catalog rows that got a
    # match — one per matched star, sorted ascending. Indexing catalog with it
    # does two things at once: it drops the unmatched rows (their index never
    # appears in the result) and it puts the matched rows in the same order as
    # nearest, so the source_id assignment below lines up row-for-row.
    catalog = catalog[np.asarray(nearest[index_col])]
    catalog["id"] = np.asarray(nearest["source_id"])
    return catalog


def _process_refcat2(catalog):
    """
    Prepare a raw Vizier refcat2 table:

    1. Filter out galaxies from the catalog.
    2. Only keep stars that are in the Gaia DR2 catalog.
    3. Add the Gaia DR2 ID number to the catalog as the ID column.
    """
    # 1.
    # The refcat2 paper says that "Virtually all galaxies can be rejected by
    # selecting objects for which Gaia provides a nonzero proper-motion
    # uncertainty," which in the Vizier download are called e_pmRA and e_pmDE,
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
    #
    # Upload only the coordinates plus a row index; uploading the full
    # 40+ column table makes the query much slower and more likely to fail,
    # and the index is the only reliable way to join the result back to the
    # catalog because XMatch does not preserve row order.
    upload = Table(
        {
            "_sp_index": np.arange(len(catalog)),
            "RA_ICRS": catalog["RA_ICRS"],
            "DE_ICRS": catalog["DE_ICRS"],
        }
    )
    start = time.perf_counter()
    result = XMatch.query(
        cat1=upload,
        cat2="vizier:gaia_dr2_j2015p5",  # "vizier:I/345/gaia2",
        max_distance=0.01 * u.arcsec,
        colRA1="RA_ICRS",
        colDec1="DE_ICRS",
    )
    elapsed = time.perf_counter() - start
    logger.info(
        "XMatch query for Gaia DR2 IDs took %.1f sec "
        "(%d rows uploaded, %d matches returned)",
        elapsed,
        len(upload),
        len(result),
    )

    return _attach_gaia_ids(catalog, result)


def apass_dr9(
    field_center,
    radius=1 * u.degree,
    clip_by_frame=False,
    padding=100,
    magnitude_limit=None,
    magnitude_limit_passband=None,
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

    magnitude_limit_passband : str, optional, default is "V"
        If provided, the passband to use for the magnitude limit. The name of
        the passband must be one of the AAVSO standard passband names.

    Returns
    -------

    `stellarphot.CatalogData`
        Table of catalog information.

    Notes
    -----
    APASS DR9 does not include an identifier column. Though Vizier does provide
    a ``recno`` column, it does not stay the same over time. This function generates
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

    if magnitude_limit is None and magnitude_limit_passband is not None:
        raise ValueError(
            "If you provide a magnitude_limit_passband, you must also "
            "provide a magnitude_limit."
        )
    if magnitude_limit is not None and magnitude_limit_passband is None:
        magnitude_limit_passband = "V"

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

    if magnitude_limit is None:
        # If no magnitude limit is provided, then we will not filter the catalog
        # by magnitude.
        magnitude_limit_passband = None

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

    magnitude_limit_passband : str, optional, default is "max"
        There is no straightforward way to limit the VSX catalog by passband. The
        magnitude limit will be applied to the variable star's magnitude at maximum
        brightness, and the only valid value for this parameter is "max".

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
    magnitude_limit_passband=None,
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

    magnitude_limit_passband : str, optional, default is "SR"
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

    if magnitude_limit is not None and magnitude_limit_passband is None:
        magnitude_limit_passband = "SR"

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

    # if not magnitude_limit:
    #     # If no magnitude limit is provided, then we will not filter the catalog
    #     # by magnitude.
    #     magnitude_limit_passband = None

    raw_catalog = CatalogData.from_vizier(
        field_center,
        "J/ApJ/867/105/refcat2",
        radius=radius,
        clip_by_frame=clip_by_frame,
        padding=padding,
        colname_map=refcat2_colnames,
        prepare_catalog=_process_refcat2,
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
