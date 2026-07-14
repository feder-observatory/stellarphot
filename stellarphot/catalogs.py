# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Catalog-fetcher factory functions.

These build a :class:`stellarphot.CatalogData` table from an online catalog
(Vizier/astroquery). They live here, separate from :mod:`stellarphot.core`,
because they are data *retrieval* helpers rather than data-structure
definitions. ``core.py`` holds the table classes only.
"""

import numpy as np
from astropy import units as u

from .core import CatalogData
from .settings import PassbandMap

__all__ = [
    "apass_dr9",
    "vsx_vizier",
    "refcat2",
]


def _iau_designation_ids(acronym, ra_deg, dec_deg):
    """
    Build coordinate-based identifiers following the guidelines in the
    `IAU designation specification <https://cds.unistra.fr/Dic/iau-spec.html>`_.

    Parameters
    ----------
    acronym : str
        Acronym the designations start with, e.g. ``"APASSSP"``.

    ra_deg, dec_deg : array-like of float
        Right ascension and declination in degrees.

    Returns
    -------
    list of str
        One designation per coordinate, e.g. ``"APASSSP J+359.9896+00.0122"``.

    Notes
    -----
    The formats below include 4 digits after the decimal point (accuracy of
    about 0.5 arcsec), a leading sign (+ or -) and leading zeros so that the
    RA is always three digits before the decimal and the Dec is always two
    digits before the decimal. IAU says there is a space between the acronym
    and the coordinates.
    """
    return [
        f"{acronym} J{ra:0=+9.4f}{dec:0=+8.4f}"
        for ra, dec in zip(ra_deg, dec_deg, strict=True)
    ]


def _process_refcat2(catalog):
    """
    Prepare a raw Vizier refcat2 table:

    1. Filter out galaxies from the catalog.
    2. Only keep stars that are in the Gaia DR2 catalog.
    3. Add a coordinate-based designation to the catalog as the ID column.
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
    # Refcat2 has no ID column, so generate an IAU-style designation from the
    # coordinates, as apass_dr9 does. Nothing downstream needs the ID to be a
    # Gaia ID, and generating it locally avoids a crossmatch against a service
    # (CDS XMatch) that has no mirrors, unlike the Vizier queries.
    catalog["id"] = _iau_designation_ids(
        "REFCAT2SP",
        np.asarray(catalog["RA_ICRS"]),
        np.asarray(catalog["DE_ICRS"]),
    )

    return catalog


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

    # IAU requires an acronym to start, so make it APASS plus SP for stellarphot
    raw_catalog["id"] = _iau_designation_ids(
        "APASSSP",
        raw_catalog["ra"].to("degree").value,
        raw_catalog["dec"].to("degree").value,
    )

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
    Refcat2 does not include an identifier column. This function generates
    an ID based on the coordinates of the refcat2 star, following the
    guidelines in the
    `IAU designation specification <https://cds.unistra.fr/Dic/iau-spec.html>`_.

    The reference for the refcat2 paper is:

    Tonry, J. L., Denneau, L., Flewelling, H., et al. 2018, ApJ, 867,
    https://iopscience.iop.org/article/10.3847/1538-4357/aae386
    """
    refcat2_colnames = {
        # There is no refcat2 ID number; a coordinate-based designation is
        # generated in _process_refcat2 and used as the ID.
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
