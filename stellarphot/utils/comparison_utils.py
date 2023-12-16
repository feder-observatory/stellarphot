from pathlib import Path

import pandas as pd

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.nddata import CCDData
from astropy import units as u

from stellarphot.photometry import *
from stellarphot import apass_dr9, vsx_vizier


__all__ = ["read_file", "set_up", "crossmatch_APASS2VSX", "mag_scale", "in_field"]

DESC_STYLE = {"description_width": "initial"}


def read_file(radec_file):
    """
    Read an AIJ radec file with target and/or comparison positions

    Parameters
    ----------

    radec_file : str
        Name of the file

    Returns
    -------

    `astropy.table.Table`
        Table with target information, including a
        `astropy.coordinates.SkyCoord` column.
    """
    df = pd.read_csv(radec_file, names=["RA", "Dec", "a", "b", "Mag"])
    target_table = Table.from_pandas(df)
    ra = target_table["RA"]
    dec = target_table["Dec"]
    target_table["coords"] = SkyCoord(ra=ra, dec=dec, unit=(u.hour, u.degree))
    return target_table


def set_up(sample_image_for_finding_stars, directory_with_images="."):
    """
    Read in sample image and find known variables in the field of view.

    Parameters
    ----------

    sample_image_for_finding_stars : str
        Name or URL of a FITS image of the field of view.

    directory_with_images : str, optional
        Folder in which the image is located. Ignored if the sample image
        is a URL.

    Returns
    -------

    ccd: `astropy.nddata.CCDData`
        Sample image.

    vsx: `astropy.table.Table`
        Table with known variables in the field of view.

    """
    if sample_image_for_finding_stars.startswith("http"):
        path = sample_image_for_finding_stars
    else:
        path = Path(directory_with_images) / sample_image_for_finding_stars

    ccd = CCDData.read(path)
    try:
        vsx = vsx_vizier(ccd.header, radius=0.5 * u.degree)
    except RuntimeError:
        vsx = []
    else:
        ra = vsx["RAJ2000"]
        dec = vsx["DEJ2000"]
        vsx["coords"] = SkyCoord(ra=ra, dec=dec, unit=(u.hour, u.degree))

    return ccd, vsx


def crossmatch_APASS2VSX(CCD, RD, vsx):
    """
    Find APASS stars in FOV and matches APASS stars to VSX and APASS to input targets.

    Parameters
    ----------

    CCD : `astropy.nddata.CCDData`
        Sample image.

    RD : `astropy.table.Table`
        Table with target information, including a
        `astropy.coordinates.SkyCoord` column.

    vsx : `astropy.table.Table`
        Table with known variables in the field of view.

    Returns
    -------

    apass : `astropy.table.Table`
        Table with APASS stars in the field of view.

    v_angle : `astropy.units.Quantity`
        Angular separation between APASS stars and VSX stars.

    RD_angle : `astropy.units.Quantity`
        Angular separation between APASS stars and input targets.
    """
    apass = apass_dr9(CCD)
    ra = apass["ra"]
    dec = apass["dec"]
    apass["coords"] = SkyCoord(ra=ra, dec=dec, unit=(u.hour, u.degree))
    apass_coord = apass["coords"]

    if vsx:
        v_index, v_angle, v_dist = apass_coord.match_to_catalog_sky(vsx["coords"])
    else:
        v_angle = []

    if RD:
        RD_index, RD_angle, RD_dist = apass_coord.match_to_catalog_sky(RD["coords"])
    else:
        RD_angle = []

    return apass, v_angle, RD_angle


def mag_scale(cmag, apass, v_angle, RD_angle, brighter_dmag=0.44, dimmer_dmag=0.75):
    """
    Select comparison stars that are 1) not close the VSX stars or to other
    target stars and 2) fall within a particular magnitude range.

    Parameters
    ----------

    cmag : float
        Magnitude of the target star.

    apass : `astropy.table.Table`
        Table with APASS stars in the field of view.

    v_angle : `astropy.units.Quantity`
        Angular separation between APASS stars and VSX stars.

    RD_angle : `astropy.units.Quantity`
        Angular separation between APASS stars and input targets.

    brighter_dmag : float, optional
        Maximum difference in magnitude between the target and comparison stars.

    dimmer_dmag : float, optional
        Minimum difference in magnitude between the target and comparison stars.

    Returns
    -------

    apass_good_coord : `astropy.coordinates.SkyCoord`
        Coordinates of the comparison stars.

    good_stars : `astropy.table.Table`
        Table with the comparison stars.
    """
    high_mag = apass["r_mag"] < cmag + dimmer_dmag
    low_mag = apass["r_mag"] > cmag - brighter_dmag
    if len(v_angle) > 0:
        good_v_angle = v_angle > 1.0 * u.arcsec
    else:
        good_v_angle = True

    if len(RD_angle) > 0:
        good_RD_angle = RD_angle > 1.0 * u.arcsec
    else:
        good_RD_angle = True

    good_stars = high_mag & low_mag & good_RD_angle & good_v_angle
    good_apass = apass[good_stars]
    apass_good_coord = good_apass["coords"]
    return apass_good_coord, good_stars


def in_field(apass_good_coord, ccd, apass, good_stars):
    """
    Return APASS stars in the field of view.

    Parameters
    ----------

    apass_good_coord : `astropy.coordinates.SkyCoord`
        Coordinates of the comparison stars.

    ccd : `astropy.nddata.CCDData`
        Sample image.

    apass : `astropy.table.Table`
        Table with APASS stars in the field of view.

    good_stars : `astropy.table.Table`
        Table with the comparison stars.

    Returns
    -------

    ent : `astropy.table.Table`
        Table with APASS stars in the field of view.
    """
    apassx, apassy = ccd.wcs.all_world2pix(apass_good_coord.ra, apass_good_coord.dec, 0)
    ccdx, ccdy = ccd.shape

    xin = (apassx < ccdx) & (0 < apassx)
    yin = (apassy < ccdy) & (0 < apassy)
    xy_in = xin & yin
    apass_good_coord[xy_in]
    nt = apass[good_stars]
    ent = nt[xy_in]
    return ent
