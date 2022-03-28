from pathlib import Path

import pandas

import ipywidgets as ipw

import numpy as np

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import CCDData
from astrowidgets import ImageWidget

from stellarphot.differential_photometry import *
from stellarphot.photometry import *
from stellarphot.visualization.seeing_profile_functions import set_keybindings


__all__ = ['read_file', 'set_up', 'match', 'mag_scale',
           'in_field', 'make_markers', 'wrap']


def read_file(radec_file):
    """
    Read an AIJ radec file with target and/or comp positions

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
    df = pandas.read_csv(radec_file, names=['RA', 'Dec', 'a', 'b', 'Mag'])
    target_table = Table.from_pandas(df)
    ra = target_table['RA']
    dec = target_table['Dec']
    target_table['coords'] = SkyCoord(ra=ra, dec=dec, unit=(u.hour, u.degree))
    return target_table


def set_up(sample_image_for_finding_stars,
           directory_with_images='.'):
    """
    Find known variable in the field of view and read sample image.

    Parameters
    ----------

    sample_image_for_finding_stars : str
        Name or URL of a fits image of the field of view.

    directory_with_images : str, optional
        Folder in which the image is located. Ignored if the sample image
        is a URL.
    """
    if sample_image_for_finding_stars.startswith('http'):
        path = sample_image_for_finding_stars
    else:
        path = Path(directory_with_images) / sample_image_for_finding_stars

    ccd = CCDData.read(path)
    try:
        vsx = find_known_variables(ccd)
    except RuntimeError:
        vsx = []
    else:
        ra = vsx['RAJ2000']
        dec = vsx['DEJ2000']
        vsx['coords'] = SkyCoord(ra=ra, dec=dec, unit=(u.hour, u.degree))

    return ccd, vsx


def match(CCD, RD, vsx):
    """
    Find APASS stars in FOV and matches APASS stars to VSX and
    APASS to input targets.
    """
    apass, apass_in_bright = find_apass_stars(
        CCD)
    ra = apass['RAJ2000']
    dec = apass['DEJ2000']
    apass['coords'] = SkyCoord(ra=ra, dec=dec, unit=(u.hour, u.degree))
    apass_coord = apass['coords']

    if vsx:
        v_index, v_angle, v_dist = \
            apass_coord.match_to_catalog_sky(vsx['coords'])
    else:
        v_angle = []
    if RD:
        RD_index, RD_angle, RD_dist = \
            apass_coord.match_to_catalog_sky(RD['coords'])
    else:
        RD_angle = []
    return apass, v_angle, RD_angle


def mag_scale(cmag, apass, v_angle, RD_angle,
              brighter_dmag=0.44, dimmer_dmag=0.75):
    """
    Select comparison stars that are 1) not close the VSX stars or to other
    target stars and 2) fall within a particular magnitude range.


    """
    high_mag = apass['r_mag'] < cmag + dimmer_dmag
    low_mag = apass['r_mag'] > cmag - brighter_dmag
    if v_angle:
        good_v_angle = v_angle > 1.0 * u.arcsec
    else:
        good_v_angle = True

    if RD_angle:
        good_RD_angle = RD_angle > 1.0 * u.arcsec
    else:
        good_RD_angle = True

    good_stars = high_mag & low_mag & good_RD_angle & good_v_angle
    good_apass = apass[good_stars]
    apass_good_coord = good_apass['coords']
    return apass_good_coord, good_stars


def in_field(apass_good_coord, ccd, apass, good_stars):
    """
    Return apass stars in the field of view
    """
    apassx, apassy = ccd.wcs.all_world2pix(
        apass_good_coord.ra, apass_good_coord.dec, 0)
    ccdx, ccdy = ccd.shape

    xin = (apassx < ccdx) & (0 < apassx)
    yin = (apassy < ccdy) & (0 < apassy)
    xy_in = xin & yin
    apass_good_coord[xy_in]
    nt = apass[good_stars]
    ent = nt[xy_in]
    return ent


def make_markers(iw, ccd, RD, vsx, ent,
                 name_or_coord=None):
    """
    Add markers for APASS, TESS targets, VSX.
    Also center on object/coordinate.
    """
    iw.load_nddata(ccd)
    iw.zoom_level = 'fit'
    try:
        iw.reset_markers()
    except AttributeError:
        iw.remove_all_markers()

    if RD:
        iw.marker = {'type': 'circle', 'color': 'green', 'radius': 10}
        iw.add_markers(RD, skycoord_colname='coords',
                       use_skycoord=True, marker_name='TESS Targets')

    if name_or_coord is not None:
        if isinstance(name_or_coord, str):
            iw.center_on(SkyCoord.from_name(name_or_coord))
        else:
            iw.center_on(name_or_coord)

    if vsx:
        iw.marker = {'type': 'circle', 'color': 'blue', 'radius': 10}
        iw.add_markers(vsx, skycoord_colname='coords',
                       use_skycoord=True, marker_name='VSX')

    iw.marker = {'type': 'circle', 'color': 'red', 'radius': 10}
    iw.add_markers(ent, skycoord_colname='coords',
                   use_skycoord=True, marker_name='APASS comparison')
    iw.marker = {'type': 'cross', 'color': 'red', 'radius': 6}


def wrap(imagewidget, outputwidget):
    """
    Make the bits that let you click to select/deselect comparisons
    """
    def cb(viewer, event, data_x, data_y):
        i = imagewidget._viewer.get_image()

        try:
            imagewidget.next_elim += 1
        except AttributeError:
            imagewidget.next_elim = 1
        pad = 15
        x = int(np.floor(event.data_x))
        y = int(np.floor(event.data_y))
        ra, dec = i.wcs.wcs.all_pix2world(event.data_x, event.data_y, 0)
        out_skycoord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree))

        try:
            all_table = imagewidget.get_markers(marker_name='all')
        except AttributeError:
            all_table = imagewidget.get_all_markers()

        with outputwidget:
            index, d2d, d3d = out_skycoord.match_to_catalog_sky(
                all_table['coord'])
            if d2d < 10 * u.arcsec:
                mouse = all_table['coord'][index].separation(
                    all_table['coord'])
                rat = mouse < 1 * u.arcsec
                elims = [name for name in all_table['marker name']
                         [rat] if name.startswith('elim')]
                if not elims:
                    imagewidget.add_markers(all_table[rat], skycoord_colname='coord', use_skycoord=True,
                                            marker_name=f'elim{imagewidget.next_elim}')
                else:
                    for elim in elims:
                        imagewidget.remove_markers(marker_name=elim)
            else:
                print('sorry try again')
                imagewidget._viewer.onscreen_message('Click closer to a star')
            print(all_table['marker name'][index])
            print(x, y, ra, dec, out_skycoord)
    return cb


def viewer():
    header = ipw.HTML(value="""
    <h2>Click and drag or use arrow keys to pan, use +/- keys to zoom</h2>
    <h3>Shift-left click (or Crtl-left click)to exclude star as target or comp. Click again to include.</h3>
    """)

    legend = ipw.HTML(value="""
    <h3>Green circles -- Gaia stars within 2.5 arcmin of target</h3>
    <h3>Red circles -- APASS stars within 1 mag of target</h3>
    <h3>Blue circles -- VSX variables</h3>
    <h3>Red × -- Exclude as target or comp</h3>
    """)

    iw = ImageWidget()
    out = ipw.Output()
    set_keybindings(iw)
    bind_map = iw._viewer.get_bindmap()
    gvc = iw._viewer.get_canvas()
    bind_map.map_event(None, ('shift',), 'ms_left', 'cursor')
    gvc.add_callback('cursor-down', wrap(iw, out))


    box = ipw.VBox()
    inner_box = ipw.HBox()
    inner_box.children = [iw, legend]
    box.children = [header, inner_box]

    return box, iw
