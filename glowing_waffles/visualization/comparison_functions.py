import pandas
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import CCDData
from pathlib import Path
from glowing_waffles.differential_photometry import *
from astrowidgets import ImageWidget
from glowing_waffles.photometry import *
import numpy as np

def read_file(radec_file):
    df = pandas.read_csv(radec_file, names=['RA','Dec','a','b','Mag'])
    RD = Table.from_pandas(df)
    ra = RD['RA']
    dec = RD['Dec']
    RD['coords']= SkyCoord(ra=ra, dec=dec,unit=(u.hour, u.degree))
    return df, RD, ra, dec, RD['coords']

def set_up(object_name, directory_with_images, sample_image_for_finding_stars):
    if object_name:
        obj_coords = SkyCoord.from_name(object_name)
    if sample_image_for_finding_stars.startswith('http'):
        path = sample_image_for_finding_stars
    else:
        path = Path(directory_with_images) / sample_image_for_finding_stars
    ccd = CCDData.read(path)
    vsx, vsx_x, vsx_y, vsx_names = find_known_variables(ccd)
    ra = vsx['RAJ2000']
    dec = vsx['DEJ2000']
    vsx['coords']= SkyCoord(ra=ra, dec=dec,unit=(u.hour, u.degree))
    ccdx,ccdy = ccd.shape
    return ccd, ccdx, ccdy, vsx, ra, dec, vsx['coords'], vsx_names

def match(CCD, RD, vsx):
    apass, apass_x, apass_y, apass_in_bright, in_apass_x, in_apass_y = find_apass_stars(CCD)
    ra = apass['RAJ2000']
    dec = apass['DEJ2000']
    apass['coords']= SkyCoord(ra=ra, dec=dec,unit=(u.hour, u.degree))
    vsx_coord = vsx['coords']
    RD_coord = RD['coords']
    apass_coord = apass['coords']
    v_index, v_angle, v_dist = apass_coord.match_to_catalog_sky(vsx['coords'])
    RD_index, RD_angle, RD_dist = apass_coord.match_to_catalog_sky(RD['coords'])
    return apass, v_angle, RD_angle

def mag_scale(cmag, apass, v_angle, RD_angle):
    high_mag = apass['r_mag'] < cmag+0.75
    low_mag = apass['r_mag'] > cmag-0.44
    good_v_angle = v_angle > 1.0 * u.arcsec
    good_RD_angle = RD_angle > 1.0 * u.arcsec
    good_stars = high_mag & low_mag & good_RD_angle & good_v_angle
    good_apass = apass[good_stars]
    apass_good_coord = good_apass['coords']
    return apass_good_coord, good_stars

def in_field(apass_good_coord, ccd, ccdx, ccdy, apass, good_stars):
    apassx, apassy = ccd.wcs.all_world2pix(apass_good_coord.ra,apass_good_coord.dec,0)
    xin = (apassx < ccdx) & (0 < apassx)
    yin = (apassy < ccdy) & (0 < apassy)
    xy_in = xin & yin
    apass_good_coord[xy_in]
    nt = apass[good_stars]
    ent = nt[xy_in]
    return ent

def make_markers(iw, image_directory, sample_image, RD, vsx, ent, object_name):
    if sample_image.startswith('http'):
        path = sample_image
    else:
        path = Path(image_directory) / sample_image
    iw.load_fits(str(path))
    iw.zoom_level = 'fit'
    iw.reset_markers()
    iw.marker = {'type': 'circle', 'color': 'lightgreen', 'radius': 10}
    iw.add_markers(RD, skycoord_colname='coords', use_skycoord=True, marker_name='TESS Targets')
    if object_name:
        iw.center_on(SkyCoord.from_name(object_name))
    iw.marker = {'type': 'circle', 'color': 'red', 'radius': 10}
    iw.add_markers(vsx, skycoord_colname='coords', use_skycoord=True, marker_name='VSX')
    iw.marker = {'type': 'circle', 'color': 'blue', 'radius': 10}
    iw.add_markers(ent, skycoord_colname='coords', use_skycoord=True, marker_name='APASS comparison')
    iw.marker = {'type': 'cross', 'color': 'red', 'radius': 6}
    iw.get_markers(marker_name='APASS comparison')

def wrap(imagewidget, outputwidget):
    def cb(viewer, event, data_x, data_y):
        i = imagewidget._viewer.get_image()
        data = i.get_data()
        try:
            imagewidget.next_elim += 1
        except AttributeError:
            imagewidget.next_elim = 1
        pad = 15
        x = int(np.floor(event.data_x))
        y = int(np.floor(event.data_y))
        ra, dec = i.wcs.wcs.all_pix2world(event.data_x, event.data_y, 0)
        out_skycoord = SkyCoord(ra=ra, dec=dec,unit=(u.degree, u.degree))
        all_table = imagewidget.get_markers(marker_name='all')
        with outputwidget:
            index, d2d, d3d = out_skycoord.match_to_catalog_sky(all_table['coord'])
            if d2d < 10 * u.arcsec:
                mouse = all_table['coord'][index].separation(all_table['coord'])
                rat = mouse < 1 * u.arcsec
                elims = [name for name in all_table['marker name'][rat] if name.startswith('elim')]
                if not elims:
                    imagewidget.add_markers(all_table[rat], skycoord_colname='coord', use_skycoord=True, 
                                            marker_name=f'elim{imagewidget.next_elim}')
                else:
                    for elim in elims:
                        imagewidget.remove_markers(marker_name=elim)
            else:
                print('sorry try again 🤮')
                imagewidget._viewer.onscreen_message('🤮') 
            print(all_table['marker name'][index])
            print(x,y,ra,dec,out_skycoord)
    return cb






