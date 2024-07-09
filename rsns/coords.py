from glob import glob
from astropy import units as u
import numpy as np

import copy
# import edge detection, erosion, dilation
from skimage import filters
from skimage.morphology import binary_erosion, binary_dilation, disk
from skimage.measure import label, regionprops

from astropy.io import fits
from casatools import msmetadata, ms, table, quanta, measures

import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time

import sunpy.data.sample
import sunpy.map
from sunpy.coordinates import frames, sun


def j2000xy(RA, DEC, t_sun):
    [RA_sun, DEC_sun] = sun.sky_position(t_sun, False)
    rotate_angel = sun.P(t_sun)

    # shift the center and transfer into arcsec

    y_shift = (DEC - DEC_sun.degree) * np.pi / 180
    x_shift = -(RA - RA_sun.degree) * np.pi / 180 * np.cos(DEC_sun.degree * np.pi / 180)


    # rotate xy according to the position angle
    xx =(x_shift) * np.cos(-rotate_angel.rad) - (y_shift) * \
        np.sin(-rotate_angel.rad)
    yy = (x_shift) * np.sin(-rotate_angel.rad) + (y_shift) * \
        np.cos(-rotate_angel.rad)

    xx = xx * 180 / np.pi * 3600
    yy = yy * 180 / np.pi * 3600
    return [xx, yy]

def radec_fits_to_helio(fits_in, helio_sunpy_fits_name =None, obs_loc = EarthLocation(lat=37.232259*u.deg, lon=-118.28479*u.deg),
                        fov=None):

    hdu = fits.open(fits_in)
    header = hdu[0].header
    data = hdu[0].data[0, 0, :, :]
    obstime = Time(header['date-obs'])
    frequency = header['crval3']*u.Hz
    obs_gcrs = SkyCoord(obs_loc.get_gcrs(obstime))
    reference_coord = SkyCoord(header['crval1']*u.Unit(header['cunit1']),
            header['crval2']*u.Unit(header['cunit2']),frame='gcrs',obstime=obstime,
            obsgeoloc=obs_gcrs.cartesian, obsgeovel=obs_gcrs.velocity.to_cartesian(),
            distance=obs_gcrs.hcrs.distance)
    reference_coord_arcsec = reference_coord.transform_to(frames.Helioprojective(observer=obs_gcrs))
    cdelt1 = (np.abs(header['cdelt1'])*u.deg).to(u.arcsec)
    cdelt2 = (np.abs(header['cdelt2'])*u.deg).to(u.arcsec)
    P1 = sun.P(obstime)
    new_header = sunpy.map.make_fitswcs_header(data, reference_coord_arcsec,
        reference_pixel=u.Quantity([header['crpix1']-1, header['crpix2']-1]*u.pixel),
        scale=u.Quantity([cdelt1, cdelt2]*u.arcsec/u.pix), wavelength=3e8/frequency.to(u.Hz).value*u.m,
        rotation_angle=-P1, observatory='obs')
    obsview_map = sunpy.map.Map(data, new_header)
    
    if fov is None:
        fov_x = new_header["naxis1"]*cdelt1
        fov_y = new_header["naxis2"]*cdelt2
    else:
        fov_x = fov[0]
        fov_y = fov[1]

    obsview_map_rotate = obsview_map.rotate()
    bl = SkyCoord(-fov_x/2, -fov_y/2, frame=obsview_map_rotate.coordinate_frame)
    tr = SkyCoord(fov_x/2,  fov_y/2,  frame=obsview_map_rotate.coordinate_frame)
    obsview_submap = obsview_map_rotate.submap(bl, top_right=tr)

    if helio_sunpy_fits_name is None:
        helio_sunpy_fits_name = fits_in.replace('.fits', '_heliosunpy.fits')
    obsview_submap.save(helio_sunpy_fits_name, overwrite=True)

    return helio_sunpy_fits_name