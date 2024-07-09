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
from scipy.optimize import curve_fit

def func_gaussian(xdata, s0, x_cent, y_cent, tile, x_sig, y_sig):
    x, y = xdata
    xp =  (x-x_cent) * np.cos(tile) + (y-y_cent) * np.sin(tile)
    yp = -(x-x_cent) * np.sin(tile) + (y-y_cent) * np.cos(tile)
    flux = s0 * (np.exp(-(xp**2)/(2*x_sig**2) - (yp**2)/(2*y_sig**2)))
    return flux

def peak_xy_coord_arcsec(data_image, xx, yy):
    peak_xy = np.unravel_index(np.argmax(data_image, axis=None), data_image.shape)
    peak_xy_arcsec = [xx[peak_xy[0]], yy[peak_xy[1]]]
    return peak_xy_arcsec

# fit the image source to a 2-d gaussian funtion with sciPy's curve_fit
def fit_gaussian2d(data_image, xx, yy, thresh=0.3 , cbox = [[0,0],[0,0]] , transpose=True,  **kwargs):

    if transpose:
        data_image = data_image.T
    idx_wanted = np.where(data_image >
                          thresh*data_image.max())
    yv, xv = np.meshgrid(yy, xx)
    boundthis = [
        [0, np.min(xv[idx_wanted]), np.min(
            yv[idx_wanted]), -1.1*np.pi, 0, 0],
        [5*np.max(data_image), np.max(xv[idx_wanted]), np.max(yv[idx_wanted]), 1.1*np.pi,
         np.max(np.abs(xv))/2, np.max(np.abs(yv))/2]]
    
    if cbox != [[0,0],[0,0]]:
        boundthis[0][1:3] = [cbox[0][0],cbox[1][0]]
        boundthis[1][1:3] = [cbox[0][1],cbox[1][1]]
        coord_x, coord_y = np.mean(cbox[0]), np.mean(cbox[1])
        idx_wanted = np.where((xv > cbox[0][0]) & (xv < cbox[0][1]) 
                              & (yv > cbox[1][0]) & (yv < cbox[1][1]))
    else:
        coord_x, coord_y = peak_xy_coord_arcsec(data_image, xx, yy)
    
    popt, pcov = curve_fit(func_gaussian, (xv[idx_wanted], yv[idx_wanted]),
                            data_image[idx_wanted],
                            p0=[np.max(data_image), coord_x,
                            coord_y, 3, 10, 10], bounds=boundthis)
    
    # y_sig as major axis, x_sig as minor axis
    if popt[4] > popt[5]:
        popt[4], popt[5] = popt[5], popt[4]
        popt[3] = popt[3] + np.pi/2
        pcov[4,4], pcov[5,5] = pcov[5,5], pcov[4,4]
        
    return popt, pcov


def angular_guassian(xdata, A, R0, theta0,sig_R, sig_theta ):
    x, y = xdata
    R = np.sqrt(x**2 + y**2)
    Theta = np.arctan2(y, x)
    angular_dist  = Theta - theta0
    angular_dist[angular_dist > np.pi] -= 2*np.pi
    angular_dist[angular_dist < -np.pi] += 2*np.pi
    dist_func = A * np.exp(- (R-R0)**2 / (2*sig_R**2) - (angular_dist)**2 / (2*sig_theta**2) )
    return dist_func

def fit_angular_gaussian(data_image, xx, yy, thresh=0.3 , cbox = [[0,0],[0,0]] , transpose=True,  **kwargs):

    if transpose:
        data_image = data_image.T
    idx_wanted = np.where(data_image >
                          thresh*data_image.max())
    yv, xv = np.meshgrid(yy, xx)

    R = np.sqrt(xv**2 + yv**2)
    R_range = [np.min(R), np.max(R)]
    Theta = np.arctan2(yv, xv)
    Theta_range = [np.min(Theta), np.max(Theta)]
    boundthis = [
        [0,                    R_range[0], Theta_range[0], np.mean(np.diff(xx)), 0],
        [2*np.max(data_image), R_range[1], Theta_range[1], R_range[1]-R_range[0] , Theta_range[1]-Theta_range[0]]]
    if cbox != [[0,0],[0,0]]:
        boundthis[0][1:3] = [cbox[0][0],cbox[1][0]]
        boundthis[1][1:3] = [cbox[0][1],cbox[1][1]]
        coord_x, coord_y = np.mean(cbox[0]), np.mean(cbox[1])
        idx_wanted = np.where((xv > cbox[0][0]) & (xv < cbox[0][1]) 
                              & (yv > cbox[1][0]) & (yv < cbox[1][1]))
    else:
        coord_x, coord_y = peak_xy_coord_arcsec(data_image, xx, yy)
    R0_start = np.sqrt(coord_x**2 + coord_y**2)
    theta0_start = np.arctan2(coord_y, coord_x)

    popt, pcov = curve_fit(angular_guassian, (xv[idx_wanted], yv[idx_wanted]),
                            data_image[idx_wanted],
                            p0=[np.max(data_image), R0_start, theta0_start, np.std(R), np.std(Theta)], bounds=boundthis)
    
    return popt, pcov