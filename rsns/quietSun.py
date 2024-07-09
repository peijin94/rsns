

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
# keep only the largest connected component
from skimage import measure
from skimage.morphology import remove_small_objects
from scipy.ndimage import center_of_mass
from skimage.morphology import convex_hull_image
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing
import sunpy.map as smap



def thresh_func(freq):  # freq in Hz
    """Return the threshold for the given frequency
    
    :param freq: frequency in Hz

    :return: threshold in Tb
    """

    return 1.1e6 * (1 - 1.8e4 * freq ** (-0.6))


def find_center_of_thresh(data_this, thresh, meta_header,
                          freq=80e6, min_size_50=1000,convex_hull=False):
    """
    Find the center of the thresholded image
    
    :param data_this: the data to be thresholded
    :param thresh: the threshold
    :param meta: the meta data of the fits file
    :param index: the index of the image contained in the fits file
    :param min_size_50: The smallest allowable object area, in pixels, at 50 MHz. min_size will scale with 1/(nu[MHz]/50MHz)**2.
    
    """
    threshed_img = (data_this > thresh)
    min_size = min_size_50/(meta_header['CDELT1']/60.)**2./(freq/50e6)**2.
    threshed_img_1st = remove_small_objects(threshed_img, min_size=min_size, connectivity=1)
    # perform erosion to remove the small features
    threshed_img_2nd = binary_erosion(threshed_img_1st, iterations=3)

    # keep only the largest connected component
    threshed_img_3rd = remove_small_objects(threshed_img_2nd, min_size=min_size, connectivity=1)

    # dialate the image back to the original size
    threshed_img_4th = binary_dilation(threshed_img_3rd, iterations=3)

    if convex_hull:
        threshed_img_4th = convex_hull_image(threshed_img_4th)

    # find the centroid of threshed_img_1st, coords in x_arr, y_arr
    com = center_of_mass(threshed_img_4th)
    # convert to arcsec

    x_arr = meta_header['CRVAL1'] + meta_header['CDELT1'] * (np.arange(meta_header['NAXIS1']) - (meta_header['CRPIX1'] - 1))
    y_arr = meta_header['CRVAL2'] + meta_header['CDELT2'] * (np.arange(meta_header['NAXIS2']) - (meta_header['CRPIX2'] - 1))

    # convert com from pixel to arcsec (linear)
    com_x_arcsec = x_arr[0] + com[1] * (x_arr[-1] - x_arr[0]) / (len(x_arr) - 1)
    com_y_arcsec = y_arr[0] + com[0] * (y_arr[-1] - y_arr[0]) / (len(y_arr) - 1)

    # move x_arr, y_arr to the center of the image
    x_arr_new = x_arr - com_x_arcsec
    y_arr_new = y_arr - com_y_arcsec

    return [com_x_arcsec, com_y_arcsec, com, x_arr_new, y_arr_new, threshed_img,
            threshed_img_1st, threshed_img_2nd, threshed_img_3rd, threshed_img_4th]