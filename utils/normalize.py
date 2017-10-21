'''
Author: Samuel Remedios

Normalizes CXR as described in:
https://www.ncbi.nlm.nih.gov/pubmed/2583851://www.ncbi.nlm.nih.gov/pubmed/25838517

'''
import numpy as np
from PIL import Image
from tqdm import *
from statistics import stdev
from scipy.ndimage.filters import gaussian_filter

def get_energy_value(ref_img_list, B):
    '''
    Gets energy value according to the energy bands defined 
    by the pixel intensity distribution from ref_img_list

    Notation and variable choice comes from aforementioned paper
    Params:
        - ref_img_list:     list of reference images to normalize to
        - B:                number of energy bands
    Returns:
        - all_bands:        list of lists, in [numRefImgs x numBands]
    '''
    # calculate bands for each reference image
    all_bands = []
    for img in ref_img_list:
        cur_bands_list = []
        cur_bands_list.append(img) # I_0 is original image as defined in paper
        for i in B:
            # gaussian convolution implmented with scipy's gaussian filter
            cur_conv = gaussian_filter(cur_bands_list[-1], sigma=2**i)
            cur_band = cur_bands_list[-1] - cur_conv
            cur_bands_list.append(cur_band)
        all_bands.append(cur_bands_list)

    return all_bands 

def energy_function(img, omega_percent, band):
    '''
    Energy value function of the energy band of the image inside region omega
    Params:
        - img:              original image
        - omega_percent:    % of original image to use as ROI
        - band:             current band to compute energy of
    Returns:
        - energy_value
    '''
    # calculate omega, assuming all images are squares
    totalArea = np.prod(img.shape)
    newSideLength = np.ceil(np.sqrt(totalArea * omega_percent))
    dL = (img.shape[0] - newSideLength) / 2
    omega = img[dl:-dl, dl:-dl]

    # calculates mean of region omega
    m_omega = np.mean(omega)

    # std using mean of omega instead of energy band
    return stdev(data=band, xbar=m_omega)

def normalize(img, omega_percent, ref_img_list, all_bands):
    '''
    Normalizes a 2D Xray image

    Params:
        - img:              numpy ndarray of image data
        - omega_percent:    % of original image to use as ROI
        - ref_img_list:     list of reference images to normalize to
        - all_bands:        list of lists, in [numRefImgs x numBands]
    Returns:
        - normalized_img:   numpy ndarray of normalized image data, intensities
                            in [0,1]
    '''
    # find energy values for each band, using all ref images for each band
    all_ref_energies = []
    for band in all_bands:
        ref_energy = 0
        for ref_img in ref_img_list:
            ref_energy += energy_function(ref_img, omega_percent, band)
        ref_energy /= len(ref_img_list)
        all_ref_energies.append(ref_energy)


    # for now, naive normalization 
    return np.divide(img, img.max())
