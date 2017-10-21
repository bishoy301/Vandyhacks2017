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

def naive_normalization(img):
    '''
    Executes naive normalization of image to scale into [0,1]
    '''
    return np.divide(img, img.max())

def decompose_into_bands(ref_img_list, B, display=True):
    '''
    Gets decomposition as energy bands defined 
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
    if display:
        print('Convolving filters...')
        for img in tqdm(ref_img_list):
            cur_bands_list = []
            cur_bands_list.append(img) # I_0 is original image as defined in paper
            for i in range(1,B+1):
                # gaussian convolution implmented with scipy's gaussian filter
                cur_conv = gaussian_filter(cur_bands_list[-1], sigma=2**i)
                cur_band = cur_bands_list[-1] - cur_conv
                cur_bands_list.append(cur_band)
            all_bands.append(cur_bands_list)
    else:
        for img in ref_img_list:
            cur_bands_list = []
            cur_bands_list.append(img) # I_0 is original image as defined in paper
            for i in range(1,B+1):
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
    dL = int((img.shape[0] - newSideLength) / 2)
    omega = img[dL:-dL, dL:-dL]

    # calculates mean of region omega
    m_omega = np.mean(omega)

    # std using mean of omega instead of energy band
    return stdev(data=np.array(band).flatten(), xbar=m_omega)

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
    print("Beginning normalization...")

    B = len(all_bands)
    # find energy values for each band, using all ref images for each band
    all_ref_energies = []
    for band in tqdm(all_bands):
        ref_energy = 0
        for ref_img, corr_band in zip(ref_img_list, band):
            ref_energy += energy_function(ref_img, omega_percent, corr_band)
        ref_energy /= len(ref_img_list)
        all_ref_energies.append(ref_energy)

    # begin iterative normalization
    lambda_omega = 0
    I_norm = img.copy()
    print("Iterative normalization...")
    while not np.isclose(lambda_omega,1,atol=1e-5):
        I = I_norm
        cur_bands = decompose_into_bands(I, B, display=False)
        all_energies = []
        for band in cur_bands:
            all_energies.append(energy_function(I, omega_percent, band))
        lambda_omega_list = []
        for ref_energy, energy in zip(all_ref_energies, all_energies):
            lambda_omega_list.append(ref_energy/energy)

        
        # set normalized image

        # initial product
        I_norm = np.multiply(lambda_omega_list[0], cur_bands[0])
        # sum with rest of products
        for i in range(1,B):
            I_norm = np.add(I_norm, np.multiply(lambda_omega_list[i], cur_bands[i]))


        # set lambda omega to closest value to 1
        smallest = 1e1000
        for k in range(B):
            diff = np.abs(1-lambda_omega_list[k])
            print("Diff:", diff)
            if diff < smallest:
                smallest = diff
                smallest_idx = k
                print("Smallest diff:", smallest)
                print("idx of small:", smallest_idx)
        lambda_omega = lambda_omega_list[smallest_idx]
        print("Lambda_omega:", lambda_omega)
        print("Lambda type:", type(lambda_omega))

    return I_norm
