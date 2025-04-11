import os
import sys
import numpy as np
import cv2 as cv
from scipy.stats import wasserstein_distance


# ==General Concept==
#
# To measure color similarity of two images, one can compare the images' histogram.
# They represent color distributions of an image.
# The difference between those histograms represent the difference in color and tone of those images.
# To actually calculate the difference, one can e.g. use the Earth Mover's Distance (or Wasserstein Distance).
# The Wasserstein-Distance is symmetric -> a to be == b to a, which makes is suitable for this application
# It describes how much work would be required to transform one histogram into another.

def normalize(array: np.ndarray, axis: int) -> np.ndarray:
    """Normalizes values of flattened array into range [0,1]"""
    if not isinstance(array, np.ndarray):
        raise ValueError(f'Input must be np.array, not {type(array)}') 
    normalized = (array - np.min(array, axis=0)) / (np.max(array, axis=0) - np.min(array, axis=0))
    normalized = np.array(normalized)
    print(f'normalized array via axis {axis}')
    return normalized
    

def get_histograms(img: np.ndarray, hist_color_space: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get histograms of each channel from given image.
    
    Args:
        img (np.ndarray): Image to get histogram from.
        img_color_space (str): Color space of the image.
        hist_color_space (str): Color space to use for histogram.
    
    Returns:
        Histogram (np.array): Histogram of the image.
    """
    # Ensure valid hist-space is calculated
    hist_color_space=hist_color_space.upper()
    accepted_color_spaces = ['BGR', 'RGB', 'HSV', 'LAB', 'LUV']
    if not hist_color_space in accepted_color_spaces:
        raise TypeError(f'Unsupported histogram color space {hist_color_space}. Must be one of {accepted_color_spaces}')
    
    img = img
    if hist_color_space not in ['BGR', 'HSV']:   # conversion to same color space unnecessary
        conversion_code = getattr(cv, f'COLOR_BGR2{hist_color_space}')
        img = cv.cvtColor(img, conversion_code)    
    elif hist_color_space == 'HSV':
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)   # Use _FULL to map 256 values to 360 (usual Hue channel range) instead of 180 (opencv default)  
 
    # split image into 3 channels
    # which they represent not too important currently -> only difference to similar channel of other image
    channels = cv.split(img)
    channel_0, channel_1, channel_2 = channels

    # calculate the histograms
    ranges = [0, 256]    # Each channel in each color spaces should have range [0,255]
    channel_0_hist = cv.calcHist(images=[channel_0], channels=[0], histSize=[256], ranges=ranges, mask=None).flatten()
    channel_1_hist = cv.calcHist(images=[channel_1], channels=[0], histSize=[256], ranges=ranges, mask=None).flatten()
    channel_2_hist = cv.calcHist(images=[channel_2], channels=[0], histSize=[256], ranges=ranges, mask=None).flatten()
    histograms = [channel_0_hist, channel_1_hist, channel_2_hist]

    # ensure image size does not distort metric -> mease
    normalized_channel_histograms = [normalize(histo, axis=0) for histo in histograms]  # axis=1 for 'horizontal' normalization

    return normalized_channel_histograms

def get_wassersteins(histograms: tuple[list, list], bins: int=None) -> np.ndarray:
    """Calculates Wasserstein-Distances per channel-histogram of two given images"""
    if len(histograms) != 2:
        if len(histograms) < 2: 
            raise ValueError(f'Too few image! Expected 2, got {len(histograms)}')
        raise ValueError(f'Too many images! Expected 2, got {len(histograms)}')
    
    img1_hists, img2_hists = histograms
    if len(img1_hists) < 3: 
        raise ValueError(f'Too few histograms from first image! Expected 3, got {len(img1_hists)}')
    elif len(img2_hists) < 3: 
        raise ValueError(f'Too few histograms from second image! Expected 3, got {len(img2_hists)}')
    elif len(img1_hists) > 3: 
        raise ValueError(f'Too many histograms from first image! Expected 3, got {len(img1_hists)}')
    elif len(img2_hists) > 3: 
        raise ValueError(f'Too many histograms from second image! Expected 3, got {len(img2_hists)}')
    
    channel_1_distance = wasserstein_distance(u_values=img1_hists[0], v_values=img2_hists[0])
    channel_2_distance = wasserstein_distance(u_values=img1_hists[1], v_values=img2_hists[1])
    channel_3_distance = wasserstein_distance(u_values=img1_hists[2], v_values=img2_hists[2])

    return np.array([channel_1_distance, channel_2_distance, channel_3_distance])

def distance_as_float(vector: np.ndarray, channel_weights: tuple[float, float, float]=[1, 1, 1]) -> float:
    if vector.shape != (3,):
        raise ValueError(f'Invalid shape of given distance-vector. Expected (3,), got {vector.shape}')
    weight1, weight2, weight3 = channel_weights
    dim1_weighted = weight1*vector[0]
    dim2_weighted = weight2*vector[1]
    dim3_weighted = weight3*vector[2]
    float_distance = np.sqrt(dim1_weighted**2 + dim2_weighted**2 + dim3_weighted**2)
    return float(float_distance)



if __name__ == '__main__':

    # Test distance (should be 0)
    img_1 = cv.imread('ColorSimilarity/image.png')
    img_2 = cv.imread('ColorSimilarity/image2.jpg')

    histograms1 = get_histograms(img=img_1, hist_color_space='bgr')
    histograms2 = get_histograms(img=img_2, hist_color_space='bgr')

    histos = [histograms1, histograms2]
    channel_distance_vector = get_wassersteins(histograms=histos)
    print(channel_distance_vector)

    float_distance = distance_as_float(vector=channel_distance_vector)
    print(float_distance)

    sys.exit()




    