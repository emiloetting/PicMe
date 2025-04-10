import os
import sys
import numpy as np
import cv2 as cv
import scipy


# General Concept
#
# To measure color similarity of two images, one can compare the images' histogram.
# They represent color distributions of an image.
# The difference between those histograms represent the difference in color and tone of those images.
# To actually calculate the difference, one can e.g. use the Earth Mover's Distance (or Wasserstein Distance).
# It describes how much work would be required to transform one histogram into another.

def get_histogram(img: np.ndarray, img_color_space: str, hist_color_space: str) -> np.array:
    """Get histogram on an image in selected color_space.
    
    Args:
        img (np.ndarray): Image to get histogram from.
        img_color_space (str): Color space of the image.
        hist_color_space (str): Color space to use for histogram.
    
    Returns:
        Histogram (np.array): Histogram of the image.
    """
    accepted_color_spaces = ['BGR', 'RGB', 'HSV', 'LAB', 'LCH', 'LUV']
    if not hist_color_space.upper() in accepted_color_spaces:
        raise TypeError(f'Unsupported histogram color space {hist_color_space}. Must be one of {accepted_color_spaces}')
    
    img = img
    match img_color_space:
        case 'bgr':
            img = cv.imread(img, )
        case 'rgb':
            pass
        case 'hsv':
            pass
        case 'lab':
            pass
        case 'lch':
            pass
        case 'luv':
            pass
        case _:
            raise TypeError(f"Unsupported input-image color space {img_color_space.upper()}. Must be one of {accepted_color_spaces}")


    